"""Run with: uv run python -m src.preprocess data=tcga."""

import csv
import hashlib
import importlib.metadata
import io
import json
import logging
import re
import subprocess
import time
from pathlib import Path

import cv2
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# Compatibility imports preserve the existing public function paths.
from src.inspection import read_thumbnail
from src.masking import tissue_mask as tissue_mask
from src.quality import clean_mask as clean_mask
from src.image_utils import save_png, tissue_hole_preview
from src.report import write_report
from src.pipeline import preprocess_masks

LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def code_provenance():
    files = ("preprocess", "export", "tiling", "pipeline", "masking", "quality", "filters",
             "inspection", "image_utils", "report")
    hashes = {f"src/{name}.py": hashlib.sha256((PROJECT_ROOT / "src" / f"{name}.py").read_bytes()).hexdigest()
              for name in files}
    try:
        revision = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"git_revision": revision, "worktree_dirty": dirty, "source_files_sha256": hashes}


def run(cfg, output):
    output = Path(output)
    run_purpose = cfg.get("run_purpose", "demo")
    if run_purpose not in {"demo", "validation"}:
        raise ValueError("run_purpose must be demo or validation")
    for name, value in (("data.max_slides", cfg.data.max_slides),
                        ("thumbnail.max_size", cfg.thumbnail.max_size),
                        ("thumbnail.max_read_pixels", cfg.thumbnail.max_read_pixels),
                        ("thumbnail.max_total_read_pixels", cfg.thumbnail.max_total_read_pixels)):
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer, not a boolean or fractional value")
    if not 32 <= cfg.thumbnail.max_size <= 4096:
        raise ValueError("thumbnail.max_size must be 32-4096")
    if not 1 <= cfg.data.max_slides <= 100:
        raise ValueError("data.max_slides must be 1-100")
    if not 1 <= cfg.thumbnail.max_read_pixels <= 64_000_000:
        raise ValueError("thumbnail.max_read_pixels must be 1-64000000")
    if not cfg.thumbnail.max_read_pixels <= cfg.thumbnail.max_total_read_pixels <= 200_000_000:
        raise ValueError("max_total_read_pixels must be between max_read_pixels and 200000000")
    if cfg.artifacts.min_tissue_area < 0 or not 0 <= cfg.artifacts.dark_threshold <= 255:
        raise ValueError("Invalid artefact threshold")
    if not 0 <= cfg.artifacts.warn_removed_fraction <= 1:
        raise ValueError("warn_removed_fraction must be 0-1")
    if cfg.data.manifest and cfg.data.paths:
        raise ValueError("Use data.manifest or data.paths, not both")
    input_manifest = None
    if cfg.data.manifest:
        manifest_path = Path(to_absolute_path(cfg.data.manifest)).resolve(strict=True)
        if manifest_path.stat().st_size > 5_000_000:
            raise ValueError("Input manifest exceeds the 5 MB bound")
        manifest_bytes = manifest_path.read_bytes()
        rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig"))))
        input_manifest = {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    else:
        rows = [{"path": str(path), "label": "unlabelled"} for path in cfg.data.paths]
    rows = rows[:cfg.data.max_slides]
    for row in rows:
        for key in ("case_submitter_id", "case_id"):
            if key in row:
                row[key] = (row[key] or "").strip() or None
    if not rows:
        raise ValueError("No slides configured")
    pixel_slide_id = cfg.export.get("pixel_slide_id")
    if cfg.export.enabled and pixel_slide_id is not None:
        if sum(Path(row["path"]).stem == pixel_slide_id for row in rows) != 1:
            raise ValueError("export.pixel_slide_id must match exactly one configured WSI filename stem")
    job_name = HydraConfig.get().job.name if HydraConfig.initialized() else "preprocess"
    if output.is_symlink() or (output.exists() and any(
            path.name not in {".hydra", f"{job_name}.log"} for path in output.iterdir())):
        raise FileExistsError(f"Choose a fresh hydra.run.dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance = code_provenance()
    with (output / "run_started.json").open("x") as stream:
        json.dump({"run_kind": "preprocessing", "run_purpose": run_purpose}, stream)
    results = output / "results"
    results.mkdir()
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    cv2.setNumThreads(1)
    summaries = []
    for index, row in enumerate(rows):
        started = time.monotonic()
        path = Path(to_absolute_path(row["path"])).resolve(strict=True)
        row = {**row, "path": str(path)}
        source_stat = path.stat()
        patient_id_source = next((key for key in ("case_submitter_id", "case_id") if row.get(key)), None)
        patient_id = row[patient_id_source] if patient_id_source else None
        LOG.info("Processing %s (%d/%d)", path.name, index + 1, len(rows))
        rgb, metadata = read_thumbnail(path, cfg.thumbnail.max_size, cfg.thumbnail.max_read_pixels,
                                       cfg.thumbnail.max_total_read_pixels)
        preprocessing = preprocess_masks(rgb, masking=OmegaConf.to_container(cfg.masking, resolve=True),
                                         artifacts=OmegaConf.to_container(cfg.artifacts, resolve=True))
        tissue, clean = preprocessing["tissue"], preprocessing["clean"]
        excluded, reasons = preprocessing["excluded"], preprocessing["reasons"]
        if np.any(clean & ~tissue) or np.any(clean & excluded):
            raise RuntimeError("Mask invariants failed")
        folder = results / f"{index:02d}"
        folder.mkdir()
        removed = tissue & ~clean
        save_png(folder / "thumbnail.png", rgb)
        for name, mask in {"tissue_mask": tissue, "clean_mask": clean, "artifact_mask": excluded,
                           "removed_tissue": removed, **reasons}.items():
            save_png(folder / f"{name}.png", mask)
        stage_files = {}
        for name, array in preprocessing["stages"].items():
            save_png(folder / f"stage_{name}.png", array)
            stage_files[name] = str((folder / f"stage_{name}.png").relative_to(output))
        masked = rgb.copy()
        masked[~clean] = 255
        save_png(folder / "filtered.png", masked)
        overlay = rgb.copy()
        overlay[clean] = (0.65 * rgb[clean] + 0.35 * np.array([30, 190, 150])).astype(np.uint8)
        overlay[removed] = (0.35 * rgb[removed] + 0.65 * np.array([255, 70, 45])).astype(np.uint8)
        save_png(folder / "overlay.png", overlay)
        segmentation_preview, holes = tissue_hole_preview(rgb, clean)
        save_png(folder / "tissue_holes.png", segmentation_preview)
        save_png(folder / "hole_mask.png", holes)
        dataset = str(row.get("project", "demo")).lower().replace("-", "_")
        if not re.fullmatch(r"[a-z0-9_]+", dataset):
            raise ValueError("Dataset identifier must be filesystem-safe")
        gallery = output / "overviews" / dataset
        gallery.mkdir(parents=True, exist_ok=True)
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", patient_id or path.stem)
        if row.get("stain"):
            prefix += "__" + re.sub(r"[^A-Za-z0-9_.-]", "_", row["stain"])
        prefix += "__" + path.stem if patient_id else ""
        for name, array in {"thumbnail": rgb, "mask": clean, "preprocessed": masked, "overlay": overlay,
                            "tissue_holes": segmentation_preview}.items():
            destination = gallery / f"{prefix}__{name}.png"
            if destination.exists():
                raise FileExistsError(f"Duplicate gallery image: {destination}")
            save_png(destination, array)
        count = int(tissue.sum())
        removed_fraction = float(removed.sum() / count) if count else None
        warnings = []
        if not count:
            warnings.append("No candidate tissue detected")
        elif not clean.any():
            warnings.append("No tissue retained after filtering")
        if removed_fraction is not None and removed_fraction > cfg.artifacts.warn_removed_fraction:
            warnings.append("Large final cleanup of candidate tissue: review postmask exclusion/component settings")
        summary = {"input": str(path), "name": patient_id or path.name,
                   "patient_id": patient_id, "patient_id_source": patient_id_source,
                   "label": row.get("label") or "unlabelled", "case_id": row.get("case_id"),
                   "gallery": str(gallery.relative_to(output)), "gallery_prefix": prefix,
                   "site": row.get("site"), "folder": str(folder.relative_to(output)),
                   "input_bytes": source_stat.st_size, "input_mtime_ns": source_stat.st_mtime_ns, **metadata,
                   "declared_checksums": {key: row[key] for key in
                       ("md5sum", "sha256", "source_checksum", "source_checksum_type") if row.get(key)},
                   "source_checksum_verified": False,
                   "thumbnail_sha256": hashlib.sha256((folder / "thumbnail.png").read_bytes()).hexdigest(),
                   "tissue_pixels": count, "retained_pixels": int(clean.sum()),
                   "removed_pixels": int(removed.sum()), "removed_tissue_fraction": removed_fraction,
                   "tissue_fraction": float(tissue.mean()), "retained_fraction": float(clean.mean()),
                   "artifact_pixels_by_reason": {key: int(mask.sum()) for key, mask in reasons.items()},
                   "hole_visualization": "retained_tissue_green_hole_blue_contours_4_connected_thumbnail",
                   "preprocessing": preprocessing["metadata"], "preprocessing_stages": stage_files,
                   "warnings": warnings, "elapsed_seconds": round(time.monotonic() - started, 3)}
        for key in ("project", "file_name", "case_submitter_id", "stain", "sample_id", "sample_id_source",
                    "enrollment_category", "primary_adjudicated_category", "file_id", "age_binned", "sex"):
            if row.get(key):
                summary[key] = row[key]
        if cfg.export.enabled:
            from src.export import export_slide

            summary["exports"] = export_slide(output, row, metadata, rgb, clean, segmentation_preview,
                                               cfg.export, cfg.masking.method)
        final_stat = path.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise RuntimeError("Source file changed during preprocessing")
        summary["source_size_mtime_unchanged"] = True
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (folder / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        for warning in warnings:
            LOG.warning("%s: %s", path.name, warning)
    versions = {name: importlib.metadata.version(name) for name in ("tiatoolbox", "hydra-core", "numpy", "pillow", "opencv-python", "h5py")}
    write_report(output, summaries, cfg.masking.method)
    (output / "manifest.json").write_text(json.dumps({"status": "complete", "schema_version": 1,
        "run_kind": "preprocessing", "run_purpose": run_purpose, "versions": versions,
        "input_manifest": input_manifest, "code": provenance,
        "mask_method": cfg.masking.method, "slides": summaries}, indent=2) + "\n")
    LOG.info("Report: %s", output / "index.html")
    return summaries


@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess")
def main(cfg: DictConfig):
    run(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()

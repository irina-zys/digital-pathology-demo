"""Run with: uv run python -m src.preprocess data=tcga."""

import csv
import hashlib
import importlib.metadata
import json
import logging
import re
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


def run(cfg, output):
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
    if cfg.data.manifest:
        with Path(to_absolute_path(cfg.data.manifest)).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = [{"path": str(path), "label": "unlabelled"} for path in cfg.data.paths]
    rows = rows[:int(cfg.data.max_slides)]
    if not rows:
        raise ValueError("No slides configured")
    pixel_slide_id = cfg.export.get("pixel_slide_id")
    if cfg.export.enabled and pixel_slide_id is not None:
        if sum(Path(row["path"]).stem == pixel_slide_id for row in rows) != 1:
            raise ValueError("export.pixel_slide_id must match exactly one configured WSI filename stem")
    output.mkdir(parents=True, exist_ok=True)
    # A reused run directory must not mix old and new artifacts.
    if (output / "results").exists():
        raise FileExistsError(f"Choose a fresh hydra.run.dir: {output}")
    if (output / "datasets").exists():
        raise FileExistsError(f"Export directory already exists: {output / 'datasets'}")
    results = output / "results"
    results.mkdir()
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    cv2.setNumThreads(1)
    summaries = []
    for index, row in enumerate(rows):
        started = time.monotonic()
        path = Path(to_absolute_path(row["path"])).resolve(strict=True)
        row = {**row, "path": str(path)}
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
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", row.get("case_submitter_id", path.stem))
        if row.get("stain"):
            prefix += "__" + re.sub(r"[^A-Za-z0-9_.-]", "_", row["stain"])
        prefix += "__" + path.stem if row.get("case_submitter_id") else ""
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
        summary = {"input": str(path), "name": row.get("case_submitter_id", path.name),
                   "label": row.get("label", "unlabelled"), "case_id": row.get("case_id"),
                   "gallery": str(gallery.relative_to(output)), "gallery_prefix": prefix,
                   "site": row.get("site"), "folder": str(folder.relative_to(output)),
                   "input_bytes": path.stat().st_size, **metadata,
                   "thumbnail_sha256": hashlib.sha256((folder / "thumbnail.png").read_bytes()).hexdigest(),
                   "tissue_pixels": count, "retained_pixels": int(clean.sum()),
                   "removed_pixels": int(removed.sum()), "removed_tissue_fraction": removed_fraction,
                   "tissue_fraction": float(tissue.mean()), "retained_fraction": float(clean.mean()),
                   "artifact_pixels_by_reason": {key: int(mask.sum()) for key, mask in reasons.items()},
                   "hole_visualization": "retained_tissue_green_hole_blue_contours_4_connected_thumbnail",
                   "preprocessing": preprocessing["metadata"], "preprocessing_stages": stage_files,
                   "warnings": warnings, "elapsed_seconds": round(time.monotonic() - started, 3)}
        for key in ("project", "file_name", "stain", "sample_id", "sample_id_source",
                    "enrollment_category", "primary_adjudicated_category", "file_id", "age_binned", "sex"):
            if row.get(key):
                summary[key] = row[key]
        if cfg.export.enabled:
            from src.export import export_slide

            summary["exports"] = export_slide(output, row, metadata, rgb, clean, segmentation_preview,
                                               cfg.export, cfg.masking.method)
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (folder / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        for warning in warnings:
            LOG.warning("%s: %s", path.name, warning)
    versions = {name: importlib.metadata.version(name) for name in ("tiatoolbox", "hydra-core", "numpy", "pillow", "opencv-python", "h5py")}
    write_report(output, summaries, cfg.masking.method)
    (output / "manifest.json").write_text(json.dumps({"status": "complete", "versions": versions,
        "mask_method": cfg.masking.method, "slides": summaries}, indent=2) + "\n")
    LOG.info("Report: %s", output / "index.html")
    return summaries


@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess")
def main(cfg: DictConfig):
    run(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()

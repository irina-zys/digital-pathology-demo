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
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from src.filters import filter_grays, filter_pen, optical_density, validate_rgb
from src.report import write_report

LOG = logging.getLogger(__name__)


def read_thumbnail(path, max_size, max_read_pixels, max_total_read_pixels):
    """Read an explicitly bounded pyramid level; never decode level 0 blindly."""
    if not path.is_file():
        raise FileNotFoundError(f"Input missing: {path}. Run the download script first.")
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as source:
            dimensions = source.size
            if source.width * source.height > max_read_pixels:
                raise ValueError(f"Raster exceeds thumbnail.max_read_pixels: {path}")
            image = source.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            image = background.convert("RGB")
        metadata = {"reader": "Pillow", "dimensions": list(dimensions), "read_level": 0,
                    "read_dimensions": list(dimensions), "mpp": None, "vendor": None,
                    "objective_power": None}
    else:
        from tiatoolbox.wsicore.wsireader import WSIReader

        reader = WSIReader.open(path)
        try:
            info = reader.info
            levels = [tuple(map(int, size)) for size in info.level_dimensions]
            safe = [i for i, (w, h) in enumerate(levels) if w * h <= max_read_pixels]
            sufficient = [i for i in safe if max(levels[i]) >= max_size]
            level = max(sufficient) if sufficient else (min(safe) if safe else len(levels) - 1)
            width, height = levels[level]
            if width * height > max_total_read_pixels:
                raise ValueError("Lowest usable level exceeds thumbnail.max_total_read_pixels")
            if safe:
                array = reader.read_bounds((0, 0, *info.slide_dimensions), resolution=level, units="level")
                validate_rgb(array)
                image = Image.fromarray(array)
            else:
                # Some real SVS files have only two levels. Resize bounded blocks.
                scale = min(1.0, max_size / max(width, height))
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                image = Image.new("RGB", size, "white")
                edge = int(np.sqrt(max_read_pixels))
                downsample = float(info.level_downsamples[level])
                for y in range(0, height, edge):
                    for x in range(0, width, edge):
                        bw, bh = min(edge, width - x), min(edge, height - y)
                        box = (round(x * size[0] / width), round(y * size[1] / height),
                               round((x + bw) * size[0] / width), round((y + bh) * size[1] / height))
                        if box[2] == box[0] or box[3] == box[1]:
                            continue
                        array = reader.read_rect((round(x * downsample), round(y * downsample)),
                                                 size=(bw, bh), resolution=level, units="level")
                        validate_rgb(array)
                        block = Image.fromarray(array).resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
                        image.paste(block, box[:2])
            metadata = {"reader": type(reader).__name__, "dimensions": list(map(int, info.slide_dimensions)),
                        "read_level": level, "read_dimensions": list(levels[level]),
                        "read_strategy": "whole_level" if safe else "bounded_blocks",
                        "level_dimensions": levels, "vendor": info.vendor,
                        "objective_power": None if info.objective_power is None else float(info.objective_power),
                        "mpp": None if info.mpp is None else np.asarray(info.mpp).tolist()}
        finally:
            handle = getattr(reader, "openslide_wsi", None)
            if handle is not None:
                handle.close()
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    rgb = np.asarray(image).copy()
    metadata["thumbnail_dimensions"] = [image.width, image.height]
    metadata["level0_pixels_per_thumbnail_pixel"] = [
        metadata["dimensions"][0] / image.width, metadata["dimensions"][1] / image.height]
    return rgb, metadata


def tissue_mask(rgb, cfg):
    if cfg.method in {"tia_otsu", "tia_morphological"}:
        from tiatoolbox.tools.tissuemask import MorphologicalMasker, OtsuTissueMasker

        masker = (OtsuTissueMasker() if cfg.method == "tia_otsu" else
                  MorphologicalMasker(kernel_size=int(cfg.kernel_size), min_region_size=int(cfg.min_region_size)))
        return masker.fit_transform(rgb[None])[0].astype(bool)
    if cfg.method == "clam_saturation":
        size = int(cfg.median_size)
        if size < 1 or size % 2 == 0 or size > 31:
            raise ValueError("median_size must be odd and between 1 and 31")
        saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
        saturation = cv2.medianBlur(saturation, size)
        threshold = 0 if cfg.use_otsu else int(cfg.saturation_threshold)
        kind = cv2.THRESH_BINARY | (cv2.THRESH_OTSU if cfg.use_otsu else 0)
        _, mask = cv2.threshold(saturation, threshold, 255, kind)
        close = int(cfg.closing_size)
        if not 0 <= close <= 31 or not 0 <= threshold <= 255:
            raise ValueError("Invalid saturation threshold or closing size")
        if close:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
        return mask.astype(bool)
    if cfg.method == "codait_od":
        return np.min(optical_density(rgb, cfg.background_intensity), axis=2) >= cfg.threshold
    raise ValueError(f"Unknown tissue masking method: {cfg.method}")


def clean_mask(rgb, tissue, cfg):
    reasons = {f"pen_{color}": ~filter_pen(rgb, color) for color in cfg.pen_colors}
    if cfg.remove_dark:
        reasons["dark"] = np.max(rgb, axis=2) < cfg.dark_threshold
    if cfg.remove_gray:
        reasons["gray"] = ~filter_grays(rgb, cfg.gray_tolerance)
    excluded = np.zeros(tissue.shape, bool)
    for mask in reasons.values():
        excluded |= mask
    clean = tissue & ~excluded
    if cfg.min_tissue_area > 0:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(clean.astype(np.uint8), connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= cfg.min_tissue_area
        keep[0] = False
        filtered = keep[labels]
        reasons["small_components"] = clean & ~filtered
        excluded |= reasons["small_components"]
        clean = filtered
    return clean, excluded, reasons


def save_png(path, array):
    Image.fromarray(array.astype(np.uint8) * 255 if array.dtype == bool else array).save(path)


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
        with Path(cfg.data.manifest).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = [{"path": str(path), "label": "unlabelled"} for path in cfg.data.paths]
    rows = rows[:int(cfg.data.max_slides)]
    if not rows:
        raise ValueError("No slides configured")
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
        path = Path(row["path"])
        LOG.info("Processing %s (%d/%d)", path.name, index + 1, len(rows))
        rgb, metadata = read_thumbnail(path, cfg.thumbnail.max_size, cfg.thumbnail.max_read_pixels,
                                       cfg.thumbnail.max_total_read_pixels)
        tissue = tissue_mask(rgb, cfg.masking)
        clean, excluded, reasons = clean_mask(rgb, tissue, cfg.artifacts)
        if np.any(clean & ~tissue) or np.any(clean & excluded):
            raise RuntimeError("Mask invariants failed")
        folder = results / f"{index:02d}"
        folder.mkdir()
        removed = tissue & ~clean
        save_png(folder / "thumbnail.png", rgb)
        for name, mask in {"tissue_mask": tissue, "clean_mask": clean, "artifact_mask": excluded,
                           "removed_tissue": removed, **reasons}.items():
            save_png(folder / f"{name}.png", mask)
        masked = rgb.copy()
        masked[~clean] = 255
        save_png(folder / "filtered.png", masked)
        overlay = rgb.copy()
        overlay[clean] = (0.65 * rgb[clean] + 0.35 * np.array([30, 190, 150])).astype(np.uint8)
        overlay[removed] = (0.35 * rgb[removed] + 0.65 * np.array([255, 70, 45])).astype(np.uint8)
        save_png(folder / "overlay.png", overlay)
        dataset = str(row.get("project", "demo")).lower().replace("-", "_")
        if not re.fullmatch(r"[a-z0-9_]+", dataset):
            raise ValueError("Dataset identifier must be filesystem-safe")
        gallery = output / "overviews" / dataset
        gallery.mkdir(parents=True, exist_ok=True)
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", row.get("case_submitter_id", path.stem))
        prefix += "__" + path.stem if row.get("case_submitter_id") else ""
        for name, array in {"thumbnail": rgb, "mask": clean, "preprocessed": masked, "overlay": overlay}.items():
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
            warnings.append("Large tissue exclusion: review pen/dark/small-component settings")
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
                   "warnings": warnings, "elapsed_seconds": round(time.monotonic() - started, 3)}
        if cfg.export.enabled:
            from src.export import export_slide

            summary["exports"] = export_slide(output, row, metadata, rgb, clean, overlay,
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

"""Steps 3/5: artefact exclusion and descriptive tile-quality measurements."""

import cv2
import numpy as np

from src.imaging_utils.filters import filter_grays, filter_pen, resolve_pen_thresholds
from src.imaging_utils.arrays import validate_rgb, validate_mask, remove_small_components
from src.imaging_utils.masking import saturation_mask


def exclude_artifacts(rgb, tissue, *, pen_colors=(), remove_dark=False, dark_threshold=30,
                      remove_gray=False, gray_tolerance=15, min_tissue_area=0, pen_thresholds=None):
    """Return (retained, excluded, reason_masks), all bool H x W.

    Exclusion/reason masks may also mark background; removed tissue is
    tissue & excluded. Rules are heuristics, not validated artefact labels.
    """
    validate_rgb(rgb)
    validate_mask(tissue, rgb.shape[:2])
    if not 0 <= dark_threshold <= 255 or not 0 <= gray_tolerance <= 255:
        raise ValueError("Colour thresholds must be in 0-255")
    if not isinstance(min_tissue_area, (int, np.integer)) or min_tissue_area < 0:
        raise ValueError("min_tissue_area must be a nonnegative integer")
    rules = resolve_pen_thresholds(pen_thresholds)
    reasons = {f"pen_{color}": ~filter_pen(rgb, color, rules[color]) for color in pen_colors}
    if remove_dark:
        reasons["dark"] = np.max(rgb, axis=2) < dark_threshold
    if remove_gray:
        reasons["gray"] = ~filter_grays(rgb, gray_tolerance)
    excluded = np.zeros(tissue.shape, bool)
    for mask in reasons.values():
        excluded |= mask
    clean = tissue & ~excluded
    if min_tissue_area > 0:
        filtered = remove_small_components(clean, min_tissue_area)
        reasons["small_components"] = clean & ~filtered
        excluded |= reasons["small_components"]
        clean = filtered
    return clean, excluded, reasons


def clean_mask(rgb, tissue, cfg):
    """Compatibility adapter; warning thresholds belong to run-level reporting."""
    return exclude_artifacts(rgb, tissue, **{k: v for k, v in dict(cfg).items()
                                             if k != "warn_removed_fraction"})


def tile_quality(rgb, *, saturation_threshold):
    """Measure coloured-pixel fraction and Laplacian variance; make no QC decision.

    Detail depends on tissue, scale and contrast as well as focus. Calibrate any
    downstream exclusion threshold independently; there is no universal cutoff.
    """
    tissue, _ = saturation_mask(rgb, use_otsu=False, threshold=saturation_threshold)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {"tissue_fraction": float(tissue.mean()),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var())}

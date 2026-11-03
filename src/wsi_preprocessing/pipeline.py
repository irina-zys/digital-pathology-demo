"""Compose preprocessing stages with explicit order and inspectable intermediates."""

import numpy as np

from src.imaging_utils.filters import resolve_pen_thresholds
from src.imaging_utils.arrays import validate_rgb, fill_small_holes, smooth_mask_edges, white_background
from src.imaging_utils.masking import create_tissue_mask, saturation_mask, otsu_mask
from src.imaging_utils.quality import exclude_artifacts


def preprocess_masks(rgb, *, masking, artifacts):
    """Return tissue/clean/excluded masks, reasons, stages and metadata.

    Arguments are ordinary mappings, not Hydra objects. Combined order:
    CODAIT exclusions -> white replacement -> TIA grayscale threshold -> white
    replacement -> CLAM-style saturation/median/closing -> small-hole filling
    -> elliptical opening/closing -> reapply CODAIT exclusions and remove small tissue components.

    Source RGB is never modified. Median filtering, closing, hole filling and smoothing can
    add pixels beyond TIA's preliminary gate; final cleanup reapplies CODAIT
    exclusions. Hole filling itself is limited to enclosed small holes.
    Legacy single-mask recipes retain their previous ordering.
    """
    validate_rgb(rgb)
    options, artifact_options = dict(masking), dict(artifacts)
    artifact_options.pop("warn_removed_fraction", None)
    method = options.pop("method")
    stages, metadata = {}, {"method": method}
    if method == "codait_tia_clam":
        use_tia = options.pop("tia_enabled", True)
        tia_threshold = options.pop("tia_threshold", None)
        max_hole_area = options.pop("max_hole_area", 16)
        smoothing_kernel_size = options.pop("smoothing_kernel_size", 0)
        if type(use_tia) is not bool:
            raise ValueError("tia_enabled must be boolean")
        # Component cleanup belongs after tissue detection, not on the background.
        prefilter = {**artifact_options, "min_tissue_area": 0}
        _, pre_excluded, _ = exclude_artifacts(rgb, np.ones(rgb.shape[:2], bool), **prefilter)
        codait_rgb = white_background(rgb, ~pre_excluded)
        tia, applied_tia = (otsu_mask(codait_rgb, threshold=tia_threshold) if use_tia
                            else (np.ones(rgb.shape[:2], bool), None))
        clam_rgb = white_background(codait_rgb, tia)
        # Config keeps the same saturation parameter name as the legacy recipe.
        options["threshold"] = options.pop("saturation_threshold", 20)
        clam, applied_clam = saturation_mask(clam_rgb, **options)
        before_holes = clam & ~pre_excluded
        before_smoothing = fill_small_holes(before_holes, max_hole_area)
        tissue = smooth_mask_edges(before_smoothing, smoothing_kernel_size)
        stages = {"codait_excluded": pre_excluded, "codait_filtered": codait_rgb,
                  "tia_otsu": tia, "clam_input": clam_rgb, "clam_mask": clam,
                  "before_holes": before_holes, "filled_holes": before_smoothing & ~before_holes,
                  "before_smoothing": before_smoothing, "smoothed_mask": tissue}
        metadata.update(order=["codait", "tia_otsu", "clam_saturation", "small_holes", "morphological_smoothing", "final_exclusion"],
                        prefilter_excluded_pixels=int(pre_excluded.sum()),
                        prefilter_excluded_fraction=float(pre_excluded.mean()),
                        tia_enabled=use_tia, tia_threshold=applied_tia,
                        tia_threshold_mode=("disabled" if not use_tia else
                                            "fitted_otsu" if tia_threshold is None else "manual"),
                        clam_saturation_threshold=applied_clam,
                        max_hole_area_mask_pixels=max_hole_area,
                        filled_hole_pixels=int(stages["filled_holes"].sum()),
                        smoothing_kernel_size=int(smoothing_kernel_size),
                        smoothing_operations=["opening", "closing"] if smoothing_kernel_size >= 3 else [],
                        smoothing_added_pixels=int((tissue & ~before_smoothing).sum()),
                        smoothing_removed_pixels=int((before_smoothing & ~tissue).sum()))
    else:
        tissue = create_tissue_mask(rgb, method=method, **options)
    clean, excluded, reasons = exclude_artifacts(rgb, tissue, **artifact_options)
    if np.any(clean & ~tissue) or np.any(clean & excluded):
        raise RuntimeError("Mask partition invariant failed")
    metadata["pen_colors"] = list(artifact_options.get("pen_colors", ()))
    metadata["pen_thresholds"] = resolve_pen_thresholds(artifact_options.get("pen_thresholds"))
    return {"tissue": tissue, "clean": clean, "excluded": excluded, "reasons": reasons,
            "stages": stages, "metadata": metadata}

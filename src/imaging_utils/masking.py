"""Step 2: candidate tissue detection; masks do not identify tumour."""

import cv2
import numpy as np

from src.imaging_utils.filters import optical_density
from src.imaging_utils.arrays import validate_rgb


def saturation_mask(rgb, *, median_size=1, use_otsu=True, threshold=20, closing_size=0):
    """Return (bool mask, applied 0-255 saturation threshold) for uint8 RGB."""
    validate_rgb(rgb)
    if not isinstance(median_size, (int, np.integer)) or not 1 <= median_size <= 31 or median_size % 2 == 0:
        raise ValueError("median_size must be odd and between 1 and 31")
    if not isinstance(closing_size, (int, np.integer)) or not 0 <= closing_size <= 31:
        raise ValueError("closing_size must be an integer between 0 and 31")
    if not np.isfinite(threshold) or not 0 <= threshold <= 255:
        raise ValueError("Saturation threshold must be finite and in 0-255")
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    saturation = cv2.medianBlur(saturation, int(median_size))
    kind = cv2.THRESH_BINARY | (cv2.THRESH_OTSU if use_otsu else 0)
    applied, mask = cv2.threshold(saturation, 0 if use_otsu else int(threshold), 255, kind)
    if closing_size:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((closing_size, closing_size), np.uint8))
    return mask.astype(bool), float(applied)


def create_tissue_mask(rgb, *, method, kernel_size=5, min_region_size=64,
                       median_size=7, use_otsu=True, saturation_threshold=20,
                       closing_size=0, background_intensity=240, threshold=0.15):
    """Return a bool candidate-tissue mask; size parameters use input-image pixels."""
    validate_rgb(rgb)
    if method in {"tia_otsu", "tia_morphological"}:
        from tiatoolbox.tools.tissuemask import MorphologicalMasker, OtsuTissueMasker

        if method == "tia_morphological":
            for name, value, minimum in (("kernel_size", kernel_size, 1),
                                         ("min_region_size", min_region_size, 0)):
                if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                        or value < minimum):
                    raise ValueError(f"{name} must be an integer >= {minimum}")
        masker = (OtsuTissueMasker() if method == "tia_otsu" else
                  MorphologicalMasker(kernel_size=int(kernel_size), min_region_size=int(min_region_size)))
        return masker.fit_transform(rgb[None])[0].astype(bool)
    if method == "clam_saturation":
        mask, _ = saturation_mask(rgb, median_size=median_size, use_otsu=use_otsu,
                                  threshold=saturation_threshold, closing_size=closing_size)
        return mask
    if method == "codait_od":
        return np.min(optical_density(rgb, background_intensity), axis=2) >= threshold
    raise ValueError(f"Unknown tissue masking method: {method}")


def otsu_mask(rgb, *, threshold=None):
    """Return TIA's grayscale Otsu mask and threshold (0-255).

    None fits Otsu on this image. An explicit threshold uses TIA's transform
    with that grayscale threshold; this is a manual override, not fitted Otsu.
    """
    from tiatoolbox.tools.tissuemask import OtsuTissueMasker

    validate_rgb(rgb)
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 255):
        raise ValueError("TIA grayscale threshold must be None or finite in 0-255")
    masker = OtsuTissueMasker()
    if threshold is None:
        masker.fit(rgb[None])
    else:
        masker.threshold = float(threshold)
        masker.fitted = True
    return masker.transform(rgb[None])[0].astype(bool), float(masker.threshold)

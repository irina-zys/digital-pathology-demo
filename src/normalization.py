"""Compare TIA normalisation with original or contour-whitened backgrounds."""

import numpy as np

from src.image_utils import validate_rgb, white_background


def _prepare(rgb, tissue_mask, max_pixels):
    validate_rgb(rgb)
    if rgb.shape[0] * rgb.shape[1] > max_pixels:
        raise ValueError("Stain input exceeds max_pixels")
    if tissue_mask is None:
        return rgb.copy()
    prepared = white_background(rgb, tissue_mask)
    if not tissue_mask.any():
        raise ValueError("Stain normalisation requires retained tissue")
    return prepared


def fit_stain_normalizer(reference_rgb, *, method, tissue_mask=None, max_pixels=1_000_000):
    """Fit TIA to a bounded reference; an optional tissue mask whitens background.

    Whitening removes background colours, not the effect of background fraction:
    TIA Reinhard statistics and Macenko concentration percentiles use all pixels.
    """
    from tiatoolbox.tools.stainnorm import get_normalizer

    prepared = _prepare(reference_rgb, tissue_mask, max_pixels)
    if method not in {"reinhard", "macenko", "ruifrok", "vahadane"}:
        raise ValueError("Demo methods are reinhard, macenko, ruifrok or vahadane")
    normalizer = get_normalizer(method)
    normalizer.fit(prepared)
    return normalizer


def normalize_tile(rgb, normalizer, *, tissue_mask=None, max_pixels=1_000_000):
    """Transform a copy; with a mask, whiten background before and after TIA.

    With tissue_mask=None, original background pixels participate in TIA defaults.
    """
    prepared = _prepare(rgb, tissue_mask, max_pixels)
    result = normalizer.transform(prepared)
    validate_rgb(result)
    if result.shape != rgb.shape or not np.isfinite(result).all():
        raise RuntimeError("Invalid stain-normalisation output")
    return result.copy() if tissue_mask is None else white_background(result, tissue_mask)

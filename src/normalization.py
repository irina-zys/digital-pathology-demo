"""Reusable TIA stain fitting, transformation and matched background comparisons.

No slide paths, plotting, Hydra configuration or file output belong here.
RGB arrays are uint8 H x W x 3; masks are bool H x W with True meaning tissue.
"""

import time

import numpy as np

from src.image_utils import validate_rgb, validate_mask, fill_background


STAIN_METHODS = ("reinhard", "macenko", "ruifrok", "vahadane")


def prepare_stain_image(rgb, *, tissue_mask=None, background_rgb=None, max_pixels=1_000_000):
    """Copy a bounded image; optionally replace non-tissue with an explicit colour.

    No mask means original background is included. With a mask, the caller must
    supply a shared colour, e.g. a rounded mean from mean_background_color.
    """
    validate_rgb(rgb)
    if not isinstance(max_pixels, (int, np.integer)) or max_pixels < 1:
        raise ValueError("max_pixels must be a positive integer")
    if rgb.shape[0] * rgb.shape[1] > max_pixels:
        raise ValueError("Stain input exceeds max_pixels")
    if tissue_mask is None:
        return rgb.copy()
    prepared = fill_background(rgb, tissue_mask, background_rgb)
    if not tissue_mask.any():
        raise ValueError("Stain normalisation requires retained tissue")
    return prepared


def fit_stain_normalizer(reference_rgb, *, method, tissue_mask=None,
                         background_rgb=None, max_pixels=1_000_000):
    """Fit a TIA normalizer using original or constant-background reference RGB.

    Background replacement does not remove background pixels from TIA statistics.
    TIA's Vahadane implementation emits its own documented stability warning.
    """
    from tiatoolbox.tools.stainnorm import get_normalizer

    if method not in STAIN_METHODS:
        raise ValueError(f"Supported methods: {', '.join(STAIN_METHODS)}")
    prepared = prepare_stain_image(reference_rgb, tissue_mask=tissue_mask,
        background_rgb=background_rgb, max_pixels=max_pixels)
    normalizer = get_normalizer(method)
    normalizer.fit(prepared)
    return normalizer


def normalize_tile(rgb, normalizer, *, tissue_mask=None, background_rgb=None, max_pixels=1_000_000):
    """Transform a copy; with a mask, replace background before and after TIA."""
    prepared = prepare_stain_image(rgb, tissue_mask=tissue_mask,
        background_rgb=background_rgb, max_pixels=max_pixels)
    result = normalizer.transform(prepared)
    validate_rgb(result)
    if result.shape != rgb.shape or not np.isfinite(result).all():
        raise RuntimeError("Invalid stain-normalisation output")
    return result.copy() if tissue_mask is None else fill_background(result, tissue_mask, background_rgb)


def compare_stain_normalization(reference_rgb, source_images, *, reference_mask,
                                 source_masks, background_rgb, methods=STAIN_METHODS,
                                 random_seed=0, max_pixels=1_000_000):
    """Return matched unmasked/constant-background images and per-method timings.

    source_images/source_masks are mappings with matching arbitrary sample IDs.
    All methods use the same reference and source images. Each fit/transform
    receives the same NumPy seed for paired starts; caller RNG state is restored.
    Run serially: TIA Vahadane exposes no per-instance random_state parameter.
    No output is written and no caller image is modified.
    """
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in STAIN_METHODS for m in methods):
        raise ValueError("methods must be a nonempty, unique selection from STAIN_METHODS")
    if not source_images or set(source_images) != set(source_masks):
        raise ValueError("source_images and source_masks must have the same nonempty sample IDs")
    # Validate every input before any costly fitting.
    prepare_stain_image(reference_rgb, tissue_mask=reference_mask,
                       background_rgb=background_rgb, max_pixels=max_pixels)
    validate_mask(reference_mask, reference_rgb.shape[:2])
    for key, rgb in source_images.items():
        validate_mask(source_masks[key], rgb.shape[:2])
        prepare_stain_image(rgb, tissue_mask=source_masks[key],
                           background_rgb=background_rgb, max_pixels=max_pixels)
    results, timings = {}, []
    rng_state = np.random.get_state()
    try:
        for method in methods:
            started = time.monotonic()
            results[method] = {}
            for mode in ("unmasked", "constant_background"):
                ref_mask = None if mode == "unmasked" else reference_mask
                np.random.seed(random_seed)
                normalizer = fit_stain_normalizer(reference_rgb, method=method,
                    tissue_mask=ref_mask, background_rgb=background_rgb, max_pixels=max_pixels)
                images = {}
                for key, rgb in source_images.items():
                    np.random.seed(random_seed)
                    images[key] = normalize_tile(rgb, normalizer,
                        tissue_mask=None if mode == "unmasked" else source_masks[key],
                        background_rgb=background_rgb, max_pixels=max_pixels)
                results[method][mode] = images
            timings.append({"method": method, "seconds": time.monotonic() - started})
    finally:
        np.random.set_state(rng_state)
    return {"images": results, "timings": timings}

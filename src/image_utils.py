"""Shared array contracts and lossless image output.

RGB images: uint8 H x W x 3. Binary masks: bool H x W, True selects pixels.
Coordinates and dimensions are (x, y)/(width, height); arrays index [y, x].
No helper modifies its input array in place.
"""

import cv2
import numpy as np
from PIL import Image


def validate_rgb(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB array, values 0-255")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected nonempty H x W x 3 RGB image")


def validate_mask(mask, shape=None):
    """Require a nonempty 2D boolean mask, optionally matching an image shape."""
    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("Expected nonempty H x W boolean mask (True selects pixels)")
    if shape is not None and mask.shape != tuple(shape):
        raise ValueError("Mask and image dimensions differ")


def remove_small_components(mask, min_area):
    """Keep 8-connected components with at least min_area mask pixels; preserve holes."""
    validate_mask(mask)
    if not isinstance(min_area, (int, np.integer)) or min_area < 0:
        raise ValueError("min_area must be a nonnegative integer in mask pixels")
    if min_area == 0:
        return mask.copy()
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def save_png(path, array):
    """Save uint8 RGB or bool mask (white=True), without altering the source."""
    if array.dtype == bool:
        validate_mask(array)
    else:
        validate_rgb(array)
    Image.fromarray(array.astype(np.uint8) * 255 if array.dtype == bool else array).save(path)


def tissue_hole_preview(rgb, clean):
    """Highlight retained tissue and enclosed thumbnail background, without area cutoffs."""
    validate_rgb(rgb)
    validate_mask(clean, rgb.shape[:2])
    # Four-connected background complements the eight-connected tissue contours.
    flooded = np.pad(clean.astype(np.uint8) * 255, 1)
    cv2.floodFill(flooded, None, (0, 0), 128, flags=4)
    holes = (flooded[1:-1, 1:-1] == 0) & ~clean
    preview = rgb.copy()
    contours, hierarchy = cv2.findContours(clean.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        # Draw child contours only in blue; a second green outline would hide holes.
        for is_hole, color in ((False, (0, 255, 0)), (True, (0, 0, 255))):
            selected = [contour for contour, relation in zip(contours, hierarchy[0])
                        if (relation[3] >= 0) == is_hole]
            cv2.drawContours(preview, selected, -1, color, 1, lineType=cv2.LINE_8)
    return preview, holes


def fill_small_holes(mask, max_area):
    """Fill enclosed 8-connected background components of at most max_area pixels.

    Never fill components touching an image border. A zero limit disables filling.
    Areas are mask pixels; this bounded raster policy is not CLAM's contour-area
    or max_n_holes policy. Large holes are always preserved.
    """
    validate_mask(mask)
    if not isinstance(max_area, (int, np.integer)) or max_area < 0:
        raise ValueError("max_area must be a nonnegative integer in mask pixels")
    if max_area == 0:
        return mask.copy()
    _, labels, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=8)
    fill = stats[:, cv2.CC_STAT_AREA] <= max_area
    fill[0] = False
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    fill[border_labels] = False
    return mask | fill[labels]


def fill_background(rgb, tissue_mask, background_rgb):
    """Copy RGB and replace non-tissue pixels with one explicit integer RGB colour."""
    validate_rgb(rgb)
    validate_mask(tissue_mask, rgb.shape[:2])
    colour = np.asarray(background_rgb)
    if colour.shape != (3,) or colour.dtype.kind not in "iu" or np.any((colour < 0) | (colour > 255)):
        raise ValueError("background_rgb must contain three integers in 0-255")
    result = rgb.copy()
    result[~tissue_mask] = colour
    return result


def mean_background_color(rgb, background_mask):
    """Return floating-point mean RGB of caller-selected background pixels.

    The caller defines background and artifact exclusions; empty samples fail.
    Round with np.rint(...).astype(np.uint8) when an integer fill colour is needed.
    """
    validate_rgb(rgb)
    validate_mask(background_mask, rgb.shape[:2])
    if not background_mask.any():
        raise ValueError("No background pixels selected for colour estimation")
    return rgb[background_mask].mean(axis=0)


def white_background(rgb, tissue_mask):
    """Compatibility helper for the white working images used by masking/QC."""
    return fill_background(rgb, tissue_mask, (255, 255, 255))

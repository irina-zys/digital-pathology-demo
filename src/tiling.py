"""Step 4: physical calibration, bounded grids, mask coverage and pixel reads.

Locations are level-0 (x, y) top-left corners. Sizes are (width, height),
rectangles are half-open, arrays index [y, x], and mask True selects pixels.
"""

import contextlib
import importlib
import io
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.image_utils import validate_mask, validate_rgb


def _size_pair(value, name):
    values = np.asarray((value, value) if np.isscalar(value) else value)
    if values.shape != (2,) or values.dtype.kind not in "iu" or np.any(values <= 0):
        raise ValueError(f"{name} must be a positive integer or (width, height) integer pair")
    return tuple(map(int, values))


def calibrated_patch_size(output_size, target_mpp, native_mpp):
    """Return source (width, height) and effective output MPP after integer rounding.

    This general geometry utility permits anisotropic MPP. Format adapters may
    require square pixels. Reject upsampling beyond the existing 1% tolerance.
    """
    output = np.asarray(_size_pair(output_size, "output_size"))
    native = np.asarray(native_mpp, dtype=float)
    if native.shape != (2,) or not np.isfinite(native).all() or np.any(native <= 0):
        raise ValueError("Positive finite native x/y MPP is required")
    if not np.isfinite(target_mpp) or target_mpp <= 0:
        raise ValueError("target_mpp must be positive and finite")
    if np.any(native > target_mpp * 1.01):
        raise ValueError("Native MPP is coarser than target_mpp; unavailable detail cannot be recovered")
    source = np.maximum(1, np.rint(output * target_mpp / native)).astype(np.int64)
    return tuple(map(int, source)), (source * native / output).tolist()


def grid_coordinates(dimensions, size, *, max_candidates):
    """Return a bounded non-overlapping row-major int64 N x 2 grid; omit edge strips."""
    width, height = _size_pair(dimensions, "dimensions")
    sw, sh = _size_pair(size, "size")
    if not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    nx, ny = width // sw, height // sh
    if nx * ny > max_candidates:
        raise ValueError("Patch grid exceeds max_candidates; raise the explicit budget or disable export")
    xx, yy = np.meshgrid(np.arange(nx) * sw, np.arange(ny) * sh)
    return np.column_stack((xx.ravel(), yy.ravel())).astype(np.int64)


def mask_bounds(coords, dimensions, size, mask_shape):
    """Map in-bounds level-0 rectangles to conservative (x0,y0,x1,y1) mask bounds.

    Floor left/top, ceil right/bottom; coverage is approximate at mask resolution.
    Independent x/y scale factors accommodate thumbnail dimension rounding.
    """
    width, height = _size_pair(dimensions, "dimensions")
    sw, sh = _size_pair(size, "size")
    mh, mw = _size_pair(mask_shape, "mask_shape")
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.dtype.kind not in "iu":
        raise ValueError("coords must be an integer N x 2 array")
    if np.any(coords < 0) or np.any(coords > np.array([width - sw, height - sh])):
        raise ValueError("Patch coordinates are outside the slide")
    x0 = np.floor(coords[:, 0] * mw / width).astype(int)
    y0 = np.floor(coords[:, 1] * mh / height).astype(int)
    x1 = np.minimum(mw, np.ceil((coords[:, 0] + sw) * mw / width).astype(int))
    y1 = np.minimum(mh, np.ceil((coords[:, 1] + sh) * mh / height).astype(int))
    return x0, y0, x1, y1


def tissue_coverage(mask, coords, dimensions, size):
    """Return mask-selected fractions per rectangle using an integral image."""
    validate_mask(mask)
    x0, y0, x1, y1 = mask_bounds(coords, dimensions, size, mask.shape)
    integral = cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_64F)
    areas = (x1 - x0) * (y1 - y0)
    return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / areas


def coordinates(mask, dimensions, size, limit, min_fraction, max_candidates):
    """Existing export API: (eligible coords, fractions, count before optional cap)."""
    if not np.isfinite(min_fraction) or not 0 <= min_fraction <= 1:
        raise ValueError("min_fraction must be finite and in 0-1")
    if limit is not None and (not isinstance(limit, (int, np.integer)) or limit < 1):
        raise ValueError("limit must be None or a positive integer")
    coords = grid_coordinates(dimensions, size, max_candidates=max_candidates)
    coverage = tissue_coverage(mask, coords, dimensions, size)
    valid = np.flatnonzero(coverage >= min_fraction)
    count = len(valid)
    if limit is not None and count > limit:
        valid = valid[np.linspace(0, count - 1, limit, dtype=int)]
    return coords[valid], coverage[valid], count


def read_tile(reader, xy, source_size, output_size, *, max_read_pixels):
    """Read one level-0 region from a CLAM/OpenSlide handle and resize with Lanczos.

    The caller owns the reader lifetime and supplies coordinates from an in-bounds
    grid. A hard pixel budget is checked before reading; no automatic colour or
    stain correction is applied. Return uint8 RGB H x W x 3.
    """
    sw, sh = _size_pair(source_size, "source_size")
    ow, oh = _size_pair(output_size, "output_size")
    location = np.asarray(xy)
    if location.shape != (2,) or location.dtype.kind not in "iu" or np.any(location < 0):
        raise ValueError("xy must be a nonnegative integer (x, y) pair")
    if not isinstance(max_read_pixels, (int, np.integer)) or max_read_pixels < 1:
        raise ValueError("max_read_pixels must be a positive integer")
    if sw * sh > max_read_pixels or ow * oh > max_read_pixels:
        raise ValueError("Source or output tile exceeds max_read_pixels")
    slide = reader.wsi if hasattr(reader, "wsi") else reader
    if np.any(location + (sw, sh) > np.asarray(slide.dimensions)):
        raise ValueError("Tile extends beyond the slide; padding is disabled")
    region = slide.read_region(tuple(map(int, location)), 0, (sw, sh))
    if region.getchannel("A").getextrema() != (255, 255):
        raise ValueError("Tile contains missing/transparent source pixels")
    patch = np.asarray(region.convert("RGB"))
    validate_rgb(patch)
    if patch.shape != (sh, sw, 3):
        raise RuntimeError("Unexpected source patch dimensions")
    return np.asarray(Image.fromarray(patch).resize((ow, oh), Image.Resampling.LANCZOS)).copy()


CLAM_COMMIT = "53e2409d4a8189c682c173382964a85f114f923c"


def _clam_class():
    vendor = Path(__file__).resolve().parents[1] / "third_party" / "CLAM"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    image_pixel_limit = Image.MAX_IMAGE_PIXELS
    try:
        module = importlib.import_module("wsi_core.WholeSlideImage")
    finally:
        # Upstream changes Pillow's process-wide limit at import; keep our caller's policy.
        Image.MAX_IMAGE_PIXELS = image_pixel_limit
    if not Path(module.__file__).resolve().is_relative_to(vendor):
        raise ImportError("A different wsi_core package is loaded; use a fresh project kernel")
    return module.WholeSlideImage


def open_clam_slide(path):
    """Open upstream CLAM's WholeSlideImage; caller must close slide.wsi."""
    return _clam_class()(str(path))


def clam_contours(mask, dimensions):
    """Convert the final raster mask to CLAM's level-0 contours and child holes.

    Uses RETR_CCOMP and independent x/y scale factors with CLAM's own scaling
    functions. No second tissue segmentation or hole-size filter is applied.
    """
    validate_mask(mask)
    width, height = _size_pair(dimensions, "dimensions")
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], []
    parents = hierarchy[0, :, 3]
    outer = np.flatnonzero(parents == -1)
    tissue = [contours[i] for i in outer]
    holes = [[contours[j] for j in np.flatnonzero(parents == i)] for i in outer]
    scale = np.asarray([width / mask.shape[1], height / mask.shape[0]])
    cls = _clam_class()
    return cls.scaleContourDim(tissue, scale), cls.scaleHolesDim(holes, scale)


def clam_coordinates(slide, mask, size, *, limit=None, min_fraction=0.6,
                     max_candidates=200_000, contour_fn="four_pt", max_contours=128,
                     exclude_hole_centers=True):
    """Call upstream CLAM process_contour, then apply explicit mask-coverage QC.

    CLAM uses contour-origin grids, x-major order, four-point/centre checks and
    optional patch-centre hole exclusion. Holes always remain excluded from mask
    coverage. Padding is disabled. All grids are budgeted
    before calling CLAM, which uses at most four coordinate workers per contour.
    Reads remain sequential. Duplicates across contours are removed stably.
    Returns (coords, coverage, eligible_count_before_optional_limit).
    """
    size, sy = _size_pair(size, "size")
    if size != sy:
        raise ValueError("Upstream CLAM tiling requires square source patches")
    if not isinstance(exclude_hole_centers, (bool, np.bool_)):
        raise ValueError("exclude_hole_centers must be a boolean")
    if contour_fn not in {"four_pt", "four_pt_hard", "center", "basic"}:
        raise ValueError("Unknown CLAM contour check")
    if not np.isfinite(min_fraction) or not 0 <= min_fraction <= 1:
        raise ValueError("min_fraction must be in 0-1")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be None or a positive integer")
    if not isinstance(max_candidates, int) or max_candidates < 1 or not isinstance(max_contours, int) or max_contours < 1:
        raise ValueError("Candidate and contour budgets must be positive integers")
    dimensions = tuple(slide.level_dim[0])
    contours, holes = clam_contours(mask, dimensions)
    if len(contours) > max_contours:
        raise ValueError("CLAM contour count exceeds max_contours; review the mask or raise the explicit budget")
    total_candidates = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        stop_x, stop_y = min(x + w, dimensions[0] - size + 1), min(y + h, dimensions[1] - size + 1)
        total_candidates += len(range(x, stop_x, size)) * len(range(y, stop_y, size))
    if total_candidates > max_candidates:
        raise ValueError("CLAM contour grids exceed max_candidates")
    batches = []
    for contour, contour_holes in zip(contours, holes, strict=True):
        x, y, w, h = cv2.boundingRect(contour)
        if min(x + w, dimensions[0] - size + 1) <= x or min(y + h, dimensions[1] - size + 1) <= y:
            continue
        with contextlib.redirect_stdout(io.StringIO()) as messages:
            assets, _ = slide.process_contour(contour, contour_holes if exclude_hole_centers else [],
                                              patch_level=0, save_path="",
                                              patch_size=size, step_size=size, contour_fn=contour_fn,
                                              use_padding=False)
        logging.getLogger(__name__).debug(messages.getvalue())
        if "coords" in assets:
            batches.append(np.asarray(assets["coords"], dtype=np.int64))
    coords = np.concatenate(batches) if batches else np.empty((0, 2), dtype=np.int64)
    if len(coords):
        _, indices = np.unique(coords, axis=0, return_index=True)
        coords = coords[np.sort(indices)]
    fractions = tissue_coverage(mask, coords, dimensions, size)
    valid = np.flatnonzero(fractions >= min_fraction)
    count = len(valid)
    if limit is not None and count > limit:
        valid = valid[np.linspace(0, count - 1, limit, dtype=int)]
    return coords[valid], fractions[valid], count


def contour_tile_mask(contours, holes, xy, source_size, output_size):
    """Rasterize level-0 tissue/child-hole contours into a bounded tile mask.

    Uses the same contours as CLAM. Holes are excluded; nested tissue islands
    remain through component union. Thumbnail-derived borders are approximate.
    """
    sw, sh = _size_pair(source_size, "source_size")
    ow, oh = _size_pair(output_size, "output_size")
    if ow * oh > 1_000_000:
        raise ValueError("Contour preview exceeds one million output pixels")
    origin = np.asarray(xy)
    if origin.shape != (2,) or origin.dtype.kind not in "iu" or np.any(origin < 0):
        raise ValueError("xy must be a nonnegative integer (x, y) pair")
    scale = np.array([ow / sw, oh / sh])
    result = np.zeros((oh, ow), dtype=bool)
    for outer, inner in zip(contours, holes, strict=True):
        component = np.zeros((oh, ow), dtype=np.uint8)
        projected = [np.rint((contour - origin) * scale).astype(np.int32)
                     for contour in [outer, *inner]]
        cv2.drawContours(component, projected[:1], -1, 1, cv2.FILLED)
        for hole in projected[1:]:
            cv2.drawContours(component, [hole], -1, 0, cv2.FILLED)
        result |= component.astype(bool)
    return result

"""CLAM four-point contour tiling followed by retained-mask coverage QC.

Independent implementation of ``process_contour`` and ``isInContourV3_Easy``:
https://github.com/mahmoodlab/CLAM/blob/53e2409d4a8189c682c173382964a85f114f923c/wsi_core/WholeSlideImage.py
https://github.com/mahmoodlab/CLAM/blob/53e2409d4a8189c682c173382964a85f114f923c/wsi_core/util_classes.py

The input is already the tissue mask after artefact filtering. Unlike upstream
segmentation defaults, no additional contour-area threshold or hole cap is used.
"""

import cv2
import numpy as np

from src.image_utils import validate_mask
from src.tiling import tissue_coverage


def clam_coordinates(mask, dimensions, size, limit, min_fraction, max_candidates):
    """Return level-0 coordinates, mask fractions and eligible count before capping.

    Use contour-anchored, x-major square grids without padding; accept any of the
    four inner probes and reject centers strictly inside associated holes. The
    candidate budget covers all contour grids before deduplication or filtering.
    Coverage QC and an optional evenly spaced cap follow CLAM eligibility.
    """
    validate_mask(mask)
    dimensions = np.asarray(dimensions)
    if (dimensions.shape != (2,) or dimensions.dtype.kind not in "iu" or
            np.any(dimensions <= 0) or np.any(dimensions > np.iinfo(np.int32).max)):
        raise ValueError("dimensions must be a positive integer (width, height) pair within int32")
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer square patch width")
    if not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if limit is not None and (not isinstance(limit, (int, np.integer)) or limit < 1):
        raise ValueError("limit must be None or a positive integer")
    if not np.isfinite(min_fraction) or not 0 <= min_fraction <= 1:
        raise ValueError("min_fraction must be finite and in 0-1")
    width, height = map(int, dimensions)
    size = int(size)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    scale = np.array([width / mask.shape[1], height / mask.shape[0]])
    contours = [np.asarray(contour * scale, dtype=np.int32) for contour in contours]
    grids = []
    candidates = 0
    if hierarchy is not None:
        parents = hierarchy[0, :, 3]
        for index in np.flatnonzero(parents == -1):
            contour = contours[index]
            if len(contour) < 3 or cv2.contourArea(contour) <= 0:
                continue
            holes = [contours[i] for i in np.flatnonzero(parents == index)
                     if len(contours[i]) >= 3 and cv2.contourArea(contours[i]) > 0]
            x, y, w, h = cv2.boundingRect(contour)
            xs = range(x, min(x + w, width - size + 1), size)
            ys = range(y, min(y + h, height - size + 1), size)
            candidates += len(xs) * len(ys)
            if candidates > max_candidates:
                raise ValueError("CLAM contour grids exceed max_candidates; raise the explicit budget or disable export")
            grids.append((contour, holes, xs, ys))

    half = size // 2
    shift = int(half * 0.5)
    offsets = ((-shift, -shift), (shift, shift), (shift, -shift), (-shift, shift)) if shift else ((0, 0),)
    accepted, seen = [], set()
    for contour, holes, xs, ys in grids:
        for x in xs:
            for y in ys:
                point = (x, y)
                if point in seen:
                    continue
                if not any(cv2.pointPolygonTest(contour, (float(x + half + dx), float(y + half + dy)), False) >= 0
                           for dx, dy in offsets):
                    continue
                # CLAM uses a floating half for hole centers, unlike its four-point probes.
                if any(cv2.pointPolygonTest(hole, (x + size / 2, y + size / 2), False) > 0 for hole in holes):
                    continue
                seen.add(point)
                accepted.append(point)
    coords = np.asarray(accepted, dtype=np.int64).reshape(-1, 2)
    coverage = tissue_coverage(mask, coords, (width, height), size)
    valid = np.flatnonzero(coverage >= min_fraction)
    count = len(valid)
    if limit is not None and count > limit:
        valid = valid[np.linspace(0, count - 1, limit, dtype=int)]
    return coords[valid], coverage[valid], count

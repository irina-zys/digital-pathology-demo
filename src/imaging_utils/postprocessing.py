"""Prediction-only instance-mask postprocessing; zero denotes background."""

import numpy as np


def filter_small_instances(labels, min_area, *, protect_edges=True):
    """Remove IDs smaller than min_area input pixels, preserving IDs, dtype and input.

    Frame-touching small objects are retained by default because their visible area
    may be truncated. protected_ids lists only small IDs retained by that rule.
    No annotations or pixel-to-physical calibration enter this operation.
    """
    labels = np.asarray(labels)
    if (labels.ndim != 2 or not labels.size or labels.dtype.kind not in 'iu'
            or np.any(labels < 0)):
        raise ValueError('Expected a nonempty 2D mask of nonnegative integer instance IDs')
    if (isinstance(min_area, (bool, np.bool_)) or not isinstance(min_area, (int, np.integer))
            or min_area < 0):
        raise ValueError('min_area must be a nonnegative integer in input pixels')
    if not isinstance(protect_edges, (bool, np.bool_)):
        raise ValueError('protect_edges must be boolean')
    ids, counts = np.unique(labels, return_counts=True)
    areas = {int(i): int(count) for i, count in zip(ids, counts) if i}
    border = set(map(int, np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))) - {0}
    small = {i for i, area in areas.items() if area < min_area}
    protected = small & border if protect_edges else set()
    removed = sorted(small - protected)
    filtered = labels.copy()
    filtered[np.isin(labels, removed)] = 0
    return {'instance_mask': filtered, 'removed_ids': removed, 'protected_ids': sorted(protected),
            'edge_ids': sorted(border), 'areas_pixels': areas, 'min_area_pixels': int(min_area),
            'protect_edges': bool(protect_edges)}

"""Bounded instance matching for annotated patches; no model or filesystem state."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.metrics.detection import detection_metrics


def _ids(labels):
    return [int(v) for v in np.unique(labels) if v]


def edge_ids(labels):
    """Original object IDs intersecting the patch frame (not the ambiguity boundary)."""
    return _ids(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))


def compare_instances(reference, prediction, ambiguous=None, *, iou_threshold=0.5, max_instances=1024):
    """Match crop-local instances one-to-one after masking ambiguous pixels.

    IDs with no included pixels are omitted, partial IDs are retained without
    relabelling, and frame-touching objects remain eligible but are reported.
    Assignment maximizes the number of pairs at/above the IoU threshold, then
    total IoU. Counts, foreground Dice and matching use the same valid pixels.
    This is an illustrative patch metric, not a full-slide benchmark protocol.
    """
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if (reference.ndim != 2 or reference.shape != prediction.shape or not reference.size
            or reference.size > 1_048_576):
        raise ValueError("Expected matched nonempty 2D masks with at most 1,048,576 pixels")
    if any(x.dtype.kind not in "iu" or np.any(x < 0) for x in (reference, prediction)):
        raise ValueError("Instance IDs must be nonnegative integers")
    if not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    if type(max_instances) is not int or not 1 <= max_instances <= 4096:
        raise ValueError("max_instances must be in 1-4096")
    if ambiguous is None:
        ambiguous = np.zeros(reference.shape, dtype=bool)
    if ambiguous.shape != reference.shape or ambiguous.dtype != bool:
        raise ValueError("Ambiguous mask must be a matching boolean array")
    valid = ~ambiguous
    r = np.where(valid, reference, 0)
    p = np.where(valid, prediction, 0)
    rids, pids = _ids(r), _ids(p)
    nr, npred = len(rids), len(pids)
    if max(nr, npred) > max_instances:
        raise ValueError("Instance matching exceeds the explicit object budget")
    # Compact indices preserve arbitrary/gapped IDs without allocating by maximum ID.
    ri = np.searchsorted([0, *rids], r.ravel())
    pi = np.searchsorted([0, *pids], p.ravel())
    table = np.bincount(ri * (npred + 1) + pi, minlength=(nr + 1) * (npred + 1)).reshape(nr + 1, npred + 1)
    intersection = table[1:, 1:]
    union = table[1:].sum(1)[:, None] + table[:, 1:].sum(0)[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
    matches = []
    if nr and npred:
        eligible = iou >= iou_threshold
        rows, cols = linear_sum_assignment(eligible * (min(nr, npred) + 1 + iou), maximize=True)
        matches = [dict(reference_id=rids[a], prediction_id=pids[b], iou=float(iou[a, b]))
                   for a, b in zip(rows, cols) if eligible[a, b]]
    matched_r = {m['reference_id'] for m in matches}
    matched_p = {m['prediction_id'] for m in matches}
    tp = len(matches)
    fg_sum = int(np.count_nonzero(r) + np.count_nonzero(p))
    report = dict(
        reference_all=len(_ids(reference)), prediction_all=len(_ids(prediction)),
        reference_evaluated=nr, prediction_evaluated=npred,
        reference_ignored_ids=sorted(set(_ids(reference)) - set(rids)),
        prediction_ignored_ids=sorted(set(_ids(prediction)) - set(pids)),
        reference_edge_ids=sorted(set(edge_ids(reference)) & set(rids)),
        prediction_edge_ids=sorted(set(edge_ids(prediction)) & set(pids)),
        included_pixels=int(valid.sum()), excluded_pixels=int(ambiguous.sum()),
        matched=tp, missed=nr-tp, extra=npred-tp, matches=matches,
        missed_ids=sorted(set(rids)-matched_r), extra_ids=sorted(set(pids)-matched_p),
        **detection_metrics(tp, npred-tp, nr-tp),
        matched_iou_mean=float(np.mean([m["iou"] for m in matches])) if matches else None,
        foreground_dice=2*int(np.count_nonzero((r > 0) & (p > 0)))/fg_sum if fg_sum else None,
        iou_threshold=iou_threshold,
        policy="Ignore ambiguous pixels; omit empty IDs; retain partial and frame-touching crop objects",
    )
    assert tp + report['missed'] == nr and tp + report['extra'] == npred
    assert len(matched_r) == len(matched_p) == tp
    return report


def comparison_row(name, model, report):
    """Flat, consistently scoped values for notebook tables and CSV export."""
    fields = ('reference_all', 'prediction_all', 'reference_evaluated', 'prediction_evaluated',
              'included_pixels', 'excluded_pixels', 'matched', 'missed', 'extra',
              'precision', 'recall', 'instance_f1', 'foreground_dice', 'false_negative_rate',
              'false_discovery_rate', 'false_positive_rate', 'false_positives_per_patch',
              'false_positives_per_mm2', 'count_error', 'absolute_count_error', 'matched_iou_mean')
    return {'patch': name, 'model': model, **{key: report[key] for key in fields},
            'reference_edge': len(report['reference_edge_ids']), 'prediction_edge': len(report['prediction_edge_ids'])}

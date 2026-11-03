"""Count-based detection metrics and aggregation; independent of models and datasets."""

from collections import defaultdict
from numbers import Integral

import numpy as np


RATE_FIELDS = ('precision', 'recall', 'false_negative_rate', 'false_discovery_rate',
               'instance_f1', 'false_positive_rate', 'false_positives_per_patch',
               'false_positives_per_mm2', 'count_bias_per_patch')


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def detection_metrics(tp, fp, fn, *, n_patches=1, tn=None, evaluated_area_mm2=None, area_verified=False):
    """Compute object-level rates from counts; undefined rates are None.

    ``tn`` is meaningful only for a separately defined negative-candidate universe,
    never background pixels in an instance mask. Area rates require the caller's
    explicit verification of physical scale and the evaluated (nonignored) area.
    Counts may represent one patch or pooled patches. Absolute pooled count error
    is not patch MAE; use ``aggregate_detection_metrics`` for the latter.
    """
    tp, fp, fn = (_count(value, name) for value, name in ((tp, 'tp'), (fp, 'fp'), (fn, 'fn')))
    n_patches = _count(n_patches, 'n_patches')
    if not n_patches:
        raise ValueError('n_patches must be positive')
    if tn is not None:
        tn = _count(tn, 'tn')
    if type(area_verified) is not bool:
        raise ValueError('area_verified must be boolean')
    if evaluated_area_mm2 is not None:
        if (not area_verified or isinstance(evaluated_area_mm2, bool)
                or not np.isfinite(evaluated_area_mm2) or evaluated_area_mm2 <= 0):
            raise ValueError('Area must be positive, finite and explicitly verified')
        evaluated_area_mm2 = float(evaluated_area_mm2)
    elif area_verified:
        raise ValueError('Provide the verified evaluated area')
    return dict(
        true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn,
        n_patches=n_patches, evaluated_area_mm2=evaluated_area_mm2,
        precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
        false_negative_rate=_ratio(fn, tp + fn), false_discovery_rate=_ratio(fp, tp + fp),
        instance_f1=_ratio(2 * tp, 2 * tp + fp + fn),
        false_positive_rate=_ratio(fp, fp + tn) if tn is not None else None,
        false_positives_per_patch=fp / n_patches,
        false_positives_per_mm2=fp / evaluated_area_mm2 if evaluated_area_mm2 is not None else None,
        count_error=fp - fn, absolute_count_error=abs(fp - fn), count_bias_per_patch=(fp - fn) / n_patches,
    )


def _pool(records):
    tn = [row.get('true_negative') for row in records]
    areas = [row.get('evaluated_area_mm2') for row in records]
    area = sum(areas) if all(value is not None for value in areas) else None
    result = detection_metrics(
        sum(row['true_positive'] for row in records), sum(row['false_positive'] for row in records),
        sum(row['false_negative'] for row in records), n_patches=len(records),
        tn=sum(tn) if all(value is not None for value in tn) else None,
        evaluated_area_mm2=area, area_verified=area is not None,
    )
    result['count_mae_per_patch'] = sum(abs(row['false_positive'] - row['false_negative']) for row in records) / len(records)
    result['patches_with_tn'] = sum(value is not None for value in tn)
    result['patches_with_verified_area'] = sum(value is not None for value in areas)
    return result


def _macro(records):
    result = {}
    for name in (*RATE_FIELDS, 'count_mae_per_patch'):
        values = [row[name] for row in records if row.get(name) is not None]
        result[name] = {'value': float(np.mean(values)) if values else None, 'contributing_units': len(values)}
    return result


def aggregate_detection_metrics(reports, *, patch_ids, group_ids=None):
    """Pool one model's per-patch reports and optionally give each known group equal weight.

    Requires unique patch IDs, one report per patch, and a common matching policy.
    ``group_ids`` must identify actual slides/patients (or another explicitly
    declared unit); no grouping is inferred. Macro means omit undefined rates and
    return the number of contributing units. Incomplete TN/area coverage makes
    the corresponding pooled rate unavailable. This helper computes descriptive
    statistics, not sampling weights or confidence intervals.
    """
    reports, patch_ids = list(reports), list(patch_ids)
    if not 1 <= len(reports) <= 10_000 or len(reports) != len(patch_ids):
        raise ValueError('Provide 1-10,000 patch reports with corresponding IDs')
    if any(not isinstance(p, str) or not p for p in patch_ids) or len(set(patch_ids)) != len(patch_ids):
        raise ValueError('Patch IDs must be nonempty, unique strings; do not combine repeat attempts')
    for key in ('iou_threshold', 'policy'):
        if len({row.get(key) for row in reports}) != 1:
            raise ValueError(f'Cannot aggregate differing {key} values')
    records = []
    for row in reports:
        if row.get('n_patches', 1) != 1:
            raise ValueError('Each report must describe exactly one patch')
        area = row.get('evaluated_area_mm2')
        record = detection_metrics(row['true_positive'], row['false_positive'], row['false_negative'],
            tn=row.get('true_negative'), evaluated_area_mm2=area, area_verified=area is not None)
        record['count_mae_per_patch'] = record['absolute_count_error']
        records.append(record)
    result = {'patch_ids': patch_ids, 'pooled': _pool(records), 'macro_patch': _macro(records),
              'groups': None, 'macro_group': None,
              'matching': {key: reports[0].get(key) for key in ('iou_threshold', 'policy')}}
    if group_ids is not None:
        group_ids = list(group_ids)
        if len(group_ids) != len(records) or any(not isinstance(g, str) or not g for g in group_ids):
            raise ValueError('Provide one nonempty, known group ID per patch')
        grouped = defaultdict(list)
        for group, record in zip(group_ids, records):
            grouped[group].append(record)
        result['groups'] = {group: _pool(rows) for group, rows in grouped.items()}
        result['macro_group'] = _macro(list(result['groups'].values()))
    return result

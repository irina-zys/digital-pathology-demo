# Nuclei segmentation evaluation

**Proposed next evaluation; not executed by this two-patch demo.** Score individual **nuclei** against annotated ground truth. Whole-cell evaluation needs whole-cell annotations and a model with that target; do not mix CytoNuke's nucleus and whole-cell categories.

### Select and freeze the sample before viewing predictions

A practical pilot is **25 annotated patches per dataset, seed 42, at most 3 patches per slide**. These are suggested workload limits, not a power calculation. Use fewer if insufficient eligible slides exist and report the actual sample size. Select slides/patients first, then patches without replacement; avoid overlapping crops. Stratify by organ and relevant acquisition conditions where metadata supports this, and report each dataset separately. Keep naturally sampled sparse or cell-free patches: they reveal false detections. Do not select examples by model performance.

Save a manifest containing dataset, image/crop ID, slide/patient ID when available, crop coordinates, annotation version, image/label hashes, ambiguity coverage, inclusion probability or sampling stratum, and verified pixel scale if available. Require complete nucleus annotation within the evaluation region; an unlabelled cell in a partially annotated region cannot safely be called a false positive. Record missing group metadata rather than treating patches as independent slides.

Freeze model weights, channels, diameter, thresholds, minimum-area/edge policies and the matching rule on a separate development set. Keep all patches from a patient/slide in one split, check documented pretraining overlap where possible, and disclose unresolved overlap. The polarity illustration above is development evidence, not an unbiased test-set result.

### Match nuclei, then report the error types

Use the same valid annotation region for both masks. Declare ambiguity and crop-edge handling before evaluation. This demo retains frame-touching nuclei without per-object text labels; for a primary analysis of intact nuclei, use a predefined interior scoring region with sufficient image/annotation context and a documented object-ownership rule. Report edge-object sensitivity separately rather than silently dropping difficult objects.

Match masks one-to-one with the declared criterion, initially **IoU ≥0.5**. A matched pair is **TP**; an unmatched annotated nucleus is **FN**; an unmatched prediction is **FP**. This still uses IoU to establish correspondence, but measures detection errors at the nucleus level. Splits and merges can create extra or missed objects; inspect them separately. The existing `src.metrics.compare_instances` returns these counts and IDs for each patch.

| Quantity | Definition | Interpretation |
|---|---|---|
| Recall / sensitivity | TP / (TP + FN) | Fraction of annotated nuclei found |
| Precision | TP / (TP + FP) | Fraction of predicted nuclei supported by annotation |
| False-negative rate (FNR) | FN / (TP + FN) = 1 − recall | Fraction of annotated nuclei missed |
| False-discovery rate (FDR) | FP / (TP + FP) = 1 − precision | Fraction of predictions that are false detections |
| Instance F1 | 2TP / (2TP + FP + FN) | Balance of precision and recall |
| False positives per patch | ΣFP / number of evaluated patches | False detections per sampled field; also report evaluated area |
| Count error | (TP + FP) − (TP + FN) = FP − FN | Over/undercount; also report absolute error |

Precision, recall and F1 follow the standard count definitions in the [scikit-learn metric documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html). Keep zero-denominator rates unavailable and report the number of contributing patches; cell-free patches still contribute FP counts and false positives per patch.

**“False-positive rate” needs a denominator.** Conventional FPR = FP / (FP + TN) requires a defined set of true-negative objects. Open-ended nucleus detection has no natural count of “non-nuclei”; background pixels are not cell-level TN. Report FDR and false positives per patch, or FP per mm² of evaluated area **only when physical scale is verified**. A formal FPR is possible for a separately defined, annotated candidate-classification task. This distinction and the need to separate object detection from mask overlap follow [Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z).

### Aggregate and inspect, without hiding heterogeneity

Save one record per patch/model with TP, FP, FN, evaluated objects/area, excluded pixels, crop-edge counts, matching threshold and source IDs. Report **pooled** rates from summed counts and **macro** summaries giving each slide equal weight; include per-dataset/organ results and distributions of per-patch errors. If strata are deliberately oversampled, retain stratum-specific results or use design weights for population estimates.

For paired model comparisons, use the exact same patches and bootstrap patients/slides together with their patches (for example, 1,000 resamples for 95% intervals). Do not bootstrap individual nuclei as independent observations. With too few known independent groups, report descriptive uncertainty and the grouping limitation rather than a precise generalization claim. This respects the hierarchical-data warning in [Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z).

Alongside detection metrics, retain foreground Dice and the distribution of **matched-object** IoU, explicitly noting that the latter excludes missed/extra objects. Audit a prespecified random sample of FP/FN, plus split/merge and boundary cases, against the RGB and annotated ground truth. If localization alone matters, predeclare a separate centroid-distance criterion in verified units; do not adjust matching after seeing test results.

### Reusable helpers in `src/metrics`

`compare_instances` matches masks and returns all per-patch metrics above, including FNR, FDR, count error and mean matched-object IoU. `detection_metrics(tp, fp, fn)` works directly from counts; supplying `tn` enables conventional FPR, and verified evaluated area enables FP/mm².

For a frozen set of predictions from **one model and one matching policy**:

```python
from src.metrics import aggregate_detection_metrics

summary = aggregate_detection_metrics(
    patch_reports,             # one compare_instances result per selected patch
    patch_ids=selected_ids,    # unique image/crop IDs; repeated attempts are rejected
    group_ids=known_slide_ids, # omit if slide/patient identity is unavailable
)
summary['pooled']             # summed TP/FP/FN, rates, count bias and patch MAE
summary['macro_group']        # equal group weights, with contributing-group counts
```

This is a usage template, not an executed subsample evaluation. The helper also returns per-group and macro-patch summaries; it does not infer grouping, sample patches, apply design weights, or compute bootstrap intervals. Those remain part of the recommended evaluation protocol.

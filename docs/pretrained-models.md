# Pretrained patch models

[README](../README.md) · [Annotation conversion](annotation-conversion.md) ·
[Model components and Lightning](model-components.md)

The [annotation and segmentation demo](../notebooks/annotation_segmentation_demo.ipynb)
walks through predictions, errors and one controlled polarity experiment on two
real 256 × 256 patches, then exports annotations and provenance before the evaluation recommendations. It uses one complete CytoNuke image and one explicit centre crop of a
NuInsSeg image. Source images, labels and ambiguous masks remain unchanged.
The separate [preprocessing demo](../notebooks/preprocessing_demo.ipynb) covers
WSI tissue masks, patch coverage and stain comparisons.

## Environments and checkpoints

From the repository root:

```sh
uv sync --locked --extra pretrained --extra train
uv sync --locked --project environments/cellpose --python 3.12
uv run --locked --extra pretrained python scripts/download_pretrained.py --model hovernext
uv run --locked --extra pretrained python scripts/download_pretrained.py --model cellpose
```

The main `.venv` adds only the optional HoVer-NeXt dependencies. The separate
`environments/cellpose/.venv` locks Cellpose 3.1.1.2 with NumPy 2.0.2; this avoids
changing the main environment's NumPy version or mixing its two OpenCV packages.
The `train` extra retains Lightning for the independent training examples.
Select the main `.venv/bin/python` as the notebook kernel. No notebook cell
installs packages or starts an unrestricted download.

Checkpoints live only under the repository's ignored `data/models/` directory.
The explicit downloader verifies/reuses matching files, bounds downloads, records
provenance and refuses mismatched existing files. Weight pins are owned by
`src/models/pretrained/assets.py` and shared by preparation and inference. The
notebook inference paths require prepared local weights; acquisition is explicit.

| Model | Local checkpoint | Purpose and input contract |
| --- | --- | --- |
| Original UNI (TIMM comparison) | `data/models/huggingface/hub/models--MahmoodLab--UNI/` | One 1024-dimensional embedding; square uint8 RGB patch, resized to 224 with ImageNet normalization |
| HoVer-NeXt PanNuke Tiny fold 1 | `data/models/hovernext/pannuke_convnextv2_tiny_1/` | Nucleus instances and five PanNuke classes; exactly 256 × 256 RGB/255, 16-view test-time augmentation and released watershed thresholds |
| Classical Cellpose nuclei (3.1.1.2) | `data/models/cellpose/nucleitorch_0` and `size_nucleitorch_0.npy` | Nucleus instances; RGB patch at most 256 × 256; H&E demo uses inverted grayscale and image-based diameter estimation |

`download_pretrained.py --model cellpose` prepares both the segmentation checkpoint
and its matching official size-model asset. Estimation requires both verified
files; neither inference loader downloads missing weights.

UNI is gated and included in the [classification comparison](../notebooks/wsi_classification_demo.ipynb).
After provider approval, sign in using the standard local Hugging Face login and
prepare the pinned comparison assets:

```sh
.venv/bin/hf auth login
.venv/bin/python scripts/prepare_classification_models.py
```

The comparison loads original UNI through TIMM at revision
`b55a5ec6cade1a39edfe6534189a9b8ca7a022f0`, verifies its checkpoint digest and
compares a mean-pooled linear classifier with a learned attention-MIL classifier
on the same frozen features. An authorization failure
does not trigger a fallback model. Hugging Face manages its local download-resume
cache; interrupted downloads can retain partial data. Keep tokens and weights
out of Git. Authentication stays in the standard user cache, independently of
the project weight-cache path.

The older single-patch `load_uni` / `embed_uni` API remains available for an
explicit authorized checkpoint at `data/models/uni/pytorch_model.bin`, prepared by
`scripts/download_pretrained.py --model uni`. The WSI comparison uses the TIMM
weight loader in `src.models.encoders.pretrained`, composed with the matched
transform by `src.inference.feature_extraction.load_comparison_encoder`, and does
not require that duplicate. The weight-only loader returns no transform.
Single-patch UNI retains its square-image resize policy; the comparison explicitly
resizes every full tile to 224 × 224 before per-patch L2 normalization.

## Reusable functions

Model-specific APIs live in `src.models.pretrained`: `uni`, `hovernext`, and
`cellpose`. The Cellpose adapter launches `src.models.pretrained.cellpose_worker`
in its isolated environment. Loading an adapter never prepares model assets.
For example:

```python
import torch
from src.models.pretrained.hovernext import load_hovernext, predict_hovernext
from src.models.pretrained.cellpose import run_cellpose_patch

torch.set_num_threads(1)
model = load_hovernext("data/models/hovernext/pannuke_convnextv2_tiny_1", device="cpu")
result = predict_hovernext(rgb_256, model, max_pixels=65_536, tta_views=16, tta_seed=42)
labels = result["instance_mask"]  # uint32, zero background

cellpose = run_cellpose_patch(
    rgb_256, "outputs/cellpose-example",
    checkpoint="data/models/cellpose/nucleitorch_0",
    diameter=0, channels=(0, 0), invert=True, timeout_seconds=120,
)
```

Here `rgb_256` must be an existing uint8 RGB array with shape `(256, 256, 3)`.
Choose a new output directory for Cellpose. Its worker writes a lossless TIFF, a
normalized-channel preview and metadata; the adapter records RGB identity instead of retaining its temporary
PNG. Callers retain source file and crop coordinates alongside these outputs.
Pass `cache_dir` to put the child process Numba cache in a writable local directory.
The preview uses the pinned Cellpose library’s channel-selection and normalization
functions before diameter rescaling; it does not change network preprocessing.
The Cellpose loader verifies the pinned local assets before use. These digests
identify verified official downloads; they are locally recorded, not publisher-supplied checksums.

HoVer-NeXt defaults to the released PanNuke configuration's **16 test-time views**
and seed **42**. Augmentation settings come from the pinned inference code;
the view count and seed come from the released model configuration. Each view
uses flips/quarter-turns and HED colour-space
perturbations (scale 0.05, bias 0.03). The adapter follows the pinned upstream
spatial transform: bilinear forward sampling, nearest-neighbour inverse sampling
of logits, then per-view softmax and probability averaging before the released
PanNuke foreground/seed thresholds and watershed postprocessing. This is performed
in float32, one view at a time, to bound CPU memory; it adds no dependencies.
The seed fixes the local stochastic recipe, without promising bitwise equivalence
to upstream GPU execution. See the [pinned inference implementation](https://github.com/digitalpathologybern/hover_next_inference/blob/bde3c9d96172d16513e292d0a2dfcd7ff80202ff/src/inference.py#L207-L267),
[colour augmentation](https://github.com/digitalpathologybern/hover_next_inference/blob/bde3c9d96172d16513e292d0a2dfcd7ff80202ff/src/augmentations.py)
and [spatial augmentation](https://github.com/digitalpathologybern/hover_next_inference/blob/bde3c9d96172d16513e292d0a2dfcd7ff80202ff/src/spatial_augmenter.py).

This remains inference on one direct 256 × 256 patch, without surrounding slide
context, WSI overlap/stitching or physical-scale correction. PanNuke's 0.25 µm/pixel
training scale is unverified for the demo PNGs. A single seeded prediction does not
reproduce the release's five evaluation rounds or its full benchmark protocol.

`diameter=0` or `None` requests the matching official `SizeModel`: a style-based
estimate is refined using predicted objects from the same normalized, inverted
grayscale signal. Ground truth is never supplied to estimation. A positive manual
diameter is an explicit alternative; the generic API default of 17 pixels remains
a manual/historical setting, not automatic H&E calibration. Final segmentation
uses `niter=None`, allowing the pinned library's diameter-dependent dynamics default.
See the [official v3 diameter guidance](https://cellpose.readthedocs.io/en/v3.1.1.1/settings.html#diameter)
and [API](https://cellpose.readthedocs.io/en/v3.1.1.1/api.html).

The `diameter_estimation` record includes the method, selected and style diameters,
refinement object count and `fallback_to_training_mean`. If refinement finds no
objects, the run saves failure/calibration metadata and fails visibly rather than
presenting a fallback to 17 pixels as a successful estimate.

`load_uni(checkpoint)` strictly loads the pinned original UNI weights;
`embed_uni(rgb, loaded)` returns one finite float32 vector with shape `(1024,)`.
No randomly initialized weights are used as a pretrained substitute. Within the
isolated environment, `load_cellpose(checkpoint, checkpoint_sha256)` and
`run_cellpose(...)` are also directly reusable. See the
[worker instructions](../environments/cellpose/README.md).

## Demo components and interpretation

`src.segmentation_demo` owns preflight, bounded patch preparation, immutable
prediction attempts, plotting and the conversion/provenance appendix. The
notebook keeps parameters, teaching questions, calls and interpretation visible.
`src.evaluation.metrics.compare_instances` is a pure, reusable instance matcher;
`comparison_row` supplies consistently scoped table values.

Preflight verifies both local model environments and checkpoint hashes, checks
input files and tests a writable Numba cache before inference. The notebook's
initial repository preparation script obtains missing inputs first; preflight
itself performs no downloads. The fixed inputs remain the first CytoNuke image and the NuInsSeg
`human_bladder_01` centre crop at `(128, 128, 256, 256)`. Annotation preparation
bounds COCO JSON parsing to 64 MB and source images to 512 × 512 pixels.

A top-to-bottom run makes four baseline model calls plus one controlled Cellpose
polarity experiment on NuInsSeg. The primary H&E Cellpose calls use **inverted
grayscale** with diameter estimated from the image alone; HoVer-NeXt uses its
16-view published augmentation recipe and released watershed thresholds. The
negative control turns inversion **off** while
holding the primary run's calibrated diameter, RGB, weights, channels and thresholds
fixed. Both outcomes are retained. Original noninverted, 17-pixel outputs remain
historical failure examples, not the primary Cellpose comparison.

The notebook also compares raw nucleus predictions with a minimum-area filter
and an edge-aware version of that filter, before the input-polarity experiment.
The teaching cutoff is 128 prediction pixels; physical area is unverified for
these PNGs. This differs from the thumbnail-pixel tissue-component threshold in
preprocessing. Filtering uses predictions only; reference annotations are used
only for the unchanged evaluation rule. Raw predictions remain available.

Small predictions may be false positives or real small nuclei. Patch borders can
truncate objects, so protecting border-touching instances avoids an early size
rejection but can also retain false positives. For overlapping windows, merge or
deduplicate corresponding instances in common slide coordinates before final
size filtering/counting. Overlap alone does not resolve duplicates or missing
context. The notebook explains this policy; it does not implement WSI stitching.
See [the reusable instance filter](../src/imaging_utils/postprocessing.py).

Each `run_prediction` call creates a fresh directory under the run's `attempts/`,
including when a cell is re-executed. Each annotation-export execution creates a fresh
`reports/` directory. Failed attempts retain a failure record; earlier predictions
and original ground truth are never overwritten. The output root defaults to
ignored `outputs/annotation_segmentation_demo/`. Commit validated demo notebooks
with their matching executed outputs so readers can inspect the figures and
tables; keep the larger run directories outside Git.

### One evaluation region, two questions

- Foreground Dice compares included foreground pixels.
- Instance matching removes ambiguous pixels from both masks and omits IDs with
  no remaining pixels. Partial objects retain their original IDs.
- **Crop-edge objects are included** and explicitly counted; metrics
  describe visible crop objects, not reconstructed intact cells.
- Matching is one-to-one at IoU ≥0.5. Assignment maximizes match count first, then
  total eligible IoU. Matched/missed/extra counts and precision/recall/F1 all use
  the evaluated IDs. Undefined denominators remain unavailable.
- Tables distinguish all-crop counts from evaluated counts. Plots shade ignored
  regions, distinguish annotated ground truth/prediction outlines and explicitly
  show empty predictions, without per-object text labels.

CytoNuke category 0 nuclei remain separate from whole cells. Last-wins
rasterization resolves overlaps in reference TIFFs; GeoJSON retains original
polygons. NuInsSeg uses `label masks modify`, and its ambiguity mask stays separate.

Automated validation records its per-cell timeout in the execution receipt and
uses one PyTorch CPU thread. Cellpose also has a 120-second worker timeout. Interactive HoVer-NeXt
calls have an input-pixel bound but no wall-clock interrupt timer.

The refined notebook was executed on both real patches on 2026-11-03: all six
code cells and five prediction requests passed in 45 seconds on one CPU thread,
within a five-minute bound. The HoVer-NeXt adapter also matched the pinned
upstream CPU probabilities and final instance mask on the real NuInsSeg crop
at 16 views/seed 42. Input hashes and caller RNG states remained unchanged;
invalid TTA/seed/pixel bounds were rejected. Local execution receipts are under
`outputs/refined_segmentation_20261103/`; original runs remain untouched.

These two patches illustrate mechanics and failure inspection, not typing
accuracy, generalization or clinical performance. Both models receive their
model-specific inference recipe: HoVer-NeXt uses 16 augmented views and released
postprocessing, while Cellpose uses inverted grayscale and its matching size
estimator. Neither uses annotations during inference. The same RGB crop, ambiguity
mask and matching policy apply to both. Both augmentation and size estimation add
inference work and must be included when reporting end-to-end runtime; equal view
counts or equal pixel normalization are not a fairness requirement.

This revised recipe follows diagnosis of the original poor results on these two
annotated patches. Image-only estimation does not make that post-hoc choice an
independent benchmark. Do not select a winning diameter, polarity or model from
their annotation scores. This remains a **development comparison of declared
recipes**. For a held-out benchmark, give both models the same development split,
separate from the evaluation groups, and declare each model's tuning budget and
search space. Freeze model-specific preprocessing/augmentation/size policies and
thresholds, then evaluate once on the same held-out groups. Training-data overlap has not been ruled out. The common
crop, ambiguity exclusion and one-to-one matching rules remain unchanged.

## Sources and licences

- [Original UNI recipe](https://github.com/mahmoodlab/UNI/blob/main/README_old.md)
  and [model card/access conditions](https://huggingface.co/MahmoodLab/UNI):
  original ViT-L/16; CC-BY-NC-ND-4.0 and provider access conditions apply.
- [HoVer-NeXt source](https://github.com/digitalpathologybern/hover_next_inference/tree/bde3c9d96172d16513e292d0a2dfcd7ff80202ff)
  is pinned under `third_party/HoVerNeXt/`, retaining GPLv3 and adaptation notices.
  The [published weights](https://zenodo.org/records/10635618) are CC-BY-NC-SA-4.0.
- [Cellpose 3.1.1.2](https://github.com/MouseLand/cellpose/tree/v3.1.1.2)
  uses BSD-3-Clause code; its release notice separately describes noncommercial
  training-data restrictions for the nuclei model. Preserve those notices.

Model weights remain outside Git. SAM, MedSAM and Cellpose-SAM are not part of
this demo. The selected weights and settings are not interchangeable with models
that happen to produce similarly shaped outputs.

### Recommended evaluation beyond two examples

The [segmentation evaluation guide](segmentation-evaluation.md) describes a proposed,
unexecuted protocol for a frozen subsample of annotated patches: grouped sampling, nucleus-level TP/FP/FN,
precision/recall/F1, false-negative and false-discovery rates, false positives per
patch, count error, and paired uncertainty across patients/slides. It distinguishes
false-discovery rate from conventional false-positive rate, which requires a
defined true-negative population. Background pixels are not cell-level negatives.
See [Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z) for metric
selection and hierarchical evaluation pitfalls. Physical-area rates require
verified scale; the current two-patch run does not provide them.

### Generic metric API

- `src.evaluation.metrics.segmentation.compare_instances`: bounded one-to-one mask matching,
  ignored regions, edge IDs, Dice, matched-object IoU, TP/FP/FN and detection rates.
- `src.evaluation.metrics.detection.detection_metrics`: count-based precision, recall, FNR,
  FDR, F1, count error/bias and FP per patch. Conventional FPR requires explicit
  TN; FP/mm² requires positive evaluated area with `area_verified=True`.
- `src.evaluation.metrics.detection.aggregate_detection_metrics`: one report per unique
  patch for one model/policy; pooled rates, patch count MAE, macro-patch summaries
  and optional group-balanced summaries from explicit known group IDs. Undefined
  rates remain unavailable with contributing-unit counts; partial TN/area coverage
  does not silently shrink pooled denominators. No sampling, weighted-population
  inference or bootstrap confidence intervals are implemented by these helpers.

All functions are exported through `src.evaluation.metrics`; the obsolete
`src.segmentation` compatibility module has been removed. No model environment,
annotation format, notebook state or filesystem access is needed to compute the
instance/detection metrics. The training Dice class is imported explicitly from
`src.evaluation.metrics.foreground`.

## Frozen-encoder WSI comparison

The classification notebook now compares original UNI through TIMM, ImageNet
DINO ViT-S/16 and ImageNet ResNet-50 on complete eligible patch bags, with both
mean pooling and attention MIL for each encoder. The six fits share the patient
split and training budget; the notebook reports and interprets the two classifier
families separately. Use
`scripts/prepare_classification_models.py` for their verified canonical assets;
UNI's TIMM cache is under `data/models/huggingface/hub`. Authentication uses the
standard local Hugging Face login, independently of the weight-cache path.

UNI plus a trained slide head is a frozen-encoder classifier, not zero-shot
diagnosis prediction. See [training recipes](training.md#pretrained-classification-and-learning-recipes)
for the separate Lightning/Hydra pretraining, fine-tuning and scratch examples.

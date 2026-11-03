# Pretrained patch models

[README](../README.md) · [Annotation conversion](annotation-conversion.md) ·
[Model components and Lightning](model-components.md)

The [annotation and segmentation demo](../notebooks/annotation_segmentation_demo.ipynb)
walks through predictions, errors and one controlled polarity experiment on two
real 256 × 256 patches, then exports annotations and provenance in the final section. It uses one complete CytoNuke image and one explicit centre crop of a
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
provenance and refuses mismatched existing files. Loaders never download weights.

| Model | Local checkpoint | Purpose and input contract |
| --- | --- | --- |
| Original UNI (TIMM comparison) | `data/models/huggingface/hub/models--MahmoodLab--UNI/` | One 1024-dimensional embedding; square uint8 RGB patch, resized to 224 with ImageNet normalization |
| HoVer-NeXt PanNuke Tiny fold 1 | `data/models/hovernext/pannuke_convnextv2_tiny_1/` | Nucleus instances and five PanNuke classes; exactly 256 × 256 RGB, divided by 255 |
| Classical Cellpose nuclei (3.1.1.2) | `data/models/cellpose/nucleitorch_0` | Nucleus instances; RGB patch at most 256 × 256; explicit diameter and polarity settings |

UNI is gated and included in the [classification comparison](../notebooks/wsi_classification_demo.ipynb).
After provider approval, sign in using the standard local Hugging Face login and
prepare the pinned comparison assets:

```sh
.venv/bin/hf auth login
.venv/bin/python scripts/prepare_classification_models.py
```

The comparison loads original UNI through TIMM at revision
`b55a5ec6cade1a39edfe6534189a9b8ca7a022f0`, verifies its checkpoint digest and
trains only the attention pooling and slide classifier. An authorization failure
does not trigger a fallback model. Hugging Face manages its local download-resume
cache; interrupted downloads can retain partial data. Keep tokens and weights
out of Git. Authentication stays in the standard user cache, independently of
the project weight-cache path.

The older single-patch `load_uni` / `embed_uni` API remains available for an
explicit authorized checkpoint at `data/models/uni/pytorch_model.bin`, prepared by
`scripts/download_pretrained.py --model uni`. The WSI comparison uses the TIMM
loader in `src.models.pretrained_encoders` and does not require that duplicate.

## Reusable functions

The main-environment API is `src.pretrained`; model-specific implementations are
`src.hovernext` and the isolated `src.cellpose_worker`. For example:

```python
import torch
from src.pretrained import load_hovernext, predict_hovernext, run_cellpose_patch

torch.set_num_threads(1)
model = load_hovernext("data/models/hovernext/pannuke_convnextv2_tiny_1", device="cpu")
result = predict_hovernext(rgb_256, model, max_pixels=65_536)
labels = result["instance_mask"]  # uint32, zero background

cellpose = run_cellpose_patch(
    rgb_256, "outputs/cellpose-example",
    checkpoint="data/models/cellpose/nucleitorch_0",
    diameter=17, channels=(0, 0), invert=False, timeout_seconds=120,
)
```

Here `rgb_256` must be an existing uint8 RGB array with shape `(256, 256, 3)`.
Choose a new output directory for Cellpose. Its worker writes a lossless TIFF, a
normalized-channel preview and metadata; the adapter records RGB identity instead of retaining its temporary
PNG. Callers retain source file and crop coordinates alongside these outputs.
Pass `cache_dir` to put the child process Numba cache in a writable local directory.
The preview uses the pinned Cellpose library’s channel-selection and normalization
functions before diameter rescaling; it does not change network preprocessing.
The Cellpose loader verifies the pinned local checkpoint before use. Its digest
identifies a verified official download; it is locally recorded, not a publisher-supplied checksum.

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
`src.metrics.compare_instances` is a pure, reusable instance matcher;
`comparison_row` supplies consistently scoped table values.

Preflight verifies both local model environments and checkpoint hashes, checks
input files and tests a writable Numba cache before inference. The notebook's
initial repository preparation script obtains missing inputs first; preflight
itself performs no downloads. The fixed inputs remain the first CytoNuke image and the NuInsSeg
`human_bladder_01` centre crop at `(128, 128, 256, 256)`. Annotation preparation
bounds COCO JSON parsing to 64 MB and source images to 512 × 512 pixels.

A top-to-bottom run makes four baseline model calls plus one **predeclared
Cellpose inversion experiment** on NuInsSeg. The same RGB, weights, channels,
diameter=17 and thresholds are used; only intensity polarity changes. Both
outcomes are retained. This is not parameter selection on a benchmark.

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
See [the reusable instance filter](../src/postprocessing.py).

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

Automated execution uses a 120-second timeout per code cell and one PyTorch CPU
thread. Cellpose also has a 120-second worker timeout. Interactive HoVer-NeXt
calls have an input-pixel bound but no wall-clock interrupt timer.

These two patches illustrate mechanics and failure inspection, not typing
accuracy, generalization or clinical performance. HoVer-NeXt uses one direct
view, no TTA and upstream postprocessing. Its PanNuke 0.25 µm/pixel training scale
is not established for these PNGs. Cellpose uses the classical nuclei model and
its matching size estimator, not another Cellpose version. Size estimation adds
inference work and is included when reporting end-to-end runtime.

This revised recipe follows diagnosis of the original poor results on these two
annotated patches. Image-only estimation does not make that post-hoc choice an
independent benchmark. Do not select a winning diameter, polarity or model from
their annotation scores. Freeze preprocessing on a separate development set before
held-out evaluation; training-data overlap has not been ruled out. The common
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

- `src.metrics.segmentation.compare_instances`: bounded one-to-one mask matching,
  ignored regions, edge IDs, Dice, matched-object IoU, TP/FP/FN and detection rates.
- `src.metrics.detection.detection_metrics`: count-based precision, recall, FNR,
  FDR, F1, count error/bias and FP per patch. Conventional FPR requires explicit
  TN; FP/mm² requires positive evaluated area with `area_verified=True`.
- `src.metrics.detection.aggregate_detection_metrics`: one report per unique
  patch for one model/policy; pooled rates, patch count MAE, macro-patch summaries
  and optional group-balanced summaries from explicit known group IDs. Undefined
  rates remain unavailable with contributing-unit counts; partial TN/area coverage
  does not silently shrink pooled denominators. No sampling, weighted-population
  inference or bootstrap confidence intervals are implemented by these helpers.

All functions are exported through `src.metrics`; `src.segmentation` retains thin
compatibility imports. No model environment, annotation format, notebook state or
filesystem access is needed to compute metrics.

## Frozen-encoder WSI comparison

The classification notebook now compares original UNI through TIMM, ImageNet
DINO ViT-S/16 and ImageNet ResNet-50 on complete eligible patch bags. Use
`scripts/prepare_classification_models.py` for their verified canonical assets;
UNI's TIMM cache is under `data/models/huggingface/hub`. Authentication uses the
standard local Hugging Face login, independently of the weight-cache path.

UNI plus a trained slide head is a frozen-encoder classifier, not zero-shot
diagnosis prediction. See [training recipes](training.md#pretrained-classification-and-learning-recipes)
for the separate Lightning/Hydra pretraining, fine-tuning and scratch examples.

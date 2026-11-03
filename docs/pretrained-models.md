# Pretrained patch models

[README](../README.md) · [Annotation conversion](annotation-conversion.md) ·
[Model components and Lightning](model-components.md)

The [annotation and segmentation demo](../notebooks/annotation_segmentation_demo.ipynb)
converts existing annotations and runs pretrained inference on two real 256 × 256
patches. It uses one complete CytoNuke image and one explicit centre crop of a
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
| Original UNI | `data/models/uni/pytorch_model.bin` | One 1024-dimensional embedding; square uint8 RGB patch, resized to 224 with ImageNet normalization |
| HoVer-NeXt PanNuke Tiny fold 1 | `data/models/hovernext/pannuke_convnextv2_tiny_1/` | Nucleus instances and five PanNuke classes; exactly 256 × 256 RGB, divided by 255 |
| Cellpose 3 nuclei | `data/models/cellpose/nucleitorch_0` | Nucleus instances; RGB patch at most 256 × 256, explicit diameter/channels/inversion |

UNI is optional and gated. After the provider grants access to your Hugging Face
account, use your existing local authentication:

```sh
uv run --locked --extra pretrained python scripts/download_pretrained.py --model uni
```

The downloader pins revision `b55a5ec6cade1a39edfe6534189a9b8ca7a022f0`
and the checkpoint digest. An authorization failure does not trigger a fallback
model. After downloading, set `UNI_CHECKPOINT` in the notebook to
`PROJECT / "data/models/uni/pytorch_model.bin"`; its default is `None`. Without
an explicitly selected local authorized checkpoint, the notebook records UNI as
**not executed**. Hugging Face manages its local resume cache; an interrupted UNI
download can retain partial data there. Never put tokens or checkpoints in Git.

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
Choose a new output directory for Cellpose. Its worker writes a lossless TIFF and
metadata; the adapter records RGB identity instead of retaining its temporary
PNG. Callers retain source file and crop coordinates alongside these outputs.
The default expected Cellpose digest identifies the verified official download;
it is a locally recorded digest, not a publisher-supplied checksum.

`load_uni(checkpoint)` strictly loads the pinned original UNI weights;
`embed_uni(rgb, loaded)` returns one finite float32 vector with shape `(1024,)`.
No randomly initialized weights are used as a pretrained substitute. Within the
isolated environment, `load_cellpose(checkpoint, checkpoint_sha256)` and
`run_cellpose(...)` are also directly reusable. See the
[worker instructions](../environments/cellpose/README.md).

## What the notebook establishes

The notebook bounds conversion to one image per dataset and prediction to two
patches per segmentation model. It limits COCO JSON size before parsing, checks
mask alignment, saves source/crop identity, checkpoint hashes and settings, and
uses one PyTorch CPU thread. Validation runs use a 120-second timeout per code
cell; Cellpose additionally has a 120-second subprocess timeout. Direct
HoVer-NeXt calls have a pixel bound but no wall-clock timeout.

Outputs go in a fresh `outputs/annotation_segmentation_demo/<run>/` directory.
Nuclei are positive integer instance IDs, not connected-component guesses.
CytoNuke nucleus category 0 is kept separate from whole-cell category 1; explicit
last-wins rasterization resolves within-category overlaps while GeoJSON preserves
them. NuInsSeg uses the recommended modified instance labels, with ambiguous
pixels excluded from its illustrative foreground Dice calculation.

The two-patch comparison is a mechanics and visual inspection exercise. It does
not measure instance detection, typing accuracy, generalization or clinical
performance. HoVer-NeXt uses one view without test-time augmentation and the
published class-specific thresholds/postprocessing. PanNuke's 0.25 µm/pixel scale
is a training assumption; these PNG crops have no verified physical calibration.
Cellpose's 17-pixel diameter and grayscale channel are explicit demo settings,
not an H&E calibration. Cropping can truncate nuclei at the image boundary.

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

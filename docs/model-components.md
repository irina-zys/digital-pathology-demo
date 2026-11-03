# Model components and Lightning

[README](../README.md) · [Training](training.md) · [Pretrained inference](pretrained-models.md)

Models can share components when their tensor interfaces and learned assumptions
match. A matching Python class name or tensor width alone is insufficient. This
guide describes the current APIs, then distinguishes possible extensions.

## Composable components

| Component | Owner | Interface |
| --- | --- | --- |
| Image encoder | [`ImageEncoder`](../src/models/encoders/image.py) | RGB `B × 3 × H × W` → pooled features `B × D`; exposes `num_features` |
| Patch/image-bag classifier | [`ImageClassifier`](../src/models/networks/classification.py) | Injected encoder and optional pooling factory → class logits |
| Cached-feature MIL | [`MILHead`](../src/models/networks/classification.py) | One `1 × N × D` complete feature bag → `1 × K` logits |
| Mean/attention pooling | [`pooling`](../src/models/pooling) | `N × D` → pooled `1 × D` and `N` normalized weights |
| Binary segmentation | [`UNet`](../src/models/networks/unet.py) | RGB `B × 3 × H × W` → foreground logits `B × 1 × H × W` |
| Contrastive network | [`ContrastiveNetwork`](../src/models/networks/contrastive.py) | Injected encoder → projection head |
| Objectives | [`losses`](../src/models/losses) | Valid-pixel BCE and NT-Xent; classification uses PyTorch cross-entropy |
| Lightning task adapters | [`lightning`](../src/lightning) | Injected `net`, `loss` and optimizer factory; task-specific steps and metrics |

Metric definitions are owned by [`src/evaluation/metrics`](../src/evaluation/metrics).
Lightning updates and logs those metrics; training callbacks and configuration
choose the monitoring and checkpoint-selection policy.

`B` is batch size, `N` patches, `D` feature width and `K` ordered classes.
Networks, pooling and losses are plain PyTorch components. Their construction
does not require a Trainer. Image and feature MIL share the same pooling/readout;
`forward_with_attention` takes an unbatched feature bag on `MILHead`, and
`forward_replaced` keeps every other embedding fixed for conditional saliency.

`ImageEncoder` supports `resnet18`, `resnet50` and
`vit_small_patch16_224.dino`. The network receives an encoder object and sizes its
head from `num_features`. Frozen encoders remain in evaluation mode during
training. Input-size requirements live with the encoder.

The compact U-Net remains a two-level foreground model with two skip connections.
It has no interchangeable pretrained segmentation encoder. Thresholded foreground
is not an instance-ID map. Reorganizing code does not change that scientific scope.

## Reassemble pretrained HoVer-NeXt in Lightning

Run from the repository root with the `train` and `pretrained` extras and the
verified local checkpoint from [pretrained setup](pretrained-models.md). This
example reuses every learned component, without downloading or initializing new
branches. It compares two forward passes on one existing CytoNuke patch.

```python
from pathlib import Path
import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from src.models.pretrained.hovernext import load_hovernext

class ReassembledHoVer(L.LightningModule):
    def __init__(self, original):
        super().__init__()
        self.encoder = original.encoder
        self.decoders = original.decoders  # Keep the registered ModuleList
        self.heads = original.heads        # Keep the registered ModuleList

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError("Expected BCHW RGB with 256-pixel sides")
        features = self.encoder(images)
        return torch.cat([
            head(decoder(*features))
            for decoder, head in zip(self.decoders, self.heads, strict=True)
        ], dim=1)

    def predict_step(self, batch, batch_idx):
        return self(batch)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)  # Run once, before parallel work in this process
loaded = load_hovernext("data/models/hovernext/pannuke_convnextv2_tiny_1", device="cpu")
original = loaded["model"].eval()
assembled = ReassembledHoVer(original).eval()
assert assembled.state_dict().keys() == original.state_dict().keys()
path = Path("data/cytonuke/extracted/images/C3L-00977-22.svsx29y31.png")
with Image.open(path) as image:
    assert image.size == (256, 256)
    rgb = np.array(image.convert("RGB"), copy=True)
x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255
with torch.inference_mode():
    expected = original(x)
trainer = L.Trainer(
    accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
    limit_predict_batches=1, enable_progress_bar=False,
)
logits = trainer.predict(assembled, DataLoader(x, batch_size=1, num_workers=0))[0]
assert logits.shape == (1, 11, 256, 256) and torch.isfinite(logits).all()
torch.testing.assert_close(logits, expected, rtol=0, atol=0)
```

The wrapper registers the same encoder, decoder and head objects under the same
names: it shares their parameters with `original`, rather than creating a copy.
Do **not** call upstream `MultiHeadModel(loaded_encoder, loaded_decoders, loaded_heads)`
to reassemble them: its constructor calls `initialize()` and resets decoder/head
weights. The wrapper above avoids that constructor and retains learned weights.

These are raw 11-channel logits, not nucleus IDs. Use `predict_hovernext` with the
existing loaded handle (or its `model` replaced by `assembled`) for the pinned
instance postprocessing. This wrapper supports inference only: training would
require matched multihead targets, losses and optimization, and is not added by
calling `Trainer.fit`. Exact output agreement checks reassembly, not accuracy.

## Feature compatibility comes before reassembly

A CNN feature map has shape `B × C × h × w`; a transformer commonly exposes
`B × T × D` tokens. A pooled vector `B × D` has discarded the explicit spatial
layout required by an ordinary segmentation decoder. In timm, the normal
`num_classes=0` forward used here retains pooling; `forward_features` exposes
unpooled representations whose layout depends on the architecture.
[Official timm feature interfaces](https://huggingface.co/docs/timm/feature_extraction).

To connect a different encoder to a segmentation decoder, specify the ordered
feature levels, channel counts, strides and spatial alignment. Transformer patch
tokens require an explicit grid mapping and handling of class/register tokens;
a single token grid does not automatically provide U-Net's multiscale skips.
Channel projections or a new feature pyramid introduce parameters that must be
trained. These adapters are possible extensions, not current repository APIs.

The input contract also includes RGB/channel order, value range, normalization,
resize/crop rules and physical resolution. Current classification defaults use
224-pixel inputs with ImageNet normalization; segmentation defaults preserve
native image dimensions and use RGB values scaled to `[0, 1]`. Stain normalization
is a separate, optional operation, not a substitute for either transform.

WSI export targets 0.5 µm/pixel, with recorded effective calibration. Resizing a
256-pixel patch at 0.5 µm/pixel to 224 pixels changes its effective sampling to
about 0.571 µm/pixel. A PNG without calibration does not establish magnification.
Track this together with tissue coverage, crop coordinates and training exposure.

## How the pretrained models fit

**Original UNI** is a ViT-L/16 encoder, with no segmentation decoder or task head.
[`load_uni` / `embed_uni`](../src/models/pretrained/uni.py) use an authorized local checkpoint,
224-pixel RGB input and ImageNet normalization to return one 1024-value vector.
A new linear classifier or bag-pooling head could learn from these embeddings,
and the complete-feature workflow already trains `MILHead` on cached UNI embeddings.
The WSI comparison uses `src.inference.feature_extraction.load_comparison_encoder`
to bind these weights to its explicit full-square resize policy. The lower-level
`src.models.encoders.pretrained.load_pretrained_encoder` returns weights and
metadata without a data transform. UNI is not in the trainable `ImageEncoder` allowlist.
Obtaining embeddings does not train the downstream head. UNI2 is a different
model and interface. [Official original UNI usage](https://github.com/mahmoodlab/UNI/blob/main/README_old.md).

**HoVer-NeXt** already separates a ConvNeXt encoder, two skip-connected decoders
and their instance/type heads. The pinned PanNuke model emits 11 channels: five
instance-branch channels and six type channels. The local adapter uses instance
channels 2–4 and the six type channels, then applies the pinned thresholds,
watershed, cleanup and class voting to obtain nucleus IDs and types. Raw logits
are not instance labels. Replacing a branch changes the learned contract and may
invalidate that postprocessing. The adapter expects exactly 256 × 256 RGB,
divides by 255 and assumes the training scale of 0.25 µm/pixel; it does not verify
PNG calibration or resample 0.5 µm/pixel WSI patches to that scale.
[Local adapter](../src/models/pretrained/hovernext.py) · [Pinned upstream architecture](https://github.com/digitalpathologybern/hover_next_inference/blob/bde3c9d96172d16513e292d0a2dfcd7ff80202ff/src/multi_head_unet.py).

**Cellpose 3 nuclei** uses a residual U-Net with style conditioning. Its output
contains two flow components and a cell-probability score; flow dynamics and
filtering produce individual instances. An argmax over these channels is not a
segmentation mask. The demo uses the complete pretrained pipeline in its
[isolated environment](../environments/cellpose/README.md), with explicit channels,
percentile normalization, inversion and diameter in input pixels. Reusing its
encoder/decoder would require model-specific adapters, losses and compatible
weights; the current API exposes inference, not interchangeable training blocks.
[Official Cellpose 3 API](https://cellpose.readthedocs.io/en/v3.1.1.1/api.html) · [Nuclei channel/diameter contract](https://cellpose.readthedocs.io/en/v3.1.1.1/models.html).

## Lightning manages training, not architectural compatibility

`ClassificationModule`, `SegmentationModule` and `ContrastiveModule` share small
lifecycle/checkpoint hooks in `BaseModule`. Their `net` owns computation and their
injected `loss` owns the objective. Hydra configures these components and an
optimizer partial; `src.runners.train` constructs them through `models/loading.py`,
validates the data interface, and calls Lightning's trainer. Architecture code
does not import Lightning. Classification and segmentation own their prediction
outputs; the short SSL recipe explicitly declines prediction.
[Official LightningModule interface](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html).

Keep checkpoint parameter names, dimensions and class order aligned with the
constructed model. UNI and HoVer-NeXt use strict state-dictionary loading; their
local adapters also verify checkpoint identity. `strict=False` is not a general
repair for incompatible shapes or semantics. A deliberately replaced head needs
explicit initialization, training and a new checkpoint identity.

Training checkpoints additionally record model/data contracts: task, labels and
configured architecture, objective, image size and normalization. Current training
checkpoints require contract version 2. Detached feature-MIL inference uses
`src.models.loading.restore_model` with an explicit data contract; it does not
restore the optimizer or training loop. Resume/evaluate through
the [training entrypoints](training.md) so these checks apply. Loading weights
alone does not establish compatible preprocessing, MPP, annotation semantics or
performance on a new cohort.

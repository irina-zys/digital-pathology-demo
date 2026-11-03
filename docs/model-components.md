# Model components and Lightning

[README](../README.md) · [Training](training.md) · [Pretrained inference](pretrained-models.md)

Models can share components when their tensor interfaces and learned assumptions
match. A matching Python class name or tensor width alone is insufficient. This
guide describes the current APIs, then distinguishes possible extensions.

## What is already composable here?

| Component | Current implementation | Interface |
| --- | --- | --- |
| Image encoder | [`ImageEncoder`](../src/models/encoders.py) | RGB `B × 3 × H × W` → pooled features `B × D` |
| Patch classifier | [`Classifier`](../src/models/vision.py), `pooling=None` | Encoder → linear head → `B × K` logits |
| MIL classifier | Same class, `pooling="mean"` or `"attention"` | One image bag `1 × N × 3 × H × W` → `N × D` → pooled `1 × D` → `1 × K` logits |
| Binary segmentation | [`UNet`](../src/models/vision.py) | RGB `B × 3 × H × W` → foreground logits `B × 1 × H × W` |

`B` is batch size, `N` the number of patches, `D` the encoder feature width and
`K` the number of ordered classes. MIL currently handles one bag per batch;
attention weights normalize across its patches. It encodes images during the
forward pass, rather than accepting cached feature bags.

`ImageEncoder` allows `resnet18`, `resnet50` and `vit_small_patch16_224.dino`.
It exposes `num_features`; `Classifier` uses this to size its linear head.
The constructor accepts an encoder **configuration dictionary**, not an arbitrary
encoder object. Its new head, and optional attention network, require training.
`pretrained=False` initializes the encoder randomly. `freeze=True` disables its
gradients and keeps it in evaluation mode when the surrounding model trains.

The compact U-Net has two skip connections: shallow features at full resolution
and deeper features at half resolution. Its decoder upsamples, concatenates the
matching skip, then applies convolutions. Its final 1 × 1 convolution produces
one foreground channel. It is randomly initialized and currently has no argument
for substituting a pretrained encoder. Thresholded foreground is not an
instance-ID map: touching nuclei may remain joined.

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
from src.pretrained import load_hovernext

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
[`load_uni` / `embed_uni`](../src/pretrained.py) use an authorized local checkpoint,
224-pixel RGB input and ImageNet normalization to return one 1024-value vector.
A new linear classifier or bag-pooling head could learn from these embeddings,
but UNI is not in `ImageEncoder`'s allowlist, and cached-UNI MIL is not implemented.
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
[Local adapter](../src/hovernext.py) · [Pinned upstream architecture](https://github.com/digitalpathologybern/hover_next_inference/blob/bde3c9d96172d16513e292d0a2dfcd7ff80202ff/src/multi_head_unet.py).

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

`Classifier` and `UNet` inherit the local `VisionModule`, which supplies optimizer,
training/evaluation hooks and checkpoint contracts. Their PyTorch submodules and
`forward` methods define the architecture. [`src.train`](../src/train.py) uses
Hydra to instantiate the model and data module, then calls Lightning's trainer.
A compatible plain `nn.Module` could be wrapped in a new Lightning task, but that
would still require suitable targets, loss, metrics and data handling.
[Official LightningModule interface](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html).

Keep checkpoint parameter names, dimensions and class order aligned with the
constructed model. UNI and HoVer-NeXt use strict state-dictionary loading; their
local adapters also verify checkpoint identity. `strict=False` is not a general
repair for incompatible shapes or semantics. A deliberately replaced head needs
explicit initialization, training and a new checkpoint identity.

Training checkpoints additionally record model/data contracts: task, labels and
configured architecture, image size and normalization. Resume/evaluate through
the [training entrypoints](training.md) so these checks apply. Loading weights
alone does not establish compatible preprocessing, MPP, annotation semantics or
performance on a new cohort.

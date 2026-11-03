# Lightning vision baselines

Run from the repository root. This setup follows the configuration-group,
`LightningDataModule`, `LightningModule` and Hydra `instantiate` pattern in
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
It integrates with the existing repository without changing WSI preprocessing.
The implementation is local; it does not vendor or run the upstream template.

## Setup and models

```sh
uv sync --locked --extra train
```

Lightning was already declared in the `train` extra. Torch, Torchvision and timm
are already in the TIA Toolbox dependency tree and lock. No new dependency or
lockfile change is required. Use `--extra train` on subsequent `uv run` commands
so synchronization retains Lightning.

| `experiment=` | Task | Encoder / pooling |
| --- | --- | --- |
| omit | Patch classification | ResNet18 |
| `segmentation` | Binary nuclear foreground | Compact two-level U-Net |
| `mil_mean` | Bag classification | ResNet18 / mean |
| `mil_attention` | Bag classification | ResNet18 / attention |
| `dino` | Patch classification | Frozen pretrained DINO ViT-S/16 |
| `mil_dino_mean` | Bag classification | Frozen pretrained DINO ViT-S/16 / mean |
| `mil_dino_attention` | Bag classification | Frozen pretrained DINO ViT-S/16 / attention |

ResNet initializes randomly by default. Set `model.encoder.pretrained=true` to
request ImageNet weights or `model.encoder.name=resnet50` to change its size.
The DINO examples use timm's
[`vit_small_patch16_224.dino`](https://huggingface.co/timm/vit_small_patch16_224.dino):
DINO v1 trained on ImageNet-1k, not a pathology-specific encoder. Its first run
downloads weights. These examples use 224 × 224 RGB inputs and ImageNet
normalization; patch resizing is explicit and does not preserve physical scale.
`model.encoder.freeze=false` fine-tunes the encoder. Frozen encoders remain in
inference mode during training; only the classifier and attention head learn.
Setting `pretrained=false` exercises the architecture without DINO-learned
weights. Do not describe that as pretrained DINO performance.

## Input contract

Supply a CSV via `data.manifest` and an explicit `data.data_root`. CSV paths are
resolved against **data_root**, not the CSV's directory. Both config paths are
relative to the original launch directory unless absolute.

All tasks require `image_path,split,group_id`. `split` is exactly `train`, `val`
or `test`. Training requires nonempty train and val sets; evaluation requires a
test set. No random split is inferred. Use stable participant IDs as `group_id`
when available, keeping their slides, stains, patches and bags together.
Duplicate image paths and groups spanning splits are rejected before sample caps.
The loader cannot establish patient independence if the supplied IDs are wrong.

Classification and MIL additionally require `label`, an integer indexing the
ordered `class_names` list (for example, `'class_names=[control,disease]'`). Label
meaning and ordering are saved in checkpoints. MIL also requires `bag_id`; repeat
it on every patch row belonging to that slide/bag. A bag must have exactly one
label, group and split. Bag labels are never used as patch-level supervision.

```csv
image_path,split,group_id,label,bag_id
patches/slide_A/001.png,train,participant_A,0,slide_A
patches/slide_A/002.png,train,participant_A,0,slide_A
patches/slide_B/001.png,val,participant_B,1,slide_B
patches/slide_C/001.png,test,participant_C,0,slide_C
```

This is a schema illustration, not a supplied scientific dataset. Existing WSI
coordinate HDF5s and thumbnails are not training images. Export actual patches,
then provide their known labels and group assignments in a manifest.

Segmentation instead requires `mask_path`; optional `ignore_path` marks excluded
pixels with nonzero values. Integer instance masks are read without truncating
IDs and converted to foreground **in memory**; source masks stay unchanged.
Use NuInsSeg's recommended modified instance masks and its ambiguous-area masks.
Loss is binary cross-entropy averaged over valid pixels within each image, then
over images. Dice aggregates foreground intersection/size across valid pixels
in the evaluated set, threshold 0.5; empty prediction plus empty target scores 1.
This predicts nuclear foreground, not separated instances or whole cells.

Training uses paired horizontal/vertical flips; validation/test never augment.
Set `data.augment=false` to disable flips. Segmentation keeps native resolution by
default; batches larger than one require equal image sizes. Resizing a mask uses
nearest-neighbor interpolation and is an explicit experimental change.

## Train, evaluate and resume

```sh
# ResNet patch classifier; replace paths and labels with your own verified cohort.
uv run --locked --extra train python -m src.train \
  data.manifest=manifests/patch_splits.csv data.data_root=data/cohort \
  'class_names=[control,disease]'

# MIL; select mil_mean, mil_attention, mil_dino_mean or mil_dino_attention.
uv run --locked --extra train python -m src.train experiment=mil_attention \
  data.manifest=manifests/bag_splits.csv data.data_root=data/cohort \
  'class_names=[control,disease]' data.max_patches_per_bag=64

# Foreground segmentation.
uv run --locked --extra train python -m src.train experiment=segmentation \
  data.manifest=manifests/nuclei_splits.csv data.data_root=data/nuinsseg

# Evaluate a saved checkpoint using the same model, label and preprocessing settings.
uv run --locked --extra train python -m src.eval experiment=mil_attention \
  data.manifest=manifests/bag_splits.csv data.data_root=data/cohort \
  'class_names=[control,disease]' ckpt_path=/absolute/path/to/checkpoints/epoch_009.ckpt
```

Use a checkpoint you trust. Evaluation refuses an absent checkpoint and checks
the model/label/preprocessing contract before loading it. Encoder weights are
restored from the checkpoint, so evaluation/resume does not download pretrained
weights again. To resume optimizer and epoch state, use the same training command
with `ckpt_path=/absolute/path/to/checkpoints/last.ckpt` and a larger
`trainer.max_epochs`. Always select a fresh output directory for the new run.
Lightning restores optimizer/epoch state but resets checkpoint ranking when the
checkpoint directory changes; the resumed run selects its best from its new
validation epochs. Retain and compare the previous best separately. Exact random
sampling replay across a resume is not guaranteed.

MIL uses one variable-size bag per batch. `data.max_patches_per_bag` bounds the
encoder workload: training randomly samples without replacement when augment is
enabled; validation/test take the first patches sorted by path. `augment=false`
also makes training bag selection fixed. Record this cap when comparing results;
a capped test is not full-slide evaluation. Use `trainer.accumulate_grad_batches`
for an effective batch of multiple bags. Mean pooling averages embeddings;
attention uses a tanh MLP and a softmax over patches before the weighted average.
Attention weights are learned aggregation weights, not verified explanations.

The default is one CPU device, two Torch threads, zero loader workers, ten epochs
and a one-hour Trainer time bound. Data/epoch/batch bounds remain configurable.
One GPU or MPS device can be selected through `trainer.accelerator`; distributed
execution is deliberately not supported by this small runner. For sweeps, Hydra
`-m model.lr=0.0001,0.001` produces separate output directories.

## Run records

Each run saves `resolved_config.yaml`, the original `input_manifest.csv`, resolved
`selected_samples.csv`, CSV epoch metrics, best/last checkpoints and
`metrics.json`. The last includes completion status, versions, input-manifest
SHA-256, sample/bag counts, metrics and the evaluated/best checkpoint path.
Selected MIL rows enumerate the available patches before per-bag sampling.
`run_started.json` reserves the directory and rejects reuse. A failed run has no
complete `metrics.json`. Hydra settings are recorded per invocation.

The best checkpoint minimizes `val/loss`. Test evaluation is opt-in (`test=true`)
and uses the best checkpoint, not arbitrary final weights. During model selection,
leave the test set untouched. Accuracy is micro accuracy across images or bags;
neither patch metrics nor bag metrics automatically become patient-level metrics.

## WSI classification notebook

[WSI classification demo](../notebooks/wsi_classification_demo.ipynb) uses the ten
local TCGA-LUAD/LUSC slides, with six training patients, two validation patients
and two held-out patients. Each split contains both classes. This is a fixed,
seeded teaching split, not an official TCGA benchmark or a performance estimate.

The `data=wsi` profile selects `src.data.wsi.WSIBagDataModule`. Its CSV requires
`wsi_path,coords_path,split,group_id,label,bag_id`, with paths relative to
`data.data_root`. It validates patient separation, slide identity, HDF5 geometry,
calibration and file-change guards. It reads eight evenly indexed tissue
coordinates per slide directly through OpenSlide; no patch PNGs or WSI copies
are written. Each native footprint becomes a 256-pixel tile at target 0.5 µm/pixel,
then the explicit DINO transform resizes it to 224 pixels. The exact coordinates,
recorded tissue fractions, effective MPP and source metadata are saved.

The default local input is `outputs/tcga-mpp05/manifest.json`, a historical
`clam_saturation`/CLAM-style export. The notebook records its actual schema and
masking/tiling methods. It does not claim those slides were rerun through the
newer default pipeline. With downloaded TCGA slides, a fresh preprocessing run
can instead be created using:

```sh
uv run --locked python -m src.preprocess data=tcga \
  export.pixel_slide_id=null hydra.run.dir=outputs/tcga-classification-preprocessing
```

Set the notebook's preprocessing-run path to that directory. Require complete
eligible 256-pixel coordinate exports at the configured MPP; capped previews,
missing files and mismatched calibration are rejected.

Use the main environment with the `train` extra. The notebook verifies a local
DINO checkpoint and runs Hugging Face offline. Its canonical cache is
`data/models/huggingface/`. On a new checkout, acquire the public weights explicitly
before running the notebook:

```sh
HF_HOME="$PWD/data/models/huggingface" uv run --locked --extra train python -c \
  'from huggingface_hub import hf_hub_download; hf_hub_download("timm/vit_small_patch16_224.dino", "model.safetensors", revision="main")'
```

The notebook checks revision `10e440b8a34dfd657a90f2fcaa988c6b6a2f0da4` and
SHA-256 `8d4795b8e4e327c6d841e15a2e3e6d88c45346256a43f8cdae8c26d92c806f7a`.
If the provider changes the branch, review the checkpoint rather than bypassing
those checks. DINO here is the ImageNet encoder, not a pathology foundation model.

The notebook composes the existing Hydra training profile, calls `src.train.run`,
trains only the attention/classification head for one epoch on CPU, and evaluates
the best validation checkpoint on the two held-out slides. It saves thumbnail
sampling overlays, slide probabilities, a confusion table, metrics, one checkpoint
and provenance in a fresh `outputs/wsi_classification_demo/<run>/` folder. It does
not use the test set to select the checkpoint or tune settings. Eight selected
patches do not represent all tissue; attention weights are not validated tissue
localization. Declared whole-WSI checksums are retained but not recomputed by this
bounded notebook.

## Bounded check on locally prepared NuInsSeg

This helper reads the existing local `data/nuinsseg/index.csv` and selects 16 real
patches in eight source-tissue groups. It retains the provided foreground and
ambiguous masks, and uses the existing human/mouse species labels for classifier
mechanics. It neither invents patient IDs nor reproduces an official benchmark
split. It must not be used to report generalization or disease performance.

```sh
uv run --locked --extra train python scripts/prepare_vision_smoke.py \
  --output outputs/vision_smoke/input.csv

uv run --locked --extra train python -m src.train experiment=mil_mean \
  data.manifest=outputs/vision_smoke/input.csv data.data_root=data/nuinsseg \
  'class_names=[human,mouse]' data.max_patches_per_bag=2 \
  trainer.max_epochs=1 trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1 trainer.limit_test_batches=1 \
  trainer.num_sanity_val_steps=0 test=true \
  hydra.run.dir=outputs/vision_smoke/mil_mean
```

Select `mil_attention`, `mil_dino_mean` or `mil_dino_attention` for the other MIL
variants. Omit `experiment` for ResNet patches, or use `experiment=dino` for DINO
patches. Use `experiment=segmentation`, omit `class_names`, and set
`data.batch_size=1` for the native 512 × 512 masks. These are bounded execution
checks; inspect losses, logs and checkpoints without interpreting their scores as
biological validation. No artificial images or labels substitute for these runs.

## Implementation verification (2026-11-03)

All seven table entries completed real CPU training (two optimizer steps), one
validation batch and best-checkpoint test evaluation on the bounded NuInsSeg
selection. DINO profiles loaded the published pretrained safetensors weights.
Standalone DINO-attention and U-Net evaluations completed, and ResNet mean MIL
resumed from step 2 to step 4 with optimizer state. Checkpoint inspection found
finite metrics and optimizer updates only for the intended trainable parameters.

The real-entrypoint checks also rejected groups crossing splits, duplicate image
paths, conflicting bag labels, changed class ordering, changed normalization,
missing evaluation checkpoints and reused output directories. Ruff passed for
the added Python code. Local logs/checkpoints are under
`outputs/lightning_validation_20261103/`; outputs are ignored by Git. No synthetic
image dataset was used. GPU/MPS execution, full-cohort training, scientific split
validity and biological accuracy remain unverified.

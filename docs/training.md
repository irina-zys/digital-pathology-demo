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
| `mil_dino_learning_curve` | Bag classification | Same encoder/head; 20 measured epochs, fixed inputs |

ResNet initializes randomly by default. Set `model.net.encoder.pretrained=true` to
request ImageNet weights or `model.net.encoder.name=resnet50` to change its size.
The DINO examples use timm's
[`vit_small_patch16_224.dino`](https://huggingface.co/timm/vit_small_patch16_224.dino):
DINO v1 trained on ImageNet-1k, not a pathology-specific encoder. Its first run
downloads weights. These examples use 224 × 224 RGB inputs and ImageNet
normalization; patch resizing is explicit and does not preserve physical scale.
`model.net.encoder.freeze=false` fine-tunes the encoder. Frozen encoders remain in
evaluation mode during training; only the classifier and attention head learn.
Setting `model.net.encoder.pretrained=false` exercises the architecture without DINO-learned
weights. Do not describe that as pretrained DINO performance.

## Input contract

Supply a CSV via `data.manifest` and an explicit `data.data_root`. CSV paths are
resolved against **data_root**, not the CSV's directory. Both config paths are
relative to the original launch directory unless absolute.

### Patch manifests

The patch-image tasks require `image_path,split,group_id`. `split` is exactly `train`, `val`
or `test`. Training requires nonempty train and val sets; evaluation requires a
test set. No random split is inferred. Use stable participant IDs as `group_id`
when available, keeping their slides, stains, patches and bags together.
Leading/trailing whitespace is removed from split, participant and bag identifiers.
Duplicate image paths and groups spanning splits are rejected before sample caps.
The loader cannot establish patient independence if the supplied IDs are wrong.
When evaluating a checkpoint with a different manifest, audit overlap with its
original training patients separately; the manifest checks cover the current run.

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
coordinate HDF5s and thumbnails are not patch-image files. For this interface,
export actual patches and provide their known labels and group assignments. The
WSI coordinate interface below reads patches directly from the original slides.

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
nearest-neighbor interpolation and is an explicit experimental change. Both source
images and resized model inputs must remain within `data.max_pixels`.

### WSI and cached-feature manifests

WSI recipes use `wsi_path,coords_path,split,group_id,label,bag_id`. Paths resolve
against `data.data_root`; `bag_id` must match the WSI stem. Coordinate files must
contain complete, unpadded level-0 exports at the requested MPP. The reader verifies
slide dimensions, calibration, coordinate identity and source size, then selects
the configured fixed coordinate subset. The SSL recipe prepares only train/val
pixels. Labels validate the source schema but never enter its objective.

Cached-feature recipes use `cache_path,cache_sha256,eligible_patches,coordinate_sha256`
plus `split,group_id,label,bag_id`. Configure `data.input_dim` and
`data.encoder_sha256` from the cache metadata. Checks include complete bag coverage,
encoder/preprocessing identity and L2-per-patch normalization. Feature bags are
never truncated to make a short validation run faster. See
[complete feature coverage](#encoder-assets-and-complete-feature-coverage).

## Train, evaluate and resume

```sh
# ResNet patch classifier; replace paths and labels with your own verified cohort.
uv run --locked --extra train python -m src.runners.train \
  data.manifest=manifests/patch_splits.csv data.data_root=data/cohort \
  'class_names=[control,disease]'

# MIL; select mil_mean, mil_attention, mil_dino_mean or mil_dino_attention.
uv run --locked --extra train python -m src.runners.train experiment=mil_attention \
  data.manifest=manifests/bag_splits.csv data.data_root=data/cohort \
  'class_names=[control,disease]' data.max_patches_per_bag=64

# Foreground segmentation.
uv run --locked --extra train python -m src.runners.train experiment=segmentation \
  data.manifest=manifests/nuclei_splits.csv data.data_root=data/nuinsseg

# Evaluate a saved checkpoint using the same model, label and preprocessing settings.
uv run --locked --extra train python -m src.runners.train --config-name eval experiment=mil_attention \
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
validation epochs. Retain and compare the previous best separately. A resumed
run must perform at least one new optimizer update; reusing an exhausted epoch/step
budget fails rather than reporting another completed training run.
Exact random sampling replay across a resume is not guaranteed.

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
`-m model.optimizer.lr=0.0001,0.001` produces separate output directories.

## Run records

Each run saves `resolved_config.yaml`, the original `input_manifest.csv`, resolved
`selected_samples.csv`, CSV epoch metrics, best/last checkpoints and
`metrics.json`. The last includes completion status, versions, input-manifest
SHA-256, sample/bag counts, metrics, optimizer updates performed in this
invocation, and the evaluated/best checkpoint path.
Selected patch-MIL rows enumerate the available patches before per-bag sampling.
WSI runs also save `sampled_coordinates.json` with the exact selected coordinates,
coordinate-file hashes and calibration recorded by the reader.
`run_started.json` reserves the directory and rejects reuse. A failed run has no
complete `metrics.json`. Hydra settings are recorded per invocation.

The best checkpoint minimizes `val/loss`. Test evaluation is opt-in (`test=true`)
and uses the best checkpoint, not arbitrary final weights. During model selection,
leave the test set untouched. Accuracy is micro accuracy across images or bags;
neither patch metrics nor bag metrics automatically become patient-level metrics.

## Pretrained classification and learning recipes

The [WSI classification notebook](../notebooks/wsi_classification_demo.ipynb)
compares frozen original UNI, ImageNet DINO ViT-S/16 and ImageNet ResNet-50 with
trained attention pooling and slide classifiers. It uses **every eligible
coordinate** in the chosen export, sharing the fixed patient split and label
order across encoders. Frozen UNI plus a trained classifier is transfer learning,
not zero-shot diagnosis classification.

The separate [brief learning notebook](../notebooks/representation_learning_demo.ipynb)
shows a SimCLR-style pretraining objective, actual encoder fine-tuning, and
random initialization. Its two-patch slide mini-bags and short epoch budgets are
explicit teaching bounds, distinct from the full-coordinate classification run.
No test-slide pixels enter that brief notebook. Individual patches do not acquire
invented diagnosis annotations: supervised updates use pooled slide mini-bags.

Both use the repository's `src.runners.train.run` / `python -m src.runners.train` entrypoint,
Lightning modules/data modules, callbacks, CSV logging and Hydra experiment
composition. This follows the
[Lightning-Hydra-Template organization](https://github.com/ashleve/lightning-hydra-template)
and its [experiment override pattern](https://github.com/ashleve/lightning-hydra-template/blob/main/configs/experiment/example.yaml).
No second optimizer loop is implemented in notebooks.

| Experiment | Model/data components | Learning behavior |
| --- | --- | --- |
| `mil_features` | `ClassificationModule` / `FeatureBagDataModule` | Frozen cached features; all eligible patches; attention and slide head learn |
| `simclr_wsi` | `ContrastiveModule` / `WSIContrastiveDataModule` | Random ResNet50, paired augmented views, NT-Xent; no diagnosis loss |
| `finetune_dino` | `ClassificationModule` / `WSIBagDataModule` | Verified local ImageNet DINO; encoder and mean-pooled slide head learn |
| `finetune_resnet50` | Same interfaces | Verified ImageNet ResNet50; encoder and slide head learn |
| `scratch_vit`, `scratch_resnet50` | Same interfaces | Matched supervised recipes with random encoder initialization |
| `finetune_local_ssl` | Same interfaces | Strict transfer of encoder weights from a checksum-bound local SimCLR checkpoint |

The relevant owners are `src/models`, `src/lightning`, `src/data`,
`src/evaluation/metrics`, `configs/model`, `configs/data` and `configs/experiment`. `src/training/recipes.py` only composes configuration and
calls the shared entrypoint. `EncoderUpdateAudit` records an actual encoder
parameter before/after fitting, so head-only training is not mislabeled fine-tuning.
Normalization running statistics stay fixed during the short supervised mini-bag
recipes; encoder parameters still receive gradients. The SSL projection head is
not transferred to the supervised classifier.

For example, using a patient-separated WSI manifest prepared by the notebook:

```sh
uv run --locked --extra train python -m src.runners.train experiment=simclr_wsi \
  data.manifest=/path/to/wsi_splits.csv data.data_root=. \
  'class_names=[LUAD,LUSC]'

uv run --locked --extra train python -m src.runners.train experiment=finetune_dino \
  data.manifest=/path/to/wsi_splits.csv data.data_root=. \
  'class_names=[LUAD,LUSC]'
```

Feature MIL takes the generated **feature manifest**, with cache checksums,
eligible counts, encoder checksum and dimensionality, rather than raw WSI rows.
`configs/wsi_comparison.yaml` configures the matched encoder comparison; each
model fit composes `experiment=mil_features`. Data, model and label contracts are
saved and checked during checkpoint reload. Interrupted budgets are not reported
as completed comparisons.

### Encoder assets and complete feature coverage

```sh
# First obtain provider approval for UNI and sign in locally; no token in source.
.venv/bin/hf auth login
.venv/bin/python scripts/prepare_classification_models.py
```

Weights are pinned and verified. UNI uses
`timm.create_model("hf-hub:MahmoodLab/UNI@<pinned revision>", pretrained=True)`.
The explicit model cache does not move authentication tokens into the project.
Original UNI, DINO and ResNet use the same full-square 256-to-224 resize and
ImageNet normalization for this comparison; this is a declared shared transform,
not each model's default center-crop recipe. Features are L2-normalized per patch,
without fitting statistics on held-out slides.

`src/inference/feature_extraction.py` binds frozen weights to the matched comparison
transform through `load_comparison_encoder`, then streams bounded batches to resumable HDF5 caches in
`outputs/wsi_features_v1`. Every feature row retains its coordinate and index.
Cache reuse requires matching source metadata, coordinate hash, weights and
preprocessing contract; partial files do not masquerade as completed bags.
Existing coordinate exports retain their recorded preprocessing/calibration.
The notebook's coverage table reports actual counts rather than a patch cap.

### Curves and interpretation

Twenty measured epochs are the default for frozen-feature MIL. The same seed,
patient split, optimizer settings and epoch budget are used across encoders;
feature dimensions and head parameter counts differ and are reported. Validation
loss selects a checkpoint; the two held-out patients do not select hyperparameters.
Training loss is measured during updates and validation after each epoch.

Whole-slide maps show **MIL pooling attention** on all eligible coordinates.
Colours are within-slide percentiles, not tumour probabilities or transformer
self-attention. H/M/L locate the highest/median/lowest attention patches.
Selected-patch maps differentiate the selected slide-class logit through one
patch with all other embeddings retained. Raw gradients/weights and display
scales are saved. Per-patch display normalization can amplify weak signals;
these maps demonstrate sensitivity, not causal or histological validation.

### Self-supervised references

- Chen et al., [SimCLR](https://arxiv.org/abs/2002.05709), ICML 2020.
- Caron et al., [DINO](https://arxiv.org/abs/2104.14294), ICCV 2021.
- Grill et al., [BYOL](https://arxiv.org/abs/2006.07733), NeurIPS 2020.

The short SSL recipe demonstrates a SimCLR-style objective with a tiny batch and
AdamW, not a reproduction of the published training recipe. DINO weights are
used for transfer; DINO and BYOL pretraining are references, not implementations
claimed by this notebook. Adequate data, augmentations, budgets and evaluation
must be established separately before interpreting representation quality.

## Bounded check on locally prepared NuInsSeg

This helper reads the existing local `data/nuinsseg/index.csv` and selects 16 real
patches in eight source-tissue groups. It retains the provided foreground and
ambiguous masks, and uses the existing human/mouse species labels for classifier
mechanics. It neither invents patient IDs nor reproduces an official benchmark
split. It must not be used to report generalization or disease performance.

```sh
uv run --locked --extra train python scripts/prepare_vision_smoke.py \
  --output outputs/vision_smoke/input.csv

uv run --locked --extra train python -m src.runners.train experiment=mil_mean \
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

The modular implementation was exercised with real NuInsSeg patches, TCGA WSI
coordinate exports and complete cached feature bags. Runs used one CPU device,
two threads, zero loader workers, two training batches and one validation/test
batch, with a five-minute per-run bound. Native segmentation geometry and complete
cached bags were retained. All 15 composed configurations resolved their targets.

Classification, segmentation, mean/attention MIL, WSI fine-tuning, SSL, transfer
from a newly produced SSL checkpoint, evaluation and optimizer-state resume passed.
Additional complete-bag recipes exercised detached checkpoint restoration. One
725-patch bag exercised DINO extraction, interrupted-cache recovery, cache reuse,
full-bag interpretation and conditional saliency. Real-input derivatives verified
leakage, checksum, contract/version mismatch and output-reuse rejection.

The original pre-refactor baseline records under `outputs/lightning_validation_20261103/`
are historical; their checkpoints do not satisfy version 2. Current refactor and
follow-up audit records are described in [the audit record](repository-audit.md).
They establish mechanics on the exercised inputs, not full-cohort convergence,
biological accuracy, a model ranking or exact stochastic replay. Full encoder-comparison/GPU
training and network acquisition branches were not part of this bounded audit.

## Modular components and the API break

Lightning task wrappers live in `src/lightning`; their `net` is a plain
PyTorch network from `src/models/networks`. Reusable encoders and pooling live in
sibling packages. **Losses are a standalone `src/models/losses` package**:
`ValidPixelBCE` preserves the per-image valid-pixel reduction, and `NTXent`
preserves the paired-view objective. Classification uses PyTorch's existing
`CrossEntropyLoss`. All objectives are injected through `model.loss`.

Hydra composes encoder, pooling, loss and optimizer groups, then recursively
instantiates their targets. Pooling and optimizer configurations use `_partial_`
so the network supplies the feature dimension and Lightning supplies parameters.
For example:

```sh
uv run --locked --extra train python -m src.runners.train experiment=mil_attention \
  data.manifest=manifests/bag_splits.csv data.data_root=data/cohort \
  'class_names=[control,disease]' model.net.pooling.attention_dim=64 \
  model.net.encoder.freeze=true model.optimizer.lr=0.0001
```

To change the pooling group, use
`model/pooling@model.net.pooling=mean`. Encoder groups use
`model/encoder@model.net.encoder=resnet50`; initialization and freezing remain
explicit encoder settings. SSL temperature is `model.loss.temperature` and
projection width is `model.net.projection_dim`.

This refactor deliberately removes old import paths and configuration keys;
there are no compatibility aliases. Training checkpoints use
`pathology_checkpoint_version: 2`. Historical training checkpoints are rejected
with an explicit retraining message. External pretrained weights and existing
feature caches retain their formats and identities. Evaluation/resume builds
without initialization weights, then restores the requested current-format
checkpoint. Local SSL transfer reads only its `net.encoder.net.*` weights.

Datamodules own preparation and expose `provenance()` with selected rows, split
counts and optional coordinate selections. The runner consumes this interface
without reaching into datasets. Datasets and manifest/coordinate utilities do
not import Lightning or Hydra. SSL validates the source manifest and prepares
only train/validation coordinate selections.

Detached feature-MIL inference restores through `src.models.loading.restore_model`,
which checks checkpoint version, model and data contracts before strict weight
loading. Trainer-managed resume also restores optimizer and loop state.

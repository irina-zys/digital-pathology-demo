# Modular source and caller audit — 2026-11-03

[Project structure](hydra-and-project-structure.md) · [Training](training.md) · [Preprocessing](preprocessing.md)

This audit covers project-owned source, operational scripts, Hydra configuration,
notebook source cells and Markdown references affected by the modular refactor.
It preserves historical artifacts, notebook outputs, dependency locks and vendored
upstream source. Existing notebook outputs were retained; validation executed
current source in separate output directories.

## Ownership after the audit

| Owner | Responsibility |
| --- | --- |
| `src/data/` | Input contracts, transforms, datasets, datamodules and feature-cache persistence |
| `src/models/` | Encoders, pooling, networks, standalone losses, pretrained adapters and strict restoration |
| `src/lightning/` | Task steps, metric updates/logging, optimization and checkpoint hooks |
| `src/training/` | Trainer execution, callbacks and composed learning recipes |
| `src/runners/` | Configuration, input/output coordination, provenance and completion records |
| `src/evaluation/metrics/` | Shared foreground Dice, instance matching, detection rates and aggregation |
| `src/inference/` | Matched encoder transforms, feature execution and cache orchestration |
| `src/imaging_utils/` | Reusable image, mask, quality, normalization and instance-cleanup operations |
| `src/wsi_preprocessing/` | Slide inspection, calibrated tiling/export and the executable preprocessing workflow |

Networks and datasets do not depend on Lightning. Models do not import the data
layer. Datamodules call plain manifest/coordinate functions and expose
`provenance()`; runners do not traverse dataset storage. Training and evaluation
share metric definitions, while monitoring policy stays in configuration/callbacks.

## Migration decisions

- Run training with `python -m src.runners.train`; add `--config-name eval` for
  evaluation. Run preprocessing with `python -m src.wsi_preprocessing.preprocess`.
  The old root forwarding modules have been removed.
- Configure computation through `model.net`, encoder options through
  `model.net.encoder`, objectives through `model.loss`, and learning rate through
  `model.optimizer.lr`. Pooling and optimizer factories are Hydra partials.
- Import task wrappers from `src.lightning` and metrics from
  `src.evaluation.metrics`. There are no aliases for the previous locations.
- Training checkpoint version 2 binds model/data contracts. Historical training
  checkpoints require retraining. External pretrained weights and feature-cache
  formats retain their identities. Detached restoration verifies the requested
  data contract; Trainer-managed resume restores optimizer and loop state.
- Use `load_comparison_encoder` in `src.inference.feature_extraction` when a
  matched comparison transform is required. The model weight loader returns
  weights and metadata only. Single-patch UNI retains its distinct policy.
- The unused independent CLAM tiler and `tissue_mask` compatibility helper were
  removed. The active tiler calls pinned upstream CLAM. The preprocessing notebook
  now records source hashes through the shared utility instead of deleted paths.
- Preprocessing presets remain starting points. Select and review dataset-specific
  thresholds, physical scale and tissue-selection policies before cohort use.

## Reproduce the training checks

Use `scripts/validate_refactor.py` with explicit real input manifests and a fresh
output directory. The patch manifest must include images, instance/ignore masks,
classification labels and bag IDs; `scripts/prepare_vision_smoke.py` prepares the
bounded NuInsSeg mechanics example. WSI and feature manifests follow the contracts
in the training guide; feature bags remain complete.

```sh
uv run --locked --extra train python scripts/validate_refactor.py \
  --patch-manifest /path/to/nuinsseg_mechanics.csv --patch-root data/nuinsseg \
  --wsi-manifest /path/to/wsi_splits.csv \
  --feature-manifest /path/to/complete_feature_bags.csv \
  --data-root . --output outputs/refactor-check
```

Replace the example paths with verified local inputs. The default label orders
are `human mouse` for the patch mechanics example and `LUAD LUSC` for WSI bags;
use `--patch-classes` and `--wsi-classes` to declare a different order. Pinned local
ResNet50 weights are required for the fine-tuning check; network access is disabled.
The script infers feature width and encoder identity from the supplied cache,
records commands/logs in `validation.json`, composes every experiment and exercises
both pooling variants, new SSL transfer, restoration and rejection protections.
It creates invalid-contract derivatives of real inputs/checkpoints in the isolated
output directory and never edits their originals.

## Earlier working-tree validation

The earlier working-tree audit recorded the checks below. Those runs preceded
separation of unrelated local comparison/interpretation edits for committing;
they do not establish execution coverage of every final commit-candidate file.

Training checks used real NuInsSeg/TCGA inputs, one CPU device, two threads,
zero loader workers, two training batches and one validation/test batch, with a
five-minute per-run bound. Native segmentation geometry and complete cached bags
were preserved. Preprocessing/adapter runs used explicit pixel and runtime bounds.

| Check | Evidence and result |
| --- | --- |
| Source/configuration | 139 source/notebook code units checked; 87 source modules imported; all 15 experiment configurations resolved their targets |
| Training/evaluation | Classification, segmentation, image mean/attention MIL, WSI fine-tuning, SSL, newly produced SSL transfer, feature MIL, evaluation and resume passed |
| Restore boundaries | Fresh native-segmentation checkpoint evaluation and two additional resume updates passed; detached feature heads rejected changed model/data contracts |
| Full feature bags | Mean/attention recipes used complete bags; DINO extraction/recovery/reuse and interpretation covered all 725 eligible patches of one real slide |
| Rejection behavior | Real-input derivatives exercised leakage, checksums, model/data contracts, historical versions and output-reuse protections |
| Pretrained/annotations | HoVer-NeXt and isolated Cellpose processed CytoNuke/NuInsSeg patches; ResNet50, DINO and both UNI interfaces ran real RGB inference; annotation conversion/overwrite checks passed |
| Preprocessing CLI | One real TCGA slide completed; independent verification and preview refresh passed; exported coordinates and eligible count matched the pre-cleanup run |
| Preprocessing notebook | Core cells completed with a 512-pixel thumbnail, all 1,005 eligible coordinates and four real QC patches; corrected provenance included the current source tree |
| Documentation/viewer | Python examples parsed and local imports/links checked; the documented HoVer-NeXt reassembly example ran on a real patch; Streamlit rendered the fresh saved run without exceptions |
| Repository checks | Ruff and whitespace checks passed; no dependency/lockfile edits; notebook outputs/metadata preserved |

Local evidence is collected under `outputs/repository_audit_20261103/` and the
initial `outputs/modular_refactor_validation_20261103/`. These ignored directories
contain commands, logs and small reports. Detailed temporary checkpoints, caches
and figures remain at the paths recorded in those reports; they are not repository
fixtures. Historical `outputs/lightning_validation_20261103/` records predate the
checkpoint break and are not current-format restoration evidence.

The optional stain-normalization notebook cell was skipped in the final core
workflow run; normalization remains off in the default pipeline. Full-cohort/GPU
training, network download branches and QuPath GUI
import were not exercised by this audit. Bounded runs establish mechanics, not
biological accuracy, generalization or a ranking of the pretrained models.

## Isolated commit-candidate validation

After separating unrelated comparison/interpretation edits, the exact candidate
source and configuration were exercised again in an isolated project snapshot.
`scripts/validate_refactor.py` completed all 17 execution/rejection cases and
resolved all 15 experiment configurations. This included classification, native
segmentation, image and cached-feature mean/attention MIL, WSI training, SSL,
transfer from the newly created SSL checkpoint, new-checkpoint evaluation/resume,
and leakage, checksum, contract, historical-version and overwrite rejection.
The CPU, thread, loader, batch and runtime bounds above were retained.

The retained comparison training caller also completed two optimizer updates
using four complete real DINO feature bags. The migrated encoder loader and
interpretation caller reused a complete 725-patch bag and produced attention and
saliency for three real patches. Ruff passed on the candidate source, scripts
and application. Notebook source references were migrated while preserving the
committed outputs, metadata and cell counts.

Local temporary evidence is in
`/private/tmp/pathology-commit-validation-runs/validation.json` and
`/private/tmp/pathology-commit-caller-validation/report.json`; these are execution
receipts, not committed fixtures or permanent storage. The broader adapter,
annotation and preprocessing checks above were not repeated during commit
separation. These bounded checks establish mechanics, not biological validity.

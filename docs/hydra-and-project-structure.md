# Hydra and the project structure

[README](../README.md) · [Preprocessing reference](preprocessing.md)

Hydra assembles the settings for a run; Python implements the work. Keeping
configuration in `configs/`, reusable functions in `src/`, and operational
commands in `scripts/` lets the CLI and notebook share the same processing code.
This directory separation is a project convention, not a Hydra requirement.

## What happens when you run preprocessing

From the repository root:

```sh
uv run --locked python -m src.preprocess data=tcga data.max_slides=1
```

The command has four parts:

| Part | Role |
| --- | --- |
| `uv run --locked` | Use the project environment and its existing dependency lock |
| `python -m src.preprocess` | Run the Python module that owns the preprocessing entrypoint |
| `data=tcga` | Select `configs/data/tcga.yaml` from the `data` configuration group |
| `data.max_slides=1` | Override one value in the composed configuration |

In [src/preprocess.py](../src/preprocess.py), the entrypoint is:

```python
@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess")
def main(cfg: DictConfig):
    run(cfg, Path(HydraConfig.get().runtime.output_dir))
```

`config_path` is relative to the Python file containing the decorator, and
`config_name` omits `.yaml`. Hydra loads that configuration, composes its selected
groups, applies command-line overrides and passes an OmegaConf `DictConfig` to
`main`. Code can read values such as `cfg.data.max_slides`.
[Hydra's configuration tutorial](https://hydra.cc/docs/1.3/tutorials/basic/your_first_app/config_file/)
explains this entrypoint pattern.

The actual call sequence here is:

```text
configs + CLI overrides
    -> Hydra main(cfg)
    -> preprocess.run(cfg, output_directory)
    -> read thumbnail -> preprocess_masks -> optional coordinate/pixel export
    -> metadata, previews and run manifest
```

Hydra does not choose a segmentation algorithm by inspecting a module name.
`run` explicitly passes the selected settings to our functions. The numerical
masking helpers receive ordinary mappings, converted with
`OmegaConf.to_container(..., resolve=True)`. Preprocessing does not use Hydra
`_target_`/`instantiate` to construct its stages. The separate
[training entrypoint](training.md) does use `instantiate` for its model, data,
callbacks and trainer; running the preprocessing entrypoint does not train a model.

## How the YAML files combine

[configs/preprocess.yaml](../configs/preprocess.yaml) contains:

```yaml
defaults:
  - _self_
  - data: demo
  - masking: codait_tia_clam
  - artifacts: default
  - export: clam_hipt
  - optional local: default
```

Each group selects a YAML file. For example, `masking: codait_tia_clam` loads
`configs/masking/codait_tia_clam.yaml` under the `masking` key. `thumbnail`
settings are written directly in `preprocess.yaml`. The `defaults` list is the
composition recipe; it is not a field passed to the processing functions.

Order matters: later values win when configurations set the same field;
dictionaries otherwise combine. `_self_` places the containing file's own
values in that order. Here it comes first, before the selected groups. The
optional local file can be absent. Command-line value overrides are applied
after composition. See [Hydra's defaults-list rules](https://hydra.cc/docs/1.3/advanced/defaults_list/).

For the TCGA command above, part of the resulting configuration is:

```yaml
data:
  paths: []
  manifest: manifests/tcga_demo.csv
  max_slides: 1
masking:
  method: codait_tia_clam
  # Other masking values come from the selected group.
```

Changing `data=tcga` replaces the group's selected file. Changing
`data.max_slides=1` adjusts a field within that group. Keep `data.manifest` and
`data.paths` mutually exclusive: when supplying explicit paths with the TCGA
profile, also set `data.manifest=null`.

The [lung profile](../configs/lung.yaml) composes `preprocess`, selects the TCGA
roster and overrides masking, component and tiling parameters. Use
`--config-name lung`; `data=tcga` alone selects only the roster. The preprocessing
notebook uses Hydra's `compose(config_name="lung")` so those overrides agree
with the CLI, rather than reading an inheriting YAML file as a flat mapping.

A named experiment can replace several groups together.
[configs/kpmp.yaml](../configs/kpmp.yaml) includes `preprocess`, then uses
`override data`, `override masking` and `override artifacts` to select the CKD
cohort and special-stain recipe:

```sh
uv run --locked python -m src.preprocess --config-name kpmp --cfg job --resolve
```

This is why choosing the KPMP profile changes more than the list of slides.
Its scientific assumptions belong in those explicit group selections.

## Inspect settings before reading slides

These commands execute the real entrypoint's configuration handling and exit
without running preprocessing:

```sh
# Default composed task configuration, with interpolations resolved.
uv run --locked python -m src.preprocess --cfg job --resolve

# One TCGA slide, smaller thumbnail, no coordinate/pixel export.
uv run --locked python -m src.preprocess data=tcga data.max_slides=1 \
  thumbnail.max_size=512 export.enabled=false --cfg job --resolve
```

Remove `--cfg job --resolve` to execute that configuration. `--cfg job` shows
application settings; `--cfg hydra` shows Hydra's own settings. `--resolve`
evaluates interpolations such as `${now:...}`. Neither command validates slide
readability or the biological suitability of a threshold.

Use existing key names exactly. Hydra normally rejects an unknown key, which
helps catch typos. Its `+key=value` syntax adds a new field, but adding a field
does not make the Python code consume it. A new processing option needs both
an implementation and a documented config entry. Quote list/dictionary overrides
in the shell, for example `'export.max_patches=[2]'`; that cap affects pixel
exports only when a slide has been selected for them.

For a small parameter sweep:

```sh
uv run --locked python -m src.preprocess -m data=tcga data.max_slides=1 \
  export.enabled=false thumbnail.max_size=128,256
```

Unlike `--cfg`, `-m` **runs the workflow**: this example processes one real slide
twice, in separate numbered job directories. It is a comparison of two overview
settings, not two independent biological samples. The default launcher runs
these jobs locally; this command does not submit an HPC job.

## Where settings and outputs are recorded

This repository sets `hydra.job.chdir: false`, so the working directory stays at
the launch location. Run commands from the repository root. The reader uses
`to_absolute_path` for input manifests and slide paths, anchored to that original
launch directory even if `chdir` is later enabled. Manifest row paths are also
relative to the launch directory, not the manifest's parent folder.

`HydraConfig.get().runtime.output_dir` supplies the output location. The project
uses a timestamped directory by default; `hydra.run.dir=outputs/my-run` chooses
one explicitly for a single run. Always choose a fresh directory. The project's
output guard refuses to mix new results with an existing run.

| Record | Written by | Purpose |
| --- | --- | --- |
| `.hydra/<invocation-timestamp>/config.yaml` | Hydra | Composed task configuration |
| `.hydra/<invocation-timestamp>/overrides.yaml` | Hydra | Command-line overrides |
| `.hydra/<invocation-timestamp>/hydra.yaml` | Hydra | Hydra runtime/launcher settings |
| `resolved_config.yaml` | Project workflow | Resolved task settings used for processing |
| `manifest.json` | Project workflow | Completed run's slide results and package versions |

The timestamp inside `.hydra/` is our customization: a refused retry preserves
its own configuration record. Only a manifest with `status: complete` establishes
completion; configuration files alone do not. See
[Hydra's working-directory documentation](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/working_directory/)
and the [project output contract](preprocessing.md#output-contract-and-visual-review).

Machine-specific settings can live in ignored `configs/local/default.yaml`.
To override root settings rather than create a `local` subtree, start it with:

```yaml
# @package _global_
data:
  manifest: /absolute/path/to/my_manifest.csv
  paths: []
```

Keep shared experimental recipes in tracked configs. Review ignored local
settings with `--cfg job --resolve` before comparing runs across machines.

## Why `src/` and `scripts/` are separate

| Location | Responsibility | Examples |
| --- | --- | --- |
| `configs/` | Declarative choices, thresholds and resource bounds | Dataset and masking profiles |
| `src/` | Importable processing functions and workflow orchestration | `pipeline.py`, `tiling.py`, `normalization.py`, `annotations.py` |
| `scripts/` | Commands for acquisition, conversion, inspection and maintenance | Downloaders, `convert_annotations.py`, `verify_run.py` |
| `notebooks/` | Teaching and exploration using the shared functions | WSI inspection and matched stain comparisons |
| `app.py` | Streamlit interface for reviewing saved results | Tables, previews and metadata filters |

For example, `src.annotations.convert_annotations(...)` owns annotation
conversion. `scripts/convert_annotations.py` parses `argparse` flags and calls
that same function. A notebook can import it without constructing a fake command
line or importing the CLI. Likewise, the preprocessing workflow and notebook
both use `src.pipeline.preprocess_masks`, so a correction has one shared owner.

The distinction is responsibility, not whether a file can run. `src.preprocess`
is deliberately an executable Hydra entrypoint, and some `scripts/` files own
substantial task-specific logic. Downloaders handle acquisition; the verifier
independently reconstructs expected results rather than just calling the exporter
again. There is no benefit in moving every script function into `src/`.

There are also two CLI styles: preprocessing uses Hydra's `key=value` overrides;
utilities such as annotation conversion use conventional `--input-format` and
`--max-images` flags. Hydra does not configure those utilities automatically.
Here `src` is the actual importable package (`from src...`), not an installed
package hidden beneath a conventional packaging `src/` directory. Commands run
from the checkout; the project is configured with `tool.uv.package = false`.

## Where to put a change

- **A new threshold or reusable run recipe:** add it to `configs/`, then make
  the owning processing function consume it explicitly.
- **An image, mask, coordinate or annotation operation:** put it in the relevant
  `src/` module with explicit inputs; avoid reading CLI arguments inside it.
- **A new operational command:** use `scripts/`, reusing `src/` functions where
  appropriate. Keep script-only orchestration there.
- **A teaching plot or interactive comparison:** keep presentation in the
  notebook or app and processing rules in shared modules.

This separation makes settings reviewable, preserves the same processing logic
across interfaces, and lets validation exercise functions through real workflows.
Hydra records choices; it does not guarantee that those choices are scientifically
appropriate or that two runs used identical data.

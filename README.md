# Digital pathology demo

A small CPU workflow for inspecting whole-slide images (WSIs), making tissue
masks and thumbnails, reviewing suspected artefacts, and exporting calibrated
CLAM patch coordinates. Source slides stay read-only in one local data directory.

## Quick start

Run from the repository root with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --locked
uv run --locked python scripts/download_examples.py
uv run --locked python -m src.preprocess hydra.run.dir=outputs/quickstart
uv run --locked python scripts/verify_run.py outputs/quickstart
```

`uv` creates the project-local `.venv` with Python 3.12 and the locked packages.
Select `.venv/bin/python` in your editor. Choose a **fresh output directory** for
each run; omit `hydra.run.dir` to use a timestamped directory automatically.
No classifier, model weights or training job is started by preprocessing.

To inspect configuration without reading slides:

```sh
uv run --locked python -m src.preprocess --cfg job --resolve
```

## Why uv?

[uv](https://docs.astral.sh/uv/concepts/projects/sync/) keeps this project's
packages isolated in `.venv` and provides one workflow for setup and execution.
`pyproject.toml` declares the Python and package requirements; the committed
`uv.lock` records resolved versions, including indirect dependencies. Teammates
can recreate the environment from these files instead of copying `.venv`.
`uv sync --locked` and `uv run --locked` use that resolution and fail if the lock
needs updating, helping prevent accidental dependency drift. `uv run` also selects
the project environment without manual activation.

## Browse results

```sh
uv run --locked streamlit run app.py
```

Open [the local viewer](http://127.0.0.1:8501), select a completed run, and filter
by dataset, participant or stain. Switch between original images, masks,
artefact overlays, green tissue/blue hole contours and patch coverage. Recorded
metadata can be downloaded as CSV. The server binds to localhost; telemetry is off.

Each run also contains one `index.html` report and flat `overviews/<dataset>/`
folders for Finder Gallery or Quick Look. Use [QuPath](docs/datasets.md#qupath)
for navigation through the original WSI at higher resolution.

## What the default does

For H&E, the pipeline applies CODAIT colour exclusions, TIA grayscale Otsu,
CLAM-style saturation masking and small-hole cleanup, then final exclusions.
Pinned upstream CLAM generates coordinates from the retained contours.
The [KPMP profile](docs/datasets.md#kpmp-ckd) uses a separate saturation-mask
recipe with H&E artefact rules disabled for its special stains.

- Export every eligible **256-pixel** tile coordinate at a target **0.5 µm/pixel**,
  recording native footprints and effective resolution after rounding.
- Save small coverage previews for every slide. Actual patch PNGs are opt-in for
  one exact `export.pixel_slide_id`, capped at 128 images by default.
- Keep intermediate masks, applied settings and provenance for review.
  Thumbnail masks and coverage previews do not require decoding every patch.

See the [preprocessing guide](docs/preprocessing.md) for controls, physical-scale
assumptions, HDF5 attributes, output folders and verification commands.

## Guides and examples

| Task | Guide |
| --- | --- |
| Understand Hydra and the code layout | [Hydra and project structure](docs/hydra-and-project-structure.md) |
| Configure masking, tiling and outputs | [Preprocessing](docs/preprocessing.md) |
| Obtain datasets and create QuPath projects | [Datasets](docs/datasets.md) |
| Walk through metadata, masks, QC and optional stain comparisons | [Preprocessing demo](notebooks/preprocessing_demo.ipynb) |
| Convert COCO polygons or instance masks to GeoJSON/TIFF | [Annotation conversion](docs/annotation-conversion.md) |
| Compare pretrained nuclei predictions on two annotated patches | [Annotation and segmentation demo](notebooks/annotation_segmentation_demo.ipynb) · [Setup](docs/pretrained-models.md) |
| Classify bounded WSI bags with patient-disjoint splits | [WSI classification demo](notebooks/wsi_classification_demo.ipynb) |
| Decompose and assemble encoders, decoders and heads | [Model components and Lightning](docs/model-components.md) |
| Run bounded Lightning classification, MIL and segmentation | [Training](docs/training.md) |
| Plan evaluation splits, metrics and reference annotations | [Computational pathology considerations](docs/computational-pathology-considerations.md) |
| Run Ruff and enable Git checks | [Linting and pre-commit hooks](docs/linting.md) |
| Review upstream sources and licences | [Third-party attribution](third_party/README.md) |

The preprocessing notebook uses the same reusable Python functions as the CLI. Its optional
stain comparison pairs original-background processing with one shared estimated
background colour. Normalization is **off in the default WSI pipeline**; its
assumptions and API are documented in the [guide](docs/preprocessing.md#optional-stain-normalization).

## Scope and sharing

Masks and pen rules are heuristics and can exclude real tissue. Review their
outputs before downstream use. These examples check preprocessing mechanics;
they do not establish biological accuracy or classifier performance. Keep all
slides, stains and patches from one participant in the same evaluation split.

Commit code, configuration, small public manifests, lockfiles and documentation.
Slides, generated outputs, `.venv`, personal guidance and local configuration
stay out of Git. Runtime artifacts can contain local paths; review them before
sharing separately. Preserve the upstream licences, including the vendored
CLAM and HoVer-NeXt GPLv3 notices. Pretrained inference uses the optional setup
above; annotation conversion and stain normalization use the base environment.

# Digital pathology demo

A small CPU preprocessing workflow for whole-slide images (WSIs): bounded thumbnails,
tissue masks, configurable pen/dark-pixel/small-component exclusion, and a local HTML
review report, plus bounded CLAM/HIPT-style patch exports. The source slides are read only. No model or weights are downloaded by
the preprocessing command.

## Setup

From this repository's root, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --locked
uv run python scripts/download_examples.py
uv run python -m src.preprocess
```

`uv` creates the project's `.venv` using Python 3.12 and the committed `uv.lock`.
Select `.venv/bin/python` in your editor. No global Python environment is modified.
TIAToolbox brings substantial image/scientific/PyTorch dependencies even for this CPU
workflow. OpenSlide's binary wheel is included through TIAToolbox.
If the default cache is not writable in a sandbox, prefix commands with
`UV_CACHE_DIR=/tmp/pathology-uv-cache`; use `MPLCONFIGDIR=/tmp/pathology-matplotlib`
if Matplotlib cannot write its font cache.

## Run and review

```sh
# Tiny public Aperio example: maximum 1024 pixels per thumbnail edge.
uv run python -m src.preprocess hydra.run.dir=outputs/my-demo

# Real pen-mark examples from CODAIT.
uv run python -m src.preprocess data=pen_examples

# Alternate masking recipes.
uv run python -m src.preprocess masking=tia_morphological
uv run python -m src.preprocess masking=clam_saturation
uv run python -m src.preprocess masking=codait_od

# Your own slide; Hydra list syntax is quoted for the shell.
uv run python -m src.preprocess 'data.paths=[/path/to/slide.svs]' thumbnail.max_size=512

# Review the impact of excluding only green and blue pen candidates.
uv run python -m src.preprocess 'artifacts.pen_colors=[green,blue]'

# Thumbnail-only review (skip coordinate and region exports).
uv run python -m src.preprocess export.enabled=false

# Check the output contract against the real images just processed.
uv run python scripts/verify_run.py outputs/my-demo
```

Open `index.html` in the run directory. It contains original, candidate tissue,
suspected artefacts, retained mask, filtered preview, and review-overlay panels.
Individual reason masks and `metadata.json` are stored in each slide's directory.
`manifest.json` records versions, dimensions, scale, counts, timings and warnings;
`resolved_config.yaml` and `.hydra/` record settings. Choose a fresh output directory
for each run; existing result sets are never overwritten.

For Finder Gallery / Quick Look, open `overviews/tcga_luad/` or
`overviews/tcga_lusc/` inside the run directory. Each is a **flat directory** with
all slides' `__thumbnail.png`, `__mask.png`, `__preprocessed.png`, and
`__overlay.png` files. Sort by name to keep views of each case together; use Finder
search for `__preprocessed` or `__mask` to browse just one view type. Select the
images and press Space for Quick Look, or Command-4 for Finder Gallery.

By default, processing is sequential and OpenCV uses one thread. Thumbnail reading is bounded
to 16 million pixels per block and 100 million pixels in total per slide. When no
whole pyramid level fits, the lowest level is read and resized in blocks; a slide
exceeding the total-pixel budget fails explicitly. Blockwise resizing may introduce
minor interpolation differences at block boundaries; this is a thumbnail preview.
Thumbnail masks are **not full-resolution masks**. Use the recorded independent x/y
scale factors for coordinate mapping; small structures are lost at thumbnail scale.
Component areas and kernel sizes are in thumbnail pixels, so changing thumbnail
size changes their physical meaning. PNG/JPEG inputs are supported for small real
examples; they are not treated as WSIs with known physical calibration.

## CLAM / HIPT output layout

The default export additionally produces the following per dataset, using the
source slide stem consistently in every artifact:

```text
outputs/<run>/
  overviews/tcga_luad/                 # All small PNGs together for Finder
  datasets/tcga_luad/
    WSIs/<slide_id>.svs               # Relative symlinks to local inputs
    dataset.csv                      # case_id, slide_id, label, wsi_path
    extracted_level0_patch256_fp/
      masks/<slide_id>.jpg            # Segmentation/artefact overlay
      tissue_masks/<slide_id>.png     # Lossless binary retained-tissue mask
      patches/<slide_id>.h5           # coords: N x 2 int64, level-0 top-left x/y
      stitches/<slide_id>.jpg         # Downsampled actual selected patch pixels
      process_list_autogen.csv
    extracted_level0_patch4096_fp/
      masks/, tissue_masks/, patches/, stitches/
      tar_patch_4096/<slide_id>.tar    # PNG + JSON pairs, WebDataset-style keys
      process_list_autogen.csv
```

The same layout is produced for `tcga_lusc/`. The HDF5 `coords` dataset includes
CLAM's `patch_size`, `patch_level`, `downsample`, `level_dim`,
`downsampled_level_dim`, `name`, and `save_path` attributes. Its extra
`tissue_fraction` dataset records estimated coverage from the retained thumbnail
mask. Regions are full, non-overlapping source-level-0 rectangles; no boundary
padding is used. At most **128 256-pixel patches and two 4096-pixel regions per
slide** are selected, spread deterministically across eligible grid positions
with at least 60% estimated tissue coverage. Eligibility scans at most 200,000
positions. Empty coordinate sets are recorded explicitly, including for slides
smaller than 4096 pixels.

Patch reads add at most about 42 million source pixels per slide at these defaults,
with only one region decoded at a time. The 4096-pixel PNG regions retain source
pixels exactly; they contain background/artefacts where the rectangular crop does,
so a tissue-coverage threshold is not a guarantee that every patch pixel is tissue.
`export.save_region_tar=false` skips the raw region archive. Adjust bounded counts
with `'export.max_patches=[64,1]'`.

This follows [CLAM fast patching](https://github.com/mahmoodlab/CLAM/blob/master/create_patches_fp.py)
and [HIPT's documented organization](https://github.com/mahmoodlab/HIPT#downloading--preprocessing--organizing-tcga-data).
We use `extracted_level0_...`, **not `extracted_mag20x_...`**, because native
magnification is not harmonized. MPP and level metadata are retained per slide.
The mask/coverage selection differs from CLAM's contour acceptance algorithm.
There are no fabricated `.pt` features or empty feature-directory placeholders.
Compatibility here concerns the preprocessing data contract, not numerical
equivalence or validation of downstream CLAM/HIPT models.

```sh
uv run python scripts/verify_run.py outputs/tcga-demo --check-regions
```

This checks mask partitions, dimensions/counts, HDF5 coordinates and bounds, TAR
member counts, and the first stored region per WSI against a fresh OpenSlide read.

## Ten TCGA lung slides

```sh
uv run python scripts/download_tcga.py
uv run python -m src.preprocess data=tcga hydra.run.dir=outputs/tcga-demo
uv run python scripts/verify_run.py outputs/tcga-demo
```

`manifests/tcga_demo.csv` freezes five TCGA-LUAD and five TCGA-LUSC open-access
diagnostic primary-tumour slides, from ten distinct patients and five tissue-source
sites per class. Selection targets approximately 60, 130, 250, 400 and 650 MB per
class, subject to 30–800 MB eligibility and site uniqueness. The complete roster is
about 2.98 GB. File size and site diversity are proxies, not proof of histological
diversity. Site is also a possible confounder for classification.

The downloader reuses the frozen manifest, resumes partial files, and verifies GDC
size and MD5 before promoting a download to `.svs`. It uses at most four downloads at a time,
a 10 GB total budget, and a disk-space check. If the manifest is absent it queries
GDC to build a new roster; preserve the committed manifest for exact replay.
Public case/file IDs, original names, labels and relative paths are included.
All large data and run outputs are Git-ignored.

These are project-level LUAD/LUSC labels, not pixel-level tumour annotations.
This is a convenience sample for plumbing and demonstrations, not a clinical
benchmark. Keep all patches from each patient in a single split. Ten patients
cannot support a credible generalization claim. The repository currently provides
preprocessing and labelled input manifests, not a trained WSI classifier.

## QuPath

Install the appropriate build from [official QuPath releases](https://github.com/qupath/qupath/releases).
On macOS use the Apple Silicon build for arm64 or Intel for x64.
In QuPath choose **File → Open** and select an `.svs` under `data/tcga/`.
The manifest maps its UUID filename to its original TCGA case and label.
For a reusable image list, create a QuPath project in an ignored directory under
`data/`, then add the ten slides. Review tissue and pen masks against the original
slide at higher magnification before using exclusions for model training.

To create the ten-slide project reproducibly, run the helper with your installed
QuPath command-line executable:

```sh
uv run python scripts/create_qupath_project.py --qupath /path/to/QuPath-executable
```

It creates `data/qupath/tcga-lung-demo/project.qpproj`, names images by class and
TCGA case, stores case/site/GDC metadata, sets H&E image type, and generates small
project thumbnails. It references the canonical absolute WSI locations under
this repository's `data/tcga/`; **it never copies WSI data**. Existing QuPath
projects are preserved. After moving a repository again, use QuPath's URI repair
dialog or create a new project; do not duplicate the slides to satisfy old paths.

## Provenance and scope

- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template):
  `src/` + composable `configs/`, overrides and per-run Hydra provenance. This is a
  focused scaffold, not a copy of its MNIST training application. Optional future
  training dependencies: `uv sync --locked --extra train`; no Lightning trainer is
  used for preprocessing.
- [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox): actual WSI reader
  and Otsu/morphological tissue maskers.
- [CLAM](https://github.com/mahmoodlab/CLAM): independently implemented
  saturation/median/threshold/closing recipe. This is not CLAM's contour/patch
  pipeline, uses different area units, and does not vendor its GPL source.
- [histolab](https://github.com/histolab/histolab): reference for composable masking
  and pen filters. Not installed: released histolab 0.7 requires NumPy ≤1.24.4 and
  scikit-image <0.19.4, which conflict with TIAToolbox 2.1.3. We use the attributed
  CODAIT filter extraction instead of bypassing dependency constraints.
- [CODAIT deep-histopath](https://github.com/CODAIT/deep-histopath): adapted pen and
  gray filters from `deephistopath/wsi/filter.py`, and optical density from
  `deephistopath/preprocessing.py`. See `third_party/README.md` for pinned sources,
  preserved licence/notices, and the precise changes.

Pen-colour thresholds can reject real stains. Dark regions can be dense tissue;
small components can be real fragments. The original candidate mask and every
exclusion reason are retained for review. Gray filtering is off by default.
No blur/fold detector, stain normalization, feature model, or training loop is
included. Add them only when the demo needs them.

## Sharing

Commit source, configs, scripts, the small public roster, `pyproject.toml`,
`.python-version`, `uv.lock`, and documentation/third-party notices in reviewable
groups. Do not commit `.venv`, slides, reports, personal `AGENTS.md`, or local configs.
Runtime logs/configs can contain local paths: review outputs before sharing them
separately. Upstream licences are preserved under `third_party/`; choose a licence
for this project's original code before publishing it publicly.

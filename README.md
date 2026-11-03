# Digital pathology demo

A small CPU preprocessing workflow for whole-slide images (WSIs): bounded thumbnails,
tissue masks, configurable pen/dark-pixel/small-component exclusion, a lightweight
Streamlit browser, and CLAM/HIPT-style coordinate exports. The source slides are read only. No model or weights are downloaded by
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

# Alternative maskers; default: CODAIT → TIA Otsu → CLAM-style masking/cleanup.
uv run python -m src.preprocess masking=clam_saturation
uv run python -m src.preprocess masking=tia_otsu
uv run python -m src.preprocess masking=tia_morphological
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

Start the local slide browser:

```sh
uv run --locked streamlit run app.py
```

Open `http://127.0.0.1:8501`. Choose a completed run, filter by dataset, patient or
stain, switch thumbnail/patch-coverage views, select rows, and download their
metadata as CSV. The table exposes recorded slide dimensions, MPP x/y, objective
power, vendor, pyramid levels, file size and public identifiers. Missing metadata
is shown as unknown; tissue-component counts are not estimated. Patch coordinate
counts, actual saved images and thumbnail-derived coverage are labelled separately.
Validation runs are hidden by default. Use QuPath for full-resolution navigation.
The server binds only to localhost and usage telemetry is disabled.

The workflow also writes one optional-to-view `index.html` per **run**, never one
HTML file per WSI. It contains original, candidate tissue,
suspected artefacts, retained mask, filtered preview, and review-overlay panels.
Individual reason masks and `metadata.json` are stored in each slide's directory.
`manifest.json` records versions, dimensions, scale, counts, timings and warnings;
`resolved_config.yaml` records the resolved task settings. Hydra writes each
invocation's config and overrides under `.hydra/<timestamp>/`, preserving earlier
provenance even when an attempted rerun is refused. Choose a fresh output directory
for each run; existing result sets are never overwritten. Relative manifest and
WSI paths are resolved from the launch directory, including when
`hydra.job.chdir=true`. Preview the composed configuration without reading slides
with `uv run --locked python -m src.preprocess --cfg job --resolve`.

For Finder Gallery / Quick Look, open `overviews/tcga_luad/` or
`overviews/tcga_lusc/` inside the run directory. Each is a **flat directory** with
all slides' `__thumbnail.png`, `__mask.png`, `__preprocessed.png`,
`__overlay.png`, and `__tissue_holes.png` files. Sort by name to keep views of each case together; use Finder
search for `__preprocessed` or `__mask` to browse just one view type. Select the
images and press Space for Quick Look, or Command-4 for Finder Gallery.

Slides and pixel reads are processed sequentially; OpenCV uses one thread.
Upstream CLAM uses at most four coordinate workers per contour. Thumbnail reading is bounded
to 16 million pixels per block and 100 million pixels in total per slide. When no
whole pyramid level fits, the lowest level is read and resized in blocks; a slide
exceeding the total-pixel budget fails explicitly. Blockwise resizing may introduce
minor interpolation differences at block boundaries; this is a thumbnail preview.
Thumbnail masks are **not full-resolution masks**. Use the recorded independent x/y
scale factors for coordinate mapping; small structures are lost at thumbnail scale.
Mask component-area thresholds and kernel sizes are in thumbnail pixels, so changing thumbnail
size changes their physical meaning. PNG/JPEG inputs are supported for small real
examples; they are not treated as WSIs with known physical calibration.

The **Tissue + holes** view draws green tissue contours and blue hole contours
using [CLAM's default colours](https://github.com/mahmoodlab/CLAM/blob/master/wsi_core/WholeSlideImage.py).
Only contour pixels change; tissue and hole interiors retain the original thumbnail
colours without fills or tinting. It uses the actual saved clean mask, without a
hole-area cutoff or tissue-piece counts. Holes are a
thumbnail-mask property, not anatomical annotations. Background connected to the
image boundary is excluded from the hole mask. Lossless `tissue_holes.png` and
`hole_mask.png` files are saved per slide; `__tissue_holes.png` copies join the
flat gallery, and the CLAM-style `masks/` JPEGs use the same visualization.
The existing `overlay.png` still shows artefact exclusions separately.

To add these visualizations to an existing completed run without reading WSIs or
regenerating patch images:

```sh
uv run --locked python -m scripts.refresh_tissue_holes outputs/tcga-mpp05 outputs/kpmp-ckd-demo
```

## CLAM / HIPT output layout

The default export additionally produces the following per dataset, using the
source slide stem consistently in every artifact:

```text
outputs/<run>/
  overviews/tcga_luad/                 # All small PNGs together for Finder
  datasets/tcga_luad/
    WSIs/<slide_id>.svs               # Relative symlinks to local inputs
    dataset.csv                      # case_id, slide_id, label, wsi_path
    extracted_mpp0p5_patch256_fp/
      masks/<slide_id>.jpg            # Green tissue contours, blue hole contours; no fill
      tissue_masks/<slide_id>.png     # Lossless binary retained-tissue mask
      patches/<slide_id>.h5           # Level-0 coords + calibrated source/output sizes
      stitches/<slide_id>.jpg         # Thumbnail pixels inside every exported patch rectangle
      coverage_masks/<slide_id>.png   # Lossless union of exported patch footprints
      patch_images/<slide_id>/       # Selected WSI only: bounded 256 PNGs + index.json
      sampled_pixel_stitches/         # Selected WSI only: actual patch pixels, downsampled
      process_list_autogen.csv
```

The same layout is produced for `tcga_lusc/`. The HDF5 `coords` dataset includes
CLAM's `patch_size`, `patch_level`, `downsample`, `level_dim`,
`downsampled_level_dim`, `name`, and `save_path` attributes. Here `patch_size` is
the **native source footprint**, while `output_patch_size` is the 256-pixel image
size after resampling. Extra attributes record native, target and effective MPP.
The `tissue_fraction` dataset records estimated coverage from the retained thumbnail
mask. The default path is **CODAIT exclusions → TIA grayscale Otsu → CLAM-style
saturation masking and bounded small-hole cleanup → final QC → upstream CLAM tiling**. Each retained tissue contour anchors its
own level-0 grid of full rectangles, with no boundary padding. CLAM's `four_pt`
rule accepts a tile when any of its four interior probes lies on or inside the
contour, then rejects a tile whose centre is inside an associated hole. The
configured 60% retained-mask coverage threshold is an additional QC step, not a
CLAM default. Every eligible 256-pixel coordinate is exported
(`export.coordinate_limit=null`), with at most 200,000 candidate positions across
contours. An explicit coordinate limit opts into subsampling. Empty coordinate
sets are recorded when no tile qualifies. Complete eligible-grid export is
not complete tissue-area coverage: boundary strips, low-coverage tiles and tissue
missed by the thumbnail mask remain outside the grid.

Actual source-pixel images are disabled unless `export.pixel_slide_id` names one
exact WSI filename stem in the run. For that slide, at most **128 256-pixel PNGs**
are saved, spread deterministically across exported locations. All other slides save only coordinates and small coverage previews.
`stitches/` uses existing thumbnail pixels to show the full exported footprint;
`sampled_pixel_stitches/` shows the bounded actual pixel samples. HDF5
`preview_indices` and the small-patch `index.json` map actual images back to the
full coordinate bag. The demo selects TCGA-55-6972 for actual images; KPMP uses
coverage previews only.

Patching targets **0.5 µm/pixel** for every WSI (`export.target_mpp=0.5`), a common
20×-equivalent resolution convention. Source objective labels alone do not define
physical pixel size. The native footprint is rounded from
`output_patch_size × target_mpp / native_mpp`; for example, 256 output pixels use
508 source pixels at 0.252 MPP and 256 source pixels at 0.5005 MPP. Output PNGs
use Pillow LANCZOS resampling. The resulting effective MPP is recorded separately
so integer rounding is visible; these images are not unchanged native crops.
Missing, invalid or anisotropic MPP fails explicitly rather than guessing.
For OpenSlide inputs, calibration must come from its recorded mpp-x/mpp-y
properties; objective-power inference is not accepted. Scans more than 1% coarser
than the target are rejected to avoid presenting enlarged images as recovered resolution.

Pixel reads are sequential and bounded by `export.max_patch_read_pixels` (100
million source pixels per crop). At 0.252 native MPP, the selected slide's default
128 patches read approximately 33 million source pixels in total. PNGs contain background/artefacts where the rectangular footprint does,
so a tissue-coverage threshold is not a guarantee that every patch pixel is tissue.
Only 256-pixel tiles are enabled (`export.patch_sizes=[256]`); region TAR export
is disabled. Adjust the bounded image count with `'export.max_patches=[64]'`.
Previously produced 4096-pixel exports are archived outside the active demo runs;
they are not included in the current Streamlit views or coordinate manifests.

This follows [CLAM fast patching](https://github.com/mahmoodlab/CLAM/blob/master/create_patches_fp.py)
and retains compatible parts of [HIPT's documented organization](https://github.com/mahmoodlab/HIPT#downloading--preprocessing--organizing-tcga-data).
The current demo does not generate HIPT's large-region inputs.
The `extracted_mpp0p5_...` names describe the calibrated target resolution; native
scanner magnification and MPP remain recorded separately. CLAM's
[coordinate-bag reader](https://github.com/mahmoodlab/CLAM/blob/master/dataset_modules/dataset_h5.py)
reads the source footprint from `coords.attrs.patch_size`. Its image transform
must resize to the recorded `output_patch_size` to retain the intended output
scale (for the 256 bag, set `--target_patch_size 256` in its feature-extraction CLI;
the current default is 224). No downstream model feature extraction is run here.
The default tiler calls upstream CLAM `WholeSlideImage.process_contour` at
revision `53e2409d4a8189c682c173382964a85f114f923c`, using the reviewed final mask
contours. The bounded thumbnail segmentation is an independent CLAM-style
recipe with CODAIT and TIA prefilters. Enclosed background components of at most
16 thumbnail pixels are filled by default; larger holes remain. This raster-area
policy differs from upstream CLAM's contour-area and maximum-hole-count policy.
CODAIT exclusions are reapplied after filling. The additional retained-mask
coverage threshold means this is a composed workflow, not unmodified upstream
CLAM segmentation. The approved source subset and GPLv3 licence are in
`third_party/CLAM/`; file hashes are recorded in `SOURCE.json`. The earlier slide-wide
grid remains available as `export.tiling_method=coverage_grid` for reproducibility.
No model features or downstream CLAM/HIPT validation are produced.

```sh
uv run --locked python scripts/verify_run.py outputs/tcga-mpp05 --check-regions
```

This checks mask partitions, dimensions/counts, HDF5 coordinates and bounds,
the complete eligible grid against the saved mask, and the first/last saved
256-pixel patches against fresh OpenSlide reads with the declared resampling.

## Ten TCGA lung slides

```sh
uv run python scripts/download_tcga.py
uv run --locked python -m src.preprocess data=tcga \
  export.pixel_slide_id=20c9afac-0d0c-4fcf-b3bf-a2beba3dd76b \
  hydra.run.dir=outputs/tcga-mpp05
uv run --locked python scripts/verify_run.py outputs/tcga-mpp05 --check-regions
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

## Five KPMP CKD participants with four stains each

The separate `kpmp-ckd-demo` QuPath project uses twenty public light-microscopy
WSIs: one H&E, PAS, trichrome (`TRI`) and silver (`SIL`) slide per participant.
`manifests/kpmp_ckd_demo.csv` freezes participant IDs, stain labels, public file and
package UUIDs, source URLs, exact byte sizes, enrollment/adjudication categories,
and canonical paths. All participants are **CKD-enrolled**; this is distinct from
their primary adjudicated disease category. The public metadata supports matching
by participant, not a claim of registered serial sections or a shared biopsy.
Filename-derived sample labels are retained only as source labels.

```sh
uv run --locked python scripts/download_kpmp.py
uv run --locked python -m src.preprocess --config-name kpmp hydra.run.dir=outputs/kpmp-ckd-demo
uv run --locked python scripts/verify_run.py outputs/kpmp-ckd-demo --check-regions
uv run --locked python scripts/create_qupath_project.py \
  --qupath /path/to/QuPath-executable \
  --manifest manifests/kpmp_ckd_demo.csv --project-name kpmp-ckd-demo \
  --expected-patients 5 --expected-slides 20
```

The downloader has a 5 GB total budget, two concurrent transfers, and a 1 GB
per-slide bound. Existing verified files are reused. It validates repository byte
sizes, bounded OpenSlide readability, and local SHA-256 receipts; the source API
does not provide a published checksum. Recorded SHA-256 values protect subsequent
replays, but the first local hash is not an independent upstream integrity check.
The KPMP proxy may ignore resume requests, in which case the same partial file is
restarted. Original WSIs live only under `data/kpmp/CKD/<participant>/`.

Choose `kpmp-ckd-demo` in Streamlit for all twenty slides, or use the flat
`outputs/kpmp-ckd-demo/overviews/kpmp_ckd/` folder in Finder Gallery. Filenames
include participant and stain. CLAM/HIPT-style exports retain patient IDs and
stains, exporting all eligible coordinates in level-0 space for the shared
0.5 MPP patch target, with thumbnail-derived coverage stitches and no actual
patch-image exports.

The KPMP config uses CLAM saturation masking and contour-based tiling. H&E pen-colour, dark-pixel,
gray and small-component exclusions are disabled: special-stain colours and dark
silver deposits must not be discarded as if they were H&E artefacts. The previews
therefore show background exclusion, not validated artefact removal across stains.
QuPath uses H&E image type for H&E and **Brightfield (other)** for PAS/TRI/SIL;
no H&E stain-deconvolution assumptions are applied to special stains.

This is a convenience cohort for browsing and preprocessing, not a disease
classifier benchmark. Keep every stain from a participant in the same data split.
Data are provided by the [Kidney Precision Medicine Project](https://www.kpmp.org/),
accessed November 2, 2026. Follow KPMP's [citation guidance](https://www.kpmp.org/help-docs/citing-kpmp-data)
when presenting or publishing results; see its [enrollment versus adjudication definitions](https://www.kpmp.org/help-docs/adjudication-vs-enrollment).

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
- [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox): bounded overview
  reader, Otsu/morphological maskers, and optional Reinhard/Macenko/Ruifrok/Vahadane
  normalisation. Tiling and tile reads use CLAM/OpenSlide.
- [CLAM](https://github.com/mahmoodlab/CLAM): pinned upstream tiler and OpenSlide
  handle; independent saturation/median/threshold/closing segmentation recipe.
  Small-hole filling and extra mask-coverage QC are explicit demo policies.
  See `third_party/README.md` for the vendored GPLv3 subset and hashes.
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
Optional TIA stain-normalisation helpers support contour-based white-background
preparation. They are not invoked by the preprocessing entrypoint.
No validated blur/fold detector, feature model or training loop is included.

## Sharing

Commit source, configs, scripts, the small public roster, `pyproject.toml`,
`.python-version`, `uv.lock`, and documentation/third-party notices in reviewable
groups. Do not commit `.venv`, slides, reports, personal `AGENTS.md`, or local configs.
Runtime logs/configs can contain local paths: review outputs before sharing them
separately. Upstream licences are preserved under `third_party/`; choose a licence
for this project's original code before publishing it publicly.


## Preprocessing stages and reusable functions

`python -m src.preprocess` remains the Hydra entrypoint. The default H&E mask
recipe is now `codait_tia_clam`; coordinate exports use upstream CLAM and record
schema version 6. Existing result folders are unchanged. The KPMP configuration
explicitly keeps its CLAM saturation recipe and disabled H&E artefact rules.

| Stage | Module | Reusable API |
| --- | --- | --- |
| Inspect/read | `src/inspection.py` | `read_slide_metadata`, `read_thumbnail` |
| Detect tissue | `src/masking.py` | `create_tissue_mask`, `saturation_mask`, `otsu_mask` |
| Review/exclude artefacts | `src/quality.py` | `exclude_artifacts`, `tile_quality` |
| Compose masking/QC | `src/pipeline.py` | `preprocess_masks` with ordinary config mappings |
| Calibrate/tile/read | `src/tiling.py` | `calibrated_patch_size`, `clam_coordinates`, `read_tile`, `contour_tile_mask` |
| Optional stain comparison | `src/normalization.py` | `fit_stain_normalizer`, `normalize_tile` |
| Export/review | `src/export.py`, `src/report.py` | CLAM-compatible exports and review report |
| Shared arrays | `src/image_utils.py` | Validation, components, `fill_small_holes`, `tissue_hole_preview`, `white_background`, `save_png` |

RGB arrays are `uint8 H×W×3`; masks are `bool H×W` with `True` selecting pixels.
Locations are level-0 `(x, y)`; sizes are `(width, height)`; arrays index `[y, x]`.
The numerical utilities take arrays and explicit arguments, without Hydra or
implicit output directories, and do not modify input arrays. Mask component and
hole areas are measured in mask pixels, not square micrometres. Coverage is an
approximation at thumbnail resolution. Effective MPP after rounding is recorded.

`read_tile` takes a caller-owned CLAM/OpenSlide handle, checks pixel budgets,
rejects padding or missing pixels, and returns RGB after Lanczos resizing.
The CLAM adapter requires square source tiles and the export adapter requires
isotropic MPP; general calibration utilities also support rectangular regions.
`tile_quality` returns measurements, not a validated focus decision.

Compatibility imports from `src.preprocess` (`read_thumbnail`, `tissue_mask`,
`clean_mask`, `save_png`, `tissue_hole_preview`), `src.filters.validate_rgb`, and
`src.export.coordinates` remain available. New callers should import the owning
stage. Legacy `tissue_mask` handles single-mask recipes; use `preprocess_masks`
for the combined recipe. `src/clam_tiling.py` is retained for historical callers;
the default uses pinned upstream CLAM through `src/tiling.py`.

Combined preprocessing records CODAIT prefilter exclusions as a percentage of
all thumbnail pixels. Final cleanup exclusions use the candidate tissue mask as
the denominator; they do not include exclusions applied before tissue detection.

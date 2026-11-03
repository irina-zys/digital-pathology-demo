# WSI preprocessing guide

[Quick start](../README.md) · [Datasets and QuPath](datasets.md) ·
[Annotation conversion](annotation-conversion.md)

Run CLI commands from the repository root. The
[Preprocessing demo](../notebooks/preprocessing_demo.ipynb) provides a guided version:
select the project's `.venv/bin/python`, restart the kernel and run all cells.
It discovers the project from ancestor directories, with an explicit project-path
override available when launched elsewhere. New notebook outputs go under the
repository's ignored `outputs/preprocessing_demo/` directory. Validated demo
notebooks retain matching executed figures and tables in Git. Bulk run artifacts
and detailed provenance remain under ignored `outputs/`. Code keeps portable
project-relative paths when outputs are retained.

**Preprocessing is a user choice and a scientific decision.** Defaults are
illustrative settings, not a universally validated best recipe. Review the
tradeoff between artefact retention and tissue loss for the stain, morphology and
downstream question. Persist the chosen settings and evaluate representative
failures as well as successful examples.

The notebook's setup obtains missing inputs through repository scripts. It
compares TIA Otsu, native CLAM segmentation and the combined recipe on a severely
pen-marked WSI. Native CLAM uses the pinned CLI-default segmentation parameters at the best
available native pyramid level for downsample 64. This is a workflow comparison,
not a scale-matched accuracy ranking. Its tiling section
exports the complete eligible HDF5/CSV coordinate grid and full coverage stitch;
12 separate patches are read for visual QC. Sampled QC does not filter the full
coordinate export or approve unsampled patches.

Comparison figures keep one patch or region per row and matching views or methods
in columns. The severe-artefact figure places original RGB, TIA Otsu, native CLAM
and the combined recipe side by side, with retained RGB above and cyan overlays
of every excluded thumbnail pixel below. Exclusions may be glass, lung airspaces,
artefacts or tissue. Native CLAM shows its actual projected green exterior and
blue hole boundary lines; contour interiors are not colour-filled. These are
display annotations, without RGB repair or inpainting. QC is shown on slide thumbnails: original, green tissue/blue hole contours, and
sampled coverage decisions marked at their slide locations. Stain figures retain
patch close-ups, with patch identity fixed across treatments and methods.

## Customize settings for each dataset

**Custom preprocessing settings are normally required.** The provided recipes
illustrate explicit choices; they do not automatically adapt to a new collection
or establish suitable tissue segmentation. Tailor settings to the stain, tissue
architecture, scanner/native MPP, artefacts and intended downstream model.

- Choose the masking method and colour/artefact rules for the actual stain.
  H&E pen or colour thresholds may remove genuine structures in special stains.
- Review component removal, hole filling and smoothing against tissue morphology.
  These thresholds use thumbnail pixels: changing `thumbnail.max_size` also
  changes their effective physical scale. An anatomical lumen is not an artefact.
- Set `export.target_mpp`, patch size and tissue-coverage/placement rules to match
  the downstream task. A valid coordinate export does not establish adequate
  sampling or disease-relevant tissue coverage.
- Start with a bounded set of representative slides, including difficult cases.
  Inspect original RGB, intermediate masks, exclusions, tissue/hole contours and
  coverage; inspect source-resolution patches where thumbnail review is unclear.
  Adjust the recipe before scaling to the cohort.

Store repeatable cohort/stain recipes in `configs/` and inspect the composed
settings with `--cfg job --resolve`. `data=tcga` changes the input roster only;
`--config-name lung` also selects the explicit lung settings. Keep each reviewed
configuration and output in a fresh run directory. Validation scripts check
mechanics and artifact consistency; they do not approve the scientific settings.

## Lung tissue: anatomy and artefacts need separate decisions

Alveolar air spaces and the thin tissue between them are normal lung architecture
([SIU histology atlas](https://histology.siu.edu/intro/IN015b.htm)). A hole in a
thresholded mask is only a topological observation. It does not distinguish an
alveolus or lumen from a tear, processing gap, blank glass or exclusion caused by
ink. Likewise, a small disconnected tissue component is not automatically debris.
Review the original RGB and, where needed, the full-resolution WSI before deciding
which structures to exclude. Low retained area is not by itself failed QC.

The TCGA preprocessing notebook composes [configs/lung.yaml](../configs/lung.yaml),
and new coordinate runs prepared by the notebook bootstrap use the same profile.
It retains the combined CODAIT/TIA/saturation pipeline with these explicit choices:

| Parameter | Generic H&E | Lung starting preset | Rationale |
| --- | ---: | ---: | --- |
| `masking.median_size` | 7 | 1 | Avoid median removal of narrow raster structures |
| `masking.closing_size` | 3 | 0 | Avoid bridging air spaces at thresholding |
| `masking.max_hole_area` | 16 | 0 | Do not fill holes merely because they are small |
| `masking.smoothing_kernel_size` | 3 | 3 | Round binary-mask boundaries; review changes to septa and air spaces |
| `artifacts.min_tissue_area` | 64 | 16 | Retain smaller disconnected fragments |
| `export.min_tissue_fraction` | 0.60 | 0.10 | Admit useful sparse tissue surrounding air spaces |
| `export.exclude_hole_centers` | true | false | Permit a patch centred in a hole when surrounding tissue meets coverage |

Hole pixels that remain after binary-mask smoothing are background and do not
contribute to the tissue fraction. Turning off CLAM's centre-in-hole rejection changes patch placement;
it does not fill a hole, recolour RGB or declare the hole anatomical. The
four-point outer-contour test, physical calibration and all-coordinate export
remain in use. The mask can retain less total area while the patch grid becomes
more inclusive.

CODAIT colour rules still flag suspected pen/dark pixels separately. At a 10%
coverage threshold, up to 90% of a selected footprint can be unretained in the
thumbnail mask. This is not a measured high-resolution tissue fraction: the raw
crop can include air spaces, glass or artefacts and is not guaranteed artefact-free.
The 16-pixel component threshold can still discard genuine small fragments; set
it to 0 only after reviewing the increased noise and contour workload. Kernel and
area units are thumbnail pixels, not fixed physical sizes. These are scientific
starting settings, not validated lung segmentation or a universal recipe.

```sh
uv run --locked python -m src.wsi_preprocessing.preprocess --config-name lung --cfg job --resolve
uv run --locked python -m src.wsi_preprocessing.preprocess --config-name lung \
  hydra.run.dir=outputs/tcga-lung
```

Selecting `data=tcga` alone changes the roster, not the masking/export policy.
Use `--config-name lung` explicitly. Existing complete runs retain their recorded
settings; choose a fresh output directory to apply the new profile. The notebook
shows generic versus lung contours on the same thumbnail and the binary masks
before and after the configured 3-pixel smoothing. The smoothed mask, after final
artefact exclusions and component filtering, is used for lung tiling.

## Configuration and run lifecycle

Hydra composes `configs/preprocess.yaml` with the `data`, `masking`, `artifacts`
and `export` groups. `--config-name lung` selects TCGA and overrides the hole,
component and tiling settings above. `--config-name kpmp` replaces the data,
masking and artefact groups for special stains. Preview the effective configuration before processing:

```sh
uv run --locked python -m src.wsi_preprocessing.preprocess --config-name lung --cfg job --resolve
uv run --locked python -m src.wsi_preprocessing.preprocess --config-name kpmp --cfg job --resolve

# A custom slide, smaller overview, no coordinate or pixel exports.
uv run --locked python -m src.wsi_preprocessing.preprocess \
  'data.paths=[/path/to/slide.svs]' thumbnail.max_size=512 export.enabled=false

# Alternative single-mask recipes.
uv run --locked python -m src.wsi_preprocessing.preprocess masking=clam_saturation
uv run --locked python -m src.wsi_preprocessing.preprocess masking=tia_otsu
uv run --locked python -m src.wsi_preprocessing.preprocess masking=tia_morphological
uv run --locked python -m src.wsi_preprocessing.preprocess masking=codait_od
```

TIA morphology kernel and region sizes also require integer values. Selecting
`masking=tia_otsu` still applies the configured repository artefact cleanup; it is
not a standalone TIA baseline unless those exclusions are disabled. Likewise,
`clam_saturation` is a local recipe, distinct from upstream CLAM segmentation.

Slide-count, thumbnail-edge and pixel-budget bounds require integers; fractional
or boolean values are rejected rather than silently rounded.
Use either `data.manifest` or `data.paths`, not both. Relative input paths resolve
from the launch directory even with `hydra.job.chdir=true`; the default is false.
Each run requires a fresh output directory. `resolved_config.yaml` records task
settings; `.hydra/<timestamp>/` preserves each invocation's Hydra configuration
and overrides, including refused attempts to reuse an existing output directory.
Only manifests with `status: complete` describe completed runs. A run reserves
its directory with `run_started.json`; existing artifacts or symlink output
folders are refused. Hydra's own log and configuration directory are permitted.

New manifests identify `run_kind: preprocessing`, `schema_version: 1` and
`run_purpose: demo` (or `validation`). Use `run_purpose=validation` for execution
checks; the viewer hides those runs until “Show validation runs” is enabled.
It also recognizes older validation/verification/review directory names.
Set `DIGITAL_PATHOLOGY_OUTPUTS=/absolute/run/root` before starting Streamlit to
browse a different output tree, including Hydra runs outside `outputs/`.

Each manifest retains the input CSV's SHA-256, Git revision/dirty state and exact
preprocessing source hashes. Slides retain declared source checksum fields,
source size and modification time. `source_checksum_verified: false` is explicit:
this bounded run did not recompute a whole-WSI checksum. Unchanged size/mtime and
a thumbnail hash are not substitutes for a verified source-file checksum.
`patient_id` comes from a nonempty declared `case_submitter_id` or `case_id`
after trimming surrounding whitespace;
missing identities remain null. The viewer counts known dataset-scoped IDs and
never counts filenames or unknown values as patients.

A Hydra sweep creates separate numbered job directories. Bounds apply per job:

```sh
uv run --locked python -m src.wsi_preprocessing.preprocess -m data=tcga data.max_slides=1 \
  export.enabled=false thumbnail.max_size=128,256
```

If a restricted environment cannot write package caches, set `UV_CACHE_DIR`,
`MPLCONFIGDIR` and `NUMBA_CACHE_DIR` to writable temporary directories. Optional
local configs are ignored by Git; a `configs/local/default.yaml` intended to
change root settings needs Hydra's `# @package _global_` header.

## Inspect metadata and bounded previews

`src.wsi_preprocessing.inspection.read_slide_metadata(path)` uses OpenSlide to collect dimensions,
pyramid levels, downsample factors, recorded x/y MPP, objective power, vendor,
file size, bounds, associated-image names, ICC information and raw properties.
The notebook displays metadata before its first overview. No label/macro pixels
are read. Missing values stay missing; objective power does not supply guessed
MPP. Physical slide extent includes background. Images remain scanner RGB; this
demo does not apply ICC colour conversion.

The CLI default overview has a maximum edge of 1024 pixels. Reading is bounded to
16 million pixels per block and 100 million pixels per slide. If no whole pyramid
level fits, the lowest usable level is resized in blocks; exceeding the total
budget fails. Block boundaries can introduce interpolation differences.
Thumbnail masks lose small structures. Recorded independent x/y scale factors
map them to level-0 coordinates. PNG/JPEG examples lack known WSI calibration
and are excluded from calibrated coordinate export.

The notebook also uses a 1024-pixel overview edge. Its main-slide budget is
4 million pixels per block and per slide. The separate severe-artefact example
allows 16 million for both limits and requires native CLAM's selected pyramid
level to fit within 16 million pixels.

## Generic combined H&E masking

`src.wsi_preprocessing.pipeline.preprocess_masks(rgb, masking=..., artifacts=...)` takes ordinary
mappings and returns candidate, clean and excluded masks, per-reason exclusions,
intermediate images and applied-threshold metadata. Source RGB is unchanged.

1. Apply configured CODAIT pen-colour, dark and grey exclusions; replace excluded
   pixels with white in a working image.
2. Apply TIA grayscale Otsu, then whiten pixels outside its gate. A manual
   threshold can replace the fitted threshold.
3. Apply CLAM-style HSV saturation, median filtering, fixed/Otsu thresholding
   and closing. This threshold is distinct from TIA's grayscale threshold.
4. Fill enclosed 8-connected background components up to `max_hole_area`.
   Border-connected background and larger holes remain.
5. Smooth the binary candidate mask with one elliptical opening (erosion then
   dilation), followed by one closing (dilation then erosion). The default
   3-pixel kernel uses constant-zero outside-image borders.
6. Reapply CODAIT exclusions and remove small foreground components. Median
   filtering, closing, hole filling and smoothing can extend beyond the preliminary
   TIA gate; final cleanup prevents restoration of CODAIT-excluded pixels.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `masking.tia_enabled` | `true` | Preliminary TIA grayscale gate |
| `masking.tia_threshold` | `null` | Fit Otsu; a number overrides it |
| `masking.median_size` | `7` | Odd saturation kernel, 1–31 mask pixels |
| `masking.use_otsu` | `true` | Fit saturation Otsu |
| `masking.saturation_threshold` | `20` | Used when saturation Otsu is disabled |
| `masking.closing_size` | `3` | Closing kernel edge; 0 disables |
| `masking.max_hole_area` | `16` | Maximum enclosed area for the small-hole step; 0 disables |
| `masking.smoothing_kernel_size` | `3` | Elliptical opening then closing; 0 or 1 disables, otherwise odd 3–31 |
| `artifacts.min_tissue_area` | `64` | Minimum 8-connected tissue area; 0 disables |
| `artifacts.pen_colors` | `[red,green,blue]` | Enabled colour rules; `[]` disables |
| `artifacts.pen_thresholds` | `null` | CODAIT defaults or per-colour replacements |

Kernel and area units are thumbnail pixels. Changing thumbnail size changes
physical meaning. The hole limit is a raster-area policy, separate from upstream
CLAM's patch-scaled contour-area and maximum-hole-count policy. Closing can bridge
spaces independently of the hole limit. Air spaces may be real anatomy; these
masks do not establish anatomical boundaries.

The post-mask smoothing step is enabled in both the generic and lung presets. It
rounds small boundary irregularities using binary erosion/dilation: pixels remain
true or false, with no soft probabilities. It does not average neighbouring RGB
values, blur the slide or inpaint image content. This preserves the original
tissue appearance while changing which regions are selected. Source RGB stays
unchanged; the earlier thresholding stages use separate working copies with
constant-white exclusions. Opening can remove narrow tissue structures; closing can bridge gaps
or fill holes even when `max_hole_area=0`. The same kernel covers different
physical areas on different slide thumbnails. This is a scientific choice:
inspect the notebook's `02c_mask_smoothing.png` before/after comparison and set
`masking.smoothing_kernel_size=0` to disable smoothing when it damages genuine
anatomy. The earlier CLAM-stage `closing_size` remains a separate setting. Saved
`before_smoothing` and `smoothed_mask` stages plus added/removed pixel counts
record its effect.

```sh
uv run --locked python -m src.wsi_preprocessing.preprocess masking.max_hole_area=0 \
  masking.use_otsu=false masking.saturation_threshold=30 \
  hydra.run.dir=outputs/fixed-mask-demo

uv run --locked python -m src.wsi_preprocessing.preprocess \
  'data.paths=[data/tcga/TCGA-LUSC/5048c79f-ed1d-4534-8da2-1a4d207b2371.svs]' \
  'artifacts.pen_thresholds={blue:[[60,100,210]]}' \
  hydra.run.dir=outputs/custom-pen-demo
```

A rule `[r,g,b]` marks red when `R>r & G<g & B<b`, green when
`R<r & G>g & B>b`, and blue when `R<r & G<g & B>b`. Rules are OR'ed within each
colour. An override replaces that colour's entire rule list; omitted colours
retain defaults, and `[]` disables a colour. Values are integers in 0–255. These
are editable heuristics, not empirically validated artefact labels. Dark tissue,
special stains and small fragments can be wrongly excluded. Grey filtering is
off by default. The [KPMP profile](datasets.md#kpmp-ckd) disables H&E exclusions
and uses the separate `clam_saturation` mask recipe.

## Upstream CLAM and physical scale

`src.wsi_preprocessing.tiling.clam_coordinates` calls upstream `WholeSlideImage.process_contour`
at commit `53e2409d4a8189c682c173382964a85f114f923c`. It converts the final mask
to level-0 tissue/child-hole contours; it does not segment the WSI again. Each
contour anchors its own grid. The default `four_pt` accepts at least one of four
interior probes. `four_pt_hard`, `center` and `basic` are also available. Patch
centres strictly inside child holes are rejected when
`export.exclude_hole_centers=true` (the generic default). The lung preset sets it
to false while preserving all holes in mask coverage and visualizations. Boundary
padding is disabled. Stable duplicate removal and the retained-mask coverage
threshold (60% generic, 10% lung) are explicit additions. `export.tiling_method=coverage_grid` retains the older
slide-wide grid for reproducibility.

The default exports **all eligible 256-pixel coordinates at 0.5 µm/pixel**.
`export.coordinate_limit=null` disables coordinate subsampling. Bounds of 128
tissue contours and 200,000 candidate positions limit work; exceeding a bound
fails rather than silently dropping eligible coordinates. CLAM uses at most
four coordinate workers per contour. Pixel reads are sequential.

Calibration requires positive recorded isotropic MPP. For OpenSlide inputs,
`mpp-x/mpp-y` properties supply calibration; objective-power inference is not
accepted. Native scans more than 1% coarser than the target are rejected.
The source footprint is rounded from `output_size × target_mpp / native_mpp`.
For example, 256 output pixels at 0.252 native MPP require 508 source pixels,
giving 0.5000625 effective MPP after Pillow Lanczos resizing. Native, target and
effective MPP are recorded separately. The 0.5 MPP target is a conventional
20×-equivalent scale, not a claim about scanner objective power.

Actual patch PNGs require one exact `export.pixel_slide_id`. At most 128 are
saved by default, spread deterministically over that slide's coordinates. Other
slides save coordinates and thumbnail-derived coverage only. Pixel decoding is
bounded by `export.max_patch_read_pixels` (100 million per crop); 128 crops of
508×508 source pixels total about 33 million pixels. Full eligible-coordinate
export still omits edge strips, failed coverage tiles and tissue missed by the
mask. Eligible rectangular crops may contain background or artefacts.

Only 256-pixel output is enabled by default; large-region TAR export is off.
Explicit `[256,4096]` sizes require matching per-size limits. This preserves parts
of [HIPT's organization](https://github.com/mahmoodlab/HIPT#downloading--preprocessing--organizing-tcga-data),
but the default does not produce HIPT's large-region inputs or model features.

## Output contract and visual review

The Hydra CLI produces this completed-run layout for the Streamlit viewer:

```text
outputs/<run>/
  manifest.json, resolved_config.yaml, index.html
  .hydra/<invocation-timestamp>/
  results/<index>/                    # Thumbnail, masks, stages, metadata
  overviews/<dataset>/                # All small PNGs in one flat directory
  datasets/<dataset>/
    WSIs/<slide_id>.svs               # Relative symlink to the original
    dataset.csv                      # Fixed case/slide/label/path/stain/sample columns
    extracted_mpp0p5_patch256_fp/
      masks/<slide_id>.jpg            # Green tissue and blue hole contours
      tissue_masks/<slide_id>.png     # Lossless retained mask
      patches/<slide_id>.h5           # Level-0 coordinate bag
      stitches/<slide_id>.jpg         # Thumbnail-derived full coordinate coverage
      coverage_masks/<slide_id>.png   # Union of all exported footprints
      patch_images/<slide_id>/       # Selected WSI only: PNGs and index.json
      sampled_pixel_stitches/         # Selected WSI only: sampled source pixels
      process_list_autogen.csv
```

The notebook calls the shared exporter directly and saves a teaching layout:

```text
outputs/preprocessing_demo/<run_id>/
  provenance.json                    # Source, settings, calibration and export metadata
  eligible_coordinates.csv           # Every eligible coordinate
  sampled_tile_qc.csv, qc_tiles/      # Separate visual-QC sample (12 by default)
  thumbnail.png, tissue_mask.png, stage_*.png
  01_overview.png, 02_tissue_mask.png, 02c_mask_smoothing.png
  03_full_coordinate_coverage.png, 04_thumbnail_qc.png, ...
  severe_artifact_comparison/         # Original, three masks, comparison figures and provenance
  stain_normalisation/               # Optional matched stain comparisons
  datasets/<dataset>/                # Shared CLAM layout above; no exporter patch PNGs
```

It does not create a CLI `manifest.json`, Hydra metadata or an HTML report, so
this notebook folder is not a completed preprocessing run in Streamlit. Open its
figures directly; use the CLI for the dataset browser. Exported coordinate counts
vary with the chosen slide and settings.

HDF5 `coords` is an `N×2` integer array of level-0 top-left `(x,y)` locations.
Its CLAM `patch_size` attribute is the **native source footprint**;
`output_patch_size` records the resized image width. Other attributes include
`patch_level=0`, downsample, level dimensions, name, save path, calibration,
backend and upstream revision. Export schema 7 also records
`exclude_hole_centers`; older exports without the field retain the original true
policy. `tissue_fraction` stores estimated retained-mask
coverage; `preview_indices` maps saved images into the full bag. Empty bags are
explicit. The selected-slide `index.json` records each PNG's coordinate index.

CLAM's coordinate-bag reader reads the native footprint from `patch_size`.
Downstream transforms must resize to the recorded output size: use
`--target_patch_size 256` for the default bag in CLAM's feature-extraction CLI.
This project does not execute or validate downstream CLAM/HIPT models.

`dataset.csv` always has `case_id,slide_id,label,wsi_path,stain,sample_id` columns;
optional values are blank. Its case ID uses the declared submitter ID, then case
ID, and stays blank when neither exists. Unknown patient identity is not replaced
with a slide filename. Mixed rows with and without stain metadata keep the same
CSV schema.

The lossless `tissue_holes.png` changes only contour pixels: green tissue
exteriors and blue holes, without fills. `hole_mask.png` identifies background
not four-connected to the thumbnail boundary. It describes topology of the
final mask, after configured cleanup, with no additional visualization cutoff.
A blue hole contour is not an artefact label or an anatomical annotation.
CLAM `masks/` uses a JPEG of this preview; `overlay.png` separately shows artefact
exclusions. `stitches/` and `sampled_pixel_stitches/` represent different evidence:
complete coordinate footprints versus bounded source-pixel samples.

Flat galleries contain `__thumbnail.png`, `__mask.png`, `__preprocessed.png`,
`__overlay.png` and `__tissue_holes.png` for each WSI. Sort by name or filter a
suffix in Finder; use Command-4 for Gallery or Space for Quick Look. The Streamlit
viewer labels coordinate counts and saved-image counts separately, supports
metadata CSV downloads, and hides validation runs by default. Each run has one
HTML report rather than a report per WSI.

Refresh only these tissue/hole visualizations from saved thumbnails and masks:

```sh
uv run --locked python -m scripts.refresh_tissue_holes outputs/my-run
```

## Reusable modules and notebook

| Stage | Module and entry points |
| --- | --- |
| Inspect/read | `src.wsi_preprocessing.inspection`: `read_slide_metadata`, `read_thumbnail` |
| Tissue masks | `src.imaging_utils.masking`: `create_tissue_mask`, `saturation_mask`, `otsu_mask` |
| Artefact and tile QC | `src.imaging_utils.quality`: `exclude_artifacts`, `tile_quality` |
| Combined masking | `src.wsi_preprocessing.pipeline`: `preprocess_masks` |
| Calibrate/tile/read | `src.wsi_preprocessing.tiling`: `calibrated_patch_size`, `clam_coordinates`, `read_tile`, `contour_tile_mask` |
| Export/review | `src.wsi_preprocessing.export`, `src.wsi_preprocessing.report` |
| Shared arrays | `src.imaging_utils.arrays`: validation, component/hole operations, `smooth_mask_edges`, background preparation, PNG output |

Numerical helpers take explicit arrays and parameters, preserve their inputs,
and do not create output directories. RGB is `uint8 H×W×3`, masks are `bool H×W`,
locations/sizes use `(x,y)`/`(width,height)`, and arrays index `[y,x]`. Mask bounds
use floor for left/top and ceil for right/bottom. `read_tile` accepts a
caller-owned CLAM/OpenSlide handle, checks bounds and rejects padding/missing
pixels before resizing. The export adapter requires square/isotropic calibration;
the general calibration utility also supports rectangular regions.

Import the owning modules listed above. `create_tissue_mask` handles individual
masking methods; `preprocess_masks` composes the configured masking and exclusion
stages. `tiling.clam_coordinates` calls the pinned upstream CLAM implementation.
The unused independent CLAM implementation and the `tissue_mask` compatibility
helper have been removed; there are no aliases for the former root modules.

The notebook reads 12 real tiles by default, selected at evenly spaced indices
from the complete eligible coordinate grid. `N_QC_PATCHES` allows 4–36; fewer
are read if the grid is smaller. The QC figure marks their decisions on the
whole-slide thumbnail; it does not project coarse thumbnail masks over
high-resolution patch close-ups. It does not deliberately select ineligible
controls. Candidate and per-read pixel budgets are explicit. Sampled QC requires
both thumbnail and raw saturation coverage. Laplacian variance is descriptive, not a validated
focus threshold. The intentionally sampled retention rate does not estimate
whole-slide quality. Settings and source coordinates are saved for review.

## Optional stain normalization

Normalization is optional and is not called by the default WSI preprocessing
entrypoint. The notebook and other callers share these functions:

| Function | Responsibility |
| --- | --- |
| `src.imaging_utils.arrays.mean_background_color(rgb, background_mask)` | Mean RGB over an explicit background sample; empty samples fail |
| `src.imaging_utils.arrays.fill_background(rgb, tissue_mask, background_rgb)` | Copy RGB and replace non-tissue pixels with an explicit colour |
| `src.wsi_preprocessing.pipeline.preprocess_masks(...)` | Build patch masks with explicitly fixed thresholds and cleanup settings |
| `src.imaging_utils.arrays.smooth_mask_edges(...)` | Apply opening/closing to a binary mask; this notebook pads boolean edges before calling |
| `src.imaging_utils.normalization.prepare_stain_image(...)` | Validate pixel bounds and prepare original or constant-background RGB |
| `src.imaging_utils.normalization.fit_stain_normalizer(...)` | Fit a supported TIA method to a prepared reference |
| `src.imaging_utils.normalization.normalize_tile(...)` | Prepare source, transform and restore background when a mask is supplied |
| `src.imaging_utils.normalization.compare_stain_normalization(...)` | Matched original/constant-background comparisons with results and timings |

These functions accept arrays, masks and explicit parameters. They do not depend
on notebook globals, slide IDs, plotting, filesystem output or Hydra. All original
arrays are preserved. The comparison accepts source-image and source-mask mappings
with arbitrary matching IDs, validates them before fitting, uses the same
reference/source identities and seed in both paths, and restores the caller's
NumPy RNG state. It runs serially because TIA Vahadane lacks a per-instance random-state
parameter. `methods` and `max_pixels` are caller-configurable.

### Use the average background colour

The notebook builds its four normalization masks from the actual 256-pixel
reference/source patches. It reuses the grayscale and saturation thresholds fitted
on the slide thumbnail, with the same CODAIT colour exclusions. Median filtering
is fixed to 1; threshold closing, dedicated hole filling and minimum component
area are 0. This avoids applying thumbnail-scale area/kernel settings to patch
pixels or enlarging coarse thumbnail contours into angular cutouts.

`NORMALISATION_MASK_KERNEL_SIZE=3` applies one elliptical opening then closing
in **output-patch pixels**, independently of the slide-mask kernel. It pads the
boolean mask by four pixels using replicated edge values, then crops back and
reapplies local artefact exclusions. Only mask values are padded: source RGB is
not padded, blurred or inpainted. Edge replication avoids imposing a zero-valued
frame but does not recover missing image context. The final patch masks control
both reference/source preparation and background restoration; they do not change
the slide mask, eligible coordinates or the shared background estimate. Provenance
records the frozen thresholds, patch settings, kernel units, border policy,
per-patch added/removed pixels, exclusions and RGB/mask hashes.

Use reviewed boolean masks to define tissue. Estimate
the **mean RGB of the slide's original non-tissue background**, excluding known
artefact candidates. Round the mean to integer RGB and use that one colour in
both the reference and every source tile, before fitting/transformation and when
restoring non-tissue pixels afterwards. The demo estimates it from the original
thumbnail outside the final mask, excluding CODAIT-flagged pixels. It displays and
records both unrounded and applied RGB, sample count and the background-sampling
mask. It does not use pure white or different fills for each tile.

```python
import numpy as np
from src.imaging_utils.arrays import mean_background_color
from src.imaging_utils.normalization import compare_stain_normalization

# Inputs are caller-owned RGB arrays and boolean masks.
mean_rgb = mean_background_color(slide_rgb, background_sample_mask)
fill_rgb = np.rint(mean_rgb).astype(np.uint8)
comparison = compare_stain_normalization(
    reference_rgb, source_images, reference_mask=reference_mask,
    source_masks=source_masks, background_rgb=fill_rgb,
    methods=['reinhard', 'macenko', 'ruifrok', 'vahadane'], random_seed=0,
)
# comparison['images'][method]['unmasked' or 'constant_background'][sample_id]
```

The original-background path fits and transforms raw reference/source RGB with
no masking or restoration. The other path replaces background with the shared
mean colour. Every method uses identical reference and source tile identities.
The notebook shows input preparation and a separate matched comparison for each
method, always with one source patch per row. Two overview figures place the
original and all methods in columns: `05c_stain_methods_unmasked.png` and
`05c_stain_methods_constant_background.png`. Raw exports are unchanged; normalised PNG filenames/provenance distinguish
`unmasked` and `constant_background`.

Constant filling is **not** tissue-only statistical fitting. Reinhard's colour
moments and stain-concentration percentiles include background pixels. The fill
colour and background fraction can therefore still affect results. Patch-level
thresholds can lose pale tissue or retain artefacts; binary smoothing can change
thin structures and anatomical spaces. This is not validated anatomical segmentation;
unflagged stitching marks and misclassified tissue can bias the estimated mean.
Review the background sampling mask and high-resolution tile borders. For a
multi-slide study, explicitly choose/fix a common background convention with the
reference rather than silently changing it across slides.

The notebook selects its reference from retained tiles by coloured-pixel fraction
and compares a bounded set of other tiles from the same slide. This illustrates
mechanics, not between-site correction or improved model performance. Choose a
study's reference using training data only. The masking pipeline still uses its own white working images for
CODAIT/TIA gating; the estimated colour applies specifically to stain normalisation.

TIA's Ruifrok implementation uses fixed H&E stain vectors with concentration
matching; Macenko and Vahadane estimate stain directions.
[TIA documents Vahadane instability](https://tia-toolbox.readthedocs.io/en/latest/_autosummary/tiatoolbox.tools.stainextract.VahadaneExtractor.html)
with newer scikit-learn dictionary learning. Its output is
included for teaching and explicitly labelled exploratory, not recommended as a
validated default. The notebook resets NumPy's seed before each fit/transform
for matched random starts, reports per-method elapsed times and saves all four
method comparisons. Remove entries from `STAIN_METHODS` to shorten the demo.
No additional packages are required.

## Validation and review

Use a fresh directory and a bounded subset of real slides:

```sh
uv run --locked python -m src.wsi_preprocessing.preprocess --config-name lung data.max_slides=1 \
  export.pixel_slide_id=20c9afac-0d0c-4fcf-b3bf-a2beba3dd76b \
  'export.max_patches=[6]' hydra.run.dir=outputs/validation-combined-clam
uv run --locked python scripts/verify_run.py outputs/validation-combined-clam --check-regions
```

The checker verifies mask partitions, combined-stage order, exclusion preservation,
bounded small-hole filling, the recorded opening/closing step, coordinate
eligibility, coverage and physical calibration.
`--check-regions` compares saved source-pixel samples with fresh OpenSlide reads
and the declared resampling. The notebook retains input-integrity and stage guards needed for execution,
but has no separate engineering-verification section. Restart and run all cells,
then inspect metadata, masks, full-coordinate coverage, thumbnail QC and stain
figures. Use the CLI checker above for saved-run engineering verification.
These checks establish mechanics on the exercised inputs, not biological accuracy,
stain invariance, artefact-detection accuracy or diagnostic performance.

## Sources and licences

- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
  informs the source/config organization. The preprocessing entrypoint does not
  run a Lightning trainer. `uv sync --locked --extra train` enables the separate
  [training workflows](training.md).
- [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox) supplies bounded
  overview reading, Otsu/morphological masking and optional stain normalization.
- [Pinned upstream CLAM](https://github.com/mahmoodlab/CLAM/blob/53e2409d4a8189c682c173382964a85f114f923c/wsi_core/WholeSlideImage.py)
  supplies contour tiling and the OpenSlide handle. The vendored subset, GPLv3
  licence and file hashes are under `third_party/CLAM/`.
- [CODAIT deep-histopath](https://github.com/CODAIT/deep-histopath) supplies the
  attributed pen/grey filters and optical-density convention. See the
  [source revisions, changes and preserved notices](../third_party/README.md).
- [histolab](https://github.com/histolab/histolab) is a composable-masking reference.
  It is not installed in this locked environment; the attributed CODAIT filters
  provide the colour-exclusion implementation.
- [OpenSlide API](https://openslide.org/api/python/),
  [TIA 2.1.3 normalization](https://github.com/TissueImageAnalytics/tiatoolbox/blob/v2.1.3/tiatoolbox/tools/stainnorm.py)
  and [stain extraction](https://github.com/TissueImageAnalytics/tiatoolbox/blob/v2.1.3/tiatoolbox/tools/stainextract.py)
  document the reader and optional normalization implementations.

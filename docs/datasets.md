# Demo datasets and QuPath

[Quick start](../README.md) · [Preprocessing](preprocessing.md) ·
[Annotation conversion](annotation-conversion.md)

Run the examples below from the repository root after `uv sync --locked`.
Downloaders resolve the project root from their script location, so invoking an
absolute script path elsewhere still reuses the same manifests and `data/` tree.
TCGA/KPMP originals remain under that canonical directory. QuPath projects
reference those files; preprocessing layouts use symlinks. Neither workflow makes
another WSI copy. Large data, download receipts and generated outputs stay out of
Git. Keep manifests for reproducible selection and choose new run directories
when repeating preprocessing.

## Small reader and pen examples

```sh
uv run --locked python scripts/download_examples.py
uv run --locked python -m src.preprocess hydra.run.dir=outputs/small-demo
uv run --locked python -m src.preprocess data=pen_examples hydra.run.dir=outputs/pen-demo
```

The first command downloads the small public Aperio reader example and real
CODAIT pen-mark images. Existing and downloaded examples are read with a 10 MB
bound and must match the pinned SHA-256; new files are created exclusively.
Raster examples support mask review but do not carry known WSI physical
calibration, so calibrated coordinate export skips them.

## TCGA lung

[The frozen roster](../manifests/tcga_demo.csv) contains five TCGA-LUAD and five
TCGA-LUSC open-access diagnostic primary-tumour slides from ten distinct patients,
with five tissue-source sites per class. Selection spans file-size ranges within
an approximately 3 GB cohort. File size and site are selection proxies, not proof
of histological diversity; site can also confound classification.

```sh
uv run --locked python scripts/download_tcga.py
uv run --locked python -m src.preprocess data=tcga \
  export.pixel_slide_id=20c9afac-0d0c-4fcf-b3bf-a2beba3dd76b \
  hydra.run.dir=outputs/tcga-mpp05
uv run --locked python scripts/verify_run.py outputs/tcga-mpp05 --check-regions
```

The selected stem names TCGA-55-6972. It saves at most 128 actual 256-pixel PNGs;
other slides retain complete eligible coordinate bags and small coverage previews.
Omit `export.pixel_slide_id` to disable actual patch images for all slides.
The default H&E workflow is the combined CODAIT/TIA/CLAM mask pipeline.

The downloader resumes partial files and verifies GDC file size and MD5 before
promoting each download to `.svs`. Defaults permit four concurrent transfers,
a 10 GB total download budget, and require disk headroom. Budget overrides must
be finite and positive. It reuses the manifest when present; if absent, it queries GDC to select a new roster. Preserve the
committed manifest for exact replay. Originals use UUID filenames under
`data/tcga/<project>/`; the manifest retains original filenames, case IDs and labels.

The project-level LUAD/LUSC labels are not pixel-level tumour annotations.
This small convenience cohort supports workflow demonstrations, not a credible
classification benchmark. Keep every slide and patch from a patient in one
split. No WSI classifier is trained by these commands.

## KPMP CKD

[The frozen KPMP roster](../manifests/kpmp_ckd_demo.csv) contains five CKD-enrolled
participants, each with H&E, PAS, trichrome (`TRI`) and silver (`SIL`) WSIs.
Enrollment category is distinct from primary adjudicated disease category.
Participant matching does not establish a shared biopsy, registered serial
sections or spatial correspondence. Filename-derived sample labels are preserved
as slide labels only.

```sh
uv run --locked python scripts/download_kpmp.py
uv run --locked python -m src.preprocess --config-name kpmp \
  hydra.run.dir=outputs/kpmp-ckd-demo
uv run --locked python scripts/verify_run.py outputs/kpmp-ckd-demo
```

The profile explicitly uses the single `clam_saturation` recipe and disables H&E
pen-colour, dark, grey and small-component exclusions. Special-stain colours and
silver deposits must not be treated automatically as H&E artefacts. The previews
show estimated tissue/background separation, not validated artefact removal.
Upstream CLAM exports eligible 256-pixel coordinates at the shared 0.5 MPP target;
actual patch images remain disabled for this profile.

Downloads default to two concurrent transfers, a 5 GB total budget and a 1 GB
per-slide bound. Existing files must match recorded size and SHA-256 before reuse.
New acquisitions also receive a bounded OpenSlide readability check. KPMP public
metadata does not expose an upstream checksum: committed/local SHA-256 values
protect replay, but a first local hash is not an independent upstream integrity
check. If the server ignores a resume request, the same partial file restarts.
Use `scripts/download_kpmp.py --verify-only` after acquisition to check existing
canonical files. Originals live under `data/kpmp/CKD/<participant>/`.

The manifest preserves public participant/file/package IDs, source URLs, sizes,
stains, enrollment/adjudication fields and download hashes. Keep all stains from
a participant in the same data split. This is a browsing/preprocessing cohort,
not a disease-classification benchmark.

Data are provided by the [Kidney Precision Medicine Project](https://www.kpmp.org/).
Follow its [citation guidance](https://www.kpmp.org/help-docs/citing-kpmp-data)
and [enrollment/adjudication definitions](https://www.kpmp.org/help-docs/adjudication-vs-enrollment)
when interpreting or sharing the data.

## Patch instance-segmentation datasets

CytoNuke and NuInsSeg contain H&E image patches with object annotations. They
support nucleus/cell instance-segmentation and annotation-conversion workflows,
rather than the WSI-level classification examples above. Their coordinates belong
to individual patch images; they are not registered annotations for the TCGA or
KPMP WSIs in this project.

Download the versioned releases below, compare archive checksums with the release
records, and retain source archives, metadata and attribution under each dataset's
`source/` directory. Extract into `extracted/`; these datasets have no project
Hydra preprocessing profile or dedicated downloader. Existing local `index.csv`
files pair inputs using paths relative to their dataset directory, while
`validation.json` records preparation checks. Generated conversions belong under
`outputs/annotations/`, leaving originals unchanged. See the
[annotation-conversion guide](annotation-conversion.md) for bounded commands,
GeoJSON/instance-TIFF exports and explicit overlap policies.

### CytoNuke

Acquire [CytoNuke v2](https://zenodo.org/records/15310323) and cite its dataset DOI
and [Raufeisen et al. paper](https://doi.org/10.1016/j.cmpb.2024.108215).
The release is CC BY 4.0 and contains 83 H&E PNG patches; the local index and
COCO dimensions are 256 × 256 pixels. Preserve the release's CPTAC-HNSCC image
attribution as well as the CytoNuke citation.

- Images: `data/cytonuke/extracted/images/`.
- Original polygon annotations: `data/cytonuke/extracted/coco.json`.
- Image/annotation index and source records: `data/cytonuke/index.csv` and
  `data/cytonuke/source/`.

The COCO classes are **0 = NUCLEUS** and **1 = CELL**; category 0 is an object
class, not background. Nucleus and whole-cell polygons overlap, and some
annotations overlap within a class. Preserve polygons when overlap must remain
explicit. Raster conversion writes separate category masks and requires a
chosen policy for within-category overlap; a single flattened mask cannot retain
all original overlapping instances.

### NuInsSeg

Acquire [NuInsSeg 1.0](https://zenodo.org/records/10518968) and cite its dataset DOI
and [Mahbod et al. paper](https://doi.org/10.1038/s41597-024-03117-2).
The saved official release metadata specifies CC BY 4.0. The local index contains
665 H&E patches, each 512 × 512 pixels, from human and mouse tissue. This is a
nuclei instance-segmentation dataset, not a whole-cell boundary dataset.

Within `data/nuinsseg/extracted/<species tissue>/`, the paired inputs are:

- `tissue images/`: original PNG images.
- `label masks modify/`: recommended integer instance-label TIFFs.
- `vague areas/mask binary/`: separate ambiguous-area masks.

`data/nuinsseg/index.csv` pairs these files and retains the raw `label masks/`
paths. Use **`label masks modify`** for evaluation, following the
[authors' mask guidance](https://github.com/masih4/NuInsSeg/blob/554543db47faedc21d2fe6cea78827bfada6dec6/README.md#codes-to-generate-segmentation-masks).
Raw label masks introduce additional numerical values where nuclei overlap;
those values are not extra nuclei. In the modified masks, positive integer IDs
identify individual nuclei and zero is background. Preserve integer IDs when
loading or converting masks.

Nonzero ambiguous-area pixels mark uncertain regions. Keep those masks paired
with the images and state their treatment explicitly in evaluation; the converter
does not automatically turn them into background or apply an ignore policy.
Patch filenames do not establish independent patients, and this preparation
creates no new train/test split or claim of biological annotation accuracy.

## Review small overviews

Start `uv run --locked streamlit run app.py` and choose a completed run. For
Finder Gallery or Quick Look, use the flat dataset directories:

- `outputs/tcga-mpp05/overviews/tcga_luad/`
- `outputs/tcga-mpp05/overviews/tcga_lusc/`
- `outputs/kpmp-ckd-demo/overviews/kpmp_ckd/`

Sort by name to group each WSI's thumbnail, retained mask, preprocessed image,
artefact overlay and tissue/hole contour view. KPMP names include participant and
stain. Filter filenames by `__preprocessed`, `__mask` or `__tissue_holes` to browse
one view type. These are small previews; inspect the originals for fine detail.

## QuPath

Install the matching platform build from [official QuPath releases](https://github.com/qupath/qupath/releases).
On macOS, choose Apple Silicon for arm64 or Intel for x64. You can open a canonical
`.svs` directly or create a reusable project with the installed command-line
executable. Replace `/path/to/QuPath-executable` below with that executable.

```sh
# Ten TCGA images.
uv run --locked python scripts/create_qupath_project.py \
  --qupath /path/to/QuPath-executable

# Twenty KPMP images from five participants.
uv run --locked python scripts/create_qupath_project.py \
  --qupath /path/to/QuPath-executable \
  --manifest manifests/kpmp_ckd_demo.csv --project-name kpmp-ckd-demo \
  --expected-patients 5 --expected-slides 20
```

Open `data/qupath/tcga-lung-demo/project.qpproj` or
`data/qupath/kpmp-ckd-demo/project.qpproj` in QuPath. The helper creates thumbnails,
records roster metadata and names images by label, participant and stain. H&E
uses the H&E image type; PAS/TRI/SIL use **Brightfield (other)** without an assumed
H&E deconvolution model. Existing projects are preserved rather than overwritten.

Projects reference canonical absolute WSI paths without copying image data.
After moving the repository, repair image URIs in QuPath or create a new project;
do not duplicate slides to satisfy old locations. Review masks and exclusions
against the original slide at higher magnification before using them downstream.

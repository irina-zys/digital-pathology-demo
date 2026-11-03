# Demo datasets and QuPath

[Quick start](../README.md) · [Preprocessing](preprocessing.md) ·
[Annotation conversion](../scripts/convert_annotations.md)

Run commands from the repository root after `uv sync --locked`. Downloaders use
the frozen public manifests and keep originals under `data/`. QuPath projects
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
CODAIT pen-mark images. Raster examples support mask review but do not carry
known WSI physical calibration, so calibrated coordinate export skips them.

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
a 10 GB total download budget, and require disk headroom. It reuses the manifest
when present; if absent, it queries GDC to select a new roster. Preserve the
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

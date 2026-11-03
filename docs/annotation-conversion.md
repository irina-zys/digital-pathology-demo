# Annotation conversion

For a bounded prediction/error walkthrough with a conversion appendix, open the
[annotation and segmentation demo](../notebooks/annotation_segmentation_demo.ipynb)
and follow its [model setup](pretrained-models.md).

`convert_annotations.py` converts either COCO polygon annotations or 2D integer instance-label images into:

- pixel-coordinate GeoJSON FeatureCollections, one feature per instance, retaining holes and disconnected parts;
- lossless uint32 instance TIFFs (0 = background), with JSON tables mapping positive pixel IDs to source annotation IDs and classes;
- or both, which is the default.

Use the project's existing `.venv/bin/python`; NumPy, Pillow, Shapely and tifffile are already available in that environment. No dependencies were added. Run the following commands from the repository root.

## Obtain the inputs

Dataset files are not included in Git. Download the public releases from
[CytoNuke v2 on Zenodo](https://zenodo.org/records/15310323) and
[NuInsSeg 1.0 on Zenodo](https://zenodo.org/records/10518968), retaining their licence
and attribution files. Compare downloaded archive checksums with those published
on the release pages before extraction. The examples below assume
`data/cytonuke/extracted/coco.json` and the original NuInsSeg tissue directories
under `data/nuinsseg/extracted/`; adjust the input paths for your local extraction.

## Python components

The reusable implementation lives in `src/data/annotations.py`; `scripts/convert_annotations.py` only parses command-line arguments and calls it. Application code and notebooks can import the same workflow directly, without importing a CLI or changing `sys.argv`:

```python
from src.data.annotations import convert_annotations

manifest = convert_annotations(
    "data/nuinsseg/extracted",
    "outputs/annotations/nuinsseg_api",
    input_format="mask",
    pattern="*/label masks modify/*.tif",
    output_format="both",
    max_images=3,
)
assert manifest["status"] == "complete"
```

`convert_annotations` validates options and paths, writes artifacts and returns the manifest. It uses standard Python logging for progress and raises exceptions on failure. Direct API calls retain the CLI's new-output-directory, pixel-budget and explicit-overlap safeguards; options are validated before output creation. The CLI reports invalid options/input as argparse errors.

The same module owns the reusable readers (`read_mask`, `coco_samples`, `mask_samples`), pixel-edge geometry conversion (`mask_objects`, `rasterize`) and per-image export (`export_sample`). Their existing coordinate, ID, category and overlap semantics are unchanged. Use the top-level conversion function for a complete validated run with a manifest.

## CytoNuke: COCO polygons

```bash
.venv/bin/python scripts/convert_annotations.py \
  data/cytonuke/extracted/coco.json \
  outputs/annotations/cytonuke \
  --input-format coco --format both --overlap last
```

COCO nuclei and whole cells overlap by design. The converter writes a separate instance TIFF for **each category**, including category 0 (NUCLEUS); category IDs are never confused with the background value. Foreground values are positive per-image instance IDs, with original COCO annotation IDs retained in `.labels.json` and GeoJSON properties. GeoJSON contains all categories and preserves overlaps.

There are also within-category overlaps in these real annotations. The safe default `--overlap error` refuses to flatten them. The example explicitly selects `last`, meaning later annotations in source order win contested pixels; `first` is also available. Neither policy establishes the correct biological boundary. The manifest records overlap-pixel counts, clipped objects, and objects that have no pixels after rasterization. Choose GeoJSON-only output (`--format geojson`) when retaining overlaps is necessary.

## NuInsSeg: existing instance masks

```bash
.venv/bin/python scripts/convert_annotations.py \
  data/nuinsseg/extracted \
  outputs/annotations/nuinsseg \
  --input-format mask --glob '*/label masks modify/*.tif' --format both
```

Use **`label masks modify`** for NuInsSeg evaluation, as [the authors specify](https://github.com/masih4/NuInsSeg/blob/554543db47faedc21d2fe6cea78827bfada6dec6/README.md#codes-to-generate-segmentation-masks). The raw `label masks` produce additional numerical values at overlaps, which are not separate nuclei. Keep each recommended mask paired with its source H&E image and ambiguous-area mask. The converter does not merge ambiguous regions into foreground/background or apply an ignore policy; retain those source masks separately for evaluation.

The mask input can be an individual TIFF/PNG or a directory selected with `--glob`. Existing positive IDs are preserved, including gaps. This is **instance** input: a binary foreground mask is interpreted as one instance, not automatically split into cells. Semantic class images, RGB masks, TIFF stacks, and COCO RLE are not supported. XML annotations, ImageJ ROI archives and other proprietary formats are not parsed and need an upstream reader.

## Coordinates, limits and output checks

Coordinates are local image pixels: x = column, y = row increasing downward, origin at the upper-left image edge. A raster pixel occupies `[x,x+1] × [y,y+1]`. Rasterization uses pixel-centre intersection and clips to image dimensions. This convention is explicit and may differ from other COCO renderers; no equivalence to pycocotools rasterization is claimed.

These are image-coordinate GeoJSON files, not geographic longitude/latitude data. No micron scale, slide origin, CRS or whole-slide registration is invented. COCO polygons remain at original resolution; mask polygons trace exact pixel edges without contour simplification. Mask-to-TIFF conversion preserves pixel IDs exactly.

Use `--max-images 3` for a bounded real-data execution. The default bound is 16,777,216 pixels per image, adjustable with `--max-pixels`. Processing is sequential on CPU. Large whole-slide masks should be tiled upstream. An output directory must be new, must not be a symlink, and must be outside a mask-input tree. Mask globs must be relative and cannot traverse parent directories. Source files and previous runs are never overwritten; TIFF creation is exclusive. The COCO JSON document is parsed in full before the image limit is applied, so `--max-images` does not bound JSON parsing memory.

Each output directory includes `manifest.json` with status, source, settings, package versions, source image paths, dimensions, counts and output paths. Treat only `status: complete` runs as usable; failed runs may contain partial files. TIFFs are read back and compared with the in-memory labels during conversion.

## Real-input validation

For a bounded check, add `--max-images 3` and write to a fresh output directory.
Confirm `manifest.json` reports `complete`; compare TIFF arrays and positive IDs
with source masks, inspect GeoJSON pixel-edge geometry and reconstruct source
labels from it. For COCO, verify the label tables retain source annotation IDs and
category 0, and review recorded overlaps before choosing `first` or `last`.
Run the same real overlapping input with the default `error` policy to confirm
rejection. Keep commands, source hashes, counts and validation results in ignored
run receipts rather than treating historical checks as guarantees for new data.

These checks establish conversion mechanics on the exercised inputs, not
biological annotation accuracy, QuPath import compatibility or segmentation-model
performance. No model is trained, and no benchmark split is changed.

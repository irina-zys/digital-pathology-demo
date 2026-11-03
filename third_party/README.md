# Source attribution

`src/imaging_utils/filters.py` adapts code from CODAIT/deep-histopath revision
`c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33` (Apache-2.0):

- [wsi/filter.py](https://github.com/CODAIT/deep-histopath/blob/c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33/deephistopath/wsi/filter.py):
  `filter_red`, `filter_green`, `filter_blue`, their pen-filter threshold sets,
  and `filter_grays`.
- [preprocessing.py](https://github.com/CODAIT/deep-histopath/blob/c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33/deephistopath/preprocessing.py):
  `optical_density`.

Changes: bool KEEP masks only, compact threshold tables, uint8 RGB validation,
replacement of obsolete `np.int` with `int16`, configurable optical-density
background, and removal of logging/Spark/slide-processing dependencies.
Default threshold values and strict inequalities are preserved. Per-colour
replacement rule lists are configurable and validated as integer RGB triples;
omitted colours keep the published defaults, and an empty list disables a colour. Optical density retains
`-ln((I+1)/240)` by default, including negative values for very bright pixels.
`codait_od` is only its OD threshold component; it does not reproduce `keep_tile`'s
edge/morphology plus OD checks. No stain-normalization function was extracted.

`CODAIT-LICENSE.txt` and `CODAIT-NOTICE.txt` are preserved from that revision.
Example pen images are fetched from that repository's documentation into ignored
`data/demo/`; the example downloader records fixed SHA-256 hashes.

Other reference revisions inspected (concepts only, source not vendored):

- lightning-hydra-template: `bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144`.
- histolab: `d189008b6da9b959d0a532d68479f3d86d4925a4`.
- HIPT: `780fafaed2e5b112bc1ed6e78852af1fe6714342` (directory layout and region archive conventions).

TIAToolbox is installed as a dependency, pinned to 2.1.3 with the full resolution
in `uv.lock`. Its own distribution retains its licensing information.

## Vendored upstream CLAM tiler

`CLAM/` contains the approved source subset from
[mahmoodlab/CLAM revision 53e2409d4a8189c682c173382964a85f114f923c](https://github.com/mahmoodlab/CLAM/tree/53e2409d4a8189c682c173382964a85f114f923c):

- `wsi_core/WholeSlideImage.py`, `wsi_core/wsi_utils.py`, `wsi_core/util_classes.py`;
- `utils/file_utils.py` and the two empty package initialisers;
- the original `LICENSE.md` (GPLv3), plus our `SOURCE.json` with revision and SHA-256 hashes.

The upstream files are unmodified. No model/training code or weights are included.
`src/wsi_preprocessing/tiling.py` calls upstream `WholeSlideImage.process_contour` and reads pixels
through its OpenSlide handle. The adapter supplies contours from the reviewed
final mask, bounds work, disables padding, deduplicates coordinates and applies
extra mask-coverage QC. The default export does not call upstream segmentation:
the combined CODAIT/TIA/CLAM-style raster pipeline and bounded small-hole policy
are local code. The notebook separately calls native CLAM segmentation for its
explicit artefact comparison. The unused independent CLAM tiler was removed.
Existing installed packages satisfy the vendored subset's imports; no dependencies
were added.

TIAToolbox supplies the normalisation implementations; `src/imaging_utils/normalization.py`
offers unmasked inputs or constant backgrounds defined by caller-supplied boolean
masks for Reinhard, Macenko, Ruifrok and Vahadane, restoring the supplied background
colour after transformation only for the masked path.
It does not alter TIA's statistical estimators. CODAIT normalisation code was not
copied. Keep the GPLv3 licence and source notices when redistributing this subset.

## HoVer-NeXt pretrained patch adapter

`HoVerNeXt/` retains the GPLv3 licence and selected source from
[digitalpathologybern/hover_next_inference](https://github.com/digitalpathologybern/hover_next_inference/tree/bde3c9d96172d16513e292d0a2dfcd7ff80202ff).
`SOURCE.json` records upstream/local hashes and the narrow adaptation: selected
array postprocessing functions use NumPy storage in place of a Zarr array.
Architecture and class constants are unchanged. The adapter in `src/models/pretrained/hovernext.py`
loads the verified PanNuke Tiny fold-1 checkpoint and official thresholds. Its
16-view test-time augmentation follows the release's `params.toml` (`tta=16`,
`seed=42`) and pinned upstream
[inference](https://github.com/digitalpathologybern/hover_next_inference/blob/bde3c9d96172d16513e292d0a2dfcd7ff80202ff/src/inference.py#L207-L267),
[colour augmentation](https://github.com/digitalpathologybern/hover_next_inference/blob/bde3c9d96172d16513e292d0a2dfcd7ff80202ff/src/augmentations.py)
and [spatial augmentation](https://github.com/digitalpathologybern/hover_next_inference/blob/bde3c9d96172d16513e292d0a2dfcd7ff80202ff/src/spatial_augmenter.py):
flips/quarter-turns, HED scale 0.05/bias 0.03 perturbations, bilinear forward and
nearest-neighbour inverse-logit sampling, per-view softmax and probability
averaging. The local implementation uses float32 and processes one view at a time
without adding dependencies. Preserve the GPLv3 attribution for this adaptation.
It still handles one direct 256-pixel patch, without WSI tiling/stitching or
verified physical-scale correction, and does not reproduce five evaluation rounds
or the full published benchmark.

Weights are downloaded separately into ignored `data/models/`, under the
[release's CC-BY-NC-SA-4.0 licence](https://zenodo.org/records/10635618).
See [pretrained model setup](../docs/pretrained-models.md) for the separate UNI
and Cellpose source/access notices. No model weights are redistributed in Git.

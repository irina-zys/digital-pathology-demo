# Source attribution

`src/filters.py` adapts code from CODAIT/deep-histopath revision
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
`src/tiling.py` calls upstream `WholeSlideImage.process_contour` and reads pixels
through its OpenSlide handle. The adapter supplies contours from the reviewed
final mask, bounds work, disables padding, deduplicates coordinates and applies
extra mask-coverage QC. Upstream segmentation is not called: the combined
CODAIT/TIA/CLAM-style raster pipeline and bounded small-hole policy are local code.
The older independent `src/clam_tiling.py` helper remains available to existing
callers but is not the default backend. Existing installed packages satisfy the
vendored subset; no Python package dependencies were added.

TIAToolbox supplies the normalisation implementations; `src/normalization.py`
offers unmasked or contour-based constant backgrounds for Reinhard, Macenko, Ruifrok
and Vahadane, restoring the supplied background colour after transformation only for the masked path.
It does not alter TIA's statistical estimators. CODAIT normalisation code was not
copied. Keep the GPLv3 licence and source notices when redistributing this subset.

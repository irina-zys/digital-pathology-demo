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
Threshold values and strict inequalities are preserved. Optical density retains
`-ln((I+1)/240)` by default, including negative values for very bright pixels.
`codait_od` is only its OD threshold component; it does not reproduce `keep_tile`'s
edge/morphology plus OD checks. No stain-normalization function was extracted.

`CODAIT-LICENSE.txt` and `CODAIT-NOTICE.txt` are preserved from that revision.
Example pen images are fetched from that repository's documentation into ignored
`data/demo/`; the example downloader records fixed SHA-256 hashes.

Other reference revisions inspected (concepts only, source not vendored):

- lightning-hydra-template: `bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144`.
- histolab: `d189008b6da9b959d0a532d68479f3d86d4925a4`.
- CLAM: `53e2409d4a8189c682c173382964a85f114f923c`.
- HIPT: `780fafaed2e5b112bc1ed6e78852af1fe6714342` (directory layout and region archive conventions).

TIAToolbox is installed as a dependency, pinned to 2.1.3 with the full resolution
in `uv.lock`. Its own distribution retains its licensing information.

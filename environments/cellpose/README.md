# Classical Cellpose nuclei inference

This separate uv environment keeps Cellpose 3.1.1.2's NumPy 2.0 requirement and
headless OpenCV apart from the main preprocessing environment. It uses the
classical `nuclei` ResUNet checkpoint, not Cellpose-SAM. Python is pinned to the
3.12 series; all resolved dependencies are recorded in this directory's lockfile.

From the repository root:

```sh
uv sync --locked --project environments/cellpose --python 3.12
uv run --locked --extra pretrained python scripts/download_pretrained.py --model cellpose
```

The worker always runs on CPU with one PyTorch compute thread. Its CLI also sets
one inter-op thread. PyTorch uses the CPU wheel index on Linux/Windows and the
standard macOS wheel on macOS; the macOS worker does not select MPS.

The downloader prepares `nucleitorch_0` and its matching official
`size_nucleitorch_0.npy` under `data/models/cellpose/`. The size asset is needed
for automatic diameter estimation and is found beside the segmentation checkpoint.
Both are checked against pinned names, sizes and SHA-256 digests before loading.
The [segmentation checkpoint](https://www.cellpose.org/models/nucleitorch_0)
is 26,563,614 bytes; the [size model](https://www.cellpose.org/models/size_nucleitorch_0.npy)
is 3,627 bytes.
The locally verified segmentation checkpoint has SHA-256
`89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303`.
This digest identifies the downloaded asset; it is not a publisher-supplied
checksum. Preserve its source URL and download receipt alongside the checkpoint.
Weights and the environment’s `.venv/` directory are excluded from Git.

```sh
environments/cellpose/.venv/bin/python -m src.models.pretrained.cellpose_worker \
  data/cytonuke/extracted/images/C3L-00977-22.svsx29y31.png \
  outputs/cellpose-example \
  --checkpoint data/models/cellpose/nucleitorch_0 \
  --checkpoint-sha256 89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303 \
  --diameter 0 --channels 0 0 --invert
```

The input must be an RGB PNG at most 256 × 256 pixels. `--max-pixels` can lower
the 65,536-pixel limit. `--channels` selects grayscale/red/green/blue as 0/1/2/3,
followed by zero for the unused second channel. `--invert` reverses intensities;
normalization uses the first/99th percentiles. The H&E demo uses inverted grayscale
so its dark nuclei become bright in the signal passed to the model.

`--diameter 0` requests the official nuclei `SizeModel`. It estimates size from
image features and refines it using predicted objects from the **same normalized,
inverted signal** used for segmentation. No annotations enter either stage.
A positive `--diameter` supplies a manual value in input pixels; the generic default
remains 17 pixels for explicit/manual use. The Python API also accepts `None` for
estimation. See the [official v3 settings](https://cellpose.readthedocs.io/en/v3.1.1.1/settings.html#diameter)
and [SizeModel/API reference](https://cellpose.readthedocs.io/en/v3.1.1.1/api.html).
Final segmentation uses `niter=None` for the pinned library's diameter-dependent
dynamics default, instead of fixing the iteration count at 200.

The `diameter_estimation` metadata records the method, selected and style diameters,
refinement object count and `fallback_to_training_mean`. A refinement with no
objects saves calibration/failure metadata and fails visibly: the training mean
of 17 pixels must not be reported as a successful image-based estimate. A user
may provide a separately justified manual diameter; do not choose it by maximizing
agreement with the two demo annotations.

Choose a new output directory: the worker writes lossless `masks.tif` (`uint32`,
zero background, positive instance IDs) and `metadata.json`. It checks local
checkpoint size/digest before loading, bounds source pixels, verifies mask
read-back, and records input/checkpoint/output hashes and actual package versions.
It performs no hidden model downloads, restoration or training. Requested size
estimation uses only the verified local assets and input image.

`src.models.pretrained.cellpose_worker.load_cellpose(checkpoint, checkpoint_sha256)` is available
inside this environment; `run_cellpose(...)` owns the bounded image/file workflow.
The main project's adapter invokes the worker using this environment's Python.
A small successful run verifies inference mechanics, not annotation accuracy.

The primary demo uses inverted grayscale and image-estimated diameter. Its polarity
negative control switches inversion off while holding the calibrated diameter
fixed. Historical noninverted/d17 outputs remain available as failure examples.
Both methods use identical crops, ambiguity exclusion and IoU matching rules.
This correction was motivated by post-hoc inspection of two annotated patches;
freeze the recipe before testing additional held-out data. These examples do not
rank model families or establish generalization, and training overlap is unknown.
Manual diameter choices and estimator fallbacks must stay visible in the records.

The [code licence](https://github.com/MouseLand/cellpose/blob/v3.1.1.2/LICENSE)
is BSD-3-Clause. The [release notice](https://github.com/MouseLand/cellpose/blob/v3.1.1.2/README.md)
separately states that nuclei-model training includes CC-BY-NC data. Retain those
notices and cite Cellpose when using its outputs in research.

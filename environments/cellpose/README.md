# Classical Cellpose nuclei inference

This separate uv environment keeps Cellpose 3.1.1.2's NumPy 2.0 requirement and
headless OpenCV apart from the main preprocessing environment. It uses the
classical `nuclei` ResUNet checkpoint, not Cellpose-SAM. Python is pinned to the
3.12 series; all resolved dependencies are recorded in this directory's lockfile.

From the repository root:

```sh
uv sync --locked --project environments/cellpose --python 3.12
```

The worker always runs on CPU with one PyTorch compute thread. Its CLI also sets
one inter-op thread. PyTorch uses the CPU wheel index on Linux/Windows and the
standard macOS wheel on macOS; the macOS worker does not select MPS.

Obtain `nucleitorch_0` from the
[official Cellpose model host](https://www.cellpose.org/models/nucleitorch_0),
keeping one copy in `data/models/cellpose/`. Its expected size is 26,563,614 bytes.
The locally verified release used here has SHA-256
`89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303`.
This digest identifies the downloaded asset; it is not a publisher-supplied
checksum. Preserve its source URL and download receipt alongside the checkpoint.
Weights and the environment’s `.venv/` directory are excluded from Git.

```sh
environments/cellpose/.venv/bin/python -m src.cellpose_worker \
  data/cytonuke/extracted/images/C3L-00977-22.svsx29y31.png \
  outputs/cellpose-example \
  --checkpoint data/models/cellpose/nucleitorch_0 \
  --checkpoint-sha256 89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303 \
  --diameter 17 --channels 0 0
```

The input must be an RGB PNG at most 256 × 256 pixels. `--max-pixels` can lower
the 65,536-pixel limit. `--diameter` is an assumed nucleus diameter in input
pixels; `--channels` selects grayscale/red/green/blue as 0/1/2/3, followed by
zero for the unused second channel. `--invert` explicitly reverses intensities.
Normalization uses the first/99th intensity percentiles. These choices are
recorded in output metadata and are not a validated H&E calibration.

Choose a new output directory: the worker writes lossless `masks.tif` (`uint32`,
zero background, positive instance IDs) and `metadata.json`. It checks local
checkpoint size/digest before loading, bounds source pixels, verifies mask
read-back, and records input/checkpoint/output hashes and actual package versions.
It performs no hidden model downloads, size estimation, restoration or training.

`src.cellpose_worker.load_cellpose(checkpoint, checkpoint_sha256)` is available
inside this environment; `run_cellpose(...)` owns the bounded image/file workflow.
The main project's adapter invokes the worker using this environment's Python.
A small successful run verifies inference mechanics, not annotation accuracy.

The [code licence](https://github.com/MouseLand/cellpose/blob/v3.1.1.2/LICENSE)
is BSD-3-Clause. The [release notice](https://github.com/MouseLand/cellpose/blob/v3.1.1.2/README.md)
separately states that nuclei-model training includes CC-BY-NC data. Retain those
notices and cite Cellpose when using its outputs in research.

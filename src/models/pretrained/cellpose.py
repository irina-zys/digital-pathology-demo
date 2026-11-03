"""Main-environment adapter for the isolated Cellpose worker."""
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from src.imaging_utils.arrays import bounded_rgb
from src.models.pretrained.assets import CELLPOSE_SHA256

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_cellpose_patch(rgb, output_dir, *, checkpoint, checkpoint_sha256=CELLPOSE_SHA256,
                       diameter=17.0, invert=False, channels=(0, 0),
                       max_pixels=65_536, timeout_seconds=120, cache_dir=None):
    """Run one real patch in the isolated Cellpose environment and retain its outputs.

    The caller owns the output directory choice; existing outputs are refused.
    Diameter is measured in input pixels; None/0 uses the verified size model
    beside the nuclei checkpoint, with the same channel and polarity settings.
    Optional cache_dir scopes the child process Numba cache to a writable directory.
    """
    import tifffile

    bounded_rgb(rgb, max_pixels)
    if max(rgb.shape[:2]) > 256 or max_pixels > 65_536:
        raise ValueError("The Cellpose demo is bounded to 256 pixels per edge")
    if diameter is not None and diameter != 0 and (not np.isfinite(diameter) or not 5 <= diameter <= 128):
        raise ValueError("diameter must be 0/None (automatic) or finite and between 5 and 128 input pixels")
    if (type(invert) is not bool or len(channels) != 2 or
            any(type(c) is not int for c in channels) or channels[0] not in range(4) or channels[1] != 0):
        raise ValueError("Use a boolean invert flag and nuclei channels [0/1/2/3, 0]")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be an integer in 1-600")
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new Cellpose output directory")
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("Cellpose requires an existing local checkpoint")
    digest = checkpoint_sha256
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("An expected lowercase checkpoint SHA-256 digest is required")
    python = PROJECT_ROOT / "environments/cellpose/.venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError("Set up the approved environments/cellpose uv environment first")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cellpose-input-", dir=output.parent) as temporary:
        image = Path(temporary) / "patch.png"
        Image.fromarray(rgb).save(image)
        command = [str(python), "-m", "src.models.pretrained.cellpose_worker", str(image), str(output),
                   "--checkpoint", str(checkpoint), "--checkpoint-sha256", digest,
                   "--diameter", str(0 if diameter is None else diameter), "--channels", *map(str, channels),
                   "--max-pixels", str(max_pixels)]
        if invert:
            command.append("--invert")
        env = os.environ.copy()
        if cache_dir is not None:
            cache = Path(cache_dir).expanduser().resolve()
            cache.mkdir(parents=True, exist_ok=True)
            env["NUMBA_CACHE_DIR"] = str(cache)
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True,
                                timeout=timeout_seconds, env=env)
    if result.returncode:
        raise RuntimeError(f"Cellpose worker failed (exit {result.returncode}): {result.stderr[-2000:]}")
    metadata = json.loads((output / "metadata.json").read_text())
    labels = tifffile.imread(output / "masks.tif")
    if labels.dtype != np.uint32 or labels.shape != rgb.shape[:2]:
        raise RuntimeError("Invalid Cellpose instance-mask output")
    metadata["input_kind"] = "in_memory_rgb"
    metadata.pop("input")
    metadata["worker_input_png_sha256"] = metadata.pop("input_sha256")
    metadata["input_rgb_sha256"] = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return {"instance_mask": labels, "metadata": metadata, "output_dir": str(output)}

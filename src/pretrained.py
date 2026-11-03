"""Explicit, bounded loading of pretrained pathology models; no hidden downloads.

UNI produces patch embeddings. HoVer-NeXt and Cellpose produce instance masks.
Cellpose runs in its isolated environment when called from the main project.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.image_utils import validate_rgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELLPOSE_SHA256 = "89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303"
UNI_REVISION = "b55a5ec6cade1a39edfe6534189a9b8ca7a022f0"
UNI_SHA256 = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"


def checkpoint_sha256(path):
    """Hash a local checkpoint without loading it into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bounded_rgb(rgb, max_pixels):
    validate_rgb(rgb)
    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError("max_pixels must be a positive integer")
    if rgb.shape[0] * rgb.shape[1] > max_pixels:
        raise ValueError("Input exceeds the explicit patch pixel budget")


def load_uni(checkpoint, *, device="cpu", expected_sha256=UNI_SHA256):
    """Load original UNI from an authorized local checkpoint, never random weights.

    The public checkpoint is gated; obtain access directly from its provider.
    This loader targets original UNI (ViT-L/16, 1024 features), not UNI2.
    """
    import timm
    import torch
    from torchvision import transforms

    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if not checkpoint.is_file() or not 0 < checkpoint.stat().st_size <= 1_300_000_000:
        raise ValueError("Expected a local UNI checkpoint no larger than 1.3 GB")
    actual = checkpoint_sha256(checkpoint)
    if actual != expected_sha256:
        raise ValueError("UNI checkpoint SHA-256 does not match the declared weights")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Supported devices are cpu, cuda and mps")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    with torch.random.fork_rng(devices=[]):
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16,
                                  init_values=1e-5, num_classes=0, dynamic_img_size=True,
                                  pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(device)
    transform = transforms.Compose([
        transforms.Resize(224, antialias=True), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return {"model": model, "transform": transform, "device": device,
            "metadata": {"model": "UNI", "repo_id": "MahmoodLab/UNI", "revision": UNI_REVISION,
                         "checkpoint_sha256": actual, "checkpoint": str(checkpoint),
                         "features": 1024, "preprocessing": "square RGB patch; resize224; ImageNet normalization",
                         "device": str(device), "torch_version": torch.__version__, "timm_version": timm.__version__}}


def embed_uni(rgb, loaded, *, max_pixels=65_536):
    """Return one finite float32 1024-vector from a bounded square RGB patch."""
    import torch

    _bounded_rgb(rgb, max_pixels)
    if rgb.shape[0] != rgb.shape[1]:
        raise ValueError("UNI demo inputs must be square; choose an explicit crop first")
    batch = loaded["transform"](Image.fromarray(rgb.copy())).unsqueeze(0).to(loaded["device"])
    with torch.inference_mode():
        result = loaded["model"](batch).detach().cpu().float().numpy()
    if result.shape != (1, 1024) or not np.isfinite(result).all():
        raise RuntimeError("Unexpected UNI feature shape or nonfinite output")
    return result[0].copy()


def load_hovernext(checkpoint_dir, *, device="cpu"):
    """Load the pinned HoVer-NeXt PanNuke model through the local adapter."""
    from src.hovernext import load_hovernext as load

    return load(checkpoint_dir, device=device)


def predict_hovernext(rgb, loaded, *, max_pixels=65_536):
    """Return HoVer-NeXt instance labels, classes and recorded inference settings."""
    from src.hovernext import predict_hovernext as predict

    _bounded_rgb(rgb, max_pixels)
    return predict(rgb, loaded, max_pixels=max_pixels)


def load_cellpose(checkpoint, checkpoint_sha256, *, device="cpu"):
    """Load Cellpose inside its dedicated environment; main-env callers use the worker."""
    from src.cellpose_worker import load_cellpose as load

    return load(checkpoint, checkpoint_sha256, device=device)


def run_cellpose_patch(rgb, output_dir, *, checkpoint, checkpoint_sha256=CELLPOSE_SHA256,
                       diameter=17.0, invert=False, channels=(0, 0),
                       max_pixels=65_536, timeout_seconds=120, cache_dir=None):
    """Run one real patch in the isolated Cellpose environment and retain its outputs.

    The caller owns the output directory choice; existing outputs are refused.
    Diameter is measured in input pixels and is an explicit model assumption.
    Optional cache_dir scopes the child process Numba cache to a writable directory.
    """
    import tifffile

    _bounded_rgb(rgb, max_pixels)
    if max(rgb.shape[:2]) > 256 or max_pixels > 65_536:
        raise ValueError("The Cellpose demo is bounded to 256 pixels per edge")
    if not np.isfinite(diameter) or not 5 <= diameter <= 128:
        raise ValueError("diameter must be finite and between 5 and 128 input pixels")
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
        command = [str(python), "-m", "src.cellpose_worker", str(image), str(output),
                   "--checkpoint", str(checkpoint), "--checkpoint-sha256", digest,
                   "--diameter", str(diameter), "--channels", *map(str, channels),
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

"""Original UNI for bounded square patch demonstrations."""
from pathlib import Path
import numpy as np
from PIL import Image
from src.imaging_utils.arrays import bounded_rgb
from src.models.pretrained.assets import UNI_SHA256, UNI_REVISION
from src.utils.files import sha256


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
    actual = sha256(checkpoint)
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

    bounded_rgb(rgb, max_pixels)
    if rgb.shape[0] != rgb.shape[1]:
        raise ValueError("UNI demo inputs must be square; choose an explicit crop first")
    batch = loaded["transform"](Image.fromarray(rgb.copy())).unsqueeze(0).to(loaded["device"])
    with torch.inference_mode():
        result = loaded["model"](batch).detach().cpu().float().numpy()
    if result.shape != (1, 1024) or not np.isfinite(result).all():
        raise RuntimeError("Unexpected UNI feature shape or nonfinite output")
    return result[0].copy()

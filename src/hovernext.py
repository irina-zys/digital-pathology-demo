"""Bounded CPU-capable HoVer-NeXt PanNuke inference on one 256-pixel RGB patch.

The pinned upstream architecture and array postprocessing live in third_party.
This adapter uses one unaugmented view, without upstream WSI tiling or TTA; it
makes no claim of reproducing published benchmark performance.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tomllib

import numpy as np
from scipy.special import softmax


_COMMIT = "bde3c9d96172d16513e292d0a2dfcd7ff80202ff"
_VENDOR = Path(__file__).resolve().parents[1] / "third_party" / "HoVerNeXt"
_CHECKPOINT_SHA256 = "0c75821cfc2a0a49f7f75ebb57a4e4842f9ddd67798841180a6e3d8091261ed0"
_PARAMS_SHA256 = "ae6ac1a2ed23e89a935a863a647978ebb3d4c84eea02ca070ca91f304125c84c"
_THRESHOLDS_SHA256 = "2032d371e0c17c179f31f459696e2d5a08fa0928bc4b47bb932e7cd843b73cc3"


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _vendor_module(name, source):
    path = _VENDOR / f"{name}.py"
    if _sha256(path) != source["local_files_sha256"][path.name]:
        raise ValueError(f"Vendored HoVer-NeXt source hash mismatch: {path.name}")
    spec = importlib.util.spec_from_file_location(f"_pathology_hovernext_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hovernext(checkpoint_dir, device="cpu"):
    """Load the verified official PanNuke Tiny fold-1 checkpoint into an eval model.

    ``checkpoint_dir`` contains params.toml, pannuke_test_param_dict.json and
    train/best_model extracted from the published ZIP. Only this pinned model is
    supported. No network access, encoder download or unsafe pickle load occurs.
    The returned dict is an in-memory model handle for ``predict_hovernext``.
    """
    try:
        import torch
        import segmentation_models_pytorch  # noqa: F401 -- optional dependency check
    except ModuleNotFoundError as error:
        raise ImportError("HoVer-NeXt requires the optional pretrained extra: uv sync --locked --extra pretrained") from error

    target = torch.device(device)
    if target.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("HoVer-NeXt device must be cpu, cuda or mps")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use device='cpu'")
    if target.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use device='cpu'")
    root = Path(checkpoint_dir).resolve(strict=True)
    files = {"checkpoint": root / "train/best_model", "params": root / "params.toml",
             "thresholds": root / "pannuke_test_param_dict.json"}
    expected = {"checkpoint": _CHECKPOINT_SHA256, "params": _PARAMS_SHA256,
                "thresholds": _THRESHOLDS_SHA256}
    digests = {}
    for name, path in files.items():
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or escaping HoVer-NeXt {name} file")
        digests[name] = _sha256(path)
        if digests[name] != expected[name]:
            raise ValueError(f"HoVer-NeXt {name} does not match the verified PanNuke Tiny fold-1 release")
    with files["params"].open("rb") as stream:
        params = tomllib.load(stream)
    if (params["dataset"], params["encoder"], params["inst_channels"], params["out_channels_cls"]) != (
            "pannuke", "convnextv2_tiny.fcmae_ft_in22k_in1k", 5, 6):
        raise ValueError("Unsupported HoVer-NeXt model parameters")
    threshold_data = json.loads(files["thresholds"].read_text())
    metric = params["optim_metric"]
    thresholds = {key: np.asarray(threshold_data[f"best_{key}_{metric}"], dtype=float)
                  for key in ("fg", "seed")}
    if any(value.shape != (5,) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1))
           for value in thresholds.values()):
        raise ValueError("Invalid official postprocessing thresholds")
    source = json.loads((_VENDOR / "SOURCE.json").read_text())
    if source["commit"] != _COMMIT:
        raise ValueError("Unexpected vendored HoVer-NeXt revision")
    model_module = _vendor_module("multi_head_unet", source)
    postprocess = _vendor_module("postprocess", source)
    constants = _vendor_module("constants", source)
    model = model_module.get_model(enc=params["encoder"], out_channels_cls=params["out_channels_cls"],
                                   out_channels_inst=params["inst_channels"], pretrained=False)
    checkpoint = torch.load(files["checkpoint"], map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                                         for k, v in state.items()):
        raise ValueError("Expected a tensor model_state_dict")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(target).eval()
    metadata = {
        "model": "HoVer-NeXt", "checkpoint": "pannuke_convnextv2_tiny_1", "device": str(target),
        "source_commit": _COMMIT, "source_sha256": source["local_files_sha256"],
        "checkpoint_sha256": digests["checkpoint"], "params_sha256": digests["params"],
        "thresholds_sha256": digests["thresholds"], "model_parameters": params,
        "training_mpp": 0.25, "input_mpp_verified": False,
        "scale_assumption": "Input pixels are assumed to match PanNuke's 0.25 micrometres/pixel; this adapter cannot establish PNG calibration and does not resample.",
        "preprocessing": "uint8 RGB divided by 255; BCHW float32; no stain or ImageNet normalization",
        "tta_views": 0, "inference_mode": "one direct 256x256 patch; no WSI tiling or border padding",
        "postprocessing_metric": metric,
        "foreground_thresholds": thresholds["fg"].tolist(), "seed_thresholds": thresholds["seed"].tolist(),
        "min_object_pixels_exclusive": constants.MIN_THRESHS_PANNUKE,
        "max_object_pixels_exclusive": constants.MAX_THRESHS_PANNUKE,
        "max_hole_size": constants.MAX_HOLE_SIZE,
        "class_vote_transform": "upstream second softmax, uint8 quantization, foreground-channel argmax and per-instance majority vote",
        "weights_license": "CC-BY-NC-SA-4.0", "code_license": "GPL-3.0",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "timm", "segmentation-models-pytorch", "numpy", "scipy", "scikit-image", "opencv-python")},
    }
    return {"model": model, "device": target, "postprocess": postprocess, "constants": constants,
            "thresholds": thresholds, "metadata": metadata}


def predict_hovernext(rgb, loaded, *, max_pixels=65536):
    """Predict one uint8 RGB 256x256 patch without resizing or file writes.

    Return ``instance_mask`` (uint32, background 0), ``classes`` (positive ID to
    class_id/class_name), raw ``probabilities`` (instance:3xHxW, class:6xHxW), and
    JSON-serializable ``metadata``. Class 0 in the probability map is background;
    final object classes are PanNuke IDs 1-5. Probabilities are not calibrated
    confidence estimates. CPU callers can bound PyTorch threads before loading.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3):
        raise ValueError("HoVer-NeXt patch input must be an exact 256x256 uint8 RGB array; crop explicitly, do not resize implicitly")
    if type(max_pixels) is not int or max_pixels < 1 or rgb.shape[0] * rgb.shape[1] > max_pixels:
        raise ValueError("HoVer-NeXt input exceeds max_pixels")
    import torch

    tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=loaded["device"], dtype=torch.float32).div_(255)
    model = loaded["model"]
    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        if logits.shape != (1, 11, 256, 256) or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid HoVer-NeXt logits")
        instance_probs = logits[:, 2:5].softmax(1)[0].cpu().numpy()
        class_probs = logits[:, 5:].softmax(1)[0].cpu().numpy()
    # Preserve upstream storage/argmax semantics before the array-only watershed.
    stored_classes = (softmax(class_probs.astype(np.float32), axis=0) * 255).astype(np.uint8)
    winners = stored_classes[1:].argmax(axis=0)
    class_votes = np.arange(5)[:, None, None] == winners[None, :, :]
    post = loaded["postprocess"]
    constants = loaded["constants"]
    instances, skip = post.faster_instance_seg(instance_probs[:2].astype(np.float16), class_votes,
                                              loaded["thresholds"]["fg"], loaded["thresholds"]["seed"])
    if not skip:
        instances = post.post_proc_inst(instances, constants.MAX_HOLE_SIZE)
        classes = post.make_ct(class_votes, instances)
        instances, classes = post.remove_obj_cls(instances, classes, constants.MIN_THRESHS_PANNUKE,
                                                 constants.MAX_THRESHS_PANNUKE)
    else:
        classes = {}
    instances = instances.astype(np.uint32)
    names = {value: key for key, value in constants.CLASS_LABELS_PANNUKE.items()}
    mapping = {int(key): {"class_id": int(value), "class_name": names[value]} for key, value in classes.items()}
    if set(np.unique(instances)) - {0} != mapping.keys():
        raise RuntimeError("HoVer-NeXt instances and class table disagree")
    metadata = {**loaded["metadata"], "input_shape": list(rgb.shape), "max_pixels": max_pixels,
                "input_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "instances": len(mapping), "foreground_pixels": int(np.count_nonzero(instances)),
                "background_id": 0, "probability_class_ids": list(range(6)), "watershed_skipped": bool(skip)}
    return {"instance_mask": instances, "classes": mapping,
            "probabilities": {"instance": instance_probs, "class": class_probs}, "metadata": metadata}

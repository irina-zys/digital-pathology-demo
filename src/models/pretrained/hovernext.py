"""Bounded CPU-capable HoVer-NeXt PanNuke inference on one 256-pixel RGB patch.

The pinned upstream architecture and array postprocessing live in third_party.
This patch adapter includes the published spatial/HED TTA and released thresholds.
It omits upstream WSI tiling/stitching and does not reproduce a full benchmark.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import tomllib
from src.utils.files import sha256
from src.models.pretrained.assets import HOVER_FILES

import numpy as np
from scipy.special import softmax


_COMMIT = "bde3c9d96172d16513e292d0a2dfcd7ff80202ff"
_VENDOR = Path(__file__).resolve().parents[3] / "third_party" / "HoVerNeXt"
_CHECKPOINT_SHA256 = HOVER_FILES["train/best_model"][1]
_PARAMS_SHA256 = HOVER_FILES["params.toml"][1]
_THRESHOLDS_SHA256 = HOVER_FILES["pannuke_test_param_dict.json"][1]




def _vendor_module(name, source):
    path = _VENDOR / f"{name}.py"
    if sha256(path) != source["local_files_sha256"][path.name]:
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
        digests[name] = sha256(path)
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
        "inference_mode": "one direct 256x256 patch; no WSI tiling or border padding",
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



def _warp(tensor, matrix, mode):
    """Pinned upstream affine sampling, including its interpolation conventions."""
    import torch
    from torch.nn import functional as F

    theta = torch.tensor([matrix], device=tensor.device, dtype=tensor.dtype)
    grid = F.affine_grid(theta, tensor.size(), align_corners=False)
    return F.grid_sample(tensor, grid, mode=mode, padding_mode="zeros", align_corners=False)


def _predict_probabilities(tensor, model, spatial_params, tta_views, tta_seed):
    """Patch-only adaptation of pinned inference.py and augmentations.py (GPLv3).

    Preserve upstream sampling and averaging; local RNGs leave caller state alone.
    One view at a time bounds model activation memory independently of TTA count.
    """
    import torch

    rng = np.random.RandomState(tta_seed)
    generator = torch.Generator(device="cpu").manual_seed(tta_seed)
    rgb_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11],
                           [0.27, 0.57, 0.78]], dtype=np.float32)
    rgb_from_hed = torch.from_numpy(rgb_matrix).to(tensor.device)
    hed_from_rgb = torch.from_numpy(np.linalg.inv(rgb_matrix)).to(tensor.device)
    log_epsilon = torch.tensor(1e-6, device=tensor.device).log()
    instance_views, class_views = [], []
    with torch.inference_mode():
        for _ in range(max(1, tta_views)):
            view = tensor
            mirror = rotation = inverse_rotation = None
            if tta_views:
                active = {key: bool(rng.binomial(1, value["prob"]))
                          for key, value in spatial_params.items()}
                if active["mirror"]:
                    x = -1 if rng.binomial(1, spatial_params["mirror"]["prob_x"]) else 1
                    y = -1 if rng.binomial(1, spatial_params["mirror"]["prob_y"]) else 1
                    mirror = [[x, 0., 0.], [0., y, 0.]]
                    view = _warp(view, mirror, "bilinear")
                if active["rotate"]:
                    angle = math.radians(rng.choice([-270, -180, -90, 90, 180, 270]))
                    cosine, sine = math.cos(angle), math.sin(angle)
                    rotation = [[cosine, -sine, 0.], [sine, cosine, 0.]]
                    inverse_rotation = [[cosine, sine, 0.], [-sine, cosine, 0.]]
                    view = _warp(view, rotation, "bilinear")
                # Upstream HED: per-channel multiplicative and additive perturbations.
                scale = torch.empty(1, 3).uniform_(-.05, .05, generator=generator).to(tensor.device)
                bias = torch.empty(1, 3).uniform_(-.03, .03, generator=generator).to(tensor.device)
                hed = (view.movedim(1, -1).clamp(min=1e-6).log() / log_epsilon) @ hed_from_rgb
                hed = hed * (1 + scale[:, None, None, :]) + bias[:, None, None, :]
                view = torch.exp((hed * log_epsilon) @ rgb_from_hed).clamp(0, 1).movedim(-1, 1)
            logits = model(view)
            if logits.shape != (1, 11, 256, 256) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid HoVer-NeXt logits")
            if rotation is not None:
                logits = _warp(logits, inverse_rotation, "nearest")
            if mirror is not None:
                logits = _warp(logits, mirror, "nearest")
            instance_views.append(logits[:, 2:5].softmax(1))
            class_views.append(logits[:, 5:].softmax(1))
        # The publication averages per-view probabilities, not per-view logits.
        instance = torch.stack(instance_views).mean(0)[0].cpu().numpy()
        classes = torch.stack(class_views).mean(0)[0].cpu().numpy()
    return instance, classes


def predict_hovernext(rgb, loaded, *, max_pixels=65536, tta_views=16, tta_seed=42):
    """Predict one uint8 RGB 256x256 patch without resizing or file writes.

    Defaults use pinned inference spatial/HED settings with the release's
    16-view count and a fixed seed.
    ``tta_views=0`` retains direct inference for explicit diagnostics. Counts
    are bounded at 16, processed sequentially; caller RNG states are unchanged.

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
    if type(tta_views) is not int or not 0 <= tta_views <= 16:
        raise ValueError("tta_views must be an integer from 0 to 16")
    if type(tta_seed) is not int or not 0 <= tta_seed < 2**32:
        raise ValueError("tta_seed must be an integer from 0 to 2**32 - 1")
    import torch

    tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=loaded["device"], dtype=torch.float32).div_(255)
    model = loaded["model"]
    model.eval()
    instance_probs, class_probs = _predict_probabilities(
        tensor, model, loaded["constants"].TTA_AUG_PARAMS, tta_views, tta_seed)
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
                "recipe": "published_spatial_hed_tta" if tta_views else "direct_unaugmented",
                "tta_views": tta_views, "tta_seed": tta_seed if tta_views else None,
                "tta": {"spatial": loaded["constants"].TTA_AUG_PARAMS,
                        "hed_scale_uniform": [-.05, .05], "hed_bias_uniform": [-.03, .03],
                        "forward_interpolation": "bilinear", "inverse_interpolation": "nearest",
                        "aggregation": "mean of inverse-transformed per-view softmax probabilities",
                        "precision": "float32", "view_batch_size": 1} if tta_views else None,
                "annotations_supplied": False,
                "input_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "instances": len(mapping), "foreground_pixels": int(np.count_nonzero(instances)),
                "background_id": 0, "probability_class_ids": list(range(6)), "watershed_skipped": bool(skip)}
    return {"instance_mask": instances, "classes": mapping,
            "probabilities": {"instance": instance_probs, "class": class_probs}, "metadata": metadata}

"""Bounded CPU nuclei inference in environments/cellpose/.venv; no model downloads."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time


CHECKPOINT_NAME = "nucleitorch_0"
CHECKPOINT_BYTES = 26_563_614
SIZE_CHECKPOINT_NAME = "size_nucleitorch_0.npy"
SIZE_CHECKPOINT_BYTES = 3627
SIZE_CHECKPOINT_SHA256 = "a79107c9f284b569dd65f3a7363014d613a3403241c062822f07b6f026b4d3ef"
MAX_INPUT_PIXELS = 256 * 256


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_cellpose(checkpoint, checkpoint_sha256, *, device="cpu"):
    """Load the explicit classical nuclei checkpoint in the isolated environment.

    Reject missing/mismatched weights before constructing Cellpose, whose own
    constructor can otherwise fall back to a downloaded default model.
    """
    checkpoint = Path(checkpoint).resolve(strict=True)
    if str(device) != "cpu":
        raise ValueError("The bounded Cellpose loader supports CPU only")
    if checkpoint.name != CHECKPOINT_NAME or checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("Expected the explicit local nucleitorch_0 checkpoint")
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    if sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Checkpoint SHA-256 does not match")
    if importlib.metadata.version("cellpose") != "3.1.1.2":
        raise RuntimeError("Run this loader in the locked Cellpose 3.1.1.2 environment")
    import torch
    from cellpose import models

    torch.set_num_threads(1)
    model = models.CellposeModel(pretrained_model=str(checkpoint), model_type=None,
        gpu=False, device=torch.device("cpu"), mkldnn=False)
    model.net.eval()
    return model



def estimate_cellpose_diameter(model, rgb, checkpoint, *, channels, invert):
    """Use the official size model on the same image signal; observe its refinement fallback."""
    import numpy as np
    import torch
    from cellpose import models

    path = Path(checkpoint).parent / SIZE_CHECKPOINT_NAME
    if (not path.is_file() or path.stat().st_size != SIZE_CHECKPOINT_BYTES
            or sha256(path) != SIZE_CHECKPOINT_SHA256):
        raise ValueError("Prepare the verified matching Cellpose size checkpoint before automatic sizing")
    estimator = models.SizeModel(cp_model=model, pretrained_size=str(path))
    original_eval = model.eval
    refinement = {}

    def observed_eval(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        if kwargs.get("compute_masks") is True:
            refinement["objects"] = int(len(np.unique(result[0][result[0] > 0])))
        return result

    # The official estimator silently substitutes its training mean after an empty refinement.
    model.eval = observed_eval
    try:
        with torch.inference_mode():
            diameter, style = estimator.eval(rgb.copy(), channels=list(channels), channel_axis=-1,
                normalize={"normalize": True, "percentile": [1, 99], "invert": invert},
                invert=invert, augment=False, batch_size=1)
    finally:
        model.eval = original_eval
    if "objects" not in refinement:
        raise RuntimeError("The pinned size estimator did not execute its expected mask-refinement pass")
    if not np.isfinite([diameter, style]).all() or not 5 <= float(diameter) <= 128:
        raise ValueError("Estimated diameter exceeds the declared 5-128 pixel bounds; supply a manual estimate")
    if sha256(path) != SIZE_CHECKPOINT_SHA256:
        raise RuntimeError("Size checkpoint changed during estimation")
    return {"method": "official_size_model", "diameter": float(diameter), "style_diameter": float(style),
            "refinement_objects": refinement["objects"], "fallback_to_training_mean": refinement["objects"] == 0,
            "training_mean_diameter": float(estimator.diam_mean), "checkpoint": str(path),
            "checkpoint_sha256": SIZE_CHECKPOINT_SHA256, "ground_truth_used": False}


def run_cellpose(input_path, output_dir, *, checkpoint, checkpoint_sha256,
                 diameter=17.0, channels=(0, 0), invert=False, max_pixels=MAX_INPUT_PIXELS):
    """Write uint32 instance labels and provenance for one existing RGB PNG.

    Diameter is in input pixels, channels use Cellpose's 0=gray/1=R/2=G/3=B
    convention, and the second channel must be zero for the nuclei model.
    diameter=None or 0 uses the matching official size estimator on this image.
    No restoration, SAM model, prompt or annotation input is used.
    """
    import numpy as np
    from PIL import Image
    import tifffile

    input_path = Path(input_path).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=True)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Choose a new Cellpose output directory")
    output_dir = output_dir.resolve()
    if input_path.suffix.lower() != ".png" or not 0 < input_path.stat().st_size <= 8_000_000:
        raise ValueError("Input must be a PNG no larger than 8 MB")
    automatic_size = diameter is None or diameter == 0
    if not automatic_size and (not np.isfinite(diameter) or not 5 <= diameter <= 128):
        raise ValueError("diameter must be 0/None (automatic) or finite and between 5 and 128 input pixels")
    channels = tuple(channels)
    if len(channels) != 2 or channels[0] not in (0, 1, 2, 3) or channels[1] != 0:
        raise ValueError("Nuclei channels must be [0/1/2/3, 0]")
    if type(invert) is not bool:
        raise ValueError("invert must be boolean")
    if type(max_pixels) is not int or not 1 <= max_pixels <= MAX_INPUT_PIXELS:
        raise ValueError("max_pixels must be a positive integer no greater than 65,536")
    input_digest = sha256(input_path)
    with Image.open(input_path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("Input must be an RGB PNG; prepare other modes explicitly")
        if max(image.size) > 256 or image.width * image.height > max_pixels:
            raise ValueError("Input must be at most 256 x 256 pixels")
        rgb = np.asarray(image).copy()
    started = time.monotonic()
    import torch
    model = load_cellpose(checkpoint, checkpoint_sha256)
    calibration = (estimate_cellpose_diameter(model, rgb, checkpoint, channels=channels, invert=invert)
                   if automatic_size else {"method": "manual", "diameter": float(diameter), "source": "caller_supplied", "annotations_supplied": False})
    if calibration.get("fallback_to_training_mean"):
        output_dir.mkdir(parents=True)
        (output_dir / "failure.json").write_text(json.dumps({"status": "failed", "diameter_estimation": calibration,
            "error": "Size refinement found no objects; inspect RGB and supply an explicit diameter"}, indent=2))
        raise ValueError("Automatic size refinement found no objects; saved fallback metadata; specify a manual diameter")
    diameter = calibration["diameter"]
    settings = dict(channels=list(channels), channel_axis=-1, diameter=float(diameter),
        normalize={"normalize": True, "percentile": [1, 99], "invert": invert},
        invert=invert, batch_size=1, do_3D=False, augment=False, resample=True,
        bsize=224, niter=None, flow_threshold=0.4, cellprob_threshold=0.0,
        min_size=15, max_size_fraction=0.4)
    from cellpose import models, transforms
    # Use the pinned library's own channel/normalization functions for the teaching preview.
    prepared = transforms.convert_image(rgb.copy(), list(channels), channel_axis=-1, nchan=model.nchan)
    prepared = transforms.normalize_img(prepared[None, ...], **{**models.normalize_default, **settings["normalize"]})
    normalized_input = prepared[0, :, :, 0].copy()
    if normalized_input.shape != rgb.shape[:2] or not np.isfinite(normalized_input).all():
        raise RuntimeError("Invalid normalized input preview")
    with torch.inference_mode():
        labels, flows, styles = model.eval(rgb.copy(), **settings)
    if (labels.shape != rgb.shape[:2] or labels.dtype.kind not in "iu" or
            not np.isfinite(labels).all() or np.any(labels < 0) or
            labels.max() > np.iinfo(np.uint32).max):
        raise RuntimeError("Cellpose returned invalid instance labels")
    if not all(np.isfinite(array).all() for array in [*flows, styles]):
        raise RuntimeError("Cellpose returned non-finite flows or styles")
    labels = labels.astype(np.uint32)
    ids = np.unique(labels[labels > 0])
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise RuntimeError("Cellpose instance IDs are not consecutive")
    if sha256(input_path) != input_digest or sha256(checkpoint) != checkpoint_sha256:
        raise RuntimeError("Source or checkpoint changed during inference")
    metadata = {
        "status": "complete", "model": "cellpose3_nuclei", "device": "cpu", "torch_threads": 1,
        "input": str(input_path), "input_sha256": input_digest,
        "dimensions": [rgb.shape[1], rgb.shape[0]], "max_input_pixels": max_pixels,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": CHECKPOINT_BYTES, "settings": settings, "diameter_estimation": calibration,
        "mask": "masks.tif", "dtype": "uint32", "background": 0,
        "instance_count": len(ids), "foreground_pixels": int(np.count_nonzero(labels)),
        "finite_labels_flows_styles": True, "source_and_checkpoint_unchanged": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "versions": {name: importlib.metadata.version(name) for name in
            ("cellpose", "numpy", "torch", "scipy", "opencv-python-headless", "pillow", "tifffile")},
        "interpretation": "Unprompted pretrained nuclei inference; not an accuracy or clinical validation",
    }
    output_dir.mkdir(parents=True)
    mask_path = output_dir / metadata["mask"]
    with mask_path.open("xb") as stream:
        tifffile.imwrite(stream, labels, photometric="minisblack", compression="deflate", metadata=None)
    if not np.array_equal(tifffile.imread(mask_path), labels):
        raise RuntimeError("Saved mask differs from inference labels")
    metadata["mask_sha256"] = sha256(mask_path)
    preview_path = output_dir / "normalized_input.npy"
    with preview_path.open("xb") as stream:
        np.save(stream, normalized_input, allow_pickle=False)
    metadata["normalized_input"] = {"path": preview_path.name, "sha256": sha256(preview_path),
        "stage": "pinned Cellpose channel selection and normalization, before diameter scaling",
        "channel": 0, "shape": list(normalized_input.shape), "dtype": str(normalized_input.dtype)}
    with (output_dir / "metadata.json").open("x") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--diameter", type=float, default=17.0, help="Input nucleus diameter; 0 uses the verified size model")
    parser.add_argument("--channels", type=int, nargs=2, default=[0, 0])
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=MAX_INPUT_PIXELS)
    args = parser.parse_args()
    import torch

    torch.set_num_interop_threads(1)
    result = run_cellpose(args.input, args.output, checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256, diameter=args.diameter,
        channels=args.channels, invert=args.invert, max_pixels=args.max_pixels)
    print(json.dumps({"status": result["status"], "instances": result["instance_count"],
                      "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()

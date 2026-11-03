"""Acquire missing demo prerequisites through the repository's bounded CLIs.

Existing originals are reused in the canonical data tree. This prepares inputs,
not environments, and never requests gated UNI weights or replaces invalid files.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_IDS = ("20c9afac-0d0c-4fcf-b3bf-a2beba3dd76b", "5048c79f-ed1d-4534-8da2-1a4d207b2371")
DINO_REVISION = "10e440b8a34dfd657a90f2fcaa988c6b6a2f0da4"
DINO_SHA256 = "8d4795b8e4e327c6d841e15a2e3e6d88c45346256a43f8cdae8c26d92c806f7a"
DINO_FOLDER = "data/models/huggingface/hub/models--timm--vit_small_patch16_224.dino"
MODEL_FILES = {
    "hovernext": [
        ("data/models/hovernext/pannuke_convnextv2_tiny_1/params.toml", 1106,
         "ae6ac1a2ed23e89a935a863a647978ebb3d4c84eea02ca070ca91f304125c84c"),
        ("data/models/hovernext/pannuke_convnextv2_tiny_1/pannuke_test_param_dict.json", 458,
         "2032d371e0c17c179f31f459696e2d5a08fa0928bc4b47bb932e7cd843b73cc3"),
        ("data/models/hovernext/pannuke_convnextv2_tiny_1/train/best_model", 144433761,
         "0c75821cfc2a0a49f7f75ebb57a4e4842f9ddd67798841180a6e3d8091261ed0"),
    ],
    "cellpose": [
        ("data/models/cellpose/nucleitorch_0", 26563614,
         "89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303"),
        ("data/models/cellpose/size_nucleitorch_0.npy", 3627,
         "a79107c9f284b569dd65f3a7363014d613a3403241c062822f07b6f026b4d3ef"),
    ],
    "dino": [(f"{DINO_FOLDER}/snapshots/{DINO_REVISION}/model.safetensors", 86676340, DINO_SHA256)],
}
SEGMENTATION_FILES = {
    "cytonuke": [("data/cytonuke/extracted/coco.json", 64_000_000),
                 ("data/cytonuke/extracted/images/C3L-00977-22.svsx29y31.png", 2_000_000)],
    "nuinsseg": [("data/nuinsseg/extracted/human bladder/tissue images/human_bladder_01.png", 4_000_000),
                 ("data/nuinsseg/extracted/human bladder/label masks modify/human_bladder_01.tif", 4_000_000),
                 ("data/nuinsseg/extracted/human bladder/vague areas/mask binary/human_bladder_01.png", 4_000_000)],
}


def _asset(path, *, size=None, digest=None, algorithm="sha256", max_bytes=None):
    """Check existing inputs before acquisition; missing files alone are recoverable."""
    path = ROOT / path
    if not path.resolve().is_relative_to((ROOT / "data").resolve()):
        raise ValueError(f"Asset leaves the canonical data directory: {path}")
    if not path.exists() and not path.is_symlink():
        return False
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Preserving invalid existing asset: {path}")
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"Preserving asset with unexpected size: {path}")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ValueError(f"Asset exceeds its read budget: {path}")
    if digest is not None:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, algorithm).hexdigest()
        if actual != digest:
            raise ValueError(f"Preserving asset with mismatched {algorithm}: {path}")
    return True


def _run(arguments, receipt, *, timeout=3600):
    command = [sys.executable, *map(str, arguments)]
    print("Preparing missing inputs:", " ".join(command), flush=True)
    # Only this explicit preparation subprocess can access the network; inference stays offline.
    env = dict(os.environ)
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=timeout)
    receipt["commands_run"].append(command)


def _tcga(demo, receipt, check_only):
    with (ROOT / "manifests/tcga_demo.csv").open(newline="") as stream:
        roster = list(csv.DictReader(stream))
    if len(roster) != 10 or len({row["file_id"] for row in roster}) != 10:
        raise ValueError("Expected the frozen ten-slide TCGA roster")
    rows = [row for row in roster if demo != "preprocessing" or row["file_id"] in PREPROCESSING_IDS]
    if demo == "preprocessing" and {row["file_id"] for row in rows} != set(PREPROCESSING_IDS):
        raise ValueError("The frozen roster lacks the two preprocessing demo slides")
    missing = []
    for row in rows:
        expected = f"data/tcga/{row['project']}/{row['file_id']}.svs"
        if row["path"] != expected:
            raise ValueError("Unexpected canonical TCGA path in roster")
        if not _asset(row["path"], size=int(row["file_size"]), digest=row["md5sum"], algorithm="md5"):
            missing.append(row)
    if missing and not check_only:
        arguments = [ROOT / "scripts/download_tcga.py", "--workers", "2", "--max-gb", "4"]
        for row in missing:
            arguments += ["--file-id", row["file_id"]]
        _run(arguments, receipt)
        for row in missing:
            if not _asset(row["path"], size=int(row["file_size"]), digest=row["md5sum"], algorithm="md5"):
                raise FileNotFoundError(row["path"])
    receipt["tcga"] = {"slides": len(rows), "bytes": sum(int(row["file_size"]) for row in rows),
                       "integrity_check": "GDC size and MD5", "missing_before": [row["path"] for row in missing]}
    if check_only:
        receipt["missing"].extend(row["path"] for row in missing)
    return roster


def _model(name, receipt, check_only):
    missing = [path for path, size, digest in MODEL_FILES[name]
               if not _asset(path, size=size, digest=digest)]
    if name == "dino":
        reference = ROOT / DINO_FOLDER / "refs/main"
        if reference.exists() or reference.is_symlink():
            _asset(reference, max_bytes=100)
            if reference.read_text().strip() != DINO_REVISION:
                raise ValueError("Preserving DINO cache with a different main revision")
        else:
            missing.append(str(reference.relative_to(ROOT)))
    if missing and not check_only:
        weight_bytes = sum(size for path, size, _ in MODEL_FILES[name] if path in missing)
        print(f"{name}: up to {weight_bytes / 1e6:.1f} MB of missing required weights; existing files are reused", flush=True)
        _run([ROOT / "scripts/download_pretrained.py", "--model", name], receipt)
        for path, size, digest in MODEL_FILES[name]:
            if not _asset(path, size=size, digest=digest):
                raise FileNotFoundError(path)
        if name == "dino" and reference.read_text().strip() != DINO_REVISION:
            raise ValueError("DINO cache revision differs from the pinned demo")
    receipt.setdefault("models", {})[name] = {"integrity_check": "pinned size and SHA-256", "missing_before": missing}
    if check_only:
        receipt["missing"].extend(missing)


def _segmentation(receipt, check_only):
    for dataset, assets in SEGMENTATION_FILES.items():
        missing = [path for path, budget in assets if not _asset(path, max_bytes=budget)]
        if missing and not check_only:
            _run([ROOT / "scripts/download_segmentation_data.py", "--dataset", dataset, "--demo-only"], receipt)
            for path, budget in assets:
                if not _asset(path, max_bytes=budget):
                    raise FileNotFoundError(path)
        receipt.setdefault("datasets", {})[dataset] = {"integrity_check": "required nonempty bounded files; notebook validates content",
                                                       "missing_before": missing}
        if check_only:
            receipt["missing"].extend(missing)
    for name in ("hovernext", "cellpose"):
        _model(name, receipt, check_only)


def _coordinates(path, roster):
    """Validate the saved ten-slide export contract without decoding WSI pixels."""
    import h5py

    manifest_path = path / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 5_000_000:
        raise ValueError(f"Preserving incomplete preprocessing directory; choose a fresh path: {path}")
    manifest = json.loads(manifest_path.read_text())
    slides = manifest.get("slides", [])
    expected = {row["file_id"]: row for row in roster}
    if manifest.get("status") != "complete" or len(slides) != 10:
        raise ValueError("Preprocessing must be a completed ten-slide run")
    seen = set()
    for slide in slides:
        source = ROOT / slide["input"]
        identifier = source.stem
        if identifier in seen or identifier not in expected:
            raise ValueError("Preprocessing does not match the ten-slide roster")
        seen.add(identifier)
        row = expected[identifier]
        if source.resolve() != (ROOT / row["path"]).resolve() or slide.get("input_bytes") != int(row["file_size"]):
            raise ValueError("Preprocessing input paths or sizes differ from the frozen roster")
        layouts = [layout for layout in slide.get("exports", {}).get("layouts", [])
                   if layout.get("patch_size") == 256 and layout.get("target_mpp") == 0.5]
        if len(layouts) != 1:
            raise ValueError("Each slide needs one 256-pixel, 0.5-MPP coordinate export")
        layout = layouts[0]
        count = layout.get("exported_count")
        if not isinstance(count, int) or not 0 < count <= 200_000 or count != layout.get("eligible_count"):
            raise ValueError("Demo requires all nonempty eligible coordinates within its budget")
        coords = (path / layout["coords"]).resolve()
        if not coords.is_relative_to(path) or not coords.is_file() or coords.stat().st_size > 100_000_000:
            raise ValueError("Missing, escaping or oversized coordinate file")
        with h5py.File(coords, "r") as handle:
            points = handle["coords"]
            if (points.shape != (count, 2) or points.dtype.kind not in "iu"
                    or points.attrs.get("name") != identifier or points.attrs.get("patch_level") != 0
                    or points.attrs.get("output_patch_size") != 256 or points.attrs.get("target_mpp") != 0.5):
                raise ValueError("Coordinate file shape or calibration differs from its manifest")
    return {"path": str(path), "slides": len(slides), "mask_method": manifest.get("mask_method"),
            "integrity_check": "manifest identity, input sizes and HDF5 shape/calibration; downstream loader validates coordinates"}


def prepare_demo_inputs(demo, *, check_only=False, prepare_coordinates=False, preprocessing_run="outputs/tcga-mpp05"):
    """Prepare only required inputs; check_only never downloads or creates outputs."""
    if demo not in {"preprocessing", "segmentation", "wsi-classification"}:
        raise ValueError("Unknown demo")
    if prepare_coordinates and demo != "wsi-classification":
        raise ValueError("Coordinate preparation applies only to the WSI classification demo")
    receipt = {"demo": demo, "project": str(ROOT), "commands_run": [], "missing": []}
    if demo == "segmentation":
        _segmentation(receipt, check_only)
    else:
        roster = _tcga(demo, receipt, check_only)
        if demo == "wsi-classification":
            _model("dino", receipt, check_only)
            path = ROOT / preprocessing_run
            if path.is_symlink() or not path.resolve().is_relative_to((ROOT / "outputs").resolve()):
                raise ValueError("Preprocessing output must stay in the project's outputs directory")
            path = path.resolve()
            if not path.exists():
                if check_only:
                    receipt["missing"].append(str(path))
                elif not prepare_coordinates:
                    raise FileNotFoundError("Coordinate run missing; use --prepare-coordinates with a fresh --preprocessing-run path")
                else:
                    _run(["-m", "src.preprocess", "--config-name", "lung", "data.max_slides=10", "export.pixel_slide_id=null",
                          "export.max_patches=[0]", f"hydra.run.dir={json.dumps(str(path))}"], receipt, timeout=1800)
                    _run([ROOT / "scripts/verify_run.py", path], receipt, timeout=300)
            if path.exists():
                receipt["preprocessing"] = _coordinates(path, roster)
    receipt["status"] = "missing" if receipt["missing"] else "ready"
    return receipt

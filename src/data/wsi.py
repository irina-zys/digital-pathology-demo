"""Bounded MIL bags read directly from calibrated WSI coordinate exports."""

import csv
import hashlib
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.data.vision import VisionDataModule
from src.tiling import calibrated_patch_size, read_tile


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_coordinates(row, count, target_mpp, max_pixels):
    """Verify one existing export and record a fixed, evenly indexed subset."""
    path = Path(row["coords_path"])
    if path.stat().st_size > 20_000_000:
        raise ValueError("Coordinate HDF5 exceeds the demo's 20 MB bound")
    digest = file_sha256(path)
    with h5py.File(path, "r") as handle:
        coords = handle["coords"]
        attrs = coords.attrs
        if (coords.ndim != 2 or coords.shape[1] != 2 or coords.dtype.kind not in "iu"
                or not 1 <= len(coords) <= 200_000):
            raise ValueError("Expected 1–200,000 integer xy coordinate pairs")
        if (handle.attrs.get("coordinate_space") != "level0_xy_top_left"
                or int(attrs.get("patch_level", -1)) != 0
                or int(attrs.get("output_patch_size", -1)) != 256
                or bool(handle.attrs.get("use_padding", True))):
            raise ValueError("Expected unpadded level-0 coordinates for 256-pixel output tiles")
        if attrs.get("name") != row["bag_id"]:
            raise ValueError("Coordinate export slide identity differs from the bag")
        if int(handle.attrs.get("eligible_count", -1)) != len(coords):
            raise ValueError("Use a complete eligible coordinate export, not a capped preview")
        if not np.isclose(float(attrs.get("target_mpp", np.nan)), target_mpp, atol=1e-8, rtol=0):
            raise ValueError("Coordinate target MPP differs from the requested scale")
        native = np.asarray(attrs["native_mpp"], dtype=float)
        source_size, effective = calibrated_patch_size(256, target_mpp, native)
        stored_size = np.asarray(attrs["patch_size"])
        if not np.all(stored_size == np.array(source_size)) or source_size[0] * source_size[1] > max_pixels:
            raise ValueError("Invalid or over-budget native patch footprint")
        if not np.allclose(attrs["effective_mpp"], effective, atol=1e-8, rtol=0):
            raise ValueError("Exported effective MPP disagrees with its footprint")
        dimensions = np.asarray(attrs["level_dim"], dtype=np.int64)
        indices = np.linspace(0, len(coords) - 1, min(count, len(coords)), dtype=int)
        selected = coords[indices]
        if (dimensions.shape != (2,) or np.any(selected < 0)
                or np.any(selected + source_size > dimensions)
                or len(np.unique(selected, axis=0)) != len(selected)):
            raise ValueError("Selected coordinates are duplicated or outside the slide")
        fractions = np.asarray(handle["tissue_fraction"][indices])
        if not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
            raise ValueError("Invalid recorded tissue fractions")
        result = dict(indices=indices.tolist(), coords=selected.tolist(), source_size=list(source_size),
                      dimensions=dimensions.tolist(), native_mpp=native.tolist(), effective_mpp=effective,
                      target_mpp=target_mpp, output_patch_size=256, selected_count=len(indices),
                      eligible_count=len(coords), tissue_fraction=fractions.tolist(),
                      coordinate_sha256=digest, selection="evenly_spaced_export_row_indices",
                      export_schema_version=int(handle.attrs.get("export_schema_version", -1)),
                      masking_method=str(handle.attrs.get("masking_method", "unknown")),
                      tiling_method=str(handle.attrs.get("tiling_method", "unknown")))
    with openslide.OpenSlide(row["wsi_path"]) as slide:
        actual_mpp = [float(slide.properties[f"openslide.mpp-{axis}"]) for axis in ("x", "y")]
        if tuple(slide.dimensions) != tuple(dimensions) or not np.allclose(actual_mpp, native, atol=1e-8, rtol=0):
            raise ValueError("Source WSI dimensions or calibration differ from the coordinate export")
    info = Path(row["wsi_path"]).stat()
    if row.get("source_size_bytes") and int(row["source_size_bytes"]) != info.st_size:
        raise ValueError("Source WSI size differs from its declared manifest")
    result.update(source_bytes=info.st_size, source_mtime_ns=info.st_mtime_ns,
                  source_checksum_verified=False)
    return result


class WSIBagDataset(Dataset):
    def __init__(self, samples, selections, image_size, mean, std, augment, max_pixels):
        self.samples, self.selections = samples, selections
        self.image_size, self.mean, self.std = image_size, mean, std
        self.augment, self.max_pixels = augment, max_pixels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index][0]
        selection = self.selections[row["bag_id"]]
        info = Path(row["wsi_path"]).stat()
        if (info.st_size, info.st_mtime_ns) != (selection["source_bytes"], selection["source_mtime_ns"]):
            raise ValueError("Source WSI changed since setup")
        if file_sha256(row["coords_path"]) != selection["coordinate_sha256"]:
            raise ValueError("Coordinate export changed since setup")
        images = []
        with openslide.OpenSlide(row["wsi_path"]) as slide:
            for xy in selection["coords"]:
                rgb = read_tile(slide, xy, selection["source_size"], 256, max_read_pixels=self.max_pixels)
                tensor = TF.to_tensor(rgb)
                if self.image_size is not None:
                    tensor = TF.resize(tensor, [self.image_size] * 2,
                                       interpolation=InterpolationMode.BILINEAR, antialias=True)
                if self.augment:
                    for axis in (-1, -2):
                        if torch.rand(()) < 0.5:
                            tensor = tensor.flip(axis)
                images.append(TF.normalize(tensor, self.mean, self.std))
        return {"image": torch.stack(images), "target": int(row["label"]), "id": row["bag_id"]}


class WSIBagDataModule(VisionDataModule):
    """Same Lightning interfaces as patch MIL, with no patch-image files required."""
    def __init__(self, *, target_mpp=0.5, max_slides=20, **kwargs):
        super().__init__(**kwargs)
        if self.hparams.task != "mil" or not np.isfinite(target_mpp) or target_mpp <= 0:
            raise ValueError("WSI coordinate input requires task=mil and positive target_mpp")
        if type(max_slides) is not int or not 1 <= max_slides <= 100:
            raise ValueError("max_slides must be an integer in 1–100")
        if self.hparams.max_patches_per_bag > 64 or self.hparams.num_workers != 0:
            raise ValueError("This bounded WSI reader permits up to 64 patches/bag and num_workers=0")
        if self.hparams.image_size is not None and self.hparams.image_size ** 2 > self.hparams.max_pixels:
            raise ValueError("Resized model input exceeds max_pixels")
        self.target_mpp, self.max_slides = target_mpp, max_slides
        self.selections = {}
        self.contract.pop("mask_policy")
        self.contract.update(input_kind="wsi_coordinates_v1", target_mpp=target_mpp, output_patch_size=256,
                             selection="evenly_spaced_export_row_indices",
                             max_patches_per_bag=self.hparams.max_patches_per_bag)

    def setup(self, stage=None):
        if not self.datasets:
            root = Path(to_absolute_path(self.hparams.data_root))
            manifest = Path(to_absolute_path(self.hparams.manifest))
            if manifest.stat().st_size > 1_000_000:
                raise ValueError("WSI manifest exceeds 1 MB")
            with manifest.open(newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"wsi_path", "coords_path", "split", "group_id", "label", "bag_id"}
                if not required <= set(reader.fieldnames or []):
                    raise ValueError(f"WSI manifest requires {sorted(required)}")
                rows = list(reader)
            if not 1 <= len(rows) <= self.max_slides:
                raise ValueError("WSI manifest exceeds max_slides or is empty")
            groups, seen, bags = {}, set(), set()
            samples = {split: [] for split in ("train", "val", "test")}
            for row in rows:
                if any(not row.get(key, "").strip() for key in required):
                    raise ValueError("WSI manifest contains an empty required field")
                split, group, bag = row["split"], row["group_id"], row["bag_id"]
                if split not in samples or groups.setdefault(group, split) != split:
                    raise ValueError("Unknown split or patient group crossing splits")
                if not 0 <= int(row["label"]) < len(self.hparams.class_names):
                    raise ValueError("Label is outside ordered class_names")
                for key in ("wsi_path", "coords_path"):
                    path = (root / row[key]).resolve(strict=True)
                    if not path.is_file():
                        raise ValueError(f"Expected an existing file: {path}")
                    row[key] = str(path)
                if row["wsi_path"] in seen or bag in bags or Path(row["wsi_path"]).stem != bag:
                    raise ValueError("Duplicate source/bag or bag does not match WSI stem")
                seen.add(row["wsi_path"])
                bags.add(bag)
                samples[split].append([row])
            # Validate all identities before any per-split cap or pixel reads.
            self.rows = rows
            for split, items in samples.items():
                if self.hparams.max_samples_per_split is not None:
                    items = items[:self.hparams.max_samples_per_split]
                for sample in items:
                    row = sample[0]
                    self.selections[row["bag_id"]] = select_coordinates(row, self.hparams.max_patches_per_bag,
                                                                       self.target_mpp, self.hparams.max_pixels)
                self.datasets[split] = WSIBagDataset(items, self.selections, self.hparams.image_size,
                    self.hparams.mean, self.hparams.std, self.hparams.augment and split == "train", self.hparams.max_pixels)
        super().setup(stage)

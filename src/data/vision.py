"""Explicit, group-disjoint manifests for patches, MIL bags and binary masks."""

import csv
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import torch
from hydra.utils import to_absolute_path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.annotations import check_size, read_mask


class PatchDataset(Dataset):
    def __init__(self, samples, task, image_size, mean, std, augment,
                 max_pixels, max_patches_per_bag):
        self.samples = samples
        self.task = task
        self.image_size = image_size
        self.mean, self.std = mean, std
        self.augment = augment
        self.max_pixels = max_pixels
        self.max_patches_per_bag = max_patches_per_bag

    def __len__(self):
        return len(self.samples)

    def _patch(self, row):
        with Image.open(row["image_path"]) as image:
            check_size(image.height, image.width, self.max_pixels)
            rgb = TF.to_tensor(image.convert("RGB"))
        target = valid = None
        if self.task == "segmentation":
            labels = read_mask(Path(row["mask_path"]), self.max_pixels)
            if labels.shape != tuple(rgb.shape[-2:]):
                raise ValueError(f"Image/mask shape mismatch: {row['image_path']}")
            target = torch.from_numpy((labels > 0).astype(np.float32))[None]
            valid = torch.ones_like(target)
            if row.get("ignore_path"):
                ignored = read_mask(Path(row["ignore_path"]), self.max_pixels)
                if ignored.shape != labels.shape:
                    raise ValueError(f"Ignore-mask shape mismatch: {row['ignore_path']}")
                valid = torch.from_numpy((ignored == 0).astype(np.float32))[None]
        if self.image_size is not None:
            size = [self.image_size, self.image_size]
            rgb = TF.resize(rgb, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
            if target is not None:
                target = TF.resize(target, size, interpolation=InterpolationMode.NEAREST)
                valid = TF.resize(valid, size, interpolation=InterpolationMode.NEAREST)
        if self.augment:
            for axis in (-1, -2):
                if torch.rand(()) < 0.5:
                    rgb = rgb.flip(axis)
                    if target is not None:
                        target, valid = target.flip(axis), valid.flip(axis)
        rgb = TF.normalize(rgb, self.mean, self.std)
        if target is not None and not valid.any():
            raise ValueError(f"No evaluable pixels: {row['image_path']}")
        return rgb, target, valid

    def __getitem__(self, index):
        rows = self.samples[index]
        if self.task == "mil":
            count = min(len(rows), self.max_patches_per_bag)
            indices = torch.randperm(len(rows))[:count].tolist() if self.augment else range(count)
            images = torch.stack([self._patch(rows[i])[0] for i in indices])
            return {"image": images, "target": int(rows[0]["label"]), "id": rows[0]["bag_id"]}
        row = rows[0]
        image, target, valid = self._patch(row)
        if self.task == "classification":
            return {"image": image, "target": int(row["label"]), "id": row["image_path"]}
        return {"image": image, "target": target, "valid": valid, "id": row["image_path"]}


class VisionDataModule(L.LightningDataModule):
    def __init__(self, manifest, data_root, task, class_names=None, batch_size=8,
                 num_workers=0, image_size=224, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), augment=True, max_pixels=4_000_000,
                 max_samples_per_split=None, max_patches_per_bag=64):
        super().__init__()
        self.save_hyperparameters()
        if task not in {"classification", "segmentation", "mil"}:
            raise ValueError("task must be classification, segmentation or mil")
        if batch_size < 1 or num_workers < 0 or max_pixels < 1 or max_patches_per_bag < 1:
            raise ValueError("Invalid data resource bounds")
        if image_size is not None and (type(image_size) is not int or image_size < 32):
            raise ValueError("image_size must be null or an integer >= 32")
        if image_size is not None and image_size ** 2 > max_pixels:
            raise ValueError("Resized model input exceeds max_pixels")
        if max_samples_per_split is not None and max_samples_per_split < 1:
            raise ValueError("max_samples_per_split must be positive or null")
        if len(mean) != 3 or len(std) != 3 or not np.isfinite([*mean, *std]).all() or min(std) <= 0:
            raise ValueError("Expected three finite normalization means and positive standard deviations")
        if task == "mil" and batch_size != 1:
            raise ValueError("Variable-size MIL bags require data.batch_size=1; use gradient accumulation")
        if task != "segmentation" and (not class_names or len(class_names) < 2 or len(set(class_names)) != len(class_names)):
            raise ValueError("Set ordered, unique class_names (at least two)")
        self.contract = dict(task=task, class_names=list(class_names or []), image_size=image_size,
                             mean=list(mean), std=list(std), mask_policy="nonzero_foreground_ignore_nonzero")
        self.datasets = {}
        self.rows = []

    def setup(self, stage=None):
        if not self.datasets:
            root = Path(to_absolute_path(self.hparams.data_root))
            with Path(to_absolute_path(self.hparams.manifest)).open(newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"image_path", "split", "group_id"}
                required |= {"mask_path"} if self.hparams.task == "segmentation" else {"label"}
                if self.hparams.task == "mil":
                    required.add("bag_id")
                if not required <= set(reader.fieldnames or []):
                    raise ValueError(f"Manifest requires columns: {sorted(required)}")
                rows = list(reader)
            if not rows:
                raise ValueError("Empty training manifest")
            groups, seen, bags = {}, set(), {}
            samples = {split: defaultdict(list) for split in ("train", "val", "test")}
            for row in rows:
                if any(not row.get(key, "").strip() for key in required):
                    raise ValueError("Manifest contains an empty required field")
                for key in ("split", "group_id", "bag_id"):
                    if row.get(key) is not None:
                        row[key] = row[key].strip()
                split, group = row["split"], row["group_id"]
                if split not in samples:
                    raise ValueError(f"Unknown split: {split}")
                if groups.setdefault(group, split) != split:
                    raise ValueError(f"Group crosses splits: {group}")
                for key in ("image_path", "mask_path", "ignore_path"):
                    if row.get(key):
                        path = (root / row[key]).resolve(strict=True)
                        if not path.is_file():
                            raise ValueError(f"Expected file: {path}")
                        row[key] = str(path)
                if row["image_path"] in seen:
                    raise ValueError(f"Duplicate image in manifest: {row['image_path']}")
                seen.add(row["image_path"])
                if self.hparams.task != "segmentation":
                    label = int(row["label"])
                    if not 0 <= label < len(self.hparams.class_names):
                        raise ValueError(f"Label outside class_names: {label}")
                key = row["image_path"]
                if self.hparams.task == "mil":
                    key = row["bag_id"]
                    identity = (split, group, int(row["label"]))
                    if bags.setdefault(key, identity) != identity:
                        raise ValueError(f"Bag has inconsistent split/group/label: {key}")
                samples[split][key].append(row)
            self.rows = rows
            for split, entries in samples.items():
                items = [sorted(value, key=lambda row: row["image_path"]) for value in entries.values()]
                if self.hparams.max_samples_per_split is not None:
                    items = items[:self.hparams.max_samples_per_split]
                self.datasets[split] = PatchDataset(
                    items, self.hparams.task, self.hparams.image_size, self.hparams.mean,
                    self.hparams.std, self.hparams.augment and split == "train",
                    self.hparams.max_pixels, self.hparams.max_patches_per_bag)
        required = {"fit": ("train", "val"), "validate": ("val",), "test": ("test",), "predict": ("test",)}
        for split in required.get(stage, ()):
            if not len(self.datasets[split]):
                raise ValueError(f"Stage {stage} requires an explicit nonempty {split} split")

    def _loader(self, split):
        return DataLoader(self.datasets[split], batch_size=self.hparams.batch_size,
                          shuffle=split == "train", num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")

    def predict_dataloader(self):
        return self.test_dataloader()

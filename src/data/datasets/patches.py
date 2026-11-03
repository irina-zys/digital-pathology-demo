"""Bounded patch and patch-bag datasets; no training lifecycle."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from src.data.annotations import check_size, read_mask
from src.data.transforms import transform_patch


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
        rgb, target, valid = transform_patch(rgb, image_size=self.image_size, mean=self.mean,
            std=self.std, augment=self.augment, target=target, valid=valid)
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

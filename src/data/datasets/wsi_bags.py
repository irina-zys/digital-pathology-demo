"""Bounded image bags from calibrated whole-slide coordinates."""
from pathlib import Path
import openslide
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from src.data.transforms import transform_patch
from src.wsi_preprocessing.tiling import read_tile
from src.utils.files import sha256


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
        if sha256(row["coords_path"]) != selection["coordinate_sha256"]:
            raise ValueError("Coordinate export changed since setup")
        images = []
        with openslide.OpenSlide(row["wsi_path"]) as slide:
            for xy in selection["coords"]:
                rgb = read_tile(slide, xy, selection["source_size"], 256, max_read_pixels=self.max_pixels)
                tensor = TF.to_tensor(rgb)
                tensor, _, _ = transform_patch(tensor, image_size=self.image_size,
                    mean=self.mean, std=self.std, augment=self.augment)
                images.append(tensor)
        return {"image": torch.stack(images), "target": int(row["label"]), "id": row["bag_id"]}

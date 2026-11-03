"""Paired views with deterministic validation augmentations."""
from pathlib import Path
import openslide
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.data.transforms import contrastive_transform
from src.wsi_preprocessing.tiling import read_tile
from src.utils.files import sha256


class PairedPatchDataset(Dataset):
    def __init__(self, samples, selections, *, training):
        self.samples = [[row] for group in samples for row in group for _ in selections[row['bag_id']]['coords']]
        self.entries = [(group[0], xy) for group in samples for xy in selections[group[0]['bag_id']]['coords']]
        self.selections, self.training = selections, training
        self.augment = contrastive_transform()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row, xy = self.entries[index]
        selection = self.selections[row['bag_id']]
        info = Path(row['wsi_path']).stat()
        if ((info.st_size, info.st_mtime_ns) != (selection['source_bytes'], selection['source_mtime_ns'])
                or sha256(row['coords_path']) != selection['coordinate_sha256']):
            raise ValueError('Source changed after SSL input selection')
        with openslide.OpenSlide(row['wsi_path']) as slide:
            rgb = read_tile(slide, xy, self.selections[row['bag_id']]['source_size'], 256, max_read_pixels=1_000_000)
        image = Image.fromarray(rgb)
        if self.training:
            views = [self.augment(image) for _ in range(2)]
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(index)
                views = [self.augment(image) for _ in range(2)]
        return {'view1': views[0], 'view2': views[1], 'id': f"{row['bag_id']}:{index}"}

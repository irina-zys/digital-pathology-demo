"""Paired augmented WSI patch views for a short, label-free Lightning recipe."""

from pathlib import Path

import lightning as L
from PIL import Image
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.wsi import WSIBagDataModule, file_sha256
from src.models.pretrained_encoders import PREPROCESSING
from src.tiling import read_tile


class PairedPatchDataset(Dataset):
    def __init__(self, samples, selections, *, training):
        self.samples = [[row] for group in samples for row in group for _ in selections[row['bag_id']]['coords']]
        self.entries = [(group[0], xy) for group in samples for xy in selections[group[0]['bag_id']]['coords']]
        self.selections, self.training = selections, training
        self.augment = transforms.Compose([transforms.RandomResizedCrop(224, scale=(.6, 1.0)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ColorJitter(.2, .2, .2, .02),
            transforms.ToTensor(), transforms.Normalize(PREPROCESSING['mean'], PREPROCESSING['std'])])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row, xy = self.entries[index]
        selection = self.selections[row['bag_id']]
        info = Path(row['wsi_path']).stat()
        if ((info.st_size, info.st_mtime_ns) != (selection['source_bytes'], selection['source_mtime_ns'])
                or file_sha256(row['coords_path']) != selection['coordinate_sha256']):
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


class WSIContrastiveDataModule(L.LightningDataModule):
    def __init__(self, manifest, data_root, class_names, patches_per_slide=2, batch_size=4, num_workers=0,
                 target_mpp=.5, task='self_supervised'):
        super().__init__()
        self.save_hyperparameters()
        if task != 'self_supervised' or not 2 <= patches_per_slide <= 8 or not 2 <= batch_size <= 16 or num_workers != 0:
            raise ValueError('Use SSL, 2-8 patches/slide, batch_size 2-16 and no loader workers')
        self.datasets = {}
        self.contract = {'task': task, 'patches_per_slide': patches_per_slide, 'labels_used': False,
                         'views': 'two_random_crop_flip_mild_jitter', 'target_mpp': target_mpp}

    def setup(self, stage=None):
        if stage in {'test', 'predict'}:
            raise ValueError('This short SSL recipe does not inspect test patches')
        if not self.datasets:
            # Construct outside this Lightning module's __init__ so hyperparameter introspection stays scoped.
            self.inner = WSIBagDataModule(manifest=self.hparams.manifest, data_root=self.hparams.data_root,
                task='mil', class_names=self.hparams.class_names, max_patches_per_bag=self.hparams.patches_per_slide,
                batch_size=1, num_workers=0, image_size=224, augment=False, target_mpp=self.hparams.target_mpp,
                max_pixels=1_000_000)
            self.inner.setup('fit')
            self.rows, self.selections = self.inner.rows, self.inner.selections
            self.datasets = {split: PairedPatchDataset(self.inner.datasets[split].samples, self.selections,
                              training=split=='train') for split in ('train', 'val')}
            if any(len(ds) < self.hparams.batch_size for ds in self.datasets.values()):
                raise ValueError('Each SSL split needs at least one complete contrastive batch')

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.hparams.batch_size, shuffle=False, drop_last=True)

"""Label-free paired-view lifecycle over explicit WSI splits."""
import numpy as np
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from src.data.datamodules.base import ManifestDataModule
from src.data.datasets.paired_patches import PairedPatchDataset
from src.data.coordinates import select_coordinates
from src.data.manifests import read_wsi_manifest


class WSIContrastiveDataModule(ManifestDataModule):
    def __init__(self, manifest, data_root, class_names, patches_per_slide=2, batch_size=4, num_workers=0,
                 target_mpp=.5, task='self_supervised'):
        super().__init__()
        self.save_hyperparameters()
        if task != 'self_supervised' or not 2 <= patches_per_slide <= 8 or not 2 <= batch_size <= 16 or num_workers != 0:
            raise ValueError('Use SSL, 2-8 patches/slide, batch_size 2-16 and no loader workers')
        self.datasets = {}
        self.contract = {'task': task, 'patches_per_slide': patches_per_slide, 'labels_used': False,
                         'views': 'two_random_crop_flip_mild_jitter', 'target_mpp': target_mpp}

        if not np.isfinite(target_mpp) or target_mpp <= 0:
            raise ValueError("target_mpp must be finite and positive")
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("Set ordered unique source class_names")

    def setup(self, stage=None):
        if stage in {"test", "predict"}:
            raise ValueError("This short SSL recipe does not inspect test patches")
        if not self.datasets:
            rows, samples = read_wsi_manifest(to_absolute_path(self.hparams.manifest),
                to_absolute_path(self.hparams.data_root), class_names=self.hparams.class_names)
            # Validate the entire manifest, but only prepare permitted SSL splits.
            selected = {split: samples[split] for split in ("train", "val")}
            selections = {sample[0]["bag_id"]: select_coordinates(sample[0], self.hparams.patches_per_slide,
                self.hparams.target_mpp, 1_000_000) for items in selected.values() for sample in items}
            datasets = {split: PairedPatchDataset(items, selections, training=split == "train")
                        for split, items in selected.items()}
            if any(len(ds) < self.hparams.batch_size for ds in datasets.values()):
                raise ValueError("Each SSL split needs at least one complete contrastive batch")
            self.rows, self.selections, self.datasets = rows, selections, datasets
            self.samples = {split: ds.samples for split, ds in datasets.items()}
        self.require_splits(stage)

    def _loader(self, split):
        return DataLoader(self.datasets[split], batch_size=self.hparams.batch_size,
                          shuffle=split == "train", drop_last=True, num_workers=0)

    def test_dataloader(self):
        raise ValueError("This short SSL recipe does not inspect test patches")

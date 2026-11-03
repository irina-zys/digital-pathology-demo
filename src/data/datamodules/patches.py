"""Lightning lifecycle for explicit patch manifests."""
from hydra.utils import to_absolute_path
from src.data.datamodules.base import ManifestDataModule
from src.data.datasets.patches import PatchDataset
from src.data.manifests import read_patch_manifest, limit_samples
from src.data.transforms import validate_image_settings


class PatchDataModule(ManifestDataModule):
    def __init__(self, manifest, data_root, task, class_names=None, batch_size=8,
                 num_workers=0, image_size=224, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), augment=True, max_pixels=4_000_000,
                 max_samples_per_split=None, max_patches_per_bag=64):
        super().__init__()
        self.save_hyperparameters()
        validate_image_settings(task=task, batch_size=batch_size, num_workers=num_workers,
            max_pixels=max_pixels, max_patches_per_bag=max_patches_per_bag, image_size=image_size,
            max_samples_per_split=max_samples_per_split, mean=mean, std=std, class_names=class_names)
        self.contract = dict(task=task, class_names=list(class_names or []), image_size=image_size,
                             mean=list(mean), std=list(std), mask_policy="nonzero_foreground_ignore_nonzero")
        self.datasets = {}
        self.rows = []

    def setup(self, stage=None):
        if not self.datasets:
            self.rows, samples = read_patch_manifest(to_absolute_path(self.hparams.manifest),
                to_absolute_path(self.hparams.data_root), task=self.hparams.task, class_names=self.hparams.class_names)
            self.samples = limit_samples(samples, self.hparams.max_samples_per_split)
            self.datasets = {split: PatchDataset(items, self.hparams.task, self.hparams.image_size,
                self.hparams.mean, self.hparams.std, self.hparams.augment and split == "train",
                self.hparams.max_pixels, self.hparams.max_patches_per_bag) for split, items in self.samples.items()}
        self.require_splits(stage)

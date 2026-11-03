"""Lightning lifecycle for calibrated WSI image bags."""
import numpy as np
from hydra.utils import to_absolute_path
from src.data.datamodules.base import ManifestDataModule
from src.data.datasets.wsi_bags import WSIBagDataset
from src.data.coordinates import select_coordinates
from src.data.manifests import read_wsi_manifest, limit_samples
from src.data.transforms import validate_image_settings


class WSIBagDataModule(ManifestDataModule):
    def __init__(self, manifest, data_root, task, class_names=None, batch_size=8,
                 num_workers=0, image_size=224, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), augment=True, max_pixels=4_000_000,
                 max_samples_per_split=None, max_patches_per_bag=64, target_mpp=0.5, max_slides=20):
        super().__init__()
        self.save_hyperparameters()
        validate_image_settings(task=task, batch_size=batch_size, num_workers=num_workers,
            max_pixels=max_pixels, max_patches_per_bag=max_patches_per_bag, image_size=image_size,
            max_samples_per_split=max_samples_per_split, mean=mean, std=std, class_names=class_names)
        self.contract = dict(task=task, class_names=list(class_names or []), image_size=image_size,
                             mean=list(mean), std=list(std), mask_policy="nonzero_foreground_ignore_nonzero")
        self.datasets = {}
        self.rows = []

        if self.hparams.task != "mil" or not np.isfinite(target_mpp) or target_mpp <= 0:
            raise ValueError("WSI coordinate input requires task=mil and positive target_mpp")
        if type(max_slides) is not int or not 1 <= max_slides <= 100:
            raise ValueError("max_slides must be an integer in 1–100")
        if self.hparams.max_patches_per_bag > 64 or self.hparams.num_workers != 0:
            raise ValueError("This bounded WSI reader permits up to 64 patches/bag and num_workers=0")
        self.target_mpp, self.max_slides = target_mpp, max_slides
        self.selections = {}
        self.contract.pop("mask_policy")
        self.contract.update(input_kind="wsi_coordinates_v1", target_mpp=target_mpp, output_patch_size=256,
                             selection="evenly_spaced_export_row_indices",
                             max_patches_per_bag=self.hparams.max_patches_per_bag)

    def setup(self, stage=None):
        if not self.datasets:
            rows, samples = read_wsi_manifest(to_absolute_path(self.hparams.manifest),
                to_absolute_path(self.hparams.data_root), class_names=self.hparams.class_names, max_slides=self.max_slides)
            samples = limit_samples(samples, self.hparams.max_samples_per_split)
            selections, datasets = {}, {}
            for split, items in samples.items():
                for sample in items:
                    row = sample[0]
                    selections[row["bag_id"]] = select_coordinates(row, self.hparams.max_patches_per_bag,
                        self.target_mpp, self.hparams.max_pixels)
                datasets[split] = WSIBagDataset(items, selections, self.hparams.image_size,
                    self.hparams.mean, self.hparams.std, self.hparams.augment and split == "train", self.hparams.max_pixels)
            self.rows, self.samples, self.selections, self.datasets = rows, samples, selections, datasets
        self.require_splits(stage)

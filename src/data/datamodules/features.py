"""Lightning lifecycle for verified complete feature bags."""
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from hydra.utils import to_absolute_path
from src.data.datamodules.base import ManifestDataModule
from src.data.datasets.feature_bags import FeatureBagDataset
from src.data.feature_cache import read_feature_cache
from src.data.manifests import read_feature_manifest
from src.utils.files import sha256


class FeatureBagDataModule(ManifestDataModule):
    def __init__(self, manifest, data_root, class_names, input_dim, encoder_sha256, task='mil',
                 batch_size=1, num_workers=0, seed=42):
        super().__init__()
        self.save_hyperparameters()
        if task != 'mil' or batch_size != 1 or num_workers != 0:
            raise ValueError('Complete feature bags require MIL, batch_size=1 and num_workers=0')
        self.datasets = {}
        self.contract = {'input_kind': 'complete_frozen_features', 'input_dim': input_dim,
                         'encoder_sha256': encoder_sha256,
                         'class_names': list(class_names), 'coverage': 'all_eligible_coordinates'}

        if len(class_names) < 2 or len(set(class_names)) != len(class_names) or input_dim < 1:
            raise ValueError("Set ordered unique class_names and a positive input_dim")

    def setup(self, stage=None):
        if not self.datasets:
            rows, samples = read_feature_manifest(to_absolute_path(self.hparams.manifest),
                to_absolute_path(self.hparams.data_root), class_names=self.hparams.class_names)
            arrays, preprocessing_contract = {}, None
            for row in rows:
                path = Path(row["cache_path"])
                if sha256(path) != row['cache_sha256']:
                    raise ValueError('Feature cache checksum mismatch')
                features, _, identity = read_feature_cache(path)
                cache_contract = {'preprocessing': identity['encoder'].get('preprocessing'),
                                  'normalization': identity.get('normalization_after_embedding')}
                if (not isinstance(cache_contract['preprocessing'], dict) or not cache_contract['preprocessing']
                        or cache_contract['normalization'] != 'l2_per_patch'):
                    raise ValueError('Feature cache must declare preprocessing and l2_per_patch normalization')
                if preprocessing_contract is not None and cache_contract != preprocessing_contract:
                    raise ValueError('Feature caches have different preprocessing/normalization contracts')
                preprocessing_contract = cache_contract
                if (features.shape != (int(row['eligible_patches']), self.hparams.input_dim)
                        or identity['encoder']['checkpoint_sha256'] != self.hparams.encoder_sha256
                        or identity['slide'] != row['bag_id']
                        or identity['coordinates_sha256'] != row['coordinate_sha256']):
                    raise ValueError('Feature, encoder, coverage or coordinate contract mismatch')
                arrays[row['bag_id']] = torch.from_numpy(features)
            self.contract.update(preprocessing_contract)
            self.rows, self.samples = rows, samples
            self.datasets = {split: FeatureBagDataset(items, arrays) for split, items in samples.items()}
        self.require_splits(stage)

    def _loader(self, split):
        # Match slide order across pooling architectures despite different initialization RNG draws.
        generator = torch.Generator().manual_seed(self.hparams.seed)
        return DataLoader(self.datasets[split], batch_size=1, shuffle=split == "train",
                          num_workers=0, generator=generator)

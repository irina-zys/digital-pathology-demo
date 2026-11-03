"""Lightning data module for complete, verified WSI feature bags."""

import csv
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from hydra.utils import to_absolute_path

from src.data.feature_bags import read_feature_cache
from src.models.pretrained_encoders import sha256


class FeatureBagDataset(Dataset):
    def __init__(self, samples, arrays):
        self.samples, self.arrays = samples, arrays

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index][0]
        return {'features': self.arrays[row['bag_id']], 'target': int(row['label']), 'id': row['bag_id']}


class FeatureBagDataModule(L.LightningDataModule):
    def __init__(self, manifest, data_root, class_names, input_dim, encoder_sha256, task='mil',
                 batch_size=1, num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        if task != 'mil' or batch_size != 1 or num_workers != 0:
            raise ValueError('Complete feature bags require MIL, batch_size=1 and num_workers=0')
        self.datasets = {}
        self.contract = {'input_kind': 'complete_frozen_features', 'input_dim': input_dim,
                         'encoder_sha256': encoder_sha256,
                         'class_names': list(class_names), 'coverage': 'all_eligible_coordinates'}

    def setup(self, stage=None):
        if not self.datasets:
            manifest = Path(to_absolute_path(self.hparams.manifest))
            if manifest.stat().st_size > 1_000_000:
                raise ValueError('Feature manifest exceeds 1 MB')
            with manifest.open(newline='') as stream:
                self.rows = list(csv.DictReader(stream))
            if not 1 <= len(self.rows) <= 20:
                raise ValueError('Feature demo expects 1-20 slides')
            groups, seen, arrays = {}, set(), {}
            preprocessing_contract = None
            split_rows = {split: [] for split in ('train', 'val', 'test')}
            for row in self.rows:
                for key in ('split', 'group_id', 'bag_id'):
                    if not row.get(key, '').strip():
                        raise ValueError('Feature manifest contains an empty required identifier')
                    row[key] = row[key].strip()
                if row['split'] not in split_rows or groups.setdefault(row['group_id'], row['split']) != row['split']:
                    raise ValueError('Invalid split or patient crossing splits')
                if row['bag_id'] in seen or not row['group_id'] or not 0 <= int(row['label']) < len(self.hparams.class_names):
                    raise ValueError('Duplicate bag, missing patient or invalid label')
                seen.add(row['bag_id'])
                path = (Path(to_absolute_path(self.hparams.data_root))/row['cache_path']).resolve(strict=True)
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
                split_rows[row['split']].append([row])
            self.contract.update(preprocessing_contract)
            self.datasets = {split: FeatureBagDataset(rows, arrays) for split, rows in split_rows.items()}
        required = {'fit': ('train', 'val'), 'validate': ('val',), 'test': ('test',), 'predict': ('test',)}
        for split in required.get(stage, ()):
            if not self.datasets[split]:
                raise ValueError(f'No {split} slides')

    def _loader(self, split):
        return DataLoader(self.datasets[split], batch_size=1, shuffle=split=='train', num_workers=0)

    def train_dataloader(self):
        return self._loader('train')

    def val_dataloader(self):
        return self._loader('val')

    def test_dataloader(self):
        return self._loader('test')

    def predict_dataloader(self):
        return self.test_dataloader()

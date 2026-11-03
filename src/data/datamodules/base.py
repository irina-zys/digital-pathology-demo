"""Shared manifest lifecycle; task-specific preparation stays in each datamodule."""
import lightning as L
from torch.utils.data import DataLoader


class ManifestDataModule(L.LightningDataModule):
    def require_splits(self, stage):
        required = {"fit": ("train", "val"), "validate": ("val",), "test": ("test",), "predict": ("test",)}
        for split in required.get(stage, ()):
            if not len(self.datasets[split]):
                raise ValueError(f"Stage {stage} requires an explicit nonempty {split} split")

    def provenance(self):
        result = {"selected_rows": [row for items in self.samples.values() for sample in items for row in sample],
                  "samples_by_split": {split: len(dataset) for split, dataset in self.datasets.items()}}
        if hasattr(self, "selections"):
            result["coordinate_selections"] = self.selections
        return result

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

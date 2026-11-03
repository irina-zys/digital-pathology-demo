"""Lightning image/MIL classifiers and a compact binary-segmentation U-Net."""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy

from src.models.encoders import ImageEncoder


class ForegroundDice(Metric):
    """Dataset-level Dice over valid pixels; empty prediction and target score 1."""
    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, prediction, target, valid):
        prediction, target = prediction & valid, target & valid
        self.intersection += (prediction & target).sum()
        self.size += prediction.sum() + target.sum()

    def compute(self):
        return torch.where(self.size > 0, 2.0 * self.intersection / self.size.clamp_min(1), 1.0)


class VisionModule(L.LightningModule):
    def configure_optimizers(self):
        return torch.optim.AdamW((p for p in self.parameters() if p.requires_grad),
                                 lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx):
        logits = self(batch["image"])
        probability = logits.sigmoid() if self.task == "segmentation" else logits.softmax(-1)
        return {"id": batch["id"], "probability": probability}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_contract"] = self.contract
        checkpoint["data_contract"] = self.trainer.datamodule.contract

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("model_contract") != self.contract:
            raise ValueError("Checkpoint model/label contract differs from the configured model")
        if self._trainer is not None and self.trainer.datamodule is not None:
            if checkpoint.get("data_contract") != self.trainer.datamodule.contract:
                raise ValueError("Checkpoint data contract differs (task/classes/image size/normalization)")


class Classifier(VisionModule):
    def __init__(self, class_names, encoder, pooling=None, attention_dim=128,
                 lr=0.0001, weight_decay=0.0001):
        super().__init__()
        self.save_hyperparameters()
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("Set at least two unique, ordered class_names")
        if pooling not in {None, "mean", "attention"}:
            raise ValueError("pooling must be null (patch classification), mean or attention")
        if attention_dim < 1 or lr <= 0 or weight_decay < 0:
            raise ValueError("Invalid attention/optimizer settings")
        self.task = "mil" if pooling else "classification"
        self.encoder = ImageEncoder(**encoder)
        features = self.encoder.num_features
        self.head = nn.Linear(features, len(class_names))
        self.attention = (nn.Sequential(nn.Linear(features, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))
                          if pooling == "attention" else None)
        self.accuracy = nn.ModuleDict({f"{stage}_metric": MulticlassAccuracy(num_classes=len(class_names), average="micro")
                                      for stage in ("train", "val", "test")})
        self.contract = dict(task=self.task, class_names=list(class_names), encoder=encoder["name"], pooling=pooling)

    def forward(self, images):
        if self.task == "mil":
            if images.ndim != 5 or images.shape[0] != 1 or images.shape[1] == 0:
                raise ValueError("MIL expects one nonempty bag: [1, patches, 3, height, width]")
            features = self.encoder(images[0])
            if self.attention is None:
                pooled = features.mean(0, keepdim=True)
            else:
                weights = self.attention(features).softmax(dim=0)
                pooled = (weights * features).sum(0, keepdim=True)
        else:
            pooled = self.encoder(images)
        return self.head(pooled)

    def _step(self, batch, stage):
        logits = self(batch["image"])
        loss = F.cross_entropy(logits, batch["target"])
        self.accuracy[f"{stage}_metric"].update(logits, batch["target"])
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True,
                 batch_size=len(batch["target"]), sync_dist=True)
        self.log(f"{stage}/accuracy", self.accuracy[f"{stage}_metric"], on_step=False, on_epoch=True)
        return loss


def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(1, out_channels), nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(1, out_channels), nn.ReLU())


class UNet(VisionModule):
    """Two-level U-Net predicting nuclear foreground, not individual instances."""
    task = "segmentation"

    def __init__(self, base_channels=16, threshold=0.5, lr=0.001, weight_decay=0.0001):
        super().__init__()
        self.save_hyperparameters()
        if base_channels < 1 or not 0 < threshold < 1 or lr <= 0 or weight_decay < 0:
            raise ValueError("Invalid U-Net/optimizer settings")
        self.down1 = conv_block(3, base_channels)
        self.down2 = conv_block(base_channels, 2 * base_channels)
        self.bottom = conv_block(2 * base_channels, 4 * base_channels)
        self.up2 = conv_block(6 * base_channels, 2 * base_channels)
        self.up1 = conv_block(3 * base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 1, 1)
        self.dice = nn.ModuleDict({f"{stage}_metric": ForegroundDice() for stage in ("train", "val", "test")})
        self.contract = dict(task=self.task, base_channels=base_channels, threshold=threshold)

    def forward(self, images):
        first = self.down1(images)
        second = self.down2(F.max_pool2d(first, 2))
        bottom = self.bottom(F.max_pool2d(second, 2))
        up = self.up2(torch.cat([F.interpolate(bottom, size=second.shape[-2:], mode="bilinear", align_corners=False), second], 1))
        up = self.up1(torch.cat([F.interpolate(up, size=first.shape[-2:], mode="bilinear", align_corners=False), first], 1))
        return self.head(up)

    def _step(self, batch, stage):
        logits = self(batch["image"])
        target, valid = batch["target"], batch["valid"].bool()
        pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        counts = valid.flatten(1).sum(1)
        if (counts == 0).any():
            raise ValueError("Each mask must contain evaluable pixels")
        loss = ((pixel_loss * valid).flatten(1).sum(1) / counts).mean()
        self.dice[f"{stage}_metric"].update(logits.sigmoid() >= self.hparams.threshold, target.bool(), valid)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, batch_size=len(target), sync_dist=True)
        self.log(f"{stage}/dice", self.dice[f"{stage}_metric"], on_step=False, on_epoch=True)
        return loss

"""Paired-view contrastive training; labels do not enter the objective."""
import torch
from src.lightning.base import BaseModule


class ContrastiveModule(BaseModule):
    def __init__(self, net, loss, optimizer):
        super().__init__(net, loss, optimizer)
        self.contract = {"task": self.task, "net": net.contract}

    def _step(self, batch, stage):
        n = len(batch["view1"])
        features = self(torch.cat((batch["view1"], batch["view2"])))
        loss = self.loss(features[:n], features[n:])
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, batch_size=n)
        return loss

    def predict_step(self, batch, batch_idx):
        raise ValueError("This short SSL recipe does not support prediction; transfer its encoder to a classifier")

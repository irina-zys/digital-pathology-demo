"""Classification lifecycle shared by image and frozen-feature networks."""
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from src.lightning.base import BaseModule


class ClassificationModule(BaseModule):
    def __init__(self, net, loss, optimizer):
        super().__init__(net, loss, optimizer)
        self.class_names = list(net.class_names)
        self.input_key = net.input_key
        self.accuracy = nn.ModuleDict({f"{stage}_metric": MulticlassAccuracy(num_classes=len(self.class_names), average="micro")
                                      for stage in ("train", "val", "test")})
        self.contract = {"task": self.task, "class_names": self.class_names, "net": net.contract}

    def validate_data(self, data):
        super().validate_data(data)
        if self.class_names != list(data.hparams.class_names):
            raise ValueError("Model and data class_names differ")
        feature_input = data.contract.get("input_kind") == "complete_frozen_features"
        if feature_input != (self.input_key == "features"):
            raise ValueError("Model and data input kinds differ")
        if feature_input and data.contract["input_dim"] != self.net.contract["input_dim"]:
            raise ValueError("Model and data feature dimensions differ")

    def _step(self, batch, stage):
        logits = self(batch[self.input_key])
        loss = self.loss(logits, batch["target"])
        self.accuracy[f"{stage}_metric"].update(logits, batch["target"])
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, batch_size=len(batch["target"]), sync_dist=True)
        self.log(f"{stage}/accuracy", self.accuracy[f"{stage}_metric"], on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        return {"id": batch["id"], "probability": self(batch[self.input_key]).softmax(-1)}

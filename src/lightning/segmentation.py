"""Binary foreground training with ignored-pixel semantics."""
from torch import nn
from src.evaluation.metrics.foreground import ForegroundDice
from src.lightning.base import BaseModule


class SegmentationModule(BaseModule):
    def __init__(self, net, loss, optimizer, threshold=.5):
        super().__init__(net, loss, optimizer)
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        self.threshold = threshold
        self.dice = nn.ModuleDict({f"{stage}_metric": ForegroundDice() for stage in ("train", "val", "test")})
        self.contract = {"task": self.task, "net": net.contract, "threshold": threshold}

    def _step(self, batch, stage):
        logits = self(batch["image"])
        target, valid = batch["target"], batch["valid"].bool()
        loss = self.loss(logits, target, valid)
        self.dice[f"{stage}_metric"].update(logits.sigmoid() >= self.threshold, target.bool(), valid)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, batch_size=len(target), sync_dist=True)
        self.log(f"{stage}/dice", self.dice[f"{stage}_metric"], on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        return {"id": batch["id"], "probability": self(batch["image"]).sigmoid()}

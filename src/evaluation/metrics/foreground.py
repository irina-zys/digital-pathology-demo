"""Dataset-level binary foreground Dice over valid pixels."""
import torch
from torchmetrics import Metric


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

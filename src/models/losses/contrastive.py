"""Paired positives and within-batch negatives, excluding self-pairs."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def nt_xent(first, second, temperature=.2):
    """Two views per patch; exclude self-pairs, retain augmented positives and batch negatives."""
    if first.shape != second.shape or first.ndim != 2 or len(first) < 2 or temperature <= 0:
        raise ValueError('Expected paired batches of at least two examples and positive temperature')
    n = len(first)
    z = F.normalize(torch.cat((first, second)), dim=1)
    logits = (z @ z.T / temperature).masked_fill(torch.eye(2*n, device=z.device, dtype=torch.bool), float('-inf'))
    targets = (torch.arange(2*n, device=z.device) + n) % (2*n)
    return F.cross_entropy(logits, targets)


class NTXent(nn.Module):
    def __init__(self, temperature=.2):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature
        self.contract = {"kind": "nt_xent", "temperature": temperature}

    def forward(self, first, second):
        return nt_xent(first, second, self.temperature)

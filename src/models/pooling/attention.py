"""Tanh attention normalized across patches within one bag."""
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, attention_dim=128):
        super().__init__()
        if input_dim < 1 or attention_dim < 1:
            raise ValueError("Pooling dimensions must be positive")
        self.input_dim = input_dim
        self.attention = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))
        self.contract = {"kind": "attention", "attention_dim": attention_dim}

    def forward(self, features):
        if features.ndim != 2 or not len(features) or features.shape[1] != self.input_dim:
            raise ValueError("Expected a nonempty [patches, input_dim] bag")
        weights = self.attention(features).softmax(dim=0)
        return (weights * features).sum(0, keepdim=True), weights[:, 0]

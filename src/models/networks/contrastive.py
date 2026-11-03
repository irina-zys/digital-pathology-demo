"""Encoder and projection head for paired-view representation learning."""
from torch import nn


class ContrastiveNetwork(nn.Module):
    task = "self_supervised"

    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        if projection_dim < 1:
            raise ValueError("projection_dim must be positive")
        self.encoder = encoder
        self.projector = nn.Sequential(nn.Linear(encoder.num_features, 256), nn.ReLU(), nn.Linear(256, projection_dim))
        self.contract = {"kind": "contrastive", "encoder": encoder.contract, "projection_dim": projection_dim}

    def forward(self, images):
        return self.projector(self.encoder(images))

"""Image encoders provided by the already-locked timm dependency."""

import timm
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, name="resnet18", pretrained=False, freeze=False):
        super().__init__()
        if name not in {"resnet18", "resnet50", "vit_small_patch16_224.dino"}:
            raise ValueError(f"Unsupported encoder: {name}")
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.num_features = self.net.num_features
        self.frozen = freeze
        if freeze:
            self.net.requires_grad_(False)
            self.net.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.net.eval()
        return self

    def forward(self, images):
        return self.net(images)

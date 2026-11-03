"""Two-level binary-foreground U-Net, independent of training."""
import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(1, out_channels), nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(1, out_channels), nn.ReLU())


class UNet(nn.Module):
    """Two-level U-Net predicting nuclear foreground, not individual instances."""
    task = "segmentation"

    def __init__(self, base_channels=16):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        self.down1 = conv_block(3, base_channels)
        self.down2 = conv_block(base_channels, 2 * base_channels)
        self.bottom = conv_block(2 * base_channels, 4 * base_channels)
        self.up2 = conv_block(6 * base_channels, 2 * base_channels)
        self.up1 = conv_block(3 * base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 1, 1)
        self.contract = {"kind": "unet", "base_channels": base_channels}

    def forward(self, images):
        first = self.down1(images)
        second = self.down2(F.max_pool2d(first, 2))
        bottom = self.bottom(F.max_pool2d(second, 2))
        up = self.up2(torch.cat([F.interpolate(bottom, size=second.shape[-2:], mode="bilinear", align_corners=False), second], 1))
        up = self.up1(torch.cat([F.interpolate(up, size=first.shape[-2:], mode="bilinear", align_corners=False), first], 1))
        return self.head(up)

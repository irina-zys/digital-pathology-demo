"""Binary BCE averaged within valid image pixels, then across images."""
from torch import nn
from torch.nn import functional as F


class ValidPixelBCE(nn.Module):
    contract = {"kind": "valid_pixel_bce", "reduction": "valid_pixels_per_image_then_images"}

    def forward(self, logits, target, valid):
        valid = valid.bool()
        pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        counts = valid.flatten(1).sum(1)
        if (counts == 0).any():
            raise ValueError("Each mask must contain evaluable pixels")
        return ((pixel_loss * valid).flatten(1).sum(1) / counts).mean()

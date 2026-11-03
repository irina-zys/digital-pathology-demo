"""Image encoders provided by the already-locked timm dependency."""

import timm
import torch
from pathlib import Path
from torchvision.models import resnet50
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, name="resnet18", pretrained=False, freeze=False, *, local_project=None,
                 freeze_norm_stats=False, source_checkpoint=None, source_checkpoint_sha256=None):
        super().__init__()
        if name not in {"resnet18", "resnet50", "vit_small_patch16_224.dino"}:
            raise ValueError(f"Unsupported encoder: {name}")
        if local_project and name in {"resnet50", "vit_small_patch16_224.dino"}:
            from src.models.encoders.pretrained import load_pretrained_encoder
            if pretrained:
                loaded = load_pretrained_encoder("dino" if name.endswith(".dino") else "resnet50", local_project)
                self.net, self.num_features = loaded['model'], loaded['metadata']['features']
                self.net.requires_grad_(True)
            elif name == "resnet50":
                self.net = resnet50(weights=None)
                self.net.fc = nn.Identity()
                self.num_features = 2048
            else:
                self.net = timm.create_model(name, pretrained=False, num_classes=0)
                self.num_features = self.net.num_features
        else:
            self.net = timm.create_model(name, pretrained=pretrained, num_classes=0)
            self.num_features = self.net.num_features
        if source_checkpoint:
            from src.utils.files import sha256
            path = Path(source_checkpoint).expanduser().resolve(strict=True)
            if pretrained or name != "resnet50" or not local_project:
                raise ValueError("Local SSL transfer requires an uninitialized local ResNet50 architecture")
            if not source_checkpoint_sha256 or sha256(path) != source_checkpoint_sha256:
                raise ValueError("SSL initialization checkpoint checksum mismatch")
            state = torch.load(path, map_location="cpu", weights_only=True)
            from src.models.checkpoints import require_checkpoint_version
            require_checkpoint_version(state)
            contract = state.get("model_contract", {})
            if (contract.get("task") != "self_supervised"
                    or contract.get("net", {}).get("encoder", {}).get("architecture") != "torchvision_resnet50"):
                raise ValueError("Unexpected SSL checkpoint architecture")
            encoder_state = {key.removeprefix("net.encoder.net."): value for key, value in state["state_dict"].items()
                             if key.startswith("net.encoder.net.")}
            self.net.load_state_dict(encoder_state, strict=True)
        self.freeze_norm_stats = freeze_norm_stats
        self.frozen = freeze
        self.required_input_size = 224 if name == "vit_small_patch16_224.dino" else None
        self.contract = {"name": name, "architecture": "torchvision_resnet50" if local_project and name == "resnet50" else "timm_" + name,
                         "num_features": self.num_features, "freeze": freeze,
                         "freeze_norm_stats": freeze_norm_stats, "required_input_size": self.required_input_size}
        if freeze:
            self.net.requires_grad_(False)
            self.net.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.net.eval()
        if self.freeze_norm_stats:
            for module in self.net.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, images):
        return self.net(images)

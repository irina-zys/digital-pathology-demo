"""Pinned frozen encoder weights, independent of data preprocessing."""
from pathlib import Path
import timm
import torch
from torchvision.models import resnet50
from src.models.pretrained.assets import DINO_REVISION, DINO_SHA256, RESNET_SHA256, UNI_REVISION, UNI_SHA256
from src.utils.files import sha256


def load_pretrained_encoder(name, project, *, device='cpu', allow_download=False):
    """Load verified frozen weights. Explicit allow_download applies to gated UNI only.

    Authentication uses the standard local Hugging Face login. Cache location is
    explicit and does not relocate or read authentication tokens into the project.
    Public DINO/ResNet weights must already exist under data/models.
    """
    project = Path(project).resolve()
    root = project/'data/models'
    if device not in {'cpu', 'mps', 'cuda'}:
        raise ValueError('Choose cpu, mps or cuda')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Apple GPU unavailable in this process; choose CPU explicitly')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if name == 'dino':
        from safetensors.torch import load_file
        checkpoint = root/f'huggingface/hub/models--timm--vit_small_patch16_224.dino/snapshots/{DINO_REVISION}/model.safetensors'
        expected, features, domain = DINO_SHA256, 384, 'ImageNet-1k, self-supervised DINO'
        if sha256(checkpoint) != expected:
            raise ValueError('DINO weight checksum mismatch')
        model = timm.create_model('vit_small_patch16_224.dino', pretrained=False, num_classes=0)
        model.load_state_dict(load_file(str(checkpoint)), strict=True)
    elif name == 'resnet50':
        checkpoint = root/'resnet50/resnet50-11ad3fa6.pth'
        expected, features, domain = RESNET_SHA256, 2048, 'ImageNet-1k, supervised IMAGENET1K_V2'
        if sha256(checkpoint) != expected:
            raise ValueError('ResNet weight checksum mismatch')
        model = resnet50(weights=None)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        model.fc = torch.nn.Identity()
    elif name == 'uni':
        from huggingface_hub import hf_hub_download
        cache = root/'huggingface/hub'
        checkpoint = Path(hf_hub_download('MahmoodLab/UNI', 'pytorch_model.bin', revision=UNI_REVISION,
                          cache_dir=cache, local_files_only=not allow_download))
        hf_hub_download('MahmoodLab/UNI', 'config.json', revision=UNI_REVISION,
                        cache_dir=cache, local_files_only=not allow_download)
        expected, features, domain = UNI_SHA256, 1024, 'Histopathology, original UNI'
        if sha256(checkpoint) != expected:
            raise ValueError('UNI weight checksum mismatch')
        model = timm.create_model(f'hf-hub:MahmoodLab/UNI@{UNI_REVISION}', pretrained=True,
                                  cache_dir=cache, init_values=1e-5, dynamic_img_size=True)
    else:
        raise ValueError('Choose dino, resnet50 or uni')
    model.eval().requires_grad_(False).to(device)
    metadata = {'name': name, 'features': features, 'training_domain': domain,
                'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': expected,
                'torch': torch.__version__, 'timm': timm.__version__,
                'device': device, 'frozen': True}
    return {'model': model, 'metadata': metadata, 'device': device}

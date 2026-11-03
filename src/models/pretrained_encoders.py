"""Pinned pretrained encoders for matched WSI comparisons; UNI uses TIMM's Hub loader."""

import hashlib
from pathlib import Path

import timm
import torch
from torchvision import transforms
from torchvision.models import resnet50

from src.pretrained import UNI_REVISION, UNI_SHA256

DINO_REVISION = '10e440b8a34dfd657a90f2fcaa988c6b6a2f0da4'
DINO_SHA256 = '8d4795b8e4e327c6d841e15a2e3e6d88c45346256a43f8cdae8c26d92c806f7a'
RESNET_SHA256 = '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
PREPROCESSING = {'version': 1, 'tile_size': 256, 'model_size': 224, 'resize': 'bilinear_antialias_square',
                 'mean': [.485, .456, .406], 'std': [.229, .224, .225],
                 'note': 'Shared full-tile resize for comparison; no model-specific center crop or stain normalization'}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


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
    transform = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor(),
                                    transforms.Normalize(PREPROCESSING['mean'], PREPROCESSING['std'])])
    metadata = {'name': name, 'features': features, 'training_domain': domain,
                'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': expected,
                'preprocessing': PREPROCESSING, 'torch': torch.__version__, 'timm': timm.__version__,
                'device': device, 'frozen': True}
    return {'model': model, 'transform': transform, 'metadata': metadata, 'device': device}


def prepare_encoder_assets(name, project):
    """Explicit acquisition entrypoint; standard local HF authentication is used unchanged."""
    from huggingface_hub import hf_hub_download
    import tempfile
    import urllib.request

    root = Path(project).resolve()/'data/models'
    if name in {'dino', 'uni'}:
        repo = 'timm/vit_small_patch16_224.dino' if name == 'dino' else 'MahmoodLab/UNI'
        revision = DINO_REVISION if name == 'dino' else UNI_REVISION
        filename = 'model.safetensors' if name == 'dino' else 'pytorch_model.bin'
        expected = DINO_SHA256 if name == 'dino' else UNI_SHA256
        checkpoint = Path(hf_hub_download(repo, filename, revision=revision, cache_dir=root/'huggingface/hub'))
        if name == 'uni':
            hf_hub_download(repo, 'config.json', revision=revision, cache_dir=root/'huggingface/hub')
    elif name == 'resnet50':
        checkpoint, expected = root/'resnet50/resnet50-11ad3fa6.pth', RESNET_SHA256
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.exists():
            with tempfile.NamedTemporaryFile(dir=checkpoint.parent, suffix='.part', delete=False) as stream:
                partial = Path(stream.name)
                try:
                    with urllib.request.urlopen('https://download.pytorch.org/models/resnet50-11ad3fa6.pth', timeout=60) as response:
                        count = 0
                        while block := response.read(1024*1024):
                            count += len(block)
                            if count > 110_000_000:
                                raise ValueError('ResNet download exceeded its size bound')
                            stream.write(block)
                    stream.flush()
                    if sha256(partial) != expected or checkpoint.exists():
                        raise ValueError('Checkpoint verification failed or another download appeared')
                    partial.rename(checkpoint)
                finally:
                    partial.unlink(missing_ok=True)
    else:
        raise ValueError('Choose uni, dino or resnet50')
    if sha256(checkpoint) != expected:
        raise ValueError('Preserving checkpoint with an unexpected checksum')
    return {'name': name, 'checkpoint': str(checkpoint.resolve()), 'sha256': expected, 'bytes': checkpoint.stat().st_size}

"""Pinned weight identities and explicit acquisition; inference does not call preparation."""
from pathlib import Path
from src.utils.files import sha256

HOVER_NAME = "pannuke_convnextv2_tiny_1"
HOVER_URL = f"https://zenodo.org/records/10635618/files/{HOVER_NAME}.zip"
HOVER_SIZE = 133721861
HOVER_MD5 = "331ce1f821b57e69e598473cc4f71769"
HOVER_SHA256 = "fc92a7807e21ad73f18dc7b41c9c13674a53383cb383561e9b2481f74a2578fa"
HOVER_FILES = {
    "params.toml": (1106, "ae6ac1a2ed23e89a935a863a647978ebb3d4c84eea02ca070ca91f304125c84c"),
    "pannuke_test_param_dict.json": (458, "2032d371e0c17c179f31f459696e2d5a08fa0928bc4b47bb932e7cd843b73cc3"),
    "train/best_model": (144433761, "0c75821cfc2a0a49f7f75ebb57a4e4842f9ddd67798841180a6e3d8091261ed0"),
}
CELLPOSE_URL = "https://www.cellpose.org/models/nucleitorch_0"
CELLPOSE_SIZE = 26563614
CELLPOSE_SHA256 = "89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303"
CELLPOSE_SIZE_MODEL_NAME = "size_nucleitorch_0.npy"
CELLPOSE_SIZE_MODEL_URL = f"https://www.cellpose.org/models/{CELLPOSE_SIZE_MODEL_NAME}"
CELLPOSE_SIZE_MODEL_BYTES = 3627
CELLPOSE_SIZE_MODEL_SHA256 = "a79107c9f284b569dd65f3a7363014d613a3403241c062822f07b6f026b4d3ef"
# Original UNI, not UNI2.
UNI_REPO = "MahmoodLab/UNI"
UNI_REVISION = "b55a5ec6cade1a39edfe6534189a9b8ca7a022f0"
UNI_FILE = "pytorch_model.bin"
UNI_SIZE = 1213527781
UNI_SHA256 = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"
DINO_REPO = "timm/vit_small_patch16_224.dino"
DINO_REVISION = "10e440b8a34dfd657a90f2fcaa988c6b6a2f0da4"
DINO_SIZE = 86676340
DINO_SHA256 = "8d4795b8e4e327c6d841e15a2e3e6d88c45346256a43f8cdae8c26d92c806f7a"

RESNET_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"


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

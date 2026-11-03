"""Explicit acquisition of pinned pretrained models under project data/models."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data/models"
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
# Original UNI pins match src.pretrained; they identify UNI, not UNI2.
UNI_REPO = "MahmoodLab/UNI"
UNI_REVISION = "b55a5ec6cade1a39edfe6534189a9b8ca7a022f0"
UNI_FILE = "pytorch_model.bin"
UNI_SIZE = 1213527781
UNI_SHA256 = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"
DINO_REPO = "timm/vit_small_patch16_224.dino"
DINO_REVISION = "10e440b8a34dfd657a90f2fcaa988c6b6a2f0da4"
DINO_SIZE = 86676340
DINO_SHA256 = "8d4795b8e4e327c6d841e15a2e3e6d88c45346256a43f8cdae8c26d92c806f7a"


def canonical(path):
    if path.is_symlink() or not path.resolve().is_relative_to(MODELS):
        raise ValueError(f"Model path must stay under canonical data/models: {path.name}")
    return path


def verify(path, size, sha256, md5=None):
    canonical(path)
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Preserving mismatched existing asset: {path}")
    sha, old_hash = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
            old_hash.update(block)
    if sha.hexdigest() != sha256 or (md5 is not None and old_hash.hexdigest() != md5):
        raise ValueError(f"Preserving asset with mismatched checksum: {path}")


def publish_stream(path, stream, size, sha256, md5=None):
    """Write one bounded, owned partial and publish only after full verification."""
    path = canonical(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = canonical(path.with_name(path.name + ".part"))
    # Exclusive creation preserves any unrelated partial left by another process.
    output = partial.open("xb")
    try:
        with output:
            count = 0
            while block := stream.read(1024 * 1024):
                count += len(block)
                if count > size:
                    raise ValueError("Model transfer exceeded its pinned byte size")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        verify(partial, size, sha256, md5)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Preserving asset created during download: {path}")
        partial.rename(path)
    finally:
        partial.unlink(missing_ok=True)


def download(path, url, size, sha256, md5=None):
    path = canonical(path)
    if path.exists():
        verify(path, size, sha256, md5)
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != size:
            raise ValueError("Server size differs from the pinned checkpoint")
        publish_stream(path, response, size, sha256, md5)


def receipt(path, values):
    canonical(path)
    if path.exists():
        return  # Preserve the original acquisition evidence on offline reuse.
    values = {**values, "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    with path.open("x") as stream:
        json.dump(values, stream, indent=2)
        stream.write("\n")


def hovernext():
    folder = MODELS / "hovernext"
    archive = folder / f"{HOVER_NAME}.zip"
    # Check all existing members before acquiring anything or adding missing files.
    for name, (size, digest) in HOVER_FILES.items():
        path = canonical(folder / HOVER_NAME / name)
        if path.exists():
            verify(path, size, digest)
    if all((folder / HOVER_NAME / name).is_file() for name in HOVER_FILES):
        if archive.exists():
            verify(archive, HOVER_SIZE, HOVER_SHA256, HOVER_MD5)
        return folder / HOVER_NAME
    download(archive, HOVER_URL, HOVER_SIZE, HOVER_SHA256, HOVER_MD5)
    allowed = {f"{HOVER_NAME}/{name}": values for name, values in HOVER_FILES.items()}
    directories = {f"{HOVER_NAME}/", f"{HOVER_NAME}/train/"}
    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        if len(infos) != len(allowed) + len(directories) or len({i.filename for i in infos}) != len(infos):
            raise ValueError("Unexpected checkpoint archive members")
        for info in infos:
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Checkpoint archive contains a symlink")
            if info.is_dir():
                if info.filename not in directories:
                    raise ValueError("Unexpected checkpoint archive directory")
            elif info.filename not in allowed or info.file_size != allowed[info.filename][0]:
                raise ValueError("Unexpected checkpoint archive file or size")
        for name, (size, digest) in allowed.items():
            path = canonical(folder / name)
            if not path.exists():
                with zipped.open(name) as stream:
                    publish_stream(path, stream, size, digest)
    receipt(folder / "download.json", {
        "model_code": HOVER_NAME, "source_url": HOVER_URL, "license": "CC-BY-NC-SA-4.0",
        "archive": archive.name, "archive_bytes": HOVER_SIZE, "archive_sha256": HOVER_SHA256,
        "published_md5": HOVER_MD5, "published_md5_verified": True,
        "extracted_files": [{"path": name, "bytes": size, "sha256": digest}
                            for name, (size, digest) in allowed.items()],
    })
    return folder / HOVER_NAME


def cellpose():
    path = MODELS / "cellpose/nucleitorch_0"
    download(path, CELLPOSE_URL, CELLPOSE_SIZE, CELLPOSE_SHA256)
    receipt(path.parent / "provenance.json", {
        "model": "Cellpose classical nuclei", "cellpose_version": "3.1.1.2", "checkpoint": path.name,
        "source_url": CELLPOSE_URL, "bytes": CELLPOSE_SIZE, "sha256": CELLPOSE_SHA256,
        "sha256_source": "pinned previously verified local acquisition; no upstream checksum supplied",
        "weights_notice": "Official release notes state nuclei training includes CC-BY-NC data; preserve model notices.",
        "release_notice": "https://github.com/MouseLand/cellpose/blob/v3.1.1.2/README.md",
    })
    return path


def uni():
    path = canonical(MODELS / "uni" / UNI_FILE)
    if path.exists():
        verify(path, UNI_SIZE, UNI_SHA256)
        return path
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError

    try:
        info = get_hf_file_metadata(hf_hub_url(UNI_REPO, UNI_FILE, revision=UNI_REVISION), token=True, timeout=20)
        if info.size != UNI_SIZE or info.etag != UNI_SHA256:
            raise ValueError("UNI server metadata differs from the pinned weights")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Hugging Face owns its locked resume cache, including interrupted transfers.
        downloaded = Path(hf_hub_download(repo_id=UNI_REPO, filename=UNI_FILE, revision=UNI_REVISION,
                                         local_dir=path.parent, token=True, etag_timeout=20))
        if downloaded.resolve() != path.resolve():
            raise RuntimeError("UNI download returned a noncanonical path")
        verify(path, UNI_SIZE, UNI_SHA256)
    except (GatedRepoError, LocalTokenNotFoundError) as error:
        raise RuntimeError("UNI requires provider-approved access and an existing Hugging Face login; no access restriction was bypassed") from error
    except HfHubHTTPError as error:
        raise RuntimeError("Hugging Face could not provide the pinned UNI checkpoint; check authorized access or connectivity") from error
    receipt(path.parent / "download.json", {"model": "UNI", "repo_id": UNI_REPO, "revision": UNI_REVISION,
                                            "checkpoint": UNI_FILE, "bytes": UNI_SIZE, "sha256": UNI_SHA256})
    return path


def dino():
    """Populate the pinned project Hugging Face cache without duplicating weights."""
    folder = MODELS / "huggingface/hub/models--timm--vit_small_patch16_224.dino"
    blob = canonical(folder / "blobs" / DINO_SHA256)
    reference = canonical(folder / "refs/main")
    snapshot = folder / "snapshots" / DINO_REVISION / "model.safetensors"
    if reference.exists() and reference.read_text().strip() != DINO_REVISION:
        raise ValueError("Preserving DINO cache with a different main revision")
    if snapshot.exists() or snapshot.is_symlink():
        if not snapshot.is_file() or not snapshot.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Preserving invalid or escaping DINO checkpoint link")
        verify(snapshot.resolve(), DINO_SIZE, DINO_SHA256)
        # Reuse a regular snapshot file too, rather than creating a second weight copy.
    else:
        url = f"https://huggingface.co/{DINO_REPO}/resolve/{DINO_REVISION}/model.safetensors"
        download(blob, url, DINO_SIZE, DINO_SHA256)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        canonical(snapshot)
        snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))
    if not reference.exists():
        reference.parent.mkdir(parents=True, exist_ok=True)
        with reference.open("x") as stream:
            stream.write(DINO_REVISION + "\n")
    receipt(folder / "download.json", {"model": "ImageNet DINO ViT-S/16", "repo_id": DINO_REPO,
        "revision": DINO_REVISION, "checkpoint": str(snapshot.relative_to(MODELS)),
        "bytes": DINO_SIZE, "sha256": DINO_SHA256})
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("hovernext", "cellpose", "uni", "dino"), required=True)
    args = parser.parse_args()
    try:
        path = {"hovernext": hovernext, "cellpose": cellpose, "uni": uni, "dino": dino}[args.model]()
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Verified {args.model}: {path}")


if __name__ == "__main__":
    main()

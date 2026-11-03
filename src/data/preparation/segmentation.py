"""Acquire pinned segmentation releases without replacing existing source files."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
from urllib.parse import quote
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[3]
CHUNK_BYTES = 1024 * 1024
DATASETS = {
    "cytonuke": {
        "record_id": 15310323,
        "archive": "CytoNuke Dataset.zip",
        "files": {
            "CytoNuke Dataset.zip": (12_827_112, "ef317b9ad26414c391e8761f46cdad31"),
            "README": (927, "f93814b2903193acc9951837544f018f"),
            "LICENSE": (18_656, "527dc6cad772ccb187d5bfe5af738204"),
        },
        "max_members": 100,
        "max_extracted_bytes": 50_000_000,
        "max_member_bytes": 40_000_000,
    },
    "nuinsseg": {
        "record_id": 10518968,
        "archive": "NuInsSeg.zip",
        "files": {"NuInsSeg.zip": (1_627_427_342, "08f9a33e23e983d82ee93aa58e69d095")},
        "max_members": 50_000,
        "max_extracted_bytes": 2_000_000_000,
        "max_member_bytes": 2_000_000,
        "demo_members": (
            "human bladder/tissue images/human_bladder_01.png",
            "human bladder/label masks modify/human_bladder_01.tif",
            "human bladder/vague areas/mask binary/human_bladder_01.png",
        ),
    },
}


def checked_path(path, root):
    """Keep source and extracted paths beneath their root; reject existing symlinks."""
    path, root = Path(path).absolute(), Path(root).absolute()
    path.relative_to(root)
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Refusing symlink in dataset path: {part}")
        if part == root:
            break
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path leaves its root: {path}")
    return path


def verify_file(path, expected_size, expected_md5):
    """Verify the publisher's fixed byte count and MD5 without loading the file."""
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"Preserving mismatched existing asset: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "md5").hexdigest()
    if actual != expected_md5:
        raise ValueError(f"Preserving asset with mismatched checksum: {path}")


def _publish_stream(path, stream, size, digest=None, max_seconds=1800):
    """Publish an exclusive bounded partial using a no-overwrite hard link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    output = partial.open("xb")
    try:
        count, checksum, started = 0, hashlib.md5(), time.monotonic()
        with output:
            while block := stream.read(CHUNK_BYTES):
                count += len(block)
                if count > size or time.monotonic() - started > max_seconds:
                    raise ValueError("Transfer exceeded its pinned size or time budget")
                output.write(block)
                checksum.update(block)
            if count != size or (digest is not None and checksum.hexdigest() != digest):
                raise ValueError("Transfer differs from the pinned size/checksum")
            output.flush()
            os.fsync(output.fileno())
        os.link(partial, path)  # Fails if another process created the destination.
    finally:
        partial.unlink(missing_ok=True)


def _download(path, url, size, md5):
    if path.exists():
        verify_file(path, size, md5)
        return False
    print(f"Downloading {path.name}: {size / 1_000_000:.2f} MB from {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != size:
            raise ValueError("Server length differs from the pinned release")
        _publish_stream(path, response, size, md5)
    return True


def _record(path, spec):
    if path.exists():
        if not path.is_file() or path.stat().st_size > 2_000_000:
            raise ValueError("Invalid or oversized saved Zenodo record")
        raw = path.read_bytes()
    else:
        url = f"https://zenodo.org/api/records/{spec['record_id']}"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Zenodo record exceeds the metadata budget")
    record = json.loads(raw)
    if record.get("id") != spec["record_id"]:
        raise ValueError("Zenodo record is not the pinned release")
    files = {entry["key"]: entry for entry in record.get("files", [])}
    for name, (size, md5) in spec["files"].items():
        entry = files.get(name, {})
        if entry.get("size") != size or entry.get("checksum") != f"md5:{md5}":
            raise ValueError(f"Zenodo metadata differs from the pinned asset: {name}")
    if not path.exists():
        import io

        _publish_stream(path, io.BytesIO(raw), len(raw))
    return record


def extract_archive(archive_path, destination, spec, *, demo_only=False):
    """Safely retain original archive paths, verifying existing members byte-for-byte.

    The caller verifies the archive's pinned checksum first. Demo preparation
    extracts only three NuInsSeg files; CytoNuke's small archive is kept complete.
    No labels, image pixels, filenames or auxiliary annotations are transformed.
    """
    destination = checked_path(destination, destination)
    selected_names = set(spec.get("demo_members", ())) if demo_only else set()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > spec["max_members"] or sum(m.file_size for m in members) > spec["max_extracted_bytes"]:
            raise ValueError("Archive exceeds its extraction budget")
        names, selected = set(), []
        for member in members:
            name = member.filename.rstrip("/")
            relative = PurePosixPath(name)
            mode = stat.S_IFMT(member.external_attr >> 16)
            if (not name or relative.is_absolute() or ".." in relative.parts
                    or "\\" in name or ":" in relative.parts[0]
                    or mode not in {0, stat.S_IFREG, stat.S_IFDIR}
                    or member.flag_bits & 1 or member.file_size > spec["max_member_bytes"]):
                raise ValueError(f"Unsupported or unsafe archive member: {member.filename}")
            key = name.casefold()
            if key in names:
                raise ValueError(f"Duplicate archive path: {member.filename}")
            names.add(key)
            path = checked_path(destination.joinpath(*relative.parts), destination)
            if not member.is_dir() and (not selected_names or name in selected_names):
                selected.append((member, path))
        if selected_names and selected_names != {m.filename for m, _ in selected}:
            raise ValueError("Archive is missing required demo members")
        # Reject a changed existing file before extracting any missing members.
        reused, missing = 0, []
        for member, path in selected:
            if not path.exists():
                missing.append((member, path))
                continue
            if not path.is_file() or path.stat().st_size != member.file_size:
                raise ValueError(f"Preserving differing extracted member: {path}")
            with path.open("rb") as existing, archive.open(member) as original:
                while block := original.read(CHUNK_BYTES):
                    if existing.read(len(block)) != block:
                        raise ValueError(f"Preserving differing extracted member: {path}")
                if existing.read(1):
                    raise ValueError(f"Preserving differing extracted member: {path}")
            reused += 1
        for member, path in missing:
            checked_path(path, destination)
            with archive.open(member) as original:
                _publish_stream(path, original, member.file_size)
    return {"scope": "demo_members" if selected_names else "complete_archive",
            "selected_files": len(selected), "reused_files": reused,
            "extracted_files": len(missing), "selected_bytes": sum(m.file_size for m, _ in selected)}


def prepare_dataset(name, *, demo_only=False):
    """Prepare one pinned dataset in this checkout's canonical data directory."""
    if name not in DATASETS:
        raise ValueError(f"Choose a dataset from {tuple(DATASETS)}")
    spec = DATASETS[name]
    root = checked_path(ROOT / "data" / name, ROOT / "data")
    source = checked_path(root / "source", root)
    record = _record(checked_path(source / "zenodo-record.json", root), spec)
    downloaded = []
    for filename, (size, md5) in spec["files"].items():
        path = checked_path(source / filename, root)
        url = f"https://zenodo.org/records/{spec['record_id']}/files/{quote(filename)}?download=1"
        if _download(path, url, size, md5):
            downloaded.append(filename)
    extraction = extract_archive(source / spec["archive"], root / "extracted", spec, demo_only=demo_only)
    return {"status": "complete", "dataset": name, "record_id": spec["record_id"],
            "doi": record.get("doi"), "root": str(root), "archive_checksum_verified": True,
            "downloaded_files": downloaded, **extraction}

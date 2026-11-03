"""Shared file identities without model or training dependencies."""
import hashlib
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes(project):
    """Record the current source tree, including every separated numerical owner."""
    project = Path(project)
    return {str(path.relative_to(project)): sha256(path) for path in sorted((project / "src").rglob("*.py"))}

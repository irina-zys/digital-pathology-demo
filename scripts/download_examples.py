"""Download small real public examples, with fixed content hashes."""

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODAIT = "https://raw.githubusercontent.com/CODAIT/deep-histopath/c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33/docs/wsi-preprocessing-in-python/images/"
EXAMPLES = {
    "CMU-1-Small-Region.svs": ("https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs", "ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7"),
    "red-pen.png": (CODAIT + "red-pen.png", "39dd2d84ef86c149cc507a3e6e370433f7805b1f5d426aa6fff92bab79e86c96"),
    "green-pen.png": (CODAIT + "green-pen.png", "db4423e05858cd503ab787381df699c5e8c8e0ce85f0016e23da32700838c1d3"),
    "blue-pen.png": (CODAIT + "blue-pen.png", "5a5d172c0a25ffa26c207a5ad36f06db89fba6e184e80936b05fbc0b898f730d"),
}


def main():
    destination = ROOT / "data/demo"
    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in EXAMPLES.items():
        path = destination / name
        if path.exists():
            with path.open("rb") as stream:
                raw = stream.read(10_000_001)
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read(10_000_001)
        if len(raw) > 10_000_000 or hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError(f"Sample verification failed: {name}; existing files were not replaced")
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(raw)
        print(f"Verified {path}: {len(raw):,} bytes")


if __name__ == "__main__":
    main()

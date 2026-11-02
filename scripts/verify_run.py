"""Inspect artifacts from a real preprocessing run; no generated test images."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--check-regions", action="store_true", help="Compare the first saved region per slide with a fresh OpenSlide read")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert (args.run / "index.html").is_file()
    assert manifest["slides"]
    for slide in manifest["slides"]:
        folder = args.run / slide["folder"]
        rgb = np.asarray(Image.open(folder / "thumbnail.png"))
        assert list(rgb.shape[1::-1]) == slide["thumbnail_dimensions"]
        assert hashlib.sha256((folder / "thumbnail.png").read_bytes()).hexdigest() == slide["thumbnail_sha256"]
        masks = {}
        for name in ("tissue_mask", "clean_mask", "artifact_mask", "removed_tissue"):
            values = np.asarray(Image.open(folder / f"{name}.png"))
            assert values.shape == rgb.shape[:2]
            assert set(np.unique(values)).issubset({0, 255})
            masks[name] = values > 0
        tissue, clean = masks["tissue_mask"], masks["clean_mask"]
        assert not np.any(clean & ~tissue)
        assert not np.any(clean & masks["artifact_mask"])
        assert np.array_equal(masks["removed_tissue"], tissue & ~clean)
        assert int(tissue.sum()) == slide["tissue_pixels"]
        assert int(clean.sum()) == slide["retained_pixels"]
        assert slide["removed_pixels"] + slide["retained_pixels"] == slide["tissue_pixels"]
        filtered = np.asarray(Image.open(folder / "filtered.png"))
        assert np.array_equal(filtered[clean], rgb[clean])
        assert np.all(filtered[~clean] == 255)
        if "gallery" in slide:
            gallery = args.run / slide["gallery"]
            for name, reference in (("thumbnail", rgb), ("mask", clean.astype(np.uint8) * 255),
                                    ("preprocessed", filtered)):
                saved = np.asarray(Image.open(gallery / f"{slide['gallery_prefix']}__{name}.png"))
                assert np.array_equal(saved, reference), "Gallery view differs from per-slide artifact"
        for layout in slide.get("exports", {}).get("layouts", []):
            import h5py

            with h5py.File(args.run / layout["coords"], "r") as handle:
                coords = handle["coords"][:]
                attrs = dict(handle["coords"].attrs)
                assert coords.shape == (layout["exported_count"], 2)
                assert np.issubdtype(coords.dtype, np.integer)
                assert attrs["patch_level"] == 0
                size = attrs["patch_size"]
                assert size == layout["patch_size"]
                assert np.all(coords >= 0)
                assert np.all(coords + size <= np.asarray(slide["dimensions"]))
                assert np.all(handle["tissue_fraction"][:] >= handle.attrs["min_tissue_fraction"])
            if layout["region_tar"]:
                tar_path = args.run / layout["folder"] / "tar_patch_4096" / (Path(slide["input"]).stem + ".tar")
                with tarfile.open(tar_path) as archive:
                    members = [m for m in archive.getmembers() if m.name.endswith(".png")]
                    assert len(members) == len(coords)
                    if members and args.check_regions:
                        import openslide

                        region = np.asarray(Image.open(archive.extractfile(members[0])).convert("RGB"))
                        with openslide.OpenSlide(slide["input"]) as wsi:
                            reference = np.asarray(wsi.read_region(tuple(map(int, coords[0])), 0, (size, size)).convert("RGB"))
                        assert region.shape == (size, size, 3)
                        assert np.array_equal(region, reference), "Saved region differs from level-0 source pixels"
                        print(f"Verified lossless {size}x{size} region against source WSI")
        print(f"Verified {slide['name']}: {rgb.shape[1]}x{rgb.shape[0]}, retained={clean.mean():.1%}")
    print(f"Verified {len(manifest['slides'])} real-input artifact sets; no biological accuracy claim.")


if __name__ == "__main__":
    main()

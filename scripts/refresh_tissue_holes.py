"""Refresh only tissue/hole visualizations: python -m scripts.refresh_tissue_holes RUN [RUN...]."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.imaging_utils.arrays import save_png, tissue_hole_preview


SEMANTICS = "retained_tissue_green_hole_blue_contours_4_connected_thumbnail"


def inside(run, relative):
    path = (run / relative).resolve()
    if not path.is_relative_to(run):
        raise ValueError("Visualization destination must stay inside its run")
    return path


def refresh(run):
    run = run.resolve(strict=True)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Only completed runs can be refreshed")
    for slide in manifest["slides"]:
        folder = inside(run, slide["folder"])
        rgb = np.asarray(Image.open(folder / "thumbnail.png").convert("RGB"))
        values = np.asarray(Image.open(folder / "clean_mask.png"))
        if values.shape != rgb.shape[:2] or not set(np.unique(values)).issubset({0, 255}):
            raise ValueError("Invalid saved retained mask")
        preview, holes = tissue_hole_preview(rgb, values > 0)
        save_png(folder / "tissue_holes.png", preview)
        save_png(folder / "hole_mask.png", holes)
        if "gallery" in slide:
            save_png(inside(run, str(Path(slide["gallery"]) / f"{slide['gallery_prefix']}__tissue_holes.png")), preview)
        for layout in slide.get("exports", {}).get("layouts", []):
            destination = inside(run, str(Path(layout["folder"]) / "masks" / (Path(slide["input"]).stem + ".jpg")))
            if not destination.is_file():
                raise FileNotFoundError(destination)
            Image.fromarray(preview).save(destination, quality=90)
            layout["mask_preview_type"] = "retained_tissue_and_enclosed_background_holes"
        slide["hole_visualization"] = SEMANTICS
        (folder / "metadata.json").write_text(json.dumps(slide, indent=2) + "\n")
        print(f"Refreshed tissue/hole thumbnails: {slide['name']}", flush=True)
    manifest["visualization_refresh"] = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "saved thumbnail.png and clean_mask.png; no WSI or patch pixels read",
        "hole_semantics": "background not four-connected to the thumbnail boundary; not anatomical annotation",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    for run in parser.parse_args().runs:
        refresh(run)


if __name__ == "__main__":
    main()

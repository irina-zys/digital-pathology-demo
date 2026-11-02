"""CLAM coordinate bags and HIPT-style region archives, with explicit bounds."""

import csv
import io
import json
import os
import re
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def coordinates(mask, dimensions, size, limit, min_fraction, max_candidates):
    """Select a spread of full, non-overlapping level-0 patches using mask coverage."""
    width, height = dimensions
    nx, ny = width // size, height // size
    if nx * ny > max_candidates:
        raise ValueError("Patch grid exceeds export.max_candidates; increase the explicit budget or disable export")
    if not nx or not ny:
        return np.empty((0, 2), dtype=np.int64), np.empty(0), 0
    xx, yy = np.meshgrid(np.arange(nx) * size, np.arange(ny) * size)
    coords = np.column_stack((xx.ravel(), yy.ravel())).astype(np.int64)
    mh, mw = mask.shape
    x0 = np.floor(coords[:, 0] * mw / width).astype(int)
    y0 = np.floor(coords[:, 1] * mh / height).astype(int)
    x1 = np.minimum(mw, np.ceil((coords[:, 0] + size) * mw / width).astype(int))
    y1 = np.minimum(mh, np.ceil((coords[:, 1] + size) * mh / height).astype(int))
    integral = cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_64F)
    areas = (x1 - x0) * (y1 - y0)
    coverage = (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / areas
    valid = np.flatnonzero(coverage >= min_fraction)
    count = len(valid)
    if count > limit:
        valid = valid[np.linspace(0, count - 1, limit, dtype=int)]
    return coords[valid], coverage[valid], count


def add_tar_member(archive, name, content):
    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mtime = 0
    archive.addfile(member, io.BytesIO(content))


def export_slide(output, row, metadata, rgb, mask, overlay, cfg, masking_method):
    """Produce actual coordinates, pixel stitches, and optional lossless 4K regions."""
    import h5py
    from tiatoolbox.wsicore.wsireader import WSIReader

    path = Path(row["path"])
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return {"status": "skipped_raster_example"}
    if list(cfg.patch_sizes) != [256, 4096] or len(cfg.max_patches) != 2:
        raise ValueError("This bounded exporter supports patch_sizes=[256,4096] with two limits")
    if not 0 < cfg.min_tissue_fraction <= 1 or not 1 <= cfg.max_candidates <= 1_000_000:
        raise ValueError("Invalid patch coverage or grid budget")
    if not 1 <= cfg.max_patches[0] <= 1024 or not 1 <= cfg.max_patches[1] <= 8:
        raise ValueError("Limits: 1-1024 small patches and 1-8 regions per slide")
    slide_id = path.stem
    project = str(row.get("project", "demo")).lower().replace("-", "_")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", slide_id) or not re.fullmatch(r"[a-z0-9_]+", project):
        raise ValueError("Export identifiers must be filesystem-safe")
    root = output / "datasets" / project
    wsi_folder = root / "WSIs"
    wsi_folder.mkdir(parents=True, exist_ok=True)
    link = wsi_folder / path.name
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Duplicate slide export: {slide_id}")
    link.symlink_to(os.path.relpath(path.resolve(), wsi_folder.resolve()))
    width, height = metadata["dimensions"]
    reader = WSIReader.open(path)
    exports = []
    try:
        for size, limit in zip(cfg.patch_sizes, cfg.max_patches, strict=True):
            size, limit = int(size), int(limit)
            coords, coverage, eligible = coordinates(mask, (width, height), size, limit,
                                                       cfg.min_tissue_fraction, cfg.max_candidates)
            folder = root / f"extracted_level0_patch{size}_fp"
            for name in ("masks", "patches", "stitches", "tissue_masks"):
                (folder / name).mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay).save(folder / "masks" / f"{slide_id}.jpg", quality=90)
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "tissue_masks" / f"{slide_id}.png")
            with h5py.File(folder / "patches" / f"{slide_id}.h5", "x") as handle:
                data = handle.create_dataset("coords", data=coords, maxshape=(None, 2), chunks=True)
                data.attrs.update({"patch_size": size, "patch_level": 0, "downsample": [1.0, 1.0],
                                   "level_dim": [width, height], "downsampled_level_dim": [width, height],
                                   "name": slide_id, "save_path": str((folder / "patches").relative_to(output))})
                handle.create_dataset("tissue_fraction", data=coverage)
                handle.attrs.update({"coordinate_space": "level0_xy_top_left", "bounded_demo": True,
                                     "eligible_count": eligible, "selection": "evenly_spaced_in_row_major_eligible_grid",
                                     "min_tissue_fraction": cfg.min_tissue_fraction, "masking_method": masking_method})
                if metadata["mpp"] is not None:
                    handle.attrs["mpp"] = metadata["mpp"]
            stitch = Image.new("RGB", (rgb.shape[1], rgb.shape[0]), "black")
            archive = None
            if size == 4096 and cfg.save_region_tar:
                (folder / "tar_patch_4096").mkdir(exist_ok=True)
                archive = tarfile.open(folder / "tar_patch_4096" / f"{slide_id}.tar", "x")
            try:
                for (x, y), fraction in zip(coords, coverage, strict=True):
                    patch = reader.read_rect((int(x), int(y)), size=(size, size), resolution=0, units="level")
                    if patch.shape != (size, size, 3) or patch.dtype != np.uint8:
                        raise RuntimeError("Unexpected patch shape or dtype")
                    image = Image.fromarray(patch)
                    x0, y0 = round(x * rgb.shape[1] / width), round(y * rgb.shape[0] / height)
                    x1, y1 = round((x + size) * rgb.shape[1] / width), round((y + size) * rgb.shape[0] / height)
                    if x1 > x0 and y1 > y0:
                        stitch.paste(image.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS), (x0, y0))
                    if archive is not None:
                        key = f"{slide_id}_{int(x)}_{int(y)}"
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        add_tar_member(archive, key + ".png", buffer.getvalue())
                        add_tar_member(archive, key + ".json", json.dumps({"slide_id": slide_id,
                            "x": int(x), "y": int(y), "patch_level": 0, "patch_size": size,
                            "tissue_fraction": float(fraction), "label": row.get("label")}).encode())
            finally:
                if archive is not None:
                    archive.close()
            stitch.save(folder / "stitches" / f"{slide_id}.jpg", quality=90)
            process = {"slide_id": path.name, "process": 0,
                       "status": "processed" if len(coords) else "no_tissue_coordinates",
                       "patch_level": 0, "patch_size": size, "step_size": size,
                       "masking_method": masking_method, "min_tissue_fraction": cfg.min_tissue_fraction,
                       "max_patches": limit, "eligible_count": eligible, "exported_count": len(coords)}
            csv_path = folder / "process_list_autogen.csv"
            exists = csv_path.exists()
            with csv_path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(process))
                if not exists:
                    writer.writeheader()
                writer.writerow(process)
            exports.append({"folder": str(folder.relative_to(output)), "patch_size": size,
                            "eligible_count": eligible, "exported_count": len(coords),
                            "coords": str((folder / "patches" / f"{slide_id}.h5").relative_to(output)),
                            "region_tar": bool(size == 4096 and cfg.save_region_tar)})
    finally:
        handle = getattr(reader, "openslide_wsi", None)
        if handle is not None:
            handle.close()
    dataset = root / "dataset.csv"
    exists = dataset.exists()
    with dataset.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "slide_id", "label", "wsi_path"])
        if not exists:
            writer.writeheader()
        writer.writerow({"case_id": row.get("case_submitter_id", slide_id), "slide_id": slide_id,
                         "label": row.get("label", "unlabelled"), "wsi_path": f"WSIs/{path.name}"})
    return {"status": "complete", "layouts": exports}

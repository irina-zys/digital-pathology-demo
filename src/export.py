"""CLAM coordinate bags and HIPT-style region archives, with explicit bounds."""

import csv
import io
import json
import os
import re
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.image_utils import save_png
from src.tiling import coordinates, calibrated_patch_size, mask_bounds, read_tile, open_clam_slide, clam_coordinates, CLAM_COMMIT


def add_tar_member(archive, name, content):
    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mtime = 0
    archive.addfile(member, io.BytesIO(content))


def export_slide(output, row, metadata, rgb, mask, segmentation_preview, cfg, masking_method):
    """Export calibrated coordinate bags, thumbnail coverage, and selected-slide pixels."""
    import h5py

    path = Path(row["path"])
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return {"status": "skipped_raster_example"}
    sizes = list(cfg.patch_sizes)
    if sizes not in ([256], [256, 4096]) or len(cfg.max_patches) != len(sizes):
        raise ValueError("Use patch_sizes=[256] (default) or [256,4096], with one limit per size")
    tiling_method = cfg.get("tiling_method", "clam_four_pt")
    if tiling_method not in {"clam_four_pt", "coverage_grid"}:
        raise ValueError("tiling_method must be clam_four_pt or coverage_grid")
    if not 0 < cfg.min_tissue_fraction <= 1 or not 1 <= cfg.max_candidates <= 1_000_000:
        raise ValueError("Invalid patch coverage or grid budget")
    if any(not isinstance(limit, int) or not 0 <= limit <= (1024 if size == 256 else 8)
           for size, limit in zip(sizes, cfg.max_patches, strict=True)):
        raise ValueError("Limits: 0-1024 small patches and 0-8 optional regions for the selected slide")
    coordinate_limit = cfg.get("coordinate_limit")
    if coordinate_limit is not None and (not isinstance(coordinate_limit, int) or
                                         not 1 <= coordinate_limit <= cfg.max_candidates):
        raise ValueError("coordinate_limit must be null (all eligible) or 1-max_candidates")
    target_mpp = float(cfg.target_mpp)
    if not np.isfinite(target_mpp) or not 0.1 <= target_mpp <= 4:
        raise ValueError("target_mpp must be finite and between 0.1 and 4 micrometres per pixel")
    native_mpp = np.asarray(metadata.get("mpp"), dtype=float)
    if native_mpp.shape != (2,) or not np.all(np.isfinite(native_mpp)) or np.any(native_mpp <= 0):
        raise ValueError("Calibrated patch export requires positive native x/y MPP metadata")
    if np.any(native_mpp > target_mpp * 1.01):
        raise ValueError("Native MPP is coarser than target_mpp; refusing to imply unavailable spatial detail")
    if not np.isclose(native_mpp[0], native_mpp[1], rtol=1e-6, atol=0):
        raise ValueError("Anisotropic native MPP is unsupported by the square CLAM patch contract")
    if not 1 <= cfg.max_patch_read_pixels <= 200_000_000:
        raise ValueError("max_patch_read_pixels must be 1-200000000")
    slide_id = path.stem
    pixel_slide_id = cfg.get("pixel_slide_id")
    if pixel_slide_id is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", str(pixel_slide_id)):
        raise ValueError("pixel_slide_id must be null or an exact filesystem-safe slide stem")
    save_pixels = pixel_slide_id == slide_id
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
    reader = open_clam_slide(path)
    exports = []
    try:
        for size, limit in zip(cfg.patch_sizes, cfg.max_patches, strict=True):
            size, limit = int(size), int(limit) if save_pixels else 0
            source_dimensions, _ = calibrated_patch_size(size, target_mpp, [native_mpp[0], native_mpp[0]])
            effective_mpp = (source_dimensions[0] * native_mpp / size).tolist()
            source_size = source_dimensions[0]
            if tiling_method == "clam_four_pt":
                coords, coverage, eligible = clam_coordinates(reader, mask, source_size,
                    limit=coordinate_limit, min_fraction=cfg.min_tissue_fraction,
                    max_candidates=cfg.max_candidates, contour_fn=cfg.contour_fn, max_contours=cfg.max_contours)
                full_selection, capped_selection = "all_eligible_clam_contours", "evenly_spaced_in_clam_contour_order"
            else:
                coords, coverage, eligible = coordinates(mask, (width, height), source_size,
                    coordinate_limit, cfg.min_tissue_fraction, cfg.max_candidates)
                full_selection, capped_selection = "all_eligible_grid", "evenly_spaced_in_row_major_eligible_grid"
            backend = "CLAM" if tiling_method == "clam_four_pt" else "coverage_grid"
            preview_indices = np.linspace(0, len(coords) - 1, min(len(coords), limit), dtype=np.int64)
            if len(preview_indices) and source_size ** 2 > cfg.max_patch_read_pixels:
                raise ValueError("Native patch footprint exceeds export.max_patch_read_pixels")
            selection = full_selection if coordinate_limit is None else capped_selection
            resolution_tag = format(target_mpp, "g").replace(".", "p")
            folder = root / f"extracted_mpp{resolution_tag}_patch{size}_fp"
            for name in ("masks", "patches", "stitches", "tissue_masks", "coverage_masks"):
                (folder / name).mkdir(parents=True, exist_ok=True)
            Image.fromarray(segmentation_preview).save(folder / "masks" / f"{slide_id}.jpg", quality=90)
            save_png(folder / "tissue_masks" / f"{slide_id}.png", mask)
            with h5py.File(folder / "patches" / f"{slide_id}.h5", "x") as handle:
                data = handle.create_dataset("coords", data=coords, maxshape=(None, 2), chunks=True)
                data.attrs.update({"patch_size": source_size, "output_patch_size": size,
                                   "target_mpp": target_mpp, "native_mpp": native_mpp,
                                   "effective_mpp": effective_mpp, "resampling": "Pillow.LANCZOS_from_level0",
                                   "patch_level": 0, "downsample": [1.0, 1.0],
                                   "level_dim": [width, height], "downsampled_level_dim": [width, height],
                                   "name": slide_id, "save_path": str((folder / "patches").relative_to(output))})
                handle.create_dataset("tissue_fraction", data=coverage)
                handle.create_dataset("preview_indices", data=preview_indices)
                handle.attrs.update({"coordinate_space": "level0_xy_top_left", "bounded_demo": True,
                                     "eligible_count": eligible, "selection": selection,
                                     "coordinate_limit": -1 if coordinate_limit is None else coordinate_limit,
                                     "max_candidates": cfg.max_candidates, "pixel_preview_limit": limit,
                                     "pixel_preview_count": len(preview_indices), "export_schema_version": 6, "tiling_method": tiling_method,
                                     "tiling_backend": backend, "clam_commit": CLAM_COMMIT,
                                     "contour_fn": cfg.contour_fn, "max_contours": cfg.max_contours, "use_padding": False,
                                     "pixel_slide_selected": save_pixels, "max_patch_read_pixels": cfg.max_patch_read_pixels,
                                     "min_tissue_fraction": cfg.min_tissue_fraction, "masking_method": masking_method})
                if metadata["mpp"] is not None:
                    handle.attrs["mpp"] = metadata["mpp"]
            # Full coordinate coverage uses existing thumbnail pixels, never extra WSI reads.
            coverage_mask = np.zeros(mask.shape, dtype=bool)
            bounds = mask_bounds(coords, (width, height), source_size, mask.shape)
            for x0, y0, x1, y1 in zip(*bounds, strict=True):
                coverage_mask[y0:y1, x0:x1] = True
            overview = rgb.copy()
            overview[~coverage_mask] = 255
            Image.fromarray(overview).save(folder / "stitches" / f"{slide_id}.jpg", quality=90)
            save_png(folder / "coverage_masks" / f"{slide_id}.png", coverage_mask)
            stitch = Image.new("RGB", (rgb.shape[1], rgb.shape[0]), "black") if len(preview_indices) else None
            patch_images = folder / "patch_images" / slide_id if size == 256 and len(preview_indices) else None
            if patch_images is not None:
                patch_images.mkdir(parents=True)
            patch_records = []
            archive = None
            save_tar = bool(size == 4096 and cfg.save_region_tar and len(preview_indices))
            if save_tar:
                (folder / "tar_patch_4096").mkdir(exist_ok=True)
                archive = tarfile.open(folder / "tar_patch_4096" / f"{slide_id}.tar", "x")
            try:
                for coordinate_index in preview_indices:
                    x, y = coords[coordinate_index]
                    fraction = coverage[coordinate_index]
                    patch = read_tile(reader, (int(x), int(y)), source_size, size,
                                      max_read_pixels=cfg.max_patch_read_pixels)
                    image = Image.fromarray(patch)
                    if patch_images is not None:
                        name = f"{int(x)}_{int(y)}.png"
                        image.save(patch_images / name)
                        patch_records.append({"file": name, "coordinate_index": int(coordinate_index),
                                              "x": int(x), "y": int(y), "patch_level": 0, "patch_size": size,
                                              "source_patch_size": source_size, "target_mpp": target_mpp,
                                              "effective_mpp": effective_mpp, "resampling": "Pillow.LANCZOS_from_level0"})
                    x0, y0 = round(x * rgb.shape[1] / width), round(y * rgb.shape[0] / height)
                    x1, y1 = round((x + source_size) * rgb.shape[1] / width), round((y + source_size) * rgb.shape[0] / height)
                    if x1 > x0 and y1 > y0:
                        stitch.paste(image.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS), (x0, y0))
                    if archive is not None:
                        key = f"{slide_id}_{int(x)}_{int(y)}"
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        add_tar_member(archive, key + ".png", buffer.getvalue())
                        add_tar_member(archive, key + ".json", json.dumps({"slide_id": slide_id,
                            "x": int(x), "y": int(y), "coordinate_index": int(coordinate_index), "patch_level": 0, "patch_size": size,
                            "source_patch_size": source_size, "target_mpp": target_mpp,
                            "effective_mpp": effective_mpp, "resampling": "Pillow.LANCZOS_from_level0",
                            "tissue_fraction": float(fraction), "label": row.get("label")}).encode())
            finally:
                if archive is not None:
                    archive.close()
            sampled_stitch = None
            if stitch is not None:
                (folder / "sampled_pixel_stitches").mkdir(exist_ok=True)
                sampled_stitch = folder / "sampled_pixel_stitches" / f"{slide_id}.jpg"
                stitch.save(sampled_stitch, quality=90)
            if patch_images is not None:
                (patch_images / "index.json").write_text(json.dumps(patch_records, indent=2) + "\n")
            process = {"slide_id": path.name, "process": 0,
                       "status": "processed" if len(coords) else "no_tissue_coordinates",
                       "patch_level": 0, "patch_size": source_size, "step_size": source_size,
                       "output_patch_size": size, "target_mpp": target_mpp,
                       "effective_mpp_x": effective_mpp[0], "effective_mpp_y": effective_mpp[1],
                       "masking_method": masking_method, "min_tissue_fraction": cfg.min_tissue_fraction,
                       "max_patches": limit, "coordinate_limit": coordinate_limit,
                       "eligible_count": eligible, "exported_count": len(coords),
                       "pixel_preview_count": len(preview_indices), "pixel_slide_selected": save_pixels,
                       "patch_image_count": len(patch_records),
                       "region_count": len(preview_indices) if save_tar else 0}
            csv_path = folder / "process_list_autogen.csv"
            exists = csv_path.exists()
            with csv_path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(process))
                if not exists:
                    writer.writeheader()
                writer.writerow(process)
            exports.append({"tiling_method": tiling_method, "tiling_backend": backend, "clam_commit": CLAM_COMMIT, "contour_fn": cfg.contour_fn,
                            "folder": str(folder.relative_to(output)), "patch_size": size,
                            "source_patch_size": source_size, "native_mpp": native_mpp.tolist(),
                            "target_mpp": target_mpp, "effective_mpp": effective_mpp,
                            "resampling": "Pillow.LANCZOS_from_level0",
                            "eligible_count": eligible, "exported_count": len(coords),
                            "coordinate_selection": selection, "coordinate_limit": coordinate_limit,
                            "pixel_preview_count": len(preview_indices), "pixel_preview_limit": limit,
                            "pixel_slide_selected": save_pixels, "patch_image_count": len(patch_records),
                            "patch_images": str(patch_images.relative_to(output)) if patch_images is not None else None,
                            "region_count": len(preview_indices) if save_tar else 0,
                            "mask_preview_type": "retained_tissue_and_enclosed_background_holes",
                            "stitch_type": "thumbnail_pixels_within_exported_coordinate_coverage",
                            "coverage_mask": str((folder / "coverage_masks" / f"{slide_id}.png").relative_to(output)),
                            "sampled_pixel_stitch": str(sampled_stitch.relative_to(output)) if sampled_stitch is not None else None,
                            "coords": str((folder / "patches" / f"{slide_id}.h5").relative_to(output)),
                            "region_tar": save_tar})
    finally:
        handle = getattr(reader, "wsi", None)
        if handle is not None:
            handle.close()
    dataset = root / "dataset.csv"
    fields = ["case_id", "slide_id", "label", "wsi_path", "stain", "sample_id"]
    exists = dataset.exists()
    if exists:
        with dataset.open(newline="") as stream:
            if next(csv.reader(stream), None) != fields:
                raise ValueError("Existing dataset.csv has a different schema; use a fresh export directory")
    with dataset.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({"case_id": (row.get("case_submitter_id") or "").strip() or (row.get("case_id") or "").strip(),
            "slide_id": slide_id, "label": row.get("label") or "unlabelled", "wsi_path": f"WSIs/{path.name}",
            "stain": row.get("stain") or "", "sample_id": row.get("sample_id") or ""})
    return {"status": "complete", "layouts": exports}

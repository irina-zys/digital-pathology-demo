"""Inspect artifacts from a real preprocessing run; no generated test images."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def clam_candidates(clean, dimensions, size, max_candidates):
    """Reconstruct CLAM's contour-anchored four-point grid without production helpers."""
    import cv2

    width, height = dimensions
    mh, mw = clean.shape
    contours, hierarchy = cv2.findContours(clean.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    scale = np.array([width / mw, height / mh])
    native = [(contour * scale).astype(np.int32) for contour in contours]
    valid = [len(contour) >= 3 and cv2.contourArea(contour) > 0 for contour in native]
    candidates, seen, candidate_count = [], set(), 0
    center_offset = size // 2
    shift = int(center_offset * 0.5)
    for index, relation in enumerate(hierarchy[0]):
        if relation[3] != -1 or not valid[index]:
            continue
        contour = native[index]
        holes = [native[child] for child, child_relation in enumerate(hierarchy[0])
                 if child_relation[3] == index and valid[child]]
        start_x, start_y, box_w, box_h = cv2.boundingRect(contour)
        xs = range(start_x, min(start_x + box_w, width - size + 1), size)
        ys = range(start_y, min(start_y + box_h, height - size + 1), size)
        candidate_count += len(xs) * len(ys)
        assert candidate_count <= max_candidates, "CLAM candidate grid exceeds declared budget"
        for x in xs:
            for y in ys:
                center_x, center_y = x + center_offset, y + center_offset
                points = [(float(center_x + dx), float(center_y + dy))
                          for dx, dy in ((-shift, -shift), (shift, shift), (shift, -shift), (-shift, shift))]
                if not any(cv2.pointPolygonTest(contour, point, False) >= 0 for point in points):
                    continue
                if any(cv2.pointPolygonTest(hole, (x + size / 2, y + size / 2), False) > 0 for hole in holes):
                    continue
                if (x, y) not in seen:
                    seen.add((x, y))
                    candidates.append((x, y))
    return candidates


def verify_upstream_clam_coverage(handle, coords, clean, dimensions, size):
    """Independently reconstruct eligibility from the saved real-image mask."""
    width, height = dimensions
    mh, mw = clean.shape
    expected, fractions = [], []
    if handle.attrs.get("tiling_backend") == "CLAM":
        import cv2

        # Independently reconstruct contour-origin grids and CLAM predicates.
        contours, hierarchy = cv2.findContours(clean.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        parents = hierarchy[0, :, 3] if hierarchy is not None else np.empty(0, dtype=int)
        outers = np.flatnonzero(parents == -1)
        assert len(outers) <= handle.attrs["max_contours"]
        scale = np.asarray([width / mw, height / mh])
        seen, candidate_count = set(), 0
        for index in outers:
            contour = (contours[index] * scale).astype(np.int32)
            holes = [(contours[j] * scale).astype(np.int32) for j in np.flatnonzero(parents == index)]
            x0, y0, w, h = cv2.boundingRect(contour)
            xs = range(x0, min(x0 + w, width - size + 1), size)
            ys = range(y0, min(y0 + h, height - size + 1), size)
            candidate_count += len(xs) * len(ys)
            for x in xs:
                for y in ys:
                    mode = handle.attrs["contour_fn"]
                    center = (x + size // 2, y + size // 2)
                    shift = int(size // 2 * 0.5)
                    points = ([(x, y)] if mode == "basic" else [center] if mode == "center" or shift == 0 else
                              [(center[0] - shift, center[1] - shift), (center[0] + shift, center[1] + shift),
                               (center[0] + shift, center[1] - shift), (center[0] - shift, center[1] + shift)])
                    inside = [cv2.pointPolygonTest(contour, tuple(map(float, point)), False) >= 0 for point in points]
                    accepted = all(inside) if mode == "four_pt_hard" else any(inside)
                    in_hole = any(cv2.pointPolygonTest(hole, (x + size / 2, y + size / 2), False) > 0 for hole in holes)
                    if not accepted or in_hole or (x, y) in seen:
                        continue
                    seen.add((x, y))
                    left, right = int(np.floor(x * mw / width)), min(mw, int(np.ceil((x + size) * mw / width)))
                    top, bottom = int(np.floor(y * mh / height)), min(mh, int(np.ceil((y + size) * mh / height)))
                    fraction = float(clean[top:bottom, left:right].mean())
                    if fraction >= handle.attrs["min_tissue_fraction"]:
                        expected.append((x, y))
                        fractions.append(fraction)
        assert candidate_count <= handle.attrs["max_candidates"]
        assert not handle.attrs["use_padding"]
    else:
        assert (width // size) * (height // size) <= handle.attrs["max_candidates"]
        for y in range(0, height - size + 1, size):
            y0, y1 = int(np.floor(y * mh / height)), min(mh, int(np.ceil((y + size) * mh / height)))
            for x in range(0, width - size + 1, size):
                x0, x1 = int(np.floor(x * mw / width)), min(mw, int(np.ceil((x + size) * mw / width)))
                fraction = float(clean[y0:y1, x0:x1].mean())
                if fraction >= handle.attrs["min_tissue_fraction"]:
                    expected.append((x, y))
                    fractions.append(fraction)
    expected = np.asarray(expected, dtype=np.int64).reshape(-1, 2)
    fractions = np.asarray(fractions)
    assert len(expected) == handle.attrs["eligible_count"]
    limit = int(handle.attrs["coordinate_limit"])
    if limit >= 0 and len(expected) > limit:
        indices = np.linspace(0, len(expected) - 1, limit, dtype=int)
        expected, fractions = expected[indices], fractions[indices]
    assert np.array_equal(coords, expected), "Exported coordinates differ from the complete eligible grid"
    assert np.allclose(handle["tissue_fraction"][:], fractions, rtol=0, atol=1e-12)
    if limit == -1:
        assert handle.attrs["selection"] == ("all_eligible_clam_contours" if handle.attrs.get("tiling_backend") == "CLAM" else "all_eligible_grid")
        assert len(coords) == handle.attrs["eligible_count"]
    coverage = np.zeros(clean.shape, dtype=bool)
    for x, y in expected:
        x0, x1 = int(np.floor(x * mw / width)), min(mw, int(np.ceil((x + size) * mw / width)))
        y0, y1 = int(np.floor(y * mh / height)), min(mh, int(np.ceil((y + size) * mh / height)))
        coverage[y0:y1, x0:x1] = True
    return coverage


def verify_coordinate_coverage(handle, coords, clean, dimensions, size):
    """Independently reconstruct eligibility from the saved real-image mask."""
    if handle.attrs.get("tiling_backend") == "CLAM":
        return verify_upstream_clam_coverage(handle, coords, clean, dimensions, size)
    width, height = dimensions
    mh, mw = clean.shape
    method = handle.attrs.get("tiling_method", "coverage_grid")
    if method == "clam_four_pt":
        assert not handle.attrs["use_padding"]
        candidates = clam_candidates(clean, dimensions, size, handle.attrs["max_candidates"])
        complete_selection = "all_eligible_clam_contour_grid"
    else:
        assert method == "coverage_grid", "Unknown tiling method"
        assert (width // size) * (height // size) <= handle.attrs["max_candidates"]
        candidates = ((x, y) for y in range(0, height - size + 1, size)
                      for x in range(0, width - size + 1, size))
        complete_selection = "all_eligible_grid"
    expected, fractions = [], []
    for x, y in candidates:
        y0, y1 = int(np.floor(y * mh / height)), min(mh, int(np.ceil((y + size) * mh / height)))
        x0, x1 = int(np.floor(x * mw / width)), min(mw, int(np.ceil((x + size) * mw / width)))
        fraction = float(clean[y0:y1, x0:x1].mean())
        if fraction >= handle.attrs["min_tissue_fraction"]:
            expected.append((x, y))
            fractions.append(fraction)
    expected = np.asarray(expected, dtype=np.int64).reshape(-1, 2)
    fractions = np.asarray(fractions)
    assert len(expected) == handle.attrs["eligible_count"]
    limit = int(handle.attrs["coordinate_limit"])
    if limit >= 0 and len(expected) > limit:
        indices = np.linspace(0, len(expected) - 1, limit, dtype=int)
        expected, fractions = expected[indices], fractions[indices]
    assert np.array_equal(coords, expected), "Exported coordinates differ from the complete eligible grid"
    assert np.allclose(handle["tissue_fraction"][:], fractions, rtol=0, atol=1e-12)
    if limit == -1:
        assert handle.attrs["selection"] == complete_selection
        assert len(coords) == handle.attrs["eligible_count"]
    coverage = np.zeros(clean.shape, dtype=bool)
    for x, y in expected:
        x0, x1 = int(np.floor(x * mw / width)), min(mw, int(np.ceil((x + size) * mw / width)))
        y0, y1 = int(np.floor(y * mh / height)), min(mh, int(np.ceil((y + size) * mh / height)))
        coverage[y0:y1, x0:x1] = True
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--check-regions", action="store_true", help="Compare saved regions and first/last individual patches with fresh OpenSlide reads")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert (args.run / "index.html").is_file()
    assert manifest["slides"]
    selected_pixel_slides = set()
    calibrated_targets = set()
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
        if slide.get("preprocessing", {}).get("method") == "codait_tia_clam":
            import cv2

            stages = {name: np.asarray(Image.open(args.run / path))
                      for name, path in slide["preprocessing_stages"].items()}
            pre_excluded = stages["codait_excluded"] > 0
            if "prefilter_excluded_pixels" in slide["preprocessing"]:
                assert slide["preprocessing"]["prefilter_excluded_pixels"] == int(pre_excluded.sum())
                assert np.isclose(slide["preprocessing"]["prefilter_excluded_fraction"], pre_excluded.mean())
            assert np.array_equal(stages["codait_filtered"][~pre_excluded], rgb[~pre_excluded])
            assert np.all(stages["codait_filtered"][pre_excluded] == 255)
            tia = stages["tia_otsu"] > 0
            if slide["preprocessing"]["tia_enabled"]:
                gray = cv2.cvtColor(stages["codait_filtered"], cv2.COLOR_RGB2GRAY)
                assert np.array_equal(tia, gray < slide["preprocessing"]["tia_threshold"])
            else:
                assert tia.all()
            assert np.all(stages["clam_input"][~tia] == 255)
            assert np.array_equal(stages["clam_input"][tia], stages["codait_filtered"][tia])
            before, filled = stages["before_holes"] > 0, stages["filled_holes"] > 0
            assert not np.any(before & filled)
            assert np.array_equal(tissue, before | filled)
            assert not np.any(clean & pre_excluded), "Hole filling must not restore CODAIT exclusions"
            assert int(filled.sum()) == slide["preprocessing"]["filled_hole_pixels"]
            if slide["preprocessing"]["max_hole_area_mask_pixels"] == 0:
                assert not filled.any()
            # Check filled components against the unfilled real-image background.
            _, labels, stats, _ = cv2.connectedComponentsWithStats((~before).astype(np.uint8), connectivity=8)
            edge_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
            ids = np.unique(labels[filled])
            assert not np.isin(ids, edge_labels).any()
            assert np.all(stats[ids, cv2.CC_STAT_AREA] <= slide["preprocessing"]["max_hole_area_mask_pixels"])
            print(f"Verified combined stage order, exclusions and {int(filled.sum())} filled-hole pixels")
        if "hole_visualization" in slide:
            import cv2

            hole_values = np.asarray(Image.open(folder / "hole_mask.png"))
            assert hole_values.shape == clean.shape and set(np.unique(hole_values)).issubset({0, 255})
            holes = hole_values > 0
            assert not np.any(holes & clean)
            assert not (holes[0].any() or holes[-1].any() or holes[:, 0].any() or holes[:, -1].any())
            # Independent topology check: label background components and exclude edge-connected labels.
            _, labels = cv2.connectedComponents((~clean).astype(np.uint8), connectivity=4)
            edge_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
            expected_holes = ~clean & ~np.isin(labels, edge_labels)
            assert np.array_equal(holes, expected_holes), "Hole mask includes exterior or misses enclosed background"
            preview = np.asarray(Image.open(folder / "tissue_holes.png"))
            assert preview.shape == rgb.shape
            if slide["hole_visualization"] == "retained_tissue_green_hole_blue_contours_4_connected_thumbnail":
                contours, hierarchy = cv2.findContours(clean.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                tissue_lines = np.zeros(clean.shape, dtype=np.uint8)
                hole_lines = np.zeros(clean.shape, dtype=np.uint8)
                if hierarchy is not None:
                    for is_hole, lines in ((False, tissue_lines), (True, hole_lines)):
                        selected = [contour for contour, relation in zip(contours, hierarchy[0])
                                    if (relation[3] >= 0) == is_hole]
                        cv2.drawContours(lines, selected, -1, 1, 1, lineType=cv2.LINE_8)
                green = (tissue_lines > 0) & (hole_lines == 0)
                blue = hole_lines > 0
                assert np.array_equal(preview[~green & ~blue], rgb[~green & ~blue]), "Contour preview changes original pixels outside contour lines"
                assert np.all(preview[green] == [0, 255, 0]), "Tissue exterior contour must be RGB green"
                assert np.all(preview[blue] == [0, 0, 255]), "Hole contour must be RGB blue"
                assert bool(hole_lines.any()) == bool(holes.any()), "Hole topology and child contours disagree"
            else:  # Historical filled previews remain inspectable.
                assert np.array_equal(preview[~clean & ~holes], rgb[~clean & ~holes])
                if clean.any():
                    assert np.any(np.all(preview == [0, 200, 80], axis=2)), "Missing green tissue contour"
                if holes.any():
                    assert np.any(np.all(preview == [220, 0, 220], axis=2)), "Missing magenta hole contour"
        if "gallery" in slide:
            gallery = args.run / slide["gallery"]
            for name, reference in (("thumbnail", rgb), ("mask", clean.astype(np.uint8) * 255),
                                    ("preprocessed", filtered)):
                saved = np.asarray(Image.open(gallery / f"{slide['gallery_prefix']}__{name}.png"))
                assert np.array_equal(saved, reference), "Gallery view differs from per-slide artifact"
            if "hole_visualization" in slide:
                saved = np.asarray(Image.open(gallery / f"{slide['gallery_prefix']}__tissue_holes.png"))
                assert np.array_equal(saved, preview)
        for layout in slide.get("exports", {}).get("layouts", []):
            import h5py

            if layout.get("mask_preview_type") == "retained_tissue_and_enclosed_background_holes":
                import io

                mask_path = args.run / layout["folder"] / "masks" / (Path(slide["input"]).stem + ".jpg")
                expected_jpeg = io.BytesIO()
                Image.fromarray(preview).save(expected_jpeg, format="JPEG", quality=90)
                assert mask_path.read_bytes() == expected_jpeg.getvalue(), "CLAM mask preview differs from tissue/hole visualization"
            with h5py.File(args.run / layout["coords"], "r") as handle:
                coords = handle["coords"][:]
                attrs = dict(handle["coords"].attrs)
                assert coords.shape == (layout["exported_count"], 2)
                assert np.issubdtype(coords.dtype, np.integer)
                assert attrs["patch_level"] == 0
                source_size = int(attrs["patch_size"])
                size = int(attrs.get("output_patch_size", source_size))
                assert size == layout["patch_size"]
                assert np.all(coords >= 0)
                assert np.all(coords + source_size <= np.asarray(slide["dimensions"]))
                assert np.all(handle["tissue_fraction"][:] >= handle.attrs["min_tissue_fraction"])
                preview_indices = np.arange(len(coords))  # Legacy outputs read every exported coordinate.
                if "preview_indices" in handle:
                    coverage = verify_coordinate_coverage(handle, coords, clean, slide["dimensions"], source_size)
                    preview_indices = handle["preview_indices"][:]
                    expected_indices = np.linspace(0, len(coords) - 1,
                        min(len(coords), int(handle.attrs["pixel_preview_limit"])), dtype=np.int64)
                    assert np.array_equal(preview_indices, expected_indices)
                    assert len(preview_indices) == handle.attrs["pixel_preview_count"] == layout["pixel_preview_count"]
                    assert len(coords) == layout["exported_count"]
                    assert handle.attrs["eligible_count"] == layout["eligible_count"]
                    if "tiling_method" in handle.attrs:
                        assert handle.attrs["tiling_method"] == layout["tiling_method"]
                        assert not handle.attrs["use_padding"]
                    saved_coverage = np.asarray(Image.open(args.run / layout["coverage_mask"]))
                    assert np.array_equal(saved_coverage, coverage.astype(np.uint8) * 255)
                    assert layout["stitch_type"] == "thumbnail_pixels_within_exported_coordinate_coverage"
                    stitch = Image.open(args.run / layout["folder"] / "stitches" / (Path(slide["input"]).stem + ".jpg"))
                    assert stitch.size == (rgb.shape[1], rgb.shape[0])
                    if layout["sampled_pixel_stitch"]:
                        sampled = Image.open(args.run / layout["sampled_pixel_stitch"])
                        assert stitch.size == sampled.size
                    if handle.attrs.get("export_schema_version", 0) >= 3:
                        assert bool(handle.attrs["pixel_slide_selected"]) == layout["pixel_slide_selected"]
                        if layout["pixel_slide_selected"]:
                            selected_pixel_slides.add(slide["input"])
                        else:
                            assert len(preview_indices) == 0 and layout["pixel_preview_limit"] == 0
                            assert layout["sampled_pixel_stitch"] is None and layout["patch_images"] is None
                            assert layout["patch_image_count"] == layout["region_count"] == 0
                            assert not layout["region_tar"]
                            export_folder = args.run / layout["folder"]
                            stem = Path(slide["input"]).stem
                            assert not (export_folder / "sampled_pixel_stitches" / (stem + ".jpg")).exists()
                            assert not (export_folder / "patch_images" / stem).exists()
                            assert not (export_folder / "tar_patch_4096" / (stem + ".tar")).exists()
                    if handle.attrs.get("export_schema_version", 0) >= 4:
                        native_mpp = np.asarray(slide["mpp"], dtype=float)
                        target_mpp = float(attrs["target_mpp"])
                        assert native_mpp.shape == (2,) and np.all(np.isfinite(native_mpp)) and np.all(native_mpp > 0)
                        assert np.all(native_mpp <= target_mpp * 1.01)
                        assert np.isclose(native_mpp[0], native_mpp[1], rtol=1e-6, atol=0)
                        assert source_size == max(1, round(size * target_mpp / native_mpp[0])) == layout["source_patch_size"]
                        assert np.array_equal(attrs["native_mpp"], native_mpp)
                        assert np.array_equal(layout["native_mpp"], native_mpp)
                        effective = source_size * native_mpp / size
                        assert np.allclose(attrs["effective_mpp"], effective, rtol=0, atol=1e-12)
                        assert np.allclose(layout["effective_mpp"], effective, rtol=0, atol=1e-12)
                        assert np.all(np.abs(effective - target_mpp) <= native_mpp / (2 * size) + 1e-12)
                        assert layout["target_mpp"] == target_mpp
                        assert attrs["resampling"] == layout["resampling"] == "Pillow.LANCZOS_from_level0"
                        if len(preview_indices):
                            assert source_size ** 2 <= handle.attrs["max_patch_read_pixels"]
                        calibrated_targets.add(target_mpp)
                        print(f"Verified calibration: {source_size} native pixels -> {size} output pixels; effective MPP {effective.tolist()}")
                    assert layout["region_count"] == (len(preview_indices) if layout["region_tar"] else 0)
                    print(f"Verified {len(coords)} coordinates ({layout['eligible_count']} eligible), {len(preview_indices)} pixel previews at size {size}")
            if layout.get("patch_images"):
                patch_folder = args.run / layout["patch_images"]
                records = json.loads((patch_folder / "index.json").read_text())
                assert size == 256
                assert len(records) == len(preview_indices) == layout["patch_image_count"]
                assert len(list(patch_folder.glob("*.png"))) == len(records)
                for record, coordinate_index in zip(records, preview_indices, strict=True):
                    x, y = map(int, coords[coordinate_index])
                    assert record["file"] == f"{x}_{y}.png"
                    assert record["coordinate_index"] == int(coordinate_index)
                    assert (record["x"], record["y"], record["patch_size"], record["patch_level"]) == (x, y, size, 0)
                    assert Image.open(patch_folder / record["file"]).size == (size, size)
                    if "source_patch_size" in record:
                        assert record["source_patch_size"] == source_size and record["target_mpp"] == layout["target_mpp"]
                        assert record["effective_mpp"] == layout["effective_mpp"]
                        assert record["resampling"] == layout["resampling"]
                if records and args.check_regions:
                    import openslide

                    with openslide.OpenSlide(slide["input"]) as wsi:
                        for index in sorted({0, len(records) - 1}):
                            record = records[index]
                            patch = np.asarray(Image.open(patch_folder / record["file"]).convert("RGB"))
                            reference = np.asarray(wsi.read_region((record["x"], record["y"]), 0,
                                (source_size, source_size)).convert("RGB").resize((size, size), Image.Resampling.LANCZOS))
                            assert np.array_equal(patch, reference), "Individual patch differs from the declared source read/resampling"
                    print(f"Verified first/last saved {size}x{size} patches against source WSI and declared resampling")
            if layout["region_tar"]:
                tar_path = args.run / layout["folder"] / "tar_patch_4096" / (Path(slide["input"]).stem + ".tar")
                with tarfile.open(tar_path) as archive:
                    members = [m for m in archive.getmembers() if m.name.endswith(".png")]
                    assert len(members) == len(preview_indices) == layout.get("region_count", len(coords))
                    for member, coordinate_index in zip(members, preview_indices, strict=True):
                        x, y = map(int, coords[coordinate_index])
                        key = f"{Path(slide['input']).stem}_{x}_{y}"
                        assert member.name == key + ".png"
                        record = json.load(archive.extractfile(key + ".json"))
                        assert (record["x"], record["y"]) == (x, y)
                        assert record["patch_size"] == size and record["patch_level"] == 0
                        if "coordinate_index" in record:
                            assert record["coordinate_index"] == int(coordinate_index)
                        if "source_patch_size" in record:
                            assert record["source_patch_size"] == source_size and record["target_mpp"] == layout["target_mpp"]
                            assert record["effective_mpp"] == layout["effective_mpp"]
                            assert record["resampling"] == layout["resampling"]
                    if members and args.check_regions:
                        import openslide

                        region = np.asarray(Image.open(archive.extractfile(members[0])).convert("RGB"))
                        with openslide.OpenSlide(slide["input"]) as wsi:
                            reference = np.asarray(wsi.read_region(tuple(map(int, coords[preview_indices[0]])), 0,
                                (source_size, source_size)).convert("RGB").resize((size, size), Image.Resampling.LANCZOS))
                        assert region.shape == (size, size, 3)
                        assert np.array_equal(region, reference), "Saved region differs from the declared source read/resampling"
                        print(f"Verified {size}x{size} region against source WSI and declared resampling")
        print(f"Verified {slide['name']}: {rgb.shape[1]}x{rgb.shape[0]}, retained={clean.mean():.1%}")
    assert len(calibrated_targets) <= 1, "Run mixes target patch resolutions"
    assert len(selected_pixel_slides) <= 1, "Source-pixel images were selected for multiple slides"
    print(f"Verified {len(manifest['slides'])} real-input artifact sets; no biological accuracy claim.")


if __name__ == "__main__":
    main()

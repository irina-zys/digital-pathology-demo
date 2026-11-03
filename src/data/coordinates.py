"""Calibrated coordinate selection with source identity checks."""
from pathlib import Path
import h5py
import numpy as np
import openslide
from src.data.manifests import read_wsi_manifest
from src.wsi_preprocessing.tiling import calibrated_patch_size
from src.utils.files import sha256


def select_coordinates(row, count, target_mpp, max_pixels):
    """Verify one existing export and record a fixed, evenly indexed subset."""
    path = Path(row["coords_path"])
    if path.stat().st_size > 20_000_000:
        raise ValueError("Coordinate HDF5 exceeds the demo's 20 MB bound")
    digest = sha256(path)
    with h5py.File(path, "r") as handle:
        coords = handle["coords"]
        attrs = coords.attrs
        if (coords.ndim != 2 or coords.shape[1] != 2 or coords.dtype.kind not in "iu"
                or not 1 <= len(coords) <= 200_000):
            raise ValueError("Expected 1–200,000 integer xy coordinate pairs")
        if (handle.attrs.get("coordinate_space") != "level0_xy_top_left"
                or int(attrs.get("patch_level", -1)) != 0
                or int(attrs.get("output_patch_size", -1)) != 256
                or bool(handle.attrs.get("use_padding", True))):
            raise ValueError("Expected unpadded level-0 coordinates for 256-pixel output tiles")
        if attrs.get("name") != row["bag_id"]:
            raise ValueError("Coordinate export slide identity differs from the bag")
        if int(handle.attrs.get("eligible_count", -1)) != len(coords):
            raise ValueError("Use a complete eligible coordinate export, not a capped preview")
        if not np.isclose(float(attrs.get("target_mpp", np.nan)), target_mpp, atol=1e-8, rtol=0):
            raise ValueError("Coordinate target MPP differs from the requested scale")
        native = np.asarray(attrs["native_mpp"], dtype=float)
        source_size, effective = calibrated_patch_size(256, target_mpp, native)
        stored_size = np.asarray(attrs["patch_size"])
        if not np.all(stored_size == np.array(source_size)) or source_size[0] * source_size[1] > max_pixels:
            raise ValueError("Invalid or over-budget native patch footprint")
        if not np.allclose(attrs["effective_mpp"], effective, atol=1e-8, rtol=0):
            raise ValueError("Exported effective MPP disagrees with its footprint")
        dimensions = np.asarray(attrs["level_dim"], dtype=np.int64)
        indices = np.linspace(0, len(coords) - 1, min(count, len(coords)), dtype=int)
        selected = coords[indices]
        if (dimensions.shape != (2,) or np.any(selected < 0)
                or np.any(selected + source_size > dimensions)
                or len(np.unique(selected, axis=0)) != len(selected)):
            raise ValueError("Selected coordinates are duplicated or outside the slide")
        fractions = np.asarray(handle["tissue_fraction"][indices])
        if not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
            raise ValueError("Invalid recorded tissue fractions")
        result = dict(indices=indices.tolist(), coords=selected.tolist(), source_size=list(source_size),
                      dimensions=dimensions.tolist(), native_mpp=native.tolist(), effective_mpp=effective,
                      target_mpp=target_mpp, output_patch_size=256, selected_count=len(indices),
                      eligible_count=len(coords), tissue_fraction=fractions.tolist(),
                      coordinate_sha256=digest, selection="evenly_spaced_export_row_indices",
                      export_schema_version=int(handle.attrs.get("export_schema_version", -1)),
                      masking_method=str(handle.attrs.get("masking_method", "unknown")),
                      tiling_method=str(handle.attrs.get("tiling_method", "unknown")))
    with openslide.OpenSlide(row["wsi_path"]) as slide:
        actual_mpp = [float(slide.properties[f"openslide.mpp-{axis}"]) for axis in ("x", "y")]
        if tuple(slide.dimensions) != tuple(dimensions) or not np.allclose(actual_mpp, native, atol=1e-8, rtol=0):
            raise ValueError("Source WSI dimensions or calibration differ from the coordinate export")
    info = Path(row["wsi_path"]).stat()
    if row.get("source_size_bytes") and int(row["source_size_bytes"]) != info.st_size:
        raise ValueError("Source WSI size differs from its declared manifest")
    result.update(source_bytes=info.st_size, source_mtime_ns=info.st_mtime_ns,
                  source_checksum_verified=False)
    return result


def complete_coordinate_manifest(manifest, project, *, class_names=("LUAD", "LUSC"), target_mpp=.5):
    """Validate source/split identities, then select every eligible coordinate."""
    if not np.isfinite(target_mpp) or target_mpp <= 0:
        raise ValueError("target_mpp must be finite and positive")
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("Set ordered unique class_names")
    rows, samples = read_wsi_manifest(manifest, project, class_names=class_names)
    if any(not items for items in samples.values()):
        raise ValueError("Complete comparison requires explicit train, val and test splits")
    result = []
    for row in rows:
        selection = select_coordinates(row, 200_000, target_mpp, 1_000_000)
        count = selection["eligible_count"]
        if selection["indices"] != list(range(count)):
            raise ValueError("All-coordinate coverage is incomplete")
        result.append({"row": row, "selection": selection, "eligible_patches": count})
    return result

"""Step 1: bounded slide reading, pyramid inspection and physical metadata."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.image_utils import validate_rgb


def read_thumbnail(path, max_size, max_read_pixels, max_total_read_pixels):
    """Read an explicitly bounded pyramid level; never decode level 0 blindly."""
    path = Path(path)
    if not 1 <= max_size <= 4096 or not 1 <= max_read_pixels <= max_total_read_pixels <= 200_000_000:
        raise ValueError("Invalid thumbnail edge or pixel-read budgets")
    if not path.is_file():
        raise FileNotFoundError(f"Input missing: {path}. Run the download script first.")
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        with Image.open(path) as source:
            dimensions = source.size
            if source.width * source.height > max_read_pixels:
                raise ValueError(f"Raster exceeds thumbnail.max_read_pixels: {path}")
            image = source.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            image = background.convert("RGB")
        metadata = {"reader": "Pillow", "dimensions": list(dimensions), "read_level": 0,
                    "read_dimensions": list(dimensions), "mpp": None, "vendor": None,
                    "objective_power": None}
    else:
        from tiatoolbox.wsicore.wsireader import WSIReader

        reader = WSIReader.open(path)
        try:
            info = reader.info
            levels = [tuple(map(int, size)) for size in info.level_dimensions]
            safe = [i for i, (w, h) in enumerate(levels) if w * h <= max_read_pixels]
            sufficient = [i for i in safe if max(levels[i]) >= max_size]
            level = max(sufficient) if sufficient else (min(safe) if safe else len(levels) - 1)
            width, height = levels[level]
            if width * height > max_total_read_pixels:
                raise ValueError("Lowest usable level exceeds thumbnail.max_total_read_pixels")
            if safe:
                array = reader.read_bounds((0, 0, *info.slide_dimensions), resolution=level, units="level")
                validate_rgb(array)
                image = Image.fromarray(array)
            else:
                # Some real SVS files have only two levels. Resize bounded blocks.
                scale = min(1.0, max_size / max(width, height))
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                image = Image.new("RGB", size, "white")
                edge = int(np.sqrt(max_read_pixels))
                downsample = float(info.level_downsamples[level])
                for y in range(0, height, edge):
                    for x in range(0, width, edge):
                        bw, bh = min(edge, width - x), min(edge, height - y)
                        box = (round(x * size[0] / width), round(y * size[1] / height),
                               round((x + bw) * size[0] / width), round((y + bh) * size[1] / height))
                        if box[2] == box[0] or box[3] == box[1]:
                            continue
                        array = reader.read_rect((round(x * downsample), round(y * downsample)),
                                                 size=(bw, bh), resolution=level, units="level")
                        validate_rgb(array)
                        block = Image.fromarray(array).resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
                        image.paste(block, box[:2])
            metadata = {"reader": type(reader).__name__, "dimensions": list(map(int, info.slide_dimensions)),
                        "read_level": level, "read_dimensions": list(levels[level]),
                        "read_strategy": "whole_level" if safe else "bounded_blocks",
                        "level_dimensions": levels, "level_downsamples": list(map(float, info.level_downsamples)),
                        "vendor": info.vendor,
                        "objective_power": None if info.objective_power is None else float(info.objective_power),
                        "mpp": None if info.mpp is None else np.asarray(info.mpp).tolist()}
            native_slide = getattr(reader, "openslide_wsi", None)
            if native_slide is not None:
                # TIAToolbox can infer calibration from objective power; require recorded MPP.
                metadata["openslide_metadata"] = _openslide_metadata(path, native_slide)
                props = native_slide.properties
                metadata["mpp"] = ([float(props[key]) for key in ("openslide.mpp-x", "openslide.mpp-y")]
                                   if all(key in props for key in ("openslide.mpp-x", "openslide.mpp-y")) else None)
                metadata["mpp_source"] = "OpenSlide mpp-x/mpp-y" if metadata["mpp"] is not None else None
                objective = props.get("openslide.objective-power")
                metadata["objective_power"] = float(objective) if objective is not None else None
        finally:
            handle = getattr(reader, "openslide_wsi", None)
            if handle is not None:
                handle.close()
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    rgb = np.asarray(image).copy()
    metadata["thumbnail_dimensions"] = [image.width, image.height]
    metadata["level0_pixels_per_thumbnail_pixel"] = [
        metadata["dimensions"][0] / image.width, metadata["dimensions"][1] / image.height]
    return rgb, metadata


def _openslide_metadata(path, slide):
    """Collect metadata only; associated-image pixels and slide pixels are not read."""
    props = dict(slide.properties)
    def number(key):
        value = props.get(key)
        return None if value is None else float(value)
    mpp = [number("openslide.mpp-x"), number("openslide.mpp-y")]
    dimensions = list(slide.dimensions)
    profile = slide.color_profile
    return {"path": str(path.resolve()), "filename": path.name, "file_size_bytes": path.stat().st_size,
            "vendor": props.get("openslide.vendor"), "objective_power": number("openslide.objective-power"),
            "native_mpp_xy": mpp, "dimensions": dimensions,
            "physical_size_mm": [dimensions[i] * v / 1000 if v is not None else None for i, v in enumerate(mpp)],
            "level_count": slide.level_count, "level_dimensions": [list(v) for v in slide.level_dimensions],
            "level_downsamples": list(slide.level_downsamples),
            "bounds_level0": {key: props.get("openslide.bounds-" + key) for key in ("x", "y", "width", "height")},
            "associated_images": list(slide.associated_images),
            "icc_profile_present": profile is not None,
            "icc_profile_description": getattr(getattr(profile, "profile", None), "profile_description", None),
            "properties": props}


def read_slide_metadata(path):
    """Read rich OpenSlide metadata without decoding any slide/associated image.

    Absent calibration remains None; objective power is never used to invent MPP.
    Raw scanner property strings are preserved for provenance and inspection.
    """
    import openslide

    path = Path(path)
    with openslide.OpenSlide(str(path)) as slide:
        return _openslide_metadata(path, slide)

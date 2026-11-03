"""Reusable COCO and instance-mask readers, geometry conversion and dataset export."""

import logging
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
import tifffile
from PIL import Image
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union


LOG = logging.getLogger(__name__)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")


def check_size(height, width, limit):
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("Image dimensions must be positive integers")
    if height * width > limit:
        raise ValueError(f"Image {width}x{height} exceeds --max-pixels={limit}; crop or raise the bound explicitly")


def read_mask(path, limit):
    """Read a bounded 2D instance mask as uint32 without changing positive IDs."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as handle:
            if len(handle.pages) != 1 or len(handle.pages[0].shape) != 2:
                raise ValueError(f"Expected a single 2D label image: {path}")
            check_size(*handle.pages[0].shape, limit)
            labels = handle.asarray()
    else:
        with Image.open(path) as image:
            check_size(image.height, image.width, limit)
            labels = np.array(image)
    if labels.ndim != 2 or labels.dtype.kind not in "ui" or labels.min() < 0:
        raise ValueError(f"Expected a nonnegative 2D integer instance mask: {path}")
    if int(labels.max()) > np.iinfo(np.uint32).max:
        raise ValueError(f"Instance IDs exceed uint32: {path}")
    return labels.astype(np.uint32, copy=False)


def mask_objects(labels):
    """Build pixel-edge geometries, preserving instance IDs, holes and disconnected parts."""
    # Union pixel runs, rather than centre-line contours, to retain one-pixel objects and holes exactly.
    runs = defaultdict(list)
    for y, row in enumerate(labels):
        changes = np.flatnonzero(row[1:] != row[:-1]) + 1
        edges = np.concatenate(([0], changes, [len(row)]))
        for left, right in zip(edges[:-1], edges[1:]):
            label = int(row[left])
            if label:
                runs[label].append(box(int(left), y, int(right), y + 1))
    return [dict(instance_id=label, source_label=label, geometry=unary_union(parts))
            for label, parts in sorted(runs.items())]


def coco_samples(path, limit, maximum):
    """Yield validated COCO images with polygon geometries and source IDs/classes."""
    data = json.loads(path.read_text())
    required = {"images": {"id", "file_name", "height", "width"},
                "annotations": {"id", "image_id", "category_id", "segmentation"},
                "categories": {"id", "name"}}
    if not isinstance(data, dict):
        raise ValueError("COCO input must be an object containing images, annotations and categories")
    for name, fields in required.items():
        items = data.get(name)
        if not isinstance(items, list) or any(not isinstance(item, dict) or not fields <= item.keys() for item in items):
            raise ValueError(f"COCO {name} must be a list of objects with fields {sorted(fields)}")
        ids = [item["id"] for item in items]
        if any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"COCO {name} IDs must be unique nonnegative integers")
    images, annotations, categories = data["images"], data["annotations"], data["categories"]
    if any(not isinstance(image["file_name"], str) or not image["file_name"] for image in images):
        raise ValueError("COCO image file_name must be a nonempty string")
    if any(not isinstance(category["name"], str) or not category["name"] for category in categories):
        raise ValueError("COCO category name must be a nonempty string")
    image_ids = {image["id"] for image in images}
    class_names = {category["id"]: category["name"] for category in categories}
    by_image = defaultdict(list)
    for annotation in annotations:
        if (type(annotation["image_id"]) is not int or type(annotation["category_id"]) is not int or
                annotation["image_id"] not in image_ids or annotation["category_id"] not in class_names):
            raise ValueError(f"Unknown image/category in annotation {annotation['id']}")
        by_image[annotation["image_id"]].append(annotation)
    for image in images[:maximum]:
        height, width = image["height"], image["width"]
        check_size(height, width, limit)
        objects = []
        for label, annotation in enumerate(by_image[image["id"]], 1):
            segmentation = annotation["segmentation"]
            if not isinstance(segmentation, list) or not segmentation:
                raise ValueError(f"Annotation {annotation['id']}: expected COCO polygon lists; RLE is not supported")
            polygons = []
            for ring in segmentation:
                coords = np.asarray(ring, dtype=float)
                if coords.ndim != 1 or coords.size < 6 or coords.size % 2 or not np.isfinite(coords).all():
                    raise ValueError(f"Invalid polygon coordinates in annotation {annotation['id']}")
                polygon = Polygon(coords.reshape(-1, 2))
                if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
                    raise ValueError(f"Invalid polygon in annotation {annotation['id']}; repair the source explicitly")
                polygons.append(polygon)
            objects.append(dict(instance_id=label, source_annotation_id=annotation["id"],
                                category_id=annotation["category_id"],
                                category_name=class_names[annotation["category_id"]],
                                iscrowd=annotation.get("iscrowd", 0), geometry=unary_union(polygons)))
        yield dict(key=f"image_{image['id']}", source_image=image["file_name"],
                   image_id=image["id"], width=width, height=height, objects=objects,
                   category_ids=list(class_names), labels=None)


def mask_samples(path, pattern, limit, maximum):
    """Yield paired label arrays and instance geometries from selected mask files."""
    paths = [path] if path.is_file() else sorted(path.glob(pattern))
    paths = [item for item in paths if item.is_file()]
    if not paths:
        raise ValueError(f"No input masks matched: {path} / {pattern}")
    for mask_path in paths[:maximum]:
        labels = read_mask(mask_path, limit)
        relative = Path(mask_path.name) if path.is_file() else mask_path.relative_to(path)
        yield dict(key=str(relative), source_image=str(relative), width=labels.shape[1],
                   height=labels.shape[0], objects=mask_objects(labels), labels=labels,
                   category_ids=[None])


def rasterize(objects, height, width, overlap):
    """Return labels, overlap-pixel count and clipped IDs using pixel-centre intersection."""
    labels = np.zeros((height, width), dtype=np.uint32)
    conflict = np.zeros((height, width), dtype=bool)
    clipped = []
    for obj in objects:
        geometry = obj["geometry"]
        x0, y0, x1, y1 = geometry.bounds
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            clipped.append(obj["instance_id"])
        left, top = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        right, bottom = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
        if left >= right or top >= bottom:
            continue
        # Pixel-edge coordinates; include a pixel when its centre intersects the polygon.
        inside = shapely.intersects_xy(geometry, np.arange(left, right)[None, :] + 0.5,
                                      np.arange(top, bottom)[:, None] + 0.5)
        target = labels[top:bottom, left:right]
        collision = inside & (target != 0)
        conflict[top:bottom, left:right] |= collision
        if collision.any() and overlap == "error":
            raise ValueError(f"Overlapping objects at instance {obj['instance_id']}; use GeoJSON or explicitly set --overlap first/last")
        if overlap == "first":
            inside &= target == 0
        target[inside] = obj["instance_id"]
    return labels, int(conflict.sum()), clipped


def export_sample(sample, output, formats, overlap):
    """Write one image’s annotations and return its output/quality-control record."""
    key = sample["key"]
    if not Path(key).parts or Path(key).is_absolute() or ".." in Path(key).parts:
        raise ValueError("Sample key must remain inside the output directory")
    base = output / key
    if not base.resolve().is_relative_to(output.resolve()):
        raise ValueError("Sample path resolves outside the output directory")
    base.parent.mkdir(parents=True, exist_ok=True)
    objects = sample["objects"]
    record = {k: v for k, v in sample.items() if k not in {"objects", "labels", "category_ids"}}
    record["objects"] = len(objects)
    record["outputs"] = []
    if "geojson" in formats:
        features = []
        for obj in objects:
            properties = {k: v for k, v in obj.items() if k != "geometry"}
            properties["objectType"] = "annotation"
            if "category_name" in obj:
                properties["classification"] = {"name": obj["category_name"]}
            features.append(dict(type="Feature", id=obj["instance_id"],
                                 geometry=mapping(obj["geometry"]), properties=properties))
        path = Path(str(base) + ".geojson")
        write_json(path, dict(type="FeatureCollection", coordinate_system="image-pixels",
                             origin="top-left", x_axis="column", y_axis="row-down",
                             width=sample["width"], height=sample["height"],
                             source_image=sample["source_image"], features=features))
        record["outputs"].append(str(path.relative_to(output)))
    if "instance-tiff" in formats:
        record["masks"] = []
        for category in sample["category_ids"]:
            group = [obj for obj in objects if obj.get("category_id") == category]
            if sample["labels"] is not None:
                labels, conflicts, clipped = sample["labels"], 0, []
            else:
                labels, conflicts, clipped = rasterize(group, sample["height"], sample["width"], overlap)
            suffix = "" if category is None else f".category_{category}"
            path = Path(str(base) + suffix + ".instances.tif")
            with path.open("xb") as stream:
                tifffile.imwrite(stream, labels, photometric="minisblack", compression="deflate", metadata=None)
            if not np.array_equal(tifffile.imread(path), labels):
                raise RuntimeError(f"TIFF read-back differs: {path}")
            present = set(int(v) for v in np.unique(labels) if v)
            missing = [obj["instance_id"] for obj in group if obj["instance_id"] not in present]
            mask_record = dict(path=str(path.relative_to(output)), category_id=category,
                               overlap_pixels=conflicts, clipped_instance_ids=clipped,
                               unrepresented_instance_ids=missing, foreground_pixels=int(np.count_nonzero(labels)),
                               instances=len(present), dtype="uint32", background=0)
            record["masks"].append(mask_record)
            record["outputs"].append(mask_record["path"])
        path = Path(str(base) + ".labels.json")
        write_json(path, [{k: v for k, v in obj.items() if k != "geometry"} for obj in objects])
        record["outputs"].append(str(path.relative_to(output)))
    return record


def convert_annotations(
    source: str | Path,
    output: str | Path,
    *,
    input_format: str,
    output_format: str = "both",
    pattern: str = "**/*.tif",
    max_images: int | None = None,
    max_pixels: int = 16_777_216,
    overlap: str = "error",
) -> dict:
    """Export COCO polygons or label masks and return the written run manifest.

    Source files are preserved; output must be a new directory outside a mask
    input tree. GeoJSON uses image pixel coordinates. TIFFs keep background 0
    and separate COCO categories, with an explicit within-class overlap policy.
    Invalid options raise ValueError before creating output. A conversion error
    leaves a failed manifest alongside any partial artifacts and is re-raised.
    """
    if input_format not in {"coco", "mask"}:
        raise ValueError("input_format must be 'coco' or 'mask'")
    if output_format not in {"geojson", "instance-tiff", "both"}:
        raise ValueError("output_format must be 'geojson', 'instance-tiff' or 'both'")
    if overlap not in {"error", "first", "last"}:
        raise ValueError("overlap must be 'error', 'first' or 'last'")
    for name, value in [("max_pixels", max_pixels), ("max_images", max_images)]:
        if name == "max_images" and value is None:
            continue
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("pattern must be a nonempty relative glob without parent traversal")
    if Path(output).is_symlink():
        raise FileExistsError("Output is a symlink; choose a new directory")
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    if source.is_dir() and output.is_relative_to(source):
        raise ValueError("Output must be outside the input tree")
    if output.exists():
        raise FileExistsError("Output already exists; choose a new directory")
    if input_format == "coco" and not source.is_file():
        raise ValueError("COCO input must be a JSON file")
    formats = ["geojson", "instance-tiff"] if output_format == "both" else [output_format]
    manifest = dict(status="running", source=str(source), input_format=input_format,
                    formats=formats, overlap=overlap, max_images=max_images,
                    max_pixels=max_pixels, glob=pattern if input_format == "mask" else None,
                    coordinates="image pixel edges; x=column, y=row, origin=top-left; no geographic CRS",
                    rasterization="pixel-centre intersection; clipped to image extent; separate TIFF per COCO category",
                    versions=dict(numpy=np.__version__, shapely=shapely.__version__, tifffile=tifffile.__version__),
                    samples=[])
    output.mkdir(parents=True)
    try:
        samples = (coco_samples(source, max_pixels, max_images) if input_format == "coco"
                   else mask_samples(source, pattern, max_pixels, max_images))
        for sample in samples:
            manifest["samples"].append(export_sample(sample, output, formats, overlap))
            LOG.info("Converted %s: %s objects", sample["key"], len(sample["objects"]))
        if not manifest["samples"]:
            raise ValueError("No images were converted")
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return manifest

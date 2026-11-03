"""Convert COCO polygons or integer instance masks to pixel GeoJSON and/or TIFF."""

import argparse
import logging
import sys
from pathlib import Path

# Keep direct script execution working without installing this source-tree project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.annotations import convert_annotations


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--input-format", choices=["coco", "mask"], required=True)
    parser.add_argument("--format", choices=["geojson", "instance-tiff", "both"], default="both")
    parser.add_argument("--glob", default="**/*.tif", help="Mask directory glob; explicitly select original instance masks")
    parser.add_argument("--max-images", type=positive, help="Process only the first N images, without changing their annotations")
    parser.add_argument("--max-pixels", type=positive, default=16_777_216, help="Per-image pixel bound")
    parser.add_argument("--overlap", choices=["error", "first", "last"], default="error",
                        help="Within-class raster overlap policy; GeoJSON always retains overlaps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        manifest = convert_annotations(
            args.input, args.output, input_format=args.input_format, output_format=args.format,
            pattern=args.glob, max_images=args.max_images, max_pixels=args.max_pixels, overlap=args.overlap,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Complete: {len(manifest['samples'])} images -> {args.output.resolve()}")


if __name__ == "__main__":
    main()

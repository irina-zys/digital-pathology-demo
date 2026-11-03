"""Download missing canonical inputs for one notebook demo; reuse existing assets."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.demo_inputs import prepare_demo_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", required=True, choices=("preprocessing", "segmentation", "wsi-classification"))
    parser.add_argument("--check-only", action="store_true", help="Inspect prerequisites without downloading or creating outputs")
    parser.add_argument("--prepare-coordinates", action="store_true", help="Create a missing bounded WSI coordinate run; never overwrite one")
    parser.add_argument("--preprocessing-run", default="outputs/tcga-mpp05")
    args = parser.parse_args()
    try:
        receipt = prepare_demo_inputs(args.demo, check_only=args.check_only,
            prepare_coordinates=args.prepare_coordinates, preprocessing_run=args.preprocessing_run)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

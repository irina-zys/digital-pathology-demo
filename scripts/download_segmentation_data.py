"""Prepare pinned CytoNuke or NuInsSeg inputs under the canonical project data tree."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    from src.data.preparation.segmentation import DATASETS, prepare_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--demo-only", action="store_true",
                        help="Extract only the three NuInsSeg notebook inputs; CytoNuke stays complete. "
                             "NuInsSeg still requires its full 1.63 GB source archive.")
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.dataset, demo_only=args.demo_only), indent=2))


if __name__ == "__main__":
    main()

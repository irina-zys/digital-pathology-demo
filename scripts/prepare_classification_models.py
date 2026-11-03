"""Explicitly acquire pinned classification encoders using the standard local HF login."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.models.pretrained_encoders import prepare_encoder_assets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['uni', 'dino', 'resnet50'], default=['uni', 'dino', 'resnet50'])
    args = parser.parse_args()
    for name in args.models:
        print(json.dumps(prepare_encoder_assets(name, PROJECT)))


if __name__ == '__main__':
    main()

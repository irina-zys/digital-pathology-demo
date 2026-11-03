"""Prepare 16 real NuInsSeg patches for mechanics checks, never a scientific split."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/nuinsseg"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.data_root / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {"train": [(species, tissue) for species in ("human", "mouse") for tissue in ("kidney", "liver")],
              "val": [("human", "spleen"), ("mouse", "spleen")],
              "test": [("human", "bladder"), ("mouse", "heart")]}
    selected = []
    for split, sources in groups.items():
        for species, tissue in sources:
            matches = [row for row in rows if row["species"] == species and row["tissue"] == tissue][:2]
            if len(matches) != 2:
                raise ValueError(f"Need two real NuInsSeg patches for {species}/{tissue}")
            for row in matches:
                selected.append(dict(image_path=row["image_path"], split=split,
                    group_id=f"{species}:{tissue}", label=0 if species == "human" else 1,
                    bag_id=f"{species}:{tissue}", mask_path=row["instance_mask_path"],
                    ignore_path=row["ambiguous_mask_path"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(f"Wrote {len(selected)} real patches / 8 source-tissue bags to {args.output}")
    print("Mechanics only: species labels; source-tissue groups are not patient identities or an official split.")


if __name__ == "__main__":
    main()

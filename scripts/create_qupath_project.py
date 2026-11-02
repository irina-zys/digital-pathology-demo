"""Create a QuPath project referencing the canonical TCGA files, without copying WSIs."""

import argparse
import csv
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qupath", required=True, type=Path, help="QuPath command-line executable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / "data/qupath/tcga-lung-demo"
    if destination.exists():
        raise FileExistsError(f"Preserving existing QuPath project: {destination}")
    with (root / "manifests/tcga_demo.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10 or len({r["case_id"] for r in rows}) != 10:
        raise ValueError("Expected ten distinct TCGA cases")
    for row in rows:
        path = (root / row["path"]).resolve(strict=True)
        if not path.is_relative_to(root / "data/tcga"):
            raise ValueError("Slide must be in the canonical data/tcga directory")
        row["absolute_path"] = str(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    roster = destination.parent / "tcga-lung-roster.json"
    roster.write_text(json.dumps(rows, indent=2) + "\n")
    subprocess.run([str(args.qupath), "script", str(root / "scripts/create_qupath_project.groovy"),
                    "--args", str(roster), "--args", str(destination)], check=True, timeout=300)
    project = destination / "project.qpproj"
    if not project.is_file():
        raise RuntimeError("QuPath did not create its project file")
    print(f"Created {project}")


if __name__ == "__main__":
    main()

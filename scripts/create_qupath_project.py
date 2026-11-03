"""Create a QuPath project referencing the canonical local files, without copying WSIs."""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qupath", required=True, type=Path, help="QuPath command-line executable")
    parser.add_argument("--manifest", type=Path, default=Path("manifests/tcga_demo.csv"))
    parser.add_argument("--project-name", default="tcga-lung-demo")
    parser.add_argument("--expected-patients", type=int, default=10)
    parser.add_argument("--expected-slides", type=int, default=10)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.project_name):
        raise ValueError("Project name must contain only letters, digits, hyphens and underscores")
    if not 1 <= args.expected_patients <= args.expected_slides <= 100:
        raise ValueError("Expected 1-100 slides and no more patients than slides")
    root = Path(__file__).resolve().parents[1]
    destination = root / "data/qupath" / args.project_name
    if destination.exists():
        raise FileExistsError(f"Preserving existing QuPath project: {destination}")
    with (root / args.manifest).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != args.expected_slides or len({r["case_id"] for r in rows}) != args.expected_patients:
        raise ValueError("Manifest does not match the expected slide and patient counts")
    paths = set()
    for row in rows:
        if not row["case_id"] or not row["case_submitter_id"]:
            raise ValueError("Each slide needs a patient ID and display ID")
        path = (root / row["path"]).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to((root / "data").resolve()):
            raise ValueError("Slide must be a file in the canonical data directory")
        if path in paths:
            raise ValueError(f"Duplicate slide path: {path}")
        paths.add(path)
        row["absolute_path"] = str(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    roster = destination.parent / f"{args.project_name}-roster.json"
    roster.write_text(json.dumps(rows, indent=2) + "\n")
    subprocess.run([str(args.qupath), "script", str(root / "scripts/create_qupath_project.groovy"),
                    "--args", str(roster), "--args", str(destination)], check=True, timeout=300)
    project = destination / "project.qpproj"
    if not project.is_file():
        raise RuntimeError("QuPath did not create its project file")
    if len(json.loads(project.read_text())["images"]) != args.expected_slides:
        raise RuntimeError("Saved QuPath project does not contain the expected number of slides")
    print(f"Created {project}")


if __name__ == "__main__":
    main()

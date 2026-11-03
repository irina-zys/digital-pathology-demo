"""Fetch/replay a bounded public GDC roster in the canonical project directory."""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.gdc.cancer.gov"
MANIFEST = ROOT / "manifests/tcga_demo.csv"
PROJECTS = ("TCGA-LUAD", "TCGA-LUSC")


def select_roster():
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": list(PROJECTS)}},
        {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
    ]}
    fields = "file_id,file_name,file_size,md5sum,cases.case_id,cases.submitter_id,cases.project.project_id,cases.samples.sample_type"
    hits = []
    while True:
        query = urllib.parse.urlencode({"filters": json.dumps(filters), "fields": fields,
                                       "size": 1000, "from": len(hits), "sort": "file_size:asc"})
        with urllib.request.urlopen(f"{API}/files?{query}", timeout=60) as response:
            page = json.load(response)["data"]
        hits.extend(page["hits"])
        if len(hits) >= page["pagination"]["total"]:
            break
        if not page["hits"]:
            raise RuntimeError("GDC pagination stopped before completion")
    candidates = []
    for hit in hits:
        if len(hit.get("cases", [])) != 1:
            continue
        case = hit["cases"][0]
        # Diagnostic primary-tumour barcode; exclude frozen TS/BS and normal slides.
        if not re.search(r"-01Z-00-DX\d+\.", hit["file_name"]):
            continue
        if not any(s.get("sample_type") == "Primary Tumor" for s in case.get("samples", [])):
            continue
        if not 30_000_000 <= hit["file_size"] <= 800_000_000:
            continue
        project = case["project"]["project_id"]
        candidates.append({"project": project, "label": project.removeprefix("TCGA-"),
                           "case_id": case["case_id"], "case_submitter_id": case["submitter_id"],
                           "site": case["submitter_id"].split("-")[1],
                           "file_id": hit["file_id"], "file_name": hit["file_name"],
                           "file_size": hit["file_size"], "md5sum": hit["md5sum"],
                           "path": f"data/tcga/{project}/{hit['file_id']}.svs"})
    rows = []
    for project in PROJECTS:
        used_sites, used_cases = set(), set()
        for target_mb in (60, 130, 250, 400, 650):
            pool = [r for r in candidates if r["project"] == project
                    and r["site"] not in used_sites and r["case_id"] not in used_cases]
            if not pool:
                raise RuntimeError(f"Insufficient distinct sites for {project}")
            chosen = min(pool, key=lambda r: (abs(r["file_size"] - target_mb * 1_000_000), r["file_id"]))
            rows.append(chosen)
            used_sites.add(chosen["site"])
            used_cases.add(chosen["case_id"])
    MANIFEST.parent.mkdir(exist_ok=True)
    with MANIFEST.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "data/manifests").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/manifests/selection.json").write_text(json.dumps({
        "queried_at_utc": datetime.now(timezone.utc).isoformat(), "api": API,
        "filters": filters, "total_hits": len(hits), "eligible_candidates": len(candidates),
        "selection": "one case per site per class; nearest to 60/130/250/400/650 MB; 30-800 MB eligibility",
        "limitations": "Convenience sample; size/site diversity does not establish biological representativeness.",
    }, indent=2) + "\n")
    return rows


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def download(row):
    path = ROOT / row["path"]
    expected_size = int(row["file_size"])
    if path.exists():
        if path.stat().st_size == expected_size and checksum(path) == row["md5sum"]:
            print(f"Verified existing {row['project']} {row['case_submitter_id']}", flush=True)
            return
        raise RuntimeError(f"Existing file failed verification; preserve and investigate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".svs.part")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                raise RuntimeError(f"Oversized partial file: {partial}")
            if offset < expected_size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                request = urllib.request.Request(f"{API}/data/{row['file_id']}", headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if offset and response.status == 206:
                        # GDC currently omits the standard 'bytes ' unit prefix.
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"(?:bytes )?(\d+)-(\d+)/(\d+)", content_range)
                        expected_range = (offset, expected_size - 1, expected_size)
                        if not match or tuple(map(int, match.groups())) != expected_range:
                            raise RuntimeError("Unexpected resume range")
                    elif response.status == 200:
                        offset = 0
                    else:
                        raise RuntimeError(f"Unexpected HTTP status {response.status}")
                    print(f"Downloading {row['project']} {row['case_submitter_id']} ({expected_size / 1e6:.1f} MB)", flush=True)
                    with partial.open("ab" if offset else "wb") as stream:
                        total = offset
                        reported = offset
                        while chunk := response.read(1024 * 1024):
                            total += len(chunk)
                            if total > expected_size:
                                raise RuntimeError("Download exceeded GDC metadata size")
                            stream.write(chunk)
                            if total - reported >= 100_000_000:
                                print(f"{row['case_submitter_id']}: {total / expected_size:.0%}", flush=True)
                                reported = total
            if partial.stat().st_size != expected_size:
                raise RuntimeError("Incomplete download")
            if checksum(partial) != row["md5sum"]:
                raise RuntimeError(f"MD5 mismatch: {partial}; investigate before retrying")
            partial.rename(path)
            print(f"Verified {row['case_submitter_id']}: {row['md5sum']}", flush=True)
            return
        except (OSError, RuntimeError) as error:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}: {error}", flush=True)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--file-id", action="append", default=[], help="Download only this frozen-roster UUID; repeat to select several")
    parser.add_argument("--max-gb", type=float, default=10)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if not math.isfinite(args.max_gb) or args.max_gb <= 0:
        parser.error("--max-gb must be a positive finite value")
    if MANIFEST.exists():
        with MANIFEST.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = select_roster()
    if Counter(row["project"] for row in rows) != Counter({p: 5 for p in PROJECTS}):
        raise ValueError("Expected five slides per project")
    if (len({row["case_id"] for row in rows}) != 10
            or len({row["file_id"] for row in rows}) != 10
            or len({row["path"] for row in rows}) != 10):
        raise ValueError("Cases, files and paths must be distinct")
    for row in rows:
        file_id = str(UUID(row["file_id"]))
        if row["path"] != f"data/tcga/{row['project']}/{file_id}.svs":
            raise ValueError("Manifest path must remain inside data/tcga")
        if not (ROOT / row["path"]).resolve().is_relative_to((ROOT / "data/tcga").resolve()):
            raise ValueError("File must stay inside the canonical TCGA data directory")
        if not re.fullmatch(r"[0-9a-f]{32}", row["md5sum"]):
            raise ValueError("Invalid MD5")
        if int(row["file_size"]) <= 0:
            raise ValueError("Invalid size")
    if args.file_id:
        requested = set(args.file_id)
        unknown = requested - {row["file_id"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown file IDs in frozen roster: {sorted(unknown)}")
        rows = [row for row in rows if row["file_id"] in requested]
    total = sum(int(row["file_size"]) for row in rows)
    print(f"{len(rows)} slides, {total / 1e9:.3f} GB; roster: {MANIFEST}", flush=True)
    if total > args.max_gb * 1e9:
        raise ValueError("Roster exceeds download budget")
    if args.select_only:
        return
    missing = sum(int(r["file_size"]) for r in rows if not (ROOT / r["path"]).exists())
    if shutil.disk_usage(ROOT).free < missing + 2_000_000_000:
        raise RuntimeError("Insufficient disk space including 2 GB headroom")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, rows))
    print(f"All {len(rows)} selected slides passed GDC size and MD5 verification.", flush=True)


if __name__ == "__main__":
    main()

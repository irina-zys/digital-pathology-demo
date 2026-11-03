"""Download the frozen public KPMP CKD multistain roster; no login or new dependencies."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/kpmp_ckd_demo.csv"
STAINS = {"H&E", "PAS", "TRI", "SIL"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(rows):
    patients = defaultdict(list)
    for row in rows:
        participant = row["case_id"]
        if not re.fullmatch(r"\d{2,3}-\d{5}", participant) or participant != row["case_submitter_id"]:
            raise ValueError("Invalid participant mapping")
        if row["access"] != "open" or row["enrollment_category"] != "CKD" or row["label"] != "CKD":
            raise ValueError("Expected only public CKD-enrolled participant files")
        for key in ("file_id", "package_id"):
            UUID(row[key])
        filename = row["file_name"]
        if Path(filename).name != filename or not filename.endswith(".svs"):
            raise ValueError("Expected safe SVS filename")
        expected_path = f"data/kpmp/CKD/{participant}/{filename}"
        expected_url = "https://atlas.kpmp.org/api/v1/file/download/" + row["package_id"] + "/" + urllib.parse.quote(filename, safe="")
        if row["path"] != expected_path or row["source_url"] != expected_url:
            raise ValueError("Manifest path or public download URL mismatch")
        if not (ROOT / row["path"]).resolve().is_relative_to((ROOT / "data/kpmp").resolve()):
            raise ValueError("File must stay inside canonical KPMP data directory")
        if not 0 < int(row["file_size"]) <= 1_000_000_000:
            raise ValueError("File exceeds one GB per-slide bound")
        if row.get("sha256") and not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid SHA-256")
        patients[participant].append(row)
    if len(rows) != 20 or len(patients) != 5 or len({r["file_id"] for r in rows}) != 20 or len({r["path"] for r in rows}) != 20:
        raise ValueError("Expected twenty distinct slides from five participants")
    for group in patients.values():
        if Counter(r["stain"] for r in group) != Counter(STAINS):
            raise ValueError("Each participant needs one H&E, PAS, TRI and SIL slide")


def download(row):
    path = ROOT / row["path"]
    receipt = path.with_suffix(".download.json")
    size = int(row["file_size"])
    expected_hash = row.get("sha256")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["source_url"] != row["source_url"] or saved["file_id"] != row["file_id"]:
            raise RuntimeError("Download receipt identity differs from roster")
        expected_hash = expected_hash or saved["sha256"]
    if path.exists():
        if not expected_hash or path.stat().st_size != size or sha256(path) != expected_hash:
            raise RuntimeError(f"Preserving existing file without matching integrity evidence: {path}")
        print(f"Verified existing {row['case_id']} {row['stain']}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".svs.part")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise RuntimeError("Partial file exceeds metadata size")
            if offset < size:
                request = urllib.request.Request(row["source_url"], headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not match or tuple(map(int, match.groups())) != (offset, size - 1, size):
                            raise RuntimeError("Unexpected partial response range")
                    elif response.status == 200:
                        # KPMP's proxy may ignore Range; safely restart the same partial file.
                        offset = 0
                    else:
                        raise RuntimeError(f"Unexpected HTTP status {response.status}")
                    print(f"Downloading {row['case_id']} {row['stain']} ({size / 1e6:.1f} MB)", flush=True)
                    with partial.open("ab" if offset else "wb") as stream:
                        count = offset
                        while chunk := response.read(1024 * 1024):
                            count += len(chunk)
                            if count > size:
                                raise RuntimeError("Download exceeds metadata size")
                            stream.write(chunk)
            if partial.stat().st_size != size:
                raise RuntimeError("Incomplete download")
            digest = sha256(partial)
            if expected_hash and digest != expected_hash:
                raise RuntimeError("SHA-256 mismatch; preserving partial for investigation")
            import openslide

            with openslide.OpenSlide(partial) as slide:
                dimensions = list(slide.dimensions)
                slide.read_region((0, 0), slide.level_count - 1, (128, 128)).load()
            # First acquisition has no upstream checksum; record local integrity explicitly.
            receipt.write_text(json.dumps({"file_id": row["file_id"], "source_url": row["source_url"],
                "bytes": size, "sha256": digest, "upstream_checksum_available": False,
                "dimensions": dimensions, "downloaded_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
            partial.rename(path)
            print(f"Verified size/readability: {row['case_id']} {row['stain']}; SHA-256 {digest}", flush=True)
            return
        except (OSError, RuntimeError) as error:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}: {error}", flush=True)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-gb", type=float, default=5)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    with MANIFEST.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    validate(rows)
    total = sum(int(r["file_size"]) for r in rows)
    print(f"5 CKD participants, 4 stains each, {total / 1e9:.3f} GB", flush=True)
    if not 0 < args.max_gb <= 20 or total > args.max_gb * 1e9:
        raise ValueError("Roster exceeds explicit download budget")
    if args.verify_only and any(not (ROOT / r["path"]).is_file() for r in rows):
        raise FileNotFoundError("Verify-only requires all twenty canonical files")
    missing = sum(int(r["file_size"]) for r in rows if not (ROOT / r["path"]).exists())
    if shutil.disk_usage(ROOT).free < missing + 2_000_000_000:
        raise RuntimeError("Insufficient free space including 2 GB headroom")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, rows))
    print("All twenty public KPMP files verified.", flush=True)


if __name__ == "__main__":
    main()

"""Explicit manifest validation, independent of Lightning and Hydra."""
import csv
from collections import defaultdict
from pathlib import Path


def read_patch_manifest(manifest, data_root, *, task, class_names):
    root = Path(data_root)
    with Path(manifest).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_path", "split", "group_id"}
        required |= {"mask_path"} if task == "segmentation" else {"label"}
        if task == "mil":
            required.add("bag_id")
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty training manifest")
    groups, seen, bags = {}, set(), {}
    samples = {split: defaultdict(list) for split in ("train", "val", "test")}
    for row in rows:
        if any(not row.get(key, "").strip() for key in required):
            raise ValueError("Manifest contains an empty required field")
        for key in ("split", "group_id", "bag_id"):
            if row.get(key) is not None:
                row[key] = row[key].strip()
        split, group = row["split"], row["group_id"]
        if split not in samples:
            raise ValueError(f"Unknown split: {split}")
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Group crosses splits: {group}")
        for key in ("image_path", "mask_path", "ignore_path"):
            if row.get(key):
                path = (root / row[key]).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Expected file: {path}")
                row[key] = str(path)
        if row["image_path"] in seen:
            raise ValueError(f"Duplicate image in manifest: {row['image_path']}")
        seen.add(row["image_path"])
        if task != "segmentation":
            label = int(row["label"])
            if not 0 <= label < len(class_names):
                raise ValueError(f"Label outside class_names: {label}")
        key = row["image_path"]
        if task == "mil":
            key = row["bag_id"]
            identity = (split, group, int(row["label"]))
            if bags.setdefault(key, identity) != identity:
                raise ValueError(f"Bag has inconsistent split/group/label: {key}")
        samples[split][key].append(row)
    return rows, {split: [sorted(value, key=lambda row: row["image_path"])
                          for value in entries.values()] for split, entries in samples.items()}


def read_wsi_manifest(manifest, data_root, *, class_names, max_slides=20):
    root, manifest = Path(data_root), Path(manifest)
    if manifest.stat().st_size > 1_000_000:
        raise ValueError("WSI manifest exceeds 1 MB")
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"wsi_path", "coords_path", "split", "group_id", "label", "bag_id"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"WSI manifest requires {sorted(required)}")
        rows = list(reader)
    if not 1 <= len(rows) <= max_slides:
        raise ValueError("WSI manifest exceeds max_slides or is empty")
    groups, seen, bags = {}, set(), set()
    samples = {split: [] for split in ("train", "val", "test")}
    for row in rows:
        if any(not row.get(key, "").strip() for key in required):
            raise ValueError("WSI manifest contains an empty required field")
        for key in ("split", "group_id", "bag_id"):
            row[key] = row[key].strip()
        split, group, bag = row["split"], row["group_id"], row["bag_id"]
        if split not in samples or groups.setdefault(group, split) != split:
            raise ValueError("Unknown split or patient group crossing splits")
        if not 0 <= int(row["label"]) < len(class_names):
            raise ValueError("Label is outside ordered class_names")
        for key in ("wsi_path", "coords_path"):
            path = (root / row[key]).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"Expected an existing file: {path}")
            row[key] = str(path)
        if row["wsi_path"] in seen or bag in bags or Path(row["wsi_path"]).stem != bag:
            raise ValueError("Duplicate source/bag or bag does not match WSI stem")
        seen.add(row["wsi_path"])
        bags.add(bag)
        samples[split].append([row])
    return rows, samples


def read_feature_manifest(manifest, data_root, *, class_names):
    manifest, root = Path(manifest), Path(data_root)
    if manifest.stat().st_size > 1_000_000:
        raise ValueError("Feature manifest exceeds 1 MB")
    required = {"split", "group_id", "bag_id", "label", "cache_path", "cache_sha256",
                "eligible_patches", "coordinate_sha256"}
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Feature manifest requires {sorted(required)}")
        rows = list(reader)
    if not 1 <= len(rows) <= 20:
        raise ValueError("Feature demo expects 1-20 slides")
    groups, seen = {}, set()
    samples = {split: [] for split in ("train", "val", "test")}
    for row in rows:
        if any(not row.get(key, "").strip() for key in required):
            raise ValueError("Feature manifest contains an empty required field")
        for key in ("split", "group_id", "bag_id"):
            row[key] = row[key].strip()
        if row["split"] not in samples or groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("Invalid split or patient crossing splits")
        if row["bag_id"] in seen or not 0 <= int(row["label"]) < len(class_names):
            raise ValueError("Duplicate bag or invalid label")
        seen.add(row["bag_id"])
        path = (root / row["cache_path"]).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Expected file: {path}")
        row["cache_path"] = str(path)
        samples[row["split"]].append([row])
    return rows, samples


def limit_samples(samples, maximum):
    return {split: items if maximum is None else items[:maximum] for split, items in samples.items()}

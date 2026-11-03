"""Exercise modular training through bounded CLI runs on supplied real inputs.

Use prepare_vision_smoke.py for a NuInsSeg mechanics manifest. WSI input must use
patient-separated calibrated coordinate exports; feature bags must be complete.
This validates execution and contracts, not biological accuracy or model ranking.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def override(key, value):
    """Quote paths and labels for Hydra without invoking a shell."""
    return f"{key}={json.dumps(value)}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, help="WSI/feature path base and local encoder project; defaults to --project")
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--wsi-manifest", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh isolated output directory")
    parser.add_argument("--patch-classes", nargs="+", default=["human", "mouse"])
    parser.add_argument("--wsi-classes", nargs="+", default=["LUAD", "LUSC"])
    args = parser.parse_args()
    for key in ("project", "patch_manifest", "patch_root", "wsi_manifest", "feature_manifest"):
        setattr(args, key, getattr(args, key).expanduser().resolve(strict=True))
    args.data_root = (args.data_root or args.project).expanduser().resolve(strict=True)
    if not (args.project / "configs/train.yaml").is_file():
        parser.error("--project must identify the checkout containing configs/train.yaml")
    for classes in (args.patch_classes, args.wsi_classes):
        if len(classes) < 2 or len(set(classes)) != len(classes) or any(not name.strip() for name in classes):
            parser.error("Class lists must contain at least two unique, nonempty labels in manifest order")
    args.output = args.output.expanduser().absolute()
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a new directory, not an existing path or symlink")
    return args


def compose_experiments(args, output, feature_dim, encoder_sha, ssl_checkpoint, ssl_sha):
    from hydra import compose, initialize_config_dir
    from hydra.utils import get_object
    from omegaconf import OmegaConf

    records = []

    def targets(value):
        found = []
        if isinstance(value, dict):
            if "_target_" in value:
                get_object(value["_target_"])
                found.append(value["_target_"])
            for item in value.values():
                found.extend(targets(item))
        elif isinstance(value, list):
            for item in value:
                found.extend(targets(item))
        return found

    experiments = [None, *sorted(p.stem for p in (args.project / "configs/experiment").glob("*.yaml"))]
    with initialize_config_dir(config_dir=str(args.project / "configs"), version_base="1.3"):
        for experiment in experiments:
            cfg = compose(config_name="train", overrides=[] if experiment is None else [f"experiment={experiment}"])
            kind = cfg.data._target_.rsplit(".", 1)[-1]
            if kind == "FeatureBagDataModule":
                cfg.data.manifest = str(args.feature_manifest)
                cfg.data.input_dim, cfg.data.encoder_sha256 = feature_dim, encoder_sha
            elif kind in {"WSIBagDataModule", "WSIContrastiveDataModule"}:
                cfg.data.manifest = str(args.wsi_manifest)
            elif kind == "PatchDataModule":
                cfg.data.manifest = str(args.patch_manifest)
            else:
                raise ValueError(f"Add explicit real-input validation for datamodule {kind}")
            is_patch = kind == "PatchDataModule"
            cfg.data.data_root = str(args.patch_root if is_patch else args.data_root)
            cfg.class_names = args.patch_classes if is_patch else args.wsi_classes
            if experiment == "finetune_local_ssl":
                cfg.model.net.encoder.source_checkpoint = str(ssl_checkpoint)
                cfg.model.net.encoder.source_checkpoint_sha256 = ssl_sha
            cfg.trainer.default_root_dir = str(output)
            cfg.logger.save_dir = str(output)
            cfg.callbacks.checkpoint.dirpath = str(output / "checkpoints")
            resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            records.append({"experiment": experiment or "default", "targets": targets(resolved)})
    (output / "compositions.json").write_text(json.dumps(records, indent=2) + "\n")
    return len(records)


def main():
    args = parse_args()
    sys.path.insert(0, str(args.project))
    import torch
    from src.data.feature_cache import read_feature_cache

    with args.feature_manifest.open(newline="") as stream:
        first = next(csv.DictReader(stream), None)
    if not first or not first.get("cache_path"):
        raise ValueError("Feature manifest needs at least one real cache_path")
    cache = (args.data_root / first["cache_path"]).resolve(strict=True)
    features, _, identity = read_feature_cache(cache)
    feature_dim, encoder_sha = int(features.shape[1]), identity["encoder"]["checkpoint_sha256"]
    del features
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONPATH": str(args.project), "PYTHONDONTWRITEBYTECODE": "1",
           "MPLCONFIGDIR": str(output / "mpl-cache"), "NUMBA_CACHE_DIR": str(output / "numba-cache"),
           "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    records = []
    report = {"scope": "Real-input execution and contract checks; not biological validation",
              "inputs": {key: {"path": str(getattr(args, key)), "sha256": sha256(getattr(args, key))}
                         for key in ("patch_manifest", "wsi_manifest", "feature_manifest")},
              "feature_dim": feature_dim, "encoder_sha256": encoder_sha, "runs": records}

    def save_report():
        (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def run(name, options, *, evaluate=False, resume=False, rejection=None, directory=None):
        run_output = directory or output / name
        command = [sys.executable, "-B", "-m", "src.runners.train",
                   *(["--config-name", "eval"] if evaluate else []), *options,
                   "trainer.accelerator=cpu", "trainer.devices=1", "num_threads=2", "data.num_workers=0",
                   f"trainer.max_epochs={2 if resume else 1}", "trainer.max_time=00:00:05:00",
                   "trainer.limit_train_batches=2", "trainer.limit_val_batches=1", "trainer.limit_test_batches=1",
                   "trainer.num_sanity_val_steps=0", "trainer.enable_progress_bar=false",
                   "trainer.enable_model_summary=false", "callbacks.checkpoint.save_last=false",
                   override("hydra.run.dir", str(run_output))]
        log = output / f"{name}.log"
        record = {"name": name, "command": command, "log": str(log)}
        records.append(record)
        start = time.monotonic()
        with log.open("x") as stream:
            try:
                process = subprocess.run(command, cwd=args.project, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, timeout=300, check=False)
            except subprocess.TimeoutExpired:
                record.update(status="timeout", seconds=round(time.monotonic() - start, 3))
                save_report()
                raise
        record.update(returncode=process.returncode, seconds=round(time.monotonic() - start, 3))
        save_report()
        if rejection:
            if process.returncode == 0 or rejection not in log.read_text():
                raise RuntimeError(f"{name} did not reject the invalid real-input derivative; inspect {log}")
            record.update(status="passed", expected_rejection=rejection)
            save_report()
            print(f"{name}: expected rejection passed", flush=True)
            return None
        if process.returncode:
            raise RuntimeError(f"{name} failed; inspect {log}")
        metrics = json.loads((run_output / "metrics.json").read_text())
        if metrics["status"] != "complete" or metrics["optimizer_steps_this_run"] != (0 if evaluate else 2):
            raise RuntimeError(f"{name} did not complete the bounded update budget")
        for filename in ("resolved_config.yaml", "input_manifest.csv", "selected_samples.csv"):
            if not (run_output / filename).is_file():
                raise RuntimeError(f"{name} is missing provenance: {filename}")
        record.update(status="passed", metrics=metrics)
        save_report()
        print(f"{name}: passed ({record['seconds']:.2f}s)", flush=True)
        return metrics

    patch = [override("data.manifest", str(args.patch_manifest)), override("data.data_root", str(args.patch_root)),
             override("class_names", args.patch_classes), "data.batch_size=2", "data.max_patches_per_bag=2", "test=true"]
    classifier = run("classification", patch)
    run("segmentation", ["experiment=segmentation", *patch])
    for pooling in ("mean", "attention"):
        run(f"image_mil_{pooling}", [f"experiment=mil_{pooling}", *patch, "data.batch_size=1"])
    wsi = [override("data.manifest", str(args.wsi_manifest)), override("data.data_root", str(args.data_root)),
           override("class_names", args.wsi_classes), "test=false"]
    run("wsi", ["experiment=finetune_resnet50", *wsi])
    ssl = run("ssl", ["experiment=simclr_wsi", *wsi])
    ssl_checkpoint = Path(ssl["checkpoint"])
    ssl_sha = sha256(ssl_checkpoint)
    transfer = ["experiment=finetune_local_ssl", *wsi,
                override("model.net.encoder.source_checkpoint", str(ssl_checkpoint)),
                override("model.net.encoder.source_checkpoint_sha256", ssl_sha)]
    run("ssl_transfer", transfer)
    feature = ["experiment=mil_features", override("data.manifest", str(args.feature_manifest)),
               override("data.data_root", str(args.data_root)), override("class_names", args.wsi_classes),
               f"data.input_dim={feature_dim}", override("data.encoder_sha256", encoder_sha), "test=true"]
    for pooling in ("mean", "attention"):
        run(f"feature_mil_{pooling}", [*feature, f"model/pooling@model.net.pooling={pooling}"])
    checkpoint = override("ckpt_path", classifier["checkpoint"])
    run("evaluation", [*patch, checkpoint], evaluate=True)
    run("resume", [*patch, checkpoint], resume=True)
    report["composed_experiments"] = compose_experiments(args, output, feature_dim, encoder_sha, ssl_checkpoint, ssl_sha)

    with args.patch_manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    original_split = rows[0]["split"]
    other = next(row for row in rows if row["split"] != original_split)
    other["group_id"] = " " + rows[0]["group_id"] + " "
    derivative = output / "leaking_manifest.csv"
    with derivative.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run("leakage", [*patch, override("data.manifest", str(derivative))], rejection="Group crosses splits")
    run("normalization_mismatch", [*patch, checkpoint, "data.mean=[0,0,0]"], evaluate=True,
        rejection="Checkpoint data contract differs")
    run("classes_mismatch", [*patch, checkpoint, override("class_names", list(reversed(args.patch_classes)))],
        evaluate=True, rejection="Checkpoint model/label contract differs")
    historical = output / "historical_contract_derivative.ckpt"
    state = torch.load(classifier["checkpoint"], map_location="cpu", weights_only=True)
    state.pop("pathology_checkpoint_version")
    torch.save(state, historical)
    del state
    run("historical_contract", [*patch, override("ckpt_path", str(historical))], evaluate=True,
        rejection="Incompatible historical checkpoint")
    run("ssl_bad_checksum", [*transfer, override("model.net.encoder.source_checkpoint_sha256", "0" * 64)],
        rejection="SSL initialization checkpoint checksum mismatch")
    metrics_path = output / "classification/metrics.json"
    digest = sha256(metrics_path)
    run("overwrite", patch, directory=output / "classification", rejection="Choose a fresh hydra.run.dir")
    if sha256(metrics_path) != digest:
        raise RuntimeError("Output-reuse rejection changed the original result")
    report["status"] = "complete"
    save_report()
    print(f"Validation complete: {output / 'validation.json'}", flush=True)


if __name__ == "__main__":
    main()

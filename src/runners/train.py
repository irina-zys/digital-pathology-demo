"""Hydra/Lightning training: python -m src.runners.train experiment=mil_mean ..."""

import csv
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf

from src.models.loading import build_model
from src.training.lightning_runtime import build_trainer, execute_training

LOG = logging.getLogger(__name__)


def run(cfg, output, *, evaluate=None):
    """Run a bounded single-device fit or checkpoint evaluation in a fresh directory."""
    if cfg.stage not in {"fit", "test"}:
        raise ValueError("stage must be fit or test")
    if evaluate is None:
        evaluate = cfg.stage == "test"
    output = Path(output)
    if output.is_symlink():
        raise ValueError("Training output must not be a symlink")
    job_name = HydraConfig.get().job.name if HydraConfig.initialized() else ("eval" if evaluate else "train")
    if output.exists() and any(p.name not in {".hydra", f"{job_name}.log"} for p in output.iterdir()):
        raise FileExistsError(f"Choose a fresh hydra.run.dir: {output}")
    if cfg.trainer.devices != 1 or cfg.trainer.get("num_nodes", 1) != 1:
        raise ValueError("This baseline runner supports one device; distributed execution is not implemented")
    if cfg.num_threads < 1:
        raise ValueError("num_threads must be positive")
    if evaluate and not cfg.ckpt_path:
        raise ValueError("Evaluation requires ckpt_path pointing to a trained checkpoint")
    checkpoint = str(Path(to_absolute_path(cfg.ckpt_path)).resolve(strict=True)) if cfg.ckpt_path else None
    torch.set_num_threads(cfg.num_threads)
    L.seed_everything(cfg.seed, workers=True)
    data = instantiate(cfg.data, _convert_="all")
    data.setup("test" if evaluate else "fit")
    if cfg.test and not evaluate:
        data.setup("test")
    model = build_model(cfg.model, checkpoint=checkpoint)
    model.validate_data(data)
    provenance = data.provenance()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run_started.json").open("x") as stream:
        json.dump({"stage": "test" if evaluate else "fit", "checkpoint": checkpoint}, stream)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    manifest_path = Path(to_absolute_path(cfg.data.manifest))
    source_manifest = manifest_path.read_bytes()
    (output / "input_manifest.csv").write_bytes(source_manifest)
    selected = provenance["selected_rows"]
    with (output / "selected_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    if "coordinate_selections" in provenance:
        (output / "sampled_coordinates.json").write_text(json.dumps(provenance["coordinate_selections"], indent=2, allow_nan=False) + "\n")
    trainer, progress = build_trainer(cfg, evaluate=evaluate)
    LOG.info("Task=%s; samples/bags by split=%s; image size=%s; manifest=%s",
             model.task, provenance["samples_by_split"], data.hparams.get("image_size"), manifest_path)
    metrics, best_path, updates = execute_training(trainer, model, data, checkpoint=checkpoint,
        evaluate=evaluate, test=cfg.test, progress=progress)
    payload = dict(status="complete", stage="test" if evaluate else "fit", task=model.task,
                   metrics=metrics, checkpoint=best_path, global_step=trainer.global_step,
                   current_epoch=trainer.current_epoch,
                   optimizer_steps_this_run=0 if evaluate else updates,
                   manifest_sha256=hashlib.sha256(source_manifest).hexdigest(),
                   samples_by_split=provenance["samples_by_split"],
                   model_contract=model.contract, data_contract=data.contract,
                   versions={name: importlib.metadata.version(name) for name in
                             ("torch", "torchvision", "timm", "lightning", "torchmetrics", "hydra-core")})
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    LOG.info("Completed: %s", output / "metrics.json")
    return payload


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg):
    run(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()

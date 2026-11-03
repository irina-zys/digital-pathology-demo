"""Hydra/Lightning training: python -m src.train experiment=mil_mean ..."""

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

LOG = logging.getLogger(__name__)


def run(cfg, output, *, evaluate=False):
    """Run a bounded single-device fit or checkpoint evaluation in a fresh directory."""
    output = Path(output)
    if output.exists() and any(p.name not in {".hydra", "train.log", "eval.log"} for p in output.iterdir()):
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
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if checkpoint and "encoder" in model_cfg:
        model_cfg["encoder"]["pretrained"] = False  # Checkpoint already contains encoder weights.
    model = instantiate(model_cfg, _convert_="all")
    if model.task != data.hparams.task:
        raise ValueError("Model and data tasks differ; select a matching experiment")
    if model.task != "segmentation" and list(model.hparams.class_names) != list(data.hparams.class_names):
        raise ValueError("Model and data class_names differ")
    if "encoder" in model_cfg and model_cfg["encoder"]["name"] == "vit_small_patch16_224.dino":
        if data.hparams.image_size != 224:
            raise ValueError("The DINO ViT-S/16 example requires data.image_size=224")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run_started.json").open("x") as stream:
        json.dump({"stage": "test" if evaluate else "fit", "checkpoint": checkpoint}, stream)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    manifest_path = Path(to_absolute_path(cfg.data.manifest))
    source_manifest = manifest_path.read_bytes()
    (output / "input_manifest.csv").write_bytes(source_manifest)
    selected = [row for dataset in data.datasets.values() for sample in dataset.samples for row in sample]
    with (output / "selected_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data.rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    callbacks = [instantiate(value) for value in cfg.callbacks.values() if value is not None]
    logger = instantiate(cfg.logger) if cfg.logger else False
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    if logger:
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    LOG.info("Task=%s; samples/bags by split=%s; image size=%s; manifest=%s",
             model.task, {k: len(v) for k, v in data.datasets.items()}, data.hparams.image_size, manifest_path)
    if evaluate:
        results = trainer.test(model=model, datamodule=data, ckpt_path=checkpoint)
        metrics = results[0]
        best_path = checkpoint
    else:
        trainer.fit(model=model, datamodule=data, ckpt_path=checkpoint)
        if trainer.global_step == 0:
            raise RuntimeError("Training performed no optimizer steps")
        if trainer.interrupted:
            raise RuntimeError("Training interrupted; no completion manifest written")
        metrics = {key: float(value.detach().cpu()) for key, value in trainer.callback_metrics.items()}
        best_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
        if cfg.test:
            if not best_path:
                raise RuntimeError("Testing after fit requires a saved best checkpoint")
            metrics.update(trainer.test(model=model, datamodule=data, ckpt_path=best_path)[0])
    payload = dict(status="complete", stage="test" if evaluate else "fit", task=model.task,
                   metrics=metrics, checkpoint=best_path, global_step=trainer.global_step,
                   current_epoch=trainer.current_epoch,
                   manifest_sha256=hashlib.sha256(source_manifest).hexdigest(),
                   samples_by_split={k: len(v) for k, v in data.datasets.items()},
                   model_contract=model.contract, data_contract=data.contract,
                   versions={name: importlib.metadata.version(name) for name in
                             ("torch", "torchvision", "timm", "lightning", "torchmetrics", "hydra-core")})
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    LOG.info("Completed: %s", output / "metrics.json")
    return payload


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg):
    run(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()

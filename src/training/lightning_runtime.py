"""Shared Trainer construction and assessed fit/evaluation execution."""
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.training.callbacks import FitProgress


def build_trainer(cfg, *, evaluate):
    callbacks = [instantiate(value) for value in cfg.callbacks.values() if value is not None]
    progress = FitProgress()
    if not evaluate:
        callbacks.append(progress)
    logger = instantiate(cfg.logger) if cfg.logger else False
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    if logger:
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    return trainer, progress


def execute_training(trainer, model, data, *, checkpoint, evaluate, test, progress):
    """Require completed work before the runner can publish its completion record."""
    if evaluate:
        results = trainer.test(model=model, datamodule=data, ckpt_path=checkpoint)
        metrics = results[0]
        best_path = checkpoint
        updates = 0
    else:
        trainer.fit(model=model, datamodule=data, ckpt_path=checkpoint)
        updates = 0 if progress.initial_step is None else trainer.global_step - progress.initial_step
        if updates <= 0:
            raise RuntimeError("Training performed no new optimizer steps; increase the epoch/step budget when resuming")
        if trainer.interrupted:
            raise RuntimeError("Training interrupted; no completion manifest written")
        metrics = {key: float(value.detach().cpu()) for key, value in trainer.callback_metrics.items()}
        best_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
        if test:
            if not best_path:
                raise RuntimeError("Testing after fit requires a saved best checkpoint")
            metrics.update(trainer.test(model=model, datamodule=data, ckpt_path=best_path)[0])
    return metrics, best_path, updates

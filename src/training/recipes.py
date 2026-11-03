"""Compose repository Hydra experiments and execute the shared Lightning entrypoint."""

import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.train import run


def run_recipe(project, experiment, manifest, output, *, overrides=(), device='cpu', test=False):
    """Notebooks and scripts use the same experiment groups as python -m src.train."""
    project, output = Path(project).resolve(), Path(output).resolve()
    values = [f'experiment={experiment}', f'data.manifest={json.dumps(str(Path(manifest).resolve()))}',
        f'data.data_root={json.dumps(str(project))}', 'class_names=[LUAD,LUSC]', 'num_threads=2',
        f'trainer.accelerator={device}', 'trainer.devices=1', 'trainer.enable_progress_bar=false',
        'trainer.enable_model_summary=false', f'trainer.default_root_dir={json.dumps(str(output))}',
        f'callbacks.checkpoint.dirpath={json.dumps(str(output/"checkpoints"))}',
        'callbacks.checkpoint.save_last=false', f'logger.save_dir={json.dumps(str(output))}',
        f'test={str(test).lower()}', *overrides]
    with initialize_config_dir(config_dir=str(project/'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=values)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return cfg, run(cfg, output)

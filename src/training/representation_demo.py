"""Run short learning recipes through the shared Hydra/Lightning entrypoint."""

import json
from pathlib import Path

from src.training.recipes import run_recipe
from src.training.history import read_epoch_history


def run_learning_recipe(project, experiment, manifest, output, *, overrides=(), device='cpu'):
    """Return actual logged epochs, trained checkpoint and encoder-update evidence."""
    cfg, result = run_recipe(project, experiment, manifest, output, overrides=overrides, device=device)
    logs = list((Path(output)/'csv').glob('version_*/metrics.csv'))
    if len(logs) != 1:
        raise ValueError('Expected one isolated epoch log')
    history = read_epoch_history(logs[0])
    if len(history) != cfg.trainer.max_epochs:
        raise RuntimeError('The requested epoch budget did not complete')
    audit = json.loads((Path(output)/'encoder_update_audit.json').read_text())
    if audit['delta_l2'] <= 0:
        raise RuntimeError('Encoder weights did not update')
    return {'experiment': experiment, 'training': result, 'history': history,
            'encoder_update': audit, 'test_evaluated': False}

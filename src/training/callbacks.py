"""Small Lightning callbacks that record what a teaching run actually updated."""

import hashlib
import json
from pathlib import Path

import lightning as L
import torch


class EncoderUpdateAudit(L.Callback):
    def __init__(self, expect_update=True):
        self.expect_update = expect_update

    def on_train_start(self, trainer, pl_module):
        self.name, parameter = next(pl_module.encoder.named_parameters())
        self.before = parameter.detach().cpu().clone()

    def on_fit_end(self, trainer, pl_module):
        parameter = dict(pl_module.encoder.named_parameters())[self.name].detach().cpu()
        delta = float(torch.linalg.vector_norm(parameter-self.before))
        if not torch.isfinite(parameter).all() or (delta > 0) != self.expect_update:
            raise RuntimeError('Encoder update witness disagrees with the training recipe')
        result = {'parameter': self.name, 'expected_update': self.expect_update, 'delta_l2': delta,
                  'before_sha256': hashlib.sha256(self.before.numpy().tobytes()).hexdigest(),
                  'after_sha256': hashlib.sha256(parameter.numpy().tobytes()).hexdigest(),
                  'scope': 'one actual encoder parameter, not only the classifier head'}
        (Path(trainer.default_root_dir)/'encoder_update_audit.json').write_text(json.dumps(result, indent=2))

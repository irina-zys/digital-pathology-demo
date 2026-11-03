"""Lightning implementation of a bounded SimCLR-style contrastive recipe."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from src.models.vision import VisionModule


def nt_xent(first, second, temperature=.2):
    """Two views per patch; exclude self-pairs, retain augmented positives and batch negatives."""
    if first.shape != second.shape or first.ndim != 2 or len(first) < 2 or temperature <= 0:
        raise ValueError('Expected paired batches of at least two examples and positive temperature')
    n = len(first)
    z = F.normalize(torch.cat((first, second)), dim=1)
    logits = (z @ z.T / temperature).masked_fill(torch.eye(2*n, device=z.device, dtype=torch.bool), float('-inf'))
    targets = (torch.arange(2*n, device=z.device) + n) % (2*n)
    return F.cross_entropy(logits, targets)


class SimCLR(VisionModule):
    """No diagnosis target is passed to the objective; projection head is pretraining-only."""
    task = 'self_supervised'

    def __init__(self, projection_dim=128, temperature=.2, lr=.0001, weight_decay=.0001):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = resnet50(weights=None)
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, projection_dim))
        self.contract = {'task': self.task, 'encoder': 'torchvision_resnet50',
                         'objective': 'nt_xent', 'projection_dim': projection_dim, 'temperature': temperature}

    def forward(self, images):
        return self.projector(self.encoder(images))

    def _step(self, batch, stage):
        n = len(batch['view1'])
        features = self(torch.cat((batch['view1'], batch['view2'])))
        loss = nt_xent(features[:n], features[n:], self.hparams.temperature)
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, batch_size=n)
        return loss

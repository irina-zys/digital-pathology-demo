"""Trainable attention MIL head for complete bags of frozen patch embeddings."""

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
from src.models.vision import VisionModule


class FeatureMIL(nn.Module):
    def __init__(self, input_dim, num_classes=2, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        if features.ndim != 2 or not len(features):
            raise ValueError('Expected one complete, nonempty [patches, features] bag')
        weights = self.attention(features).softmax(dim=0)
        return self.classifier((weights * features).sum(0, keepdim=True)), weights[:, 0]

    def forward_replaced(self, features, index, replacement):
        """Preserve the complete bag while making one patch differentiable for saliency."""
        return self(torch.cat((features[:index], replacement, features[index+1:]), dim=0))


class FeatureMILClassifier(VisionModule):
    """Lightning classifier over complete frozen-embedding bags."""
    task = 'mil'

    def __init__(self, input_dim, class_names, attention_dim=128, lr=.001, weight_decay=.0001):
        super().__init__()
        self.save_hyperparameters()
        self.network = FeatureMIL(input_dim, len(class_names), attention_dim)
        self.accuracy = nn.ModuleDict({f'{stage}_metric': MulticlassAccuracy(num_classes=len(class_names), average='micro')
                                      for stage in ('train', 'val', 'test')})
        self.contract = {'task': 'mil', 'input_kind': 'complete_frozen_features', 'input_dim': input_dim,
                         'class_names': list(class_names), 'attention_dim': attention_dim}

    def forward(self, features):
        if features.ndim != 3 or len(features) != 1:
            raise ValueError('Expected one [1, patches, features] bag')
        return self.network(features[0])[0]

    def _step(self, batch, stage):
        logits = self(batch['features'])
        loss = F.cross_entropy(logits, batch['target'])
        self.accuracy[f'{stage}_metric'].update(logits, batch['target'])
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, batch_size=1)
        self.log(f'{stage}/accuracy', self.accuracy[f'{stage}_metric'], on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        return {'id': batch['id'], 'probability': self(batch['features']).softmax(-1)}

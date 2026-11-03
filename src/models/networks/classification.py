"""Plain PyTorch image classifiers and feature-bag readouts."""
import torch
from torch import nn


class MILHead(nn.Module):
    """Shared pooling/readout, with full-bag attention and saliency access."""
    task = "mil"
    input_key = "features"

    def __init__(self, input_dim, class_names, pooling):
        super().__init__()
        if input_dim < 1 or len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("Set positive input_dim and ordered unique class_names")
        self.class_names = list(class_names)
        self.classifier = nn.Linear(input_dim, len(class_names))
        self.pooling = pooling(input_dim=input_dim)
        self.contract = {"kind": "feature_mil", "input_dim": input_dim,
                         "class_names": self.class_names, "pooling": self.pooling.contract}

    def forward_with_attention(self, features):
        if features.ndim != 2 or not len(features):
            raise ValueError("Expected one nonempty [patches, features] bag")
        pooled, weights = self.pooling(features)
        return self.classifier(pooled), weights

    def forward(self, features):
        if features.ndim != 3 or len(features) != 1:
            raise ValueError("Expected one [1, patches, features] bag")
        return self.forward_with_attention(features[0])[0]

    def forward_replaced(self, features, index, replacement):
        if not 0 <= index < len(features) or replacement.shape != features[index:index + 1].shape:
            raise ValueError("Replacement must match one existing patch embedding")
        return self.forward_with_attention(torch.cat((features[:index], replacement, features[index + 1:]), dim=0))


class ImageClassifier(nn.Module):
    input_key = "image"

    def __init__(self, encoder, class_names, pooling=None):
        super().__init__()
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("Set at least two unique, ordered class_names")
        self.encoder = encoder
        self.class_names = list(class_names)
        self.task = "classification" if pooling is None else "mil"
        self.head = (nn.Linear(encoder.num_features, len(class_names)) if pooling is None
                     else MILHead(encoder.num_features, class_names, pooling))
        self.contract = {"kind": "image_classifier", "encoder": encoder.contract,
                         "class_names": self.class_names,
                         "pooling": None if pooling is None else self.head.pooling.contract}

    def forward(self, images):
        if self.task == "mil":
            if images.ndim != 5 or images.shape[0] != 1 or images.shape[1] == 0:
                raise ValueError("MIL expects one nonempty bag: [1, patches, 3, height, width]")
            return self.head.forward_with_attention(self.encoder(images[0]))[0]
        return self.head(self.encoder(images))

    def forward_with_attention(self, images):
        if self.task != "mil" or images.ndim != 5 or images.shape[0] != 1 or images.shape[1] == 0:
            raise ValueError("Attention requires one nonempty image bag")
        return self.head.forward_with_attention(self.encoder(images[0]))

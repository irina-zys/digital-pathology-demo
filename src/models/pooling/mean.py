"""Mean pooling over one complete or explicitly selected bag."""
from torch import nn


class MeanPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.contract = {"kind": "mean"}

    def forward(self, features):
        if features.ndim != 2 or not len(features) or features.shape[1] != self.input_dim:
            raise ValueError("Expected a nonempty [patches, input_dim] bag")
        return features.mean(0, keepdim=True), features.new_full((len(features),), 1 / len(features))

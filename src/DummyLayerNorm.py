
import torch
import torch.nn as nn

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(DummyLayerNorm, self).__init__()

    def forward(self, x):
        return x

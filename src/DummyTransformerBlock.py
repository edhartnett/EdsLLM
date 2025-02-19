import torch
import torch.nn as nn

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(DummyTransformerBlock, self).__init__()
        
    def forward(self, x):
        return x
    
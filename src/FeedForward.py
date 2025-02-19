import torch
import torch.nn as nn
import GELU as GELU
import math

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            GELU.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
           
    def forward(self, x):
        return self.layers(x)
    
#import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super(DummyGPTModel, self).__init__()
        self.tok_emb = nn.embedding(cfg["vocab_size"] cfg["emb_dim"])
        self.pos_emb = nn.embedding(cfg["context_length"] cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (seq_len, emb_dim)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x  self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(DummyTransformerBlock, self).__init__()
        
    def forward(self, x):
        return x
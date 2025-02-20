import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from LayerNorm import LayerNorm
from DummyTransformerBlock import DummyTransformerBlock
import FeedForward as FeedForward
import GELU as GELU
import TransformerBlock as TransformerBlock


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super(DummyGPTModel, self).__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (seq_len, emb_dim)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "emb_dim": 768,
        "context_length": 1024,
        "n_layers": 12,
        "n_heads": 12,
        "ffn_dim": 3072,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(txt1)
    print(batch)

    ffn = FeedForward.FeedForward(GPT_CONFIG_124M)
    x = torch.randn(2, 10, GPT_CONFIG_124M["emb_dim"])
    y = ffn(x)
    print(y.shape)
    print(y)
    
    torch.manual_seed(123)
    x = torch.randn(2, 4, GPT_CONFIG_124M["emb_dim"])
    block = TransformerBlock.TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print(output.shape)
    print(output)
    
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print(logits.shape)
    print(logits)
    
if __name__=="__main__":
    main()
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


        self.model = torch.nn.GPT2Model.from_pretrained('gpt2')
        self.tokenizer = torch.nn.GPT2Tokenizer.from_pretrained('gpt2')
        self.linear = nn.Linear(768, 1) # 768 is the hidden size of GPT2 
         
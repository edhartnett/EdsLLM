import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from LayerNorm import LayerNorm
from TransformerBlock import TransformerBlock
import FeedForward as FeedForward
import GELU as GELU
import TransformerBlock as TransformerBlock


class EdsGPTModel(nn.Module):
    def __init__(self, cfg):
        super(EdsGPTModel, self).__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock.TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
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

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
    

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
    
    # tokenizer = tiktoken.get_encoding("gpt2")
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack(batch, dim=0)
    # print(txt1)
    # print(batch)

    # ffn = FeedForward.FeedForward(GPT_CONFIG_124M)
    # x = torch.randn(2, 10, GPT_CONFIG_124M["emb_dim"])
    # y = ffn(x)
    # print(y.shape)
    # print(y)
    
    # torch.manual_seed(123)
    # x = torch.randn(2, 4, GPT_CONFIG_124M["emb_dim"])
    # block = TransformerBlock.TransformerBlock(GPT_CONFIG_124M)
    # output = block(x)
    # print(output.shape)
    # print(output)

    torch.manual_seed(123)
    model = EdsGPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)

  `  # model = EdsGPTModel(GPT_CONFIG_124M)
    # logits = model(batch)
 `   # print(logits.shape)
    # print(logits)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")
    total_size_mb = total_params * 4 / 1024 / 1024
    print(f"Total size: {total_size_mb:.2f} MB")
    
if __name__=="__main__":
    main()
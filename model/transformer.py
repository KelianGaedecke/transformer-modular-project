import torch
import torch.nn as nn
from core.config import TransformerConfig
from core import registry
from model.block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # 1. Token Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        
        # 2. Positional Embedding (Created ONCE here)
        self.pos_emb = registry.build('pe', config.pe, config)
        
        # 3. Transformer Blocks (Stacked)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # 4. Final Output Head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        
        # Embed tokens
        x = self.token_emb(idx) # (B, T, d_model)
        
        # Add positions (if the strategy is additive)
        pe_additive = self.pos_emb(T, x.device)
        if pe_additive is not None:
            x = x + pe_additive
            
        # Pass through all blocks
        for block in self.blocks:
            # We pass self.pos_emb so attention layers can use RoPE if needed
            x, _ = block(x, pe=self.pos_emb)
            
        x = self.ln_f(x)
        logits = self.head(x) # (B, T, vocab_size)
        
        return logits
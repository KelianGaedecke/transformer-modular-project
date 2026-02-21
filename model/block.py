import torch
import torch.nn as nn
from typing import Optional, Tuple
from core.config import TransformerConfig
from core import registry

class TransformerBlock(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()

    # 1. Component Lookup (The Registry in action!)
        # We don't import MultiHeadAttention here; the registry finds it.
        self.attn = registry.build('attn', config.attn, config)
        self.ffn  = registry.build('ffn', config.ffn, config)

    # 2. Normalization
        # Standard Transformers use two LayerNorms per block
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
    
    def forward(
            self,
            x : torch.tensor,
            pe : Optional[nn.Module] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, d_model)
        pe: The PositionalEmbedding object (passed down from the top-level Model)
        """
        # --- Sublayer 1: Attention + Residual Connection ---
        # We pass 'pe' into attention because RoPE/ALiBi need it inside
        attn_out, weights = self.attn(self.norm1(x), pe=pe)
        x = x + attn_out
        
        # --- Sublayer 2: FFN + Residual Connection ---
        x = x + self.ffn(self.norm2(x))
        
        return x, weights
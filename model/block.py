import torch
import torch.nn as nn
from typing import Optional, Tuple
from core.config import PointCloudTransformerConfig
from core import registry

class TransformerBlock(nn.Module):
    def __init__(self, config: PointCloudTransformerConfig):
        super().__init__()

        # Component lookup via registry — no direct imports of implementations
        self.attn = registry.build('attn', config.attn, config)
        self.ffn  = registry.build('ffn',  config.ffn,  config)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x:      torch.Tensor,                    # (batch, N, d_model)
        pe:     Optional[nn.Module]   = None,    # PositionalEmbedding object
        coords: Optional[torch.Tensor] = None,   # (batch, N, coord_dim) for distance PE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sublayer 1: Attention + residual
        attn_out, weights = self.attn(self.norm1(x), pe=pe, coords=coords)
        x = x + attn_out

        # Sublayer 2: FFN + residual
        x = x + self.ffn(self.norm2(x))

        return x, weights

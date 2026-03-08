import torch
import torch.nn as nn
from typing import Optional, Tuple
from core.config import PointCloudTransformerConfig
from core import registry

class TransformerBlock(nn.Module):
    def __init__(self, config: PointCloudTransformerConfig):
        super().__init__()

        # Component lookup via registry — no direct imports of implementations
        self.attn = registry.build('attn', config.attn, config, **config.attn_kwargs)
        self.ffn  = registry.build('ffn',  config.ffn,  config)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Optional message passing sublayer (norm3 + mp)
        self.mp = None
        if config.mp != 'none':
            cutoff_kw = (
                {'cutoff_radius': config.cutoff_radius}
                if config.cutoff_radius is not None else {}
            )
            self.mp    = registry.build('mp', config.mp, config, **cutoff_kw, **config.mp_kwargs)
            self.norm3 = nn.LayerNorm(config.d_model)

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

        # Sublayer 3 (optional): Message passing + residual
        if self.mp is not None and coords is not None:
            x = x + self.mp(self.norm3(x), coords)

        return x, weights

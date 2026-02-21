import torch
import torch.nn as nn
from core.interfaces import PositionalEmbedding
from core.registry import register
from core.config import TransformerConfig

@register('pe', 'sinusoidal')  # <--- This hooks it into your 'Menu'
class SinusoidalEmbedding(PositionalEmbedding):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model

        self.register_buffer('inv_freq', 1.0 / (10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model)))

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Implementation of the 'Contract'"""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :] # (1, seq_len, d_model)
    

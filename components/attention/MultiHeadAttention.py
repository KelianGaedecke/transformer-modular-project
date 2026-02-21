import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from core.interfaces import AttentionStrategy, PositionalEmbedding
from core.registry import register
from core.config import TransformerConfig

@register('attn', 'mha')
class MultiHeadAttention(AttentionStrategy):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k = config.d_k  
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        pe: Optional[PositionalEmbedding] = None,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        
        # 1. Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Apply Position logic (The Contract in action!)
        if pe is not None:
            # RoPE happens here
            q, k = pe.apply_to_qk(q, k)
            
        # 3. Scaled Dot-Product Attention
        # Calculate scores
        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # ALiBi bias happens here
        if pe is not None:
            bias = pe.get_bias(T, x.device)
            if bias is not None:
                attn_scores = attn_scores + bias

        # Causal Mask (Standard for Transformers)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T) #for 'future' value ?
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(attn_scores, dim=-1)
        weights = self.dropout(weights)
        
        # 4. Combine heads
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), weights
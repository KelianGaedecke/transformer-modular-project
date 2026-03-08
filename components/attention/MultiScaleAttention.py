"""
components/attention/MultiScaleAttention.py — Multi-Scale Attention

Splits the attention heads into groups, each group attending at a different
spatial scale (cutoff distance). This allows the model to simultaneously
capture short-range (covalent bonds), medium-range (non-bonded), and
long-range (electrostatic) interactions in the same layer.

Configuration:
    attn='multiscale',
    attn_kwargs={'cutoffs': [2.0, 5.0, None]}

cutoffs is a list of per-group cutoff distances (same units as coords).
    - float → hard mask at that distance (pairs beyond → -inf)
    - None  → no cutoff (full attention, like standard MHA)

The number of groups must divide n_heads evenly.
Heads are partitioned in order: first n_heads//n_groups heads get cutoffs[0],
next group get cutoffs[1], etc.

Example with n_heads=8, cutoffs=[2.0, 5.0, None]:
    - Heads 0,1 : only atoms within 2.0 Å attend to each other  (bonded)
    - Heads 2,3 : atoms within 5.0 Å                            (non-bonded)
    - Heads 4,5,6,7 : full N² attention                         (long-range)

Note: The additive PE bias from distance_bias is still applied on top of
the scale masks if a PositionalEmbedding with get_bias() is present.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from core.interfaces import AttentionStrategy, PositionalEmbedding
from core.registry import register
from core.config import PointCloudTransformerConfig


@register('attn', 'multiscale')
class MultiScaleAttention(AttentionStrategy):
    """
    Multi-scale attention: each head-group has its own spatial cutoff.

    Args:
        config.n_heads:   total number of attention heads
        config.d_model:   embedding dimension
        cutoffs:          list of per-group cutoffs (float or None).
                          len(cutoffs) must divide n_heads.
    """

    def __init__(
        self,
        config:  PointCloudTransformerConfig,
        cutoffs: List[Optional[float]] = None,
    ):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k     = config.d_k

        if cutoffs is None:
            cutoffs = [None]   # default: one group, no cutoff (= standard MHA)

        n_groups = len(cutoffs)
        assert config.n_heads % n_groups == 0, (
            f"n_heads ({config.n_heads}) must be divisible by len(cutoffs) ({n_groups})"
        )
        self.cutoffs       = cutoffs
        self.n_groups      = n_groups
        self.heads_per_grp = config.n_heads // n_groups

        self.q_proj   = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj   = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj   = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def _scale_masks(
        self,
        coords: torch.Tensor,           # (B, N, coord_dim)
    ) -> torch.Tensor:
        """
        Build a (B, n_heads, N, N) boolean mask where True means "allowed to attend".
        Groups with cutoff=None are fully True; others restrict to within the cutoff.
        """
        B, N, _ = coords.shape
        device = coords.device

        # Pairwise distances: (B, N, N)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = diff.norm(dim=-1)

        # Start with all-True for all heads
        mask = torch.ones(B, self.n_heads, N, N, dtype=torch.bool, device=device)

        for g, cutoff in enumerate(self.cutoffs):
            if cutoff is None:
                continue
            h_start = g * self.heads_per_grp
            h_end   = h_start + self.heads_per_grp
            within  = dist < cutoff                          # (B, N, N)
            mask[:, h_start:h_end] = within.unsqueeze(1)    # broadcast over heads

        return mask   # True = can attend

    def forward(
        self,
        x:       torch.Tensor,
        pe:      Optional[PositionalEmbedding] = None,
        coords:  Optional[torch.Tensor]        = None,
        context: Optional[torch.Tensor]        = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        if pe is not None:
            q, k = pe.apply_to_qk(q, k)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Additive PE bias (e.g. distance_bias)
        if pe is not None and coords is not None:
            bias = pe.get_bias(coords)
            if bias is not None:
                attn_scores = attn_scores + bias

        # Apply scale-group masks: pairs beyond a group's cutoff → -inf
        if coords is not None and any(c is not None for c in self.cutoffs):
            scale_mask = self._scale_masks(coords)           # (B, n_heads, N, N)
            attn_scores = attn_scores.masked_fill(~scale_mask, float('-inf'))

        weights = F.softmax(attn_scores, dim=-1)
        weights = self.dropout(weights)

        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), weights

"""
model/transformer.py — Point Cloud Transformer

Encodes a set of N 3D points, each with a coordinate (x, y, z) and a
discrete type label (e.g. atom element), into per-point feature vectors.

Input:
    coords: (batch, N, coord_dim)  — floating-point 3D positions
    types:  (batch, N)             — integer point-type indices

Encoding pipeline:
    1. coord_encoder : Linear(coord_dim → d_model)   — projects xyz into embedding space
    2. type_emb      : Embedding(num_point_types, d_model) — discrete type lookup
    3. pos_emb       : optional additive spatial PE (e.g. Fourier3D on coords)
    4. n × TransformerBlock — full (non-causal) self-attention + FFN
    5. LayerNorm

Output:
    features: (batch, N, d_model)  — per-point representations
    OR, if config.output_dim is set:
    out:      (batch, N, output_dim) — projected per-point predictions
              (e.g. forces, charges, per-atom energies)
"""

import torch
import torch.nn as nn
from core.config import PointCloudTransformerConfig
from core import registry
from model.block import TransformerBlock


class PointCloudTransformer(nn.Module):
    def __init__(self, config: PointCloudTransformerConfig):
        super().__init__()
        self.config = config

        # 1. Coordinate encoder: projects (x,y,z) → d_model
        self.coord_encoder = nn.Linear(config.coord_dim, config.d_model, bias=config.bias)

        # 2. Point-type embedding: discrete label → d_model
        self.type_emb = nn.Embedding(config.num_point_types, config.d_model)

        # 3. Positional embedding.
        #    cutoff_radius is a first-class config field — only forwarded when set,
        #    since PE strategies that don't support it (none, fourier3d) would error.
        #    pe_kwargs carries any remaining PE-specific settings (n_rbf, sigma, …).
        cutoff_kw = {'cutoff_radius': config.cutoff_radius} if config.cutoff_radius is not None else {}
        self.pos_emb = registry.build(
            'pe', config.pe, config,
            **cutoff_kw,
            **config.pe_kwargs,
        )

        # 4. Transformer blocks (stacked)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # 5. Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # 6. Optional output projection head
        self.head = (
            nn.Linear(config.d_model, config.output_dim, bias=config.bias)
            if config.output_dim is not None else None
        )

    def forward(
        self,
        coords: torch.Tensor,   # (batch, N, coord_dim)  — 3D positions
        types:  torch.Tensor,   # (batch, N)             — integer type labels
    ) -> torch.Tensor:
        # Encode point types (always); only add raw coords if PE uses absolute positions.
        # Relative PE (e.g. distance_bias) skips coord_encoder to stay
        # translation- and rotation-invariant — all spatial info lives in attention bias.
        x = self.type_emb(types)  # (B, N, d_model)
        if self.pos_emb.uses_absolute_positions:
            x = x + self.coord_encoder(coords)

        # Add spatial positional encoding if the strategy provides one
        pe_additive = self.pos_emb(coords)
        if pe_additive is not None:
            x = x + pe_additive

        # Pass through all transformer blocks (full attention, non-causal)
        # coords is forwarded so distance-based PE can compute pairwise biases
        for block in self.blocks:
            x, _ = block(x, pe=self.pos_emb, coords=coords)

        x = self.ln_f(x)  # (B, N, d_model)

        # Optional task-specific projection
        if self.head is not None:
            x = self.head(x)  # (B, N, output_dim)

        return x


# Keep old name as alias for backwards compatibility
Transformer = PointCloudTransformer

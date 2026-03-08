"""
core/config.py — PointCloudTransformerConfig

All hyperparameters in one typed, serializable, saveable object.
Designed for 3D point cloud data: chemistry, molecular dynamics,
protein structures, crystallography, and other spatial 3D fields.

Each point in the cloud has:
  - A 3D coordinate (x, y, z)
  - A discrete type label (e.g. atom element, particle species)

Usage:
    # From scratch:
    config = PointCloudTransformerConfig(d_model=256, num_point_types=118)

    # From a preset:
    config = PointCloudTransformerConfig.molecule()

    # Save and reload:
    config.save('experiments/run_001.json')
    config = PointCloudTransformerConfig.load('experiments/run_001.json')

    # Derive a variant (e.g., for ablation):
    bigger = config.replace(n_layers=8, d_model=512)
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class PointCloudTransformerConfig:
    # ── Input ────────────────────────────────────────────────────────────────
    num_point_types: int   = 118      # number of discrete point type labels
                                      # (e.g. 118 chemical elements, or custom)
    coord_dim:       int   = 3        # spatial dimensions (3 for x,y,z)
    max_points:      int   = 512      # max number of points per cloud

    # ── Architecture ─────────────────────────────────────────────────────────
    d_model:  int   = 256             # embedding / hidden dimension
    n_heads:  int   = 4               # attention heads
    n_layers: int   = 4               # transformer blocks
    d_ff:     int   = None            # FFN hidden dim (default: 4 * d_model)

    # ── Strategy names (looked up in registry) ────────────────────────────────
    pe:   str = 'fourier3d'   # positional encoding:  none | fourier3d | distance_bias
    attn: str = 'mha'         # attention:            mha | multiscale
    ffn:  str = 'standard'    # feed-forward:         standard | swiglu
    mp:   str = 'none'        # message passing:      none | schnet

    # ── Attention-specific ────────────────────────────────────────────────────
    n_kv_heads:   Optional[int]  = None  # for GQA: number of key/value heads
                                         # (None = same as n_heads, i.e. standard MHA)
    attn_kwargs:  dict = field(default_factory=dict)
    # Extra kwargs for attention constructor.
    # Used by multiscale: attn_kwargs={'cutoffs': [2.0, 5.0, None]}
    # cutoffs is a list (one per head-group); None = full attention for that group.

    # ── Message passing kwargs ────────────────────────────────────────────────
    mp_kwargs: dict = field(default_factory=dict)
    # Extra kwargs for the message passing constructor.
    # Example: mp='schnet', mp_kwargs={'n_rbf': 32}

    # ── Output ───────────────────────────────────────────────────────────────
    output_dim: Optional[int] = None  # if set, adds a final Linear(d_model → output_dim)
                                      # e.g. 1 for energy prediction, None for raw features

    # ── Interaction cutoff ────────────────────────────────────────────────────
    cutoff_radius: Optional[float] = None
    # Maximum interaction distance for distance_bias PE (same units as coords).
    # None = full N² attention (fine for small molecules, N < ~200).
    # Recommended for large systems: 5.0 Å for molecules, 6–8 Å for proteins.
    #
    # When set:
    #   - Pairs beyond the cutoff are masked to -inf in attention (hard mask).
    #   - A cosine envelope tapers RBF features smoothly to 0 at the boundary.
    # Ignored by 'none' and 'fourier3d' PE strategies.

    # ── PE-specific kwargs ────────────────────────────────────────────────────
    pe_kwargs: dict = field(default_factory=dict)
    # Extra keyword arguments forwarded to the PE constructor.
    # Use for less-common PE settings (n_rbf, rbf_range, sigma, …).
    # Do NOT put cutoff_radius here — use the dedicated field above.
    # Example:
    #   pe='distance_bias', pe_kwargs={'n_rbf': 32, 'rbf_range': [0.0, 5.0]}

    # ── Training ─────────────────────────────────────────────────────────────
    dropout: float = 0.1
    bias:    bool  = False            # use bias in linear layers?

    def __post_init__(self):
        """Validate and fill in derived fields."""
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.pe in ('none', 'fourier3d', 'distance_bias'), (
            f"Unknown pe '{self.pe}'. Use: none, fourier3d, distance_bias"
        )
        assert self.attn in ('mha', 'multiscale'), (
            f"Unknown attn '{self.attn}'. Use: mha, multiscale"
        )
        assert self.ffn in ('standard', 'swiglu'), (
            f"Unknown ffn '{self.ffn}'. Use: standard, swiglu"
        )

    @property
    def d_k(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save config to JSON for experiment reproducibility."""
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> PointCloudTransformerConfig:
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def replace(self, **kwargs) -> PointCloudTransformerConfig:
        """
        Create a copy with some fields changed.
        Perfect for ablation studies.

        Example:
            base = PointCloudTransformerConfig.molecule()
            variant = base.replace(pe='none', n_layers=6)
        """
        d = asdict(self)
        d.update(kwargs)
        return PointCloudTransformerConfig(**d)

    def __repr__(self) -> str:
        params = self._estimate_params()
        cutoff = f", cutoff={self.cutoff_radius}" if self.cutoff_radius is not None else ""
        return (
            f"PointCloudTransformerConfig(\n"
            f"  input:      coord_dim={self.coord_dim}, num_point_types={self.num_point_types}, max_points={self.max_points}\n"
            f"  arch:       d_model={self.d_model}, n_heads={self.n_heads}, n_layers={self.n_layers}\n"
            f"  strategies: pe={self.pe!r}{cutoff}, attn={self.attn!r}, ffn={self.ffn!r}\n"
            f"  output:     output_dim={self.output_dim}\n"
            f"  ~params:    {params:,}\n"
            f")"
        )

    def _estimate_params(self) -> int:
        """Rough parameter count estimate."""
        # Input encoders
        coord_encoder = self.coord_dim * self.d_model
        type_emb = self.num_point_types * self.d_model
        # Per layer: attention (4 * d^2) + FFN (2 * d * d_ff)
        per_layer = 4 * self.d_model**2 + 2 * self.d_model * self.d_ff
        output = (self.d_model * self.output_dim) if self.output_dim else 0
        return coord_encoder + type_emb + self.n_layers * per_layer + output

    # ── Presets ───────────────────────────────────────────────────────────────

    @classmethod
    def molecule(cls) -> PointCloudTransformerConfig:
        """Small molecule / drug-like compound (~2M params).
        num_point_types=118 covers all chemical elements."""
        return cls(
            num_point_types=118, d_model=128, n_heads=4, n_layers=4,
            max_points=64, pe='fourier3d',
        )

    @classmethod
    def protein(cls) -> PointCloudTransformerConfig:
        """Protein structure (~10M params).
        Residue-level (num_point_types=20) or all-atom (37).
        Uses distance_bias with 8 Å cutoff — standard for protein force fields."""
        return cls(
            num_point_types=20, d_model=512, n_heads=8, n_layers=8,
            max_points=1024, pe='distance_bias', cutoff_radius=8.0,
            attn='mha', ffn='swiglu',
        )

    @classmethod
    def crystal(cls) -> PointCloudTransformerConfig:
        """Crystallographic / materials science (~5M params).
        118 elements, large unit cells.
        Uses distance_bias with 6 Å cutoff — typical for periodic structures."""
        return cls(
            num_point_types=118, d_model=256, n_heads=8, n_layers=6,
            max_points=256, pe='distance_bias', cutoff_radius=6.0, ffn='swiglu',
        )


if __name__ == "__main__":
    cfg = PointCloudTransformerConfig.molecule()
    print(cfg)

    variant = cfg.replace(pe='none', n_layers=6)
    print(f"\nVariant: {variant.pe=}, {variant.n_layers=}")

    cfg.save('/tmp/test_config.json')
    loaded = PointCloudTransformerConfig.load('/tmp/test_config.json')
    assert loaded.d_model == cfg.d_model
    print("\nConfig save/load OK")

"""
components/positional/SinusoidalEmbedding.py — 3D Positional Embedding Strategies

Three strategies for 3D point clouds:

  'none'         — No additive PE. Coordinate information is captured by
                   the coord_encoder (Linear projection) in the top-level model.

  'fourier3d'    — Random Fourier Features on (x, y, z). Per-point additive
                   embedding, absolute positions only.

  'distance_bias'— Pairwise distance bias on attention scores.
                   Computes ||r_i - r_j|| for every pair, expands into a
                   learned RBF bank, and projects to a per-head attention bias.
                   Translation and permutation invariant.
                   Does NOT add anything to the input embedding (forward → None).
"""

import torch
import torch.nn as nn
from typing import Optional
from core.interfaces import PositionalEmbedding
from core.registry import register
from core.config import PointCloudTransformerConfig


# ─── No-op baseline ───────────────────────────────────────────────────────────

@register('pe', 'none')
class NonePositionalEmbedding(PositionalEmbedding):
    """
    No positional embedding added to the input.
    The coord_encoder in the Transformer already projects (x,y,z) → d_model,
    so this is a valid and clean baseline.
    """
    def __init__(self, _config: PointCloudTransformerConfig):
        super().__init__()

    def forward(self, _coords: torch.Tensor) -> Optional[torch.Tensor]:
        return None


# ─── Fourier 3D ───────────────────────────────────────────────────────────────

@register('pe', 'fourier3d')
class Fourier3DPositionalEmbedding(PositionalEmbedding):
    """
    Random Fourier Feature positional encoding for 3D coordinates.

    For each point at position (x, y, z) this computes:
        PE(x,y,z) = Linear([sin(B·p), cos(B·p)])

    where B is a fixed random matrix of shape (coord_dim, num_frequencies)
    sampled from N(0, sigma^2). The sin/cos outputs are concatenated to give
    2*num_frequencies features, then projected to d_model.

    This gives the model a continuous, translation-equivariant sense of
    spatial position that generalises well across scales.

    Args:
        config.d_model:   output embedding dimension
        config.coord_dim: input spatial dimensions (default 3)
        num_frequencies:  number of random frequency samples (default d_model//2)
        sigma:            bandwidth of the Gaussian kernel (default 1.0)
                          larger sigma → higher-frequency features
    """

    def __init__(
        self,
        config: PointCloudTransformerConfig,
        num_frequencies: int = None,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.d_model = config.d_model
        num_freq = num_frequencies or (config.d_model // 2)

        # Fixed random projection matrix — not a learned parameter
        B = torch.randn(config.coord_dim, num_freq) * sigma
        self.register_buffer('B', B)

        # Learned projection from Fourier features → d_model
        self.proj = nn.Linear(2 * num_freq, config.d_model, bias=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (batch, N, coord_dim)
        Returns:
            pe: (batch, N, d_model)
        """
        # (batch, N, num_freq)
        proj = coords @ self.B
        # Concatenate sin and cos → (batch, N, 2*num_freq)
        fourier = torch.cat([proj.sin(), proj.cos()], dim=-1)
        # Project to d_model
        return self.proj(fourier)


# ─── Distance Bias ────────────────────────────────────────────────────────────

@register('pe', 'distance_bias')
class DistanceBiasPositionalEmbedding(PositionalEmbedding):
    """
    Relative positional encoding via pairwise distance bias on attention scores.

    For every pair of points (i, j), computes the Euclidean distance
    d_ij = ||r_i - r_j||, then encodes it with K Gaussian RBFs and projects
    to a per-head attention bias.

    Cutoff radius (optional, recommended for large systems):
        When cutoff_radius is set:
        - Pairs beyond the cutoff get attention bias = -inf (hard mask).
          Those atoms cannot attend to each other at all.
        - Within the cutoff, a cosine envelope smoothly tapers the RBF
          signal to 0 at the boundary, avoiding gradient discontinuities:

              envelope(d) = 0.5 * (cos(π * d / r_cut) + 1)

        This is the standard approach in SchNet, MACE, NequIP.
        Memory is still O(N²) but information flow is local.
        For truly sparse computation see sparse-attention extensions.

    Configure via pe_kwargs in the config:
        pe='distance_bias', pe_kwargs={'cutoff_radius': 5.0, 'n_rbf': 32}

    Properties:
        - Translation & rotation invariant (distances are geometry-invariant)
        - Permutation equivariant
        - Learned: RBF width and the projection to attention heads are trained

    Args:
        config.n_heads:   number of attention heads
        n_rbf:            number of Gaussian RBF basis functions (default 16)
        rbf_range:        (d_min, d_max) for RBF center placement (default 0–5 Å)
        cutoff_radius:    hard interaction cutoff in the same units as coords.
                          None = full N² attention (default).
    """

    uses_absolute_positions: bool = False

    def __init__(
        self,
        config:        PointCloudTransformerConfig,
        n_rbf:         int            = 16,
        rbf_range:     tuple          = (0.0, 5.0),
        cutoff_radius: Optional[float] = None,
    ):
        super().__init__()
        self.n_heads       = config.n_heads
        self.cutoff_radius = cutoff_radius

        d_min, d_max = rbf_range

        # Fixed RBF centers evenly spaced across rbf_range
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer('centers', centers)

        # Learnable log-width (kept positive via exp)
        init_width = (d_max - d_min) / n_rbf
        self.log_width = nn.Parameter(torch.tensor(init_width).log())

        # Project RBF features → one scalar bias per attention head
        self.proj = nn.Linear(n_rbf, config.n_heads, bias=False)

    def forward(self, _coords: torch.Tensor) -> None:
        # No additive input embedding — spatial info goes into attention only
        return None

    def get_bias(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (batch, N, coord_dim)
        Returns:
            bias: (batch, n_heads, N, N)
                  Pairs beyond cutoff_radius have bias = -inf.
        """
        # Pairwise distances: (B, N, N)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = diff.norm(dim=-1)

        # RBF expansion: (B, N, N, n_rbf)
        width = self.log_width.exp()
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.centers) / width) ** 2)

        if self.cutoff_radius is not None:
            within = dist < self.cutoff_radius          # (B, N, N) bool

            # Cosine envelope: 1 at d=0, smoothly → 0 at d=r_cut
            # Avoids discontinuous gradients at the boundary
            envelope = 0.5 * (torch.cos(torch.pi * dist / self.cutoff_radius) + 1)
            envelope = torch.where(within, envelope, torch.zeros_like(envelope))

            # Apply envelope to RBF features
            rbf = rbf * envelope.unsqueeze(-1)

        # Project to per-head bias: (B, N, N, n_heads) → (B, n_heads, N, N)
        bias = self.proj(rbf).permute(0, 3, 1, 2)

        if self.cutoff_radius is not None:
            # Hard mask: atoms beyond cutoff cannot attend to each other
            within_h = within.unsqueeze(1)              # (B, 1, N, N)
            bias = bias.masked_fill(~within_h, float('-inf'))

        return bias

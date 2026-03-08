"""
components/message_passing/schnet.py — SchNet-style continuous-filter convolution

SchNet reference:
    Schütt et al., "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions", NeurIPS 2017.

Operation for each node i:
    agg_i = sum_{j} x_j ⊙ filter_net(rbf(d_ij))

where:
    - d_ij = ||r_i − r_j||  (pairwise Euclidean distance)
    - rbf(d)                (Gaussian RBF expansion to n_rbf features)
    - filter_net            (small MLP: n_rbf → d_model → d_model, SiLU activations)
    - ⊙                     (element-wise product — the "filter" gates the neighbour)

Cutoff (optional):
    When cutoff_radius is set, interactions beyond the cutoff are masked to 0
    with a smooth cosine envelope tapering the RBF features to 0 at r_cut:

        envelope(d) = 0.5 * (cos(π * d / r_cut) + 1)

    This matches the envelope used in the distance_bias PE and in MACE/NequIP.

Configuration:
    mp='schnet', mp_kwargs={'n_rbf': 32, 'cutoff_radius': 5.0}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from core.interfaces import MessagePassingStrategy
from core.registry import register
from core.config import PointCloudTransformerConfig


@register('mp', 'schnet')
class SchNetMessagePassing(MessagePassingStrategy):
    """
    SchNet continuous-filter convolution for 3D point clouds.

    Args:
        config.d_model:   node feature dimension (in and out)
        n_rbf:            number of Gaussian RBF basis functions (default 32)
        rbf_range:        (d_min, d_max) for RBF center placement (default 0–5 Å)
        cutoff_radius:    hard interaction cutoff in the same units as coords.
                          None = all pairs interact (default).
    """

    def __init__(
        self,
        config:        PointCloudTransformerConfig,
        n_rbf:         int            = 32,
        rbf_range:     tuple          = (0.0, 5.0),
        cutoff_radius: Optional[float] = None,
    ):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        d = config.d_model

        d_min, d_max = rbf_range

        # Fixed RBF centers evenly spaced across rbf_range
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer('centers', centers)

        # Learnable log-width (kept positive via exp)
        init_width = (d_max - d_min) / n_rbf
        self.log_width = nn.Parameter(torch.tensor(init_width).log())

        # Filter network: maps rbf features to a per-channel gate (d_model)
        # Two-layer MLP with SiLU, following SchNet convention
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # Output projection after aggregation
        self.out_proj = nn.Linear(d, d, bias=config.bias)

    def forward(
        self,
        x:      torch.Tensor,   # (batch, N, d_model)
        coords: torch.Tensor,   # (batch, N, coord_dim)
    ) -> torch.Tensor:
        """
        Returns aggregated node features of shape (batch, N, d_model).
        """
        B, N, _ = x.shape

        # Pairwise distances: (B, N, N)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)   # (B, N, N, 3)
        dist = diff.norm(dim=-1)                            # (B, N, N)

        # RBF expansion: (B, N, N, n_rbf)
        width = self.log_width.exp()
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.centers) / width) ** 2)

        # Cosine envelope + hard mask when cutoff is set
        if self.cutoff_radius is not None:
            within = dist < self.cutoff_radius              # (B, N, N)
            envelope = 0.5 * (torch.cos(torch.pi * dist / self.cutoff_radius) + 1)
            envelope = torch.where(within, envelope, torch.zeros_like(envelope))
            rbf = rbf * envelope.unsqueeze(-1)              # taper to 0 at boundary

        # Filter: (B, N, N, d_model) — one filter vector per pair
        W = self.filter_net(rbf)                            # (B, N, N, d_model)

        # Aggregate: sum_j x_j * W_ij  →  (B, N, d_model)
        # x_j: (B, 1, N, d_model) broadcast over query dim i
        agg = (x.unsqueeze(1) * W).sum(dim=2)              # (B, N, d_model)

        return self.out_proj(agg)

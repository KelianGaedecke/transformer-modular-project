"""
core/interfaces.py — Abstract Base Classes (Strategy Interfaces)

These ABCs define the *contract* that every strategy must satisfy.
Designed for 3D point cloud data: chemistry, molecular dynamics,
protein structures, and other spatial fields.

Adding a new strategy:
  1. Subclass the right ABC
  2. Implement all @abstractmethod methods
  3. Add @register decorator
  Done. Nothing else to change.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn


# ─── Positional Embedding Interface ──────────────────────────────────────────

class PositionalEmbedding(ABC, nn.Module):
    """
    Contract for all 3D positional embedding strategies.

    Two modes:
      A) Absolute (Fourier3D): forward() returns (batch, N, d_model) added to input.
         uses_absolute_positions = True  → coord_encoder also applied in Transformer.

      B) Relative (DistanceBias): forward() returns None, get_bias() returns
         (batch, n_heads, N, N) added to attention scores.
         uses_absolute_positions = False → coord_encoder is skipped so the model
         is translation- and rotation-invariant.
    """

    uses_absolute_positions: bool = True

    @abstractmethod
    def forward(self, coords: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Args:
            coords: (batch, N, coord_dim) — 3D coordinates of N points.

        Returns:
            Additive positional embedding of shape (batch, N, d_model),
            or None if this PE method doesn't add to the input.
        """

    def get_bias(self, _coords: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Returns additive attention score bias of shape (batch, n_heads, N, N),
        or None. Default: no bias. Override for distance-based / relative PE.

        Args:
            coords: (batch, N, coord_dim) — raw 3D coordinates of the point cloud.
        """
        return None

    def apply_to_qk(
        self,
        q: torch.Tensor,   # (batch, heads, N, d_k)
        k: torch.Tensor,   # (batch, heads, N, d_k)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally modify Q and K before attention.
        Default: identity (no modification).
        """
        return q, k


# ─── Attention Interface ──────────────────────────────────────────────────────

class AttentionStrategy(ABC, nn.Module):
    """
    Contract for all attention mechanisms.

    Input x is a sequence of point embeddings (batch, N, d_model).
    Attention is full (non-causal): every point attends to every other point.
    """

    @abstractmethod
    def forward(
        self,
        x:    torch.Tensor,                        # (batch, N, d_model)
        pe:   Optional[PositionalEmbedding] = None,
        context: Optional[torch.Tensor]    = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output:  (batch, N, d_model)
            weights: (batch, n_heads, N, N)
        """
        ...


# ─── FFN Interface ────────────────────────────────────────────────────────────

class FFNStrategy(ABC, nn.Module):
    """
    Contract for feed-forward network variants.
    Input and output are both (batch, N, d_model).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, N, d_model) → (batch, N, d_model)"""
        ...

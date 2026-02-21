"""
core/interfaces.py — Abstract Base Classes (Strategy Interfaces)

These ABCs define the *contract* that every strategy must satisfy.
The rest of the codebase only talks to these interfaces, never to
concrete implementations. This is the heart of the Strategy pattern.

Why ABCs over duck typing?
  - Explicit: you immediately know what methods a strategy needs
  - Helpful errors: failing at class definition, not at runtime
  - IDE support: autocomplete works, type checking works
  - Documentation: the ABC IS the specification

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
    Contract for all positional embedding strategies.

    Two operation modes exist in practice:
      A) Absolute (Sinusoidal, Learned): add a vector to input embeddings.
         → implement forward() to return an additive embedding.
         → get_bias() returns None.

      B) Relative / Score-based (ALiBi, RoPE): modify attention scores
         or Q/K vectors inside attention.
         → forward() may return None (nothing to add to input).
         → get_bias() returns an additive bias for attention scores (ALiBi).
         → RoPE is handled differently (modifies Q/K directly via apply_to_qk).

    The TransformerBlock checks which mode applies by calling both.
    """

    @abstractmethod
    def forward(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Returns additive positional embedding of shape (1, seq_len, d_model),
        or None if this PE method doesn't add to the input (e.g. ALiBi, RoPE).
        """
        

    def get_bias(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Returns additive attention score bias of shape (1, n_heads, seq_len, seq_len),
        or None. Default: no bias. Override for ALiBi.
        """
        return None

    def apply_to_qk(
        self,
        q: torch.Tensor,   # (batch, heads, seq, d_k)
        k: torch.Tensor,   # (batch, heads, seq, d_k)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally modify Q and K before attention (used by RoPE).
        Default: identity (no modification). Override for RoPE.
        """
        return q, k


# ─── Attention Interface ──────────────────────────────────────────────────────

class AttentionStrategy(ABC, nn.Module):
    """
    Contract for all attention mechanisms.

    Every attention implementation receives (x, pe) and returns (output, weights).
    The positional embedding is passed in so attention can call pe.apply_to_qk()
    for RoPE, or pe.get_bias() for ALiBi — keeping PE logic in the PE class.
    """

    @abstractmethod
    def forward(
        self,
        x:    torch.Tensor,                        # (batch, seq, d_model)
        pe:   Optional[PositionalEmbedding] = None,  # for RoPE / ALiBi
        context: Optional[torch.Tensor]    = None,   # for cross-attention
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output:  (batch, seq, d_model)
            weights: (batch, n_heads, seq, seq) — or approximation thereof
        """
        ...
        


# ─── FFN Interface ────────────────────────────────────────────────────────────

class FFNStrategy(ABC, nn.Module):
    """
    Contract for feed-forward network variants.
    Input and output are both (batch, seq, d_model).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model) → (batch, seq, d_model)"""
        ...
        
"""
core/config.py — TransformerConfig

All hyperparameters in one typed, serializable, saveable object.
This is Pattern 3: Dataclass Config.

Design goals:
  - One object to pass around (not 15 kwargs)
  - Validates itself on creation (catches mistakes early)
  - Serializes to/from JSON (experiment reproducibility)
  - HuggingFace-compatible (can subclass PretrainedConfig)
  - Supports named presets for common model sizes

Usage:
    # From scratch:
    config = TransformerConfig(d_model=256, pe='rope', attn='gqa', n_kv_heads=2)

    # From a preset:
    config = TransformerConfig.gpt_nano()

    # Save and reload:
    config.save('experiments/run_001.json')
    config = TransformerConfig.load('experiments/run_001.json')

    # Derive a variant (e.g., for ablation):
    bigger = config.replace(n_layers=8, d_model=512)
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TransformerConfig:
    # ── Architecture ────────────────────────────────────────────────────────
    vocab_size:  int   = 256       # character-level default
    d_model:     int   = 256       # embedding dimension
    n_heads:     int   = 4         # attention heads
    n_layers:    int   = 4         # transformer blocks
    d_ff:        int   = None      # FFN hidden dim (default: 4 * d_model)
    max_seq_len: int   = 512       # max context length

    # ── Strategy names (looked up in registry) ──────────────────────────────
    pe:   str = 'sinusoidal'  # positional embedding: sinusoidal|learned|rope|alibi
    attn: str = 'mha'         # attention: mha|gqa|flash
    ffn:  str = 'standard'    # feed-forward: standard|swiglu

    # ── Attention-specific ──────────────────────────────────────────────────
    n_kv_heads:  Optional[int] = None   # for GQA: number of key/value heads
                                        # (None = same as n_heads, i.e. standard MHA)
    # ── Training ────────────────────────────────────────────────────────────
    dropout:     float = 0.1
    bias:        bool  = False     # use bias in linear layers? (False = GPT-2 style)

    def __post_init__(self):
        """Validate and fill in derived fields."""
        # Fill defaults
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # Validate
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.pe in ('sinusoidal', 'learned', 'rope', 'alibi'), (
            f"Unknown pe '{self.pe}'. Use: sinusoidal, learned, rope, alibi"
        )
        assert self.attn in ('mha', 'gqa', 'flash'), (
            f"Unknown attn '{self.attn}'. Use: mha, gqa, flash"
        )
        assert self.ffn in ('standard', 'swiglu'), (
            f"Unknown ffn '{self.ffn}'. Use: standard, swiglu"
        )

    @property
    def d_k(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    # ── Serialization ────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save config to JSON for experiment reproducibility."""
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> TransformerConfig:
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def replace(self, **kwargs) -> TransformerConfig:
        """
        Create a copy with some fields changed.
        Perfect for ablation studies.

        Example:
            base = TransformerConfig.gpt_nano()
            variant = base.replace(pe='rope', n_layers=6)
        """
        d = asdict(self)
        d.update(kwargs)
        return TransformerConfig(**d)

    def __repr__(self) -> str:
        params = self._estimate_params()
        return (
            f"TransformerConfig(\n"
            f"  arch:    d_model={self.d_model}, n_heads={self.n_heads}, n_layers={self.n_layers}\n"
            f"  strategies: pe={self.pe!r}, attn={self.attn!r}, ffn={self.ffn!r}\n"
            f"  context: max_seq_len={self.max_seq_len}, vocab_size={self.vocab_size}\n"
            f"  ~params: {params:,}\n"
            f")"
        )

    def _estimate_params(self) -> int:
        """Rough parameter count estimate (useful for quick sanity checks)."""
        # Embeddings
        embed = self.vocab_size * self.d_model
        # Per layer: attention (4 * d^2) + FFN (2 * d * d_ff) + 2 LayerNorms
        per_layer = 4 * self.d_model**2 + 2 * self.d_model * self.d_ff
        return embed + self.n_layers * per_layer

    # ── Presets ──────────────────────────────────────────────────────────────

    @classmethod
    def gpt_nano(cls) -> TransformerConfig:
        """Tiny model for fast experimentation (~1M params)."""
        return cls(vocab_size=65, d_model=128, n_heads=4, n_layers=3,
                   max_seq_len=256, pe='sinusoidal')

    @classmethod
    def gpt_micro(cls) -> TransformerConfig:
        """Small but expressive model (~10M params)."""
        return cls(vocab_size=50257, d_model=512, n_heads=8, n_layers=6,
                   max_seq_len=1024, pe='rope', attn='mha', ffn='swiglu')

    @classmethod
    def llama_style(cls, vocab_size: int = 32000) -> TransformerConfig:
        """LLaMA-inspired: RoPE + SwiGLU + GQA."""
        return cls(vocab_size=vocab_size, d_model=512, n_heads=8, n_layers=8,
                   max_seq_len=2048, pe='rope', attn='gqa', ffn='swiglu',
                   n_kv_heads=2, bias=False)


if __name__ == "__main__":
    # Demo
    cfg = TransformerConfig.gpt_nano()
    print(cfg)

    # Ablation: try RoPE on the same base
    rope_cfg = cfg.replace(pe='rope')
    print(f"\nVariant: {rope_cfg.pe=}, {rope_cfg.d_model=}")

    # Save/load round-trip
    cfg.save('/tmp/test_config.json')
    loaded = TransformerConfig.load('/tmp/test_config.json')
    assert loaded.d_model == cfg.d_model
    print("\n Config save/load OK")
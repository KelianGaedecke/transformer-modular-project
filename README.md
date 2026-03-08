# Point Cloud Transformer

A modular Transformer architecture for 3D point cloud data — designed for chemistry, molecular dynamics, protein structures, crystallography, and any other field where inputs are sets of spatially located, typed points.

---

## Core idea

Each input is a **set of N points**, where every point has:
- A **3D coordinate** `(x, y, z)`
- A **discrete type label** (e.g. atom element, amino acid, particle species)

The model produces **per-point feature vectors** that capture both local geometry and point identity. There is no sequence ordering — the model is **permutation equivariant** by design.

---

## Architecture

```
coords (B, N, 3)  ──→  coord_encoder  ─┐
                                        +──→  x (B, N, d_model)
types  (B, N)     ──→  type_emb       ─┘
                              │
                        + pos_emb(coords)      [optional additive PE]
                              │
                    ┌─────────▼──────────┐
                    │  TransformerBlock  │ × n_layers
                    │  ├─ LayerNorm      │
                    │  ├─ Attention ─────┼── + distance bias (optional)
                    │  ├─ residual       │
                    │  ├─ LayerNorm      │
                    │  ├─ FFN            │
                    │  └─ residual       │
                    └─────────┬──────────┘
                              │
                         LayerNorm
                              │
                    [optional Linear → output_dim]
                              │
                     output (B, N, d_model)
                    or (B, N, output_dim)
```

Attention is **full and non-causal** — every point attends to every other point. No causal mask. No positional ordering.

---

## Quick start

```python
import torch
import components                          # registers all strategies
from core.config import PointCloudTransformerConfig
from model.transformer import PointCloudTransformer

config = PointCloudTransformerConfig.molecule()
model  = PointCloudTransformer(config)

coords = torch.randn(4, 32, 3)            # 4 molecules, 32 atoms each
types  = torch.randint(0, 118, (4, 32))   # atom element indices

out = model(coords, types)                # (4, 32, 128) per-atom features
```

---

## Configuration

All hyperparameters live in `PointCloudTransformerConfig`:

```python
config = PointCloudTransformerConfig(
    # Input
    num_point_types = 118,        # distinct type labels (118 = all elements)
    coord_dim       = 3,          # spatial dimensions
    max_points      = 512,        # informational; no hard runtime limit

    # Architecture
    d_model  = 256,               # embedding width
    n_heads  = 8,                 # attention heads (must divide d_model)
    n_layers = 6,                 # number of transformer blocks
    d_ff     = None,              # FFN hidden dim (default: 4 × d_model)

    # Strategies — swappable via the registry
    pe   = 'distance_bias',       # positional encoding (see below)
    attn = 'mha',                 # attention mechanism
    ffn  = 'swiglu',              # feed-forward variant

    # Cutoff radius (distance_bias PE only)
    cutoff_radius = 5.0,          # Angstrom; None = full N² attention

    # Output
    output_dim = 1,               # None → raw (B,N,d_model); int → (B,N,output_dim)

    # Training
    dropout = 0.1,
    bias    = False,

    # Advanced PE settings (forwarded to PE constructor)
    pe_kwargs = {'n_rbf': 32, 'rbf_range': [0.0, 5.0]},
)
```

Configs are **JSON-serializable**:

```python
config.save('experiments/run_001.json')
config = PointCloudTransformerConfig.load('experiments/run_001.json')

# Ablation variant
variant = config.replace(n_layers=8, pe='fourier3d')
```

---

## Positional encoding strategies

The `pe=` option controls how spatial information enters the model.

### `'none'`
No positional embedding. Coordinates are still projected into the embedding via `coord_encoder` (a linear layer), but no extra spatial signal is added. Good baseline.

- Uses absolute positions — **not** rotation/translation invariant.

### `'fourier3d'`
Random Fourier Features on the raw `(x, y, z)` coordinates:

```
PE(x,y,z) = Linear([sin(B·p), cos(B·p)])
```

`B` is a fixed random matrix; the result is added to the input embedding. Provides a rich, continuous, multi-scale sense of spatial location.

- Uses absolute positions — **not** rotation/translation invariant.
- Good for fixed-frame data (crystal structures, MD snapshots in a fixed box).

### `'distance_bias'`
Pairwise distance bias on attention scores. For each pair `(i, j)`:

```
d_ij = ‖r_i − r_j‖

φ_k(d) = exp(−((d − μ_k) / σ)²)     [Gaussian RBF expansion]

attn_score[h,i,j] += Linear(φ)(h)    [projected to per-head bias]
```

- **Translation invariant** — shifting all atoms leaves distances unchanged.
- **Rotation invariant** — rotating all atoms leaves distances unchanged.
- `coord_encoder` is automatically **skipped** when using this PE (it would break invariance).

Comparison:

| Property             | `none` | `fourier3d` | `distance_bias` |
|----------------------|:------:|:-----------:|:---------------:|
| Permutation equivariant | ✓   | ✓           | ✓               |
| Translation invariant   | ✓*  | ✗           | ✓               |
| Rotation invariant      | ✓*  | ✗           | ✓               |
| Captures geometry       | ✗   | ✓           | ✓               |

*`none` is invariant only because it ignores coordinates entirely.

#### Cutoff radius

For large systems, set `cutoff_radius` to limit interactions to a sphere:

```python
config = PointCloudTransformerConfig(
    pe='distance_bias',
    cutoff_radius=5.0,              # only atoms within 5 Å interact
    pe_kwargs={'n_rbf': 32},        # finer RBF grid
)
```

Within the cutoff, a **cosine envelope** tapers RBF features smoothly to zero at the boundary — avoiding gradient discontinuities at `r_cut`:

```
envelope(d) = 0.5 × (cos(π × d / r_cut) + 1)
```

Beyond the cutoff, the attention bias is set to `-∞` (hard mask) — those pairs contribute zero to the attention-weighted sum.

Note: memory is still O(N²). For truly sparse computation (N > ~2000), a neighbor-list approach would be needed.

---

## FFN strategies

| `ffn=`       | Operation |
|--------------|-----------|
| `'standard'` | `Linear → GELU → Linear → Dropout` |
| `'swiglu'`   | `(W₁x) × silu(W₂x) → Linear → Dropout` — used in LLaMA, PaLM |

---

## Attention strategies

| `attn=`        | Description |
|----------------|-------------|
| `'mha'`        | Standard multi-head attention — every point attends to every other |
| `'multiscale'` | Head-group attention with per-group spatial cutoffs (see below) |

### `'multiscale'`

Splits `n_heads` into groups, each group attending at a different spatial scale.
Configure via `attn_kwargs`:

```python
config = PointCloudTransformerConfig(
    d_model=128, n_heads=6,   # 6 heads, 3 groups of 2
    attn='multiscale',
    attn_kwargs={'cutoffs': [2.0, 5.0, None]},
    # cutoffs[0]=2.0 → heads 0,1 see only atoms within 2 Å  (covalent bonds)
    # cutoffs[1]=5.0 → heads 2,3 see atoms within 5 Å       (non-bonded)
    # cutoffs[2]=None → heads 4,5 have full N² attention     (long-range)
)
```

`len(cutoffs)` must divide `n_heads` evenly.
The additive PE bias from `distance_bias` is still applied on top of the scale masks.

---

## Message passing strategies

An optional SchNet-style message passing sublayer can be added after the FFN
in every TransformerBlock. It aggregates local neighbour information using
learned distance filters — complementing global attention with explicitly
local structure.

| `mp=`      | Description |
|------------|-------------|
| `'none'`   | No message passing (default) |
| `'schnet'` | SchNet continuous-filter convolution |

### `'schnet'`

For each node `i`:
```
agg_i = Σ_{j} x_j ⊙ filter_net(rbf(d_ij))
```

- `d_ij = ‖r_i − r_j‖` — pairwise distance
- `rbf(d)` — Gaussian RBF expansion (fixed centers, learnable width)
- `filter_net` — two-layer MLP with SiLU: maps RBF features to a per-channel gate
- `⊙` — element-wise product gates each neighbour's features

Configure via `mp_kwargs`:

```python
config = PointCloudTransformerConfig(
    pe='distance_bias',
    mp='schnet',
    cutoff_radius=5.0,          # cutoff is shared with the PE
    mp_kwargs={'n_rbf': 32},    # finer RBF grid for the filter network
)
```

SchNet is **translation and rotation invariant** (uses only pairwise distances).
Memory cost is O(N²) per layer.

---

## Built-in presets

```python
# Small molecule (~820K params) — Fourier3D PE, all 118 elements
config = PointCloudTransformerConfig.molecule()

# Protein (~34M params) — distance_bias + 8 Å cutoff, SwiGLU, 20 residue types
config = PointCloudTransformerConfig.protein()

# Crystal / materials (~10M params) — distance_bias + 6 Å cutoff, SwiGLU
config = PointCloudTransformerConfig.crystal()
```

---

## Output modes

| `output_dim` | Output shape | Typical use |
|---|---|---|
| `None` (default) | `(B, N, d_model)` | Embeddings for downstream tasks |
| `1` | `(B, N, 1)` | Per-atom energy, charge, … |
| `3` | `(B, N, 3)` | Per-atom force vectors |
| `K` | `(B, N, K)` | Any per-point regression |

---

## Adding a new strategy

The registry pattern makes it trivial. Subclass the right ABC, decorate:

```python
# components/positional/my_pe.py
from core.interfaces import PositionalEmbedding
from core.registry import register

@register('pe', 'my_pe')
class MyPositionalEmbedding(PositionalEmbedding):
    uses_absolute_positions = False   # set True if you use raw coords

    def __init__(self, config, **kwargs):
        super().__init__()
        ...

    def forward(self, coords):
        return None   # or (B, N, d_model) if additive

    def get_bias(self, coords):
        return ...    # (B, n_heads, N, N) or None
```

Then add it to `components/positional/__init__.py` and use `pe='my_pe'` in config. Nothing else needs to change.

The same pattern applies to `AttentionStrategy` (`@register('attn', ...)`) and `FFNStrategy` (`@register('ffn', ...)`).

---

## File layout

```
core/
  config.py        — PointCloudTransformerConfig (all hyperparameters)
  interfaces.py    — ABCs: PositionalEmbedding, AttentionStrategy, FFNStrategy,
                     MessagePassingStrategy
  registry.py      — @register decorator + build() factory

model/
  transformer.py   — PointCloudTransformer (top-level model)
  block.py         — TransformerBlock (attention + FFN + optional MP + residuals)

components/
  positional/
    SinusoidalEmbedding.py  — none | fourier3d | distance_bias
  attention/
    MultiHeadAttention.py   — mha
    MultiScaleAttention.py  — multiscale
  ffn/
    ffn.py                  — standard | swiglu
  message_passing/
    schnet.py               — schnet

run_test.py        — 12 integration tests
```

---

## Running tests

```bash
python run_test.py
```

Tests cover: forward pass shapes, no-PE baseline, output head, permutation equivariance, config save/load, protein preset (SwiGLU), variable point counts, rotation/translation invariance of `distance_bias`, cutoff radius behavior, `pe_kwargs` round-trip serialization, SchNet message passing (shape + translation invariance), and multi-scale attention (shape, isolated-atom NaN safety, permutation equivariance).

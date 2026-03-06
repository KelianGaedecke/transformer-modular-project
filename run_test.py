"""
run_test.py — Integration tests for the Point Cloud Transformer

Tests cover:
  1. Basic forward pass (molecule preset, Fourier3D PE)
  2. No positional encoding baseline
  3. Output projection head (e.g. per-atom energy prediction)
  4. Config save/load round-trip
  5. Permutation equivariance — reordering points should reorder outputs
     identically (no causal mask, no positional ordering bias)
  6. Protein and crystal presets
  7. Variable batch sizes and point counts
"""

import torch
import components  # IMPORTANT: triggers @register decorators
from core.config import PointCloudTransformerConfig
from model.transformer import PointCloudTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_batch(batch_size, n_points, num_point_types, coord_dim=3):
    """Create a random batch of 3D point cloud data."""
    coords = torch.randn(batch_size, n_points, coord_dim, device=DEVICE)
    types  = torch.randint(0, num_point_types, (batch_size, n_points), device=DEVICE)
    return coords, types

def param_count(model):
    return sum(p.numel() for p in model.parameters())

def separator(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_molecule_preset():
    separator("Test 1: molecule() preset — Fourier3D PE")
    config = PointCloudTransformerConfig.molecule()
    model  = PointCloudTransformer(config).to(DEVICE)

    B, N = 4, 32
    coords, types = make_batch(B, N, config.num_point_types)

    out = model(coords, types)

    assert out.shape == (B, N, config.d_model), f"Expected ({B}, {N}, {config.d_model}), got {out.shape}"
    print(f"  Config:  {config.d_model=}, {config.n_layers=}, {config.pe=}")
    print(f"  Input:   coords={tuple(coords.shape)}, types={tuple(types.shape)}")
    print(f"  Output:  {tuple(out.shape)}")
    print(f"  Params:  {param_count(model):,}")
    print("  PASSED")


def test_no_pe():
    separator("Test 2: no positional encoding (pe='none')")
    config = PointCloudTransformerConfig(
        num_point_types=10, d_model=64, n_heads=4, n_layers=2, pe='none'
    )
    model = PointCloudTransformer(config).to(DEVICE)

    B, N = 2, 16
    coords, types = make_batch(B, N, config.num_point_types)
    out = model(coords, types)

    assert out.shape == (B, N, config.d_model)
    print(f"  Output:  {tuple(out.shape)}")
    print("  PASSED")


def test_output_head():
    separator("Test 3: output projection head (per-atom energy)")
    config = PointCloudTransformerConfig(
        num_point_types=118, d_model=128, n_heads=4, n_layers=3,
        pe='fourier3d', output_dim=1   # scalar energy per atom
    )
    model = PointCloudTransformer(config).to(DEVICE)

    B, N = 3, 24
    coords, types = make_batch(B, N, config.num_point_types)
    out = model(coords, types)

    assert out.shape == (B, N, 1), f"Expected ({B}, {N}, 1), got {out.shape}"
    print(f"  Output:  {tuple(out.shape)}  ← per-atom scalar (e.g. energy)")
    print(f"  Params:  {param_count(model):,}")
    print("  PASSED")


def test_permutation_equivariance():
    separator("Test 4: permutation equivariance")
    config = PointCloudTransformerConfig(
        num_point_types=20, d_model=64, n_heads=4, n_layers=2,
        pe='none', dropout=0.0
    )
    model = PointCloudTransformer(config).to(DEVICE).eval()

    B, N = 1, 8
    coords, types = make_batch(B, N, config.num_point_types)

    # Random permutation of the N points
    perm = torch.randperm(N, device=DEVICE)

    with torch.no_grad():
        out_orig = model(coords, types)
        out_perm = model(coords[:, perm, :], types[:, perm])

    # Permuting input should permute output identically
    max_diff = (out_orig[:, perm, :] - out_perm).abs().max().item()
    assert max_diff < 1e-5, f"Equivariance broken: max diff = {max_diff:.2e}"
    print(f"  Max absolute difference after permutation: {max_diff:.2e}")
    print("  PASSED  (outputs are permutation-equivariant)")


def test_config_save_load(tmp_path='/tmp/pointcloud_config.json'):
    separator("Test 5: config save / load round-trip")
    config = PointCloudTransformerConfig.crystal()
    config.save(tmp_path)
    loaded = PointCloudTransformerConfig.load(tmp_path)

    assert loaded.d_model        == config.d_model
    assert loaded.num_point_types == config.num_point_types
    assert loaded.pe             == config.pe
    assert loaded.n_layers       == config.n_layers
    print(f"  Saved to {tmp_path}")
    print(f"  Loaded: {loaded.d_model=}, {loaded.pe=}, {loaded.num_point_types=}")
    print("  PASSED")


def test_protein_preset():
    separator("Test 6: protein() preset")
    config = PointCloudTransformerConfig.protein()
    model  = PointCloudTransformer(config).to(DEVICE)

    B, N = 2, 64          # small batch; real proteins can be 1000+ residues
    coords, types = make_batch(B, N, config.num_point_types)
    out = model(coords, types)

    assert out.shape == (B, N, config.d_model)
    print(f"  Config:  {config.d_model=}, {config.n_layers=}, {config.n_heads=}")
    print(f"  Output:  {tuple(out.shape)}")
    print(f"  Params:  {param_count(model):,}")
    print("  PASSED")


def test_variable_point_count():
    separator("Test 7: variable number of points per call")
    config = PointCloudTransformerConfig(
        num_point_types=50, d_model=64, n_heads=4, n_layers=2
    )
    model = PointCloudTransformer(config).to(DEVICE).eval()

    with torch.no_grad():
        for n in [4, 16, 64, 128]:
            coords, types = make_batch(1, n, config.num_point_types)
            out = model(coords, types)
            assert out.shape == (1, n, config.d_model)
            print(f"  N={n:4d} → output {tuple(out.shape)}  OK")

    print("  PASSED")


def test_distance_bias():
    separator("Test 8: distance_bias PE — rotation & translation invariance")
    config = PointCloudTransformerConfig(
        num_point_types=20, d_model=64, n_heads=4, n_layers=2,
        pe='distance_bias', dropout=0.0
    )
    model = PointCloudTransformer(config).to(DEVICE).eval()

    B, N = 1, 8
    coords, types = make_batch(B, N, config.num_point_types)

    with torch.no_grad():
        out_orig = model(coords, types)

        # Translation: shift all atoms by a constant vector — distances unchanged
        shift = torch.tensor([[[1.5, -2.0, 0.7]]], device=DEVICE)
        out_translated = model(coords + shift, types)

        # Rotation: apply a random 3×3 orthogonal matrix
        Q, _ = torch.linalg.qr(torch.randn(3, 3, device=DEVICE))
        out_rotated = model((coords @ Q.T), types)

    trans_diff  = (out_orig - out_translated).abs().max().item()
    rotate_diff = (out_orig - out_rotated).abs().max().item()

    assert trans_diff  < 1e-4, f"Translation not invariant: max diff = {trans_diff:.2e}"
    assert rotate_diff < 1e-4, f"Rotation not invariant:    max diff = {rotate_diff:.2e}"

    print(f"  Translation invariance error: {trans_diff:.2e}")
    print(f"  Rotation    invariance error: {rotate_diff:.2e}")
    print("  PASSED  (distance_bias is translation & rotation invariant)")


def test_cutoff_radius():
    separator("Test 9: distance_bias with cutoff_radius")
    # Coords tightly clustered around origin (all within cutoff)
    # vs one atom placed far away (beyond cutoff)
    cutoff = 3.0
    config = PointCloudTransformerConfig(
        num_point_types=10, d_model=64, n_heads=4, n_layers=2,
        pe='distance_bias', dropout=0.0,
        cutoff_radius=cutoff,
        pe_kwargs={'n_rbf': 16, 'rbf_range': (0.0, cutoff)},
    )
    model = PointCloudTransformer(config).to(DEVICE).eval()

    B, N = 1, 6
    # Close cluster: all atoms within cutoff of each other
    coords_close = torch.rand(B, N, 3, device=DEVICE) * 0.5   # max dist ~0.87
    types = torch.zeros(B, N, dtype=torch.long, device=DEVICE)

    # Same cluster but with one atom moved far beyond the cutoff
    coords_far = coords_close.clone()
    coords_far[0, -1] = torch.tensor([100.0, 100.0, 100.0])   # atom 5 is isolated

    with torch.no_grad():
        out_close = model(coords_close, types)
        out_far   = model(coords_far, types)

    # Forward pass should not produce NaNs even with an isolated atom
    assert not out_close.isnan().any(), "NaN in close-cluster output"
    assert not out_far.isnan().any(),   "NaN in output with isolated atom"

    # The representation of atoms 0–4 should differ between the two cases
    # (they now have one fewer neighbour within the cutoff)
    diff = (out_close[0, :N-1] - out_far[0, :N-1]).abs().max().item()
    print(f"  Cutoff={cutoff:.1f} — close cluster output:    {tuple(out_close.shape)}")
    print(f"  Representation shift (neighbours lost):   {diff:.2e}")
    assert diff > 0, "Outputs should differ when a neighbour moves beyond cutoff"

    # Verify invariance is preserved with cutoff
    shift = torch.tensor([[[5.0, -3.0, 1.0]]], device=DEVICE)
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=DEVICE))
    with torch.no_grad():
        out_trans  = model(coords_close + shift, types)
        out_rotate = model(coords_close @ Q.T, types)

    trans_err  = (out_close - out_trans).abs().max().item()
    rotate_err = (out_close - out_rotate).abs().max().item()
    assert trans_err  < 1e-4, f"Translation broken with cutoff: {trans_err:.2e}"
    assert rotate_err < 1e-4, f"Rotation broken with cutoff: {rotate_err:.2e}"
    print(f"  Translation invariance (with cutoff): {trans_err:.2e}")
    print(f"  Rotation    invariance (with cutoff): {rotate_err:.2e}")
    print("  PASSED")


def test_pe_kwargs_config_roundtrip():
    separator("Test 10: pe_kwargs survive config save/load")
    config = PointCloudTransformerConfig(
        num_point_types=10, d_model=64, n_heads=4, n_layers=2,
        pe='distance_bias', cutoff_radius=4.0,
        pe_kwargs={'n_rbf': 8, 'rbf_range': [0.0, 4.0]},
    )
    config.save('/tmp/cutoff_config.json')
    loaded = PointCloudTransformerConfig.load('/tmp/cutoff_config.json')

    assert loaded.cutoff_radius == 4.0
    assert loaded.pe_kwargs['n_rbf'] == 8
    print(f"  Saved:  cutoff_radius={config.cutoff_radius}, pe_kwargs={config.pe_kwargs}")
    print(f"  Loaded: cutoff_radius={loaded.cutoff_radius}, pe_kwargs={loaded.pe_kwargs}")
    print("  PASSED")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nPoint Cloud Transformer — Test Suite")
    print(f"Device: {DEVICE}")

    tests = [
        test_molecule_preset,
        test_no_pe,
        test_output_head,
        test_permutation_equivariance,
        test_config_save_load,
        test_protein_preset,
        test_variable_point_count,
        test_distance_bias,
        test_cutoff_radius,
        test_pe_kwargs_config_roundtrip,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  FAILED: {e}")

    separator(f"Results: {passed}/{len(tests)} tests passed")

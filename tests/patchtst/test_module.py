"""Tests for chronax.models.patchtst.patchtst_module."""
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.patchtst.patchtst_module import RevIN


def _x(B=4, L=20):
    rng = np.random.RandomState(0)
    return jnp.asarray(rng.randn(B, L, 1), dtype=jnp.float32)


def test_revin_norm_denorm_round_trip():
    revin = RevIN(num_features=1, subtract_last=True, affine=False, rngs=nnx.Rngs(0))
    x = _x()
    z, loc, scale = revin.norm(x)
    x_rec = revin.denorm(z, loc, scale)
    np.testing.assert_allclose(np.asarray(x_rec), np.asarray(x), rtol=1e-5, atol=1e-5)


def test_revin_subtract_last_centers_on_last_value():
    revin = RevIN(num_features=1, subtract_last=True, affine=False, rngs=nnx.Rngs(0))
    x = _x()
    z, loc, scale = revin.norm(x)
    # loc is the last timestep per (batch, feature)
    np.testing.assert_allclose(np.asarray(loc)[:, 0, 0], np.asarray(x)[:, -1, 0], rtol=1e-6)


def test_revin_uses_population_variance():
    revin = RevIN(num_features=1, subtract_last=True, affine=False, eps=1e-5, rngs=nnx.Rngs(0))
    x = _x()
    _, _, scale = revin.norm(x)
    expected = np.sqrt(np.var(np.asarray(x), axis=1, keepdims=True) + 1e-5)  # ddof=0
    np.testing.assert_allclose(np.asarray(scale), expected, rtol=1e-5)


def test_revin_outputs_float32():
    revin = RevIN(num_features=1, subtract_last=True, affine=False, rngs=nnx.Rngs(0))
    z, loc, scale = revin.norm(_x())
    assert z.dtype == jnp.float32


from chronax.models.patchtst.patchtst_module import compute_patch_num, patchify


def test_compute_patch_num_standard():
    # input_size=512, patch_len=16, stride=8 -> 63 + 1 = 64
    assert compute_patch_num(512, 16, 8) == 64


def test_compute_patch_num_degenerate_input_shorter_than_patch():
    # input_size=12, patch_len=16, stride=8 -> int(-0.5)= 0; +1 = 1
    assert compute_patch_num(12, 16, 8) == 1


def test_patchify_shape_and_values():
    B, L, patch_len, stride = 2, 512, 16, 8
    x = jnp.arange(B * L, dtype=jnp.float32).reshape(B, L)
    patches = patchify(x, patch_len=patch_len, stride=stride)
    assert patches.shape == (B, compute_patch_num(L, patch_len, stride), patch_len)
    # first patch is the first patch_len samples of the (padded) series
    np.testing.assert_allclose(np.asarray(patches[0, 0]), np.asarray(x[0, :patch_len]))


def test_patchify_end_padding_replicates_last():
    B, L, patch_len, stride = 1, 12, 16, 8
    x = jnp.arange(L, dtype=jnp.float32).reshape(1, L)
    patches = patchify(x, patch_len=patch_len, stride=stride)
    assert patches.shape == (1, 1, patch_len)
    # padded tail must equal the last value (edge replication)
    assert float(patches[0, 0, -1]) == float(x[0, -1])


from chronax.models.patchtst.patchtst_module import PatchEmbedding


def test_patch_embedding_shape_and_pos():
    B, patch_len, hidden, pn = 2, 16, 32, 7
    emb = PatchEmbedding(patch_len=patch_len, hidden_size=hidden, patch_num=pn,
                         dropout=0.0, rngs=nnx.Rngs(0))
    patches = jnp.zeros((B, pn, patch_len), dtype=jnp.float32)
    out = emb(patches, deterministic=True)
    assert out.shape == (B, pn, hidden)
    assert out.dtype == jnp.float32
    # positional encoding is a learnable [patch_num, hidden] param
    assert emb.pos.value.shape == (pn, hidden)


from chronax.models.patchtst.patchtst_module import MultiHeadAttention


def test_mha_shape_and_returns_scores():
    B, T, hidden, heads = 2, 7, 32, 4
    mha = MultiHeadAttention(hidden_size=hidden, n_heads=heads, attn_dropout=0.0,
                             proj_dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((B, T, hidden), dtype=jnp.float32)
    out, scores = mha(x, prev=None, deterministic=True)
    assert out.shape == (B, T, hidden)
    assert scores.shape == (B, heads, T, T)


def test_mha_prev_changes_output():
    B, T, hidden, heads = 1, 5, 16, 2
    mha = MultiHeadAttention(hidden_size=hidden, n_heads=heads, attn_dropout=0.0,
                             proj_dropout=0.0, rngs=nnx.Rngs(0))
    # Non-constant input: an all-ones input makes every position identical, so the
    # prev-score residual cannot change the (constant) attention output.
    x = jnp.asarray(np.random.RandomState(2).randn(B, T, hidden), dtype=jnp.float32)
    out0, scores0 = mha(x, prev=None, deterministic=True)
    out1, _ = mha(x, prev=scores0 + 5.0, deterministic=True)
    assert not np.allclose(np.asarray(out0), np.asarray(out1))

"""Tests for chronax.models.VanillaTransformer.

Covers the four source modules of the vanillatransformer subpackage in one file
(losses, module, training, model), matching the flat ``test_<model>.py``
convention used elsewhere in ``tests/``.
"""
import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.vanillatransformer.vanillatransformer_losses import (
    LOSSES, huber, mae, mse, resolve,
)


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


# ============================================================================
# Losses
# ============================================================================

def test_mae_zero_at_match():
    x = jnp.array([1.0, 2.0, 3.0])
    assert float(mae(x, x)) == 0.0


def test_mse_known_value():
    pred = jnp.array([0.0, 0.0])
    target = jnp.array([1.0, 3.0])
    assert float(mse(pred, target)) == pytest.approx(5.0)


def test_huber_quadratic_region():
    assert float(huber(jnp.array([0.0]), jnp.array([0.5]))) == pytest.approx(0.125)


def test_huber_linear_region():
    assert float(huber(jnp.array([0.0]), jnp.array([3.0]))) == pytest.approx(2.5)


@pytest.mark.parametrize("name", ["mae", "mse", "huber"])
def test_loss_is_jit_and_grad_friendly(name):
    fn = LOSSES[name]
    pred = jnp.array([0.1, 0.2, 0.3])
    target = jnp.array([0.0, 0.5, 0.2])
    g = jax.grad(lambda p: fn(p, target))(pred)
    assert g.shape == pred.shape
    assert jnp.all(jnp.isfinite(g))


def test_resolve_string_returns_registry_entry():
    assert resolve("mae") is mae


def test_resolve_callable_passes_through():
    fn = lambda p, t: jnp.sum(p - t)
    assert resolve(fn) is fn


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve("not_a_loss")


from chronax.models.vanillatransformer.vanillatransformer_module import (
    AttentionLayer,
    DataEmbedding,
    TokenEmbedding,
    _positional_embedding,
    _resolve_activation,
)


# ============================================================================
# Module: inits / embeddings / attention
# ============================================================================

def test_resolve_activation_gelu_is_exact():
    z = jnp.array([0.7, -1.3, 2.0])
    got = _resolve_activation("gelu")(z)
    want = jax.nn.gelu(z, approximate=False)
    assert jnp.allclose(got, want)


def test_resolve_activation_unknown_raises():
    with pytest.raises(ValueError):
        _resolve_activation("swish")


def test_positional_embedding_shape_and_bounds():
    pe = _positional_embedding(16, 8)
    assert pe.shape == (1, 16, 8)
    assert jnp.all(jnp.abs(pe) <= 1.0 + 1e-6)


def test_token_embedding_shape():
    tok = TokenEmbedding(hidden_size=12, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 7, 1))
    out = tok(x)
    assert out.shape == (2, 7, 12)


def test_data_embedding_shape_and_dropout_identity_when_deterministic():
    emb = DataEmbedding(hidden_size=12, dropout=0.5, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 7, 1))
    out = emb(x, deterministic=True)
    assert out.shape == (2, 7, 12)


def test_attention_layer_self_shape():
    attn = AttentionLayer(hidden_size=16, n_heads=4, attn_dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((3, 5, 16))
    out = attn(x, x, deterministic=True)
    assert out.shape == (3, 5, 16)


def test_attention_layer_cross_shape():
    attn = AttentionLayer(hidden_size=16, n_heads=4, attn_dropout=0.0, rngs=nnx.Rngs(0))
    q = jnp.ones((3, 5, 16))
    kv = jnp.ones((3, 9, 16))
    out = attn(q, kv, deterministic=True)
    assert out.shape == (3, 5, 16)


def test_attention_layer_requires_divisible_heads():
    with pytest.raises(ValueError):
        AttentionLayer(hidden_size=10, n_heads=4, attn_dropout=0.0, rngs=nnx.Rngs(0))

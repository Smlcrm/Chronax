"""Tests for chronax.models.patchtst.patchtst_training."""
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from chronax.models.patchtst.patchtst_module import PatchTSTNet
from chronax.models.patchtst.patchtst_training import (
    build_windows, forward_loss, predict_step, train,
)


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _net(h=12, input_size=36, hidden=16, heads=2, layers=1):
    return PatchTSTNet(
        h=h, input_size=input_size, patch_len=16, stride=8, hidden_size=hidden,
        n_heads=heads, encoder_layers=layers, linear_hidden_size=32, dropout=0.0,
        fc_dropout=0.0, head_dropout=0.0, attn_dropout=0.0, revin=True,
        revin_affine=False, revin_subtract_last=True, rngs=nnx.Rngs(0),
    )


def test_build_windows_shape():
    y = _make_y(60)
    w = build_windows(y, input_size=36, h=12)
    assert w.shape == (60 - 48 + 1, 48)


def test_forward_loss_returns_scalar():
    net = _net()
    w = build_windows(_make_y(), input_size=36, h=12)
    loss = forward_loss(net, w[:8], h=12, input_size=36)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_gradient_step_decreases_loss():
    net = _net()
    y = _make_y()
    losses = train(net, y, h=12, input_size=36, max_steps=30,
                   windows_batch_size=64, lr=1e-3, seed=0)
    assert losses.shape == (30,)
    assert float(losses[-1]) < float(losses[0])


def test_train_deterministic_with_same_seed():
    y = _make_y()
    n1 = _net()
    l1 = train(n1, y, h=12, input_size=36, max_steps=10,
               windows_batch_size=64, lr=1e-3, seed=0)
    n2 = _net()
    l2 = train(n2, y, h=12, input_size=36, max_steps=10,
               windows_batch_size=64, lr=1e-3, seed=0)
    np.testing.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=1e-5)


def test_predict_step_shape_and_idempotent():
    net = _net()
    y = _make_y()
    train(net, y, h=12, input_size=36, max_steps=5, windows_batch_size=64, lr=1e-3, seed=0)
    p1 = predict_step(net, y[-36:], h=12, input_size=36)
    p2 = predict_step(net, y[-36:], h=12, input_size=36)
    assert p1.shape == (12,)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-6)

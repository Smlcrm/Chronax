"""Tests for chronax.models.kan.kan_training."""
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.kan.kan_module import KANNet
from chronax.models.kan.kan_scaler import IdentityScaler
from chronax.models.kan.kan_training import build_windows, predict_step, scaled_forward_loss, train


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _net(h=12, input_size=36, hidden=16):
    return KANNet(h=h, input_size=input_size, n_hidden_layers=1, hidden_size=hidden, grid_size=5,
                  spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                  enable_standalone_scale_spline=True, grid_range=(-1.0, 1.0), rngs=nnx.Rngs(0))


def test_build_windows_shape():
    assert build_windows(_make_y(60), input_size=36, h=12).shape == (60 - 48 + 1, 48)


def test_scaled_forward_loss_returns_scalar():
    net = _net()
    w = build_windows(_make_y(), input_size=36, h=12)
    loss = scaled_forward_loss(net, w[:8], h=12, input_size=36, scaler=IdentityScaler())
    assert loss.shape == () and jnp.isfinite(loss)


def test_train_gradient_step_decreases_loss():
    net = _net()
    losses = train(net, _make_y(), h=12, input_size=36, max_steps=30, windows_batch_size=64,
                   lr=1e-3, seed=0, scaler=IdentityScaler())
    assert losses.shape == (30,) and float(losses[-1]) < float(losses[0])


def test_train_deterministic_with_same_seed():
    y = _make_y()
    l1 = train(_net(), y, h=12, input_size=36, max_steps=10, windows_batch_size=64, lr=1e-3, seed=0, scaler=IdentityScaler())
    l2 = train(_net(), y, h=12, input_size=36, max_steps=10, windows_batch_size=64, lr=1e-3, seed=0, scaler=IdentityScaler())
    np.testing.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=1e-5)


def test_train_oversample_with_replacement_small_n():
    # y(60), input_size=36, h=12 -> n_windows=13 < windows_batch_size=64
    losses = train(_net(), _make_y(60), h=12, input_size=36, max_steps=8, windows_batch_size=64,
                   lr=1e-3, seed=0, scaler=IdentityScaler())
    assert losses.shape == (8,) and jnp.all(jnp.isfinite(losses))


def test_train_raises_on_divergence():
    with pytest.raises(RuntimeError, match="diverged"):
        train(_net(), _make_y(), h=12, input_size=36, max_steps=10, windows_batch_size=64,
              lr=1e9, seed=0, scaler=IdentityScaler())


def test_predict_step_shape_and_idempotent():
    net = _net()
    y = _make_y()
    train(net, y, h=12, input_size=36, max_steps=5, windows_batch_size=64, lr=1e-3, seed=0, scaler=IdentityScaler())
    p1 = predict_step(net, y[-36:], h=12, input_size=36, scaler=IdentityScaler())
    p2 = predict_step(net, y[-36:], h=12, input_size=36, scaler=IdentityScaler())
    assert p1.shape == (12,)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-6)

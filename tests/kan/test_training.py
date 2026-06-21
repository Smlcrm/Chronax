"""Tests for chronax.models.kan.kan_training."""
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.kan.kan_module import KANNet
from chronax.models.kan.kan_scaler import IdentityScaler
from chronax.models.kan.kan_training import build_windows, scaled_forward_loss


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

"""Tests for chronax.models.gru.gru_training — windows, forward+loss, train, predict."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.gru.gru_module import GRUNet
from chronax.models.gru.gru_scaler import RobustScaler
from chronax.models.gru.gru_training import build_windows, scaled_forward_loss


def test_build_windows_shape_and_no_leakage():
    y = jnp.arange(20.0)
    w = build_windows(y, input_size=5, h=3)
    # n_windows = T - L - h + 1 = 20 - 5 - 3 + 1 = 13
    assert w.shape == (13, 8)
    np.testing.assert_allclose(w[0], jnp.arange(8.0))
    np.testing.assert_allclose(w[-1], jnp.arange(12.0, 20.0))


def test_build_windows_raises_on_short_series():
    with pytest.raises(ValueError):
        build_windows(jnp.arange(5.0), input_size=5, h=3)


def test_scaled_forward_loss_returns_scalar():
    net = GRUNet(
        in_features=1, encoder_hidden=8, encoder_layers=1,
        decoder_hidden=4, decoder_layers=2, dropout=0.0,
        h=3, input_size=5, rngs=nnx.Rngs(0),
    )
    windows = jnp.arange(2 * 8, dtype=jnp.float32).reshape(2, 8)
    loss = scaled_forward_loss(net, windows, h=3, input_size=5, scaler=RobustScaler())
    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert jnp.isfinite(loss)


def test_scaled_forward_loss_is_scale_invariant():
    """Scaling the input series by 100x shouldn't change the loss in scaled
    space — the per-window scaler normalizes both insample and target."""
    net = GRUNet(
        in_features=1, encoder_hidden=8, encoder_layers=1,
        decoder_hidden=4, decoder_layers=2, dropout=0.0,
        h=3, input_size=5, rngs=nnx.Rngs(0),
    )
    rng = np.random.default_rng(0)
    base = jnp.asarray(rng.standard_normal((2, 8)), dtype=jnp.float32)
    sc = RobustScaler()
    l1 = scaled_forward_loss(net, base, h=3, input_size=5, scaler=sc)
    l2 = scaled_forward_loss(net, base * 100.0, h=3, input_size=5, scaler=sc)
    np.testing.assert_allclose(float(l1), float(l2), rtol=1e-3)

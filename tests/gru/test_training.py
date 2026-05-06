"""Tests for chronax.models.gru.gru_training — windows, forward+loss, train, predict."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.gru.gru_module import GRUNet
from chronax.models.gru.gru_scaler import RobustScaler
from chronax.models.gru.gru_training import build_windows


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

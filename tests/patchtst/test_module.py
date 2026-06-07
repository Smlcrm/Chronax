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

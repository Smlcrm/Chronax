"""Tests for chronax.models.kan.kan_scaler."""
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.kan.kan_scaler import IdentityScaler, RobustScaler, resolve_scaler


def _x():
    return jnp.asarray(np.random.RandomState(0).randn(4, 12), dtype=jnp.float32)


def test_identity_is_noop():
    s = IdentityScaler()
    x = _x()
    shift, scale = s.stats(x, axis=1)
    np.testing.assert_array_equal(np.asarray(shift), np.zeros_like(np.asarray(shift)))
    np.testing.assert_array_equal(np.asarray(scale), np.ones_like(np.asarray(scale)))
    np.testing.assert_array_equal(np.asarray(s.transform(x, shift, scale)), np.asarray(x))


def test_robust_round_trip():
    s = RobustScaler()
    x = _x()
    shift, scale = s.stats(x, axis=1)
    rec = s.inverse(s.transform(x, shift, scale), shift, scale)
    np.testing.assert_allclose(np.asarray(rec), np.asarray(x), rtol=1e-5, atol=1e-5)


def test_robust_mad_zero_fallback_constant_row():
    s = RobustScaler()
    x = jnp.ones((2, 8), dtype=jnp.float32)
    _, scale = s.stats(x, axis=1)
    assert jnp.all(jnp.isfinite(scale)) and jnp.all(scale > 0)


def test_resolve_scaler():
    assert isinstance(resolve_scaler("identity"), IdentityScaler)
    assert isinstance(resolve_scaler("robust"), RobustScaler)
    with pytest.raises(ValueError, match="Unknown scaler"):
        resolve_scaler("nope")

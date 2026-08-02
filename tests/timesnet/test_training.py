"""Tests for chronax.models.timesnet training infra + module."""
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.timesnet.timesnet_losses import mae as loss_mae, resolve as resolve_loss
from chronax.models.timesnet.timesnet_scaler import IdentityScaler, resolve_scaler


def test_masked_mae_drops_masked_elements():
    pred = jnp.array([[1.0, 2.0]]); target = jnp.array([[0.0, 0.0]])
    mask = jnp.array([[1.0, 0.0]])
    assert float(loss_mae(pred, target, mask)) == 1.0     # only the unmasked |1-0|


def test_resolve_loss_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        resolve_loss("nope")


def test_scaler_resolve_and_roundtrip():
    x = jnp.array([[1.0, 2.0, 3.0, 100.0]])
    for name in ("identity", "robust", "standard"):
        s = resolve_scaler(name)
        shift, scale = s.stats(x, axis=1)
        z = s.transform(x, shift, scale)
        np.testing.assert_allclose(np.asarray(s.inverse(z, shift, scale)), np.asarray(x), rtol=1e-5)
    with pytest.raises(ValueError, match="Unknown scaler"):
        resolve_scaler("nope")


def test_standard_scaler_hand_computed():
    from chronax.models.timesnet.timesnet_scaler import StandardScaler
    x = jnp.array([[1.0, 2.0, 3.0, 6.0]])
    s = StandardScaler()
    shift, scale = s.stats(x, axis=1)
    np.testing.assert_allclose(float(shift[0, 0]), 3.0, rtol=1e-6)
    np.testing.assert_allclose(float(scale[0, 0]), np.std([1.0, 2.0, 3.0, 6.0]) + 1e-6, rtol=1e-6)
    z = s.transform(x, shift, scale)
    np.testing.assert_allclose(np.asarray(s.inverse(z, shift, scale)), np.asarray(x), rtol=1e-5)


def test_standard_scaler_constant_window():
    from chronax.models.timesnet.timesnet_scaler import StandardScaler
    shift, scale = StandardScaler().stats(jnp.full((1, 8), 5.0), axis=1)
    np.testing.assert_allclose(float(scale[0, 0]), 1.0 + 1e-6, rtol=1e-6)  # std==0 -> 1.0 THEN +eps


def test_resolve_scaler_standard():
    from chronax.models.timesnet.timesnet_scaler import StandardScaler, resolve_scaler
    assert isinstance(resolve_scaler("standard"), StandardScaler)

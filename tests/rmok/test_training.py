"""Tests for chronax.models.rmok training infra + module."""
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.rmok.rmok_losses import mae as loss_mae, resolve as resolve_loss
from chronax.models.rmok.rmok_scaler import IdentityScaler, resolve_scaler


def test_masked_mae_drops_masked_elements():
    pred = jnp.array([[1.0, 2.0]]); target = jnp.array([[0.0, 0.0]])
    mask = jnp.array([[1.0, 0.0]])
    assert float(loss_mae(pred, target, mask)) == 1.0     # only the unmasked |1-0|


def test_resolve_loss_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        resolve_loss("nope")


def test_scaler_resolve_and_roundtrip():
    x = jnp.array([[1.0, 2.0, 3.0, 100.0]])
    for name in ("identity", "robust"):
        s = resolve_scaler(name)
        shift, scale = s.stats(x, axis=1)
        z = s.transform(x, shift, scale)
        np.testing.assert_allclose(np.asarray(s.inverse(z, shift, scale)), np.asarray(x), rtol=1e-5)
    with pytest.raises(ValueError, match="Unknown scaler"):
        resolve_scaler("nope")

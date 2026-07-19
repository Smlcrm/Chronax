"""Tests for chronax.models.nlinear training infra + module."""
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.nlinear.nlinear_losses import mae as loss_mae, resolve as resolve_loss
from chronax.models.nlinear.nlinear_scaler import IdentityScaler, RobustScaler, resolve_scaler


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


from chronax.models.nlinear.nlinear_module import NLinearNet


def test_forward_matches_hand_computed():
    net = NLinearNet(h=2, input_size=3, rngs=nnx.Rngs(0))
    W = jnp.array([[1.0, 0.0, 2.0], [0.5, -1.0, 0.0]])   # [h=2, in=3]
    b = jnp.array([0.1, -0.2])
    net.weight.value = W; net.bias.value = b
    y = jnp.array([[1.0, 2.0, 4.0]])                      # last = 4
    out = net(y[..., None])                               # [1, 2, 1]
    norm = y - 4.0                                        # [-3, -2, 0]
    expected = norm @ W.T + b + 4.0
    np.testing.assert_allclose(np.asarray(out[..., 0]), np.asarray(expected), rtol=1e-6)


def test_init_matches_torch_bounds():
    net = NLinearNet(h=24, input_size=72, rngs=nnx.Rngs(0))
    bound = 1.0 / np.sqrt(72)
    w = np.asarray(net.weight.value); b = np.asarray(net.bias.value)
    assert w.shape == (24, 72) and b.shape == (24,)
    assert np.abs(w).max() <= bound and np.abs(b).max() <= bound
    assert w.std() > 0.2 * bound          # non-degenerate uniform, not zeros/normal
    assert np.abs(b).max() > 0.0          # torch uses uniform bias, NOT zeros

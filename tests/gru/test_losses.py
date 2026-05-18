"""Tests for chronax.models.gru.gru_losses."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.gru.gru_losses import LOSSES, huber, mae, mse, resolve


def test_mae_zero_at_match():
    y = jnp.asarray([1.0, 2.0, 3.0])
    assert float(mae(y, y)) == 0.0


def test_mae_known_value():
    pred = jnp.asarray([1.0, 2.0, 3.0])
    target = jnp.asarray([2.0, 0.0, 6.0])
    # |−1| + |2| + |−3| = 6 ; mean = 2.0
    np.testing.assert_allclose(float(mae(pred, target)), 2.0)


def test_mse_known_value():
    pred = jnp.asarray([1.0, 2.0])
    target = jnp.asarray([2.0, 0.0])
    # (1 + 4) / 2 = 2.5
    np.testing.assert_allclose(float(mse(pred, target)), 2.5)


def test_huber_quadratic_region():
    """For residuals with |r| <= 1, Huber equals 0.5 * r**2."""
    pred = jnp.asarray([0.0, 0.5])
    target = jnp.asarray([0.5, 0.0])
    # both residuals have |r| = 0.5 -> 0.5 * 0.25 = 0.125
    np.testing.assert_allclose(float(huber(pred, target)), 0.125)


def test_huber_linear_region():
    """For residuals with |r| > 1, Huber equals |r| - 0.5."""
    pred = jnp.asarray([0.0])
    target = jnp.asarray([3.0])
    # |r| = 3 -> 3 - 0.5 = 2.5
    np.testing.assert_allclose(float(huber(pred, target)), 2.5)


def test_huber_smooth_at_transition():
    """Quadratic and linear branches must agree at |r| = 1."""
    pred = jnp.asarray([0.0])
    target_just_under = jnp.asarray([1.0 - 1e-6])
    target_just_over = jnp.asarray([1.0 + 1e-6])
    np.testing.assert_allclose(
        float(huber(pred, target_just_under)),
        float(huber(pred, target_just_over)),
        atol=1e-5,
    )


@pytest.mark.parametrize("name", ["mae", "mse", "huber"])
def test_loss_is_jit_and_grad_friendly(name):
    """Every registered loss must trace through jit + grad cleanly."""
    fn = LOSSES[name]
    pred = jnp.asarray([1.0, 2.0, 3.0])
    target = jnp.asarray([2.0, 1.0, 4.0])

    jit_fn = jax.jit(fn)
    np.testing.assert_allclose(float(jit_fn(pred, target)), float(fn(pred, target)))

    grad_fn = jax.grad(fn)
    g = grad_fn(pred, target)
    assert g.shape == pred.shape
    assert jnp.all(jnp.isfinite(g))


def test_resolve_string_returns_registry_entry():
    assert resolve("mae") is mae
    assert resolve("mse") is mse
    assert resolve("huber") is huber


def test_resolve_callable_passes_through():
    def custom(pred, target):
        return jnp.mean(jnp.abs(pred - target))

    assert resolve(custom) is custom


def test_resolve_unknown_string_raises_value_error():
    with pytest.raises(ValueError, match="Unknown loss"):
        resolve("rmse")


def test_registry_keys_are_stable():
    """If we ever change registry keys, the GRU.loss=string API breaks."""
    assert set(LOSSES.keys()) == {"mae", "mse", "huber"}

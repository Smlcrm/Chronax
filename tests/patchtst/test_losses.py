"""Tests for chronax.models.patchtst.patchtst_losses."""
import jax
import jax.numpy as jnp
import pytest

from chronax.models.patchtst.patchtst_losses import LOSSES, huber, mae, mse, resolve


def test_mae_zero_at_match():
    x = jnp.array([1.0, 2.0, 3.0])
    assert float(mae(x, x)) == 0.0


def test_mse_known_value():
    pred = jnp.array([0.0, 0.0])
    target = jnp.array([1.0, 3.0])
    assert float(mse(pred, target)) == pytest.approx(5.0)  # (1 + 9) / 2


def test_huber_quadratic_region():
    pred = jnp.array([0.0])
    target = jnp.array([0.5])
    # |r| = 0.5 <= 1 -> 0.5 * r^2 = 0.125
    assert float(huber(pred, target)) == pytest.approx(0.125)


def test_huber_linear_region():
    pred = jnp.array([0.0])
    target = jnp.array([3.0])
    # |r| = 3 > 1 -> |r| - 0.5 = 2.5
    assert float(huber(pred, target)) == pytest.approx(2.5)


@pytest.mark.parametrize("name", ["mae", "mse", "huber"])
def test_loss_is_jit_and_grad_friendly(name):
    fn = LOSSES[name]
    pred = jnp.array([0.1, 0.2, 0.3])
    target = jnp.array([0.0, 0.5, 0.2])
    g = jax.grad(lambda p: fn(p, target))(pred)
    assert g.shape == pred.shape
    assert jnp.all(jnp.isfinite(g))


def test_resolve_string_returns_registry_entry():
    assert resolve("mae") is mae


def test_resolve_callable_passes_through():
    fn = lambda p, t: jnp.sum(p - t)
    assert resolve(fn) is fn


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        resolve("nope")


def test_registry_keys_are_stable():
    assert set(LOSSES.keys()) == {"mae", "mse", "huber"}

"""Tests for chronax.models.FEDformer — BaseForecaster contract conformance."""
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.fedformer.forecaster import FEDformer


def _make_y(n: int = 200) -> jnp.ndarray:
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _tiny() -> FEDformer:
    return FEDformer(
        h=12,
        input_size=36,
        hidden_size=16,
        n_heads=8,
        modes=8,
        encoder_layers=1,
        decoder_layers=1,
        max_steps=20,
        batch_size=8,
        random_seed=0,
        val_fraction=0.0,
    )


def test_fedformer_inherits_baseforecaster():
    assert issubclass(FEDformer, BaseForecaster)


def test_input_size_default_is_3h():
    m = FEDformer(h=12, hidden_size=16, n_heads=8, modes=8, max_steps=1, batch_size=2)
    assert m.input_size == 36


def test_fit_returns_self_and_sets_model_attribute():
    m = _tiny()
    out = m.fit(_make_y())
    assert out is m
    assert m.model_ is not None


def test_predict_default_returns_h_steps():
    m = _tiny().fit(_make_y())
    fcst = m.predict(h=12)
    assert isinstance(fcst, dict) and "mean" in fcst
    assert fcst["mean"].shape == (12,)


def test_predict_with_smaller_h_slices():
    m = _tiny().fit(_make_y())
    assert m.predict(h=6)["mean"].shape == (6,)


def test_predict_with_larger_h_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError):
        m.predict(h=24)


def test_forecast_stateless():
    m = _tiny()
    fcst = m.forecast(_make_y(), h=12)
    assert "mean" in fcst and fcst["mean"].shape == (12,)


def test_constructor_core_args_for_registry():
    """Neural harness requires h, input_size, random_seed in __init__."""
    import inspect
    params = set(inspect.signature(FEDformer.__init__).parameters)
    assert {"h", "input_size", "random_seed"} <= params


def test_uses_exog_true_but_none_valid():
    assert FEDformer(h=12).uses_exog is True
    m = _tiny().fit(_make_y())
    assert m._futr_size == 0
    assert m.predict(h=12)["mean"].shape == (12,)


def test_fit_X_not_implemented():
    with pytest.raises(NotImplementedError, match="futr_exog"):
        _tiny().fit(_make_y(), X=jnp.ones((200, 1), jnp.float32))


def test_futr_exog_fit_predict_shapes():
    y = _make_y(240)
    T = int(y.shape[0])
    futr = jnp.asarray(np.random.RandomState(1).randn(T, 2), jnp.float32)
    m = _tiny()
    m.max_steps = 10
    m = m.fit(y, futr_exog=futr)
    out = m.predict(h=12, futr_exog=jnp.asarray(np.random.RandomState(2).randn(12, 2), jnp.float32))
    assert out["mean"].shape == (12,)
    assert bool(jnp.all(jnp.isfinite(out["mean"])))


def test_futr_required_at_predict_raises():
    y = _make_y(240)
    futr = jnp.asarray(np.random.RandomState(1).randn(int(y.shape[0]), 1), jnp.float32)
    m = _tiny()
    m.max_steps = 5
    m = m.fit(y, futr_exog=futr)
    with pytest.raises(ValueError, match="futr_exog"):
        m.predict(h=12)


def test_futr_wrong_shape_raises():
    y = _make_y(240)
    futr = jnp.asarray(np.random.RandomState(1).randn(int(y.shape[0]), 2), jnp.float32)
    m = _tiny()
    m.max_steps = 5
    m = m.fit(y, futr_exog=futr)
    with pytest.raises(ValueError, match="futr_exog"):
        m.predict(h=12, futr_exog=jnp.ones((12, 3), jnp.float32))


def test_different_futr_yields_different_preds():
    y = _make_y(240)
    T = int(y.shape[0])
    futr = jnp.asarray(np.random.RandomState(1).randn(T, 1), jnp.float32)
    m = _tiny()
    m.max_steps = 15
    m = m.fit(y, futr_exog=futr)
    a = np.asarray(m.predict(h=12, futr_exog=jnp.zeros((12, 1), jnp.float32))["mean"])
    b = np.asarray(m.predict(h=12, futr_exog=jnp.ones((12, 1), jnp.float32))["mean"])
    assert not np.allclose(a, b)
"""Tests for chronax.models.Autoformer — BaseForecaster contract conformance."""
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.autoformer.forecaster import Autoformer
from chronax.models.base_forecaster import BaseForecaster


def _make_y(n: int = 200) -> jnp.ndarray:
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _tiny() -> Autoformer:
    return Autoformer(
        h=12,
        input_size=36,
        hidden_size=16,
        n_heads=2,
        encoder_layers=1,
        decoder_layers=1,
        max_steps=20,
        batch_size=8,
        random_seed=0,
        val_fraction=0.0,
    )


def test_autoformer_inherits_baseforecaster():
    assert issubclass(Autoformer, BaseForecaster)


def test_input_size_default_is_3h():
    m = Autoformer(h=12, hidden_size=8, n_heads=2, max_steps=1, batch_size=2)
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
    params = set(inspect.signature(Autoformer.__init__).parameters)
    assert {"h", "input_size", "random_seed"} <= params

"""Tests for chronax.models.gru.gru_model — BaseForecaster contract conformance."""
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.gru.gru_model import GRU


def _make_y(n: int = 200) -> jnp.ndarray:
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _tiny() -> GRU:
    return GRU(
        h=12, input_size=36, hidden_size=16, n_layers=1,
        max_steps=20, batch_size=8, random_seed=0,
    )


def test_gru_inherits_baseforecaster():
    assert issubclass(GRU, BaseForecaster)


def test_gru_init_sets_conformal_params_to_none():
    m = GRU(h=12, input_size=36, hidden_size=8, n_layers=1, max_steps=1, batch_size=2)
    assert m.conformal_params is None


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
    """BaseForecaster contract: predict(h) accepts any h <= self.h."""
    m = _tiny().fit(_make_y())
    assert m.predict(h=6)["mean"].shape == (6,)


def test_predict_with_larger_h_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError):
        m.predict(h=24)


def test_forecast_stateless():
    m = _tiny()
    fcst = m.forecast(_make_y(), h=12)
    assert fcst["mean"].shape == (12,)


def test_predict_deterministic_with_same_seed():
    def run():
        m = GRU(
            h=12, input_size=36, hidden_size=16, n_layers=1,
            max_steps=20, batch_size=8, random_seed=42,
        )
        m.fit(_make_y())
        return m.predict(h=12)["mean"]

    np.testing.assert_allclose(run(), run(), rtol=1e-5)

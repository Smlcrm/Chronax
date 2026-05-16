"""Tests for chronax.models.gru.gru_model — BaseForecaster contract conformance."""
import pickle

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


def test_fit_raises_on_exog():
    m = _tiny()
    with pytest.raises(NotImplementedError, match="Exogenous"):
        m.fit(_make_y(), X=jnp.zeros((10, 2)))


def test_fit_raises_on_short_series():
    m = _tiny()  # h=12, input_size=36 -> needs 48
    with pytest.raises(ValueError, match="too short"):
        m.fit(jnp.arange(30, dtype=jnp.float32))


def test_predict_before_fit_raises():
    m = _tiny()
    with pytest.raises(RuntimeError, match="fit"):
        m.predict(h=12)


def test_predict_with_level_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(NotImplementedError, match="intervals"):
        m.predict(h=12, level=[80, 95])


def test_forecast_raises_on_exog():
    m = _tiny()
    with pytest.raises(NotImplementedError, match="Exogenous"):
        m.forecast(_make_y(), h=12, X=jnp.zeros((10, 2)))
    with pytest.raises(NotImplementedError, match="Exogenous"):
        m.forecast(_make_y(), h=12, X_future=jnp.zeros((12, 2)))


def test_forecast_raises_on_fitted_request():
    m = _tiny()
    with pytest.raises(NotImplementedError, match="fitted"):
        m.forecast(_make_y(), h=12, fitted=True)


def test_forecast_equals_fit_then_predict_with_same_seed():
    """`forecast` is the stateless fit-then-predict path; must agree numerically."""
    y = _make_y()

    def fresh():
        return GRU(
            h=12, input_size=36, hidden_size=16, n_layers=1,
            max_steps=20, batch_size=8, random_seed=7,
        )

    a = fresh().forecast(y, h=12)["mean"]
    b = fresh().fit(y).predict(h=12)["mean"]
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_model_beats_naive_last_value_on_easy_signal():
    """End-to-end sanity: with a clean periodic signal and a few hundred
    training steps, the model should beat a naive 'last value' forecast.
    Catches broken loss / optimizer / scaler-inverse wiring that would still
    pass shape and determinism tests."""
    n = 400
    t = np.arange(n)
    y = jnp.asarray(np.sin(t / 5.0), dtype=jnp.float32)
    train_y = y[:-12]
    test_y = np.asarray(y[-12:])

    m = GRU(
        h=12, input_size=36, hidden_size=32, n_layers=1,
        max_steps=200, batch_size=32, random_seed=0,
    )
    m.fit(train_y)
    pred = np.asarray(m.predict(h=12)["mean"])

    naive_last = np.full(12, float(train_y[-1]))
    mae_model = float(np.mean(np.abs(pred - test_y)))
    mae_naive = float(np.mean(np.abs(naive_last - test_y)))
    assert mae_model < mae_naive, (
        f"model MAE {mae_model:.4f} did not beat naive-last MAE {mae_naive:.4f}"
    )


def test_gru_pickle_round_trip():
    """A fitted GRU survives pickle.dumps/loads and produces identical predictions."""
    y = _make_y()
    model = _tiny()
    model.fit(y)
    pred_before = np.asarray(model.predict(h=12)["mean"])

    blob = pickle.dumps(model)
    restored = pickle.loads(blob)

    pred_after = np.asarray(restored.predict(h=12)["mean"])
    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)


def test_fit_predict_on_constant_series_returns_finite():
    """Constant input → MAD=0 → scaler fallback fires. Forecast must be finite."""
    y = jnp.ones(200, dtype=jnp.float32) * 5.0
    model = _tiny()  # h=12, input_size=36
    model.fit(y)
    pred = np.asarray(model.predict(h=12)["mean"])
    assert np.all(np.isfinite(pred)), f"got non-finite forecast: {pred}"


def test_fit_works_at_minimum_series_length():
    """Series of exactly input_size + h trains without ValueError (n_windows == 1)."""
    h, L = 4, 12
    y = jnp.asarray(np.sin(np.arange(L + h) / 3.0), dtype=jnp.float32)
    model = GRU(h=h, input_size=L, hidden_size=8, n_layers=1,
                max_steps=5, batch_size=1, random_seed=0)
    model.fit(y)
    pred = model.predict(h=h)["mean"]
    assert pred.shape == (h,)
    assert np.all(np.isfinite(np.asarray(pred)))
    assert model._context.shape == (L,)


def test_h_equals_one_and_tiny_input_size():
    """Smallest non-degenerate h=1, input_size=2 configuration."""
    y = jnp.asarray(np.sin(np.arange(50) / 5.0), dtype=jnp.float32)
    model = GRU(h=1, input_size=2, hidden_size=4, n_layers=1,
                max_steps=5, batch_size=2, random_seed=0)
    model.fit(y)
    pred = model.predict(h=1)["mean"]
    assert pred.shape == (1,)
    assert np.isfinite(float(pred[0]))

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


def test_predict_with_level_raises_with_helpful_message():
    """`predict(h, level=...)` raises with a clear message pointing the user
    at the manual conformity_scores + add_confidence_intervals workflow.

    The inherited path is not just slow on GRU — it's broken under vmap
    (see test_conformity_scores_inheritance_smoke_xfail). So the right UX
    is a clear NotImplementedError that documents the manual escape.
    """
    m = _tiny().fit(_make_y())
    with pytest.raises(NotImplementedError) as exc_info:
        m.predict(h=12, level=[80, 95])
    msg = str(exc_info.value)
    assert "conformity_scores" in msg
    assert "add_confidence_intervals" in msg


def test_conformity_scores_raises_with_helpful_message():
    """GRU.conformity_scores overrides the inherited (vmap-broken) path with
    a clear NotImplementedError that documents the constraint and the manual
    escape hatch.
    """
    m = _tiny()
    with pytest.raises(NotImplementedError) as exc_info:
        m.conformity_scores(_make_y())
    msg = str(exc_info.value)
    assert "vmap" in msg.lower() or "incompatible" in msg.lower()
    assert "add_confidence_intervals" in msg


def test_forecast_raises_on_exog():
    m = _tiny()
    with pytest.raises(NotImplementedError, match="Exogenous"):
        m.forecast(_make_y(), h=12, X=jnp.zeros((10, 2)))
    with pytest.raises(NotImplementedError, match="Exogenous"):
        m.forecast(_make_y(), h=12, X_future=jnp.zeros((12, 2)))


def test_forecast_with_fitted_returns_fitted_key():
    """forecast(y, h, fitted=True) returns the 'fitted' key (replaces the
    previous NotImplementedError contract). Detailed shape behavior covered
    in test_forecast_fitted_returns_fitted_values."""
    m = _tiny()
    result = m.forecast(_make_y(), h=12, fitted=True)
    assert "fitted" in result


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


@pytest.mark.xfail(
    reason=(
        "BaseForecaster.conformity_scores uses jax.vmap over windows. GRU's "
        "train() does a host-side float(loss) each step, which raises "
        "ConcretizationTypeError under vmap. The GRU class therefore "
        "overrides conformity_scores to raise NotImplementedError "
        "immediately with a helpful pointer to the manual workflow — see "
        "test_conformity_scores_raises_with_helpful_message. This test "
        "pins the gap: a future refactor that makes train() vmap-safe AND "
        "removes the override should flip this from xfail to xpass."
    ),
    strict=True,
)
def test_conformity_scores_inheritance_smoke_xfail():
    """Documents the known incompatibility between BaseForecaster.conformity_scores
    and GRU's vmap-unsafe training loop. If a future refactor makes train()
    vmap-compatible AND removes the conformity_scores override, this test
    should flip from xfail to xpass."""
    from chronax.utils import ConformalIntervals

    y = jnp.asarray(np.sin(np.arange(120) / 5.0), dtype=jnp.float32)
    model = GRU(h=6, input_size=18, hidden_size=8, n_layers=1,
                max_steps=1, batch_size=2, random_seed=0)
    model.conformal_params = ConformalIntervals(
        n_windows=2, h=6, method="conformal_distribution"
    )
    model.fit(y)
    cs = model.conformity_scores(y)  # expected to raise — xfail
    assert cs.ndim == 2 and np.all(np.isfinite(np.asarray(cs)))


def test_forecast_fitted_returns_fitted_values():
    """forecast(y, h, fitted=True) returns {'mean', 'fitted'}. The 'fitted'
    array has shape (len(y),) with the first input_size entries NaN and the
    rest finite. Matches the convention used by HistoricAverage etc.
    """
    y = jnp.asarray(np.sin(np.arange(120) / 5.0), dtype=jnp.float32)
    model = GRU(h=4, input_size=12, hidden_size=8, n_layers=1, max_steps=10,
                batch_size=4, random_seed=0)
    result = model.forecast(y, h=4, fitted=True)
    assert isinstance(result, dict)
    assert "mean" in result and result["mean"].shape == (4,)
    assert "fitted" in result, f"missing 'fitted' key in {list(result.keys())}"
    fitted = np.asarray(result["fitted"])
    assert fitted.shape == (len(y),), f"got shape {fitted.shape}"
    assert np.all(np.isnan(fitted[:model.input_size]))
    assert np.all(np.isfinite(fitted[model.input_size:]))


def test_forecast_without_fitted_returns_mean_only():
    """forecast(y, h) without fitted=True returns only the 'mean' key."""
    y = jnp.asarray(np.sin(np.arange(120) / 5.0), dtype=jnp.float32)
    model = GRU(h=4, input_size=12, hidden_size=8, n_layers=1, max_steps=5,
                batch_size=4, random_seed=0)
    result = model.forecast(y, h=4)
    assert "mean" in result
    assert "fitted" not in result

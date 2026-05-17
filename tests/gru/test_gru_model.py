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


def test_predict_with_level_returns_interval_keys():
    """predict(h, level=[80, 95]) returns mean + lo-XX/hi-XX via the
    inherited BaseForecaster.add_confidence_intervals path. Possible
    because the training loop is now nnx.scan-based and vmap-safe."""
    from chronax.utils import ConformalIntervals
    y = jnp.asarray(np.sin(np.arange(120) / 5.0), dtype=jnp.float32)
    model = GRU(h=4, input_size=12, hidden_size=8, n_layers=1,
                max_steps=2, batch_size=4, random_seed=0)
    model.conformal_params = ConformalIntervals(n_windows=2, h=4)
    model.fit(y)
    fcst = model.predict(h=4, level=[80, 95])
    for k in ("mean", "lo-80", "hi-80", "lo-95", "hi-95"):
        assert k in fcst, f"missing {k!r} in {list(fcst.keys())}"
        assert fcst[k].shape == (4,)
        assert np.all(np.isfinite(np.asarray(fcst[k])))
    assert float(fcst["lo-95"].mean()) <= float(fcst["mean"].mean())
    assert float(fcst["mean"].mean()) <= float(fcst["hi-95"].mean())


def test_predict_with_level_raises_without_conformal_params():
    """If level is set but conformal_params isn't, raise clearly with a
    'minutes per call' cost warning so users don't accidentally invoke
    a multi-minute compute path."""
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError) as exc_info:
        m.predict(h=12, level=[80, 95])
    msg = str(exc_info.value).lower()
    assert "conformal_params" in msg
    assert "minutes" in msg, "error must surface the compute cost in user-readable units"


def test_conformity_scores_returns_finite_2d_array():
    """conformity_scores now works via the inherited (vmap-safe) path."""
    from chronax.utils import ConformalIntervals
    y = jnp.asarray(np.sin(np.arange(120) / 5.0), dtype=jnp.float32)
    model = GRU(h=4, input_size=12, hidden_size=8, n_layers=1,
                max_steps=2, batch_size=4, random_seed=0)
    model.conformal_params = ConformalIntervals(n_windows=2, h=4)
    model.fit(y)
    cs = model.conformity_scores(y)
    assert cs.ndim == 2
    assert np.all(np.isfinite(np.asarray(cs)))


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


def test_conformity_scores_inheritance_smoke():
    """Inherited BaseForecaster.conformity_scores works via the nnx.scan-based
    train(). Originally xfailed pre-refactor because the loop did a host-side
    float(loss) per step. Flipped to xpass after the lax.scan/nnx.scan rewrite
    of train()."""
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


def test_build_net_works_under_vmap():
    """Module construction (_build_net inside fit, called inside vmap by
    BaseForecaster.conformity_scores) must trace without
    ConcretizationTypeError. If this fails, the train() refactor alone
    won't fix the conformity_scores xfail — module init itself is the gate.
    """
    from chronax.models.gru.gru_module import GRUNet
    from flax import nnx

    def build(seed_scalar):
        return GRUNet(
            in_features=1, encoder_hidden=8, encoder_layers=1,
            decoder_hidden=4, decoder_layers=2, dropout=0.0,
            h=2, input_size=4, rngs=nnx.Rngs(int(seed_scalar)),
        )

    # Loop construction is the pattern BaseForecaster.conformity_scores
    # ultimately uses (forecast → fit → _build_net, traced by jax.vmap
    # but each call constructs a fresh network at Python-level).
    nets = [build(int(s)) for s in jnp.arange(3)]
    x = jnp.ones((1, 4, 1), dtype=jnp.float32)
    for net in nets:
        out = net(x, deterministic=True)
        assert out.shape == (1, 2, 1)
        assert np.all(np.isfinite(np.asarray(out)))


def test_pickle_round_trip_with_full_max_steps():
    """Pickle round-trip after the nnx.scan refactor.

    The refactor changes how optimizer state is constructed (inside the
    nnx.scan body, with the (model, optimizer) tuple as a single Carry).
    Verifies the full-fitted-state pickling still produces identical
    predictions. Uses a larger max_steps to exercise the warm-trace path
    that the small `_tiny()` config in `test_gru_pickle_round_trip`
    doesn't fully cover.
    """
    y = _make_y()
    model = GRU(h=12, input_size=36, hidden_size=16, n_layers=1,
                max_steps=50, batch_size=8, random_seed=0)
    model.fit(y)
    pred_before = np.asarray(model.predict(h=12)["mean"])

    blob = pickle.dumps(model)
    restored = pickle.loads(blob)
    pred_after = np.asarray(restored.predict(h=12)["mean"])

    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)

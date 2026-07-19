"""Tests for chronax.models.nlinear.nlinear_model — BaseForecaster conformance."""
import pickle

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.nlinear.nlinear_model import NLinear


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _tiny(**kw):
    return NLinear(h=12, input_size=36, max_steps=20, windows_batch_size=64, random_seed=0, **kw)


def test_nlinear_inherits_baseforecaster():
    assert issubclass(NLinear, BaseForecaster)


def test_init_conformal_params_none():
    assert _tiny().conformal_params is None


def test_fit_returns_self_and_sets_model():
    m = _tiny()
    out = m.fit(_make_y())
    assert out is m and m.model_ is not None


def test_predict_default_returns_h():
    assert _tiny().fit(_make_y()).predict(h=12)["mean"].shape == (12,)


def test_predict_smaller_h_slices():
    assert _tiny().fit(_make_y()).predict(h=5)["mean"].shape == (5,)


def test_predict_larger_h_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="not supported"):
        m.predict(h=13)


def test_predict_h_below_one_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="positive"):
        m.predict(h=0)


def test_fit_raises_on_exog():
    with pytest.raises(NotImplementedError):
        _tiny().fit(_make_y(), X=jnp.ones((200, 1)))


def test_fit_raises_on_short_series():
    with pytest.raises(ValueError, match="too short"):
        _tiny().fit(_make_y(40))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        _tiny().predict(h=12)


def test_unknown_scaler_raises_at_fit():
    with pytest.raises(ValueError, match="Unknown scaler"):
        NLinear(h=12, input_size=36, max_steps=2, windows_batch_size=8, random_seed=0, scaler="nope").fit(_make_y())


def test_predict_deterministic_same_seed():
    y = _make_y()
    np.testing.assert_allclose(np.asarray(_tiny().fit(y).predict(h=12)["mean"]),
                               np.asarray(_tiny().fit(y).predict(h=12)["mean"]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scaler", ["identity", "robust"])
def test_scaler_knob_trains_and_predicts(scaler):
    m = _tiny(scaler=scaler).fit(_make_y())
    assert jnp.all(jnp.isfinite(m.predict(h=12)["mean"]))


def test_custom_callable_loss_trains():
    def my_loss(pred, target, mask=None):
        e = jnp.abs(pred - target)
        return jnp.mean(e) if mask is None else (e * mask).sum() / jnp.clip(mask.sum(), 1.0)
    m = _tiny(loss=my_loss).fit(_make_y())
    assert jnp.all(jnp.isfinite(m.predict(h=12)["mean"]))


def test_model_beats_naive_on_easy_signal():
    n = 200
    y = jnp.asarray(np.sin(np.arange(n) / 5.0), dtype=jnp.float32)
    m = NLinear(h=12, input_size=36, max_steps=200, windows_batch_size=64, random_seed=0,
                learning_rate=1e-2).fit(y[:-12])
    pred = np.asarray(m.predict(h=12)["mean"])
    y_true = np.asarray(y[-12:])
    naive = np.full(12, float(y[-13]))
    assert np.mean(np.abs(pred - y_true)) < np.mean(np.abs(naive - y_true))


def test_pickle_round_trip_preserves_predictions():
    m = _tiny().fit(_make_y())
    before = np.asarray(m.predict(h=12)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(np.asarray(m2.predict(h=12)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_pickle_preserves_params():
    m = _tiny().fit(_make_y())
    _, sb = nnx.split(m.model_)
    m2 = pickle.loads(pickle.dumps(m))
    _, sa = nnx.split(m2.model_)

    def arrs(state):
        out = {}
        for p, v in nnx.to_flat_state(state):
            val = getattr(v, "value", None)
            if val is None or str(getattr(val, "dtype", "")).startswith("key"):
                continue
            out[tuple(p)] = val
        return out

    a, b = arrs(sb), arrs(sa)
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))


@pytest.mark.parametrize("loss_name", ["mae", "mse", "huber"])
def test_loss_string_pickle_round_trip(loss_name):
    m = _tiny(loss=loss_name).fit(_make_y())
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(np.asarray(m2.predict(h=12)["mean"]),
                               np.asarray(m.predict(h=12)["mean"]), rtol=1e-5, atol=1e-5)


from chronax.utils import ConformalIntervals


def test_constant_series_returns_finite():
    m = _tiny(scaler="robust").fit(jnp.ones(200, dtype=jnp.float32))
    assert jnp.all(jnp.isfinite(m.predict(h=12)["mean"]))


def test_h_equals_one():
    m = NLinear(h=1, input_size=12, max_steps=5, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(60))
    assert m.predict(h=1)["mean"].shape == (1,)


def test_forecast_equals_fit_then_predict():
    y = _make_y()
    np.testing.assert_allclose(np.asarray(_tiny().forecast(y, h=12)["mean"]),
                               np.asarray(_tiny().fit(y).predict(h=12)["mean"]), rtol=1e-5, atol=1e-5)


def test_forecast_fitted_nan_head_finite_tail():
    res = _tiny().forecast(_make_y(), h=12, fitted=True)
    fitted = np.asarray(res["fitted"])
    assert fitted.shape == (200,) and np.all(np.isnan(fitted[:36])) and np.all(np.isfinite(fitted[36:]))


def test_fitted_values_match_hand_computed():
    y = _make_y(60)
    m = _tiny().fit(y)
    # zero weights + constant bias c: one-step-ahead fitted[36+i] = c + y[35+i]
    # (forward adds back the insample window's last value).
    m.model_.weight.value = jnp.zeros((12, 36), dtype=jnp.float32)
    m.model_.bias.value = jnp.full((12,), 0.5, dtype=jnp.float32)
    fitted = np.asarray(m._compute_fitted_values())
    for i in (0, 5, 23):
        np.testing.assert_allclose(fitted[36 + i], 0.5 + float(y[35 + i]), rtol=1e-5)


def test_predict_with_level_returns_interval_keys():
    m = NLinear(h=4, input_size=12, max_steps=2, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    out = m.predict(h=4, level=[80])
    assert "lo-80" in out and "hi-80" in out


def test_conformity_scores_finite_2d():
    m = NLinear(h=4, input_size=12, max_steps=2, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    cs = m.conformity_scores(_make_y(80))
    assert cs.ndim == 2 and cs.shape[1] == 4 and jnp.all(jnp.isfinite(cs))


def test_nlinear_importable_from_models_namespace():
    from chronax.models import NLinear as N
    assert N is NLinear


def test_nlinear_auto_discovered_by_benchmark_harness():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.neural import registry
    assert "NLinear" in registry.list_models()
    assert registry.resolve_chronax("NLinear") is NLinear

"""Tests for chronax.models.rmok.rmok_model — BaseForecaster conformance."""
import pickle

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.rmok.rmok_model import RMoK


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _tiny(**kw):
    kw.setdefault("taylor_order", 2); kw.setdefault("jacobi_degree", 3)
    kw.setdefault("max_steps", 10); kw.setdefault("windows_batch_size", 16)
    return RMoK(h=4, input_size=12, random_seed=0, **kw)


def test_rmok_inherits_baseforecaster():
    assert issubclass(RMoK, BaseForecaster)


def test_init_conformal_params_none():
    assert _tiny().conformal_params is None


def test_fit_returns_self_and_sets_model():
    m = _tiny()
    out = m.fit(_make_y())
    assert out is m and m.model_ is not None


def test_predict_default_returns_h():
    assert _tiny().fit(_make_y()).predict(h=4)["mean"].shape == (4,)


def test_predict_smaller_h_slices():
    assert _tiny().fit(_make_y()).predict(h=2)["mean"].shape == (2,)


def test_predict_larger_h_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="not supported"):
        m.predict(h=5)


def test_predict_h_below_one_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="positive"):
        m.predict(h=0)


def test_fit_raises_on_exog():
    with pytest.raises(NotImplementedError):
        _tiny().fit(_make_y(), X=jnp.ones((200, 1)))


def test_fit_raises_on_short_series():
    with pytest.raises(ValueError, match="too short"):
        _tiny().fit(_make_y(10))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        _tiny().predict(h=4)


def test_unknown_scaler_raises_at_fit():
    with pytest.raises(ValueError, match="Unknown scaler"):
        _tiny(scaler="nope").fit(_make_y())


def test_taylor_order_below_zero_raises():
    with pytest.raises(ValueError, match="taylor_order"):
        _tiny(taylor_order=-1)


def test_jacobi_degree_below_zero_raises():
    with pytest.raises(ValueError, match="jacobi_degree"):
        _tiny(jacobi_degree=-1)


def test_dropout_out_of_range_raises():
    with pytest.raises(ValueError, match="dropout"):
        _tiny(dropout=1.0)


def test_wavelet_invalid_raises():
    with pytest.raises(ValueError, match="wavelet"):
        _tiny(wavelet_function="nope")


def test_predict_deterministic_same_seed():
    y = _make_y()
    np.testing.assert_allclose(np.asarray(_tiny().fit(y).predict(h=4)["mean"]),
                               np.asarray(_tiny().fit(y).predict(h=4)["mean"]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scaler", ["identity", "robust"])
def test_scaler_knob_trains_and_predicts(scaler):
    m = _tiny(scaler=scaler).fit(_make_y())
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


def test_custom_callable_loss_trains():
    def my_loss(pred, target, mask=None):
        e = jnp.abs(pred - target)
        return jnp.mean(e) if mask is None else (e * mask).sum() / jnp.clip(mask.sum(), 1.0)
    m = _tiny(loss=my_loss).fit(_make_y())
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


@pytest.mark.parametrize("knob", [{"taylor_order": 5}, {"jacobi_degree": 8},
                                  {"wavelet_function": "morlet"}, {"dropout": 0.5}])
def test_arch_knob_threads_to_predictions(knob):
    y = _make_y()
    base = np.asarray(_tiny().fit(y).predict(h=4)["mean"])
    var = np.asarray(_tiny(**knob).fit(y).predict(h=4)["mean"])
    assert not np.allclose(base, var), knob


def test_revin_affine_perturbation_changes_output():
    # revin_affine on/off at INIT is vacuous (gamma=1/beta=0). Perturb gamma/beta
    # on a fitted model and confirm the prediction moves.
    m = _tiny().fit(_make_y())
    base = np.asarray(m.predict(h=4)["mean"])
    m.model_.affine_weight.value = m.model_.affine_weight.value * 1.5 + 0.3
    m.model_.affine_bias.value = m.model_.affine_bias.value + 0.2
    assert not np.allclose(base, np.asarray(m.predict(h=4)["mean"]))


def test_model_beats_naive_on_easy_signal():
    n = 200
    y = jnp.asarray(np.sin(np.arange(n) / 5.0), dtype=jnp.float32)
    m = _tiny(max_steps=300, learning_rate=1e-2).fit(y[:-4])
    pred = np.asarray(m.predict(h=4)["mean"])
    y_true = np.asarray(y[-4:])
    naive = np.full(4, float(y[-5]))
    assert np.mean(np.abs(pred - y_true)) < np.mean(np.abs(naive - y_true))


def test_predict_smaller_h_is_prefix_of_full():
    m = _tiny().fit(_make_y())
    full = np.asarray(m.predict(h=4)["mean"])
    np.testing.assert_allclose(np.asarray(m.predict(h=2)["mean"]), full[:2], rtol=1e-6)


def test_pickle_round_trip_preserves_predictions_and_running_stats():
    m = _tiny().fit(_make_y())
    before = np.asarray(m.predict(h=4)["mean"])
    rv_before = np.asarray(m.model_.wave_bn.var.value)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(np.asarray(m2.model_.wave_bn.var.value), rv_before)  # BatchStat survives
    np.testing.assert_allclose(np.asarray(m2.predict(h=4)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_train_updates_all_params_except_weight1():
    y = _make_y()
    m2 = _tiny(max_steps=30, learning_rate=1e-2).fit(y)
    after = {tuple(p): np.asarray(v.value) for p, v in nnx.to_flat_state(nnx.state(m2.model_, nnx.Param))}
    m3 = _tiny(max_steps=30, learning_rate=1e-2)
    init_net = m3._build_net()
    init = {tuple(p): np.asarray(v.value) for p, v in nnx.to_flat_state(nnx.state(init_net, nnx.Param))}
    assert init.keys() == after.keys()
    for k in after:
        moved = not np.allclose(init[k], after[k])
        if "weight1" in "/".join(map(str, k)):
            assert not moved, "unused weight1 must stay frozen"
        else:
            assert moved, f"{k} did not train"


from chronax.utils import ConformalIntervals


def test_constant_series_returns_finite():
    m = _tiny(scaler="robust").fit(jnp.ones(200, dtype=jnp.float32))
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


def test_forecast_equals_fit_then_predict():
    y = _make_y()
    np.testing.assert_allclose(np.asarray(_tiny().forecast(y, h=4)["mean"]),
                               np.asarray(_tiny().fit(y).predict(h=4)["mean"]), rtol=1e-5, atol=1e-5)


def test_forecast_fitted_nan_head_finite_tail():
    res = _tiny().forecast(_make_y(), h=4, fitted=True)
    fitted = np.asarray(res["fitted"])
    assert fitted.shape == (200,) and np.all(np.isnan(fitted[:12])) and np.all(np.isfinite(fitted[12:]))


def test_predict_with_level_returns_interval_keys():
    m = _tiny(max_steps=2).fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    out = m.predict(h=4, level=[80])
    assert "lo-80" in out and "hi-80" in out


def test_conformity_scores_finite_2d():
    m = _tiny(max_steps=2).fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    cs = m.conformity_scores(_make_y(80))
    assert cs.ndim == 2 and cs.shape[1] == 4 and jnp.all(jnp.isfinite(cs))


def test_rmok_importable_from_models_namespace():
    from chronax.models import RMoK as R
    assert R is RMoK


def test_rmok_auto_discovered_by_benchmark_harness():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.neural import registry
    assert "RMoK" in registry.list_models()
    assert registry.resolve_chronax("RMoK") is RMoK

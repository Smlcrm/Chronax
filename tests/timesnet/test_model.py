"""Tests for chronax.models.timesnet.timesnet_model — BaseForecaster conformance."""
import pickle

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.timesnet.timesnet_model import TimesNet


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _tiny(**kw):
    kw.setdefault("hidden_size", 8); kw.setdefault("conv_hidden_size", 8)
    kw.setdefault("top_k", 2); kw.setdefault("num_kernels", 2)
    kw.setdefault("encoder_layers", 1)
    kw.setdefault("max_steps", 10); kw.setdefault("windows_batch_size", 16)
    return TimesNet(h=4, input_size=12, random_seed=0, **kw)


def test_timesnet_inherits_baseforecaster():
    assert issubclass(TimesNet, BaseForecaster)


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


def test_top_k_below_one_raises():
    with pytest.raises(ValueError, match="top_k"):
        _tiny(top_k=0)


def test_top_k_too_large_raises():
    # (input_size + h) // 2 = (12 + 4) // 2 = 8 nonzero rfft bins
    with pytest.raises(ValueError, match="top_k"):
        _tiny(top_k=9)


def test_num_kernels_below_one_raises():
    with pytest.raises(ValueError, match="num_kernels"):
        _tiny(num_kernels=0)


def test_encoder_layers_below_one_raises():
    with pytest.raises(ValueError, match="encoder_layers"):
        _tiny(encoder_layers=0)


def test_predict_deterministic_same_seed():
    y = _make_y()
    np.testing.assert_allclose(np.asarray(_tiny().fit(y).predict(h=4)["mean"]),
                               np.asarray(_tiny().fit(y).predict(h=4)["mean"]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scaler", ["identity", "robust", "standard"])
def test_scaler_knob_trains_and_predicts(scaler):
    m = _tiny(scaler=scaler).fit(_make_y())
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


def test_custom_callable_loss_trains():
    def my_loss(pred, target, mask=None):
        e = jnp.abs(pred - target)
        return jnp.mean(e) if mask is None else (e * mask).sum() / jnp.clip(mask.sum(), 1.0)
    m = _tiny(loss=my_loss).fit(_make_y())
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


@pytest.mark.parametrize("knob", [{"hidden_size": 4}, {"conv_hidden_size": 4}, {"top_k": 1},
                                  {"num_kernels": 1}, {"encoder_layers": 2}, {"dropout": 0.5}])
def test_arch_knob_threads_to_predictions(knob):
    y = _make_y()
    base = np.asarray(_tiny().fit(y).predict(h=4)["mean"])
    var = np.asarray(_tiny(**knob).fit(y).predict(h=4)["mean"])
    assert not np.allclose(base, var), knob


def test_periods_set_after_fit_and_data_dependent():
    t = np.arange(120, dtype=np.float32)
    fast = jnp.asarray(np.sin(2 * np.pi * t / 8.0), dtype=jnp.float32)
    slow = jnp.asarray(np.sin(2 * np.pi * t / 25.0), dtype=jnp.float32)
    m1 = _tiny(top_k=1).fit(fast)
    m2 = _tiny(top_k=1).fit(slow)
    T = 12 + 4
    for m in (m1, m2):
        assert m.periods_ is not None and m.freqs_ is not None
        assert len(m.periods_) == 1 and len(m.freqs_) == 1
        assert m.periods_[0] == T // m.freqs_[0] and 1 <= m.freqs_[0] <= T // 2
    # period-8 tone -> freq bin 2 of a length-16 window; period-25 tone -> bin 1
    assert m1.periods_ != m2.periods_


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


def test_predict_conditions_on_series_end():
    y = jnp.asarray(np.linspace(0.0, 100.0, 120), dtype=jnp.float32)
    m = _tiny().fit(y)
    pred = np.asarray(m.predict(h=4)["mean"])
    assert np.all(pred > 50.0), pred     # standard scaler anchors near end-window level


def test_pickle_round_trip_preserves_predictions_and_periods():
    m = _tiny().fit(_make_y())
    before = np.asarray(m.predict(h=4)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    assert m2.periods_ == m.periods_ and m2.freqs_ == m.freqs_
    np.testing.assert_allclose(np.asarray(m2.predict(h=4)["mean"]), before, rtol=1e-5, atol=1e-5)


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
    np.testing.assert_allclose(np.asarray(m2.predict(h=4)["mean"]),
                               np.asarray(m.predict(h=4)["mean"]), rtol=1e-5, atol=1e-5)


def test_train_updates_all_params():
    y = _make_y()
    m2 = _tiny(max_steps=30, learning_rate=1e-2).fit(y)
    _, s1 = nnx.split(m2.model_)
    after = {tuple(p): np.asarray(v.value) for p, v in nnx.to_flat_state(s1)
             if not str(getattr(v.value, "dtype", "")).startswith("key")}
    # compare a fresh init (same seed) against the trained net: every leaf must move
    m3 = _tiny(max_steps=30, learning_rate=1e-2)
    m3.periods_, m3.freqs_ = m2.periods_, m2.freqs_
    init_net = m3._build_net()
    _, si = nnx.split(init_net)
    init = {tuple(p): np.asarray(v.value) for p, v in nnx.to_flat_state(si)
            if not str(getattr(v.value, "dtype", "")).startswith("key")}
    assert init.keys() == after.keys()
    for k in after:
        assert not np.allclose(init[k], after[k]), f"{k} did not train"


from chronax.utils import ConformalIntervals


def test_constant_series_returns_finite():
    m = _tiny().fit(jnp.ones(200, dtype=jnp.float32))
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


def test_h_equals_one():
    m = TimesNet(h=1, input_size=12, hidden_size=8, conv_hidden_size=8, top_k=2,
                 num_kernels=2, encoder_layers=1, max_steps=5, windows_batch_size=16,
                 random_seed=0)
    m.fit(_make_y(60))
    assert m.predict(h=1)["mean"].shape == (1,)


def test_forecast_equals_fit_then_predict():
    y = _make_y()
    np.testing.assert_allclose(np.asarray(_tiny().forecast(y, h=4)["mean"]),
                               np.asarray(_tiny().fit(y).predict(h=4)["mean"]), rtol=1e-5, atol=1e-5)


def test_forecast_fitted_nan_head_finite_tail():
    res = _tiny().forecast(_make_y(), h=4, fitted=True)
    fitted = np.asarray(res["fitted"])
    assert fitted.shape == (200,) and np.all(np.isnan(fitted[:12])) and np.all(np.isfinite(fitted[12:]))


def test_fitted_values_match_hand_computed():
    y = _make_y(60)
    m = _tiny().fit(y)
    # Zero every weight, keep biases zero except projection bias c=0.5. Then
    # emb+PE is killed by the zero predict_linear, blocks pass 0 through
    # (zero convs, residual 0, LN(0)=0), so pred_z == c everywhere and
    # fitted[12+i] = c*scale_i + shift_i (standard scaler stats of window i).
    net = m.model_
    net.w_token.value = jnp.zeros_like(net.w_token.value)
    net.w_predict.value = jnp.zeros_like(net.w_predict.value)
    net.b_predict.value = jnp.zeros_like(net.b_predict.value)
    for layer_ws, layer_bs in zip(net.conv_ws, net.conv_bs):
        for ws, bs in zip(layer_ws, layer_bs):
            for w, b in zip(ws, bs):
                w.value = jnp.zeros_like(w.value)
                b.value = jnp.zeros_like(b.value)
    net.w_proj.value = jnp.zeros_like(net.w_proj.value)
    net.b_proj.value = jnp.full((1,), 0.5, dtype=jnp.float32)
    fitted = np.asarray(m._compute_fitted_values())
    yn = np.asarray(y)
    for i in (0, 5, 23):
        win = yn[i:i + 12]
        expected = 0.5 * (win.std() + 1e-6) + win.mean()
        np.testing.assert_allclose(fitted[12 + i], expected, rtol=1e-5)


def test_predict_with_level_returns_interval_keys():
    m = _tiny(max_steps=2).fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    out = m.predict(h=4, level=[80])
    assert "lo-80" in out and "hi-80" in out


def test_conformity_scores_finite_2d_and_reuses_parent_periods():
    m = _tiny(max_steps=2).fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    periods_before = m.periods_
    cs = m.conformity_scores(_make_y(80))
    assert cs.ndim == 2 and cs.shape[1] == 4 and jnp.all(jnp.isfinite(cs))
    assert m.periods_ == periods_before      # traced re-fits reuse, never clobber


def test_timesnet_importable_from_models_namespace():
    from chronax.models import TimesNet as T
    assert T is TimesNet


def test_timesnet_auto_discovered_by_benchmark_harness():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.neural import registry
    assert "TimesNet" in registry.list_models()
    assert registry.resolve_chronax("TimesNet") is TimesNet

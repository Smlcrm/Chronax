"""Tests for chronax.models.patchtst.patchtst_model — BaseForecaster conformance."""
import pickle

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.patchtst.patchtst_model import PatchTST


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _tiny():
    return PatchTST(h=12, input_size=36, hidden_size=16, n_heads=2, encoder_layers=1,
                    linear_hidden_size=32, max_steps=20, windows_batch_size=64, random_seed=0)


def test_patchtst_inherits_baseforecaster():
    assert issubclass(PatchTST, BaseForecaster)


def test_init_sets_conformal_params_to_none():
    assert _tiny().conformal_params is None


def test_fit_returns_self_and_sets_model():
    m = _tiny()
    out = m.fit(_make_y())
    assert out is m
    assert m.model_ is not None


def test_predict_deterministic_with_same_seed():
    y = _make_y()
    p1 = _tiny().fit(y).predict(h=12)["mean"]
    p2 = _tiny().fit(y).predict(h=12)["mean"]
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-5, atol=1e-5)


def test_model_beats_naive_last_value_on_easy_signal():
    # End-to-end correctness floor: catches broken loss/optimizer/RevIN-denorm/
    # flatten-order wiring that still passes shape + determinism tests.
    n = 200
    y = jnp.asarray(np.sin(np.arange(n) / 5.0), dtype=jnp.float32)
    m = PatchTST(h=12, input_size=36, hidden_size=32, n_heads=4, encoder_layers=2,
                 linear_hidden_size=64, max_steps=200, windows_batch_size=64, random_seed=0)
    m.fit(y[:-12])
    pred = np.asarray(m.predict(h=12)["mean"])
    y_true = np.asarray(y[-12:])
    naive = np.full(12, float(y[-13]))
    assert np.mean(np.abs(pred - y_true)) < np.mean(np.abs(naive - y_true))


def test_predict_default_returns_h_steps():
    m = _tiny().fit(_make_y())
    assert m.predict(h=12)["mean"].shape == (12,)


def test_predict_smaller_h_slices():
    m = _tiny().fit(_make_y())
    assert m.predict(h=5)["mean"].shape == (5,)


def test_predict_larger_h_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="not supported"):
        m.predict(h=13)


def test_fit_raises_on_exog():
    with pytest.raises(NotImplementedError):
        _tiny().fit(_make_y(), X=jnp.ones((200, 1)))


def test_fit_raises_on_short_series():
    with pytest.raises(ValueError, match="too short"):
        _tiny().fit(_make_y(40))  # < input_size + h = 48


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        _tiny().predict(h=12)


def test_hidden_not_divisible_by_heads_raises_at_fit():
    m = PatchTST(h=12, input_size=36, hidden_size=30, n_heads=4, encoder_layers=1,
                 max_steps=2, windows_batch_size=8, random_seed=0)
    with pytest.raises(ValueError, match="divisible"):
        m.fit(_make_y())


def test_pickle_round_trip_preserves_predictions():
    m = _tiny().fit(_make_y())
    before = np.asarray(m.predict(h=12)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    after = np.asarray(m2.predict(h=12)["mean"])
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-5)


def test_pickle_round_trip_preserves_batchnorm_running_stats():
    m = _tiny().fit(_make_y())
    _, state_before = nnx.split(m.model_)
    m2 = pickle.loads(pickle.dumps(m))
    _, state_after = nnx.split(m2.model_)
    flat_b = nnx.to_flat_state(state_before)
    flat_a = nnx.to_flat_state(state_after)

    def _arrays(flat):
        # Skip Rngs PRNGKey variables — np.asarray on a key<fry> dtype raises.
        out = {}
        for p, v in flat:
            val = getattr(v, "value", None)
            if val is None or str(getattr(val, "dtype", "")).startswith("key"):
                continue
            out[tuple(p)] = val
        return out

    bn_before, bn_after = _arrays(flat_b), _arrays(flat_a)
    assert bn_before.keys() == bn_after.keys()
    for k in bn_before:
        np.testing.assert_array_equal(np.asarray(bn_before[k]), np.asarray(bn_after[k]))


@pytest.mark.parametrize("loss_name", ["mae", "mse", "huber"])
def test_loss_string_pickle_round_trip(loss_name):
    m = PatchTST(h=12, input_size=36, hidden_size=16, n_heads=2, encoder_layers=1,
                 linear_hidden_size=32, max_steps=10, windows_batch_size=64,
                 random_seed=0, loss=loss_name).fit(_make_y())
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(
        np.asarray(m2.predict(h=12)["mean"]), np.asarray(m.predict(h=12)["mean"]),
        rtol=1e-5, atol=1e-5,
    )


def test_patchtst_importable_from_models_namespace():
    from chronax.models import PatchTST as P
    assert P is PatchTST


from chronax.utils import ConformalIntervals


def test_constant_series_returns_finite():
    m = _tiny().fit(jnp.ones(200, dtype=jnp.float32))
    assert jnp.all(jnp.isfinite(m.predict(h=12)["mean"]))


def test_h_equals_one():
    m = PatchTST(h=1, input_size=12, hidden_size=8, n_heads=2, encoder_layers=1,
                 linear_hidden_size=16, max_steps=5, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(60))
    assert m.predict(h=1)["mean"].shape == (1,)


def test_forecast_equals_fit_then_predict_same_seed():
    y = _make_y()
    a = _tiny().forecast(y, h=12)["mean"]
    b = _tiny().fit(y).predict(h=12)["mean"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


def test_forecast_fitted_has_nan_head_and_finite_tail():
    res = _tiny().forecast(_make_y(), h=12, fitted=True)
    fitted = np.asarray(res["fitted"])
    assert fitted.shape == (200,)
    assert np.all(np.isnan(fitted[:36]))
    assert np.all(np.isfinite(fitted[36:]))


def test_build_net_works_under_vmap_loop_construction():
    # precursor to conformity_scores' vmap: constructing the net in a loop
    nets = [_tiny()._build_net() for _ in range(3)]
    assert len(nets) == 3


def test_predict_with_level_returns_interval_keys():
    m = PatchTST(h=4, input_size=12, hidden_size=8, n_heads=2, encoder_layers=1,
                 linear_hidden_size=16, max_steps=2, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    out = m.predict(h=4, level=[80])
    assert "lo-80" in out and "hi-80" in out


def test_predict_with_level_raises_without_conformal_params():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="conformal_params"):
        m.predict(h=12, level=[80])


def test_conformity_scores_returns_finite_2d_array():
    m = PatchTST(h=4, input_size=12, hidden_size=8, n_heads=2, encoder_layers=1,
                 linear_hidden_size=16, max_steps=2, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    cs = m.conformity_scores(_make_y(80))
    assert cs.ndim == 2 and cs.shape[1] == 4
    assert jnp.all(jnp.isfinite(cs))


def test_patch_len_clamp_fires_on_short_input():
    # input_size + stride < patch_len forces the clamp; forward must still work.
    m = PatchTST(h=2, input_size=4, patch_len=16, stride=2, hidden_size=8, n_heads=2,
                 encoder_layers=1, linear_hidden_size=16, max_steps=3,
                 windows_batch_size=8, random_seed=0)
    m.fit(_make_y(40))
    out = m.predict(h=2)["mean"]
    assert out.shape == (2,) and jnp.all(jnp.isfinite(out))

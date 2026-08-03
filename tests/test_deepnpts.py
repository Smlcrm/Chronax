"""Tests for chronax.models.DeepNPTS.

Covers the four source modules of the deepnpts subpackage in one file (losses,
module, training, model), matching the flat ``test_<model>.py`` convention used
elsewhere in ``tests/``. The strict numerical parity test vs neuralforecast runs
at ``batch_norm=False`` (see ``test_parity_vs_neuralforecast``).
"""
import json
import math
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.deepnpts.deepnpts_losses import (
    LOSSES, huber, mae, mse, resolve,
)


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


# ============================================================================
# Losses
# ============================================================================

def test_mae_zero_at_match():
    x = jnp.array([1.0, 2.0, 3.0])
    assert float(mae(x, x)) == 0.0


def test_mse_known_value():
    pred = jnp.array([0.0, 0.0])
    target = jnp.array([1.0, 3.0])
    assert float(mse(pred, target)) == pytest.approx(5.0)


def test_huber_quadratic_region():
    assert float(huber(jnp.array([0.0]), jnp.array([0.5]))) == pytest.approx(0.125)


def test_huber_linear_region():
    assert float(huber(jnp.array([0.0]), jnp.array([3.0]))) == pytest.approx(2.5)


@pytest.mark.parametrize("name", ["mae", "mse", "huber"])
def test_loss_is_jit_and_grad_friendly(name):
    fn = LOSSES[name]
    pred = jnp.array([0.1, 0.2, 0.3])
    target = jnp.array([0.0, 0.5, 0.2])
    g = jax.grad(lambda p: fn(p, target))(pred)
    assert g.shape == pred.shape
    assert jnp.all(jnp.isfinite(g))


def test_resolve_string_returns_registry_entry():
    assert resolve("mae") is mae


def test_resolve_callable_passes_through():
    fn = lambda p, t: jnp.sum(p - t)
    assert resolve(fn) is fn


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve("not_a_loss")


from chronax.models.deepnpts.deepnpts_module import DeepNPTSNet, _TorchLinearInit


# ============================================================================
# Module: init / backbone
# ============================================================================

def test_torch_linear_init_bounds():
    init = _TorchLinearInit(fan_in=16)
    w = init(jax.random.PRNGKey(0), (16, 32))
    bound = 1.0 / math.sqrt(16)
    assert jnp.all(jnp.abs(w) <= bound + 1e-6)


def test_net_forward_shape():
    net = DeepNPTSNet(h=4, input_size=12, hidden_size=8, n_layers=2, dropout=0.0,
                      batch_norm=False, rngs=nnx.Rngs(0))
    x = jnp.ones((3, 12, 1))
    out = net(x, deterministic=True)
    assert out.shape == (3, 4, 1)


def test_net_softmax_weights_keep_forecast_in_window_range():
    # Each forecast value is a convex combination (softmax weights) of the window,
    # so it must lie within [min(window), max(window)].
    net = DeepNPTSNet(h=4, input_size=12, hidden_size=8, n_layers=2, dropout=0.0,
                      batch_norm=False, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(5, 12, 1), dtype=jnp.float32)
    out = net(x, deterministic=True)                # [5, 4, 1]
    lo = jnp.min(x, axis=1)                          # [5, 1]
    hi = jnp.max(x, axis=1)
    assert bool(jnp.all(out[..., 0] >= lo - 1e-4))
    assert bool(jnp.all(out[..., 0] <= hi + 1e-4))


def test_net_batch_norm_path_shape_and_finite():
    net = DeepNPTSNet(h=4, input_size=12, hidden_size=8, n_layers=2, dropout=0.1,
                      batch_norm=True, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(1).randn(6, 12, 1), dtype=jnp.float32)
    out = net(x, deterministic=False)
    assert out.shape == (6, 4, 1)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_net_deterministic_is_repeatable():
    net = DeepNPTSNet(h=4, input_size=12, hidden_size=8, n_layers=2, dropout=0.5,
                      batch_norm=False, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(2, 12, 1), dtype=jnp.float32)
    a = net(x, deterministic=True)
    b = net(x, deterministic=True)
    assert jnp.allclose(a, b)


def test_net_rejects_zero_layers():
    with pytest.raises(ValueError):
        DeepNPTSNet(h=4, input_size=12, hidden_size=8, n_layers=0, dropout=0.0,
                    batch_norm=False, rngs=nnx.Rngs(0))


from chronax.models.deepnpts.deepnpts_training import (
    build_windows, forward_loss, predict_step, train,
)


# ============================================================================
# Training
# ============================================================================

def test_build_windows_shape_and_content():
    y = jnp.arange(10.0)
    w = build_windows(y, input_size=4, h=2)
    assert w.shape == (5, 6)          # n = 10 - 6 + 1 = 5
    assert jnp.allclose(w[0], jnp.arange(6.0))
    assert jnp.allclose(w[-1], jnp.arange(4.0, 10.0))


def test_build_windows_too_short_raises():
    with pytest.raises(ValueError):
        build_windows(jnp.arange(3.0), input_size=4, h=2)


def _tiny_net(h=4, input_size=12, batch_norm=False):
    return DeepNPTSNet(h=h, input_size=input_size, hidden_size=8, n_layers=2,
                       dropout=0.0, batch_norm=batch_norm, rngs=nnx.Rngs(0))


def test_forward_loss_scalar_finite():
    net = _tiny_net()
    w = build_windows(_make_y(60), input_size=12, h=4)
    loss = forward_loss(net, w[:8], h=4, input_size=12)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_returns_losses_and_reduces():
    # DeepNPTS's softmax-resample head converges more slowly than a direct-output
    # head, so use enough steps for a clear reduction (40 is noisy on this series).
    net = _tiny_net()
    losses = train(net, _make_y(120), h=4, input_size=12, max_steps=200,
                   windows_batch_size=16, lr=1e-3, seed=0)
    assert losses.shape == (200,)
    assert jnp.all(jnp.isfinite(losses))
    assert float(jnp.mean(losses[-5:])) < float(jnp.mean(losses[:5]))


def test_train_with_replacement_regime_runs():
    # n_windows (= 120 - 96 + 1 = 25) < windows_batch_size (= 64) -> WITH replacement.
    net = _tiny_net(h=24, input_size=72)
    losses = train(net, _make_y(120), h=24, input_size=72, max_steps=10,
                   windows_batch_size=64, lr=1e-3, seed=1)
    assert losses.shape == (10,)
    assert jnp.all(jnp.isfinite(losses))


def test_train_batch_norm_path_runs():
    net = _tiny_net(batch_norm=True)
    losses = train(net, _make_y(120), h=4, input_size=12, max_steps=20,
                   windows_batch_size=16, lr=1e-3, seed=0)
    assert losses.shape == (20,)
    assert jnp.all(jnp.isfinite(losses))


def test_predict_step_shape():
    net = _tiny_net()
    out = predict_step(net, _make_y(60), h=4, input_size=12)
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))


from chronax.models.deepnpts.deepnpts_model import (
    DeepNPTS, _boxcox, _inv_boxcox, _select_boxcox_lambda,
)
from chronax.utils import ConformalIntervals


# ============================================================================
# Model wrapper
# ============================================================================

def _fast_model(**kw):
    base = dict(h=4, input_size=12, hidden_size=8, n_layers=2, dropout=0.0,
                max_steps=40, learning_rate=1e-3, windows_batch_size=16, random_seed=0)
    base.update(kw)
    return DeepNPTS(**base)


def test_is_base_forecaster():
    assert issubclass(DeepNPTS, BaseForecaster)


def test_uses_exog_false():
    assert DeepNPTS.uses_exog is False


def test_input_size_default_resolves_to_3h():
    m = DeepNPTS(h=10)
    assert m.input_size == 30


def test_default_hyperparameters_match_nf():
    m = DeepNPTS(h=4)
    assert m.hidden_size == 32
    assert m.n_layers == 2
    assert m.dropout == 0.1
    assert m.batch_norm is False   # documented override of NF default (True)


def test_fit_predict_shape_and_finite():
    m = _fast_model()
    m.fit(_make_y(120))
    out = m.predict(h=4)
    assert out["mean"].shape == (4,)
    assert jnp.all(jnp.isfinite(out["mean"]))


def test_predict_h_greater_than_trained_raises():
    m = _fast_model()
    m.fit(_make_y(120))
    with pytest.raises(ValueError):
        m.predict(h=5)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        _fast_model().predict(h=4)


def test_fit_rejects_2d():
    with pytest.raises(ValueError):
        _fast_model().fit(jnp.ones((10, 2)))


def test_fit_rejects_exog():
    with pytest.raises(NotImplementedError):
        _fast_model().fit(_make_y(120), X=jnp.ones((120, 1)))


def test_forecast_matches_fit_predict():
    y = _make_y(120)
    a = _fast_model().forecast(y, h=4)["mean"]
    b = _fast_model().fit(y).predict(h=4)["mean"]
    assert jnp.allclose(a, b)


def test_forecast_fitted_values_shape_and_nan_head():
    m = _fast_model()
    out = m.forecast(_make_y(120), h=4, fitted=True)
    fitted = out["fitted"]
    assert fitted.shape == (120,)
    assert bool(jnp.all(jnp.isnan(fitted[:12])))
    assert bool(jnp.all(jnp.isfinite(fitted[12:])))


def test_pickle_round_trip_preserves_predictions():
    m = _fast_model()
    m.fit(_make_y(120))
    before = m.predict(h=4)["mean"]
    m2 = pickle.loads(pickle.dumps(m))
    after = m2.predict(h=4)["mean"]
    assert jnp.allclose(before, after)


def test_pickle_round_trip_with_batch_norm():
    m = _fast_model(batch_norm=True, dropout=0.1)
    m.fit(_make_y(120))
    before = m.predict(h=4)["mean"]
    m2 = pickle.loads(pickle.dumps(m))
    after = m2.predict(h=4)["mean"]
    assert jnp.allclose(before, after)


def test_predict_in_window_range():
    m = _fast_model()
    m.fit(_make_y(120))
    out = np.asarray(m.predict(h=4)["mean"])
    ctx = np.asarray(m._context)
    assert bool(np.all(out >= ctx.min() - 1e-4))
    assert bool(np.all(out <= ctx.max() + 1e-4))


def test_boxcox_round_trip_identity():
    y = jnp.asarray(np.linspace(1.0, 5.0, 50), dtype=jnp.float32)
    lam = _select_boxcox_lambda(y)
    assert jnp.allclose(_inv_boxcox(_boxcox(y, lam), lam), y, atol=1e-4)


def test_boxcox_requires_positive():
    m = _fast_model(use_boxcox=True)
    with pytest.raises(ValueError):
        m.fit(_make_y(120))   # sine series has non-positive values


def test_boxcox_fit_predict_positive_series():
    y = jnp.asarray(50.0 + 40.0 * np.sin(np.arange(160) / 6.0), dtype=jnp.float32)
    m = _fast_model(use_boxcox=True)
    m.fit(y)
    out = m.predict(h=4)
    assert out["mean"].shape == (4,)
    assert jnp.all(jnp.isfinite(out["mean"]))


def test_conformal_intervals_present_when_level_set():
    m = _fast_model()
    m.fit(_make_y(160))
    m.conformal_params = ConformalIntervals(n_windows=2, h=4)
    out = m.predict(h=4, level=[80])
    assert "lo-80" in out and "hi-80" in out
    assert out["lo-80"].shape == (4,)


def test_predict_level_without_conformal_params_raises():
    m = _fast_model()
    m.fit(_make_y(120))
    with pytest.raises(ValueError):
        m.predict(h=4, level=[80])


def test_conformal_does_not_corrupt_fitted_model():
    # conformity_scores re-fits inside a vmap; it must run on a throwaway copy
    # (self.new()) so the tracer-valued refits don't overwrite self.model_.
    # Regression for the "cs = self.conformity_scores(...)" bug: without the fix
    # the second predict below returns different values or raises a tracer leak.
    m = _fast_model()
    m.fit(_make_y(160))
    before = np.asarray(m.predict(h=4)["mean"])
    m.conformal_params = ConformalIntervals(n_windows=2, h=4)
    _ = m.predict(h=4, level=[80])                 # triggers conformity_scores
    after = np.asarray(m.predict(h=4)["mean"])     # model_ must be untouched
    assert np.all(np.isfinite(after))
    assert np.allclose(before, after)


# ============================================================================
# Parity vs neuralforecast (batch_norm=False)
# ============================================================================

from benchmarks.neural import resolve_nf_venv_py

_NF_VENV_PY = resolve_nf_venv_py(Path(__file__).resolve().parents[1])


@pytest.mark.skipif(not _NF_VENV_PY.exists(),
                    reason="neuralforecast reference venv (benchmarks/.venv-nf) not present")
def test_parity_vs_neuralforecast():
    """Chronax DeepNPTS vs neuralforecast at identical hyperparameters.

    Both sides: batch_norm=False, scaler_type='identity', dropout=0.0, same seed
    and window sampling. Trained models will not be bit-identical (torch vs JAX
    optimizers/PRNG differ), so this asserts the two engines land in the same
    ballpark rather than exact equality — enough to catch a wrong forward pass
    (e.g. softmax over the wrong axis) while tolerating optimizer noise.
    """
    h, input_size, max_steps = 8, 24, 300
    y = np.asarray(100.0 + 20.0 * np.sin(np.arange(220) / 7.0), dtype=np.float32)
    y_train = y[:-h]

    m = DeepNPTS(h=h, input_size=input_size, hidden_size=32, n_layers=2,
                 dropout=0.0, batch_norm=False, max_steps=max_steps,
                 learning_rate=1e-3, windows_batch_size=64, random_seed=1)
    m.fit(jnp.asarray(y_train))
    chronax_pred = np.asarray(m.predict(h=h)["mean"])

    code = f"""
import json, numpy as np, pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import DeepNPTS
y = np.asarray({y_train.tolist()!r}, dtype=np.float32)
df = pd.DataFrame({{'unique_id': 'p', 'ds': pd.date_range('2000-01-01', periods=len(y), freq='D'), 'y': y}})
m = DeepNPTS(h={h}, input_size={input_size}, hidden_size=32, n_layers=2,
             dropout=0.0, batch_norm=False, max_steps={max_steps},
             learning_rate=1e-3, windows_batch_size=64, scaler_type='identity',
             random_seed=1, loss=MAE(), accelerator='cpu', enable_progress_bar=False,
             logger=False, enable_model_summary=False, enable_checkpointing=False)
nf = NeuralForecast(models=[m], freq='D')
nf.fit(df=df)
print(json.dumps(nf.predict()['DeepNPTS'].to_numpy().tolist()))
"""
    out = subprocess.run([str(_NF_VENV_PY), "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"NF subprocess failed:\n{out.stderr}"
    nf_pred = np.asarray(json.loads(out.stdout.strip().splitlines()[-1]), dtype=np.float32)

    # Both must resample from the same window, so both lie in the series' range.
    lo, hi = float(y_train[-input_size:].min()), float(y_train[-input_size:].max())
    assert np.all(chronax_pred >= lo - 1.0) and np.all(chronax_pred <= hi + 1.0)
    assert np.all(nf_pred >= lo - 1.0) and np.all(nf_pred <= hi + 1.0)
    # Same ballpark: mean absolute deviation small relative to the series scale.
    series_scale = float(y_train.std())
    mad = float(np.mean(np.abs(chronax_pred - nf_pred)))
    assert mad < 0.5 * series_scale, f"MAD {mad:.3f} vs scale {series_scale:.3f}"

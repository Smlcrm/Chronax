"""Tests for chronax.models.TSMixerx.

Covers the five source modules of the tsmixerx subpackage in one file (loss,
data, model, train, forecaster), matching the flat ``test_<model>.py``
convention used elsewhere in ``tests/`` (see ``test_tsmixer.py``). Adds
exogenous-specific coverage (``create_windows_exog`` / ``make_batch_exog``,
``MixingLayerWithStaticExogenous``, and the full-exog train/eval path)
mirroring the pattern established in ``test_tide.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.tsmixerx.loss import LOSSES, masked_mae, masked_mse, resolve
from chronax.models.tsmixerx.data import (
    create_windows,
    create_windows_exog,
    make_batch,
    make_batch_exog,
)
from chronax.models.tsmixerx.model import (
    FeatureMixing,
    MixingLayer,
    MixingLayerWithStaticExogenous,
    TemporalMixing,
    TSMixerx,
    TSMixerxConfig,
)
from chronax.models.tsmixerx.train import create_train_state, eval_step, train_step
from chronax.models.tsmixerx.forecaster import TSMixerxForecaster


def _make_y(n=120, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float32)
    return jnp.asarray(np.sin(2 * np.pi * t / 12) + 0.05 * rng.randn(n), dtype=jnp.float32)


# ============================================================================
# Losses
# ============================================================================


def test_masked_mae_basic():
    y = jnp.array([[[1.0], [2.0], [3.0]]])
    y_hat = jnp.array([[[1.1], [2.0], [2.8]]])
    assert float(masked_mae(y, y_hat)) == pytest.approx(0.1, abs=1e-6)


def test_masked_mse_basic():
    y = jnp.array([[[1.0], [2.0]]])
    y_hat = jnp.array([[[2.0], [4.0]]])
    assert float(masked_mse(y, y_hat)) == pytest.approx(2.5, abs=1e-6)


def test_masked_mae_respects_mask():
    y = jnp.array([[[1.0], [2.0], [3.0]]])
    y_hat = jnp.array([[[10.0], [2.0], [3.0]]])
    mask = jnp.array([[[0.0], [1.0], [1.0]]])
    assert float(masked_mae(y, y_hat, mask)) == pytest.approx(0.0, abs=1e-6)


def test_masked_mae_horizon_weight():
    y = jnp.array([[[1.0], [2.0], [3.0]]])
    y_hat = jnp.array([[[2.0], [4.0], [6.0]]])
    horizon_weight = jnp.array([0.0, 0.0, 1.0])
    assert float(masked_mae(y, y_hat, horizon_weight=horizon_weight)) == pytest.approx(3.0)


def test_loss_all_masked_returns_zero():
    y = jnp.ones((2, 4, 1))
    y_hat = jnp.zeros((2, 4, 1))
    mask = jnp.zeros_like(y)
    assert float(masked_mae(y, y_hat, mask)) == 0.0
    assert float(masked_mse(y, y_hat, mask)) == 0.0


@pytest.mark.parametrize("name", ["mae", "mse"])
def test_loss_is_jit_and_grad_friendly(name):
    fn = LOSSES[name]
    y = jnp.array([[[1.0], [2.0], [3.0]]])
    g = jax.grad(lambda yh: fn(y, yh))(jnp.array([[[1.1], [2.2], [2.7]]]))
    assert g.shape == y.shape
    assert jnp.all(jnp.isfinite(g))
    jitted = jax.jit(fn)
    assert jnp.isfinite(jitted(y, y))


def test_resolve_string_returns_registry_entry():
    assert resolve("mae") is masked_mae


def test_resolve_callable_passes_through():
    fn = lambda y, yh: jnp.sum(y - yh)
    assert resolve(fn) is fn


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve("not_a_loss")


# ============================================================================
# Data pipeline
# ============================================================================


def test_create_windows_shapes():
    y = jnp.asarray(np.arange(20.0).reshape(20, 1))
    y = jnp.tile(y, (1, 2))  # [T=20, N=2]
    insample, outsample = create_windows(y, input_size=6, h=3)
    n_windows = 20 - (6 + 3) + 1
    assert insample.shape == (n_windows, 6, 2)
    assert outsample.shape == (n_windows, 3, 2)


def test_create_windows_content():
    y = jnp.arange(10.0)[:, None]  # [T=10, N=1]
    insample, outsample = create_windows(y, input_size=4, h=2)
    assert jnp.allclose(insample[0, :, 0], jnp.arange(4.0))
    assert jnp.allclose(outsample[0, :, 0], jnp.array([4.0, 5.0]))
    assert jnp.allclose(insample[-1, :, 0], jnp.arange(4.0, 8.0))
    assert jnp.allclose(outsample[-1, :, 0], jnp.array([8.0, 9.0]))


def test_create_windows_pads_short_series():
    y = jnp.arange(5.0)[:, None]  # too short for input_size=6, h=2 (need 8)
    insample, outsample = create_windows(y, input_size=6, h=2)
    assert insample.shape == (1, 6, 1)
    assert outsample.shape == (1, 2, 1)
    assert jnp.allclose(insample[0, :, 0], jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0]))
    assert jnp.allclose(outsample[0, :, 0], jnp.array([3.0, 4.0]))


def test_create_windows_accepts_1d_input():
    y = jnp.arange(10.0)
    insample, outsample = create_windows(y, input_size=4, h=2)
    assert insample.shape == (5, 4, 1)
    assert outsample.shape == (5, 2, 1)


def test_make_batch_selects_indices_and_builds_mask():
    insample = jnp.arange(24.0).reshape(4, 3, 2)
    outsample = jnp.arange(16.0).reshape(4, 2, 2)
    batch = make_batch(insample, outsample, jnp.array([0, 2]))
    assert batch["insample_y"].shape == (2, 3, 2)
    assert batch["outsample_y"].shape == (2, 2, 2)
    assert jnp.allclose(batch["insample_y"][0], insample[0])
    assert jnp.allclose(batch["insample_y"][1], insample[2])
    assert jnp.all(batch["sample_mask"] == 1.0)


def test_create_windows_exog_shapes_and_content():
    T, N, X, F = 20, 3, 1, 2
    y = jnp.asarray(np.arange(T * N).reshape(T, N), dtype=jnp.float32)
    hist_exog = jnp.ones((T, X, N))
    futr_exog = jnp.ones((T, F, N))
    windows = create_windows_exog(y, input_size=6, h=3, hist_exog=hist_exog, futr_exog=futr_exog)

    n_windows = T - (6 + 3) + 1
    assert windows["insample_y"].shape == (n_windows, 6, N)
    assert windows["outsample_y"].shape == (n_windows, 3, N)
    assert windows["hist_exog"].shape == (n_windows, X, 6, N)
    assert windows["futr_exog"].shape == (n_windows, F, 9, N)
    assert jnp.all(windows["hist_exog"] == 1.0)


def test_create_windows_exog_without_exog_returns_none():
    y = jnp.asarray(np.arange(20 * 2).reshape(20, 2), dtype=jnp.float32)
    windows = create_windows_exog(y, input_size=6, h=3)
    assert windows["hist_exog"] is None
    assert windows["futr_exog"] is None


def test_create_windows_exog_too_short_raises():
    y = jnp.arange(5.0)[:, None]
    with pytest.raises(ValueError):
        create_windows_exog(y, input_size=6, h=3)


def test_make_batch_exog_selects_indices_and_carries_stat():
    T, N, X = 20, 3, 1
    y = jnp.asarray(np.arange(T * N).reshape(T, N), dtype=jnp.float32)
    hist_exog = jnp.ones((T, X, N))
    windows = create_windows_exog(y, input_size=6, h=3, hist_exog=hist_exog)
    stat_exog = jnp.asarray(np.random.RandomState(0).randn(N, 4), dtype=jnp.float32)

    idx = jnp.array([0, 2])
    batch = make_batch_exog(windows, idx, stat_exog=stat_exog)
    assert batch["insample_y"].shape == (2, 6, N)
    assert batch["hist_exog"].shape == (2, X, 6, N)
    assert batch["futr_exog"] is None
    assert jnp.allclose(batch["stat_exog"], stat_exog)
    assert jnp.all(batch["sample_mask"] == 1.0)


# ============================================================================
# Model
# ============================================================================


def test_temporal_mixing_preserves_shape():
    m = TemporalMixing(dropout=0.0, use_batchnorm=False)
    x = jnp.asarray(np.random.RandomState(0).randn(2, 8, 3), dtype=jnp.float32)  # [B, T, C]
    params = m.init(jax.random.PRNGKey(0), x, deterministic=True)
    out = m.apply(params, x, deterministic=True)
    assert out.shape == x.shape
    assert "batch_stats" not in params


def test_temporal_mixing_with_batchnorm_has_batch_stats():
    m = TemporalMixing(dropout=0.0, use_batchnorm=True)
    x = jnp.asarray(np.random.RandomState(0).randn(2, 8, 3), dtype=jnp.float32)
    variables = m.init(jax.random.PRNGKey(0), x, deterministic=False)
    out, updates = m.apply(variables, x, deterministic=False, mutable=["batch_stats"])
    assert out.shape == x.shape
    assert "batch_stats" in updates


def test_feature_mixing_changes_trailing_dim():
    m = FeatureMixing(ff_dim=8, out_features=5, dropout=0.0, use_batchnorm=False)
    x = jnp.asarray(np.random.RandomState(1).randn(2, 4, 3), dtype=jnp.float32)  # [B, T, C_in=3]
    params = m.init(jax.random.PRNGKey(0), x, deterministic=True)
    out = m.apply(params, x, deterministic=True)
    assert out.shape == (2, 4, 5)
    assert "residual_proj" in params["params"]  # in != out -> residual projection exists


def test_feature_mixing_same_dim_has_no_residual_proj():
    m = FeatureMixing(ff_dim=8, out_features=3, dropout=0.0, use_batchnorm=False)
    x = jnp.ones((2, 4, 3))
    params = m.init(jax.random.PRNGKey(0), x, deterministic=True)
    assert "residual_proj" not in params["params"]


def test_mixing_layer_changes_trailing_dim():
    m = MixingLayer(ff_dim=8, out_features=6, dropout=0.0, use_batchnorm=False)
    x = jnp.asarray(np.random.RandomState(2).randn(2, 5, 9), dtype=jnp.float32)
    params = m.init(jax.random.PRNGKey(0), x, deterministic=True)
    out = m.apply(params, x, deterministic=True)
    assert out.shape == (2, 5, 6)


def test_mixing_layer_with_static_exogenous_shapes():
    B, h, ff_dim, S = 3, 4, 8, 2
    m = MixingLayerWithStaticExogenous(ff_dim=ff_dim, dropout=0.0, use_batchnorm=False)
    x = jnp.asarray(np.random.RandomState(3).randn(B, h, ff_dim), dtype=jnp.float32)
    stat = jnp.asarray(np.random.RandomState(4).randn(B, h, S), dtype=jnp.float32)
    params = m.init(jax.random.PRNGKey(0), (x, stat), deterministic=True)
    x_out, stat_out = m.apply(params, (x, stat), deterministic=True)
    assert x_out.shape == (B, h, ff_dim)
    assert jnp.allclose(stat_out, stat)  # passed through unchanged


def _tiny_config(**kw):
    base = dict(h=4, input_size=8, n_series=2, n_block=1, ff_dim=8, dropout=0.0)
    base.update(kw)
    return TSMixerxConfig(**base)


def test_tsmixerx_forward_shape_no_exog():
    cfg = _tiny_config()
    model = TSMixerx(cfg)
    x = jnp.asarray(np.random.RandomState(0).randn(3, cfg.input_size, cfg.n_series), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    assert "batch_stats" not in params  # use_batchnorm defaults False
    out = model.apply(params, x, deterministic=True)
    assert out.shape == (3, cfg.h, cfg.n_series)
    assert jnp.all(jnp.isfinite(out))


def test_tsmixerx_forward_shape_with_batchnorm():
    cfg = _tiny_config(use_batchnorm=True)
    model = TSMixerx(cfg)
    x = jnp.asarray(np.random.RandomState(0).randn(3, cfg.input_size, cfg.n_series), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), x, deterministic=False)
    out, updates = model.apply(variables, x, deterministic=False, mutable=["batch_stats"])
    assert out.shape == (3, cfg.h, cfg.n_series)
    assert "batch_stats" in updates


def test_tsmixerx_forward_shape_with_all_exog():
    cfg = _tiny_config(futr_exog_size=2, hist_exog_size=1, stat_exog_size=3)
    model = TSMixerx(cfg)
    B, L, h, N = 2, cfg.input_size, cfg.h, cfg.n_series
    rng = np.random.RandomState(1)
    insample_y = jnp.asarray(rng.randn(B, L, N), dtype=jnp.float32)
    hist_exog = jnp.asarray(rng.randn(B, 1, L, N), dtype=jnp.float32)
    futr_exog = jnp.asarray(rng.randn(B, 2, L + h, N), dtype=jnp.float32)
    stat_exog = jnp.asarray(rng.randn(N, 3), dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(0), insample_y,
        hist_exog=hist_exog, futr_exog=futr_exog, stat_exog=stat_exog,
        deterministic=True,
    )
    out = model.apply(
        params, insample_y,
        hist_exog=hist_exog, futr_exog=futr_exog, stat_exog=stat_exog,
        deterministic=True,
    )
    assert out.shape == (B, h, N)
    assert jnp.all(jnp.isfinite(out))


def test_tsmixerx_forward_no_revin():
    cfg = _tiny_config(revin=False, n_series=1)
    model = TSMixerx(cfg)
    x = jnp.asarray(np.random.RandomState(3).randn(2, cfg.input_size, 1), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    assert "revin_gamma" not in params["params"]
    out = model.apply(params, x, deterministic=True)
    assert out.shape == (2, cfg.h, 1)
    assert jnp.all(jnp.isfinite(out))


def test_tsmixerx_revin_affine_params_present():
    cfg = _tiny_config(revin=True, revin_affine=True)
    model = TSMixerx(cfg)
    x = jnp.ones((2, cfg.input_size, cfg.n_series))
    params = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    assert "revin_gamma" in params["params"]
    assert "revin_beta" in params["params"]
    assert params["params"]["revin_gamma"].shape == (cfg.n_series,)


def test_tsmixerx_deterministic_eval_repeatable():
    cfg = _tiny_config()
    model = TSMixerx(cfg)
    x = jnp.asarray(np.random.RandomState(5).randn(2, cfg.input_size, cfg.n_series), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    out1 = model.apply(params, x, deterministic=True)
    out2 = model.apply(params, x, deterministic=True)
    assert jnp.allclose(out1, out2)


# ============================================================================
# Train
# ============================================================================


def test_create_train_state_no_batchnorm_has_empty_batch_stats():
    cfg = _tiny_config()
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    assert int(state.step) == 0
    assert len(jax.tree_util.tree_leaves(state.batch_stats)) == 0


def test_create_train_state_with_batchnorm_has_batch_stats():
    cfg = _tiny_config(use_batchnorm=True)
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    assert len(jax.tree_util.tree_leaves(state.batch_stats)) > 0


def test_train_step_updates_params_no_exog():
    cfg = _tiny_config()
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    rng = np.random.RandomState(0)
    B = 5
    batch = {
        "insample_y": jnp.asarray(rng.randn(B, cfg.input_size, cfg.n_series), dtype=jnp.float32),
        "outsample_y": jnp.asarray(rng.randn(B, cfg.h, cfg.n_series), dtype=jnp.float32),
        "sample_mask": jnp.ones((B, cfg.h, cfg.n_series), dtype=jnp.float32),
    }
    new_state, loss, preds = train_step(state, batch, jax.random.PRNGKey(1))
    assert preds.shape == (B, cfg.h, cfg.n_series)
    assert jnp.isfinite(loss)

    leaves_before = jax.tree_util.tree_leaves(state.params)
    leaves_after = jax.tree_util.tree_leaves(new_state.params)
    diffs = [float(jnp.abs(a - b).max()) for a, b in zip(leaves_after, leaves_before)]
    assert max(diffs) > 0.0


def test_train_step_with_exog_runs_and_decreases_loss():
    cfg = _tiny_config(futr_exog_size=2, hist_exog_size=1, stat_exog_size=2)
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    B, L, h, N = 4, cfg.input_size, cfg.h, cfg.n_series
    rng = np.random.RandomState(2)
    batch = {
        "insample_y": jnp.asarray(rng.randn(B, L, N), dtype=jnp.float32),
        "outsample_y": jnp.asarray(rng.randn(B, h, N), dtype=jnp.float32),
        "sample_mask": jnp.ones((B, h, N), dtype=jnp.float32),
        "hist_exog": jnp.asarray(rng.randn(B, 1, L, N), dtype=jnp.float32),
        "futr_exog": jnp.asarray(rng.randn(B, 2, L + h, N), dtype=jnp.float32),
        "stat_exog": jnp.asarray(rng.randn(N, 2), dtype=jnp.float32),
    }

    rng_key = jax.random.PRNGKey(1)
    losses = []
    for _ in range(20):
        rng_key, sub = jax.random.split(rng_key)
        state, loss, preds = train_step(state, batch, sub)
        losses.append(float(loss))
    assert preds.shape == (B, h, N)
    assert jnp.isfinite(losses[-1])
    assert losses[-1] < losses[0]

    eval_loss, eval_preds = eval_step(state, batch)
    assert eval_preds.shape == (B, h, N)
    assert jnp.isfinite(eval_loss)


def test_training_decreases_loss_on_sine_signal():
    cfg = _tiny_config(h=4, input_size=12, n_series=1)
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    y = _make_y(60, seed=7)[:, None]  # [T, 1]
    insample, outsample = create_windows(y, cfg.input_size, cfg.h)
    batch = make_batch(insample, outsample, jnp.arange(insample.shape[0]))

    rng = jax.random.PRNGKey(1)
    losses = []
    for _ in range(60):
        rng, sub = jax.random.split(rng)
        state, loss, _ = train_step(state, batch, sub)
        losses.append(float(loss))

    assert float(np.mean(losses[-5:])) < float(np.mean(losses[:5]))


def test_eval_step_is_deterministic():
    cfg = _tiny_config()
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    rng = np.random.RandomState(2)
    B = 4
    batch = {
        "insample_y": jnp.asarray(rng.randn(B, cfg.input_size, cfg.n_series), dtype=jnp.float32),
        "outsample_y": jnp.asarray(rng.randn(B, cfg.h, cfg.n_series), dtype=jnp.float32),
        "sample_mask": jnp.ones((B, cfg.h, cfg.n_series), dtype=jnp.float32),
    }
    loss1, preds1 = eval_step(state, batch)
    loss2, preds2 = eval_step(state, batch)
    assert jnp.allclose(preds1, preds2)
    assert float(loss1) == pytest.approx(float(loss2))


# ============================================================================
# Forecaster
# ============================================================================


def _fast_forecaster(**kw):
    base = dict(
        h=6, input_size=18, n_series=1, n_block=1, ff_dim=8, dropout=0.0,
        max_steps=20, learning_rate=1e-2, batch_size=4, random_seed=0, val_fraction=0.0,
    )
    base.update(kw)
    return TSMixerxForecaster(**base)


def test_is_base_forecaster():
    assert issubclass(TSMixerxForecaster, BaseForecaster)


def test_input_size_default_resolves_to_3h():
    fc = TSMixerxForecaster(h=10)
    assert fc.input_size == 30
    assert fc.config.input_size == 30


def test_fit_predict_univariate_shape_and_finite():
    fc = _fast_forecaster()
    preds = fc.fit_predict(_make_y(120))
    assert preds.shape == (fc.config.h,)
    assert jnp.all(jnp.isfinite(preds))


def test_fit_predict_multivariate_shape_and_finite():
    series = [_make_y(100, seed=i) for i in range(3)]
    fc = _fast_forecaster(n_series=3)
    preds = fc.fit_predict(series)
    assert preds.shape == (fc.config.h, 3)
    assert jnp.all(jnp.isfinite(preds))


def test_fit_rejects_n_series_mismatch():
    fc = _fast_forecaster(n_series=2)
    with pytest.raises(ValueError):
        fc.fit(_make_y(120))


def test_forecast_before_fit_raises():
    with pytest.raises(RuntimeError):
        _fast_forecaster().forecast(y=_make_y(60))


def test_forecast_wrong_h_raises():
    fc = _fast_forecaster()
    fc.fit(_make_y(120))
    with pytest.raises(ValueError):
        fc.forecast(y=_make_y(120), h=999)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        _fast_forecaster().predict()


def test_predict_uses_cached_fit_y():
    fc = _fast_forecaster()
    fc.fit(_make_y(120))
    out = fc.predict()
    assert "mean" in out
    assert out["mean"].shape == (fc.config.h,)


def test_predict_level_not_supported():
    fc = _fast_forecaster()
    fc.fit(_make_y(120))
    with pytest.raises(NotImplementedError):
        fc.predict(level=[80])


def test_with_config_returns_new_unfitted_forecaster():
    fc = _fast_forecaster()
    fc.fit(_make_y(120))
    fc2 = fc.with_config(ff_dim=16)
    assert fc2.config.ff_dim == 16
    assert not fc2.fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

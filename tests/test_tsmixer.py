"""Tests for chronax.models.TSMixer.

Covers the five source modules of the tsmixer subpackage in one file (loss,
data, model, train, forecaster), matching the flat ``test_<model>.py``
convention used elsewhere in ``tests/`` (see ``test_bitcn.py``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.tsmixer.loss import LOSSES, masked_mae, masked_mse, resolve
from chronax.models.tsmixer.data import create_windows, make_batch
from chronax.models.tsmixer.model import (
    FeatureMixing,
    MixingLayer,
    TemporalMixing,
    TSMixer,
    TSMixerConfig,
)
from chronax.models.tsmixer.train import create_train_state, eval_step, train_step
from chronax.models.tsmixer.forecaster import TSMixerForecaster


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
    # Only the third horizon step counts: |3-6| = 3.
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
    # 3 zeros left-padded onto [0,1,2,3,4]: insample takes the first 6, outsample the rest.
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


# ============================================================================
# Model
# ============================================================================


def test_temporal_mixing_shape_with_batchnorm():
    m = TemporalMixing(dropout=0.0, use_batchnorm=True)
    x = jnp.asarray(np.random.RandomState(0).randn(2, 8, 3), dtype=jnp.float32)  # [B, L, N]
    variables = m.init(jax.random.PRNGKey(0), x, deterministic=False)
    out, updates = m.apply(variables, x, deterministic=False, mutable=["batch_stats"])
    assert out.shape == x.shape
    assert "batch_stats" in updates


def test_temporal_mixing_shape_with_layernorm_has_no_batch_stats():
    m = TemporalMixing(dropout=0.0, use_batchnorm=False)
    x = jnp.ones((2, 8, 3))
    variables = m.init(jax.random.PRNGKey(0), x, deterministic=True)
    assert "batch_stats" not in variables
    out = m.apply(variables, x, deterministic=True)
    assert out.shape == x.shape


def test_feature_mixing_shape():
    m = FeatureMixing(ff_dim=8, dropout=0.0, use_batchnorm=True)
    x = jnp.asarray(np.random.RandomState(1).randn(2, 5, 3), dtype=jnp.float32)  # [B, L, N]
    variables = m.init(jax.random.PRNGKey(0), x, deterministic=False)
    out, updates = m.apply(variables, x, deterministic=False, mutable=["batch_stats"])
    assert out.shape == x.shape
    assert "batch_stats" in updates


def test_mixing_layer_preserves_shape():
    m = MixingLayer(ff_dim=8, dropout=0.0, use_batchnorm=True)
    x = jnp.asarray(np.random.RandomState(2).randn(2, 6, 3), dtype=jnp.float32)
    variables = m.init(jax.random.PRNGKey(0), x, deterministic=False)
    out, updates = m.apply(variables, x, deterministic=False, mutable=["batch_stats"])
    assert out.shape == x.shape


def _tiny_config(**kw):
    base = dict(h=4, input_size=8, n_series=2, n_block=1, ff_dim=8, dropout=0.0)
    base.update(kw)
    return TSMixerConfig(**base)


def test_tsmixer_forward_shape_default():
    cfg = _tiny_config()
    model = TSMixer(cfg)
    x = jnp.asarray(np.random.RandomState(0).randn(3, cfg.input_size, cfg.n_series), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), x, deterministic=False)
    out, updates = model.apply(variables, x, deterministic=False, mutable=["batch_stats"])
    assert out.shape == (3, cfg.h, cfg.n_series)
    assert "batch_stats" in updates

    out_eval = model.apply(variables, x, deterministic=True)
    assert out_eval.shape == (3, cfg.h, cfg.n_series)
    assert jnp.all(jnp.isfinite(out_eval))


def test_tsmixer_forward_no_batchnorm_no_revin_no_skip():
    cfg = _tiny_config(revin=False, use_batchnorm=False, use_global_skip=False, n_series=1)
    model = TSMixer(cfg)
    x = jnp.asarray(np.random.RandomState(3).randn(2, cfg.input_size, 1), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    assert "batch_stats" not in params
    out = model.apply(params, x, deterministic=True)
    assert out.shape == (2, cfg.h, 1)
    assert jnp.all(jnp.isfinite(out))


def test_tsmixer_forward_with_global_skip():
    cfg = _tiny_config(use_global_skip=True)
    model = TSMixer(cfg)
    x = jnp.asarray(np.random.RandomState(4).randn(2, cfg.input_size, cfg.n_series), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    assert "skip_out" in variables["params"]
    out = model.apply(variables, x, deterministic=True)
    assert out.shape == (2, cfg.h, cfg.n_series)


def test_tsmixer_revin_affine_params_present():
    cfg = _tiny_config(revin=True, revin_affine=True)
    model = TSMixer(cfg)
    x = jnp.ones((2, cfg.input_size, cfg.n_series))
    variables = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    assert "revin_gamma" in variables["params"]
    assert "revin_beta" in variables["params"]
    assert variables["params"]["revin_gamma"].shape == (cfg.n_series,)


def test_tsmixer_deterministic_eval_repeatable():
    cfg = _tiny_config()
    model = TSMixer(cfg)
    x = jnp.asarray(np.random.RandomState(5).randn(2, cfg.input_size, cfg.n_series), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    out1 = model.apply(variables, x, deterministic=True)
    out2 = model.apply(variables, x, deterministic=True)
    assert jnp.allclose(out1, out2)


# ============================================================================
# Train
# ============================================================================


def test_create_train_state_has_batch_stats():
    cfg = _tiny_config()
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    assert int(state.step) == 0
    assert len(jax.tree_util.tree_leaves(state.batch_stats)) > 0


def test_train_step_updates_params_and_batch_stats():
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

    param_leaves_before = jax.tree_util.tree_leaves(state.params)
    param_leaves_after = jax.tree_util.tree_leaves(new_state.params)
    diffs = [float(jnp.abs(a - b).max()) for a, b in zip(param_leaves_after, param_leaves_before)]
    assert max(diffs) > 0.0

    stats_before = jax.tree_util.tree_leaves(state.batch_stats)
    stats_after = jax.tree_util.tree_leaves(new_state.batch_stats)
    stat_diffs = [float(jnp.abs(a - b).max()) for a, b in zip(stats_after, stats_before)]
    assert max(stat_diffs) > 0.0


def test_training_decreases_loss_on_sine_signal():
    """A per-window-constant target is trivial under RevIN (normalises to zero),
    so use a real sine series where the model must learn the mixing weights."""
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


def test_eval_step_uses_running_stats_and_is_deterministic():
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
    # eval_step must not mutate batch_stats.
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda a, b: bool(jnp.all(a == b)), state.batch_stats, state.batch_stats)
    )


# ============================================================================
# Forecaster
# ============================================================================


def _fast_forecaster(**kw):
    base = dict(
        h=6, input_size=18, n_series=1, n_block=1, ff_dim=8, dropout=0.0,
        max_steps=20, learning_rate=1e-2, batch_size=4, random_seed=0, val_fraction=0.0,
    )
    base.update(kw)
    return TSMixerForecaster(**base)


def test_is_base_forecaster():
    assert issubclass(TSMixerForecaster, BaseForecaster)


def test_input_size_default_resolves_to_3h():
    fc = TSMixerForecaster(h=10)
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

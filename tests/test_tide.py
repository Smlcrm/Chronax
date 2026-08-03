"""Tests for chronax.models.TiDE.

Covers the five source modules of the tide subpackage in one file (loss,
data, model, train, forecaster), matching the flat ``test_<model>.py``
convention used elsewhere in ``tests/`` (see ``test_bitcn.py``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.tide.loss import LOSSES, masked_mae, masked_mse, resolve
from chronax.models.tide.data import (
    RobustScaler,
    align_covariates,
    build_windows,
    create_batch,
    pad_sequence,
    split_train_val_windows,
)
from chronax.models.tide.model import MLPResidual, TiDE, TiDEConfig
from chronax.models.tide.train import (
    create_train_state,
    eval_step,
    eval_step_windows,
    eval_step_windows_raw,
    eval_step_windows_std,
    train_step,
    train_step_windows,
    train_step_windows_raw,
    train_step_windows_std,
)
from chronax.models.tide.forecaster import TiDEForecaster


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


def test_build_windows_shape_and_content():
    y = jnp.arange(10.0)
    w = build_windows(y, input_size=4, h=2)
    assert w.shape == (5, 6)
    assert jnp.allclose(w[0], jnp.arange(6.0))
    assert jnp.allclose(w[-1], jnp.arange(4.0, 10.0))


def test_build_windows_too_short_raises():
    with pytest.raises(ValueError):
        build_windows(jnp.arange(3.0), input_size=4, h=2)


def test_split_train_val_windows_basic():
    windows = jnp.arange(20).reshape(10, 2).astype(jnp.float32)
    train, val = split_train_val_windows(windows, val_fraction=0.3)
    assert train.shape[0] == 7
    assert val.shape[0] == 3


def test_split_train_val_windows_zero_fraction_keeps_all_train():
    windows = jnp.arange(10).reshape(5, 2).astype(jnp.float32)
    train, val = split_train_val_windows(windows, val_fraction=0.0)
    assert train.shape[0] == 5
    assert val.shape[0] == 0


def test_split_train_val_windows_empty_raises():
    with pytest.raises(ValueError):
        split_train_val_windows(jnp.zeros((0, 2)))


def test_robust_scaler_round_trip():
    x = jnp.asarray(np.random.RandomState(0).randn(4, 10), dtype=jnp.float32)
    scaler = RobustScaler()
    shift, scale = scaler.stats(x, axis=1)
    z = scaler.transform(x, shift, scale)
    back = scaler.inverse(z, shift, scale)
    assert jnp.allclose(back, x, atol=1e-4)


def test_robust_scaler_constant_row_no_nan():
    x = jnp.ones((2, 6))
    scaler = RobustScaler()
    shift, scale = scaler.stats(x, axis=1)
    z = scaler.transform(x, shift, scale)
    assert jnp.all(jnp.isfinite(z))


def test_pad_sequence_left_pads():
    padded, mask = pad_sequence(jnp.array([1.0, 2.0, 3.0]), 5)
    assert padded.tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_pad_sequence_truncates_when_longer():
    padded, mask = pad_sequence(jnp.arange(10.0), 4)
    assert padded.tolist() == [6.0, 7.0, 8.0, 9.0]
    assert mask.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_align_covariates_pad_and_truncate():
    aligned_pad = align_covariates(
        hist_exog=jnp.ones((3, 2)),
        futr_exog=jnp.ones((4, 1)),
        stat_exog=jnp.ones(5),
        input_size=10,
        h=4,
    )
    assert aligned_pad["hist_exog"].shape == (10, 2)
    assert aligned_pad["futr_exog"].shape == (14, 1)
    assert aligned_pad["stat_exog"].shape == (5,)
    assert jnp.all(aligned_pad["hist_exog"][:7] == 0)
    assert jnp.all(aligned_pad["hist_exog"][7:] == 1)

    aligned_trunc = align_covariates(
        hist_exog=jnp.arange(20).reshape(20, 1).astype(jnp.float32),
        futr_exog=None,
        stat_exog=None,
        input_size=5,
        h=2,
    )
    assert aligned_trunc["hist_exog"].shape == (5, 1)
    assert aligned_trunc["hist_exog"].ravel().tolist() == [15, 16, 17, 18, 19]
    assert aligned_trunc["futr_exog"] is None
    assert aligned_trunc["stat_exog"] is None


def test_create_batch_shapes_no_exog():
    y_series = [jnp.arange(20.0), jnp.arange(5.0, 25.0)]
    batch = create_batch(y_series, input_size=10, h=5)
    assert batch["insample_y"].shape == (2, 10, 1)
    assert batch["outsample_y"].shape == (2, 5, 1)
    assert batch["available_mask"].shape == (2, 10, 1)
    assert batch["sample_mask"].shape == (2, 5, 1)
    assert batch["hist_exog"] is None
    assert batch["futr_exog"] is None
    assert batch["stat_exog"] is None


def test_create_batch_with_exog():
    y_series = [jnp.arange(20.0), jnp.arange(5.0, 25.0)]
    hist = [jnp.ones((20, 2)), jnp.ones((20, 2))]
    futr = [jnp.ones((25, 3)), jnp.ones((25, 3))]
    stat = [jnp.ones(4), jnp.ones(4)]
    batch = create_batch(
        y_series, input_size=10, h=5,
        hist_exog_list=hist, futr_exog_list=futr, stat_exog_list=stat,
    )
    assert batch["hist_exog"].shape == (2, 10, 2)
    assert batch["futr_exog"].shape == (2, 15, 3)
    assert batch["stat_exog"].shape == (2, 4)


# ============================================================================
# Model
# ============================================================================


def test_mlp_residual_shape_and_layernorm():
    m = MLPResidual(hidden_size=8, output_dim=4, dropout_rate=0.0, use_layernorm=True)
    x = jnp.ones((3, 6))
    params = m.init(jax.random.PRNGKey(0), x)
    out = m.apply(params, x)
    assert out.shape == (3, 4)


def test_mlp_residual_3d_input():
    """MLPResidual operates on the last axis, so [B, T, D] inputs work too."""
    m = MLPResidual(hidden_size=8, output_dim=4, dropout_rate=0.0, use_layernorm=False)
    x = jnp.ones((2, 5, 6))
    params = m.init(jax.random.PRNGKey(0), x)
    out = m.apply(params, x)
    assert out.shape == (2, 5, 4)


def _tiny_config(**kw):
    base = dict(
        h=4, input_size=8, hidden_size=8, decoder_output_dim=4,
        temporal_decoder_dim=8, dropout=0.0, num_encoder_layers=1,
        num_decoder_layers=1, temporal_width=2,
    )
    base.update(kw)
    return TiDEConfig(**base)


def test_tide_forward_shape_no_exog():
    cfg = _tiny_config()
    model = TiDE(config=cfg)
    x = jnp.asarray(np.random.RandomState(0).randn(3, cfg.input_size, 1), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x)
    out = model.apply(params, x)
    assert out.shape == (3, cfg.h, cfg.output_size)
    assert jnp.all(jnp.isfinite(out))


def test_tide_forward_shape_with_all_exog():
    cfg = _tiny_config(hist_exog_size=2, futr_exog_size=3, stat_exog_size=1)
    model = TiDE(config=cfg)
    rng = np.random.RandomState(1)
    B = 2
    x = jnp.asarray(rng.randn(B, cfg.input_size, 1), dtype=jnp.float32)
    hist = jnp.asarray(rng.randn(B, cfg.input_size, 2), dtype=jnp.float32)
    futr = jnp.asarray(rng.randn(B, cfg.input_size + cfg.h, 3), dtype=jnp.float32)
    stat = jnp.asarray(rng.randn(B, 1), dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(0), x, hist_exog=hist, futr_exog=futr, stat_exog=stat
    )
    out = model.apply(params, x, hist_exog=hist, futr_exog=futr, stat_exog=stat)
    assert out.shape == (B, cfg.h, cfg.output_size)
    assert jnp.all(jnp.isfinite(out))


def test_tide_deterministic_repeatable():
    cfg = _tiny_config()
    model = TiDE(config=cfg)
    x = jnp.ones((2, cfg.input_size, 1))
    params = model.init(jax.random.PRNGKey(0), x, deterministic=True)
    out1 = model.apply(params, x, deterministic=True)
    out2 = model.apply(params, x, deterministic=True)
    assert jnp.allclose(out1, out2)


# ============================================================================
# Train
# ============================================================================


def test_create_train_state_starts_at_step_zero():
    state = create_train_state(jax.random.PRNGKey(0), _tiny_config(), learning_rate=1e-2)
    assert int(state.step) == 0


def test_train_step_windows_updates_params_and_finite_loss():
    cfg = _tiny_config()
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    windows = jnp.asarray(
        np.random.RandomState(0).randn(6, cfg.input_size + cfg.h), dtype=jnp.float32
    )
    new_state, loss, preds = train_step_windows(
        state, windows, jax.random.PRNGKey(1), input_size=cfg.input_size
    )
    assert preds.shape == (6, cfg.h, 1)
    assert jnp.isfinite(loss)

    leaves_before = jax.tree_util.tree_leaves(state.params)
    leaves_after = jax.tree_util.tree_leaves(new_state.params)
    diffs = [float(jnp.abs(a - b).max()) for a, b in zip(leaves_after, leaves_before)]
    assert max(diffs) > 0.0


def test_training_decreases_loss_on_constant_signal():
    cfg = _tiny_config(h=3, input_size=6)
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-1)
    windows = jnp.tile(jnp.arange(cfg.input_size + cfg.h, dtype=jnp.float32), (8, 1))

    rng = jax.random.PRNGKey(1)
    losses = []
    for _ in range(30):
        rng, sub = jax.random.split(rng)
        state, loss, _ = train_step_windows(state, windows, sub, input_size=cfg.input_size)
        losses.append(float(loss))

    assert losses[-1] < losses[0]


def test_eval_step_windows_is_deterministic():
    cfg = _tiny_config(h=3, input_size=6)
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    windows = jnp.asarray(
        np.random.RandomState(1).randn(4, cfg.input_size + cfg.h), dtype=jnp.float32
    )
    loss1, preds1 = eval_step_windows(state, windows, input_size=cfg.input_size)
    loss2, preds2 = eval_step_windows(state, windows, input_size=cfg.input_size)
    assert jnp.allclose(preds1, preds2)
    assert float(loss1) == pytest.approx(float(loss2))


def test_train_step_windows_std_and_raw_variants_run():
    cfg = _tiny_config(h=3, input_size=6)
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    windows = jnp.asarray(
        np.random.RandomState(2).randn(5, cfg.input_size + cfg.h), dtype=jnp.float32
    )

    new_state, loss_std, preds_std = train_step_windows_std(
        state, windows, jax.random.PRNGKey(3), input_size=cfg.input_size
    )
    assert preds_std.shape == (5, cfg.h, 1)
    assert jnp.isfinite(loss_std)
    eval_loss_std, _ = eval_step_windows_std(new_state, windows, input_size=cfg.input_size)
    assert jnp.isfinite(eval_loss_std)

    new_state2, loss_raw, preds_raw = train_step_windows_raw(
        state, windows, jax.random.PRNGKey(4), input_size=cfg.input_size
    )
    assert preds_raw.shape == (5, cfg.h, 1)
    assert jnp.isfinite(loss_raw)
    eval_loss_raw, _ = eval_step_windows_raw(new_state2, windows, input_size=cfg.input_size)
    assert jnp.isfinite(eval_loss_raw)


def test_train_step_batch_based_with_exog_runs():
    cfg = _tiny_config(hist_exog_size=2, futr_exog_size=1, stat_exog_size=1)
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    B = 4
    rng = np.random.RandomState(5)
    batch = {
        "insample_y": jnp.asarray(rng.randn(B, cfg.input_size, 1), dtype=jnp.float32),
        "outsample_y": jnp.asarray(rng.randn(B, cfg.h, 1), dtype=jnp.float32),
        "sample_mask": jnp.ones((B, cfg.h, 1), dtype=jnp.float32),
        "hist_exog": jnp.asarray(rng.randn(B, cfg.input_size, 2), dtype=jnp.float32),
        "futr_exog": jnp.asarray(rng.randn(B, cfg.input_size + cfg.h, 1), dtype=jnp.float32),
        "stat_exog": jnp.asarray(rng.randn(B, 1), dtype=jnp.float32),
    }
    new_state, loss, preds = train_step(state, batch, jax.random.PRNGKey(6))
    assert preds.shape == (B, cfg.h, 1)
    assert jnp.isfinite(loss)
    eval_loss, _ = eval_step(new_state, batch)
    assert jnp.isfinite(eval_loss)


# ============================================================================
# Forecaster
# ============================================================================


def _fast_forecaster(**kw):
    base = dict(
        h=6, input_size=18, hidden_size=8, decoder_output_dim=4,
        temporal_decoder_dim=8, dropout=0.0, temporal_width=2,
        max_steps=20, learning_rate=1e-2, batch_size=4, random_seed=0, val_fraction=0.0,
    )
    base.update(kw)
    return TiDEForecaster(**base)


def test_is_base_forecaster():
    assert issubclass(TiDEForecaster, BaseForecaster)


def test_input_size_default_resolves_to_3h():
    fc = TiDEForecaster(h=10)
    assert fc.input_size == 30
    assert fc.config.input_size == 30


def test_fit_predict_univariate_shape_and_finite():
    fc = _fast_forecaster()
    preds = fc.fit_predict(_make_y(120))
    assert preds.shape == (fc.config.h,)
    assert jnp.all(jnp.isfinite(preds))


def test_fit_predict_panel_shape_and_finite():
    series = [_make_y(100, seed=i) for i in range(3)]
    fc = _fast_forecaster()
    preds = fc.fit_predict(series)
    assert preds.shape == (3, fc.config.h)
    assert jnp.all(jnp.isfinite(preds))


def test_fit_predict_with_global_scale():
    fc = _fast_forecaster(global_scale=True)
    preds = fc.fit_predict(_make_y(120))
    assert preds.shape == (fc.config.h,)
    assert jnp.all(jnp.isfinite(preds))


def test_fit_predict_with_std_scaler():
    fc = _fast_forecaster(use_std_scaler=True)
    preds = fc.fit_predict(_make_y(120))
    assert preds.shape == (fc.config.h,)
    assert jnp.all(jnp.isfinite(preds))


def test_forecast_before_fit_raises():
    with pytest.raises(RuntimeError):
        _fast_forecaster().forecast(y=_make_y(60))


def test_forecast_requires_y():
    fc = _fast_forecaster()
    fc.fit(_make_y(120))
    with pytest.raises(ValueError):
        fc.forecast(y=None)


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
    fc2 = fc.with_config(hidden_size=16)
    assert fc2.config.hidden_size == 16
    assert not fc2.fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

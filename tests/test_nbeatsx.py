"""Tests for chronax.models.NBEATSx.

Covers the five source modules of the nbeatsx subpackage in one file (loss,
data, model, train, forecaster), matching the flat ``test_<model>.py``
convention used elsewhere in ``tests/`` (see ``test_nbeats.py``). Adds
exogenous-specific coverage (``ExogenousBasis``, the ``"exogenous"`` stack
type, and the batch-based exog train/eval path) mirroring the pattern
established in ``test_tide.py``.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.nbeatsx.loss import LOSSES, masked_mae, masked_mse, resolve
from chronax.models.nbeatsx.data import (
    RobustScaler,
    build_windows,
    create_batch,
    pad_sequence,
    split_train_val_windows,
)
from chronax.models.nbeatsx.model import (
    ExogenousBasis,
    NBEATSx,
    NBEATSxBlock,
    NBEATSxConfig,
    _harmonic_size,
    _n_theta,
    _polynomial_basis,
    _seasonality_basis,
)
from chronax.models.nbeatsx.train import (
    create_train_state,
    eval_batch_step,
    eval_step,
    train_batch_step,
    train_step,
)
from chronax.models.nbeatsx.forecaster import NBEATSxForecaster


def _make_y(n=120, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float32)
    return jnp.asarray(np.sin(2 * np.pi * t / 12) + 0.05 * rng.randn(n), dtype=jnp.float32)


# ============================================================================
# Losses
# ============================================================================


def test_masked_mae_basic():
    y = jnp.array([[1.0, 2.0, 3.0]])
    y_hat = jnp.array([[1.1, 2.0, 2.8]])
    assert float(masked_mae(y, y_hat)) == pytest.approx(0.1, abs=1e-6)


def test_masked_mse_basic():
    y = jnp.array([[1.0, 2.0]])
    y_hat = jnp.array([[2.0, 4.0]])
    assert float(masked_mse(y, y_hat)) == pytest.approx(2.5, abs=1e-6)


def test_masked_mae_respects_mask():
    y = jnp.array([[1.0, 2.0, 3.0]])
    y_hat = jnp.array([[10.0, 2.0, 3.0]])
    mask = jnp.array([[0.0, 1.0, 1.0]])
    assert float(masked_mae(y, y_hat, mask)) == pytest.approx(0.0, abs=1e-6)


def test_loss_all_masked_returns_zero():
    y = jnp.ones((2, 4))
    y_hat = jnp.zeros((2, 4))
    mask = jnp.zeros_like(y)
    assert float(masked_mae(y, y_hat, mask)) == 0.0
    assert float(masked_mse(y, y_hat, mask)) == 0.0


@pytest.mark.parametrize("name", ["mae", "mse"])
def test_loss_is_jit_and_grad_friendly(name):
    fn = LOSSES[name]
    y = jnp.array([[1.0, 2.0, 3.0]])
    g = jax.grad(lambda yh: fn(y, yh))(jnp.array([[1.1, 2.2, 2.7]]))
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
    assert jnp.allclose(train, windows[:7])
    assert jnp.allclose(val, windows[7:])


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


def test_create_batch_shapes_no_exog():
    y_series = [jnp.arange(20.0), jnp.arange(5.0, 25.0)]
    batch = create_batch(y_series, input_size=10, h=5)
    assert batch["insample_y"].shape == (2, 10)
    assert batch["outsample_y"].shape == (2, 5)
    assert batch["available_mask"].shape == (2, 10)
    assert batch["sample_mask"].shape == (2, 5)
    assert jnp.all(batch["sample_mask"] == 1.0)
    assert batch["hist_exog"] is None
    assert batch["futr_exog"] is None
    assert batch["stat_exog"] is None


def test_create_batch_with_sample_mask():
    y_series = [jnp.arange(20.0)]
    mask = [jnp.array([1.0, 1.0, 0.0, 1.0, 1.0])]
    batch = create_batch(y_series, input_size=10, h=5, sample_mask_list=mask)
    assert batch["sample_mask"][0].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0]


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


def test_create_batch_with_partial_exog_fills_zeros():
    """When only some series carry exog, missing entries fill with zeros."""
    y_series = [jnp.arange(20.0), jnp.arange(5.0, 25.0)]
    hist = [jnp.ones((20, 2)), None]
    batch = create_batch(y_series, input_size=10, h=5, hist_exog_list=hist)
    assert batch["hist_exog"].shape == (2, 10, 2)
    assert jnp.all(batch["hist_exog"][1] == 0.0)


# ============================================================================
# Model
# ============================================================================


def test_n_theta_identity():
    assert _n_theta("identity", input_size=24, h=12, n_basis=2, n_harmonics=2) == 36


def test_n_theta_trend():
    assert _n_theta("trend", input_size=24, h=12, n_basis=2, n_harmonics=2) == 6


def test_n_theta_seasonality():
    n = _n_theta("seasonality", input_size=24, h=12, n_basis=2, n_harmonics=2)
    assert n == 2 * _harmonic_size(2, 12)


def test_n_theta_exogenous():
    n = _n_theta(
        "exogenous", input_size=24, h=12, n_basis=2, n_harmonics=2,
        futr_input_size=3, stat_input_size=1,
    )
    assert n == 2 * (3 + 1)


def test_n_theta_unknown_block_raises():
    with pytest.raises(ValueError):
        _n_theta("bogus", 10, 5, 2, 2)


def test_polynomial_basis_shape_and_constant_term():
    b = _polynomial_basis(length=4, n_terms=3)
    assert b.shape == (3, 4)
    assert jnp.allclose(b[0], jnp.ones(4))  # t**0 == 1


def test_seasonality_basis_shapes_match_harmonic_size():
    bb, fb = _seasonality_basis(backcast_size=24, forecast_size=12, harmonics=2)
    K = _harmonic_size(2, 12)
    assert bb.shape == (K, 24)
    assert fb.shape == (K, 12)


def test_exogenous_basis_shapes():
    B, L, h, C = 3, 8, 4, 5
    basis = ExogenousBasis(forecast_size=h)
    theta = jnp.asarray(np.random.RandomState(0).randn(B, 2 * C), dtype=jnp.float32)
    futr_exog = jnp.asarray(np.random.RandomState(1).randn(B, L + h, C), dtype=jnp.float32)
    params = basis.init(jax.random.PRNGKey(0), theta, futr_exog)
    backcast, forecast = basis.apply(params, theta, futr_exog)
    assert backcast.shape == (B, L)
    assert forecast.shape == (B, h)
    assert jnp.all(jnp.isfinite(backcast))
    assert jnp.all(jnp.isfinite(forecast))


@pytest.mark.parametrize("block_type", ["identity", "trend", "seasonality"])
def test_block_shapes_no_exog(block_type):
    cfg = NBEATSxConfig(h=6, input_size=12, mlp_units=8, mlp_layers=1, layer_norm=False)
    block = NBEATSxBlock(config=cfg, block_type=block_type)
    x = jnp.ones((3, 12))
    params = block.init(jax.random.PRNGKey(0), x)
    backcast, forecast = block.apply(params, x)
    assert backcast.shape == (3, 12)
    assert forecast.shape == (3, 6)


def test_block_exogenous_type_shapes():
    cfg = NBEATSxConfig(
        h=6, input_size=12, mlp_units=8, mlp_layers=1, layer_norm=False,
        futr_exog_size=2, stat_exog_size=1,
    )
    block = NBEATSxBlock(config=cfg, block_type="exogenous")
    B = 3
    residuals = jnp.ones((B, 12))
    futr_exog = jnp.asarray(np.random.RandomState(0).randn(B, 18, 2), dtype=jnp.float32)
    stat_exog = jnp.asarray(np.random.RandomState(1).randn(B, 1), dtype=jnp.float32)
    params = block.init(
        jax.random.PRNGKey(0), residuals, futr_exog=futr_exog, stat_exog=stat_exog
    )
    backcast, forecast = block.apply(
        params, residuals, futr_exog=futr_exog, stat_exog=stat_exog
    )
    assert backcast.shape == (3, 12)
    assert forecast.shape == (3, 6)
    assert jnp.all(jnp.isfinite(forecast))


def test_block_exogenous_type_without_exog_raises():
    cfg = NBEATSxConfig(h=4, input_size=8, mlp_units=8, mlp_layers=1, layer_norm=False)
    block = NBEATSxBlock(config=cfg, block_type="exogenous")
    x = jnp.ones((2, 8))
    with pytest.raises(ValueError):
        block.init(jax.random.PRNGKey(0), x)


def test_block_unknown_type_raises():
    cfg = NBEATSxConfig(h=4, input_size=8, mlp_units=8, mlp_layers=1, layer_norm=False)
    block = NBEATSxBlock(config=cfg, block_type="bogus")
    x = jnp.ones((2, 8))
    with pytest.raises(ValueError):
        block.init(jax.random.PRNGKey(0), x)


@pytest.mark.parametrize(
    "stack_types,n_blocks",
    [
        (("identity",), (1,)),
        (("trend",), (1,)),
        (("seasonality",), (1,)),
        (("identity", "trend", "seasonality"), (1, 1, 1)),
    ],
)
def test_nbeatsx_forward_shape_no_exog(stack_types, n_blocks):
    cfg = NBEATSxConfig(
        h=6, input_size=12, stack_types=stack_types, n_blocks=n_blocks,
        mlp_units=8, mlp_layers=1, layer_norm=False,
    )
    model = NBEATSx(config=cfg)
    x = jnp.asarray(np.random.RandomState(0).randn(4, 12), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x)
    out = model.apply(params, x)
    assert out.shape == (4, 6)
    assert jnp.all(jnp.isfinite(out))


def test_nbeatsx_forward_shape_with_all_exog():
    cfg = NBEATSxConfig(
        h=6, input_size=12, mlp_units=8, mlp_layers=1, layer_norm=False,
        stack_types=("identity", "exogenous"), n_blocks=(1, 1),
        futr_exog_size=2, hist_exog_size=3, stat_exog_size=1,
    )
    model = NBEATSx(config=cfg)
    B = 4
    rng = np.random.RandomState(1)
    insample_y = jnp.asarray(rng.randn(B, 12), dtype=jnp.float32)
    hist_exog = jnp.asarray(rng.randn(B, 12, 3), dtype=jnp.float32)
    futr_exog = jnp.asarray(rng.randn(B, 18, 2), dtype=jnp.float32)
    stat_exog = jnp.asarray(rng.randn(B, 1), dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(0), insample_y,
        hist_exog=hist_exog, futr_exog=futr_exog, stat_exog=stat_exog,
    )
    out = model.apply(
        params, insample_y,
        hist_exog=hist_exog, futr_exog=futr_exog, stat_exog=stat_exog,
    )
    assert out.shape == (B, 6)
    assert jnp.all(jnp.isfinite(out))


def test_nbeatsx_forward_defaults_missing_exog_to_zeros():
    """When exog sizes are configured but not passed, the model fills zeros
    rather than raising — matches the TiDE-style optional-exog convention."""
    cfg = NBEATSxConfig(
        h=4, input_size=8, mlp_units=8, mlp_layers=1, layer_norm=False,
        stack_types=("identity",), n_blocks=(1,),
        hist_exog_size=2,
    )
    model = NBEATSx(config=cfg)
    x = jnp.ones((2, 8))
    params = model.init(jax.random.PRNGKey(0), x, hist_exog=jnp.zeros((2, 8, 2)))
    out = model.apply(params, x)  # hist_exog omitted
    assert out.shape == (2, 4)
    assert jnp.all(jnp.isfinite(out))


def test_nbeatsx_deterministic_repeatable():
    cfg = NBEATSxConfig(h=4, input_size=8, mlp_units=8, mlp_layers=1, layer_norm=False)
    model = NBEATSx(config=cfg)
    x = jnp.ones((2, 8))
    params = model.init(jax.random.PRNGKey(0), x)
    out1 = model.apply(params, x, deterministic=True)
    out2 = model.apply(params, x, deterministic=True)
    assert jnp.allclose(out1, out2)


def test_nbeatsx_shared_weights_uses_fewer_parameters():
    cfg_shared = NBEATSxConfig(
        h=4, input_size=8, stack_types=("identity",), n_blocks=(3,),
        mlp_units=8, mlp_layers=1, layer_norm=False, shared_weights=True,
    )
    cfg_unshared = replace(cfg_shared, shared_weights=False)
    x = jnp.ones((2, 8))
    p_shared = NBEATSx(config=cfg_shared).init(jax.random.PRNGKey(0), x)
    p_unshared = NBEATSx(config=cfg_unshared).init(jax.random.PRNGKey(0), x)
    n_shared = sum(leaf.size for leaf in jax.tree_util.tree_leaves(p_shared))
    n_unshared = sum(leaf.size for leaf in jax.tree_util.tree_leaves(p_unshared))
    assert n_shared < n_unshared


def test_nbeatsx_forward_accepts_insample_mask():
    cfg = NBEATSxConfig(h=4, input_size=8, mlp_units=8, mlp_layers=1, layer_norm=False)
    model = NBEATSx(config=cfg)
    x = jnp.asarray(np.random.RandomState(1).randn(1, 8), dtype=jnp.float32)
    mask = jnp.ones_like(x)
    params = model.init(jax.random.PRNGKey(0), x)
    out_default = model.apply(params, x)
    out_explicit_mask = model.apply(params, x, insample_mask=mask)
    assert jnp.allclose(out_default, out_explicit_mask)


# ============================================================================
# Train
# ============================================================================


def _tiny_config(**kw):
    base = dict(h=4, input_size=8, mlp_units=8, mlp_layers=1, layer_norm=False)
    base.update(kw)
    return NBEATSxConfig(**base)


def test_create_train_state_starts_at_step_zero():
    state = create_train_state(jax.random.PRNGKey(0), _tiny_config(), learning_rate=1e-2)
    assert int(state.step) == 0


def test_train_step_updates_params_and_finite_loss():
    cfg = _tiny_config()
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    windows = jnp.asarray(
        np.random.RandomState(0).randn(6, cfg.input_size + cfg.h), dtype=jnp.float32
    )
    new_state, loss, preds = train_step(
        state, windows, jax.random.PRNGKey(1), input_size=cfg.input_size
    )
    assert preds.shape == (6, cfg.h)
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
        state, loss, _ = train_step(state, windows, sub, input_size=cfg.input_size)
        losses.append(float(loss))

    assert losses[-1] < losses[0]


def test_eval_step_is_deterministic():
    cfg = _tiny_config(h=3, input_size=6)
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    windows = jnp.asarray(
        np.random.RandomState(1).randn(4, cfg.input_size + cfg.h), dtype=jnp.float32
    )
    loss1, preds1 = eval_step(state, windows, input_size=cfg.input_size)
    loss2, preds2 = eval_step(state, windows, input_size=cfg.input_size)
    assert jnp.allclose(preds1, preds2)
    assert float(loss1) == pytest.approx(float(loss2))


def test_train_batch_step_with_exog_runs_and_decreases_loss():
    cfg = _tiny_config(
        stack_types=("identity", "exogenous"), n_blocks=(1, 1),
        futr_exog_size=2, hist_exog_size=1, stat_exog_size=1,
    )
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-2)
    B = 5
    rng = np.random.RandomState(2)
    batch = {
        "insample_y": jnp.asarray(rng.randn(B, cfg.input_size), dtype=jnp.float32),
        "outsample_y": jnp.asarray(rng.randn(B, cfg.h), dtype=jnp.float32),
        "sample_mask": jnp.ones((B, cfg.h), dtype=jnp.float32),
        "hist_exog": jnp.asarray(rng.randn(B, cfg.input_size, 1), dtype=jnp.float32),
        "futr_exog": jnp.asarray(rng.randn(B, cfg.input_size + cfg.h, 2), dtype=jnp.float32),
        "stat_exog": jnp.asarray(rng.randn(B, 1), dtype=jnp.float32),
    }

    rng_key = jax.random.PRNGKey(1)
    losses = []
    for _ in range(20):
        rng_key, sub = jax.random.split(rng_key)
        state, loss, preds = train_batch_step(state, batch, sub, scale=False)
        losses.append(float(loss))
    assert preds.shape == (B, cfg.h)
    assert jnp.isfinite(losses[-1])
    assert losses[-1] < losses[0]

    eval_loss, eval_preds = eval_batch_step(state, batch, scale=False)
    assert eval_preds.shape == (B, cfg.h)
    assert jnp.isfinite(eval_loss)


# ============================================================================
# Forecaster
# ============================================================================


def _fast_forecaster(**kw):
    base = dict(
        h=6, input_size=18, mlp_units=8, mlp_layers=1, layer_norm=False,
        max_steps=20, learning_rate=1e-2, batch_size=4, random_seed=0, val_fraction=0.0,
    )
    base.update(kw)
    return NBEATSxForecaster(**base)


def test_is_base_forecaster():
    assert issubclass(NBEATSxForecaster, BaseForecaster)


def test_input_size_default_resolves_to_3h():
    fc = NBEATSxForecaster(h=10)
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
    fc2 = fc.with_config(mlp_units=16)
    assert fc2.config.mlp_units == 16
    assert not fc2.fitted


def test_fit_predict_with_validation_split():
    fc = _fast_forecaster(val_fraction=0.2, val_check_steps=5)
    preds = fc.fit_predict(_make_y(150))
    assert preds.shape == (fc.config.h,)
    assert jnp.all(jnp.isfinite(preds))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

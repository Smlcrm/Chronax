"""End-to-end tests for the Chronax JAX/Flax RNN.

The forward-parity tests against an inline PyTorch mirror of Nixtla
NeuralForecast's ``RNN`` live with the benchmark platform, not here.

Test plan:

* ``test_loss_*``               — numerical correctness of MAE/MSE under
                                  masking and horizon-weighting.
* ``test_data_pipeline``        — batch shapes, padding, available/sample masks.
* ``test_data_alignment``       — covariate alignment to ``input_size + h``.
* ``test_train_step_runs``      — full train + eval step on a JIT path produces
                                  finite losses and parameter changes.
* ``test_forecaster_*``         — high-level RNNForecaster fit/forecast smoke.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chronax.models.rnn.data import (  # noqa: E402
    align_covariates,
    create_batch,
    pad_sequence,
)
from chronax.models.rnn.loss import masked_mae, masked_mse  # noqa: E402
from chronax.models.rnn.model import (  # noqa: E402
    RNN,
    RNNConfig,
    autoregressive_predict,
)
from chronax.models.rnn.train import (  # noqa: E402
    create_train_state,
    eval_step,
    train_step,
)


# ===========================================================================
# Loss tests
# ===========================================================================


def test_loss_basic_mae_mse():
    y = jnp.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    y_hat = jnp.array([[[1.1], [2.0], [2.8]], [[3.5], [5.0], [6.2]]])
    mask = jnp.array([[[1.0], [1.0], [0.0]], [[1.0], [0.0], [1.0]]])

    mae = float(masked_mae(y, y_hat, mask))
    mse = float(masked_mse(y, y_hat, mask))

    assert mae == pytest.approx(0.2, abs=1e-6)
    assert mse == pytest.approx(0.075, abs=1e-6)


def test_loss_no_mask_matches_unmasked():
    y = jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    y_hat = jnp.array([[[1.5], [1.5]], [[3.5], [3.5]]])
    expected_mae = float(jnp.mean(jnp.abs(y - y_hat)))
    expected_mse = float(jnp.mean((y - y_hat) ** 2))
    assert float(masked_mae(y, y_hat)) == pytest.approx(expected_mae, abs=1e-6)
    assert float(masked_mse(y, y_hat)) == pytest.approx(expected_mse, abs=1e-6)


def test_loss_horizon_weight():
    y = jnp.array([[[1.0], [2.0], [3.0]]])
    y_hat = jnp.array([[[2.0], [4.0], [6.0]]])
    horizon_weight = jnp.array([0.0, 0.0, 1.0])
    # Only the third position counts: |3-6| = 3.
    assert float(masked_mae(y, y_hat, horizon_weight=horizon_weight)) == pytest.approx(3.0)


def test_loss_all_masked_returns_zero():
    y = jnp.ones((2, 4, 1))
    y_hat = jnp.zeros((2, 4, 1))
    mask = jnp.zeros_like(y)
    assert float(masked_mae(y, y_hat, mask)) == 0.0
    assert float(masked_mse(y, y_hat, mask)) == 0.0


def test_loss_is_jit_compatible():
    y = jnp.ones((2, 4, 1))
    y_hat = jnp.zeros((2, 4, 1))
    mask = jnp.ones_like(y)
    jit_loss = jax.jit(masked_mae)
    out = float(jit_loss(y, y_hat, mask))
    assert out == pytest.approx(1.0, abs=1e-6)


def test_loss_grad_finite():
    """Gradient should flow and never produce NaN even with empty masks."""
    y = jnp.array([[[1.0], [2.0]]])
    mask = jnp.array([[[0.0], [0.0]]])

    def loss(y_hat):
        return masked_mse(y, y_hat, mask)

    g = jax.grad(loss)(jnp.zeros_like(y))
    assert jnp.all(jnp.isfinite(g))


# ===========================================================================
# Data pipeline tests
# ===========================================================================


def test_data_pipeline_basic():
    input_size, h, B = 10, 5, 4
    rng = np.random.RandomState(0)
    y_series = [rng.randn(15 + i) for i in range(B)]
    hist = [rng.randn(15 + i, 2) for i in range(B)]
    futr = [rng.randn(15 + i + h, 3) for i in range(B)]
    stat = [rng.randn(4) for _ in range(B)]
    sample_masks = [np.ones(h) for _ in range(B)]

    batch = create_batch(
        y_series, input_size, h,
        hist_exog_list=hist,
        futr_exog_list=futr,
        stat_exog_list=stat,
        sample_mask_list=sample_masks,
    )

    assert batch["insample_y"].shape == (B, input_size, 1)
    assert batch["outsample_y"].shape == (B, h, 1)
    assert batch["available_mask"].shape == (B, input_size, 1)
    assert batch["sample_mask"].shape == (B, h, 1)
    assert batch["hist_exog"].shape == (B, input_size, 2)
    assert batch["futr_exog"].shape == (B, input_size + h, 3)
    assert batch["stat_exog"].shape == (B, 4)
    assert (np.asarray(batch["sample_mask"]) == 1.0).all()


def test_data_pipeline_no_exog():
    input_size, h, B = 8, 3, 2
    y_series = [np.arange(20.0) for _ in range(B)]
    batch = create_batch(y_series, input_size, h)
    assert batch["hist_exog"] is None
    assert batch["futr_exog"] is None
    assert batch["stat_exog"] is None
    assert batch["insample_y"].shape == (B, input_size, 1)


def test_data_pipeline_padding():
    """A short series should be left-padded; available_mask reflects that."""
    input_size, h = 10, 3
    short_series = np.arange(5.0)  # 5 history points (after horizon split = 2)
    batch = create_batch([short_series], input_size, h)
    avail = np.asarray(batch["available_mask"])[0, :, 0]
    insample = np.asarray(batch["insample_y"])[0, :, 0]

    n_pad = input_size - 2  # we keep 2 history points
    assert avail[:n_pad].sum() == 0.0
    assert avail[n_pad:].sum() == 2.0
    assert (insample[:n_pad] == 0).all()


def test_align_covariates_pad_and_truncate():
    aligned_pad = align_covariates(
        hist_exog=np.ones((3, 2)),
        futr_exog=np.ones((4, 1)),
        stat_exog=np.ones(5),
        input_size=10,
        h=4,
    )
    assert aligned_pad["hist_exog"].shape == (10, 2)
    assert aligned_pad["futr_exog"].shape == (14, 1)
    assert aligned_pad["stat_exog"].shape == (5,)
    assert (aligned_pad["hist_exog"][:7] == 0).all()
    assert (aligned_pad["hist_exog"][7:] == 1).all()

    aligned_trunc = align_covariates(
        hist_exog=np.arange(20).reshape(20, 1).astype(np.float32),
        futr_exog=None,
        stat_exog=None,
        input_size=5,
        h=2,
    )
    assert aligned_trunc["hist_exog"].shape == (5, 1)
    assert aligned_trunc["hist_exog"].ravel().tolist() == [15, 16, 17, 18, 19]


def test_pad_sequence_left_pads():
    padded, mask = pad_sequence(np.array([1.0, 2.0, 3.0]), 5)
    assert padded.tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


# ===========================================================================
# Masking interaction
# ===========================================================================


def test_loss_under_random_mask_matches_manual_average():
    """Compute MSE on random predictions with mask, sanity-check vs numpy."""
    rng = np.random.RandomState(7)
    y = rng.randn(3, 6, 1).astype(np.float32)
    y_hat = rng.randn(3, 6, 1).astype(np.float32)
    mask = rng.randint(0, 2, size=(3, 6, 1)).astype(np.float32)

    j_loss = float(masked_mse(jnp.array(y), jnp.array(y_hat), jnp.array(mask)))
    diff_sq = (y - y_hat) ** 2
    expected = (diff_sq * mask).sum() / max(mask.sum(), 1.0)
    assert j_loss == pytest.approx(float(expected), abs=1e-5)


# ===========================================================================
# Train/eval steps
# ===========================================================================


def test_train_and_eval_step_run():
    cfg = RNNConfig(h=3, input_size=8, encoder_hidden_size=8, encoder_n_layers=1)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, cfg, learning_rate=1e-2)

    rng_step = jax.random.PRNGKey(1)
    rng_data = np.random.RandomState(0)
    y_series = [rng_data.randn(20) for _ in range(4)]
    batch = create_batch(y_series, cfg.input_size, cfg.h)

    new_state, loss, preds = train_step(state, batch, rng_step)
    assert preds.shape == (4, cfg.h, cfg.output_size)
    assert jnp.isfinite(loss)
    # Params should change after a step.
    leaves_before = jax.tree_util.tree_leaves(state.params)
    leaves_after = jax.tree_util.tree_leaves(new_state.params)
    diffs = [
        float(jnp.abs(a - b).max()) for a, b in zip(leaves_after, leaves_before)
    ]
    assert max(diffs) > 0.0

    eval_loss, eval_preds = eval_step(new_state, batch)
    assert eval_preds.shape == (4, cfg.h, cfg.output_size)
    assert jnp.isfinite(eval_loss)


def test_training_decreases_loss_on_constant_signal():
    """Optimiser actually reduces loss on an easy synthetic task."""
    cfg = RNNConfig(h=3, input_size=6, encoder_hidden_size=8, encoder_n_layers=1)
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-1)

    # Constant series — model should learn the mean quickly.
    y_series = [np.ones(20, dtype=np.float32) * 3.0 for _ in range(8)]
    batch = create_batch(y_series, cfg.input_size, cfg.h)

    rng = jax.random.PRNGKey(123)
    losses = []
    for _ in range(50):
        rng, sub = jax.random.split(rng)
        state, loss, _ = train_step(state, batch, sub)
        losses.append(float(loss))

    assert losses[-1] < losses[0] * 0.5, f"expected loss to drop, got {losses[0]} -> {losses[-1]}"


# ===========================================================================
# RNNForecaster (high-level fit/forecast adapter)
# ===========================================================================


def test_forecaster_fit_predict_runs_univariate():
    """RNNForecaster.fit_predict on a single seasonal series returns finite [h]."""
    from chronax.models.rnn.forecaster import RNNForecaster

    fc = RNNForecaster(
        h=12, input_size=24, encoder_hidden_size=8, encoder_n_layers=1,
        recurrent=True,
        max_steps=30, learning_rate=1e-2, batch_size=4, random_seed=0,
    )
    t = np.arange(120, dtype=np.float32)
    y = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.RandomState(0).randn(120)

    preds = fc.fit_predict(y.astype(np.float32))
    assert preds.shape == (fc.config.h,)
    assert np.all(np.isfinite(preds))


def test_forecaster_fit_predict_runs_panel():
    """RNNForecaster on a panel of 3 series returns [n_series, h] preds."""
    from chronax.models.rnn.forecaster import RNNForecaster

    rng = np.random.RandomState(1)
    series = [
        np.sin(2 * np.pi * np.arange(80) / 12).astype(np.float32) + 0.05 * rng.randn(80)
        for _ in range(3)
    ]
    fc = RNNForecaster(
        h=6, input_size=18, encoder_hidden_size=8, encoder_n_layers=1,
        recurrent=True,
        max_steps=30, learning_rate=1e-2, batch_size=3, random_seed=0,
    )
    preds = fc.fit_predict(series)
    assert preds.shape == (3, fc.config.h)
    assert np.all(np.isfinite(preds))


def test_forecaster_direct_decoder_runs():
    """Direct (non-recurrent) MLP-decoder path also produces finite predictions."""
    from chronax.models.rnn.forecaster import RNNForecaster

    y = np.sin(np.linspace(0, 8 * np.pi, 60)).astype(np.float32)
    fc = RNNForecaster(
        h=4, input_size=12, encoder_hidden_size=8, encoder_n_layers=1,
        decoder_hidden_size=8, decoder_layers=2, recurrent=False,
        max_steps=20, learning_rate=1e-2, batch_size=2, random_seed=0,
    )
    preds = fc.fit_predict(y)
    assert preds.shape == (fc.config.h,)
    assert np.all(np.isfinite(preds))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

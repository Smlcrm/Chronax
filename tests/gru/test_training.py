"""Tests for chronax.models.gru.gru_training — windows, forward+loss, train, predict."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.gru.gru_module import GRUNet
from chronax.models.gru.gru_scaler import RobustScaler
from chronax.models.gru.gru_training import (
    build_windows,
    predict_step,
    scaled_forward_loss,
    train,
)


def test_build_windows_shape_and_no_leakage():
    y = jnp.arange(20.0)
    w = build_windows(y, input_size=5, h=3)
    # n_windows = T - L - h + 1 = 20 - 5 - 3 + 1 = 13
    assert w.shape == (13, 8)
    np.testing.assert_allclose(w[0], jnp.arange(8.0))
    np.testing.assert_allclose(w[-1], jnp.arange(12.0, 20.0))


def test_build_windows_raises_on_short_series():
    with pytest.raises(ValueError):
        build_windows(jnp.arange(5.0), input_size=5, h=3)


def test_scaled_forward_loss_returns_scalar():
    net = GRUNet(
        in_features=1, encoder_hidden=8, encoder_layers=1,
        decoder_hidden=4, decoder_layers=2, dropout=0.0,
        h=3, input_size=5, rngs=nnx.Rngs(0),
    )
    windows = jnp.arange(2 * 8, dtype=jnp.float32).reshape(2, 8)
    loss = scaled_forward_loss(net, windows, h=3, input_size=5, scaler=RobustScaler())
    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert jnp.isfinite(loss)


def test_scaled_forward_loss_is_scale_invariant():
    """Scaling the input series by 100x shouldn't change the loss in scaled
    space — the per-window scaler normalizes both insample and target."""
    net = GRUNet(
        in_features=1, encoder_hidden=8, encoder_layers=1,
        decoder_hidden=4, decoder_layers=2, dropout=0.0,
        h=3, input_size=5, rngs=nnx.Rngs(0),
    )
    rng = np.random.default_rng(0)
    base = jnp.asarray(rng.standard_normal((2, 8)), dtype=jnp.float32)
    sc = RobustScaler()
    l1 = scaled_forward_loss(net, base, h=3, input_size=5, scaler=sc)
    l2 = scaled_forward_loss(net, base * 100.0, h=3, input_size=5, scaler=sc)
    np.testing.assert_allclose(float(l1), float(l2), rtol=1e-3)


def test_train_gradient_step_decreases_loss_on_fixed_batch():
    """One gradient step on a FIXED batch must strictly decrease the loss
    evaluated on that same batch. This is a true invariant of gradient
    descent (with small enough lr) and isolates correctness from the
    stochastic batch-sampling in `train`.
    """
    import optax
    from chronax.models.gru.gru_training import build_windows, scaled_forward_loss

    y = jnp.asarray(np.sin(np.arange(120) / 5.0), dtype=jnp.float32)
    net = GRUNet(
        in_features=1, encoder_hidden=16, encoder_layers=1,
        decoder_hidden=8, decoder_layers=2, dropout=0.0,
        h=6, input_size=18, rngs=nnx.Rngs(0),
    )
    windows = build_windows(y, input_size=18, h=6)
    batch = windows[:8]
    sc = RobustScaler()

    loss_before = float(
        scaled_forward_loss(net, batch, h=6, input_size=18, scaler=sc)
    )
    optimizer = nnx.Optimizer(net, optax.adam(1e-2), wrt=nnx.Param)
    grads = nnx.grad(
        lambda m: scaled_forward_loss(m, batch, h=6, input_size=18, scaler=sc)
    )(net)
    optimizer.update(grads)
    loss_after = float(
        scaled_forward_loss(net, batch, h=6, input_size=18, scaler=sc)
    )
    assert loss_after < loss_before, f"{loss_before=} {loss_after=}"


def test_train_deterministic_with_same_seed_dropout_enabled():
    """Determinism with dropout active — exercises the RNG threading path."""
    y = jnp.asarray(np.sin(np.arange(200) / 10.0), dtype=jnp.float32)

    def run():
        net = GRUNet(
            in_features=1, encoder_hidden=16, encoder_layers=2,
            decoder_hidden=8, decoder_layers=2, dropout=0.1,
            h=6, input_size=18, rngs=nnx.Rngs(7),
        )
        return train(net, y, h=6, input_size=18, max_steps=10,
                     batch_size=8, lr=1e-3, seed=7)

    np.testing.assert_allclose(run(), run(), rtol=1e-5)


def test_predict_step_shape_and_idempotent():
    y = jnp.asarray(np.sin(np.arange(200) / 10.0), dtype=jnp.float32)
    net = GRUNet(
        in_features=1, encoder_hidden=16, encoder_layers=1,
        decoder_hidden=8, decoder_layers=2, dropout=0.0,
        h=6, input_size=18, rngs=nnx.Rngs(0),
    )
    train(net, y, h=6, input_size=18, max_steps=5, batch_size=4, lr=1e-3, seed=0)
    sc = RobustScaler()
    p1 = predict_step(net, y, h=6, input_size=18, scaler=sc)
    p2 = predict_step(net, y, h=6, input_size=18, scaler=sc)
    assert p1.shape == (6,)
    np.testing.assert_allclose(p1, p2, rtol=1e-5)


def test_train_losses_match_pre_refactor_trajectory():
    """After the nnx.scan refactor of train(), per-step losses on a fixed
    seed must stay close to the pre-refactor trajectory (captured
    2026-05-17 against `yash/gru`'s HEAD before the refactor).

    Bit-for-bit equality is not promised: the RNG-threading order changed
    (per-step `jax.random.split` + per-step `choice` → upfront single
    `choice` of shape [max_steps, batch_size]). The same seed produces a
    different sample sequence with the new layout, so per-step losses
    drift. Mean within rtol=0.20 catches real refactor bugs (NaN, wildly
    different magnitudes) while allowing the legitimate sampling drift.
    With only max_steps=10 the loss-mean is particularly noise-sensitive;
    we use a looser bound on the mean than on the final value.
    """
    # Reference values from the pre-refactor capture; if these need to be
    # updated for a legitimate algorithmic change, re-run the capture on
    # the pre-change branch and update here.
    PRE_REFACTOR_LOSSES_MEAN = 2.8072
    PRE_REFACTOR_LOSSES_FINAL = 1.3303

    from chronax.models.gru.gru_module import GRUNet
    from chronax.models.gru.gru_training import train
    from flax import nnx

    y = jnp.asarray(np.sin(np.arange(200) / 10.0), dtype=jnp.float32)
    net = GRUNet(in_features=1, encoder_hidden=16, encoder_layers=1,
                 decoder_hidden=8, decoder_layers=2, dropout=0.0,
                 h=4, input_size=12, rngs=nnx.Rngs(42))
    losses = np.asarray(train(net, y, h=4, input_size=12, max_steps=10,
                              batch_size=4, lr=1e-3, seed=42))
    # Mean: 20% slack — captures noise from a 10-step stochastic average
    # of differently-ordered batches.
    np.testing.assert_allclose(losses.mean(), PRE_REFACTOR_LOSSES_MEAN, rtol=0.20)
    # Final loss: 30% slack — single-step value, even noisier.
    np.testing.assert_allclose(losses[-1], PRE_REFACTOR_LOSSES_FINAL, rtol=0.30)


def test_predict_matches_pre_refactor_trajectory():
    """End-to-end fit().predict() output stays close in magnitude after
    the refactor. Allowed slack: atol=0.5 because the prediction's
    magnitude is O(1) and we accept trajectory drift, not bit-for-bit."""
    PRE_REFACTOR_PRED_MEAN = 0.4585

    from chronax.models.gru.gru_model import GRU

    y = jnp.asarray(np.sin(np.arange(200) / 10.0), dtype=jnp.float32)
    m = GRU(h=4, input_size=12, hidden_size=16, n_layers=1,
            max_steps=10, batch_size=4, random_seed=42)
    m.fit(y)
    pred = np.asarray(m.predict(h=4)["mean"])
    np.testing.assert_allclose(pred.mean(), PRE_REFACTOR_PRED_MEAN, atol=0.5)

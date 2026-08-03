"""Window construction, LR schedule and JIT-compiled training/predict steps for DilatedRNN.

Training runs in SCALED space: each rolling window's insample slice sets the
robust (median/MAD) statistics, the target is scaled with those same statistics,
and the loss is taken there — which is what neuralforecast does via ``TemporalNorm``
(``scaler_type="robust"``) before ``DilatedRNN.forward``. Predictions are inverted
with the statistics of the prediction context.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chronax.models.dilated_rnn.dilated_rnn_losses import LossFn, mae
from chronax.models.dilated_rnn.dilated_rnn_module import DilatedRNNNet
from chronax.models.dilated_rnn.dilated_rnn_scaler import RobustScaler, Scaler


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> jnp.ndarray:
    """Return [n_windows, input_size+h] rolling windows; step=1."""
    window_size = input_size + h
    n = y.shape[0] - window_size + 1
    if n <= 0:
        raise ValueError(
            f"Series length {y.shape[0]} too short for input_size={input_size}, h={h}"
        )
    idx = jnp.arange(window_size)[None, :] + jnp.arange(n)[:, None]
    return y[idx]


def make_lr_schedule(learning_rate, max_steps: int, num_lr_decays: int):
    """Reproduce NF's ``StepLR(step_size=max_steps // num_lr_decays, gamma=0.5)``.

    neuralforecast's ``DilatedRNN`` defaults to ``num_lr_decays=3``, i.e. the learning
    rate halves three times over training — a real part of the recipe, not a detail:
    without it the model keeps taking full-size steps at the end of the run. Torch's
    StepLR is a staircase, ``lr * gamma^floor(step / step_size)``, which is exactly
    ``optax.exponential_decay(..., staircase=True)``.

    ``num_lr_decays <= 0`` disables decay (NF's convention). A callable
    ``learning_rate`` is assumed to be a user-supplied ``optax`` schedule and is
    passed through untouched.
    """
    if callable(learning_rate) or num_lr_decays <= 0:
        return learning_rate
    step_size = max(max_steps // num_lr_decays, 1)
    return optax.exponential_decay(
        init_value=learning_rate, transition_steps=step_size,
        decay_rate=0.5, staircase=True,
    )


def scaled_forward_loss(model: DilatedRNNNet, windows: jnp.ndarray, *, h: int,
                        input_size: int, scaler: Scaler,
                        loss_fn: LossFn = mae) -> jnp.ndarray:
    """Forward + point loss in SCALED space. windows: [B, input_size+h] -> scalar.

    The insample slice sets (shift, scale); the target is scaled with the SAME
    statistics, so the loss is computed exactly where NF computes it.
    """
    insample = windows[:, :input_size]
    target = windows[:, input_size:]
    shift, scale = scaler.stats(insample, axis=1)
    insample_z = scaler.transform(insample, shift, scale)
    target_z = scaler.transform(target, shift, scale)
    pred = model(insample_z[..., None], deterministic=False)   # [B, h, 1]
    return loss_fn(pred[..., 0], target_z)


@nnx.jit
def _jit_forward_deterministic(model: DilatedRNNNet, x: jnp.ndarray) -> jnp.ndarray:
    """Inference-mode forward. Cached across calls."""
    return model(x, deterministic=True)


def train(model: DilatedRNNNet, y: jnp.ndarray, *, h: int, input_size: int,
          max_steps: int, windows_batch_size: int, lr: optax.ScalarOrSchedule,
          seed: int, scaler: Scaler | None = None,
          loss_fn: LossFn = mae) -> jnp.ndarray:
    """Train ``model`` in place via a single ``nnx.scan``. Returns per-step losses.

    The whole loop is one ``nnx.scan`` (carry = (model, optimizer)), which keeps the
    function ``jax.vmap``-traceable for ``BaseForecaster.conformity_scores``. Window
    sampling replicates neuralforecast's REGIME-DEPENDENT scheme (``_base_model.py``
    training_step): when ``n_windows < windows_batch_size`` NF draws
    ``windows_batch_size`` indices WITH replacement (oversampling with duplicates —
    the regime the small benchmark series hit, e.g. ~25 windows for AirlinePassengers
    at input_size=72, h=24, far below the default batch of 128); otherwise it takes a
    without-replacement permutation of ``windows_batch_size`` windows. Getting this
    branch right is load-bearing for accuracy parity, so we do NOT collapse it to a
    full batch.

    ``lr`` may be a scalar or an ``optax`` schedule — pass the output of
    :func:`make_lr_schedule` to get NF's StepLR behaviour.

    Note: ``batches`` materializes a ``[max_steps, windows_batch_size, input_size+h]``
    tensor up front. At NF's DilatedRNN defaults (windows_batch_size=128,
    max_steps=1000, input_size=72, h=24) that is ~49 MB of float32; raising
    ``windows_batch_size`` much further wants a per-step sampler instead.
    """
    if scaler is None:
        scaler = RobustScaler()
    windows = build_windows(y, input_size, h)
    n_windows = windows.shape[0]

    key = jax.random.PRNGKey(seed)
    step_keys = jax.random.split(key, max_steps)

    if n_windows < windows_batch_size:
        # NF: torch.randint(0, n_windows, size=(windows_batch_size,)) — WITH replacement.
        def sample_one(k):
            return jax.random.choice(k, n_windows, shape=(windows_batch_size,), replace=True)
    else:
        # NF: torch.randperm(n_windows)[:windows_batch_size] — WITHOUT replacement.
        def sample_one(k):
            return jax.random.permutation(k, n_windows)[:windows_batch_size]

    batch_idx = jax.vmap(sample_one)(step_keys)   # [max_steps, windows_batch_size]
    batches = windows[batch_idx]                  # [max_steps, windows_batch_size, L+h]

    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, batch_windows):
        model, opt = carry
        loss, grads = nnx.value_and_grad(
            lambda m: scaled_forward_loss(
                m, batch_windows, h=h, input_size=input_size,
                scaler=scaler, loss_fn=loss_fn,
            )
        )(model)
        opt.update(grads)
        return (model, opt), loss

    _, losses = step((model, optimizer), batches)

    # Finite check runs only when concrete. Under a higher-level trace (e.g.
    # BaseForecaster.conformity_scores's vmap) the loss array is a tracer and
    # np.asarray raises; in that path we return the traced array and let any
    # non-finite values surface downstream as NaN predictions.
    try:
        losses_host = np.asarray(losses)
    except jax.errors.TracerArrayConversionError:
        return losses
    bad = ~np.isfinite(losses_host)
    if bad.any():
        first_bad = int(np.argmax(bad))
        raise RuntimeError(
            f"Non-finite loss ({losses_host[first_bad]}) at step {first_bad}. "
            f"Training diverged. Consider lowering `learning_rate`, reducing "
            f"`windows_batch_size`, or checking the input series for extreme values."
        )
    return jnp.asarray(losses_host)


def predict_step(model: DilatedRNNNet, y: jnp.ndarray, *, h: int, input_size: int,
                 scaler: Scaler) -> jnp.ndarray:
    """Forecast next h steps from the final ``input_size`` of y. Returns (h,)."""
    insample = y[-input_size:][None, :]                          # [1, L]
    shift, scale = scaler.stats(insample, axis=1)                # [1, 1]
    x_z = scaler.transform(insample, shift, scale)[..., None]    # [1, L, 1]
    pred_z = _jit_forward_deterministic(model, x_z)              # [1, h, 1]
    return scaler.inverse(pred_z[..., 0], shift, scale)[0]

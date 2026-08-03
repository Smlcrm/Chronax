"""Window construction and JIT-compiled training/predict steps for SOFTSSharp.

The training loop is identical to the SOFTS port's: STADSharp carries its own
``nnx.Rngs`` reference, so its train-time multinomial-pooling key AND its
position-encoding Bernoulli key are threaded through the ``nnx.scan`` carry
automatically (same mechanism as ``nnx.Dropout``); no extra key plumbing is
needed in ``train`` beyond the window-sampling seed.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chronax.models.softssharp.softssharp_losses import LossFn, mae
from chronax.models.softssharp.softssharp_module import SOFTSSharpNet


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


def forward_loss(model: SOFTSSharpNet, windows: jnp.ndarray, *, h: int,
                 input_size: int, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Forward + point loss in ORIGINAL scale (RevIN denorms inside the net).

    windows: [B, input_size+h] -> scalar. The univariate series carries a channel
    dim of 1, so the insample window is reshaped to [B, L, 1].
    """
    insample = windows[:, :input_size][..., None]   # [B, L, 1]
    target = windows[:, input_size:]                # [B, h]
    pred = model(insample, deterministic=False)     # [B, h, 1]
    return loss_fn(pred[..., 0], target)


@nnx.jit
def _jit_forward_deterministic(model: SOFTSSharpNet, x: jnp.ndarray) -> jnp.ndarray:
    """Inference-mode forward (dropout disabled, deterministic core, position
    encoding applied at its expectation scale). Cached across calls."""
    return model(x, deterministic=True)


def train(model: SOFTSSharpNet, y: jnp.ndarray, *, h: int, input_size: int,
          max_steps: int, windows_batch_size: int, lr: optax.ScalarOrSchedule,
          seed: int, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Train ``model`` in place via a single ``nnx.scan``. Returns per-step losses.

    The whole loop is one ``nnx.scan`` (carry = (model, optimizer)), which keeps the
    function ``jax.vmap``-traceable for ``BaseForecaster.conformity_scores``. Window
    sampling replicates neuralforecast's REGIME-DEPENDENT scheme (``_base_model.py``
    training_step): when ``n_windows < windows_batch_size`` NF draws
    ``windows_batch_size`` indices WITH replacement (oversampling with duplicates —
    the regime small benchmark series hit, e.g. ~25 windows for AirlinePassengers at
    input_size=72, h=24, both << the batch size); otherwise it takes a
    without-replacement permutation of ``windows_batch_size`` windows. Getting this
    branch right is load-bearing for accuracy parity, so we do NOT collapse it to a
    full batch.

    STADSharp's stochastic pooling, its position-encoding gate, and the dropout
    layers all draw from the model's ``nnx.Rngs``; because the model is the scan
    carry, those key streams advance per step without any explicit threading here.

    Note: ``batches`` materializes a ``[max_steps, windows_batch_size, input_size+h]``
    tensor up front. At SOFTSSharp's defaults (windows_batch_size=32) this is small;
    if you raise ``windows_batch_size`` substantially, reduce ``max_steps`` or sample
    per-step instead to bound memory.
    """
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
            lambda m: forward_loss(m, batch_windows, h=h, input_size=input_size, loss_fn=loss_fn)
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


def predict_step(model: SOFTSSharpNet, y: jnp.ndarray, *, h: int, input_size: int) -> jnp.ndarray:
    """Forecast next h steps from the final ``input_size`` of y. Returns (h,)."""
    x = y[-input_size:][None, :, None]                 # [1, L, 1]
    pred = _jit_forward_deterministic(model, x)        # [1, h, 1]
    return pred[0, :, 0]

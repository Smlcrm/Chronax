"""Window construction and JIT-compiled training/predict steps for PatchTST."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chronax.models.patchtst.patchtst_losses import LossFn, mae
from chronax.models.patchtst.patchtst_module import PatchTSTNet


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


def forward_loss(model: PatchTSTNet, windows: jnp.ndarray, *, h: int,
                 input_size: int, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Forward + point loss in ORIGINAL scale (RevIN denorms inside the net).

    windows: [B, input_size+h] -> scalar.
    """
    insample = windows[:, :input_size][..., None]   # [B, L, 1]
    target = windows[:, input_size:]                # [B, h]
    pred = model(insample, deterministic=False, use_running_average=False)
    return loss_fn(pred[..., 0], target)


@nnx.jit
def _jit_forward_deterministic(model: PatchTSTNet, x: jnp.ndarray) -> jnp.ndarray:
    """Inference-mode forward with running BatchNorm stats. Cached across calls."""
    return model(x, deterministic=True, use_running_average=True)


def train(model: PatchTSTNet, y: jnp.ndarray, *, h: int, input_size: int,
          max_steps: int, windows_batch_size: int, lr: optax.ScalarOrSchedule,
          seed: int, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Train ``model`` in place via a single ``nnx.scan``. Returns per-step losses.

    The whole loop is one ``nnx.scan`` (carry = (model, optimizer)), which keeps
    the function ``jax.vmap``-traceable for ``BaseForecaster.conformity_scores``.
    Window sampling replicates neuralforecast's REGIME-DEPENDENT scheme
    (``_base_model.py`` training_step): when ``n_windows < windows_batch_size`` NF
    draws ``windows_batch_size`` indices WITH replacement (oversampling with
    duplicates — the regime every small benchmark series hits, e.g. ~24 windows
    for AirlinePassengers or ~245 for DailyFemaleBirths, both << 1024); otherwise
    it takes a without-replacement permutation of ``windows_batch_size`` windows.
    Getting this branch right is load-bearing for accuracy parity, so we do NOT
    collapse it to full-batch.

    Memory: only the int32 index tensor ``[max_steps, windows_batch_size]`` is
    pre-sampled (~40 MB at benchmark defaults); the windows themselves are
    gathered inside the scan step from the small ``windows`` array, avoiding a
    ~1.9 GB float32 materialization — a large peak-memory win on GPU/TPU HBM,
    neutral on CPU wall-clock.
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

    batch_idx = jax.vmap(sample_one)(step_keys)   # [max_steps, windows_batch_size] int32

    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    # Scan over the int32 index tensor and gather the windows INSIDE the step
    # (windows is a small closure constant), rather than pre-materializing a
    # ~1.9 GB [max_steps, windows_batch_size, input_size+h] float32 tensor.
    # Bit-identical; a large peak-memory win on GPU/TPU HBM, neutral on CPU.
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, idx):
        model, opt = carry
        batch_windows = windows[idx]
        loss, grads = nnx.value_and_grad(
            lambda m: forward_loss(m, batch_windows, h=h, input_size=input_size, loss_fn=loss_fn)
        )(model)
        opt.update(grads)
        return (model, opt), loss

    _, losses = step((model, optimizer), batch_idx)

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


def predict_step(model: PatchTSTNet, y: jnp.ndarray, *, h: int, input_size: int) -> jnp.ndarray:
    """Forecast next h steps from the final ``input_size`` of y. Returns (h,)."""
    x = y[-input_size:][None, :, None]                 # [1, L, 1]
    pred = _jit_forward_deterministic(model, x)        # [1, h, 1]
    return pred[0, :, 0]

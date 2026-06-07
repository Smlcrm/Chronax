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


def forward_loss(model, windows, *, h, input_size, loss_fn: LossFn = mae):
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


def train(model, y, *, h, input_size, max_steps, windows_batch_size, lr, seed,
          loss_fn: LossFn = mae):
    """Train ``model`` in place via a single ``nnx.scan``. Returns per-step losses.

    The whole loop is one ``nnx.scan`` (carry = (model, optimizer)), which keeps
    the function ``jax.vmap``-traceable for ``BaseForecaster.conformity_scores``.
    Window sampling replicates neuralforecast's REGIME-DEPENDENT scheme
    (``_base_model.py`` training_step): when ``n_windows < windows_batch_size`` NF
    draws ``windows_batch_size`` indices WITH replacement (oversampling with
    duplicates — the regime every small benchmark series hits, e.g. 25/246
    windows vs 1024); otherwise it takes a without-replacement permutation of
    ``windows_batch_size`` windows. Getting this branch right is load-bearing for
    accuracy parity, so we do NOT collapse it to full-batch.
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

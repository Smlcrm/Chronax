"""Window construction and training/predict steps for VanillaTransformer.

Unlike the iTransformer training loop, windows are sampled *inside* the
``nnx.scan`` (the per-step PRNG key is the scanned input), so memory stays
bounded to one batch even at NF defaults (``max_steps=5000``,
``windows_batch_size=1024``). The whole loop is one ``nnx.scan`` so it remains
``jax.vmap``-traceable for ``BaseForecaster.conformity_scores``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chronax.models.vanillatransformer.vanillatransformer_losses import LossFn, mae
from chronax.models.vanillatransformer.vanillatransformer_module import VanillaTransformerNet


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


def forward_loss(model: VanillaTransformerNet, windows: jnp.ndarray, *, h: int,
                 input_size: int, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Forward + point loss in raw scale. windows: [B, input_size+h] -> scalar."""
    insample = windows[:, :input_size][..., None]   # [B, L, 1]
    target = windows[:, input_size:]                # [B, h]
    pred = model(insample, deterministic=False)     # [B, h, 1]
    return loss_fn(pred[..., 0], target)


@nnx.jit
def _jit_forward_deterministic(model: VanillaTransformerNet, x: jnp.ndarray) -> jnp.ndarray:
    """Inference-mode forward (dropout disabled). Cached across calls."""
    return model(x, deterministic=True)


def train(model: VanillaTransformerNet, y: jnp.ndarray, *, h: int, input_size: int,
          max_steps: int, windows_batch_size: int, lr: optax.ScalarOrSchedule,
          seed: int, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Train ``model`` in place via a single ``nnx.scan``. Returns per-step losses.

    Window sampling replicates neuralforecast's regime-dependent scheme
    (``_base_model.py`` training_step): when ``n_windows < windows_batch_size``
    NF draws ``windows_batch_size`` indices WITH replacement; otherwise it takes a
    without-replacement permutation. The branch is chosen on concrete shapes
    (Python-static), so it bakes into the scan once. Sampling happens per step
    inside the scan to keep memory bounded to one batch.
    """
    windows = build_windows(y, input_size, h)
    n_windows = windows.shape[0]
    replace = n_windows < windows_batch_size

    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, step_key):
        model, opt = carry
        if replace:
            idx = jax.random.choice(step_key, n_windows, shape=(windows_batch_size,), replace=True)
        else:
            idx = jax.random.permutation(step_key, n_windows)[:windows_batch_size]
        batch = windows[idx]                         # [windows_batch_size, L+h]
        loss, grads = nnx.value_and_grad(
            lambda m: forward_loss(m, batch, h=h, input_size=input_size, loss_fn=loss_fn)
        )(model)
        opt.update(grads)
        return (model, opt), loss

    step_keys = jax.random.split(jax.random.PRNGKey(seed), max_steps)
    _, losses = step((model, optimizer), step_keys)

    # Finite check only when concrete. Under a higher-level vmap (conformity_scores)
    # losses is a tracer; return it and let non-finite values surface downstream.
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


def predict_step(model: VanillaTransformerNet, y: jnp.ndarray, *, h: int, input_size: int) -> jnp.ndarray:
    """Forecast next h steps from the final ``input_size`` of y. Returns (h,)."""
    x = y[-input_size:][None, :, None]                 # [1, L, 1]
    pred = _jit_forward_deterministic(model, x)        # [1, h, 1]
    return pred[0, :, 0]

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

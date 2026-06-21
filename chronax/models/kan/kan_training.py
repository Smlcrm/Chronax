"""Window construction and JIT-compiled training/predict steps for KAN."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chronax.models.kan.kan_losses import LossFn, mae
from chronax.models.kan.kan_module import KANNet
from chronax.models.kan.kan_scaler import Scaler


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> jnp.ndarray:
    """Return [n_windows, input_size+h] rolling windows; step=1."""
    window_size = input_size + h
    n = y.shape[0] - window_size + 1
    if n <= 0:
        raise ValueError(f"Series length {y.shape[0]} too short for input_size={input_size}, h={h}")
    idx = jnp.arange(window_size)[None, :] + jnp.arange(n)[:, None]
    return y[idx]


def scaled_forward_loss(model: KANNet, windows: jnp.ndarray, *, h: int, input_size: int,
                        scaler: Scaler, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Forward + point loss in SCALED space. windows: [B, input_size+h] -> scalar."""
    insample = windows[:, :input_size]
    target = windows[:, input_size:]
    shift, scale = scaler.stats(insample, axis=1)
    insample_z = scaler.transform(insample, shift, scale)
    target_z = scaler.transform(target, shift, scale)
    pred = model(insample_z[..., None])
    return loss_fn(pred[..., 0], target_z)


@nnx.jit
def _jit_forward(model: KANNet, x: jnp.ndarray) -> jnp.ndarray:
    return model(x)

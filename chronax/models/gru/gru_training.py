"""Window construction and JIT-compiled training/predict steps for the GRU."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from chronax.models.gru.gru_module import GRUNet
from chronax.models.gru.gru_scaler import RobustScaler, Scaler


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


def scaled_forward_loss(
    model: GRUNet,
    windows: jnp.ndarray,
    *,
    h: int,
    input_size: int,
    scaler: Scaler,
) -> jnp.ndarray:
    """MAE in scaled space, mirrors Nixtla. windows: [B, input_size+h] -> scalar."""
    insample = windows[:, :input_size]
    target = windows[:, input_size:]
    shift, scale = scaler.stats(insample, axis=1)
    insample_z = scaler.transform(insample, shift, scale)
    target_z = scaler.transform(target, shift, scale)
    pred = model(insample_z[..., None], deterministic=False)
    return jnp.mean(jnp.abs(pred[..., 0] - target_z))

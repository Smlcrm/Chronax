"""DLinear network: moving-average decomposition + two linear heads (NF-faithful).

``_series_decomp`` replicates NF's SeriesDecomp exactly: replicate-pad both ends
with (k-1)//2 edge copies, then an unpadded size-k moving mean (pool padding is 0
in NF, so the divisor is always k). Weight init replicates the *distribution* of
torch ``nn.Linear.reset_parameters`` for BOTH heads (draws differ — JAX vs torch
RNG): U(+-1/sqrt(input_size)) for weights and biases, torch layout
(``[h, input_size]``) so NF weight transplant is a pure copy. float32 throughout,
matching torch/neuralforecast defaults.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


def _series_decomp(y: jnp.ndarray, kernel_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """NF SeriesDecomp: (trend, seasonal) for ``y: [B, L]`` -> each ``[B, L]``."""
    pad = (kernel_size - 1) // 2
    y_pad = jnp.pad(y, ((0, 0), (pad, pad)), mode="edge")
    trend = jax.lax.reduce_window(
        y_pad, 0.0, jax.lax.add,
        window_dimensions=(1, kernel_size), window_strides=(1, 1), padding="VALID",
    ) / kernel_size
    return trend, y - trend


class DLinearNet(nnx.Module):
    """``forecast = trend @ Wt.T + bt + seasonal @ Ws.T + bs`` (no last-value
    add-back — that is NLinear's trick). I/O ``[B, L, 1] -> [B, h, 1]``."""

    def __init__(self, h: int, input_size: int, moving_avg_window: int, *, rngs: nnx.Rngs):
        self.h = h
        self.input_size = input_size
        self.moving_avg_window = moving_avg_window
        bound = 1.0 / math.sqrt(input_size)
        kt, kbt, ks, kbs = jax.random.split(rngs.params(), 4)

        def u(k: jax.Array, shape: tuple) -> jnp.ndarray:
            return jax.random.uniform(k, shape, minval=-bound, maxval=bound, dtype=jnp.float32)

        self.w_trend = nnx.Param(u(kt, (h, input_size)))
        self.b_trend = nnx.Param(u(kbt, (h,)))
        self.w_season = nnx.Param(u(ks, (h, input_size)))
        self.b_season = nnx.Param(u(kbs, (h,)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = x[..., 0]                                    # [B, L]
        trend, seasonal = _series_decomp(y, self.moving_avg_window)
        out = (trend @ self.w_trend.value.T + self.b_trend.value
               + seasonal @ self.w_season.value.T + self.b_season.value)
        return out[..., None]                            # [B, h, 1]

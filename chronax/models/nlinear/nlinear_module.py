"""NLinear network: one linear layer with last-value normalization (NF-faithful).

Weight init replicates the *distribution* of torch ``nn.Linear.reset_parameters``
(draws differ — JAX vs torch RNG): kaiming-uniform (a=sqrt(5)) on the weight —
equivalent to U(+-1/sqrt(fan_in)) — and U(+-1/sqrt(fan_in)) on the bias. Params
are stored in TORCH layout (weight ``[h, input_size]``) so NF weight transplant
is a pure copy. float32 throughout, matching torch/neuralforecast defaults.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class NLinearNet(nnx.Module):
    """``forecast = (y - y_last) @ W.T + b + y_last``. I/O ``[B, L, 1] -> [B, h, 1]``."""

    def __init__(self, h: int, input_size: int, *, rngs: nnx.Rngs):
        self.h = h
        self.input_size = input_size
        bound = 1.0 / math.sqrt(input_size)
        kw, kb = jax.random.split(rngs.params())
        self.weight = nnx.Param(jax.random.uniform(
            kw, (h, input_size), minval=-bound, maxval=bound, dtype=jnp.float32))
        self.bias = nnx.Param(jax.random.uniform(
            kb, (h,), minval=-bound, maxval=bound, dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = x[..., 0]                                   # [B, L]
        last = y[:, -1:]                                # [B, 1]  (the "N" trick)
        out = (y - last) @ self.weight.value.T + self.bias.value + last
        return out[..., None]                           # [B, h, 1]

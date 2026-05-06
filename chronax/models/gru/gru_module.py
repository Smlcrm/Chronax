"""Flax NNX modules for the GRU forecaster: encoder, decoder, full network."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def pytorch_uniform_init(hidden_size: int):
    """Match PyTorch nn.GRU / nn.Linear default: Uniform(-1/sqrt(H), 1/sqrt(H))."""
    bound = float(1.0 / jnp.sqrt(jnp.asarray(hidden_size, dtype=jnp.float32)))

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init

"""XLinear network: RevIN + projection + global token + dual gating + head (NF-faithful).

Univariate (n_series=1) adaptation of neuralforecast.XLinear's forward. Weight
init replicates the *distribution* of torch defaults per layer (draws differ —
JAX vs torch RNG): U(+-1/sqrt(fan_in)) for every Linear's weight AND bias, with
each layer's own fan_in; the global token is standard normal (torch.randn).
Params are stored in TORCH layout ([out, in]) so NF weight transplant is a pure
copy. float32 throughout, matching torch/neuralforecast defaults.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


def _revin(y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """NF ``use_norm`` block for ``y: [B, L]`` -> (y_norm, mean, stdev), stats [B, 1].

    Biased variance, eps 1e-5 INSIDE the sqrt, stats under stop_gradient
    (replicates torch ``.detach()``; a no-op for parameter gradients since the
    stats depend only on the data, kept for exactness of intent).
    """
    mean = jnp.mean(y, axis=1, keepdims=True)
    var = jnp.mean((y - mean) ** 2, axis=1, keepdims=True)
    stdev = jnp.sqrt(var + 1e-5)
    mean = jax.lax.stop_gradient(mean)
    stdev = jax.lax.stop_gradient(stdev)
    return (y - mean) / stdev, mean, stdev


def _gating(x: jnp.ndarray, w1: jnp.ndarray, b1: jnp.ndarray,
            w2: jnp.ndarray, b2: jnp.ndarray) -> jnp.ndarray:
    """NF GatingBlock at dropout p=0: ``x * sigmoid(relu(x @ w1.T + b1) @ w2.T + b2)``."""
    hidden = jax.nn.relu(x @ w1.T + b1)
    return x * jax.nn.sigmoid(hidden @ w2.T + b2)


class XLinearNet(nnx.Module):
    """Forward per NF XLinear at n_series=1, no exog. I/O ``[B, L, 1] -> [B, h, 1]``."""

    def __init__(self, h: int, input_size: int, hidden_size: int, temporal_ff: int,
                 channel_ff: int, use_norm: bool, *, rngs: nnx.Rngs):
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.temporal_ff = temporal_ff
        self.channel_ff = channel_ff
        self.use_norm = use_norm
        keys = jax.random.split(rngs.params(), 13)

        def lin(kw: jax.Array, kb: jax.Array, out_f: int, in_f: int) -> tuple[nnx.Param, nnx.Param]:
            bound = 1.0 / math.sqrt(in_f)
            w = jax.random.uniform(kw, (out_f, in_f), minval=-bound, maxval=bound, dtype=jnp.float32)
            b = jax.random.uniform(kb, (out_f,), minval=-bound, maxval=bound, dtype=jnp.float32)
            return nnx.Param(w), nnx.Param(b)

        self.w_proj, self.b_proj = lin(keys[0], keys[1], hidden_size, input_size)
        self.glob_token = nnx.Param(jax.random.normal(keys[2], (1, 1, hidden_size), dtype=jnp.float32))
        self.w_tg1, self.b_tg1 = lin(keys[3], keys[4], temporal_ff, 2 * hidden_size)
        self.w_tg2, self.b_tg2 = lin(keys[5], keys[6], 2 * hidden_size, temporal_ff)
        self.w_cg1, self.b_cg1 = lin(keys[7], keys[8], channel_ff, 2)   # 2 == 2*n_series at N=1
        self.w_cg2, self.b_cg2 = lin(keys[9], keys[10], 2, channel_ff)
        self.w_head, self.b_head = lin(keys[11], keys[12], h, 2 * hidden_size)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = x[..., 0]                                                  # [B, L]
        if self.use_norm:
            yn, mean, stdev = _revin(y)
        else:
            yn = y
        emb = yn @ self.w_proj.value.T + self.b_proj.value             # [B, H]
        glob = jnp.broadcast_to(self.glob_token.value[0, 0], emb.shape)
        en_emb = jnp.concatenate([emb, glob], axis=-1)                 # [B, 2H]
        en_atten = _gating(en_emb, self.w_tg1.value, self.b_tg1.value,
                           self.w_tg2.value, self.b_tg2.value)
        origin_atten = en_atten[:, :self.hidden_size]                  # [B, H]
        glob_atten = en_atten[:, self.hidden_size:]                    # [B, H]
        # cross-channel: channels are [emb, glob_atten]; gate over the channel dim
        ex = jnp.stack([emb, glob_atten], axis=1)                      # [B, 2, H]
        ex_atten = _gating(jnp.transpose(ex, (0, 2, 1)),               # [B, H, 2]
                           self.w_cg1.value, self.b_cg1.value,
                           self.w_cg2.value, self.b_cg2.value)
        glob2 = ex_atten[:, :, 1]                                      # [B, H]  (channels[n_series:])
        en = jnp.concatenate([origin_atten, glob2], axis=-1)           # [B, 2H]
        out = en @ self.w_head.value.T + self.b_head.value             # [B, h]
        if self.use_norm:
            out = out * stdev + mean
        return out[..., None]                                          # [B, h, 1]

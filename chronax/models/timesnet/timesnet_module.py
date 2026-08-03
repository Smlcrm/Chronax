"""TimesNet network: data-embedding + static-period TimesBlocks (NF-faithful forward).

Univariate port of neuralforecast.TimesNet's forward with STATIC per-fit periods
(see timesnet_model docstring for the disclosed period-selection divergence).
Inits replicate torch distributions per layer (draws differ): token conv
kaiming-normal fan_in/leaky_relu (std=sqrt(2/3), no bias); Inception Conv2d
kaiming-normal fan_out/relu (std=sqrt(2/(C_out*k^2))), zero bias; Linears
U(+-1/sqrt(fan_in)) w+b; ONE shared LayerNorm (gamma=1, beta=0) applied after
every TimesBlock, exactly as NF's single `layer_norm`. float32 throughout.
Convs use lax.conv_general_dilated (cross-correlation — torch semantics).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


def _positional_encoding(length: int, hidden_size: int) -> jnp.ndarray:
    """NF PositionalEmbedding: fixed sinusoidal buffer [1, length, hidden_size]."""
    position = jnp.arange(length, dtype=jnp.float32)[:, None]
    div = jnp.exp(jnp.arange(0, hidden_size, 2, dtype=jnp.float32)
                  * -(math.log(10000.0) / hidden_size))
    pe = jnp.zeros((length, hidden_size), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div))
    return pe[None]


def _token_embed(y: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """NF TokenEmbedding: circular Conv1d(k=3, no bias). y [B, T] -> [B, T, hidden].

    Torch 'circular' padding == jnp.pad(mode='wrap') then VALID conv;
    w is torch layout [hidden, 1, 3]."""
    x = jnp.pad(y[:, None, :], ((0, 0), (0, 0), (1, 1)), mode="wrap")  # [B, 1, T+2]
    out = jax.lax.conv_general_dilated(
        x, w, window_strides=(1,), padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"))
    return jnp.transpose(out, (0, 2, 1))                               # [B, T, hidden]


def _inception(x: jnp.ndarray, ws: list, bs: list) -> jnp.ndarray:
    """NF Inception_Block_V1: mean of parallel SAME Conv2d (kernels 2i+1). NCHW."""
    outs = []
    for w, b in zip(ws, bs):
        o = jax.lax.conv_general_dilated(
            x, w, window_strides=(1, 1), padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"))
        outs.append(o + b[None, :, None, None])
    return sum(outs) / len(outs)


def _layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """torch LayerNorm over the last dim (eps 1e-5, biased variance)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5) * scale + bias


class TimesNetNet(nnx.Module):
    """Forward per NF TimesNet (univariate, no exog) at STATIC periods/freqs.

    ``periods``/``freqs`` are parallel static tuples chosen at fit time.
    I/O ``[B, L, 1] -> [B, h, 1]``; ``dropout_key=None`` => eval (no dropout)."""

    def __init__(self, h: int, input_size: int, hidden_size: int, conv_hidden_size: int,
                 top_k: int, num_kernels: int, encoder_layers: int, dropout: float,
                 periods: tuple[int, ...], freqs: tuple[int, ...], *, rngs: nnx.Rngs):
        T = input_size + h
        # periods/freqs are parallel (period = T // freq); each freq must index a
        # nonzero rfft bin, else the in-graph `amp[:, freqs]` gather silently
        # clamps to a wrong index instead of erroring. The wrapper's _compute_periods
        # guarantees this; validate so a hand-built net fails loudly, not numerically.
        if len(periods) != len(freqs) or len(freqs) != top_k:
            raise ValueError(f"periods/freqs must be parallel tuples of length top_k={top_k}; "
                             f"got len(periods)={len(periods)}, len(freqs)={len(freqs)}.")
        if any(not (1 <= int(f) <= T // 2) for f in freqs):
            raise ValueError(f"every freq must be in [1, {T // 2}] (nonzero rfft bins of a "
                             f"length-{T} window); got freqs={tuple(int(f) for f in freqs)}.")
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        self.periods = tuple(int(p) for p in periods)
        self.freqs = tuple(int(f) for f in freqs)
        n_lin = 2                                   # predict_linear + projection
        # keys: 1 token conv + 2 per linear + 1 per conv2d (bias is zero-init, no draw)
        keys = jax.random.split(rngs.params(), 1 + 2 * n_lin + 2 * encoder_layers * num_kernels)
        ki = iter(range(len(keys)))

        def nk():
            return keys[next(ki)]

        # token embedding conv: [hidden, 1, 3], kaiming-normal fan_in(=3) leaky_relu -> std sqrt(2/3)
        self.w_token = nnx.Param(jax.random.normal(nk(), (hidden_size, 1, 3), dtype=jnp.float32)
                                 * math.sqrt(2.0 / 3.0))

        def lin(out_f: int, in_f: int) -> tuple[nnx.Param, nnx.Param]:
            bound = 1.0 / math.sqrt(in_f)
            w = jax.random.uniform(nk(), (out_f, in_f), minval=-bound, maxval=bound, dtype=jnp.float32)
            b = jax.random.uniform(nk(), (out_f,), minval=-bound, maxval=bound, dtype=jnp.float32)
            return nnx.Param(w), nnx.Param(b)

        self.w_predict, self.b_predict = lin(T, input_size)        # predict_linear over TIME
        self.w_proj, self.b_proj = lin(1, hidden_size)             # projection

        def conv2d(out_c: int, in_c: int, k: int) -> tuple[nnx.Param, nnx.Param]:
            # kaiming-normal fan_out relu: std = sqrt(2 / (out_c * k * k)); zero bias
            std = math.sqrt(2.0 / (out_c * k * k))
            w = jax.random.normal(nk(), (out_c, in_c, k, k), dtype=jnp.float32) * std
            return nnx.Param(w), nnx.Param(jnp.zeros((out_c,), dtype=jnp.float32))

        # per encoder layer: inception A (hidden->conv_hidden), inception B (conv_hidden->hidden)
        self.conv_ws = []
        self.conv_bs = []
        for _ in range(encoder_layers):
            layer_ws, layer_bs = [], []
            for (ci, co) in ((hidden_size, conv_hidden_size), (conv_hidden_size, hidden_size)):
                ws, bs = [], []
                for i in range(num_kernels):
                    w, b = conv2d(co, ci, 2 * i + 1)
                    ws.append(w); bs.append(b)
                layer_ws.append(ws); layer_bs.append(bs)
            self.conv_ws.append(layer_ws)
            self.conv_bs.append(layer_bs)
        # ONE shared LayerNorm (NF: single self.layer_norm reused after every block)
        self.ln_scale = nnx.Param(jnp.ones((hidden_size,), dtype=jnp.float32))
        self.ln_bias = nnx.Param(jnp.zeros((hidden_size,), dtype=jnp.float32))

    def _times_block(self, x: jnp.ndarray, layer: int) -> jnp.ndarray:
        """One TimesBlock at static periods. x [B, T, C] -> [B, T, C]."""
        B, T, C = x.shape
        # in-graph per-sample weights at the static freqs (NF formula)
        xf = jnp.fft.rfft(x, axis=1)
        amp = jnp.abs(xf).mean(axis=-1)                      # [B, T//2+1]
        weights = amp[:, jnp.asarray(self.freqs)]            # [B, k]
        weights = jax.nn.softmax(weights, axis=1)
        res = []
        for p in self.periods:
            rows = -(-T // p)                                # ceil
            pad = rows * p - T
            xp = jnp.pad(x, ((0, 0), (0, pad), (0, 0))) if pad else x
            grid = jnp.transpose(xp.reshape(B, rows, p, C), (0, 3, 1, 2))  # [B, C, rows, p]
            ws_a, ws_b = self.conv_ws[layer]
            bs_a, bs_b = self.conv_bs[layer]
            o = _inception(grid, [w.value for w in ws_a], [b.value for b in bs_a])
            o = jax.nn.gelu(o, approximate=False)
            o = _inception(o, [w.value for w in ws_b], [b.value for b in bs_b])
            back = jnp.transpose(o, (0, 2, 3, 1)).reshape(B, rows * p, C)[:, :T]
            res.append(back)
        stacked = jnp.stack(res, axis=-1)                    # [B, T, C, k]
        out = jnp.sum(stacked * weights[:, None, None, :], axis=-1)
        return out + x

    def __call__(self, x: jnp.ndarray, *, dropout_key: jax.Array | None = None) -> jnp.ndarray:
        y = x[..., 0]                                        # [B, L]
        emb = _token_embed(y, self.w_token.value)            # [B, L, hidden]
        emb = emb + _positional_encoding(self.input_size, self.hidden_size)
        if dropout_key is not None and self.dropout > 0.0:
            keep = 1.0 - self.dropout
            mask = jax.random.bernoulli(dropout_key, keep, emb.shape)
            emb = jnp.where(mask, emb / keep, 0.0)
        # predict_linear over the TIME axis: [B, L, C] -> [B, C, L] @ W.T -> [B, C, T]
        ext = jnp.transpose(emb, (0, 2, 1)) @ self.w_predict.value.T + self.b_predict.value
        out = jnp.transpose(ext, (0, 2, 1))                  # [B, T, hidden]
        for layer in range(self.encoder_layers):
            out = _layer_norm(self._times_block(out, layer),
                              self.ln_scale.value, self.ln_bias.value)
        dec = out @ self.w_proj.value.T + self.b_proj.value  # [B, T, 1]
        return dec[:, -self.h:, :]                           # [B, h, 1]

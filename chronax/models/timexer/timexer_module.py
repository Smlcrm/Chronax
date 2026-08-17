"""Flax NNX modules for the TimeXer forecaster (port of neuralforecast.TimeXer).

TimeXer splits the endogenous window into patches embedded as tokens, appends
one learnable GLOBAL token per variate, and runs encoder layers in which patch
tokens self-attend while ONLY the global token cross-attends to variate-level
embeddings of the full window (the reference's exogenous pathway; without
exogenous inputs the cross context is the endogenous variates themselves).
A flatten head maps each variate's token stack onto the horizon, wrapped in
non-stationary normalization (per-window mean/std, statistics carry no
parameter dependence so no stop-gradient is needed for gradient equivalence
with the reference's detach).

The reference's position-wise FFN uses kernel-size-1 convolutions — pointwise
Linears here, mathematically identical. ``float32`` throughout.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _TorchLinearInit:
    """Picklable initializer matching torch ``nn.Linear`` default:
    ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` for weight and bias."""

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    def __eq__(self, other):
        # Value equality keeps same-config graphdefs cache-equal across refits
        # for the module-level @nnx.jit inference forward.
        return type(other) is type(self) and other.fan_in == self.fan_in

    def __hash__(self):
        return hash((type(self), self.fan_in))


def _torch_linear(n_in: int, n_out: int, *, use_bias: bool = True, rngs: nnx.Rngs) -> nnx.Linear:
    init = _TorchLinearInit(n_in)
    return nnx.Linear(n_in, n_out, use_bias=use_bias,
                      kernel_init=init, bias_init=init, rngs=rngs)


def _positional_embedding(length: int, hidden_size: int) -> jnp.ndarray:
    """Fixed sinusoidal positional embedding ``[1, length, hidden_size]``:
    ``sin`` on even channels, ``cos`` on odd. Constant, no parameters."""
    position = jnp.arange(length, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(
        jnp.arange(0, hidden_size, 2, dtype=jnp.float32)
        * -(math.log(10000.0) / hidden_size)
    )
    pe = jnp.zeros((length, hidden_size), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe[None]


class EnEmbedding(nnx.Module):
    """Endogenous patch embedding with a learnable per-variate global token.

    ``x [B, N, L]`` is cut into ``L // patch_len`` non-overlapping patches (any
    tail shorter than a patch is dropped, as in the reference's unfold), each
    projected by a bias-free Linear and shifted by the sinusoidal positional
    code of its patch index; the global token is appended LAST. Returns
    ``([B * N, patch_num + 1, d_model], n_vars)``.
    """

    def __init__(self, n_vars: int, d_model: int, patch_len: int, dropout: float,
                 *, rngs: nnx.Rngs):
        self.patch_len = patch_len
        self.d_model = d_model
        self.value_embedding = _torch_linear(patch_len, d_model, use_bias=False, rngs=rngs)
        self.glb_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, n_vars, 1, d_model), jnp.float32))
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> tuple[jnp.ndarray, int]:
        B, n_vars, L = x.shape
        pn = L // self.patch_len
        patches = x[:, :, : pn * self.patch_len].reshape(B, n_vars, pn, self.patch_len)
        e = self.value_embedding(patches)                     # [B, N, pn, d]
        e = e + _positional_embedding(pn, self.d_model)[None]
        glb = jnp.broadcast_to(self.glb_token.value, (B, n_vars, 1, self.d_model))
        e = jnp.concatenate([e, glb], axis=2)                 # [B, N, pn+1, d]
        e = e.reshape(B * n_vars, pn + 1, self.d_model)
        return self.dropout(e, deterministic=deterministic), n_vars


class DataEmbeddingInverted(nnx.Module):
    """Variate-as-token embedding: each variate's full window is one token,
    projected time -> hidden. ``x [B, L, N] -> [B, N, d_model]``."""

    def __init__(self, c_in: int, d_model: int, dropout: float, *, rngs: nnx.Rngs):
        self.value_embedding = _torch_linear(c_in, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        e = self.value_embedding(jnp.transpose(x, (0, 2, 1)))
        return self.dropout(e, deterministic=deterministic)


class AttentionLayer(nnx.Module):
    """Multi-head full (softmax dot-product) attention, self- or cross-.

    Folds the reference ``AttentionLayer`` + ``FullAttention`` (no mask,
    weights-dropout, scale ``d_k ** -0.5``). ``q`` from ``q_in``; ``k``/``v``
    from ``kv_in`` (equal for self-attention). torch ``nn.Linear`` init.
    """

    def __init__(self, *, hidden_size: int, n_heads: int, attn_dropout: float, rngs: nnx.Rngs):
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})."
            )
        self.n_heads = n_heads
        self.d_k = hidden_size // n_heads
        self.scale = float(self.d_k ** -0.5)
        self.w_q = _torch_linear(hidden_size, hidden_size, rngs=rngs)
        self.w_k = _torch_linear(hidden_size, hidden_size, rngs=rngs)
        self.w_v = _torch_linear(hidden_size, hidden_size, rngs=rngs)
        self.w_o = _torch_linear(hidden_size, hidden_size, rngs=rngs)
        self.attn_dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)

    def _split(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def __call__(self, q_in, kv_in, *, deterministic: bool):
        q = self._split(self.w_q(q_in))
        k = self._split(self.w_k(kv_in))
        v = self._split(self.w_v(kv_in))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, deterministic=deterministic)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        B, _, Lq, _ = ctx.shape
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, Lq, self.n_heads * self.d_k)
        return self.w_o(ctx)


class TimeXerEncoderLayer(nnx.Module):
    """One TimeXer encoder layer: patch-token self-attention, global-token-only
    cross-attention into the variate context, position-wise FFN.

    ``x [B * n_vars, pn + 1, d]`` with the global token LAST; ``cross`` is
    ``[B, n_cross_tokens, d]``. Only the global token row reads the cross
    context inside a layer — patch tokens see it through later layers.
    """

    def __init__(self, *, hidden_size: int, n_heads: int, d_ff: int, dropout: float,
                 rngs: nnx.Rngs):
        self.self_attention = AttentionLayer(hidden_size=hidden_size, n_heads=n_heads,
                                             attn_dropout=dropout, rngs=rngs)
        self.cross_attention = AttentionLayer(hidden_size=hidden_size, n_heads=n_heads,
                                              attn_dropout=dropout, rngs=rngs)
        self.ffn_w1 = _torch_linear(hidden_size, d_ff, rngs=rngs)   # reference conv1, k=1
        self.ffn_w2 = _torch_linear(d_ff, hidden_size, rngs=rngs)   # reference conv2, k=1
        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.norm3 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, cross: jnp.ndarray, *, n_vars: int,
                 deterministic: bool) -> jnp.ndarray:
        d = x.shape[-1]
        B = cross.shape[0]
        x = x + self.dropout(self.self_attention(x, x, deterministic=deterministic),
                             deterministic=deterministic)
        x = self.norm1(x)

        x_glb_ori = x[:, -1:, :]                              # [B*N, 1, d]
        x_glb = x_glb_ori.reshape(B, -1, d)                   # [B, N, d]
        x_glb_attn = self.dropout(
            self.cross_attention(x_glb, cross, deterministic=deterministic),
            deterministic=deterministic)
        x_glb_attn = x_glb_attn.reshape(B * n_vars, d)[:, None, :]
        x_glb = self.norm2(x_glb_ori + x_glb_attn)

        x = jnp.concatenate([x[:, :-1, :], x_glb], axis=1)
        y = self.dropout(jax.nn.relu(self.ffn_w1(x)), deterministic=deterministic)
        y = self.dropout(self.ffn_w2(y), deterministic=deterministic)
        return self.norm3(x + y)


class TimeXerNet(nnx.Module):
    """Full TimeXer: NS-norm -> patch + variate embeddings -> encoder stack ->
    per-variate flatten head -> NS-denorm.

    ``__call__(insample_y [B, L, N], deterministic) -> [B, h, N * mult]``
    (the reference reshapes its ``[B, h * mult, N]`` head output with
    ``reshape(B, h, -1)`` — kept verbatim).
    """

    def __init__(self, *, h: int, input_size: int, n_series: int, patch_len: int = 16,
                 hidden_size: int = 512, n_heads: int = 8, e_layers: int = 2,
                 d_ff: int = 2048, dropout: float = 0.1, use_norm: bool = True,
                 outputsize_multiplier: int = 1, rngs: nnx.Rngs):
        if input_size < patch_len:
            raise ValueError(
                f"input_size ({input_size}) must be >= patch_len ({patch_len}); "
                "the patch embedding needs at least one full patch."
            )
        self.h = h
        self.input_size = input_size
        self.n_series = n_series
        self.patch_len = patch_len
        self.hidden_size = hidden_size
        self.use_norm = use_norm
        self.outputsize_multiplier = outputsize_multiplier
        self.patch_num = input_size // patch_len

        self.en_embedding = EnEmbedding(n_series, hidden_size, patch_len, dropout, rngs=rngs)
        self.ex_embedding = DataEmbeddingInverted(input_size, hidden_size, dropout, rngs=rngs)
        self.layers = [
            TimeXerEncoderLayer(hidden_size=hidden_size, n_heads=n_heads, d_ff=d_ff,
                                dropout=dropout, rngs=rngs)
            for _ in range(e_layers)
        ]
        self.final_norm = nnx.LayerNorm(hidden_size, rngs=rngs)
        head_nf = hidden_size * (self.patch_num + 1)
        self.head = _torch_linear(head_nf, h * outputsize_multiplier, rngs=rngs)
        self.head_dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, insample_y: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        x = insample_y.astype(jnp.float32)                    # [B, L, N]
        B, _, N = x.shape
        if self.use_norm:
            means = jnp.mean(x, axis=1, keepdims=True)
            x = x - means
            stdev = jnp.sqrt(jnp.var(x, axis=1, keepdims=True) + 1e-5)
            x = x / stdev

        en, n_vars = self.en_embedding(jnp.transpose(x, (0, 2, 1)),
                                       deterministic=deterministic)
        ex = self.ex_embedding(x, deterministic=deterministic)  # [B, N, d]

        out = en
        for layer in self.layers:
            out = layer(out, ex, n_vars=n_vars, deterministic=deterministic)
        out = self.final_norm(out)

        out = out.reshape(B, n_vars, self.patch_num + 1, self.hidden_size)
        out = jnp.transpose(out, (0, 1, 3, 2))                # [B, N, d, pn+1]
        flat = out.reshape(B, n_vars, self.hidden_size * (self.patch_num + 1))
        dec = self.head_dropout(self.head(flat), deterministic=deterministic)
        dec = jnp.transpose(dec, (0, 2, 1))                   # [B, h*mult, N]

        if self.use_norm:
            dec = dec * stdev[:, 0, :][:, None, :] + means[:, 0, :][:, None, :]
        return dec.reshape(B, self.h, -1)

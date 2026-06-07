"""Flax NNX modules for the PatchTST forecaster.

Univariate port of neuralforecast.PatchTST. The series carries a channel
dimension of size 1 (``c_in = 1``); RevIN operates on ``[B, L, 1]`` and the
transformer encoder operates on ``[B, patch_num, hidden_size]``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class RevIN(nnx.Module):
    """Reversible instance normalization (Kim et al. 2022), per-window.

    Faithful to neuralforecast's RevIN at PatchTST's defaults: centers on the
    last timestep (``subtract_last=True``), divides by ``sqrt(var + eps)`` with
    population variance (``ddof=0``), and applies no learnable affine
    (``affine=False``). Statistics are returned explicitly rather than cached,
    so the module is pure and vmap/scan-safe.
    """

    def __init__(
        self,
        num_features: int,
        *,
        subtract_last: bool = True,
        affine: bool = False,
        eps: float = 1e-5,
        rngs: nnx.Rngs,
    ):
        self.num_features = num_features
        self.subtract_last = subtract_last
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nnx.Param(jnp.ones((num_features,), dtype=jnp.float32))
            self.beta = nnx.Param(jnp.zeros((num_features,), dtype=jnp.float32))

    def norm(self, x: jnp.ndarray):
        """x: [B, L, C] -> (z: [B, L, C], loc: [B, 1, C], scale: [B, 1, C])."""
        x = x.astype(jnp.float32)
        if self.subtract_last:
            loc = x[:, -1:, :]
        else:
            loc = jnp.mean(x, axis=1, keepdims=True)
        var = jnp.var(x, axis=1, keepdims=True)  # ddof=0 (population)
        scale = jnp.sqrt(var + self.eps)
        z = (x - loc) / scale
        if self.affine:
            z = z * self.gamma.value[None, None, :] + self.beta.value[None, None, :]
        return z, loc, scale

    def denorm(self, z: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        """Invert norm. z: [B, T, C] with broadcastable loc/scale [B, 1, C]."""
        if self.affine:
            z = (z - self.beta.value[None, None, :]) / (self.gamma.value[None, None, :] + self.eps * self.eps)
        return z * scale + loc


def compute_patch_num(input_size: int, patch_len: int, stride: int) -> int:
    """Number of patches after end-padding by ``stride`` (NF padding_patch='end').

    Mirrors NF's ``int((input_size - patch_len) / stride + 1) + 1`` exactly
    (truncation toward zero, not Python floor) so it stays correct even if called
    with an unclamped ``patch_len``; callers normally pass the clamped
    ``patch_len = min(input_size + stride, patch_len)``.
    """
    return int((input_size - patch_len) / stride + 1) + 1


def patchify(x: jnp.ndarray, *, patch_len: int, stride: int) -> jnp.ndarray:
    """x: [B, L] -> patches: [B, patch_num, patch_len].

    Replicates torch ``ReplicationPad1d((0, stride))`` then
    ``unfold(dim=-1, size=patch_len, step=stride)``.
    """
    x = x.astype(jnp.float32)
    orig_L = x.shape[1]                                 # input_size, before padding
    x = jnp.pad(x, ((0, 0), (0, stride)), mode="edge")  # end padding
    n = compute_patch_num(orig_L, patch_len, stride)
    starts = jnp.arange(n) * stride
    offs = jnp.arange(patch_len)
    idx = starts[:, None] + offs[None, :]               # [patch_num, patch_len]
    return x[:, idx]                                     # [B, patch_num, patch_len]


class _UniformPos:
    """Picklable Uniform(-0.02, 0.02) initializer for the positional encoding."""

    __slots__ = ()

    def __call__(self, key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-0.02, maxval=0.02)


class PatchEmbedding(nnx.Module):
    """Linear patch projection + learnable positional encoding + residual dropout.

    Mirrors NF's ``W_P`` (``Linear(patch_len -> hidden_size)``) plus
    ``positional_encoding(pe='zeros', learn_pe=True)`` — a learnable
    ``[patch_num, hidden_size]`` parameter initialized ``Uniform(-0.02, 0.02)``.
    Receives pre-cut patches; patchify lives in ``PatchTSTNet``.
    """

    def __init__(self, *, patch_len, hidden_size, patch_num, dropout, rngs: nnx.Rngs):
        self.proj = nnx.Linear(patch_len, hidden_size, rngs=rngs)
        key = rngs.params()
        self.pos = nnx.Param(_UniformPos()(key, (patch_num, hidden_size)))
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, patches: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """patches: [B, patch_num, patch_len] -> tokens: [B, patch_num, hidden_size]."""
        z = self.proj(patches.astype(jnp.float32))
        z = z + self.pos.value[None, :, :]
        return self.dropout(z, deterministic=deterministic)


class MultiHeadAttention(nnx.Module):
    """Multi-head self-attention with Realformer residual-attention threading.

    Faithful to NF's ``res_attention=True`` path: pre-softmax scores are added
    across layers via ``prev`` and the scaling factor ``d_k**-0.5`` is a frozen
    constant (NF uses ``lsa=False``). Hand-rolled (not
    ``jax.nn.dot_product_attention``) because that fused path only matches NF's
    ``res_attention=False`` branch.
    """

    def __init__(self, *, hidden_size, n_heads, attn_dropout, proj_dropout, rngs: nnx.Rngs):
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.d_k = hidden_size // n_heads
        self.scale = float(self.d_k ** -0.5)
        self.w_q = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.w_k = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.w_v = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.w_o = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.attn_dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)
        self.proj_dropout = nnx.Dropout(rate=proj_dropout, rngs=rngs)

    def _split(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)  # [B, H, T, d_k]

    def __call__(self, x, prev, deterministic: bool):
        """x: [B, T, hidden] -> (out: [B, T, hidden], scores: [B, H, T, T])."""
        q = self._split(self.w_q(x))
        k = self._split(self.w_k(x))
        v = self._split(self.w_v(x))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        if prev is not None:
            scores = scores + prev
        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, deterministic=deterministic)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, v)  # [B, H, T, d_k]
        B, _, T, _ = ctx.shape
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.d_k)
        out = self.proj_dropout(self.w_o(ctx), deterministic=deterministic)
        return out, scores


def _resolve_activation(name: str):
    """Map an activation name to a callable. NF uses EXACT (erf) GELU by default —
    ``jax.nn.gelu`` defaults to the tanh approximation, which diverges ~1e-3 per
    call and would fail the weight-parity gate, so pass ``approximate=False``.
    """
    if name == "gelu":
        return lambda z: jax.nn.gelu(z, approximate=False)  # == torch nn.GELU()
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unknown activation {name!r}. Available: 'gelu', 'relu'.")


class TSTEncoderLayer(nnx.Module):
    """One transformer encoder layer: residual MHA + BatchNorm, then FFN + BatchNorm.

    Post-norm (NF default ``pre_norm=False``). BatchNorm with ``axis=-1`` over
    ``[B, patch_num, hidden]`` matches NF's transpose/BatchNorm1d sandwich;
    ``momentum=0.9`` matches torch BatchNorm1d's ``momentum=0.1`` running-stat
    decay. (flax updates running-var with the biased/ddof=0 batch variance vs
    torch's unbiased/ddof=1 — negligible at the benchmark's effective batch size
    of thousands, and irrelevant to the parity gate, which loads NF's stats and
    only runs eval.)
    """

    def __init__(self, *, hidden_size, n_heads, linear_hidden_size, dropout,
                 attn_dropout, activation="gelu", rngs: nnx.Rngs):
        self.activation = activation
        self.attn = MultiHeadAttention(
            hidden_size=hidden_size, n_heads=n_heads, attn_dropout=attn_dropout,
            proj_dropout=dropout, rngs=rngs,
        )
        self.dropout_attn = nnx.Dropout(rate=dropout, rngs=rngs)
        self.norm_attn = nnx.BatchNorm(hidden_size, axis=-1, momentum=0.9,
                                       epsilon=1e-5, rngs=rngs)
        self.ff1 = nnx.Linear(hidden_size, linear_hidden_size, rngs=rngs)
        self.ff2 = nnx.Linear(linear_hidden_size, hidden_size, rngs=rngs)
        self.dropout_ff = nnx.Dropout(rate=dropout, rngs=rngs)
        self.dropout_ffn = nnx.Dropout(rate=dropout, rngs=rngs)
        self.norm_ffn = nnx.BatchNorm(hidden_size, axis=-1, momentum=0.9,
                                      epsilon=1e-5, rngs=rngs)

    def __call__(self, x, prev, deterministic: bool, use_running_average: bool):
        act = _resolve_activation(self.activation)
        attn_out, scores = self.attn(x, prev=prev, deterministic=deterministic)
        x = x + self.dropout_attn(attn_out, deterministic=deterministic)
        x = self.norm_attn(x, use_running_average=use_running_average)
        ff = self.ff2(self.dropout_ff(act(self.ff1(x)), deterministic=deterministic))
        x = x + self.dropout_ffn(ff, deterministic=deterministic)
        x = self.norm_ffn(x, use_running_average=use_running_average)
        return x, scores


class TSTEncoder(nnx.Module):
    """Stack of ``n_layers`` encoder layers, threading residual attention scores."""

    def __init__(self, *, n_layers, hidden_size, n_heads, linear_hidden_size,
                 dropout, attn_dropout, activation="gelu", rngs: nnx.Rngs):
        self.layers = [
            TSTEncoderLayer(
                hidden_size=hidden_size, n_heads=n_heads,
                linear_hidden_size=linear_hidden_size, dropout=dropout,
                attn_dropout=attn_dropout, activation=activation, rngs=rngs,
            )
            for _ in range(n_layers)
        ]

    def __call__(self, x, deterministic: bool, use_running_average: bool):
        scores = None
        for layer in self.layers:
            x, scores = layer(x, prev=scores, deterministic=deterministic,
                              use_running_average=use_running_average)
        return x


class FlattenHead(nnx.Module):
    """Flatten and project to the horizon (univariate).

    NF flattens its encoder output ``[B, hidden, patch_num]`` with
    ``nn.Flatten(start_dim=-2)`` — HIDDEN-major (hidden is the slow axis). Our
    encoder produces ``[B, patch_num, hidden]``, so we transpose to
    ``[B, hidden, patch_num]`` BEFORE flattening, otherwise the linear-head weight
    columns are permuted relative to NF and the parity gate fails (measured:
    max|Δ|=1.81 patch-major vs 7.4e-5 hidden-major).
    """

    def __init__(self, *, hidden_size, patch_num, h, head_dropout, rngs: nnx.Rngs):
        self.linear = nnx.Linear(hidden_size * patch_num, h, rngs=rngs)
        self.dropout = nnx.Dropout(rate=head_dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        B = x.shape[0]
        flat = x.transpose(0, 2, 1).reshape(B, -1)   # [B, patch_num, hidden] -> hidden-major
        return self.dropout(self.linear(flat), deterministic=deterministic)


class PatchTSTNet(nnx.Module):
    """Full PatchTST backbone: RevIN -> patchify -> embed -> encoder -> head -> denorm.

    I/O mirrors the GRU network: ``__call__(x: [B, L, 1]) -> [B, h, 1]``.
    """

    def __init__(self, *, h, input_size, patch_len, stride, hidden_size, n_heads,
                 encoder_layers, linear_hidden_size, dropout, fc_dropout,
                 head_dropout, attn_dropout, revin, revin_affine,
                 revin_subtract_last, activation="gelu", rngs: nnx.Rngs):
        # fc_dropout is accepted for NF-signature parity but inert: NF uses it only
        # in the disabled pretrain head; the active residual dropout is `dropout`.
        self.h = h
        self.input_size = input_size
        self.patch_len = min(input_size + stride, patch_len)  # NF clamp
        self.stride = stride
        self.revin_enabled = revin
        patch_num = compute_patch_num(input_size, self.patch_len, stride)
        if revin:
            self.revin = RevIN(num_features=1, subtract_last=revin_subtract_last,
                               affine=revin_affine, rngs=rngs)
        self.embedding = PatchEmbedding(
            patch_len=self.patch_len, hidden_size=hidden_size, patch_num=patch_num,
            dropout=dropout, rngs=rngs,   # NF residual/positional dropout uses the main `dropout`
        )
        self.encoder = TSTEncoder(
            n_layers=encoder_layers, hidden_size=hidden_size, n_heads=n_heads,
            linear_hidden_size=linear_hidden_size, dropout=dropout,
            attn_dropout=attn_dropout, activation=activation, rngs=rngs,
        )
        self.head = FlattenHead(hidden_size=hidden_size, patch_num=patch_num, h=h,
                                head_dropout=head_dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool, use_running_average: bool) -> jnp.ndarray:
        x = x.astype(jnp.float32)                     # [B, L, 1]
        if self.revin_enabled:
            z, loc, scale = self.revin.norm(x)        # z: [B, L, 1]
        else:
            z, loc, scale = x, None, None
        series = z[:, :, 0]                           # [B, L]
        patches = patchify(series, patch_len=self.patch_len, stride=self.stride)
        tokens = self.embedding(patches, deterministic=deterministic)
        enc = self.encoder(tokens, deterministic=deterministic,
                           use_running_average=use_running_average)
        out = self.head(enc, deterministic=deterministic)   # [B, h]
        out = out[:, :, None]                               # [B, h, 1]
        if self.revin_enabled:
            out = self.revin.denorm(out, loc, scale)
        return out

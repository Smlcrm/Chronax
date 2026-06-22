"""Flax NNX modules for the iTransformer forecaster.

Univariate JAX/Flax-NNX port of ``neuralforecast.iTransformer`` (Liu et al. 2024,
"iTransformer: Inverted Transformers Are Effective for Time Series Forecasting").

The "inverted" design treats each *variate* (series) as a token and applies
attention ACROSS variates, with the lookback window L embedded as the token
feature vector. The series carries a channel dimension of size 1 (``c_in = 1``),
so for the univariate case the encoder operates on a single token of shape
``[B, 1, hidden_size]``. The module code is written N-generically (it accepts
``[B, L, N]`` and produces ``[B, h, N]``), so a multivariate path can reuse it
later; only the :class:`iTransformer` wrapper fixes ``N = 1``.

Parity with neuralforecast (PyTorch) is preserved by matching torch's default
``nn.Linear`` / ``nn.Conv1d`` init (``_TorchLinearInit``), exact (erf) GELU, and
``LayerNorm`` placement. ``float32`` throughout.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _TorchLinearInit:
    """Picklable initializer matching torch ``nn.Linear`` / ``nn.Conv1d`` default.

    torch draws both weight and bias from ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``
    (Kaiming-uniform with ``a=sqrt(5)`` reduces to this bound). Flax's
    ``nnx.Linear`` defaults to ``lecun_normal`` instead, so matching torch here is
    what makes training trajectories — and thus accuracy — comparable to
    neuralforecast. ``fan_in`` is the layer's ``in_features`` (the bias shares the
    weight's fan_in). A 1-D conv with ``kernel_size=1`` has ``fan_in = in_channels``,
    so the same initializer applies to the pointwise-conv FFN.
    """

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _resolve_activation(name: str):
    """Map an activation name to a callable. neuralforecast's iTransformer passes
    ``activation=F.gelu`` (EXACT erf GELU). ``jax.nn.gelu`` defaults to the tanh
    approximation, which diverges ~1e-3 per call, so pass ``approximate=False``.
    """
    if name == "gelu":
        return lambda z: jax.nn.gelu(z, approximate=False)  # == torch nn.GELU()
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unknown activation {name!r}. Available: 'gelu', 'relu'.")


class RevIN(nnx.Module):
    """Reversible instance normalization (Kim et al. 2022), per-window.

    Faithful to neuralforecast's iTransformer ``use_norm`` block, which centers on
    the per-window MEAN (``subtract_last=False``), divides by ``sqrt(var + eps)``
    with population variance (``unbiased=False`` / ``ddof=0``), and applies no
    learnable affine. Statistics are returned explicitly rather than cached, so the
    module is pure and vmap/scan-safe.
    """

    def __init__(
        self,
        num_features: int,
        *,
        subtract_last: bool = False,
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

    def norm(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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


class DataEmbeddingInverted(nnx.Module):
    """Inverted embedding: each variate's lookback becomes a token.

    Mirrors NF ``DataEmbedding_inverted``: permute ``[B, L, N] -> [B, N, L]`` then
    ``Linear(input_size -> hidden_size)`` (the lookback length is the feature dim),
    followed by dropout. Exogenous/time marks are not supported (univariate, no
    covariates), so the ``x_mark`` concat path is omitted.
    """

    def __init__(self, *, input_size, hidden_size, dropout, rngs: nnx.Rngs):
        init = _TorchLinearInit(input_size)
        self.value_embedding = nnx.Linear(
            input_size, hidden_size, kernel_init=init, bias_init=init, rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """x: [B, L, N] -> tokens: [B, N, hidden_size]."""
        x = x.astype(jnp.float32).transpose(0, 2, 1)   # [B, N, L]
        x = self.value_embedding(x)                    # [B, N, hidden]
        return self.dropout(x, deterministic=deterministic)


class AttentionLayer(nnx.Module):
    """Multi-head full (softmax dot-product) self-attention over the token axis.

    Folds NF's ``AttentionLayer`` + ``FullAttention`` into one module: q/k/v/out
    projections (``Linear(hidden -> n_heads*d_k)``, torch init) and a frozen scale
    ``d_k**-0.5`` applied pre-softmax (NF ``scale = 1/sqrt(E)``). Dropout is applied
    to the attention weights (NF ``FullAttention(attention_dropout=dropout)``); the
    output projection carries no dropout (the residual dropout lives in the encoder
    layer). No causal mask (NF ``mask_flag=False``), no residual-attention threading.
    """

    def __init__(self, *, hidden_size, n_heads, attn_dropout, rngs: nnx.Rngs):
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.d_k = hidden_size // n_heads
        self.scale = float(self.d_k ** -0.5)
        init = _TorchLinearInit(hidden_size)
        self.w_q = nnx.Linear(hidden_size, hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        self.w_k = nnx.Linear(hidden_size, hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        self.w_v = nnx.Linear(hidden_size, hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        self.w_o = nnx.Linear(hidden_size, hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        self.attn_dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)

    def _split(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)  # [B, H, T, d_k]

    def __call__(self, x, deterministic: bool):
        """x: [B, T, hidden] -> [B, T, hidden] (T = number of variate tokens)."""
        q = self._split(self.w_q(x))
        k = self._split(self.w_k(x))
        v = self._split(self.w_v(x))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, deterministic=deterministic)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, v)         # [B, H, T, d_k]
        B, _, T, _ = ctx.shape
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.d_k)
        return self.w_o(ctx)


class TransEncoderLayer(nnx.Module):
    """One iTransformer encoder layer (post-norm).

    NF ``TransEncoderLayer.forward``::

        new_x = attention(x, x, x)
        x = x + dropout(new_x)
        y = x = norm1(x)
        y = dropout(activation(conv1(y)))   # conv1d kernel=1 == pointwise Linear
        y = dropout(conv2(y))
        return norm2(x + y)

    The two ``Conv1d(kernel_size=1)`` layers are mathematically pointwise linear
    maps over the channel axis, so they are implemented as ``nnx.Linear`` (no
    transpose needed). ``activation`` is exact GELU (NF passes ``F.gelu``).
    """

    def __init__(self, *, hidden_size, n_heads, d_ff, dropout,
                 activation="gelu", rngs: nnx.Rngs):
        self.activation = activation
        self.attn = AttentionLayer(
            hidden_size=hidden_size, n_heads=n_heads, attn_dropout=dropout, rngs=rngs,
        )
        self.conv1 = nnx.Linear(hidden_size, d_ff,
                                kernel_init=_TorchLinearInit(hidden_size),
                                bias_init=_TorchLinearInit(hidden_size), rngs=rngs)
        self.conv2 = nnx.Linear(d_ff, hidden_size,
                                kernel_init=_TorchLinearInit(d_ff),
                                bias_init=_TorchLinearInit(d_ff), rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x, deterministic: bool):
        act = _resolve_activation(self.activation)
        new_x = self.attn(x, deterministic=deterministic)
        x = x + self.dropout(new_x, deterministic=deterministic)
        y = x = self.norm1(x)
        y = self.dropout(act(self.conv1(y)), deterministic=deterministic)
        y = self.dropout(self.conv2(y), deterministic=deterministic)
        return self.norm2(x + y)


class TransEncoder(nnx.Module):
    """Stack of ``e_layers`` encoder layers plus a final LayerNorm (NF ``norm_layer``)."""

    def __init__(self, *, e_layers, hidden_size, n_heads, d_ff, dropout,
                 activation="gelu", rngs: nnx.Rngs):
        self.layers = [
            TransEncoderLayer(
                hidden_size=hidden_size, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, activation=activation, rngs=rngs,
            )
            for _ in range(e_layers)
        ]
        self.norm = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)

    def __call__(self, x, deterministic: bool):
        for layer in self.layers:
            x = layer(x, deterministic=deterministic)
        return self.norm(x)


class ITransformerNet(nnx.Module):
    """Full iTransformer backbone: (RevIN) -> invert-embed -> encoder -> project -> (denorm).

    I/O mirrors the other Chronax neural nets: ``__call__(x: [B, L, 1]) -> [B, h, 1]``.
    Written N-generically: an ``[B, L, N]`` input yields ``[B, h, N]``.
    """

    def __init__(self, *, h, input_size, hidden_size, n_heads, e_layers, d_ff,
                 dropout, use_norm, activation="gelu", rngs: nnx.Rngs):
        self.h = h
        self.input_size = input_size
        self.use_norm = use_norm
        if use_norm:
            self.revin = RevIN(num_features=1, subtract_last=False, affine=False, rngs=rngs)
        self.enc_embedding = DataEmbeddingInverted(
            input_size=input_size, hidden_size=hidden_size, dropout=dropout, rngs=rngs,
        )
        self.encoder = TransEncoder(
            e_layers=e_layers, hidden_size=hidden_size, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, activation=activation, rngs=rngs,
        )
        init = _TorchLinearInit(hidden_size)
        self.projector = nnx.Linear(hidden_size, h, kernel_init=init, bias_init=init, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        x = x.astype(jnp.float32)                       # [B, L, N]
        if self.use_norm:
            z, loc, scale = self.revin.norm(x)          # z: [B, L, N]
        else:
            z, loc, scale = x, None, None
        enc_out = self.enc_embedding(z, deterministic=deterministic)   # [B, N, hidden]
        enc_out = self.encoder(enc_out, deterministic=deterministic)   # [B, N, hidden]
        dec_out = self.projector(enc_out).transpose(0, 2, 1)           # [B, N, h] -> [B, h, N]
        if self.use_norm:
            dec_out = self.revin.denorm(dec_out, loc, scale)
        return dec_out

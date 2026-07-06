"""Flax NNX modules for the VanillaTransformer forecaster.

Univariate JAX/Flax-NNX port of ``neuralforecast.VanillaTransformer`` (classic
encoder-decoder Transformer with full softmax attention — the Informer baseline,
Zhou et al. 2021). The encoder embeds the lookback window (token conv +
sinusoidal positional embedding) and runs full self-attention; the decoder
embeds ``concat(last label_len of input, zeros for h)`` and runs full
self-attention + cross-attention against the encoder output before a linear
projection to one channel.

Parity with neuralforecast (PyTorch): torch ``nn.Linear`` default init
(``_TorchLinearInit``), torch ``Conv1d`` kaiming-normal init for the token conv
(``_KaimingConvInit``), exact (erf) GELU, fixed sinusoidal positional embedding,
and NO causal mask anywhere (NF runs ``output_attention=False`` with
``attn_mask=None``). ``float32`` throughout.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _TorchLinearInit:
    """Picklable initializer matching torch ``nn.Linear`` / ``nn.Conv1d`` default.

    torch draws weight and bias from ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``.
    ``fan_in`` is the layer's ``in_features``.
    """

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class _KaimingConvInit:
    """Picklable init matching torch ``kaiming_normal_(fan_in, leaky_relu)``.

    NF ``TokenEmbedding`` initializes its ``Conv1d`` with
    ``kaiming_normal_(mode="fan_in", nonlinearity="leaky_relu")``: gain
    ``sqrt(2 / (1 + 0.01**2))`` and ``fan_in = in_channels * kernel_size``.
    For the flax ``nnx.Conv`` kernel of shape ``(*kernel_size, in, out)``,
    ``fan_in = prod(kernel_size) * in``; the same scalar ``std`` applies.
    """

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        gain = math.sqrt(2.0 / (1.0 + 0.01 ** 2))
        std = gain / math.sqrt(self.fan_in)
        return std * jax.random.normal(key, shape, dtype)


def _resolve_activation(name: str):
    """Map an activation name to a callable. NF passes ``F.gelu`` (EXACT erf
    GELU); ``jax.nn.gelu`` defaults to the tanh approximation, so pass
    ``approximate=False``.
    """
    if name == "gelu":
        return lambda z: jax.nn.gelu(z, approximate=False)
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unknown activation {name!r}. Available: 'gelu', 'relu'.")


def _positional_embedding(length: int, hidden_size: int) -> jnp.ndarray:
    """Fixed sinusoidal positional embedding ``[1, length, hidden_size]``.

    Mirrors NF ``PositionalEmbedding``: ``sin`` on even channels, ``cos`` on odd.
    Computed as a constant (no parameters). Assumes ``hidden_size`` is even
    (NF default 128); odd sizes are unsupported, as in NF.
    """
    position = jnp.arange(length, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(
        jnp.arange(0, hidden_size, 2, dtype=jnp.float32)
        * -(math.log(10000.0) / hidden_size)
    )
    pe = jnp.zeros((length, hidden_size), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe[None]


class TokenEmbedding(nnx.Module):
    """Circular k=3 Conv mapping the univariate channel to ``hidden_size``.

    NF ``TokenEmbedding`` is ``Conv1d(c_in=1, hidden, kernel_size=3, padding=1,
    padding_mode='circular', bias=False)`` over ``[B, C, L]``. flax ``nnx.Conv``
    is channels-last, so the input ``[B, L, 1]`` is consumed directly with
    ``padding='CIRCULAR'``; no permute needed.
    """

    def __init__(self, *, hidden_size: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=1,
            out_features=hidden_size,
            kernel_size=(3,),
            padding="CIRCULAR",
            use_bias=False,
            kernel_init=_KaimingConvInit(1 * 3),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x.astype(jnp.float32))


class DataEmbedding(nnx.Module):
    """Token conv + fixed sinusoidal positional embedding + dropout.

    Mirrors NF ``DataEmbedding`` for the univariate, no-exogenous case
    (``temporal_embedding`` omitted; ``pos_embedding=True``).
    """

    def __init__(self, *, hidden_size: int, dropout: float, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.value_embedding = TokenEmbedding(hidden_size=hidden_size, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """x: [B, T, 1] -> [B, T, hidden]."""
        x = self.value_embedding(x)
        pe = _positional_embedding(x.shape[1], self.hidden_size)
        x = x + pe
        return self.dropout(x, deterministic=deterministic)


class AttentionLayer(nnx.Module):
    """Multi-head full (softmax dot-product) attention, self- or cross-.

    Folds NF ``AttentionLayer`` + ``FullAttention`` (``output_attention=False``,
    no mask) into one module. ``q`` is projected from ``q_in``; ``k``/``v`` from
    ``kv_in`` (equal for self-attention). Scale ``d_k**-0.5``; dropout on the
    attention weights; torch ``nn.Linear`` init.
    """

    def __init__(self, *, hidden_size: int, n_heads: int, attn_dropout: float, rngs: nnx.Rngs):
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})."
            )
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
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)  # [B,H,T,d_k]

    def __call__(self, q_in, kv_in, deterministic: bool):
        """q_in: [B, Lq, hidden], kv_in: [B, Lk, hidden] -> [B, Lq, hidden]."""
        q = self._split(self.w_q(q_in))
        k = self._split(self.w_k(kv_in))
        v = self._split(self.w_v(kv_in))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights, deterministic=deterministic)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, v)              # [B,H,Lq,d_k]
        B, _, Lq, _ = ctx.shape
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, Lq, self.n_heads * self.d_k)
        return self.w_o(ctx)


class TransEncoderLayer(nnx.Module):
    """One encoder layer (post-norm), matching NF ``TransEncoderLayer``.

    ``new_x = attn(x, x); x = x + drop(new_x); y = x = norm1(x);
    y = drop(act(conv1(y))); y = drop(conv2(y)); return norm2(x + y)``.
    The two ``Conv1d(kernel_size=1)`` are pointwise linears with bias.
    """

    def __init__(self, *, hidden_size, n_heads, conv_hidden_size, dropout,
                 activation="gelu", rngs: nnx.Rngs):
        self.activation = activation
        self.attn = AttentionLayer(
            hidden_size=hidden_size, n_heads=n_heads, attn_dropout=dropout, rngs=rngs,
        )
        self.conv1 = nnx.Linear(hidden_size, conv_hidden_size,
                                kernel_init=_TorchLinearInit(hidden_size),
                                bias_init=_TorchLinearInit(hidden_size), rngs=rngs)
        self.conv2 = nnx.Linear(conv_hidden_size, hidden_size,
                                kernel_init=_TorchLinearInit(conv_hidden_size),
                                bias_init=_TorchLinearInit(conv_hidden_size), rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x, deterministic: bool):
        act = _resolve_activation(self.activation)
        new_x = self.attn(x, x, deterministic=deterministic)
        x = x + self.dropout(new_x, deterministic=deterministic)
        y = x = self.norm1(x)
        y = self.dropout(act(self.conv1(y)), deterministic=deterministic)
        y = self.dropout(self.conv2(y), deterministic=deterministic)
        return self.norm2(x + y)


class TransEncoder(nnx.Module):
    """Stack of ``encoder_layers`` encoder layers + final LayerNorm."""

    def __init__(self, *, encoder_layers, hidden_size, n_heads, conv_hidden_size,
                 dropout, activation="gelu", rngs: nnx.Rngs):
        self.layers = [
            TransEncoderLayer(
                hidden_size=hidden_size, n_heads=n_heads,
                conv_hidden_size=conv_hidden_size, dropout=dropout,
                activation=activation, rngs=rngs,
            )
            for _ in range(encoder_layers)
        ]
        self.norm = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)

    def __call__(self, x, deterministic: bool):
        for layer in self.layers:
            x = layer(x, deterministic=deterministic)
        return self.norm(x)


class TransDecoderLayer(nnx.Module):
    """One decoder layer, matching NF ``TransDecoderLayer`` (no causal mask).

    ``x = x + drop(self_attn(x, x)); x = norm1(x);
    x = x + drop(cross_attn(x, cross)); y = x = norm2(x);
    y = drop(act(conv1(y))); y = drop(conv2(y)); return norm3(x + y)``.
    """

    def __init__(self, *, hidden_size, n_heads, conv_hidden_size, dropout,
                 activation="gelu", rngs: nnx.Rngs):
        self.activation = activation
        self.self_attn = AttentionLayer(
            hidden_size=hidden_size, n_heads=n_heads, attn_dropout=dropout, rngs=rngs,
        )
        self.cross_attn = AttentionLayer(
            hidden_size=hidden_size, n_heads=n_heads, attn_dropout=dropout, rngs=rngs,
        )
        self.conv1 = nnx.Linear(hidden_size, conv_hidden_size,
                                kernel_init=_TorchLinearInit(hidden_size),
                                bias_init=_TorchLinearInit(hidden_size), rngs=rngs)
        self.conv2 = nnx.Linear(conv_hidden_size, hidden_size,
                                kernel_init=_TorchLinearInit(conv_hidden_size),
                                bias_init=_TorchLinearInit(conv_hidden_size), rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.norm3 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x, cross, deterministic: bool):
        act = _resolve_activation(self.activation)
        x = x + self.dropout(self.self_attn(x, x, deterministic=deterministic),
                             deterministic=deterministic)
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attn(x, cross, deterministic=deterministic),
                             deterministic=deterministic)
        y = x = self.norm2(x)
        y = self.dropout(act(self.conv1(y)), deterministic=deterministic)
        y = self.dropout(self.conv2(y), deterministic=deterministic)
        return self.norm3(x + y)


class TransDecoder(nnx.Module):
    """Stack of ``decoder_layers`` decoder layers + LayerNorm + projection."""

    def __init__(self, *, decoder_layers, hidden_size, n_heads, conv_hidden_size,
                 dropout, activation="gelu", c_out=1, rngs: nnx.Rngs):
        self.layers = [
            TransDecoderLayer(
                hidden_size=hidden_size, n_heads=n_heads,
                conv_hidden_size=conv_hidden_size, dropout=dropout,
                activation=activation, rngs=rngs,
            )
            for _ in range(decoder_layers)
        ]
        self.norm = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.projection = nnx.Linear(
            hidden_size, c_out,
            kernel_init=_TorchLinearInit(hidden_size),
            bias_init=_TorchLinearInit(hidden_size), rngs=rngs,
        )

    def __call__(self, x, cross, deterministic: bool):
        for layer in self.layers:
            x = layer(x, cross, deterministic=deterministic)
        x = self.norm(x)
        return self.projection(x)


class VanillaTransformerNet(nnx.Module):
    """Full encoder-decoder backbone: ``[B, input_size, 1] -> [B, h, 1]``.

    Identity scaling (NF default): operates in raw scale. During training pass
    ``deterministic=False``; ``nnx.Rngs`` supplies the dropout stream.
    """

    def __init__(self, *, h, input_size, hidden_size, n_heads, conv_hidden_size,
                 encoder_layers, decoder_layers, dropout, activation="gelu",
                 decoder_input_size_multiplier=0.5, rngs: nnx.Rngs):
        self.h = h
        self.input_size = input_size
        self.label_len = int(math.ceil(input_size * decoder_input_size_multiplier))
        self.enc_embedding = DataEmbedding(hidden_size=hidden_size, dropout=dropout, rngs=rngs)
        self.dec_embedding = DataEmbedding(hidden_size=hidden_size, dropout=dropout, rngs=rngs)
        self.encoder = TransEncoder(
            encoder_layers=encoder_layers, hidden_size=hidden_size, n_heads=n_heads,
            conv_hidden_size=conv_hidden_size, dropout=dropout, activation=activation, rngs=rngs,
        )
        self.decoder = TransDecoder(
            decoder_layers=decoder_layers, hidden_size=hidden_size, n_heads=n_heads,
            conv_hidden_size=conv_hidden_size, dropout=dropout, activation=activation,
            c_out=1, rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        x = x.astype(jnp.float32)                       # [B, L, 1]
        b = x.shape[0]
        zeros = jnp.zeros((b, self.h, 1), dtype=x.dtype)
        x_dec = jnp.concatenate([x[:, -self.label_len:, :], zeros], axis=1)  # [B, label_len+h, 1]
        enc_out = self.enc_embedding(x, deterministic=deterministic)
        enc_out = self.encoder(enc_out, deterministic=deterministic)
        dec_out = self.dec_embedding(x_dec, deterministic=deterministic)
        dec_out = self.decoder(dec_out, enc_out, deterministic=deterministic)  # [B, label_len+h, 1]
        return dec_out[:, -self.h:, :]

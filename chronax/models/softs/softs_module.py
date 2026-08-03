"""Flax NNX modules for the SOFTS forecaster.

Univariate JAX/Flax-NNX port of ``neuralforecast.SOFTS`` (Han et al. 2024,
"SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion",
arXiv:2404.14197).

SOFTS shares iTransformer's "inverted" skeleton — each *variate* (series) is a
token, with the lookback window L embedded as the token feature vector — but
replaces the O(C^2) self-attention across tokens with the O(C) **STAD** (STar
Aggregate-Dispatch) module: every series is fused into a single global *core*
representation, which is then dispatched back and concatenated onto each series.
The series carries a channel dimension of size 1 (``c_in = 1``) for the
univariate case, so the encoder operates on a single token of shape
``[B, 1, hidden_size]``. The module code is written N-generically (it accepts
``[B, L, N]`` and produces ``[B, h, N]``), so a multivariate path can reuse it
later; only the :class:`SOFTS` wrapper fixes ``N = 1``.

Parity with neuralforecast (PyTorch) is preserved by matching torch's default
``nn.Linear`` init (``_TorchLinearInit``), exact (erf) GELU, ``LayerNorm``
placement, and the STAD train/eval pooling split (stochastic multinomial pooling
during training, deterministic softmax-weighted mean at inference). ``float32``
throughout. The encoder skeleton (``DataEmbeddingInverted``, ``RevIN``,
``TransEncoderLayer``, ``TransEncoder``) mirrors the iTransformer port verbatim,
since neuralforecast's SOFTS reuses ``common._modules.TransEncoder`` unchanged.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _TorchLinearInit:
    """Picklable initializer matching torch ``nn.Linear`` default.

    torch draws both weight and bias from ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``
    (Kaiming-uniform with ``a=sqrt(5)`` reduces to this bound). Flax's
    ``nnx.Linear`` defaults to ``lecun_normal`` instead, so matching torch here is
    what makes training trajectories — and thus accuracy — comparable to
    neuralforecast. ``fan_in`` is the layer's ``in_features`` (the bias shares the
    weight's fan_in).
    """

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _resolve_activation(name: str):
    """Map an activation name to a callable. neuralforecast's SOFTS passes
    ``activation=F.gelu`` (EXACT erf GELU). ``jax.nn.gelu`` defaults to the tanh
    approximation, which diverges ~1e-3 per call, so pass ``approximate=False``.
    """
    if name == "gelu":
        return lambda z: jax.nn.gelu(z, approximate=False)  # == torch nn.GELU()
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unknown activation {name!r}. Available: 'gelu', 'relu'.")


def _gelu(z: jnp.ndarray) -> jnp.ndarray:
    """Exact (erf) GELU — matches torch ``F.gelu`` used inside NF's STAD."""
    return jax.nn.gelu(z, approximate=False)


class RevIN(nnx.Module):
    """Reversible instance normalization (Kim et al. 2022), per-window.

    Faithful to neuralforecast's SOFTS ``use_norm`` block, which centers on the
    per-window MEAN (``subtract_last=False``), divides by ``sqrt(var + eps)`` with
    population variance (``unbiased=False`` / ``ddof=0``), and applies no learnable
    affine. Statistics are returned explicitly rather than cached, so the module is
    pure and vmap/scan-safe.
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


class STAD(nnx.Module):
    """STar Aggregate-Dispatch module — SOFTS's series-core fusion block.

    Replaces the encoder's self-attention. Given per-series tokens
    ``input: [B, C, d_series]`` (``d_series = hidden_size``, ``C`` = number of
    variate tokens):

    1. **Set FFN** ``h = gen2(gelu(gen1(input)))`` -> ``[B, C, d_core]``.
    2. **Aggregate into a core** (pool across the ``C`` axis into a single
       representation, then broadcast it back to all ``C`` series):

       - *train* (``deterministic=False``): STOCHASTIC pooling — for each
         ``(batch, core-dim)`` sample one series index from ``softmax(h)`` over the
         ``C`` axis (NF ``torch.multinomial``) and gather that series' value.
       - *eval* (``deterministic=True``): the softmax-weighted mean over the ``C``
         axis (NF's inference branch).
    3. **Dispatch + fuse** ``output = gen4(gelu(gen3([input, core])))`` -> ``[B, C, d_series]``.

    Cost is O(C) in the number of series, versus O(C^2) for attention — the point
    of SOFTS. The multinomial sample draws a fresh key from ``rngs`` each forward;
    under the training ``nnx.scan`` the key stream is threaded through the carry
    exactly like ``nnx.Dropout``'s, so successive steps sample independently.
    """

    def __init__(self, *, hidden_size, d_core, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.d_core = d_core
        self.gen1 = nnx.Linear(hidden_size, hidden_size,
                               kernel_init=_TorchLinearInit(hidden_size),
                               bias_init=_TorchLinearInit(hidden_size), rngs=rngs)
        self.gen2 = nnx.Linear(hidden_size, d_core,
                               kernel_init=_TorchLinearInit(hidden_size),
                               bias_init=_TorchLinearInit(hidden_size), rngs=rngs)
        self.gen3 = nnx.Linear(hidden_size + d_core, hidden_size,
                               kernel_init=_TorchLinearInit(hidden_size + d_core),
                               bias_init=_TorchLinearInit(hidden_size + d_core), rngs=rngs)
        self.gen4 = nnx.Linear(hidden_size, hidden_size,
                               kernel_init=_TorchLinearInit(hidden_size),
                               bias_init=_TorchLinearInit(hidden_size), rngs=rngs)
        # Held for the train-time multinomial pooling key. Shared RngStream state
        # threads through nnx.scan (same mechanism as the dropout modules above).
        self.rngs = rngs

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """x: [B, C, hidden] -> [B, C, hidden]."""
        B, C, _ = x.shape
        combined = self.gen2(_gelu(self.gen1(x)))          # [B, C, d_core]

        if deterministic:
            # Inference: softmax-weighted mean over the series (channel) axis.
            weight = jax.nn.softmax(combined, axis=1)       # over C
            core = jnp.sum(combined * weight, axis=1, keepdims=True)  # [B, 1, d_core]
        else:
            # Training: stochastic pooling. jax.random.categorical takes LOGITS and
            # softmaxes internally, so passing `combined` reproduces NF's
            # softmax(combined) -> multinomial. nan_to_num matches NF exactly.
            logits = jnp.nan_to_num(combined)               # [B, C, d_core]
            key = self.rngs.pooling()
            idx = jax.random.categorical(key, logits, axis=1)          # [B, d_core]
            core = jnp.take_along_axis(combined, idx[:, None, :], axis=1)  # [B, 1, d_core]

        core = jnp.broadcast_to(core, (B, C, self.d_core))  # dispatch to all series
        fused = _gelu(self.gen3(jnp.concatenate([x, core], axis=-1)))   # [B, C, hidden]
        return self.gen4(fused)


class TransEncoderLayer(nnx.Module):
    """One SOFTS encoder layer (post-norm), STAD in place of attention.

    NF ``TransEncoderLayer.forward`` (with STAD as the "attention")::

        new_x, _ = STAD(x)
        x = x + dropout(new_x)
        y = x = norm1(x)
        y = dropout(activation(conv1(y)))   # conv1d kernel=1 == pointwise Linear
        y = dropout(conv2(y))
        return norm2(x + y)

    The two ``Conv1d(kernel_size=1)`` layers are mathematically pointwise linear
    maps over the channel axis, so they are implemented as ``nnx.Linear`` (no
    transpose needed). ``activation`` is exact GELU (NF passes ``F.gelu``).
    """

    def __init__(self, *, hidden_size, d_core, d_ff, dropout,
                 activation="gelu", rngs: nnx.Rngs):
        self.activation = activation
        self.stad = STAD(hidden_size=hidden_size, d_core=d_core, rngs=rngs)
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
        new_x = self.stad(x, deterministic=deterministic)
        x = x + self.dropout(new_x, deterministic=deterministic)
        y = x = self.norm1(x)
        y = self.dropout(act(self.conv1(y)), deterministic=deterministic)
        y = self.dropout(self.conv2(y), deterministic=deterministic)
        return self.norm2(x + y)


class TransEncoder(nnx.Module):
    """Stack of ``e_layers`` encoder layers. NO final LayerNorm — see below.

    NF's ``common._modules.TransEncoder`` takes an OPTIONAL ``norm_layer`` and
    applies it only ``if self.norm is not None``. SOFTS builds the encoder
    positionally::

        TransEncoder([TransEncoderLayer(STAD(...), ...) for l in range(e_layers)])

    with no ``norm_layer`` argument, so the reference has no final normalization
    and the encoder output feeds ``projection`` directly — NF's ``state_dict``
    has no ``encoder.norm.*`` entry at all.

    The difference is subtler than it looks, and worth stating precisely: every
    layer already ENDS in ``norm2``, so an extra final LayerNorm is near-identity
    at initialization (re-normalizing an already-normalized vector). What it is
    not is free — its scale and bias are learnable, so it would hand the port
    ``2 * hidden_size`` trainable parameters the reference does not have, and a
    learned per-feature affine applied immediately before the projector.
    """

    def __init__(self, *, e_layers, hidden_size, d_core, d_ff, dropout,
                 activation="gelu", rngs: nnx.Rngs):
        self.layers = [
            TransEncoderLayer(
                hidden_size=hidden_size, d_core=d_core, d_ff=d_ff,
                dropout=dropout, activation=activation, rngs=rngs,
            )
            for _ in range(e_layers)
        ]

    def __call__(self, x, deterministic: bool):
        for layer in self.layers:
            x = layer(x, deterministic=deterministic)
        return x


class SOFTSNet(nnx.Module):
    """Full SOFTS backbone: (RevIN) -> invert-embed -> encoder -> project -> (denorm).

    I/O mirrors the other Chronax neural nets: ``__call__(x: [B, L, 1]) -> [B, h, 1]``.
    Written N-generically: an ``[B, L, N]`` input yields ``[B, h, N]``.
    """

    def __init__(self, *, h, input_size, hidden_size, d_core, e_layers, d_ff,
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
            e_layers=e_layers, hidden_size=hidden_size, d_core=d_core, d_ff=d_ff,
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

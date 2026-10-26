"""Flax NNX modules for the SOFTSSharp forecaster.

Univariate JAX/Flax-NNX port of ``neuralforecast.SOFTSSharp`` (SOFTS# — Ljubic,
"SOFTSSharp: SOFTS extension with stochastic variable-position encoding",
reference implementation at https://github.com/hljubic/SOFTSsharp; manuscript in
preparation), which itself extends SOFTS (Han et al. 2024, arXiv:2404.14197).

SOFTSSharp keeps the whole SOFTS skeleton — inverted embedding, post-norm
encoder, STAD (STar Aggregate-Dispatch) series-core fusion in place of
self-attention, RevIN-style ``use_norm`` — and changes exactly one block, STAD,
into :class:`STADSharp`:

1. **Stochastic variable-position encoding.** A fixed sinusoidal table indexed by
   the *variate* position (not time — the encoder's token axis is the series
   axis) is added to the STAD input, scaled by a learnable scalar ``pe_scale``.
   During training it is applied with probability ``pe_keep_prob`` (a single
   Bernoulli draw per forward, shared across the batch — matching NF's
   ``torch.rand(1).item() < pe_keep_prob``); at inference it is always applied,
   scaled down by ``pe_keep_prob``. This is the usual expectation-matching trick
   (inference sees the training-time mean), the same one plain dropout uses.
2. **Three extra dropout layers inside STAD** — after the first FFN activation,
   after the dispatched core, and after the fusion activation.

Everything else is byte-for-byte the SOFTS port, deliberately duplicated rather
than imported: the neural packages under ``chronax/models/`` are self-contained
by convention (``chronax.models.softs.softs_module`` says the same about its own
copy of the iTransformer skeleton), so a future edit to one port cannot silently
change another.

Parity with neuralforecast (PyTorch) is preserved by matching torch's default
``nn.Linear`` init (``_TorchLinearInit``), exact (erf) GELU, ``LayerNorm``
placement, and the STAD train/eval pooling split (stochastic multinomial pooling
during training, deterministic softmax-weighted mean at inference). ``float32``
throughout.

Note on the univariate case: the position table is indexed by variate, so with
``n_series = 1`` only row 0 is ever read — ``sin(0), cos(0), ...`` — i.e. the
encoding degenerates to a fixed constant vector modulated by the learnable
``pe_scale`` and the Bernoulli gate. That is faithful to the reference (which
behaves the same way at ``n_series = 1``); the position encoding only starts to
differentiate series in a multivariate setting.
"""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
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
    """Map an activation name to a callable. neuralforecast's SOFTSSharp passes
    ``activation=F.gelu`` (EXACT erf GELU). ``jax.nn.gelu`` defaults to the tanh
    approximation, which diverges ~1e-3 per call, so pass ``approximate=False``.
    """
    if name == "gelu":
        return lambda z: jax.nn.gelu(z, approximate=False)  # == torch nn.GELU()
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unknown activation {name!r}. Available: 'gelu', 'relu'.")


def _gelu(z: jnp.ndarray) -> jnp.ndarray:
    """Exact (erf) GELU — matches torch ``F.gelu`` used inside NF's STADSharp."""
    return jax.nn.gelu(z, approximate=False)


@functools.lru_cache(maxsize=None)
def positional_table(d_series: int, max_len: int = 5000) -> np.ndarray:
    """Fixed sinusoidal table of shape ``[1, max_len, d_series]``, as ``float32``.

    Mirrors NF's ``PositionalEmbedding`` buffer: even feature indices carry
    ``sin(pos * w_i)``, odd indices ``cos(pos * w_i)``, with
    ``w_i = exp(-i * log(10000) / d_series)`` for ``i in range(0, d_series, 2)``.

    The table is a deterministic function of its two arguments and is never
    trained (torch registers it as a *buffer*), so it is cached at module level
    and closed over as a jit constant instead of being carried as NNX state —
    which keeps it out of ``nnx.Optimizer``'s ``wrt=nnx.Param`` selection with no
    extra plumbing, and keeps the pickled state free of a 5000-row constant.

    Built with NUMPY, not ``jnp``, and cached as a host array: a cached ``jnp``
    value would be created the first time this runs, which under the training
    ``nnx.scan`` is inside a trace — the cache would then hand a leaked
    ``DynamicJaxprTracer`` to every later call (``UnexpectedTracerError``). A host
    array is trace-independent and is staged into each computation as a constant.

    NF's implementation assumes an even ``d_series`` (its ``0::2`` / ``1::2``
    assignment shapes disagree otherwise and torch raises). Here the cosine half
    is truncated to ``d_series // 2`` columns so an odd ``hidden_size`` degrades
    to the natural interleaving instead of crashing.
    """
    if d_series < 1:
        raise ValueError(f"d_series must be >= 1; got {d_series}.")
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1; got {max_len}.")
    position = np.arange(max_len, dtype=np.float32)[:, None]            # [max_len, 1]
    div_term = np.exp(
        np.arange(0, d_series, 2, dtype=np.float32) * -(math.log(10000.0) / d_series)
    )                                                                    # [ceil(d/2)]
    table = np.zeros((max_len, d_series), dtype=np.float32)
    table[:, 0::2] = np.sin(position * div_term)
    table[:, 1::2] = np.cos(position * div_term[: d_series // 2])
    table.setflags(write=False)  # cached and shared — never mutate in place
    return table[None, :, :]                                             # [1, max_len, d]


class PositionalEmbedding(nnx.Module):
    """Additive sinusoidal encoding over the token (variate) axis.

    ``__call__(x, scale)`` returns ``x + scale * table[:, :x.shape[1]]``, matching
    NF ``PositionalEmbedding.forward``. ``scale`` is a scalar (array or float) so
    the caller can pass a learnable parameter, a gated parameter, or a constant.
    """

    def __init__(self, d_series: int, max_len: int = 5000):
        self.d_series = d_series
        self.max_len = max_len

    def __call__(self, x: jnp.ndarray, scale=1.0) -> jnp.ndarray:
        """x: [B, C, d_series] -> same shape. Requires ``C <= max_len``."""
        n_tokens = x.shape[1]
        if n_tokens > self.max_len:
            raise ValueError(
                f"PositionalEmbedding built for max_len={self.max_len} but got "
                f"{n_tokens} tokens; raise max_len."
            )
        table = positional_table(self.d_series, self.max_len)[:, :n_tokens, :]
        return x + scale * table


class STADSharp(nnx.Module):
    """STAD with stochastic variable-position encoding — the SOFTSSharp block.

    Given per-series tokens ``input: [B, C, d_series]`` (``d_series = hidden_size``,
    ``C`` = number of variate tokens):

    0. **Variable-position encoding** (new vs SOFTS): add the sinusoidal table
       indexed by variate position, scaled by the learnable ``pe_scale``. Applied
       with probability ``pe_keep_prob`` in training (one Bernoulli per forward,
       shared across the batch); always applied at inference, scaled by
       ``pe_keep_prob * pe_scale``. The encoded tensor is what feeds *both* the
       set FFN and the later fusion concat, exactly as in NF.
    1. **Set FFN** ``h = gen2(dropout1(gelu(gen1(input))))`` -> ``[B, C, d_core]``.
    2. **Aggregate into a core** (pool across the ``C`` axis into a single
       representation, then broadcast it back to all ``C`` series):

       - *train* (``deterministic=False``): STOCHASTIC pooling — for each
         ``(batch, core-dim)`` sample one series index from ``softmax(h)`` over the
         ``C`` axis (NF ``torch.multinomial``) and gather that series' value.
       - *eval* (``deterministic=True``): the softmax-weighted mean over the ``C``
         axis (NF's inference branch).

       followed by ``dropout2`` on the dispatched core (new vs SOFTS).
    3. **Dispatch + fuse** ``output = gen4(dropout3(gelu(gen3([input, core]))))``
       -> ``[B, C, d_series]`` (``dropout3`` new vs SOFTS).

    Cost stays O(C) in the number of series, versus O(C^2) for attention. The
    multinomial sample and the position-encoding Bernoulli both draw fresh keys
    from ``rngs`` each forward; under the training ``nnx.scan`` the key streams are
    threaded through the carry exactly like ``nnx.Dropout``'s, so successive steps
    sample independently.
    """

    def __init__(self, *, hidden_size, d_core, dropout: float = 0.1,
                 pe_keep_prob: float = 0.5, pe_max_len: int = 5000, rngs: nnx.Rngs):
        if not 0.0 <= pe_keep_prob <= 1.0:
            raise ValueError(f"pe_keep_prob must be in [0, 1]; got {pe_keep_prob}.")
        self.hidden_size = hidden_size
        self.d_core = d_core
        self.pe_keep_prob = pe_keep_prob
        self.positional_embedding = PositionalEmbedding(hidden_size, max_len=pe_max_len)
        # NF: nn.Parameter(torch.tensor(1.0)) — a learnable scalar gain on the
        # position table (kept as a rank-0 Param so it trains under wrt=nnx.Param).
        self.pe_scale = nnx.Param(jnp.asarray(1.0, dtype=jnp.float32))
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
        self.dropout1 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=dropout, rngs=rngs)
        # Held for the train-time multinomial-pooling and PE-gate keys. Shared
        # RngStream state threads through nnx.scan (same mechanism as dropout).
        self.rngs = rngs

    def add_positional_embedding(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """NF ``STADSharp.add_positional_embedding``, as a branch-free JAX gate.

        NF draws ONE uniform per forward and keeps the encoding iff it is below
        ``pe_keep_prob``; that Python ``if`` cannot survive tracing, so the draw is
        folded into the scale (``gate * pe_scale``, gate in {0, 1}). Multiplying by
        a zero scale is exactly the "return input unchanged" branch, and keeps the
        gradient path to ``pe_scale`` intact on the kept branch.
        """
        if deterministic:
            return self.positional_embedding(x, scale=self.pe_keep_prob * self.pe_scale.value)
        gate = (jax.random.uniform(self.rngs.pe(), ()) < self.pe_keep_prob).astype(jnp.float32)
        return self.positional_embedding(x, scale=gate * self.pe_scale.value)

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """x: [B, C, hidden] -> [B, C, hidden]."""
        x = self.add_positional_embedding(x, deterministic=deterministic)
        B, C, _ = x.shape

        combined = _gelu(self.gen1(x))
        combined = self.dropout1(combined, deterministic=deterministic)
        combined = self.gen2(combined)                     # [B, C, d_core]

        if deterministic:
            # Inference: softmax-weighted mean over the series (channel) axis.
            weight = jax.nn.softmax(combined, axis=1)       # over C
            core = jnp.sum(combined * weight, axis=1, keepdims=True)  # [B, 1, d_core]
        else:
            # Training: stochastic pooling. jax.random.categorical takes LOGITS and
            # softmaxes internally, so passing `combined` reproduces NF's
            # softmax(combined) -> multinomial. nan_to_num matches NF exactly.
            logits = jnp.nan_to_num(combined)               # [B, C, d_core]
            idx = jax.random.categorical(self.rngs.pooling(), logits, axis=1)  # [B, d_core]
            core = jnp.take_along_axis(combined, idx[:, None, :], axis=1)      # [B, 1, d_core]

        core = jnp.broadcast_to(core, (B, C, self.d_core))  # dispatch to all series
        core = self.dropout2(core, deterministic=deterministic)

        fused = _gelu(self.gen3(jnp.concatenate([x, core], axis=-1)))   # [B, C, hidden]
        fused = self.dropout3(fused, deterministic=deterministic)
        return self.gen4(fused)


class RevIN(nnx.Module):
    """Reversible instance normalization (Kim et al. 2022), per-window.

    Faithful to neuralforecast's SOFTSSharp ``use_norm`` block, which centers on
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
            z = (z - self.beta.value[None, None, :]) / (
                self.gamma.value[None, None, :] + self.eps * self.eps
            )
        return z * scale + loc


class DataEmbeddingInverted(nnx.Module):
    """Inverted embedding: each variate's lookback becomes a token.

    Mirrors NF ``DataEmbedding_inverted`` (SOFTSSharp imports SOFTS's verbatim):
    permute ``[B, L, N] -> [B, N, L]`` then ``Linear(input_size -> hidden_size)``
    (the lookback length is the feature dim), followed by dropout.
    Exogenous/time marks are not supported (univariate, no covariates), so the
    ``x_mark`` concat path is omitted.
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


class TransEncoderLayer(nnx.Module):
    """One SOFTSSharp encoder layer (post-norm), STADSharp in place of attention.

    NF ``TransEncoderLayer.forward`` (with STADSharp as the "attention")::

        new_x, _ = STADSharp(x)
        x = x + dropout(new_x)
        y = x = norm1(x)
        y = dropout(activation(conv1(y)))   # conv1d kernel=1 == pointwise Linear
        y = dropout(conv2(y))
        return norm2(x + y)

    The two ``Conv1d(kernel_size=1)`` layers are mathematically pointwise linear
    maps over the channel axis, so they are implemented as ``nnx.Linear`` (no
    transpose needed). ``activation`` is exact GELU (NF passes ``F.gelu``). Note
    the residual adds the layer input BEFORE the position encoding — the encoding
    lives entirely inside STADSharp, as in NF.
    """

    def __init__(self, *, hidden_size, d_core, d_ff, dropout, pe_keep_prob,
                 pe_max_len=5000, activation="gelu", rngs: nnx.Rngs):
        self.activation = activation
        self.stad = STADSharp(hidden_size=hidden_size, d_core=d_core, dropout=dropout,
                              pe_keep_prob=pe_keep_prob, pe_max_len=pe_max_len, rngs=rngs)
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
    applies it only ``if self.norm is not None``. SOFTSSharp (like SOFTS) builds
    the encoder positionally::

        TransEncoder([TransEncoderLayer(STADSharp(...), ...) for l in range(e_layers)])

    with no ``norm_layer`` argument, so the reference has no final normalization
    and the encoder output feeds ``projection`` directly.

    The difference is subtler than it looks, and worth stating precisely: every
    layer already ENDS in ``norm2``, so an extra final LayerNorm is near-identity
    at initialization (re-normalizing an already-normalized vector). What it is
    not is free — its scale and bias are learnable, so it would hand the port
    ``2 * hidden_size`` trainable parameters the reference does not have, and a
    learned per-feature affine applied immediately before the projector. That is
    a real architectural divergence, so it is omitted.
    """

    def __init__(self, *, e_layers, hidden_size, d_core, d_ff, dropout, pe_keep_prob,
                 pe_max_len=5000, activation="gelu", rngs: nnx.Rngs):
        self.layers = [
            TransEncoderLayer(
                hidden_size=hidden_size, d_core=d_core, d_ff=d_ff, dropout=dropout,
                pe_keep_prob=pe_keep_prob, pe_max_len=pe_max_len,
                activation=activation, rngs=rngs,
            )
            for _ in range(e_layers)
        ]

    def __call__(self, x, deterministic: bool):
        for layer in self.layers:
            x = layer(x, deterministic=deterministic)
        return x


class SOFTSSharpNet(nnx.Module):
    """Full SOFTSSharp backbone: (RevIN) -> invert-embed -> encoder -> project -> (denorm).

    I/O mirrors the other Chronax neural nets: ``__call__(x: [B, L, 1]) -> [B, h, 1]``.
    Written N-generically: an ``[B, L, N]`` input yields ``[B, h, N]``, so a
    multivariate path can reuse it later; only the :class:`SOFTSSharp` wrapper
    fixes ``N = 1``.
    """

    def __init__(self, *, h, input_size, hidden_size, d_core, e_layers, d_ff,
                 dropout, use_norm, pe_keep_prob=0.5, pe_max_len=5000,
                 activation="gelu", rngs: nnx.Rngs):
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
            dropout=dropout, pe_keep_prob=pe_keep_prob, pe_max_len=pe_max_len,
            activation=activation, rngs=rngs,
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

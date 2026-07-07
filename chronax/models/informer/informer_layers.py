"""Layer primitives for Informer (flax.nnx).

Faithful port of the building blocks in ``neuralforecast/models/informer.py``:
torch-matching linear/conv initializers, the activation resolver,
``DataEmbedding`` (circular-conv token embedding + on-the-fly sinusoidal
position code + optional time-feature linear for future-known marks),
ProbSparse attention (``_prob_attention`` + ``AttentionLayer``), the distilling
``ConvLayer``, and the post-norm ``TransEncoderLayer``/``TransDecoderLayer``.
``float32`` throughout.
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


def _torch_linear(in_features: int, out_features: int, *, use_bias: bool = True, rngs: nnx.Rngs):
    init = _TorchLinearInit(in_features)
    return nnx.Linear(
        in_features, out_features, use_bias=use_bias,
        kernel_init=init, bias_init=init, rngs=rngs,
    )


class _KaimingNormalConvInit:
    """Picklable initializer matching torch's ``Conv1d`` init in Informer's ``TokenEmbedding``.

    The reference calls ``nn.init.kaiming_normal_(weight, mode='fan_in',
    nonlinearity='leaky_relu')`` (``a=0``), which reduces to drawing from
    ``N(0, 2/fan_in)``. For a 1-D conv, ``fan_in = kernel_size * in_channels``.
    Plain class with ``__slots__`` (no lambda/closure) so a fitted estimator
    holding this initializer still pickles, matching the ``_TorchLinearInit``
    precedent.
    """

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * math.sqrt(2.0 / self.fan_in)


def _resolve_activation(name: str):
    """Map an activation name to a callable. Informer's default is ``F.gelu``
    (EXACT erf GELU, matching torch's ``nn.GELU()``). ``jax.nn.gelu`` defaults to
    the tanh approximation, which diverges ~1e-3 per call, so pass
    ``approximate=False``.
    """
    if name == "gelu":
        return lambda z: jax.nn.gelu(z, approximate=False)  # == torch nn.GELU()
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unknown activation {name!r}. Available: 'gelu', 'relu'.")


def sinusoid_position_embedding(L: int, hidden_size: int) -> jnp.ndarray:
    """Fixed (non-learned) sinusoidal position code, shape ``[L, hidden_size]``.

    Standard transformer encoding: even columns are ``sin(pos*freq)``, odd
    columns are ``cos(pos*freq)``, with ``freq`` geometrically spaced via
    ``exp(-ln(10000)*i/hidden_size)``. ``L`` and ``hidden_size`` are static
    Python ints, so this constant-folds under jit; there is no learned
    parameter and no cached max-length buffer (unlike the torch reference's
    preallocated ``pe``), matching this port's params-only nnx.Module style.
    ``hidden_size`` may be odd: ``0::2``/``1::2`` need equally-sized targets,
    so the sin/cos grids are built on a one-wider padded buffer when
    ``hidden_size`` is odd, then sliced back to ``hidden_size``.
    """
    hidden_pad = hidden_size + (hidden_size % 2)
    pos = jnp.arange(L, dtype=jnp.float32)[:, None]                      # [L, 1]
    freqs = jnp.exp(
        jnp.arange(0, hidden_size, 2, dtype=jnp.float32) * -(math.log(10000.0) / hidden_size)
    )                                                                      # [ceil(hidden_size/2)]
    pe = jnp.zeros((L, hidden_pad), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(pos * freqs))
    pe = pe.at[:, 1::2].set(jnp.cos(pos * freqs))
    return pe[:, :hidden_size]


class TokenEmbedding(nnx.Module):
    """Circular 1-D conv token embedding (NF ``TokenEmbedding``).

    A kernel-3 conv with ``padding="CIRCULAR"`` is length-preserving and wraps
    values at the sequence boundary, matching torch's ``Conv1d(kernel_size=3,
    padding=1, padding_mode='circular')``. Flax is feature-last (``[B, L,
    C]``), so — unlike a literal torch port — no axis permutes are needed. No
    bias (NF default); Kaiming-normal init with ``fan_in = 3 * c_in``.
    """

    def __init__(self, c_in: int, hidden_size: int, *, rngs: nnx.Rngs) -> None:
        self.conv = nnx.Conv(
            c_in, hidden_size, kernel_size=(3,), padding="CIRCULAR", use_bias=False,
            kernel_init=_KaimingNormalConvInit(3 * c_in), rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: ``[B, L, c_in]`` -> ``[B, L, hidden_size]``."""
        return self.conv(x.astype(jnp.float32))


class TimeFeatureEmbedding(nnx.Module):
    """Bias-free linear projection of time-feature marks (NF ``TimeFeatureEmbedding``)."""

    def __init__(self, input_size: int, hidden_size: int, *, rngs: nnx.Rngs) -> None:
        self.lin = _torch_linear(input_size, hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: ``[B, L, input_size]`` -> ``[B, L, hidden_size]``."""
        return self.lin(x.astype(jnp.float32))


class DataEmbedding(nnx.Module):
    """Token + sinusoidal-position (+ optional time-feature) embedding (NF ``DataEmbedding``).

    Sums the circular-conv token embedding, the fixed sinusoidal position code,
    and — only when ``exog_input_size > 0`` — a bias-free linear embedding of
    future-known time-feature marks, then applies dropout. ``temporal_embedding``
    is ``None`` (rather than a zero-input layer) when there are no exogenous
    marks, so ``x_mark`` may be passed as ``None`` in that case.
    """

    def __init__(
        self, *, c_in: int, exog_input_size: int, hidden_size: int, dropout: float, rngs: nnx.Rngs,
    ) -> None:
        self.hidden_size = hidden_size
        self.token_embedding = TokenEmbedding(c_in, hidden_size, rngs=rngs)
        self.temporal_embedding = (
            TimeFeatureEmbedding(exog_input_size, hidden_size, rngs=rngs)
            if exog_input_size > 0 else None
        )
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, x_mark: jnp.ndarray | None, deterministic: bool) -> jnp.ndarray:
        """x: ``[B, L, c_in]``, x_mark: ``[B, L, exog_input_size]`` or ``None`` -> ``[B, L, hidden_size]``."""
        L = x.shape[1]
        out = self.token_embedding(x) + sinusoid_position_embedding(L, self.hidden_size)[None]
        if self.temporal_embedding is not None:
            out = out + self.temporal_embedding(x_mark)
        return self.dropout(out, deterministic=deterministic)


def _prob_attention(q, k, v, *, factor: int, mask_flag: bool, key) -> jnp.ndarray:
    """ProbSparse attention (Informer). q: [B,H,L_Q,E]; k,v: [B,H,L_K,E] -> [B,H,L_Q,E].

    No dropout (the reference defines but never applies it). ``key`` drives the
    data-independent key subsample (shared across B and H, matching torch).
    """
    B, H, L_Q, E = q.shape
    L_K = k.shape[2]
    # Static Python ints (shapes are static under jit). max(1, .) guards L=1 where
    # ceil(ln 1) = 0 would make top_k/gather empty (the reference would crash).
    U_part = max(min(factor * math.ceil(math.log(L_K)), L_K), 1)   # sampled keys
    u = max(min(factor * math.ceil(math.log(L_Q)), L_Q), 1)        # scored queries
    # --- _prob_QK ---
    index_sample = jax.random.randint(key, (L_Q, U_part), 0, L_K)
    k_sample = k[:, :, index_sample, :]                            # [B,H,L_Q,U_part,E]
    qk_sample = jnp.einsum("bhle,bhlue->bhlu", q, k_sample)
    m = qk_sample.max(-1) - qk_sample.sum(-1) / L_K                # max - mean; / L_K NOT U_part (reference quirk)
    m_top = jax.lax.top_k(m, u)[1]                                 # [B,H,u]
    q_reduce = jnp.take_along_axis(q, m_top[..., None], axis=2)    # [B,H,u,E]
    scores = jnp.einsum("bhue,bhke->bhuk", q_reduce, k) / math.sqrt(E)   # scale AFTER selection
    # --- initial context ---
    if mask_flag:
        assert L_Q == L_K, "masked ProbAttention requires L_Q == L_V (self-attention only)"
        context = jnp.cumsum(v, axis=-2)                           # running SUM, not mean (reference quirk)
    else:
        context = jnp.broadcast_to(v.mean(axis=-2, keepdims=True), (B, H, L_Q, E))
    # --- update context at selected rows ---
    if mask_flag:
        causal = jnp.triu(jnp.ones((L_Q, L_K), dtype=bool), k=1)
        scores = jnp.where(causal[m_top], -1e9, scores)            # ProbMask: triu rows gathered at m_top
    attn = jax.nn.softmax(scores, axis=-1)
    b_grid = jnp.arange(B)[:, None, None]
    h_grid = jnp.arange(H)[None, :, None]
    return context.at[b_grid, h_grid, m_top].set(attn @ v)


class AttentionLayer(nnx.Module):
    """Multi-head ProbSparse attention (NF ``AttentionLayer`` + ``ProbAttention`` folded into one).

    q/k/v/out projections are ``hidden_size -> hidden_size`` (torch init, bias=True — NF
    default), split into ``n_head`` heads of size ``hidden_size // n_head``. Unlike
    ``itransformer``'s ``AttentionLayer`` (self-attention only), queries and keys/values
    here may have DIFFERENT lengths (``L_Q`` vs ``L_K``) — this is what lets the decoder's
    cross-attention run queries from the decoder against keys/values from the encoder.
    The inner attention is :func:`_prob_attention` rather than dense softmax attention.

    **Deliberate correctness deviation from Nixtla main:** NF's ``AttentionLayer.forward``
    flattens the ProbAttention context straight from its ``[B, H, L, E]`` layout (``out =
    out.view(B, L, -1)``), which silently interleaves the head and time axes whenever
    ``H > 1`` (reinterpreting ``[B, H, L, E]``-ordered memory as ``[B, L, H*E]`` mixes
    each head's features with the wrong time steps). The original Informer2020 code (and
    every other multi-head attention layer in this repo, e.g. ``itransformer_module.py``'s
    ``AttentionLayer``) transposes to ``[B, L, H, E]`` *before* flattening. We follow that
    correct convention — ``ctx.transpose(0, 2, 1, 3).reshape(B, L_Q, hidden_size)`` — here
    rather than reproduce NF's bug.

    No dropout (``_prob_attention`` never applies its own), no ``deterministic`` arg.
    """

    def __init__(
        self, *, hidden_size: int, n_head: int, factor: int, mask_flag: bool, rngs: nnx.Rngs,
    ) -> None:
        if hidden_size % n_head != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_head ({n_head}).")
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.factor = factor
        self.mask_flag = mask_flag
        self.w_q = _torch_linear(hidden_size, hidden_size, rngs=rngs)
        self.w_k = _torch_linear(hidden_size, hidden_size, rngs=rngs)
        self.w_v = _torch_linear(hidden_size, hidden_size, rngs=rngs)
        self.w_o = _torch_linear(hidden_size, hidden_size, rngs=rngs)

    def _split_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        B, L, _ = x.shape
        E = self.hidden_size // self.n_head
        return x.reshape(B, L, self.n_head, E).transpose(0, 2, 1, 3)   # [B,L,H,E] -> [B,H,L,E]

    def __call__(
        self, queries: jnp.ndarray, keys: jnp.ndarray, values: jnp.ndarray, *, sample_key,
    ) -> jnp.ndarray:
        """queries: ``[B,L_Q,hid]``; keys/values: ``[B,L_K,hid]`` -> ``[B,L_Q,hid]``."""
        B, L_Q, _ = queries.shape
        q = self._split_heads(self.w_q(queries))
        k = self._split_heads(self.w_k(keys))
        v = self._split_heads(self.w_v(values))
        ctx = _prob_attention(q, k, v, factor=self.factor, mask_flag=self.mask_flag, key=sample_key)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, L_Q, self.hidden_size)  # [B,H,L_Q,E]->[B,L_Q,H,E]->[B,L_Q,hid]
        return self.w_o(ctx)


class ConvLayer(nnx.Module):
    """Self-attention distilling layer (NF ``ConvLayer``).

    Roughly halves the sequence length between encoder stacks. NF pins torch
    ``Conv1d(kernel_size=3, padding=2, padding_mode='circular')`` — the paper
    repo's padding is torch-version-conditional and the NF benchmark twin uses
    padding=2, which EXPANDS the sequence to ``L + 2`` before the pool (the
    length-preserving padding=1 variant used here until 2026-07-07 was a parity
    bug, not a decision). Torch circular padding draws the left pad from the
    sequence tail and the right pad from its head, so we pre-pad explicitly and
    run a VALID conv. Init matches torch ``Conv1d`` default — same
    ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` bound as ``_torch_linear``, with
    ``fan_in = 3 * c_in`` for both kernel and bias. The conv feeds a
    ``BatchNorm`` + ELU, then a stride-2 max-pool. ``BatchNorm`` (rather than
    dropout/LayerNorm) is what the reference actually uses here, so — like
    ``patchtst_module.py``'s ``TSTEncoderLayer`` — the running-stat toggle is
    threaded as its own ``use_running_average`` flag; there is no dropout in
    this layer, so there is no ``deterministic`` argument.

    The max-pool ``padding=((1, 1),)`` pads with ``-inf`` (flax's ``max_pool``
    convention), matching torch's ``MaxPool1d(kernel_size=3, stride=2,
    padding=1)`` behavior on the boundary. Output length is ``(L+1)//2 + 1``.
    """

    def __init__(self, c_in: int, *, rngs: nnx.Rngs) -> None:
        self.conv = nnx.Conv(
            c_in, c_in, kernel_size=(3,), padding="VALID", use_bias=True,
            kernel_init=_TorchLinearInit(3 * c_in), bias_init=_TorchLinearInit(3 * c_in), rngs=rngs,
        )
        self.norm = nnx.BatchNorm(c_in, axis=-1, momentum=0.9, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, use_running_average: bool) -> jnp.ndarray:
        """x: ``[B, L, C]`` -> ``[B, (L+1)//2 + 1, C]``."""
        x = jnp.concatenate([x[:, -2:], x, x[:, :2]], axis=1)  # torch circular padding=2
        x = self.conv(x)  # VALID k=3 on L+4 -> L+2
        x = self.norm(x, use_running_average=use_running_average)
        x = jax.nn.elu(x)
        return nnx.max_pool(x, window_shape=(3,), strides=(2,), padding=((1, 1),))


class TransEncoderLayer(nnx.Module):
    """One Informer encoder layer (post-norm; NF ``EncoderLayer``).

    Adapted from ``itransformer_module.py``'s ``TransEncoderLayer`` (same
    post-norm skeleton: residual attention -> norm1 -> pointwise FFN ->
    residual -> norm2), swapping in the ProbSparse ``AttentionLayer``
    (``mask_flag=False``, unmasked self-attention) and threading a
    ``sample_key`` for its data-independent key subsample. The two
    ``Conv1d(kernel_size=1)`` layers in NF are pointwise linear maps over the
    channel axis, so — as in ``itransformer``'s port — they are plain
    ``nnx.Linear`` (named ``conv1``/``conv2`` to keep the NF correspondence
    visible), with FFN width ``conv_hidden_size``.
    """

    def __init__(
        self, *, hidden_size: int, n_head: int, conv_hidden_size: int, factor: int,
        dropout: float, activation: str = "gelu", rngs: nnx.Rngs,
    ) -> None:
        self.activation = activation
        self.attn = AttentionLayer(
            hidden_size=hidden_size, n_head=n_head, factor=factor, mask_flag=False, rngs=rngs,
        )
        self.conv1 = _torch_linear(hidden_size, conv_hidden_size, rngs=rngs)
        self.conv2 = _torch_linear(conv_hidden_size, hidden_size, rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, sample_key, deterministic: bool) -> jnp.ndarray:
        """x: ``[B, L, hidden]`` -> ``[B, L, hidden]``."""
        act = _resolve_activation(self.activation)
        new_x = self.attn(x, x, x, sample_key=sample_key)
        x = x + self.dropout(new_x, deterministic=deterministic)
        y = x = self.norm1(x)
        y = self.dropout(act(self.conv1(y)), deterministic=deterministic)
        y = self.dropout(self.conv2(y), deterministic=deterministic)
        return self.norm2(x + y)


class TransDecoderLayer(nnx.Module):
    """One Informer decoder layer (post-norm; NF ``DecoderLayer``).

    Same FFN/config shape as :class:`TransEncoderLayer` but with two attention
    sub-layers and three norms: masked ProbSparse self-attention
    (``mask_flag=True``) over the decoder's own sequence, then unmasked
    ProbSparse cross-attention (``mask_flag=False``) with queries from the
    decoder against keys/values from the encoder output (``cross``). Each
    attention call gets its own independent sampling key (``self_key`` /
    ``cross_key``) since the two ``AttentionLayer``s draw unrelated key
    subsamples.
    """

    def __init__(
        self, *, hidden_size: int, n_head: int, conv_hidden_size: int, factor: int,
        dropout: float, activation: str = "gelu", rngs: nnx.Rngs,
    ) -> None:
        self.activation = activation
        self.self_attn = AttentionLayer(
            hidden_size=hidden_size, n_head=n_head, factor=factor, mask_flag=True, rngs=rngs,
        )
        self.cross_attn = AttentionLayer(
            hidden_size=hidden_size, n_head=n_head, factor=factor, mask_flag=False, rngs=rngs,
        )
        self.conv1 = _torch_linear(hidden_size, conv_hidden_size, rngs=rngs)
        self.conv2 = _torch_linear(conv_hidden_size, hidden_size, rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.norm3 = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, cross: jnp.ndarray, *, self_key, cross_key, deterministic: bool,
    ) -> jnp.ndarray:
        """x: ``[B, L_dec, hidden]``, cross: ``[B, L_enc, hidden]`` -> ``[B, L_dec, hidden]``."""
        act = _resolve_activation(self.activation)
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, sample_key=self_key), deterministic=deterministic))
        new_x = self.cross_attn(x, cross, cross, sample_key=cross_key)
        x = x + self.dropout(new_x, deterministic=deterministic)
        y = x = self.norm2(x)
        y = self.dropout(act(self.conv1(y)), deterministic=deterministic)
        y = self.dropout(self.conv2(y), deterministic=deterministic)
        return self.norm3(x + y)

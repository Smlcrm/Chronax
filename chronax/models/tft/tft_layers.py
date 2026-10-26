"""Gated primitives and selection/attention layers for TFT (flax.nnx).

Faithful port of the building blocks in ``neuralforecast/models/tft.py``:
GLU, the Gated Residual Network (GRN), the Variable Selection Network, the
continuous embedding, and the interpretable multi-head attention. ``float32``
throughout; all ``nnx.Linear`` use torch-style init for trajectory parity.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _TorchLinearInit:
    """Picklable init matching torch ``nn.Linear`` default: ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``."""

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    def __eq__(self, other):  # I2 value equality — see informer_layers._TorchLinearInit
        return type(other) is type(self) and other.fan_in == self.fan_in

    def __hash__(self):
        return hash((type(self), self.fan_in))


_GRN_ACTIVATIONS = {
    "ELU": jax.nn.elu,
    "RELU": jax.nn.relu,
    "SOFTPLUS": jax.nn.softplus,
    "TANH": jax.nn.tanh,
    "SELU": jax.nn.selu,
    "LEAKYRELU": jax.nn.leaky_relu,
    "SIGMOID": jax.nn.sigmoid,
}


def _resolve_grn_activation(name: str):
    """Map a GRN activation name (NF ``grn_activation``) to a callable. Default ELU."""
    key = name.upper()
    if key not in _GRN_ACTIVATIONS:
        raise ValueError(
            f"Unknown grn activation {name!r}. Available: {sorted(_GRN_ACTIVATIONS)}."
        )
    return _GRN_ACTIVATIONS[key]


def _torch_linear(in_features: int, out_features: int, *, use_bias: bool = True, rngs: nnx.Rngs):
    init = _TorchLinearInit(in_features)
    return nnx.Linear(
        in_features, out_features, use_bias=use_bias,
        kernel_init=init, bias_init=init, rngs=rngs,
    )


class GLU(nnx.Module):
    """Gated Linear Unit: ``a * sigmoid(b)`` where ``[a, b] = Linear(x)`` (== torch F.glu)."""

    def __init__(self, input_size: int, output_size: int, *, rngs: nnx.Rngs):
        self.lin = _torch_linear(input_size, output_size * 2, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        a, b = jnp.split(self.lin(x), 2, axis=-1)
        return a * jax.nn.sigmoid(b)


class GRN(nnx.Module):
    """Gated Residual Network (NF GRN).

    ``y = MaybeLayerNorm(residual + GLU(lin_i(act(lin_a(a) + lin_c(c)))))``,
    where ``residual = a`` (or ``out_proj(a)`` when ``output_size`` is given) and
    ``MaybeLayerNorm`` is identity when ``output_size == 1``. The context ``c`` is
    static (no time axis) and is broadcast across leading non-feature dims.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        context_size: int | None = None,
        dropout: float = 0.0,
        activation: str = "ELU",
        *,
        rngs: nnx.Rngs,
    ):
        out = output_size or hidden_size
        self.lin_a = _torch_linear(input_size, hidden_size, rngs=rngs)
        self.lin_c = (
            _torch_linear(context_size, hidden_size, use_bias=False, rngs=rngs)
            if context_size
            else None
        )
        self.lin_i = _torch_linear(hidden_size, hidden_size, rngs=rngs)
        self.glu = GLU(hidden_size, out, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.out_proj = _torch_linear(input_size, out, rngs=rngs) if output_size is not None else None
        self.layer_norm = None if out == 1 else nnx.LayerNorm(out, epsilon=1e-3, rngs=rngs)
        self.activation = _resolve_grn_activation(activation)

    def __call__(self, a: jnp.ndarray, c: jnp.ndarray | None = None, deterministic: bool = True) -> jnp.ndarray:
        x = self.lin_a(a)
        if c is not None and self.lin_c is not None:
            cproj = self.lin_c(c)
            while cproj.ndim < x.ndim:           # broadcast static context over time
                cproj = jnp.expand_dims(cproj, axis=-2)
            x = x + cproj
        x = self.activation(x)
        x = self.lin_i(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.glu(x)
        residual = a if self.out_proj is None else self.out_proj(a)
        x = x + residual
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class VariableSelectionNetwork(nnx.Module):
    """Selects and combines ``num_inputs`` embedded variables (NF VSN).

    A joint GRN over the flattened variable embeddings (plus optional static
    context) produces softmax selection weights; each variable is transformed by
    its own GRN; the outputs are combined by the weights. Works with or without a
    time axis (static inputs have none).
    """

    def __init__(
        self,
        hidden_size: int,
        num_inputs: int,
        dropout: float = 0.0,
        context_size: int | None = None,
        activation: str = "ELU",
        *,
        rngs: nnx.Rngs,
    ):
        self.num_inputs = num_inputs
        # NF parity: the joint/selection GRN takes NO dropout — only the per-variable
        # GRNs do. Regularizing the selection logits under-commits the target channel
        # on seasonal series, leaving weight on constant inputs.
        self.joint_grn = GRN(
            hidden_size * num_inputs, hidden_size, output_size=num_inputs,
            context_size=context_size, dropout=0.0, activation=activation, rngs=rngs,
        )
        self.var_grns = [
            GRN(hidden_size, hidden_size, dropout=dropout, activation=activation, rngs=rngs)
            for _ in range(num_inputs)
        ]

    def __call__(self, x: jnp.ndarray, context: jnp.ndarray | None = None, deterministic: bool = True):
        flat = x.reshape(*x.shape[:-2], -1)                                  # [..., num_inputs*hidden]
        weights = jax.nn.softmax(
            self.joint_grn(flat, context, deterministic=deterministic), axis=-1
        )                                                                    # [..., num_inputs]
        transformed = jnp.stack(
            [self.var_grns[i](x[..., i, :], deterministic=deterministic) for i in range(self.num_inputs)],
            axis=-1,
        )                                                                    # [..., hidden, num_inputs]
        out = jnp.matmul(transformed, weights[..., None])[..., 0]            # [..., hidden]
        return out, weights


class ContinuousEmbedding(nnx.Module):
    """Per-feature continuous embedding (NF TFTEmbedding, continuous path).

    Each scalar feature ``j`` maps to ``x[..., j, None] * vec[j] + bias[j]``
    with learned ``vec, bias`` of shape ``[num_features, hidden]``. ``float32``.
    """

    def __init__(self, num_features: int, hidden_size: int, *, rngs: nnx.Rngs):
        self.num_features = num_features
        self.hidden_size = hidden_size
        if num_features > 0:
            k = rngs.params()
            # NF parity: torch.nn.init.xavier_normal_ on the [num_features, hidden]
            # vectors, std = sqrt(2/(num_features+hidden)). A fixed small std (e.g.
            # normal*0.02) starts the input coupling far weaker and costs accuracy on
            # seasonal data within a fixed step budget.
            std = (2.0 / (num_features + hidden_size)) ** 0.5
            vec = jax.random.normal(k, (num_features, hidden_size), dtype=jnp.float32) * std
            self.vectors = nnx.Param(vec.astype(jnp.float32))
            self.bias = nnx.Param(jnp.zeros((num_features, hidden_size), dtype=jnp.float32))
        else:
            self.vectors = None
            self.bias = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.num_features == 0:
            return jnp.zeros((*x.shape, self.hidden_size), dtype=jnp.float32)
        return x[..., None] * self.vectors.value + self.bias.value


class InterpretableMultiHeadAttention(nnx.Module):
    """TFT interpretable multi-head attention.

    Q and K are multi-head; **V is shared** across heads (a single value
    projection), and head outputs are **averaged** (not concatenated) before the
    output projection -- this is what makes the attention weights interpretable.
    Causal: each query attends only to keys at <= its own time index. The mask is
    computed with ``jnp.tril`` (no stored buffer) so the module is vmap/scan-pure.
    """

    def __init__(self, n_head: int, hidden_size: int, attn_dropout: float = 0.0,
                 dropout: float = 0.0, *, rngs: nnx.Rngs):
        if hidden_size % n_head != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_head ({n_head}).")
        self.n_head = n_head
        self.d_head = hidden_size // n_head
        self.scale = self.d_head ** -0.5
        self.qkv = nnx.Linear(
            hidden_size, (2 * n_head + 1) * self.d_head, use_bias=False,
            kernel_init=_TorchLinearInit(hidden_size), rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            self.d_head, hidden_size, use_bias=False,
            kernel_init=_TorchLinearInit(self.d_head), rngs=rngs,
        )
        self.attn_dropout = nnx.Dropout(rate=attn_dropout, rngs=rngs)
        # NF parity: a second dropout at rate `dropout` (not attn_dropout) after
        # the output projection.
        self.out_dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True, query_start: int = 0):
        """Compute attention only for query rows ``>= query_start``.

        A deliberate schedule deviation from NF, which computes all T query rows and
        discards all but the last h in the decoder — most of that work is wasted at
        long input lengths. Per-row math is identical (row-wise softmax/dot); the
        deterministic path matches compute-then-slice to ~1 ULP of f32 reassociation
        (guarded by test_imha_query_slice_matches_compute_then_slice). Training
        differs only through the out_dropout mask SHAPE ([B,h,·] vs [B,T,·]).
        ``query_start=0`` is the identity schedule. Returned ``attn`` is
        ``[B, n_head, T-query_start, T]``.
        """
        B, T, _ = x.shape
        qkv = self.qkv(x)                    # k/v need all rows; q's unused rows are dropped below
        nh = self.n_head * self.d_head
        q, k, v = jnp.split(qkv, [nh, 2 * nh], axis=-1)
        Tq = T - query_start
        q = q[:, query_start:].reshape(B, Tq, self.n_head, self.d_head)
        k = k.reshape(B, T, self.n_head, self.d_head)
        # v stays [B, T, d_head] -- shared across heads
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * self.scale            # [B, n_head, Tq, T]
        causal = jnp.tril(jnp.ones((T, T), dtype=bool))[query_start:]        # [Tq, T]
        scores = jnp.where(causal[None, None], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.attn_dropout(attn, deterministic=deterministic)
        ctx = jnp.einsum("bhqk,bkd->bhqd", attn, v)                          # v broadcast over heads
        ctx = jnp.mean(ctx, axis=1)                                          # average heads -> [B, Tq, d_head]
        return self.out_dropout(self.out_proj(ctx), deterministic=deterministic), attn

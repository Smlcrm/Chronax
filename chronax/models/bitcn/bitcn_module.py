"""Flax NNX modules for the BiTCN forecaster.

Univariate JAX/Flax-NNX port of ``neuralforecast.BiTCN`` (Bidirectional
Temporal Convolutional Network; Sprangers, Schelter & de Rijke, 2023). BiTCN
stacks two dilated temporal convolutional networks (TCNs): a *backward* network
that encodes past observations and a *forward* network that encodes future
covariates. This is a univariate model; Chronax (like the iTransformer /
VanillaTransformer ports) targets the **no-exogenous** case, so only the
backward TCN is instantiated (NF instantiates the forward TCN solely when
``futr_exog_size > 0``). With no exogenous inputs the NF ``forward`` reduces to::

    x = drop_hist(lin_hist(insample_y))           # [B, L, 1]      -> [B, L, H]
    x = x.permute(0, 2, 1)                         #                -> [B, H, L]
    _, x = net_bwd((x, 0))                         # backward TCN   -> [B, H, L]
    x = drop_temporal(gelu(temporal_lin1(x)))      # over L         -> [B, H, H]
    x = temporal_lin2(x)                           #                -> [B, H, h]
    forecast = output_lin(x.permute(0, 2, 1))      #                -> [B, h, 1]

Parity with neuralforecast (PyTorch): torch ``nn.Linear`` default init
(``_TorchLinearInit``: ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``), torch
``nn.Conv1d`` default init (``_TorchConvInit``: ``U(-k, k)``,
``k = sqrt(1 / (in_channels * kernel_size))``), exact (erf) GELU, causal
(left-padded) dilated convolutions with ``kernel_size=2`` and dilation
``2**i``, and residual + skip accumulation inside each ``TCNCell``.
``float32`` throughout.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _TorchLinearInit:
    """Picklable initializer matching torch ``nn.Linear`` default.

    torch draws weight and bias from ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``,
    where ``fan_in`` is the layer's ``in_features``.
    """

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class _TorchConvInit:
    """Picklable initializer matching torch ``nn.Conv1d`` default.

    NF ``CustomConv1d`` reproduces torch's default conv init explicitly:
    both weight and bias are drawn from ``U(-k, k)`` with
    ``k = sqrt(1 / (in_channels * kernel_size))``. The same scalar bound applies
    to the weight tensor ``(out, in, kernel_size)`` and the bias ``(out,)``.
    """

    __slots__ = ("bound",)

    def __init__(self, in_channels: int, kernel_size: int) -> None:
        self.bound = math.sqrt(1.0 / (in_channels * kernel_size))

    def __call__(self, key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-self.bound, maxval=self.bound)


def _gelu(z: jnp.ndarray) -> jnp.ndarray:
    """Exact (erf) GELU. NF uses ``F.gelu`` (exact); ``jax.nn.gelu`` defaults to
    the tanh approximation, so ``approximate=False`` is required for parity."""
    return jax.nn.gelu(z, approximate=False)


class CustomConv1d(nnx.Module):
    """Forward- or backward-looking causal dilated 1-D convolution.

    Mirrors NF ``CustomConv1d``. Operates in torch's channel-first layout
    ``[B, C, L]`` via ``jax.lax.conv_general_dilated`` with dimension numbers
    ``('NCH', 'OIH', 'NCH')`` and a ``(low, high)`` causal pad:
    ``mode="backward"`` pads on the left (looks into the past), ``mode="forward"``
    pads on the right. ``groups=1`` throughout (as in NF). Weight shape is torch's
    ``(out_channels, in_channels, kernel_size)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        mode: str = "backward",
        *,
        rngs: nnx.Rngs,
    ):
        init = _TorchConvInit(in_channels, kernel_size)
        self.weight = nnx.Param(init(rngs.params(), (out_channels, in_channels, kernel_size)))
        self.bias = nnx.Param(init(rngs.params(), (out_channels,)))
        self.dilation = dilation
        if mode == "backward":
            self.pad = (padding, 0)
        elif mode == "forward":
            self.pad = (0, padding)
        else:
            raise ValueError(f"mode must be 'backward' or 'forward'; got {mode!r}.")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [B, C_in, L] -> [B, C_out, L]."""
        out = jax.lax.conv_general_dilated(
            x,
            self.weight.value,
            window_strides=(1,),
            padding=[self.pad],
            rhs_dilation=(self.dilation,),
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        return out + self.bias.value[None, :, None]


class TCNCell(nnx.Module):
    """One Temporal Convolutional Network cell (NF ``TCNCell``).

    ``conv1`` is a dilated causal conv (``kernel_size=2``); ``conv2`` is a
    pointwise (``kernel_size=1``) conv widening to ``2 * in_channels`` whose
    output is split into a residual update ``h_next`` and a skip contribution
    ``out_next``::

        h = drop(gelu(conv1(h_prev)))
        h_next, out_next = conv2(h).chunk(2, dim=channels)
        return (h_prev + h_next, out_prev + out_next)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        mode: str,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.conv1 = CustomConv1d(
            in_channels, out_channels, kernel_size, padding, dilation, mode, rngs=rngs,
        )
        # conv2: pointwise (kernel_size=1, padding=0, dilation=1, backward is a no-op).
        self.conv2 = CustomConv1d(out_channels, in_channels * 2, 1, rngs=rngs)
        self.drop = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, h_prev: jnp.ndarray, out_prev: jnp.ndarray, deterministic: bool):
        h = self.drop(_gelu(self.conv1(h_prev)), deterministic=deterministic)
        h_next, out_next = jnp.split(self.conv2(h), 2, axis=1)   # split channels
        return h_prev + h_next, out_prev + out_next


class BiTCNNet(nnx.Module):
    """Univariate BiTCN backbone: ``[B, input_size, 1] -> [B, h, 1]``.

    No-exogenous case (backward TCN only). Identity scaling (NF default): the
    network operates in raw scale. The number of backward layers is set so the
    dilated receptive field covers the lookback window::

        n_layers_bwd = ceil(log2((input_size - 1) / (kernel_size - 1) + 1))

    (with ``kernel_size=2`` this is ``ceil(log2(input_size))``). During training
    pass ``deterministic=False``; the ``nnx.Rngs`` dropout stream supplies noise.
    """

    def __init__(self, *, h: int, input_size: int, hidden_size: int, dropout: float, rngs: nnx.Rngs):
        kernel_size = 2
        self.kernel_size = kernel_size
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers_bwd = int(
            math.ceil(math.log2(((input_size - 1) / (kernel_size - 1)) + 1))
        )

        # Input projection: univariate channel (1) -> hidden_size (no exog).
        self.lin_hist = nnx.Linear(
            1, hidden_size,
            kernel_init=_TorchLinearInit(1), bias_init=_TorchLinearInit(1), rngs=rngs,
        )
        self.drop_hist = nnx.Dropout(rate=dropout, rngs=rngs)

        # Backward TCN: dilation doubles each layer.
        self.net_bwd = [
            TCNCell(
                hidden_size, hidden_size, kernel_size,
                padding=(kernel_size - 1) * 2 ** i, dilation=2 ** i,
                mode="backward", dropout=dropout, rngs=rngs,
            )
            for i in range(self.n_layers_bwd)
        ]

        # Temporal projection to the forecast horizon (operates over the L axis).
        self.drop_temporal = nnx.Dropout(rate=dropout, rngs=rngs)
        self.temporal_lin1 = nnx.Linear(
            input_size, hidden_size,
            kernel_init=_TorchLinearInit(input_size),
            bias_init=_TorchLinearInit(input_size), rngs=rngs,
        )
        self.temporal_lin2 = nnx.Linear(
            hidden_size, h,
            kernel_init=_TorchLinearInit(hidden_size),
            bias_init=_TorchLinearInit(hidden_size), rngs=rngs,
        )
        # output_lin_dim_multiplier = 1 (no futr TCN); outputsize_multiplier = 1 (point loss).
        self.output_lin = nnx.Linear(
            hidden_size, 1,
            kernel_init=_TorchLinearInit(hidden_size),
            bias_init=_TorchLinearInit(hidden_size), rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        x = x.astype(jnp.float32)                                   # [B, L, 1]
        x = self.drop_hist(self.lin_hist(x), deterministic=deterministic)   # [B, L, H]
        x = x.transpose(0, 2, 1)                                    # [B, H, L]

        # Backward TCN: thread (hidden state, skip accumulator). NF seeds out=0.
        h_state = x
        out = jnp.zeros_like(x)
        for cell in self.net_bwd:
            h_state, out = cell(h_state, out, deterministic=deterministic)
        x = out                                                    # [B, H, L]

        # Temporal dense layers over the L axis, then to the horizon.
        x = self.drop_temporal(_gelu(self.temporal_lin1(x)), deterministic=deterministic)  # [B, H, H]
        x = self.temporal_lin2(x)                                  # [B, H, h]
        x = x.transpose(0, 2, 1)                                   # [B, h, H]
        return self.output_lin(x)                                  # [B, h, 1]

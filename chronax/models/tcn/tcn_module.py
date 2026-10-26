"""Flax NNX modules for the TCN forecaster (port of neuralforecast.TCN).

The building blocks are ``CausalConv1d`` (dilated causal 1-D convolution),
``TemporalConvolutionEncoder`` (a plain sequential stack of causal convs with
exponentially increasing dilations), the ``MLP`` decoder, and the full
``TCNNet`` (encoder -> context adapter ``Linear(input_size -> h)`` over the
time axis -> optional future-exog residual concat -> per-timestep MLP decoder).

Causality is realized as K left-shifted matmul taps (see ``CausalConv1d``),
mathematically identical to a symmetric pad plus a right-trim. All shift
amounts are static ctor config, so everything traces under ``jax.vmap`` as
``BaseForecaster.conformity_scores`` requires. ``float32`` throughout.
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

    def __eq__(self, other):  # value equality: same-config initializers make
        return type(other) is type(self) and other.fan_in == self.fan_in

    def __hash__(self):  # same-config graphdefs EQUAL, so nnx.jit caches hit
        return hash((type(self), self.fan_in))  # across net instances


class _TorchConvInit:
    """Picklable initializer matching torch ``nn.Conv1d`` default.

    Both weight and bias are drawn from ``U(-k, k)`` with
    ``k = sqrt(1 / (in_channels * kernel_size))``. The same scalar bound applies
    to the weight tensor ``(out, in, kernel_size)`` and the bias ``(out,)``.
    """

    __slots__ = ("bound",)

    def __init__(self, in_channels: int, kernel_size: int) -> None:
        self.bound = math.sqrt(1.0 / (in_channels * kernel_size))

    def __call__(self, key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-self.bound, maxval=self.bound)

    def __eq__(self, other):  # value equality — see _TorchLinearInit
        return type(other) is type(self) and other.bound == self.bound

    def __hash__(self):
        return hash((type(self), self.bound))


# Supported encoder activations.
ACTIVATIONS = {"ReLU": jax.nn.relu, "Tanh": jnp.tanh}


class CausalConv1d(nnx.Module):
    """Causal dilated 1-D convolution followed by an activation.

    Operates in channel-first layout ``[B, C, L]``; causality is realized as K
    left-shifted matmul taps (see ``__call__``). Weight shape is
    ``(out_channels, in_channels, kernel_size)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        activation: str,
        *,
        rngs: nnx.Rngs,
    ):
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(ACTIVATIONS)}; got {activation!r}."
            )
        init = _TorchConvInit(in_channels, kernel_size)
        self.weight = nnx.Param(init(rngs.params(), (out_channels, in_channels, kernel_size)))
        self.bias = nnx.Param(init(rngs.params(), (out_channels,)))
        self.dilation = dilation
        self.pad = (padding, 0)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [B, C_in, L] -> [B, C_out, L].

        Computed as K shifted matmuls (``out[t] = Σ_k W[:,:,k] · x[t-(K-1-k)·d]``)
        rather than ``lax.conv_general_dilated``: the XLA CPU backend pessimizes
        conv primitives inside the ``lax.scan`` training loop, while matmuls keep
        their fast dot path there. K and the shift amounts are static ctor
        config, so this traces unchanged under jit/vmap.
        """
        x = x.astype(self.weight.value.dtype)   # enforce float32 (x64-safe)
        w = self.weight.value                   # [C_out, C_in, K]
        n_taps = w.shape[2]
        xt = jnp.transpose(x, (0, 2, 1))        # [B, L, C_in]
        length = xt.shape[1]
        out = None
        for k in range(n_taps):
            shift = (n_taps - 1 - k) * self.dilation
            xk = jnp.pad(xt, ((0, 0), (shift, 0), (0, 0)))[:, :length, :] if shift else xt
            term = xk @ w[:, :, k].T
            out = term if out is None else out + term
        out = jnp.transpose(out + self.bias.value, (0, 2, 1))
        return ACTIVATIONS[self.activation](out)


class TemporalConvolutionEncoder(nnx.Module):
    """Sequential stack of causal dilated convs.

    Layer i uses ``padding = (kernel_size-1) * dilations[i]``; layer 0 maps
    ``in_channels -> out_channels``, the rest ``out_channels -> out_channels``.
    Input/output are time-first ``[B, L, C]`` (transposed to channel-first
    internally).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilations: tuple,
        activation: str = "ReLU",
        *,
        rngs: nnx.Rngs,
    ):
        layers = []
        for dilation in dilations:
            layers.append(
                CausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    dilation=dilation,
                    activation=activation,
                    rngs=rngs,
                )
            )
            in_channels = out_channels
        self.layers = layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [B, L, C_in] -> [B, L, C_out]."""
        x = jnp.transpose(x, (0, 2, 1))         # [B, C_in, L]
        for layer in self.layers:
            x = layer(x)
        return jnp.transpose(x, (0, 2, 1))      # [B, L, C_out]


class MLP(nnx.Module):
    """MLP decoder head: ``num_layers`` Linears total, ReLU between.

    ``num_layers=1`` is a direct linear projection. For ``num_layers>=2``:
    input Linear, ``num_layers-2`` hidden Linears, output Linear — ReLU after
    every layer but the last. Dropout is omitted (TCN runs it at ``0.0``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}.")

        def _linear(n_in: int, n_out: int) -> nnx.Linear:
            init = _TorchLinearInit(n_in)
            return nnx.Linear(n_in, n_out, kernel_init=init, bias_init=init, rngs=rngs)

        if num_layers == 1:
            self.layers = [_linear(in_features, out_features)]
        else:
            self.layers = (
                [_linear(in_features, hidden_size)]
                + [_linear(hidden_size, hidden_size) for _ in range(num_layers - 2)]
                + [_linear(hidden_size, out_features)]
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


class TCNNet(nnx.Module):
    """Full TCN: encoder -> context adapter -> futr residual concat -> MLP decoder.

    Forward:
      1. concat scaled insample_y with the historical exog and the input slice
         of future-known exog as extra channels -> encoder input
         ``[B, L, 1+H+F]`` (insample, then hist, then futr — matching the
         reference);
      2. ``TemporalConvolutionEncoder`` -> ``[B, L, C]``;
      3. transpose to ``[B, C, L]``, ``context_adapter = Linear(L -> h)`` over the
         time axis -> ``[B, C, h]`` (natively handles ``h > input_size``);
      4. concat the horizon slice of futr exog as extra channels -> ``[B, C+F, h]``
         (historical exog is encoder-only, not part of the decoder residual);
      5. transpose to ``[B, h, C+F]``, per-timestep MLP decoder -> ``[B, h, mult]``.

    The forward is fully deterministic (no dropout or batchnorm), so no RNG or
    train/eval mode flags are needed.
    """

    def __init__(
        self,
        *,
        h: int,
        input_size: int,
        kernel_size: int = 2,
        dilations: tuple = (1, 2, 4, 8, 16),
        encoder_hidden_size: int = 128,
        encoder_activation: str = "ReLU",
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        hist_exog_size: int = 0,
        futr_exog_size: int = 0,
        outputsize_multiplier: int = 1,
        rngs: nnx.Rngs,
    ):
        self.h = h
        self.input_size = input_size
        self.hist_exog_size = hist_exog_size
        self.futr_exog_size = futr_exog_size
        self.hist_encoder = TemporalConvolutionEncoder(
            1 + hist_exog_size + futr_exog_size,
            encoder_hidden_size,
            kernel_size,
            tuple(dilations),
            encoder_activation,
            rngs=rngs,
        )
        ctx_init = _TorchLinearInit(input_size)
        self.context_adapter = nnx.Linear(
            input_size, h, kernel_init=ctx_init, bias_init=ctx_init, rngs=rngs
        )
        self.mlp_decoder = MLP(
            encoder_hidden_size + futr_exog_size,
            outputsize_multiplier,
            decoder_hidden_size,
            decoder_layers,
            rngs=rngs,
        )

    def __call__(self, insample_z: jnp.ndarray, hist_exog: jnp.ndarray | None = None,
                 futr_exog: jnp.ndarray | None = None) -> jnp.ndarray:
        """insample_z: [B, L, 1] scaled target; hist_exog: [B, L, H] or None;
        futr_exog: [B, L+h, F] or None.

        Returns [B, h, outputsize_multiplier] in scaled space.
        """
        x = insample_z.astype(jnp.float32)
        if self.hist_exog_size > 0:
            x = jnp.concatenate([x, hist_exog], axis=2)                       # [B, L, 1+H]
        if self.futr_exog_size > 0:
            x = jnp.concatenate([x, futr_exog[:, : self.input_size]], axis=2)  # [B, L, 1+H+F]
        hidden = self.hist_encoder(x)                     # [B, L, C]
        hidden = jnp.transpose(hidden, (0, 2, 1))         # [B, C, L]
        context = self.context_adapter(hidden)            # [B, C, h]
        if self.futr_exog_size > 0:
            futr_futr = jnp.transpose(futr_exog[:, self.input_size :], (0, 2, 1))  # [B, F, h]
            context = jnp.concatenate([context, futr_futr], axis=1)                # [B, C+F, h]
        context = jnp.transpose(context, (0, 2, 1))       # [B, h, C+F]
        return self.mlp_decoder(context)                  # [B, h, mult]

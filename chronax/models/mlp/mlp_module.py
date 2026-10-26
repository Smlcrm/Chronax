"""Flax NNX module for the MLP forecaster (port of neuralforecast.MLP).

The network flattens its inputs — the scaled insample target, the
historical exogenous window over the input span, and the future-known
exogenous window spanning input and horizon (each present only when its
feature count is nonzero) — into one vector, passes it through ``num_layers``
fully connected layers with a ReLU after every one, and projects with a
separate raw output head whose width is ``h * outputsize_multiplier`` (the
multiplier is loss-driven: 1 for point losses, Q for multi-quantile heads,
``(2+weighted)*K`` for the GMM distribution head). The flatten order is
``[insample | hist | futr]``, matching the reference so a weight transplant
lines up column-for-column. ``float32`` throughout.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _TorchLinearInit:
    """Picklable initializer matching torch ``nn.Linear`` default.

    torch draws both weight and bias from ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``
    (Kaiming-uniform with ``a=sqrt(5)`` reduces to this bound); ``fan_in`` is the
    layer's ``in_features`` (the bias shares the weight's fan_in).
    """

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    def __eq__(self, other):
        # Value equality. Initializer instances are static graphdef leaves; an
        # identity-based __eq__ makes same-config graphdefs UNEQUAL, so the
        # module-level @nnx.jit inference cache could never hit across nets built
        # by a later fit — costing a full recompile on every refit->predict.
        return type(other) is type(self) and other.fan_in == self.fan_in

    def __hash__(self):
        return hash((type(self), self.fan_in))


def _torch_linear(n_in: int, n_out: int, *, rngs: nnx.Rngs) -> nnx.Linear:
    init = _TorchLinearInit(n_in)
    return nnx.Linear(n_in, n_out, kernel_init=init, bias_init=init, rngs=rngs)


class MLPNet(nnx.Module):
    """Full MLP: flatten ``[insample_y | hist_exog | futr_exog]`` -> ReLU'd Linear stack -> raw head.

    Mirrors the reference layer structure exactly: ``num_layers`` Linears (the
    first maps the flattened input to ``hidden_size``, the rest are
    hidden-to-hidden), each followed by ReLU, then a separate un-activated
    ``out`` head. The flattened input is the scaled insample target of length
    ``input_size``, concatenated (when the corresponding feature count is
    nonzero) with the row-major flattened historical exog window of shape
    ``[input_size, F_hist]`` and then the future-known exog window of shape
    ``[input_size + h, F_futr]`` — hist before futr, matching the reference.
    """

    def __init__(
        self,
        *,
        h: int,
        input_size: int,
        hist_exog_size: int = 0,
        futr_exog_size: int = 0,
        num_layers: int = 2,
        hidden_size: int = 1024,
        outputsize_multiplier: int = 1,
        rngs: nnx.Rngs,
    ):
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}.")
        self.h = h
        self.input_size = input_size
        self.hist_exog_size = hist_exog_size
        self.futr_exog_size = futr_exog_size
        self.outputsize_multiplier = outputsize_multiplier
        first_in = (input_size + hist_exog_size * input_size
                    + futr_exog_size * (input_size + h))
        layers = [_torch_linear(first_in, hidden_size, rngs=rngs)]
        layers += [
            _torch_linear(hidden_size, hidden_size, rngs=rngs)
            for _ in range(num_layers - 1)
        ]
        self.mlp = layers
        self.out = _torch_linear(hidden_size, h * outputsize_multiplier, rngs=rngs)

    def __call__(self, insample_z: jnp.ndarray, hist_exog: jnp.ndarray | None = None,
                 futr_exog: jnp.ndarray | None = None) -> jnp.ndarray:
        """insample_z: [B, L, 1] scaled target; hist_exog: [B, L, F] or None;
        futr_exog: [B, L+h, F] or None.

        Returns [B, h, outputsize_multiplier] (scaled space for point/quantile
        heads; raw pre-``domain_map`` parameters for distribution heads).
        """
        x = insample_z.astype(jnp.float32)[..., 0]                 # [B, L]
        if self.hist_exog_size > 0:
            x = jnp.concatenate([x, hist_exog.reshape(x.shape[0], -1)], axis=1)
        if self.futr_exog_size > 0:
            x = jnp.concatenate([x, futr_exog.reshape(x.shape[0], -1)], axis=1)
        for layer in self.mlp:
            x = jax.nn.relu(layer(x))
        y = self.out(x)
        return y.reshape(x.shape[0], self.h, self.outputsize_multiplier)

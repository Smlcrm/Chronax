"""Flax NNX module for the DeepNPTS forecaster.

Univariate JAX/Flax-NNX port of ``neuralforecast.DeepNPTS`` (Deep Non-Parametric
Time Series forecaster; Rangapuram et al., 2023, arXiv:2312.14657). DeepNPTS is a
baseline deep model: a small MLP reads the lookback window and emits, for each
horizon step, a softmax weight over every position of the window; the forecast is
the weighted sum of the in-sample values. It is a *learned non-parametric
resample* of the context, not a parametric-distribution head. Point forecaster,
univariate.

Chronax (like the iTransformer / BiTCN / VanillaTransformer ports) targets the
**no-exogenous** case, so NF's general ``forward`` reduces to::

    z = insample_y.reshape(B, L)          # [B, L, 1] -> [B, L]  (== input_size)
    z = MLP(z)                            # n_layers x (Linear -> ReLU -> [BN] -> [Drop])
    w = out_linear(z).reshape(B, L, h)    # -> [B, L, h]
    w = softmax(w, axis=L)                # weights over the window
    forecast = sum(w * insample_y, L)     # [B, L, h]*[B, L, 1] -> [B, h] -> [B, h, 1]

Parity with neuralforecast (PyTorch): torch ``nn.Linear`` default init
(``_TorchLinearInit``: ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``), exact ReLU, and
softmax taken over the window (``L``) axis. ``float32`` throughout.

``batch_norm`` is off by default (NF defaults it True). When enabled it uses
``nnx.BatchNorm`` with ``momentum=0.9`` (flax decay convention, equivalent to
torch ``BatchNorm1d``'s ``momentum=0.1``) and ``epsilon=1e-5`` (torch default).
BatchNorm reads ``use_running_average=deterministic``: batch statistics during
training, accumulated running statistics at inference.
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


class DeepNPTSNet(nnx.Module):
    """Univariate DeepNPTS backbone: ``[B, input_size, 1] -> [B, h, 1]``.

    No-exogenous case: the MLP input dimension equals ``input_size``. ``n_layers``
    blocks of ``Linear -> ReLU -> [BatchNorm] -> [Dropout]`` feed a final linear
    that produces ``input_size * h`` logits; these are reshaped to ``[B, L, h]``,
    softmaxed over ``L`` (the window axis), and used to weight-sum the raw
    in-sample values into the horizon. Identity scaling (NF default): the network
    operates in raw scale. During training pass ``deterministic=False``; the
    ``nnx.Rngs`` dropout stream supplies noise and BatchNorm updates running stats.
    """

    def __init__(
        self,
        *,
        h: int,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        batch_norm: bool,
        rngs: nnx.Rngs,
    ):
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1; got {n_layers}.")
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.dropout = dropout

        # MLP: first layer maps input_size -> hidden_size, the rest hidden -> hidden.
        in_dims = [input_size] + [hidden_size] * (n_layers - 1)
        self.linears = [
            nnx.Linear(
                d_in, hidden_size,
                kernel_init=_TorchLinearInit(d_in),
                bias_init=_TorchLinearInit(d_in), rngs=rngs,
            )
            for d_in in in_dims
        ]
        # BatchNorm: momentum=0.9 matches torch BatchNorm1d momentum=0.1
        # (flax uses the running-average decay, torch uses the update weight).
        self.norms = (
            [nnx.BatchNorm(hidden_size, momentum=0.9, epsilon=1e-5, rngs=rngs)
             for _ in range(n_layers)]
            if batch_norm else None
        )
        self.drops = (
            [nnx.Dropout(rate=dropout, rngs=rngs) for _ in range(n_layers)]
            if dropout > 0.0 else None
        )
        # Output projection: hidden_size -> input_size * h (one weight per (pos, step)).
        self.out = nnx.Linear(
            hidden_size, input_size * h,
            kernel_init=_TorchLinearInit(hidden_size),
            bias_init=_TorchLinearInit(hidden_size), rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        x = x.astype(jnp.float32)                          # [B, L, 1]
        insample = x                                       # keep raw values
        batch_size, seq_len = x.shape[0], x.shape[1]
        z = x.reshape(batch_size, seq_len)                 # [B, L] (== input_size)

        for i, lin in enumerate(self.linears):
            z = jax.nn.relu(lin(z))                         # [B, H]
            if self.norms is not None:
                z = self.norms[i](z, use_running_average=deterministic)
            if self.drops is not None:
                z = self.drops[i](z, deterministic=deterministic)

        w = self.out(z)                                    # [B, L*h]
        w = w.reshape(batch_size, seq_len, self.h)         # [B, L, h]
        w = jax.nn.softmax(w, axis=1)                      # softmax over the window
        forecast = jnp.sum(w * insample, axis=1)           # [B, L, h]*[B, L, 1] -> [B, h]
        return forecast[..., None]                         # [B, h, 1]

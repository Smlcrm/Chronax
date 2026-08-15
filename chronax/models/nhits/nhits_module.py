"""Flax NNX module for the NHITS forecaster (port of neuralforecast.NHITS).

NHITS (Challu et al. 2023, arXiv:2201.12886) stacks fully connected blocks
with doubly residual links. Each block pools the (time-reversed) input window
at a stack-specific rate, regresses a coefficient vector ``theta`` through an
MLP, splits it into a backcast — subtracted from the running residual — and a
small set of forecast knots, and interpolates the knots up to the full
horizon. Multi-rate pooling and hierarchical interpolation specialize each
stack to a frequency band. Interpolation is expressed as a constant-matrix
GEMM (torch's ``F.interpolate`` is a fixed linear operator once sizes and
mode are known), so the forward stays a pure matmul pipeline. ``float32``
throughout.
"""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

ACTIVATIONS = ("ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid")
POOLING = ("MaxPool1d", "AvgPool1d")
INTERPOLATION = ("linear", "nearest", "cubic")

_ACTIVATION_FNS = {
    "ReLU": jax.nn.relu,
    "Softplus": jax.nn.softplus,
    "Tanh": jnp.tanh,
    "SELU": jax.nn.selu,
    "LeakyReLU": lambda x: jax.nn.leaky_relu(x, negative_slope=0.01),
    "Sigmoid": jax.nn.sigmoid,
}


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


@functools.lru_cache(maxsize=None)
def _interp_weights(n_in: int, n_out: int, mode: str) -> np.ndarray:
    """Constant ``[n_in, n_out]`` matrix replicating ``torch.nn.functional.
    interpolate`` along the last axis, so ``knots @ W`` equals the reference
    interpolation.

    ``linear``: align_corners=False — source position ``(j+0.5)*n_in/n_out - 0.5``
    clamped at zero, two taps. ``nearest``: ``floor(j * n_in/n_out)``. ``cubic``:
    the reference routes cubic through bicubic on a height-1 image and reads row
    0; every row tap then clamps to the single row (cubic weights sum to one),
    which collapses to 1-D cubic convolution along the knot axis — Keys kernel
    ``A=-0.75``, unclamped source position, border-clamped taps. Keyed on
    config-derived statics only; built in float64 once per configuration.
    """
    W = np.zeros((n_in, n_out))
    if mode == "nearest":
        scale = n_in / n_out
        for j in range(n_out):
            W[min(int(np.floor(j * scale)), n_in - 1), j] = 1.0
    elif mode == "linear":
        scale = n_in / n_out
        for j in range(n_out):
            src = max(scale * (j + 0.5) - 0.5, 0.0)
            i0 = min(int(np.floor(src)), n_in - 1)
            t = src - i0
            W[i0, j] += 1.0 - t
            W[min(i0 + 1, n_in - 1), j] += t
    elif mode == "cubic":
        A = -0.75

        def keys(x: float) -> float:
            x = abs(x)
            if x <= 1.0:
                return (A + 2.0) * x**3 - (A + 3.0) * x**2 + 1.0
            if x < 2.0:
                return A * (x**3 - 5.0 * x**2 + 8.0 * x - 4.0)
            return 0.0

        scale = n_in / n_out
        for j in range(n_out):
            src = scale * (j + 0.5) - 0.5
            i0 = int(np.floor(src))
            t = src - i0
            for tap, w in zip(
                range(i0 - 1, i0 + 3), (keys(1.0 + t), keys(t), keys(1.0 - t), keys(2.0 - t))
            ):
                W[min(max(tap, 0), n_in - 1), j] += w
    else:
        raise ValueError(f"interpolation_mode must be one of {INTERPOLATION}; got {mode!r}.")
    W.setflags(write=False)
    return W


def _pool1d(x: jnp.ndarray, k: int, avg: bool) -> jnp.ndarray:
    """``MaxPool1d``/``AvgPool1d(kernel_size=k, stride=k, ceil_mode=True)`` over
    the last axis. The average divides by the in-bounds element count only —
    torch excludes the ceil-mode overhang from the divisor when there is no
    explicit padding."""
    if k == 1:
        return x
    L = x.shape[-1]
    n_out = -(-L // k)
    pad = [(0, 0)] * (x.ndim - 1) + [(0, n_out * k - L)]
    if avg:
        sums = jnp.pad(x, pad).reshape(*x.shape[:-1], n_out, k).sum(-1)
        counts = np.minimum(k, L - k * np.arange(n_out))
        return sums / jnp.asarray(counts, x.dtype)
    return jnp.pad(x, pad, constant_values=-jnp.inf).reshape(*x.shape[:-1], n_out, k).max(-1)


class NHITSBlock(nnx.Module):
    """One NHITS block: pool -> MLP -> theta -> (backcast, interpolated forecast).

    The MLP mirrors the reference layer structure exactly: an entry Linear onto
    ``mlp_units[0][0]`` with NO activation of its own, then one activated (and
    optionally dropped-out) Linear per ``mlp_units`` pair, then a raw theta
    head of width ``input_size + out_features * n_knots``. The backcast slice
    stays in the caller's (time-reversed) orientation; the knot slice is
    reshaped ``[B, out_features, n_knots]`` (feature-major, the reference's
    row-major reshape) and interpolated up to ``h`` with the constant matrix.
    ``PReLU`` is a single learnable scalar shared by all of the block's
    activations, matching the reference's one reused module instance.
    """

    def __init__(
        self,
        *,
        input_size: int,
        h: int,
        n_knots: int,
        out_features: int,
        mlp_units: tuple,
        n_pool_kernel_size: int,
        pooling_mode: str,
        interpolation_mode: str,
        dropout_prob: float,
        activation: str,
        futr_exog_size: int,
        rngs: nnx.Rngs,
    ):
        self.input_size = input_size
        self.h = h
        self.n_knots = n_knots
        self.out_features = out_features
        self.k = n_pool_kernel_size
        self.avg_pool = pooling_mode == "AvgPool1d"
        self.interpolation_mode = interpolation_mode
        self.dropout_prob = float(dropout_prob)
        self.activation = activation
        self.futr_exog_size = futr_exog_size
        pooled_hist = -(-input_size // self.k)
        pooled_futr = -(-(input_size + h) // self.k)
        first_in = pooled_hist + futr_exog_size * pooled_futr
        n_theta = input_size + out_features * n_knots
        widths = ((first_in, mlp_units[0][0]),) + mlp_units
        self.layers = [_torch_linear(a, b, rngs=rngs) for a, b in widths]
        self.out_layer = _torch_linear(mlp_units[-1][1], n_theta, rngs=rngs)
        if activation == "PReLU":
            self.prelu_a = nnx.Param(jnp.full((1,), 0.25, jnp.float32))
        if self.dropout_prob > 0.0:
            self.dropouts = [
                nnx.Dropout(rate=self.dropout_prob, rngs=rngs) for _ in mlp_units
            ]

    def __call__(
        self,
        insample_y: jnp.ndarray,
        futr_exog: jnp.ndarray | None = None,
        *,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """insample_y: [B, L] time-reversed residuals; futr_exog: [B, L+h, F]
        in natural time order (the reference pools but never reverses exog).
        Returns ``(backcast [B, L] in reversed orientation, forecast [B, h, Q])``.
        """
        x = _pool1d(insample_y, self.k, self.avg_pool)
        if self.futr_exog_size > 0:
            f = _pool1d(futr_exog.transpose(0, 2, 1), self.k, self.avg_pool)
            x = jnp.concatenate([x, f.transpose(0, 2, 1).reshape(x.shape[0], -1)], axis=1)
        if self.activation == "PReLU":
            a = self.prelu_a.value
            act = lambda v: jnp.where(v >= 0, v, a * v)
        else:
            act = _ACTIVATION_FNS[self.activation]
        x = self.layers[0](x)
        for i, layer in enumerate(self.layers[1:]):
            x = act(layer(x))
            if self.dropout_prob > 0.0:
                x = self.dropouts[i](x, deterministic=deterministic)
        theta = self.out_layer(x)
        backcast = theta[:, : self.input_size]
        knots = theta[:, self.input_size :].reshape(-1, self.out_features, self.n_knots)
        W = jnp.asarray(
            _interp_weights(self.n_knots, self.h, self.interpolation_mode), theta.dtype
        )
        forecast = knots @ W                                   # [B, Q, h]
        return backcast, forecast.transpose(0, 2, 1)           # [B, h, Q]


class NHITSNet(nnx.Module):
    """Full NHITS: doubly residual stack of :class:`NHITSBlock`.

    The scaled insample window is time-reversed once at entry; every block
    consumes and backcasts in that reversed orientation. The forecast
    accumulator starts at the Naive1 anchor — the window's last value,
    broadcast onto ALL ``outputsize_multiplier`` output features (for a
    distribution head that includes the raw pre-``domain_map`` parameters;
    reference behavior, kept). ``mlp_units`` is shared whole by every block
    (the reference passes the full list to each block rather than indexing
    per stack), so each block carries ``1 + len(mlp_units)`` hidden Linears
    plus the theta head.
    """

    def __init__(
        self,
        *,
        h: int,
        input_size: int,
        futr_exog_size: int = 0,
        n_blocks=(1, 1, 1),
        mlp_units=((512, 512), (512, 512), (512, 512)),
        n_pool_kernel_size=(2, 2, 1),
        n_freq_downsample=(4, 2, 1),
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        dropout_prob_theta: float = 0.0,
        activation: str = "ReLU",
        outputsize_multiplier: int = 1,
        rngs: nnx.Rngs,
    ):
        n_blocks = tuple(int(b) for b in n_blocks)
        kernels = tuple(int(k) for k in n_pool_kernel_size)
        freqs = tuple(int(f) for f in n_freq_downsample)
        units = tuple((int(a), int(b)) for a, b in mlp_units)
        if not (len(n_blocks) == len(kernels) == len(freqs)):
            raise ValueError(
                "n_blocks, n_pool_kernel_size and n_freq_downsample must have equal "
                f"lengths (one entry per stack); got {len(n_blocks)}, {len(kernels)}, "
                f"{len(freqs)}."
            )
        if any(k < 1 for k in kernels):
            raise ValueError(f"n_pool_kernel_size entries must be >= 1; got {kernels}.")
        if any(f < 1 for f in freqs):
            raise ValueError(f"n_freq_downsample entries must be >= 1; got {freqs}.")
        if any(b < 0 for b in n_blocks) or sum(n_blocks) < 1:
            raise ValueError(f"n_blocks must be non-negative with at least one block; got {n_blocks}.")
        if not units:
            raise ValueError("mlp_units must contain at least one [in, out] pair.")
        for (_, b0), (a1, _) in zip(units, units[1:]):
            if b0 != a1:
                raise ValueError(
                    f"mlp_units pairs must chain (units[i][1] == units[i+1][0]); got {units}."
                )
        if pooling_mode not in POOLING:
            raise ValueError(f"pooling_mode must be one of {POOLING}; got {pooling_mode!r}.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}; got {activation!r}.")
        if interpolation_mode not in INTERPOLATION:
            raise ValueError(
                f"interpolation_mode must be one of {INTERPOLATION}; got {interpolation_mode!r}."
            )
        if interpolation_mode == "cubic" and outputsize_multiplier > 1:
            raise ValueError(
                "Cubic interpolation is not available with multi-output heads "
                "(quantile or distribution losses); use 'linear' or 'nearest'."
            )
        if not 0.0 <= dropout_prob_theta < 1.0:
            raise ValueError(f"dropout_prob_theta must be in [0, 1); got {dropout_prob_theta}.")
        self.h = h
        self.input_size = input_size
        self.futr_exog_size = futr_exog_size
        self.outputsize_multiplier = outputsize_multiplier
        blocks = []
        for i in range(len(n_blocks)):
            n_knots = max(h // freqs[i], 1)
            for _ in range(n_blocks[i]):
                blocks.append(
                    NHITSBlock(
                        input_size=input_size,
                        h=h,
                        n_knots=n_knots,
                        out_features=outputsize_multiplier,
                        mlp_units=units,
                        n_pool_kernel_size=kernels[i],
                        pooling_mode=pooling_mode,
                        interpolation_mode=interpolation_mode,
                        dropout_prob=dropout_prob_theta,
                        activation=activation,
                        futr_exog_size=futr_exog_size,
                        rngs=rngs,
                    )
                )
        self.blocks = blocks

    def __call__(
        self,
        insample_z: jnp.ndarray,
        futr_exog: jnp.ndarray | None = None,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """insample_z: [B, L, 1] scaled target; futr_exog: [B, L+h, F] or None.

        Returns [B, h, outputsize_multiplier] (scaled space for point/quantile
        heads; raw pre-``domain_map`` parameters for distribution heads).
        """
        y = insample_z.astype(jnp.float32)[..., 0]             # [B, L]
        residuals = y[:, ::-1]
        forecast = y[:, -1:, None]                             # [B, 1, 1] Naive1 anchor
        for block in self.blocks:
            backcast, block_forecast = block(
                residuals, futr_exog, deterministic=deterministic
            )
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return forecast

"""Flax NNX modules for the KAN forecaster.

Faithful port of neuralforecast.KAN's KANLinear: each layer learns the activation
on every input edge as a B-spline. Weights are kept in torch layout so the
weight-parity gate is a pure copy. Univariate; all-float32.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _Grid(nnx.Variable):
    """Non-trainable B-spline knot grid. ``nnx.Optimizer(wrt=nnx.Param)`` skips it;
    ``nnx.split`` captures it (so pickle restores it)."""


def build_grid(in_features: int, grid_size: int, spline_order: int,
               grid_range: tuple[float, float]) -> jnp.ndarray:
    """Deterministic knot grid [in_features, grid_size + 2*spline_order + 1] (NF kan.py:42-51)."""
    lo, hi = grid_range
    h = (hi - lo) / grid_size
    knots = jnp.arange(-spline_order, grid_size + spline_order + 1, dtype=jnp.float32) * h + lo
    return jnp.broadcast_to(knots[None, :], (in_features, knots.shape[0]))


def b_splines(x: jnp.ndarray, grid: jnp.ndarray, spline_order: int) -> jnp.ndarray:
    """Cox-de Boor B-spline bases. x:[B, in], grid:[in, G] -> [B, in, grid_size+spline_order]."""
    x = x[..., None]                                                   # [B, in, 1]
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(jnp.float32)
    for k in range(1, spline_order + 1):                              # static unroll
        bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) \
              + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
    return bases


def _kaiming_uniform(key, shape, fan_in: int, a: float) -> jnp.ndarray:
    """torch.nn.init.kaiming_uniform_ for a weight whose fan_in is given explicitly."""
    gain = math.sqrt(2.0 / (1.0 + a * a))            # calculate_gain('leaky_relu', a)
    bound = math.sqrt(3.0) * gain / math.sqrt(fan_in)
    return jax.random.uniform(key, shape, jnp.float32, minval=-bound, maxval=bound)


def _curve2coeff(x_pts: jnp.ndarray, y: jnp.ndarray, grid: jnp.ndarray,
                 spline_order: int) -> jnp.ndarray:
    """Least-squares spline coeffs. x_pts:[N, in], y:[N, in, out] -> [out, in, coeff] (NF kan.py:132-162)."""
    A = jnp.transpose(b_splines(x_pts, grid, spline_order), (1, 0, 2))   # [in, N, coeff]
    B = jnp.transpose(y, (1, 0, 2))                                       # [in, N, out]
    sol = jax.vmap(lambda a, b: jnp.linalg.lstsq(a, b)[0])(A, B)          # [in, coeff, out]
    return jnp.transpose(sol, (2, 0, 1))                                  # [out, in, coeff]


class KANLinear(nnx.Module):
    """A KAN edge layer: base = Linear(SiLU(x), base_weight); spline = Linear(b_splines(x), spline*scaler)."""

    def __init__(self, in_features, out_features, *, grid_size, spline_order, scale_noise,
                 scale_base, scale_spline, enable_standalone_scale_spline, grid_range, rngs: nnx.Rngs):
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        grid = build_grid(in_features, grid_size, spline_order, grid_range)
        self.grid = _Grid(grid)

        kb, ks, kn = jax.random.split(rngs.params(), 3)
        self.base_weight = nnx.Param(
            _kaiming_uniform(kb, (out_features, in_features), in_features, math.sqrt(5) * scale_base)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nnx.Param(
                _kaiming_uniform(ks, (out_features, in_features), in_features, math.sqrt(5) * scale_spline)
            )
        # spline_weight via curve2coeff on noise at the interior grid points (NF kan.py:75-90)
        noise = (jax.random.uniform(kn, (grid_size + 1, in_features, out_features), jnp.float32) - 0.5) \
                * scale_noise / grid_size
        x_pts = grid.T[spline_order:-spline_order]                        # [grid_size+1, in]
        coeff = _curve2coeff(x_pts, noise, grid, spline_order)
        self.spline_weight = nnx.Param(coeff if enable_standalone_scale_spline else scale_spline * coeff)

    def _scaled_spline_weight(self) -> jnp.ndarray:
        if self.enable_standalone_scale_spline:
            return self.spline_weight.value * self.spline_scaler.value[..., None]
        return self.spline_weight.value

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [B, in_features] -> [B, out_features]."""
        x = x.astype(jnp.float32)
        base_out = jax.nn.silu(x) @ self.base_weight.value.T
        bs = b_splines(x, self.grid.value, self.spline_order)            # [B, in, coeff]
        scaled = self._scaled_spline_weight()                            # [out, in, coeff]
        spline_out = bs.reshape(bs.shape[0], -1) @ scaled.reshape(self.out_features, -1).T
        return base_out + spline_out

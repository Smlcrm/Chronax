"""Scaler protocol for GRU (and future neural models) + RobustScaler.

The protocol intentionally exposes (stats, transform, inverse) as separate
methods so the same shift/scale can be reused for inverse-transforming
predictions without re-computing them. Pure functions; no module state.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp

# scipy.stats.norm.ppf(0.75) — relates MAD to std under normality.
_MAD_TO_STD = 0.6744897501960817
_EPS = 1e-6


@runtime_checkable
class Scaler(Protocol):
    """Per-window scaler. Implementations are pure."""

    def stats(
        self, x: jnp.ndarray, axis: int = 1, mask: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ...

    def transform(self, x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        ...

    def inverse(self, z: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        ...


def _masked_median(x: jnp.ndarray, mask: jnp.ndarray | None, axis: int) -> jnp.ndarray:
    if mask is None:
        return jnp.median(x, axis=axis, keepdims=True)
    x_nan = jnp.where(mask, x, jnp.nan)
    return jnp.nanmedian(x_nan, axis=axis, keepdims=True)


def _masked_mean(x: jnp.ndarray, mask: jnp.ndarray | None, axis: int) -> jnp.ndarray:
    if mask is None:
        return jnp.mean(x, axis=axis, keepdims=True)
    x_nan = jnp.where(mask, x, jnp.nan)
    return jnp.nanmean(x_nan, axis=axis, keepdims=True)


class RobustScaler:
    """Median + MAD scaler with 0.6745*std fallback when MAD=0.

    The fallback uses the Gaussian relationship `MAD ≈ 0.6745·σ` to estimate
    MAD when the empirical MAD degenerates to zero (e.g. near-constant input).
    """

    def stats(
        self, x: jnp.ndarray, axis: int = 1, mask: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        median = _masked_median(x, mask, axis)
        mad = _masked_median(jnp.abs(x - median), mask, axis)
        mean = _masked_mean(x, mask, axis)
        std = jnp.sqrt(_masked_mean((x - mean) ** 2, mask, axis))
        fallback = std * _MAD_TO_STD
        scale = jnp.where(mad == 0.0, fallback, mad)
        scale = jnp.where(scale == 0.0, 1.0, scale) + _EPS
        return median, scale

    def transform(self, x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return (x - shift) / scale

    def inverse(self, z: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return z * scale + shift

"""Per-window scaler for DilatedRNN: RobustScaler (median / MAD).

neuralforecast's ``DilatedRNN`` defaults to ``scaler_type="robust"``, which is
``TemporalNorm`` with ``robust_statistics`` — median as the shift and the median
absolute deviation as the scale, computed over the TIME axis of each insample
window, with the same statistics reused to scale the target and to invert the
prediction.

The protocol intentionally exposes (stats, transform, inverse) as separate
methods so the same shift/scale can be reused for inverse-transforming
predictions without re-computing them. Pure functions; no module state.

Self-contained per the per-package convention used across the neural ports (an
identical RobustScaler lives in ``chronax/models/gru/gru_scaler.py``).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp

# scipy.stats.norm.ppf(0.75) — relates MAD to std under normality. Matches the
# constant NF's robust_statistics uses for its MAD==0 fallback.
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


class IdentityScaler:
    """No-op scaler — the ``scaler_type="identity"`` path."""

    def stats(
        self, x: jnp.ndarray, axis: int = 1, mask: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        shape = list(x.shape)
        shape[axis] = 1
        return (jnp.zeros(shape, dtype=jnp.float32), jnp.ones(shape, dtype=jnp.float32))

    def transform(self, x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return x

    def inverse(self, z: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return z


class StandardScaler:
    """Mean / population-std scaler — the ``scaler_type="standard"`` path."""

    def stats(
        self, x: jnp.ndarray, axis: int = 1, mask: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean = _masked_mean(x, mask, axis)
        std = jnp.sqrt(_masked_mean((x - mean) ** 2, mask, axis))
        std = jnp.where(std == 0.0, 1.0, std) + _EPS
        return mean, std

    def transform(self, x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return (x - shift) / scale

    def inverse(self, z: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return z * scale + shift


class RobustScaler:
    """Median + MAD scaler with 0.6745*std fallback when MAD=0.

    Mirrors NF ``robust_statistics``: the fallback uses the Gaussian relationship
    ``MAD ~= 0.6745*sigma`` to estimate MAD when the empirical MAD degenerates to
    zero (e.g. a near-constant window), then forces any remaining zero to 1.0 and
    adds ``eps``.
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


SCALERS: dict[str, type] = {
    "robust": RobustScaler,
    "standard": StandardScaler,
    "identity": IdentityScaler,
}


def resolve(scaler_type: "str | Scaler") -> Scaler:
    """Return a Scaler instance from a registry name (or pass an instance through)."""
    if isinstance(scaler_type, str):
        if scaler_type not in SCALERS:
            raise ValueError(
                f"Unknown scaler_type {scaler_type!r}. Available: {sorted(SCALERS)}."
            )
        return SCALERS[scaler_type]()
    return scaler_type

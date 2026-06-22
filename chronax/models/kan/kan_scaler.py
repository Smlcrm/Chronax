"""Per-window scalers for KAN: identity (NF default) and robust (median/MAD).

Self-contained (no cross-model import). Pure functions exposing
(stats, transform, inverse) so the same shift/scale is reused for inverse.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp

_MAD_TO_STD = 0.6744897501960817  # scipy.stats.norm.ppf(0.75)
_EPS = 1e-6


@runtime_checkable
class Scaler(Protocol):
    def stats(self, x: jnp.ndarray, axis: int = 1) -> tuple[jnp.ndarray, jnp.ndarray]: ...
    def transform(self, x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray: ...
    def inverse(self, z: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray: ...


class IdentityScaler:
    """No-op scaler (shift=0, scale=1). Matches neuralforecast's scaler_type='identity'."""

    def stats(self, x: jnp.ndarray, axis: int = 1) -> tuple[jnp.ndarray, jnp.ndarray]:
        shape = list(x.shape)
        shape[axis] = 1
        return jnp.zeros(tuple(shape), x.dtype), jnp.ones(tuple(shape), x.dtype)

    def transform(self, x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return x

    def inverse(self, z: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return z


class RobustScaler:
    """Median + MAD scaler with 0.6745*std fallback when MAD=0."""

    def stats(self, x: jnp.ndarray, axis: int = 1) -> tuple[jnp.ndarray, jnp.ndarray]:
        median = jnp.median(x, axis=axis, keepdims=True)
        mad = jnp.median(jnp.abs(x - median), axis=axis, keepdims=True)
        mean = jnp.mean(x, axis=axis, keepdims=True)
        std = jnp.sqrt(jnp.mean((x - mean) ** 2, axis=axis, keepdims=True))
        scale = jnp.where(mad == 0.0, std * _MAD_TO_STD, mad)
        scale = jnp.where(scale == 0.0, 1.0, scale) + _EPS
        return median, scale

    def transform(self, x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return (x - shift) / scale

    def inverse(self, z: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return z * scale + shift


def resolve_scaler(scaler: "str | Scaler") -> Scaler:
    """Resolve a scaler from a name (``'identity'``/``'robust'``) or a Scaler instance."""
    if isinstance(scaler, str):
        if scaler == "identity":
            return IdentityScaler()
        if scaler == "robust":
            return RobustScaler()
        raise ValueError(f"Unknown scaler {scaler!r}. Available: 'identity', 'robust'.")
    return scaler

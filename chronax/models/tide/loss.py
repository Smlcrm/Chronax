"""Loss functions for Chronax TiDE.

Mirrors the RNN / N-BEATS loss API::

    masked_mae(y, y_hat, mask=None) -> scalar
    masked_mse(y, y_hat, mask=None) -> scalar

Shapes follow the batch convention: ``y`` and ``y_hat`` are ``[B, h, output_size]``
(or any broadcast-compatible shape). The optional ``mask`` is ``[B, h, 1]``.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp

LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def _divide_no_nan(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    safe_b = jnp.where(b == 0.0, 1.0, b)
    out = a / safe_b
    return jnp.where(b == 0.0, jnp.zeros_like(out), out)


def _weighted_mean(losses: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return _divide_no_nan(jnp.sum(losses * weights), jnp.sum(weights))


def masked_mae(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Mean Absolute Error, optionally masked.

    Args:
        y:     Target values ``[B, h, output_size]`` (or broadcastable).
        y_hat: Predicted values, same shape as ``y``.
        mask:  Float mask broadcastable to ``y``; 1 = include, 0 = exclude.

    Returns:
        Scalar loss value.
    """
    losses = jnp.abs(y - y_hat)
    if mask is None:
        return jnp.mean(losses)
    return _weighted_mean(losses, mask)


def masked_mse(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Mean Squared Error, optionally masked."""
    losses = (y - y_hat) ** 2
    if mask is None:
        return jnp.mean(losses)
    return _weighted_mean(losses, mask)


LOSSES = {
    "mae": masked_mae,
    "mse": masked_mse,
}


def resolve(loss):
    """Return a callable loss from either a registry string or a callable."""
    if callable(loss):
        return loss
    if loss not in LOSSES:
        raise ValueError(
            f"Unknown loss {loss!r}. Available: {sorted(LOSSES)}. "
            "Pass a callable for custom losses."
        )
    return LOSSES[loss]

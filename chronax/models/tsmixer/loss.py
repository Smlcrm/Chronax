"""Masked point-loss functions for the Chronax TSMixer model.

Pure JAX implementations of MAE and MSE with multiplicative masking, JIT- and
grad-compatible. Shapes follow the ``[B, H, N]`` convention produced by TSMixer
(batch, horizon, n_series), but the functions are generic and work for any
matching broadcast-compatible shapes.

These mirror the loss functions in :mod:`chronax.models.rnn.loss`; they are
kept as a separate copy so each model sub-package is self-contained.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp

LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _divide_no_nan(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Safe division returning 0 wherever ``b == 0`` (including 0/0).

    Avoids ``nan`` poisoning in ``value_and_grad`` by keeping gradients zero
    through the masked positions rather than using ``jax.nan_to_num``.
    """
    safe_b = jnp.where(b == 0.0, 1.0, b)
    return jnp.where(b == 0.0, jnp.zeros_like(a), a / safe_b)


def _weighted_mean(losses: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """``sum(losses * weights) / sum(weights)`` with safe-divide semantics."""
    return _divide_no_nan(jnp.sum(losses * weights), jnp.sum(weights))


def _compute_weights(
    y: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    horizon_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Combine ``mask`` and optional per-horizon weights into one tensor.

    ``mask`` defaults to all-ones. ``horizon_weight`` is broadcast from
    ``[H]`` to match ``y``'s shape ``[B, H, N]``.
    """
    if mask is None:
        mask = jnp.ones_like(y)
    if horizon_weight is None:
        return mask
    hw = horizon_weight[None, :, None]
    return mask * jnp.broadcast_to(hw, mask.shape)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def masked_mae(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    horizon_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Mean Absolute Error with multiplicative masking."""
    weights = _compute_weights(y=y, mask=mask, horizon_weight=horizon_weight)
    return _weighted_mean(jnp.abs(y - y_hat), weights)


def masked_mse(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    horizon_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Mean Squared Error with multiplicative masking."""
    weights = _compute_weights(y=y, mask=mask, horizon_weight=horizon_weight)
    return _weighted_mean((y - y_hat) ** 2, weights)


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

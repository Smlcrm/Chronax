"""Masked point-loss functions for the Chronax RNN model.

These mirror Nixtla NeuralForecast's ``BasePointLoss`` family (``MAE`` /
``MSE``) but in pure JAX, so they are JIT- and grad-friendly:

* multiplicative masking via ``jnp.where``-style safe arithmetic;
* numerically stable weighted mean that returns 0 when the mask is empty
  (matches ``_divide_no_nan`` in Nixtla NeuralForecast's PyTorch losses);
* optional per-horizon weighting.

All shapes follow the ``[B, H, output_size]`` convention used by the model.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _divide_no_nan(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Safe division that returns 0 wherever ``b == 0`` (incl. 0/0).

    Implemented without ``jax.nan_to_num`` so that gradients through the
    "would-be-NaN" branch are also zero (avoids ``nan`` poisoning during
    ``value_and_grad``).
    """
    safe_b = jnp.where(b == 0.0, 1.0, b)
    out = a / safe_b
    return jnp.where(b == 0.0, jnp.zeros_like(out), out)


def _weighted_mean(losses: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """``sum(losses * weights) / sum(weights)`` with safe-divide semantics."""
    return _divide_no_nan(jnp.sum(losses * weights), jnp.sum(weights))


def _compute_weights(
    y: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    horizon_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Combine ``mask`` and ``horizon_weight`` into a single weights tensor.

    * ``mask`` defaults to ones of the same shape as ``y``.
    * ``horizon_weight`` is broadcast from ``[H]`` to ``[1, H, 1]`` to match
      ``[B, H, output_size]`` targets.
    """
    if mask is None:
        mask = jnp.ones_like(y)
    if horizon_weight is None:
        weights = jnp.ones_like(mask)
    else:
        hw = horizon_weight[None, :, None]
        weights = jnp.broadcast_to(hw, mask.shape)
    return weights * mask


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
    losses = jnp.abs(y - y_hat)
    weights = _compute_weights(y=y, mask=mask, horizon_weight=horizon_weight)
    return _weighted_mean(losses=losses, weights=weights)


def masked_mse(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    horizon_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Mean Squared Error with multiplicative masking."""
    losses = (y - y_hat) ** 2
    weights = _compute_weights(y=y, mask=mask, horizon_weight=horizon_weight)
    return _weighted_mean(losses=losses, weights=weights)

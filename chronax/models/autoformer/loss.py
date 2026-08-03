"""Loss functions for the Chronax Autoformer model.

Two families are provided:

* **Window losses** — ``mae``, ``mse``, ``huber``: reduce over all elements,
  signature ``(pred, target) -> scalar``. Used by the window-based training
  loop (``AutoformerForecaster``) where batches are already per-window scaled.

* **Masked losses** — ``masked_mae``, ``masked_mse``: reduce with a
  multiplicative mask and optional per-horizon weighting, signature
  ``(y, y_hat, mask, horizon_weight) -> scalar``. Used with the
  ``create_batch``-style training loop (explicit ``sample_mask``).

A ``LOSSES`` registry and ``resolve`` helper allow callers to specify a loss
by name string or a callable.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


# ---------------------------------------------------------------------------
# Window losses (simple element-wise, no masking)
# ---------------------------------------------------------------------------


def mae(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error: ``mean(|pred - target|)``."""
    return jnp.mean(jnp.abs(pred - target))


def mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error: ``mean((pred - target)**2)``."""
    return jnp.mean((pred - target) ** 2)


def huber(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Huber loss with delta = 1.0 (smooth L1).

    Quadratic for ``|r| <= 1``, linear outside. Smooth at the transition so
    gradients are well-behaved everywhere.
    """
    r = pred - target
    abs_r = jnp.abs(r)
    quadratic = 0.5 * r * r
    linear = abs_r - 0.5
    return jnp.mean(jnp.where(abs_r <= 1.0, quadratic, linear))


LOSSES: Mapping[str, LossFn] = {"mae": mae, "mse": mse, "huber": huber}


def resolve(loss: "str | LossFn") -> LossFn:
    """Return a callable from a registry key or pass-through a callable."""
    if callable(loss):
        return loss
    if loss not in LOSSES:
        raise ValueError(
            f"Unknown loss {loss!r}. Available: {sorted(LOSSES)}. "
            "Pass a callable for custom losses."
        )
    return LOSSES[loss]


# ---------------------------------------------------------------------------
# Masked losses (for create_batch-style training with explicit masks)
# ---------------------------------------------------------------------------


def _divide_no_nan(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Safe division: returns 0 where ``b == 0`` (avoids NaN in gradients)."""
    safe_b = jnp.where(b == 0.0, 1.0, b)
    out = a / safe_b
    return jnp.where(b == 0.0, jnp.zeros_like(out), out)


def _weighted_mean(losses: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    return _divide_no_nan(jnp.sum(losses * weights), jnp.sum(weights))


def _compute_weights(
    y: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    horizon_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    if mask is None:
        mask = jnp.ones_like(y)
    if horizon_weight is None:
        weights = jnp.ones_like(mask)
    else:
        hw = horizon_weight[None, :, None]
        weights = jnp.broadcast_to(hw, mask.shape)
    return weights * mask


def masked_mae(
    y: jnp.ndarray,
    y_hat: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    horizon_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Mean Absolute Error with multiplicative masking.

    Args:
        y:               Target values ``[B, H, 1]``.
        y_hat:           Predictions ``[B, H, 1]``.
        mask:            Optional ``[B, H, 1]`` mask (1 = include, 0 = ignore).
        horizon_weight:  Optional ``[H]`` per-step weights.
    """
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

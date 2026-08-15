"""Loss functions for the Chronax FEDformer model (self-contained, pure JAX).

WHAT: The objective functions optimised during training and used for validation.
WHY:  Forecasting models are trained by minimising a discrepancy between the
      predicted horizon and the observed horizon. Different losses encode
      different notions of "error" (absolute vs squared vs robust).
HOW:  Two families are provided:

  * **Window losses** -- ``mae``, ``mse``, ``huber``: signature
    ``(pred, target) -> scalar``; reduce over every element. Used by the
    window-based training loop where each batch is already per-window scaled.

  * **Masked losses** -- ``masked_mae``, ``masked_mse``: signature
    ``(y, y_hat, mask, horizon_weight) -> scalar``; reduce with a multiplicative
    mask (and optional per-step weighting). Used when some horizon steps must be
    ignored (e.g. padded series).

A ``LOSSES`` registry plus ``resolve`` lets callers pass a loss by name string
or by callable.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

import jax.numpy as jnp


# Type alias for window-loss callables (optional mask for NF-style padding).
LossFn = Callable[..., jnp.ndarray]


# ---------------------------------------------------------------------------
# Window losses (element-wise; optional outsample mask for NF-style padding)
# ---------------------------------------------------------------------------


def _reduce(e: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    if mask is None:
        return jnp.mean(e)
    return (e * mask).sum() / jnp.clip(mask.sum(), 1.0)


def mae(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Mean Absolute Error; masked mean when ``mask`` is given (NF padder)."""
    return _reduce(jnp.abs(pred - target), mask)


def mse(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Mean Squared Error; masked mean when ``mask`` is given."""
    return _reduce((pred - target) ** 2, mask)


def huber(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Huber loss with ``delta = 1.0`` (smooth L1); masked mean when ``mask`` is given.

    Quadratic for small residuals (``|r| <= 1``) and linear beyond, giving the
    smooth gradients of MSE near zero with the outlier-robustness of MAE in the
    tails.
    """
    r = pred - target
    abs_r = jnp.abs(r)
    quadratic = 0.5 * r * r
    linear = abs_r - 0.5
    return _reduce(jnp.where(abs_r <= 1.0, quadratic, linear), mask)


# Registry of built-in window losses, addressable by name.
LOSSES: Mapping[str, LossFn] = {"mae": mae, "mse": mse, "huber": huber}


def resolve(loss: "str | LossFn") -> LossFn:
    """Return a loss callable from a registry key or pass a callable through.

    Args:
        loss: Either a registry key (``"mae"`` / ``"mse"`` / ``"huber"``) or any
            ``(pred, target) -> scalar`` callable.

    Raises:
        ValueError: If ``loss`` is an unknown string key.
    """
    if callable(loss):
        return loss
    if loss not in LOSSES:
        raise ValueError(
            f"Unknown loss {loss!r}. Available: {sorted(LOSSES)}. "
            "Pass a callable for custom losses."
        )
    return LOSSES[loss]


# ---------------------------------------------------------------------------
# Masked losses (for training with explicit per-step masks)
# ---------------------------------------------------------------------------


def _divide_no_nan(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Safe division returning 0 where ``b == 0`` (keeps gradients finite)."""
    safe_b = jnp.where(b == 0.0, 1.0, b)
    out = a / safe_b
    return jnp.where(b == 0.0, jnp.zeros_like(out), out)


def _weighted_mean(losses: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Weighted average ``sum(losses*weights) / sum(weights)`` (NaN-safe)."""
    return _divide_no_nan(jnp.sum(losses * weights), jnp.sum(weights))


def _compute_weights(
    y: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    horizon_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Combine an inclusion mask with optional per-horizon-step weights."""
    if mask is None:
        mask = jnp.ones_like(y)
    if horizon_weight is None:
        weights = jnp.ones_like(mask)
    else:
        hw = horizon_weight[None, :, None]  # [1, H, 1] -> broadcast over B and C
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
        y: Target values ``[B, H, 1]``.
        y_hat: Predictions ``[B, H, 1]``.
        mask: Optional ``[B, H, 1]`` mask (1 = include, 0 = ignore).
        horizon_weight: Optional ``[H]`` per-step weights.
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
    """Mean Squared Error with multiplicative masking (see :func:`masked_mae`)."""
    losses = (y - y_hat) ** 2
    weights = _compute_weights(y=y, mask=mask, horizon_weight=horizon_weight)
    return _weighted_mean(losses=losses, weights=weights)

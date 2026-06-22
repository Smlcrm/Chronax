"""Pluggable point-loss functions for the KAN forecaster (self-contained).

Each loss has signature ``(pred, target, mask=None) -> scalar``. When ``mask`` is
given it is a per-element 0/1 weight and the reduction is the masked mean
``sum(loss*mask)/sum(mask)`` — matching neuralforecast's ``_weighted_mean``, used
to drop right-padded horizon steps from the training loss.
"""
from __future__ import annotations

from typing import Callable, Mapping

import jax.numpy as jnp

LossFn = Callable[..., jnp.ndarray]


def _reduce(e: jnp.ndarray, mask: "jnp.ndarray | None") -> jnp.ndarray:
    if mask is None:
        return jnp.mean(e)
    return (e * mask).sum() / jnp.clip(mask.sum(), 1.0)


def mae(pred: jnp.ndarray, target: jnp.ndarray, mask: "jnp.ndarray | None" = None) -> jnp.ndarray:
    """Mean absolute error (masked mean when ``mask`` is given)."""
    return _reduce(jnp.abs(pred - target), mask)


def mse(pred: jnp.ndarray, target: jnp.ndarray, mask: "jnp.ndarray | None" = None) -> jnp.ndarray:
    """Mean squared error (masked mean when ``mask`` is given)."""
    return _reduce((pred - target) ** 2, mask)


def huber(pred: jnp.ndarray, target: jnp.ndarray, mask: "jnp.ndarray | None" = None) -> jnp.ndarray:
    """Huber loss with delta = 1.0 (masked mean when ``mask`` is given)."""
    r = pred - target
    abs_r = jnp.abs(r)
    return _reduce(jnp.where(abs_r <= 1.0, 0.5 * r * r, abs_r - 0.5), mask)


LOSSES: Mapping[str, LossFn] = {"mae": mae, "mse": mse, "huber": huber}


def resolve(loss: "str | LossFn") -> LossFn:
    """Return a callable loss from either a registry string or a callable.

    Callables receive ``(pred, target, mask)``; a custom loss should accept an
    optional ``mask`` (``def my_loss(pred, target, mask=None): ...``).
    """
    if callable(loss):
        return loss
    if loss not in LOSSES:
        raise ValueError(
            f"Unknown loss {loss!r}. Available: {sorted(LOSSES)}. "
            "Pass a callable for custom losses."
        )
    return LOSSES[loss]

"""Pluggable point-loss functions for the iTransformer forecaster.

Self-contained (no dependency on other models' loss modules). Each loss has
signature ``(pred, target) -> scalar`` and reduces by mean over all elements.
Functions are module-level so they pickle cleanly when stored as the ``loss``
attribute of a fitted :class:`chronax.models.itransformer.iTransformer`.
"""
from __future__ import annotations

from typing import Callable, Mapping

import jax.numpy as jnp

LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def mae(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error: ``mean(|pred - target|)``."""
    return jnp.mean(jnp.abs(pred - target))


def mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error: ``mean((pred - target)**2)``."""
    return jnp.mean((pred - target) ** 2)


def huber(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Huber loss with delta = 1.0. Quadratic for ``|r| <= 1``, linear outside."""
    r = pred - target
    abs_r = jnp.abs(r)
    quadratic = 0.5 * r * r
    linear = abs_r - 0.5
    return jnp.mean(jnp.where(abs_r <= 1.0, quadratic, linear))


LOSSES: Mapping[str, LossFn] = {"mae": mae, "mse": mse, "huber": huber}


def resolve(loss: "str | LossFn") -> LossFn:
    """Return a callable loss from either a registry string or a callable."""
    if callable(loss):
        return loss
    if loss not in LOSSES:
        raise ValueError(
            f"Unknown loss {loss!r}. Available: {sorted(LOSSES)}. "
            "Pass a callable for custom losses."
        )
    return LOSSES[loss]

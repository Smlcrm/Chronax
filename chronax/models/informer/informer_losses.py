"""Pluggable losses for the Informer forecaster.

Point losses keep the iTransformer signature ``(pred, target) -> scalar`` and
reduce by mean. ``MultiQuantileLoss`` is Informer's headline multi-quantile (pinball)
loss. Every loss carries an ``outputsize_multiplier`` so the network's output
head width is loss-driven (NF ``loss.outputsize_multiplier``). All are
module-level / class-based so a fitted estimator pickles cleanly.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import jax.numpy as jnp

LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def mae(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error."""
    return jnp.mean(jnp.abs(pred - target))


def mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error."""
    return jnp.mean((pred - target) ** 2)


def huber(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Huber loss, delta = 1.0."""
    r = pred - target
    abs_r = jnp.abs(r)
    return jnp.mean(jnp.where(abs_r <= 1.0, 0.5 * r * r, abs_r - 0.5))


LOSSES: Mapping[str, LossFn] = {"mae": mae, "mse": mse, "huber": huber}


class MultiQuantileLoss:
    """Multi-quantile (pinball) loss. ``__call__(pred[...,h,Q], target[...,h])``.

    ``QL(y, y_hat, q) = q*(y-y_hat)+ + (1-q)*(y_hat-y)+``, SUMMED over quantiles
    and averaged over all other elements — neuralforecast's effective reduction
    (its ``1/len(quantiles)`` factor hits a Q-broadcast tensor, so ``len == 1``
    and the division is dead); the trainer's masked inline branch computes the
    same reduction. Quantiles are sorted, must lie in (0, 1), and must include
    0.5 (the median / ``"mean"`` head). Picklable (holds a plain tuple).
    """

    def __init__(self, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> None:
        qs = tuple(sorted(float(q) for q in quantiles))
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"quantiles must be strictly between 0 and 1; got {qs}.")
        if 0.5 not in qs:
            raise ValueError(f"quantiles must include 0.5 (the median head); got {qs}.")
        self.quantiles = qs
        self.outputsize_multiplier = len(qs)

    def __call__(self, pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        q = jnp.asarray(self.quantiles, dtype=pred.dtype)        # [Q]
        err = target[..., None] - pred                            # [..., h, Q]
        ql = jnp.maximum(q * err, (q - 1.0) * err)                # pinball
        return jnp.mean(jnp.sum(ql, axis=-1))                     # NF: sum over Q, mean elsewhere

    def __eq__(self, other):  # value equality: same-quantile instances are
        return type(other) is type(self) and other.quantiles == self.quantiles

    def __hash__(self):  # interchangeable, which keeps them usable as jit static args
        return hash((type(self), self.quantiles))


def outputsize_multiplier(loss) -> int:
    """Output-head width implied by ``loss`` (1 for point losses)."""
    return int(getattr(loss, "outputsize_multiplier", 1))


def resolve(loss: "str | LossFn | MultiQuantileLoss"):
    """Return a callable loss from a registry string, a callable, or an MQ instance."""
    if callable(loss):
        return loss
    if loss not in LOSSES:
        raise ValueError(
            f"Unknown loss {loss!r}. Available: {sorted(LOSSES)}, or pass a callable / "
            "MultiQuantileLoss."
        )
    return LOSSES[loss]

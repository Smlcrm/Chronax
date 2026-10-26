"""Pluggable losses for the MLP forecaster.

Point losses keep the shared signature ``(pred, target) -> scalar`` and
reduce by mean. ``MultiQuantileLoss`` is the multi-quantile (pinball) loss.
``GMM`` is a Gaussian-mixture distribution loss (port of neuralforecast's
``GMM``): the network head emits distribution parameters instead of point
predictions, the training objective is the mixture's negative log-likelihood,
and predictive quantiles come from Monte-Carlo samples. Every loss carries an
``outputsize_multiplier`` so the network's output head width is loss-driven.
All are module-level / class-based so a fitted estimator pickles cleanly.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import jax
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
    (its ``1/len(quantiles)`` factor is dead); the trainer's masked inline
    branch computes the same reduction. Quantiles are sorted, must lie in (0, 1), and must include
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


def weighted_average(x: jnp.ndarray, weights: jnp.ndarray | None = None, axis=None) -> jnp.ndarray:
    """Weighted average with zero-weight masking, matching neuralforecast's
    ``weighted_average``: cells with weight 0 contribute exactly 0 (so garbage
    or NaN values under a zero mask never leak in), and the denominator is
    ``max(sum(weights), 1.0)`` — clamped to one, not to an epsilon.
    """
    if weights is None:
        return jnp.mean(x, axis=axis)
    weighted = jnp.where(weights != 0, x * weights, jnp.zeros_like(x))
    sum_w = weights.sum(axis=axis) if axis is not None else weights.sum()
    sum_w = jnp.maximum(sum_w, 1.0)
    num = weighted.sum(axis=axis) if axis is not None else weighted.sum()
    return num / sum_w


class GMM:
    """Gaussian Mixture Model distribution loss (port of neuralforecast ``GMM``).

    The network head emits ``(2 + weighted) * n_components`` values per horizon
    step: component means and (pre-softplus) standard deviations, plus mixture
    weight logits when ``weighted=True``; otherwise weights are uniform ``1/K``.
    ``__call__`` is the mixture negative log-likelihood
    ``-logsumexp(log N(y; mu_k, sigma_k) + log w_k)`` reduced by
    :func:`weighted_average` over the mask. ``batch_correlation`` /
    ``horizon_correlation`` sum the component log-likelihoods over the batch /
    horizon axis before mixing (composite-likelihood variants; they assume
    ``[batch, horizon, components]`` inputs).

    ``scale_decouple`` maps scaled-space parameters back to the data scale
    (``means*scale + loc``, ``stds = (softplus(stds) + eps) * scale``): the
    network optimizes in the scaler's normalized space while the likelihood is
    evaluated in original units. The ``eps`` floor applies whenever an anchor is
    given — the identity scaler anchors with real 0/1 tensors, so the floor is
    active there too, matching the reference exactly.

    ``sample`` draws ``num_samples`` Monte-Carlo paths directly from the mixture
    with an explicit PRNG key. The torch reference reaches the same distribution
    through an evenly-spaced quantile grid that is then bootstrap-resampled with
    an unseeded generator (a workaround for its column-oriented predict path);
    direct seeded draws are equal in distribution and reproducible.

    The point forecast surface uses :meth:`analytic_mean` (the exact mixture
    mean) rather than the Monte-Carlo mean of the samples.

    ``return_params`` is accepted for API parity but not supported.

    References:
        Olivares et al., "Probabilistic Hierarchical Forecasting with Deep
        Poisson Mixtures" — https://arxiv.org/abs/2110.13179
    """

    is_distribution_output = True

    def __init__(
        self,
        n_components: int = 1,
        level: Sequence[float] = (80, 90),
        quantiles: Sequence[float] | None = None,
        num_samples: int = 1000,
        return_params: bool = False,
        batch_correlation: bool = False,
        horizon_correlation: bool = False,
        weighted: bool = False,
    ) -> None:
        if return_params:
            raise NotImplementedError(
                "return_params=True is not supported by the chronax GMM port; "
                "the predict dict carries mean and interval columns only."
            )
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1; got {n_components}.")
        # Loss-level quantiles mirror the reference's level/quantile resolution:
        # levels expand to [50 - l/2, 50 + l/2], sorted, with the median prepended;
        # explicit quantiles are kept in the given order (deduplicated).
        if quantiles is not None:
            qs = list(dict.fromkeys(float(q) for q in quantiles))
        else:
            lv = list(dict.fromkeys(level))
            expanded = sorted(q for l in lv for q in (50.0 - l / 2.0, 50.0 + l / 2.0))
            qs = [q / 100.0 for q in [50.0] + expanded]
        self.quantiles = tuple(qs)
        self.n_components = n_components
        self.num_samples = num_samples
        self.batch_correlation = batch_correlation
        self.horizon_correlation = horizon_correlation
        self.weighted = weighted
        self.n_outputs = 2 + weighted
        self.outputsize_multiplier = self.n_outputs * n_components

    # ---- parameter plumbing ---------------------------------------------------
    def domain_map(self, output: jnp.ndarray) -> tuple:
        """Split the raw head ``[..., (2+weighted)*K]`` into equal parameter
        chunks ``(means, stds[, weight_logits])`` of ``[..., K]`` each."""
        return tuple(jnp.split(output, self.n_outputs, axis=-1))

    def scale_decouple(self, output: tuple, loc=None, scale=None, eps: float = 0.2) -> tuple:
        """Positivity-map the stds (softplus) and, when an anchor is given, map
        parameters to the data scale. ``loc``/``scale`` must broadcast against
        the ``[..., h, K]`` parameter arrays (e.g. ``[B, 1, 1]``)."""
        if self.weighted:
            means, stds, weights = output
            weights = jax.nn.softmax(weights, axis=-1)
        else:
            means, stds = output
        stds = jax.nn.softplus(stds)
        if loc is not None and scale is not None:
            means = means * scale + loc
            stds = (stds + eps) * scale
        return (means, stds, weights) if self.weighted else (means, stds)

    def _log_weights(self, distr_args: tuple) -> jnp.ndarray:
        if self.weighted:
            return jnp.log(distr_args[2])
        return jnp.full((self.n_components,), -math.log(self.n_components),
                        dtype=distr_args[0].dtype)

    # ---- objective --------------------------------------------------------------
    def __call__(self, y: jnp.ndarray, distr_args: tuple, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        """Masked negative log-likelihood. ``y [..., h]`` in the same space as the
        (decoupled) ``distr_args``; ``mask`` weights positions per
        :func:`weighted_average`."""
        means, stds = distr_args[0], distr_args[1]
        if mask is not None:
            # Zero-weight cells never contribute; neutralize their targets so a
            # NaN there cannot poison values (or gradients through where).
            y = jnp.where(mask != 0, y, jnp.zeros_like(y))
        z = (y[..., None] - means) / stds
        log_prob = -0.5 * z * z - jnp.log(stds) - 0.5 * math.log(2.0 * math.pi)
        if self.batch_correlation:
            log_prob = jnp.sum(log_prob, axis=0, keepdims=True)
        if self.horizon_correlation:
            log_prob = jnp.sum(log_prob, axis=1, keepdims=True)
        loss_values = -jax.nn.logsumexp(log_prob + self._log_weights(distr_args), axis=-1)
        return weighted_average(loss_values, mask)

    # ---- prediction surface -------------------------------------------------
    def analytic_mean(self, distr_args: tuple) -> jnp.ndarray:
        """Exact mixture mean ``sum_k w_k mu_k`` over the component axis."""
        means = distr_args[0]
        if self.weighted:
            return jnp.sum(means * distr_args[2], axis=-1)
        return jnp.mean(means, axis=-1)

    def sample(self, distr_args: tuple, num_samples: int | None = None, *, key) -> jnp.ndarray:
        """Draw ``[..., num_samples]`` Monte-Carlo paths from the mixture."""
        if num_samples is None:
            num_samples = self.num_samples
        means, stds = distr_args[0], distr_args[1]
        k_comp, k_norm = jax.random.split(key)
        logits = jnp.broadcast_to(self._log_weights(distr_args), means.shape)
        idx = jax.random.categorical(
            k_comp, logits[..., None, :], axis=-1, shape=(*means.shape[:-1], num_samples)
        )
        mu = jnp.take_along_axis(means, idx, axis=-1)
        sd = jnp.take_along_axis(stds, idx, axis=-1)
        return mu + sd * jax.random.normal(k_norm, idx.shape, dtype=means.dtype)

    def __eq__(self, other):  # value equality over the full config surface:
        return type(other) is type(self) and self.__dict__ == other.__dict__

    def __hash__(self):  # equal configs hash equal, usable as jit static args
        return hash((type(self), self.n_components, self.quantiles, self.num_samples))


def outputsize_multiplier(loss) -> int:
    """Output-head width implied by ``loss`` (1 for point losses)."""
    return int(getattr(loss, "outputsize_multiplier", 1))


def resolve(loss: "str | LossFn | MultiQuantileLoss | GMM"):
    """Return a loss object from a registry string, a callable, an MQ instance,
    or a GMM instance."""
    if isinstance(loss, GMM) or callable(loss):
        return loss
    if loss not in LOSSES:
        raise ValueError(
            f"Unknown loss {loss!r}. Available: {sorted(LOSSES)}, or pass a callable / "
            "MultiQuantileLoss / GMM."
        )
    return LOSSES[loss]

"""Pure conformal prediction interval helpers (distribution vs signed scores)."""

from typing import Callable, Dict, List, Union

import jax.numpy as jnp


def add_conformal_distribution_intervals(
    fcst: dict,
    cs: jnp.ndarray,
    level: Union[List[float], List[int]],
) -> dict:
    """Add symmetric conformal intervals using absolute residuals.

    Takes the absolute value of signed conformity scores and constructs
    2W forecast paths (mean +/- |scores|), producing intervals that are
    always symmetric around the mean.

    Args:
        fcst: Forecast dict containing 'mean'.
        cs: Signed conformal scores of shape (W, h).
        level: Confidence levels (0-100).

    Returns:
        Updated fcst dict with 'lo-{lv}' and 'hi-{lv}' keys.
    """
    level = sorted(level)
    alphas = jnp.array([100 - lv for lv in level], dtype=jnp.float32)
    cuts_lower = (alphas / 200.0)[::-1]
    cuts_upper = 1.0 - (alphas / 200.0)
    cuts = jnp.concatenate([cuts_lower, cuts_upper])

    mean = fcst["mean"].reshape(1, -1)
    cs_abs = jnp.abs(cs)
    scores = jnp.vstack([mean - cs_abs, mean + cs_abs])
    quantiles = jnp.quantile(scores, cuts, axis=0)

    lo_cols = [f"lo-{lv}" for lv in reversed(level)]
    hi_cols = [f"hi-{lv}" for lv in level]
    out_cols = lo_cols + hi_cols

    for i, col in enumerate(out_cols):
        fcst[col] = quantiles[i]

    return fcst


def add_conformal_signed_intervals(
    fcst: dict,
    cs: jnp.ndarray,
    level: Union[List[float], List[int]],
) -> dict:
    """Add asymmetric conformal intervals using signed residuals.

    Uses raw signed conformity scores (actual - forecast) to construct
    W plausible values per horizon, allowing asymmetric intervals.

    Args:
        fcst: Forecast dict containing 'mean'.
        cs: Signed conformal scores array of shape (W, h).
        level: Sorted list of confidence levels (0-100).

    Returns:
        Updated fcst dict with 'lo-{lv}' and 'hi-{lv}' keys.
    """
    level = sorted(level)
    alphas = jnp.array([100 - lv for lv in level], dtype=jnp.float32)
    cuts_lower = (alphas / 200.0)[::-1]
    cuts_upper = 1.0 - (alphas / 200.0)
    cuts = jnp.concatenate([cuts_lower, cuts_upper])

    mean = fcst["mean"].reshape(1, -1)
    scores = mean + cs
    quantiles = jnp.quantile(scores, cuts, axis=0)

    lo_cols = [f"lo-{lv}" for lv in reversed(level)]
    hi_cols = [f"hi-{lv}" for lv in level]
    out_cols = lo_cols + hi_cols

    for i, col in enumerate(out_cols):
        fcst[col] = quantiles[i]

    return fcst


def get_conformal_method(method: str) -> Callable:
    """Look up a conformal prediction interval method by name.

    Args:
        method: Method name ('conformal_distribution' or 'conformal_signed').

    Returns:
        The corresponding interval function.

    Raises:
        ValueError: If method is not supported.
    """
    available_methods: Dict[str, Callable] = {
        "conformal_distribution": add_conformal_distribution_intervals,
        "conformal_signed": add_conformal_signed_intervals,
    }
    if method not in available_methods:
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f"please choose one of {', '.join(available_methods.keys())}"
        )
    return available_methods[method]

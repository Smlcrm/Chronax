"""
utils.py — Shared utilities for all Chronax forecasting models.

Sections
--------
1.  Type / Dtype Helpers          ensure_float, calculate_sigma, _jax_norm_ppf, _quantiles
2.  Data Extraction Helpers       extract_demand, extract_probability
3.  Forecast Output Helpers       _repeat_val, _repeat_val_seas, _calculate_intervals,
                                   _add_fitted_pi, _add_fitted_pi_1
4.  Conformal Interval Helpers    add_conformal_distribution_intervals, add_conformal_signed_intervals,
                                   get_conformal_method, _add_conformal_distribution_intervals,
                                   _add_conformal_signed_intervals, _get_conformal_method,
                                   _conformal_method, _store_cs, _add_conformal_intervals,
                                   _add_predict_conformal_intervals
5.  SES Core                      _ses_forecast_nan, _ses_sse, _ses_forecast,
                                   _ses_sse_masked, _ses_forecast_last_masked,
                                   _golden_bounded_minimize,
                                   _optimized_ses_forecast, _optimized_ses_forecast_masked
6.  Aggregation / Chunking        _window_average_core, _window_average,
                                   _chunk_sums, _chunk_forecast
7.  Intermittent Demand Helpers   _demand, _intervals_c, _intervals,
                                   _expand_fitted_demand, _expand_fitted_intervals
8.  Seasonal & Decomposition      _seasonal_exponential_smoothing, _seasonal_naive,
                                   seasonal_decompose, _linear_extrapolate_tail
9.  IMAPA                         _imapa_aggregate_jit, _imapa
10. Miscellaneous                 is_constant, acf, calculate_information_criteria

Public API (imported by other modules)
---------------------------------------
ensure_float, calculate_sigma, extract_demand, extract_probability,
_repeat_val, _repeat_val_seas, _quantiles, _calculate_intervals,
_add_fitted_pi, _add_fitted_pi_1,
add_conformal_distribution_intervals, add_conformal_signed_intervals, get_conformal_method,
_add_conformal_distribution_intervals, _get_conformal_method,
_conformal_method, _store_cs, _add_conformal_intervals, _add_predict_conformal_intervals,
_seasonal_naive, _seasonal_exponential_smoothing, _window_average,
_intervals, _intervals_c, _expand_fitted_intervals, _expand_fitted_demand, _imapa,
calculate_information_criteria, is_constant, acf, results
"""

# ============================================================
# IMPORTS & CONFIG
# ============================================================

import math
import warnings
from collections import namedtuple
from functools import partial
_partial = partial  # alias used in @_partial(jax.jit, ...) decorators
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, lax
from jax.scipy.stats import norm

from chronax.utils.conformal_methods import (
    add_conformal_distribution_intervals,
    add_conformal_signed_intervals,
    get_conformal_method,
)
from chronax.utils.conformal_workflow import (
    add_confidence_intervals as _add_confidence_intervals,
    add_conformal_intervals as _add_conformal_intervals,
    add_predict_conformal_intervals as _add_predict_conformal_intervals,
    compute_conformity_scores as _compute_conformity_scores,
    resolve_conformal_params as _resolve_conformal_config,
    store_conformity_scores as _store_cs,
)

# Enable float64 precision — required by the golden-section SES optimizer.
# Note: this affects the entire JAX session.
jax.config.update("jax_enable_x64", True)

# Named tuple returned by optimize_theta_target_fn and used by ets_functions.
results = namedtuple("results", "x fn nit simplex")


# ============================================================
# SECTION 1 — Type / Dtype Helpers
# ============================================================

def ensure_float(y: jnp.ndarray) -> jnp.ndarray:
    """Cast array to float32 if it is not already a floating-point dtype.

    Args:
        y: Input JAX array of any dtype.

    Returns:
        The same array if already floating-point, otherwise cast to float32.
    """
    if not jnp.issubdtype(y.dtype, jnp.floating):
        return y.astype(jnp.float32)
    return y


@jax.jit
def calculate_sigma(residuals: jnp.ndarray, n: int) -> jnp.ndarray:
    """Compute the root-mean-square of residuals (RMS sigma).

    Args:
        residuals: Residual values as a JAX array.
        n: Number of degrees of freedom (denominator).

    Returns:
        Scalar sigma value; returns 0.0 when n <= 0.
    """
    return jnp.where(
        n > 0,
        jnp.sqrt(jnp.nansum(residuals ** 2) / n),
        0.0,
    )


def _jax_norm_ppf(p: jnp.ndarray) -> jnp.ndarray:
    """Inverse normal CDF (percent-point function) implemented in JAX.

    Uses the Beasley-Springer-Moro rational approximation.

    Args:
        p: Probability value(s) in (0, 1).

    Returns:
        Corresponding z-score(s).
    """
    p = jnp.clip(p, 1e-10, 1 - 1e-10)
    sign = jnp.where(p > 0.5, 1.0, -1.0)
    p_adj = jnp.where(p > 0.5, p, 1.0 - p)

    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    t = jnp.sqrt(-2 * jnp.log(1 - p_adj))
    z = t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
    return sign * z


def _quantiles(level: List[Union[int, float]]) -> jnp.ndarray:
    """Convert confidence levels to z-scores using the normal inverse CDF.

    JAX equivalent of statsforecast.utils._quantiles().

    Args:
        level: List of confidence levels in [0, 100], e.g. [80, 95].

    Returns:
        Array of z-scores, one per level.
    """
    level_arr = jnp.atleast_1d(jnp.asarray(level, jnp.float32))
    p = 0.5 + (level_arr / 200.0)
    return jax.vmap(_jax_norm_ppf)(p)


# ============================================================
# SECTION 2 — Data Extraction Helpers
# ============================================================

def extract_demand(y: jnp.ndarray) -> jnp.ndarray:
    """Extract positive (non-zero) demand values from a time series.

    Used for intermittent demand models like TSB and Croston,
    where we need to separate demand occurrences from no-demand periods.

    Args:
        y: Time series array that may contain zeros.

    Returns:
        Array containing only positive values from y.

    Example:
        >>> y = jnp.array([0, 5, 0, 0, 3, 2, 0])
        >>> extract_demand(y)
        Array([5., 3., 2.], dtype=float32)
    """
    return y[y > 0]


def extract_probability(y: jnp.ndarray) -> jnp.ndarray:
    """Convert time series to binary indicator (1=demand, 0=no demand).

    Used for intermittent demand models like TSB to track the
    probability of demand occurrence at each time step.

    Args:
        y: Time series array.

    Returns:
        Binary array where 1 indicates demand occurred, 0 indicates no demand.

    Example:
        >>> y = jnp.array([0, 5, 0, 0, 3, 2, 0])
        >>> extract_probability(y)
        Array([0., 1., 0., 0., 1., 1., 0.], dtype=float32)
    """
    return (y != 0).astype(y.dtype)


# ============================================================
# SECTION 3 — Forecast Output Helpers
# ============================================================

@_partial(jax.jit, static_argnums=(1,))
def _repeat_val(val: float, h: int) -> jnp.ndarray:
    """Repeat scalar value h times.

    JAX equivalent of statsforecast.utils._repeat_val().

    Args:
        val: Scalar value to repeat.
        h: Number of repetitions (forecast horizon).

    Returns:
        Array of length h filled with val.
    """
    return jnp.full(h, val, dtype=jnp.float32)


@_partial(jax.jit, static_argnums=(1,))
def _repeat_val_seas(season_vals: jnp.ndarray, h: int) -> jnp.ndarray:
    """Tile seasonal values to cover forecast horizon h.

    JAX equivalent of statsforecast.utils._repeat_val_seas().

    Args:
        season_vals: Seasonal pattern of shape (season_length,).
        h: Forecast horizon (static — must be known at compile time).

    Returns:
        Tiled pattern of length h.

    Example:
        >>> season_vals = jnp.array([10.0, 20.0, 30.0])
        >>> _repeat_val_seas(season_vals, h=7)
        array([10., 20., 30., 10., 20., 30., 10.])
    """
    repeats = math.ceil(h / season_vals.size)
    return jnp.tile(season_vals, repeats)[:h]


def _calculate_intervals(
    res: dict,
    level: List[int],
    h: int,
    sigmah: Union[jnp.ndarray, float],
) -> dict:
    """Calculate native (non-conformal) prediction intervals using normal quantiles.

    Args:
        res: Forecast result dict containing 'mean'.
        level: List of confidence levels (0-100).
        h: Forecast horizon.
        sigmah: Standard error (scalar or array of length h).

    Returns:
        Dict with 'lo-{lv}' and 'hi-{lv}' keys for each level.
    """
    mean = jnp.asarray(res["mean"], dtype=jnp.float32)
    sigmah = jnp.asarray(sigmah, dtype=jnp.float32)

    if sigmah.ndim == 0:
        sigmah = jnp.ones(h, dtype=jnp.float32) * sigmah
    elif sigmah.shape[0] != h:
        raise ValueError(f"sigmah shape {sigmah.shape} does not match h={h}")

    z = jnp.asarray(_quantiles(level), dtype=jnp.float32)

    lo = mean[:, None] - sigmah[:, None] * z[None, :]
    hi = mean[:, None] + sigmah[:, None] * z[None, :]

    out = {}
    for i, lv in enumerate(level[::-1]):
        out[f"lo-{int(lv)}"] = lo[:, len(level) - 1 - i]
    for i, lv in enumerate(level):
        out[f"hi-{int(lv)}"] = hi[:, i]

    return out


def _add_fitted_pi(
    res: dict,
    se: jnp.ndarray,
    level: Union[List[int], jnp.ndarray],
) -> dict:
    """Add in-sample prediction intervals to a fitted result dict.

    Used by theta/HW/ETS models. Works with scalar or vector se via reshaping.

    Args:
        res: Result dict containing 'fitted'.
        se: Standard error (scalar or vector).
        level: Confidence levels (0-100).

    Returns:
        Updated res dict with 'fitted-lo-{lv}' and 'fitted-hi-{lv}' keys.
    """
    level = sorted(level)
    level = jnp.asarray(level)
    quantiles = _quantiles(level=level)
    lo = res["fitted"].reshape(-1, 1) - quantiles * se.reshape(-1, 1)
    hi = res["fitted"].reshape(-1, 1) + quantiles * se.reshape(-1, 1)
    lo = lo[:, ::-1]
    lo = {f"fitted-lo-{l}": lo[:, i] for i, l in enumerate(reversed(level))}
    hi = {f"fitted-hi-{l}": hi[:, i] for i, l in enumerate(level)}
    res = {**res, **lo, **hi}
    return res


def _add_fitted_pi_1(
    fitted: jnp.ndarray,
    sigmah: Union[jnp.ndarray, float],
    level: List[int],
) -> dict:
    """Calculate native (non-conformal) fitted (in-sample) prediction intervals.

    JAX equivalent of statsforecast.models._add_fitted_pi().
    Used by historic_average and croston_classic models.

    Args:
        fitted: Fitted values of shape (t,).
        sigmah: Standard error for predictions (scalar or array).
        level: Sorted list of confidence levels (0-100).

    Returns:
        Dict with 'fitted-lo-{lv}' and 'fitted-hi-{lv}' keys for each level.
    """
    z = _quantiles(level)
    lo = fitted[:, None] - z[None, :] * sigmah
    hi = fitted[:, None] + z[None, :] * sigmah

    out = {}
    for i, lv in enumerate(level[::-1]):
        out[f"fitted-lo-{int(lv)}"] = lo[:, len(level) - 1 - i]
    for i, lv in enumerate(level):
        out[f"fitted-hi-{int(lv)}"] = hi[:, i]
    return out


# ============================================================
# SECTION 4 — Conformal Interval Helpers
# ============================================================

_add_conformal_distribution_intervals = add_conformal_distribution_intervals
_add_conformal_signed_intervals = add_conformal_signed_intervals
_get_conformal_method = get_conformal_method


def _conformal_method(self) -> Callable:
    """Retrieve the conformal method from a model's conformal config.

    Args:
        self: A forecaster instance with conformal configuration.

    Returns:
        The conformal interval function.
    """
    conformal_cfg = _resolve_conformal_config(self)
    if conformal_cfg is None:
        raise ValueError("No conformal configuration is set on this model instance.")
    return _get_conformal_method(conformal_cfg.method)


# ============================================================
# SECTION 5 — SES Core
# ============================================================

@jax.jit
def _ses_forecast_nan(x: jnp.ndarray, alpha: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simple Exponential Smoothing forecast with NaN handling.

    Skips NaN values in computation — useful for padded arrays from Croston models.

    Args:
        x: Input array (may contain NaNs).
        alpha: Smoothing parameter.

    Returns:
        Tuple of (forecast, fitted) arrays.
    """
    complement = 1 - alpha
    n = x.size
    fitted = jnp.full_like(x, jnp.nan)

    # Find first non-NaN value
    is_valid = ~jnp.isnan(x)
    first_valid_idx = jnp.argmax(is_valid)
    first_valid_val = x[first_valid_idx]
    fitted = fitted.at[first_valid_idx].set(first_valid_val)

    def body_fun(i, fitted_arr):
        val = x[i]
        prev_fitted = fitted_arr[i - 1]
        new_fitted = jnp.where(
            jnp.isnan(val),
            jnp.nan,
            jnp.where(
                jnp.isnan(prev_fitted),
                val,
                alpha * val + complement * prev_fitted
            )
        )
        fitted_arr = fitted_arr.at[i].set(new_fitted)
        return fitted_arr

    fitted = jax.lax.fori_loop(first_valid_idx + 1, n, body_fun, fitted)

    # Forecast from last non-NaN fitted value
    last_valid_idx = n - 1 - jnp.argmax(is_valid[::-1])
    forecast = fitted[last_valid_idx]

    # Set first fitted to NaN to match original behavior
    fitted = fitted.at[first_valid_idx].set(jnp.nan)
    return forecast, fitted


@jax.jit
def _ses_sse(alpha: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Residual sum of squares for simple exponential smoothing (JIT-compiled).

    Uses lax.fori_loop for efficient JAX compilation.

    Args:
        alpha: Smoothing parameter.
        x: Clean time series of shape (n,).

    Returns:
        SSE scalar value.
    """
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha
    n = x.shape[0]

    def body_fun(i, state):
        forecast, sse = state
        forecast_new = alpha * x[i - 1] + complement * forecast
        err = x[i] - forecast_new
        return (forecast_new, sse + err * err)

    init_state = (x[0], jnp.asarray(0.0, dtype=dtype))
    _, sse = lax.fori_loop(1, n, body_fun, init_state)
    return sse


@jax.jit
def _ses_forecast(x: jnp.ndarray, alpha: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One-step ahead forecast and in-sample fitted values for SES (JIT-compiled).

    Args:
        x: Clean time series of shape (n,).
        alpha: Smoothing parameter.

    Returns:
        Tuple of (forecast, fitted) where forecast is the next-step prediction
        and fitted is the in-sample array with fitted[0] = NaN.
    """
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha

    n = x.shape[0]
    fitted = jnp.empty_like(x)
    fitted = fitted.at[0].set(x[0])

    def body_fun(i, carry):
        j, fitted_arr = carry
        next_val = alpha * x[j] + complement * fitted_arr[j]
        fitted_arr = fitted_arr.at[i].set(next_val)
        return (j + 1, fitted_arr)

    _, fitted = lax.fori_loop(1, n, body_fun, (0, fitted))
    forecast = alpha * x[n - 1] + complement * fitted[n - 1]
    fitted = fitted.at[0].set(jnp.asarray(jnp.nan, dtype=dtype))
    return forecast, fitted


@jax.jit
def _ses_sse_masked(alpha: jnp.ndarray, x: jnp.ndarray, n_eff: jnp.ndarray) -> jnp.ndarray:
    """SSE for SES over the first n_eff elements of a padded array (JIT-compiled).

    Args:
        alpha: Smoothing parameter.
        x: Padded time series of shape (n,).
        n_eff: Number of effective (non-padding) elements.

    Returns:
        SSE scalar value computed only over valid elements.
    """
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha
    n = x.shape[0]

    def body_fun(i, state):
        forecast, sse = state

        def do_update():
            forecast_new = alpha * x[i - 1] + complement * forecast
            err = x[i] - forecast_new
            return (forecast_new, sse + err * err)

        return lax.cond(i < n_eff, do_update, lambda: (forecast, sse))

    init_state = (x[0], jnp.asarray(0.0, dtype=dtype))
    _, sse = lax.fori_loop(1, n, body_fun, init_state)
    return sse


@jax.jit
def _ses_forecast_last_masked(
    x: jnp.ndarray, alpha: jnp.ndarray, n_eff: jnp.ndarray
) -> jnp.ndarray:
    """One-step SES forecast over the first n_eff elements of a padded array.

    Args:
        x: Padded time series.
        alpha: Smoothing parameter.
        n_eff: Number of effective (non-padding) elements.

    Returns:
        One-step forecast scalar.
    """
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha
    n = x.shape[0]

    def body_fun(i, forecast):
        def do_update():
            return alpha * x[i - 1] + complement * forecast
        return lax.cond(i < n_eff, do_update, lambda: forecast)

    init_forecast = x[0]
    forecast = lax.fori_loop(1, n, body_fun, init_forecast)
    return forecast


def _golden_bounded_minimize(
    f: Callable,
    a: float,
    b: float,
    dtype: jnp.dtype = jnp.float64,
    xatol: Optional[float] = None,
    maxiter: int = 1000,
    early_stop_eps: Optional[float] = None,
    early_stop_patience: int = 50,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic golden-section search matching SciPy's 'bounded' behavior.

    All arithmetic performed in the specified dtype. Uses lax.while_loop
    and lax.cond for full JIT compatibility.

    Args:
        f: Objective function to minimize.
        a: Lower bound of search interval.
        b: Upper bound of search interval.
        dtype: Arithmetic dtype (float64 for SciPy-matching precision).
        xatol: Absolute tolerance; defaults to 1e-12 (float64) or 1e-7 (float32).
        maxiter: Maximum number of iterations.
        early_stop_eps: Early stopping threshold; defaults to xatol.
        early_stop_patience: Iterations without improvement before stopping.

    Returns:
        Tuple of (x_star, f_star) — the minimizer and its objective value.
    """
    if xatol is None:
        xatol = 1e-12 if dtype == jnp.float64 else 1e-7

    a = jnp.asarray(a, dtype=dtype)
    b = jnp.asarray(b, dtype=dtype)

    invphi = (jnp.sqrt(jnp.asarray(5.0, dtype=dtype)) - 1.0) / 2.0   # ~0.6180339887
    invphi2 = 1.0 - invphi                                            # ~0.3819660113

    xatol_arr = jnp.asarray(xatol, dtype=dtype)
    if early_stop_eps is None:
        early_stop_eps = xatol
    early_eps_arr = jnp.asarray(early_stop_eps, dtype=dtype)

    # Initial interior points
    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    fc = jnp.asarray(f(c), dtype=dtype)
    fd = jnp.asarray(f(d), dtype=dtype)
    it0 = jnp.asarray(0, dtype=jnp.int32)
    no_improve0 = jnp.asarray(0, dtype=jnp.int32)
    best0 = jnp.minimum(fc, fd)

    def cond_fun(state):
        a_, b_, c_, d_, fc_, fd_, it_, best_, no_improve_ = state
        return (
            (it_ < maxiter)
            & ((d_ - c_) > xatol_arr)
            & (no_improve_ < early_stop_patience)
        )

    def body_fun(state):
        a_, b_, c_, d_, fc_, fd_, it_, best_, no_improve_ = state

        def step_left():
            b_new = d_
            d_new = c_
            fd_new = fc_
            h_new = b_new - a_
            c_new = a_ + invphi2 * h_new
            fc_new = jnp.asarray(f(c_new), dtype=dtype)
            return a_, b_new, c_new, d_new, fc_new, fd_new, it_ + 1

        def step_right():
            a_new = c_
            c_new = d_
            fc_new = fd_
            h_new = b_ - a_new
            d_new = a_new + invphi * h_new
            fd_new = jnp.asarray(f(d_new), dtype=dtype)
            return a_new, b_, c_new, d_new, fc_new, fd_new, it_ + 1

        a_new, b_new, c_new, d_new, fc_new, fd_new, it_new = lax.cond(
            fc_ < fd_, step_left, step_right
        )
        new_best = jnp.minimum(fc_new, fd_new)
        improved = jnp.abs(best_ - new_best) > (early_eps_arr * (1.0 + jnp.abs(best_)))
        no_improve_new = jnp.where(improved, 0, no_improve_ + 1)
        return a_new, b_new, c_new, d_new, fc_new, fd_new, it_new, new_best, no_improve_new

    a, b, c, d, fc, fd, _, _, _ = lax.while_loop(
        cond_fun, body_fun, (a, b, c, d, fc, fd, it0, best0, no_improve0)
    )

    xstar = jnp.where(fc < fd, c, d)
    fstar = jnp.where(fc < fd, fc, fd)
    return jnp.asarray(xstar, dtype=dtype), jnp.asarray(fstar, dtype=dtype)


def _optimized_ses_forecast(
    x: jnp.ndarray, bounds: Tuple[float, float] = (0.1, 0.3)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SES forecast with alpha chosen by golden-section minimization of SSE.

    Heuristic dtype policy:
      - If input is float32 AND (sequence is degenerate (<=2 nonzero) OR has
        any negatives), run BOTH the optimizer and SES recursion in float32 to
        match NumPy reference.
      - Otherwise, run both in float64 for numerical stability.

    Args:
        x: Clean time series.
        bounds: (lower, upper) bounds for alpha search.

    Returns:
        Tuple of (forecast, fitted) arrays in the original input dtype.
    """
    x = ensure_float(x)
    out_dtype = x.dtype

    nonzero_cnt = jnp.sum(x != 0)
    has_neg = jnp.any(x < 0)
    prefer_fp32 = (out_dtype == jnp.float32) & ((nonzero_cnt <= 2) | has_neg)

    run_dtype = jnp.float32 if prefer_fp32 else jnp.float64
    x_run = x.astype(run_dtype)

    def obj(a):
        return _ses_sse(a, x_run)

    xatol = 1e-8 if run_dtype == jnp.float32 else 1e-13

    alpha_star, _ = _golden_bounded_minimize(
        obj, bounds[0], bounds[1], dtype=run_dtype, xatol=xatol, maxiter=5000
    )

    forecast_run, fitted_run = _ses_forecast(x_run, alpha_star)
    forecast = forecast_run.astype(out_dtype)
    fitted = fitted_run.astype(out_dtype)
    return forecast, fitted


def _optimized_ses_forecast_masked(
    x: jnp.ndarray,
    n_eff: jnp.ndarray,
    bounds: Tuple[float, float] = (0.1, 0.3),
) -> jnp.ndarray:
    """SES forecast over the first n_eff elements of a padded array.

    Uses golden-section search to optimize alpha, then computes forecast
    over the effective portion of the array.

    Args:
        x: Padded time series.
        n_eff: Number of effective (non-padding) elements.
        bounds: (lower, upper) bounds for alpha search.

    Returns:
        One-step forecast scalar in the original input dtype.
    """
    x = ensure_float(x)
    out_dtype = x.dtype

    nonzero_cnt = jnp.sum(x != 0)
    has_neg = jnp.any(x < 0)
    prefer_fp32 = (out_dtype == jnp.float32) & ((nonzero_cnt <= 2) | has_neg)

    def run(dtype, xatol):
        x_run = x.astype(dtype)

        def obj(a):
            return _ses_sse_masked(a, x_run, n_eff)

        alpha_star, _ = _golden_bounded_minimize(
            obj, bounds[0], bounds[1], dtype=dtype, xatol=xatol, maxiter=5000
        )
        forecast_run = _ses_forecast_last_masked(x_run, alpha_star, n_eff)
        return forecast_run.astype(out_dtype)

    return lax.cond(
        prefer_fp32,
        lambda: run(jnp.float32, 1e-8),
        lambda: run(jnp.float64, 1e-13),
    )


# ============================================================
# SECTION 6 — Aggregation / Chunking
# ============================================================

@_partial(jax.jit, static_argnums=(1, 2))
def _window_average_core(y: jnp.ndarray, window_size: int, h: int) -> jnp.ndarray:
    """JIT-able core: average the last ``window_size`` values, repeat to length h.

    Args:
        y: Input time series.
        window_size: Number of trailing values to average (static).
        h: Forecast horizon (static).

    Returns:
        Constant forecast array of length h.
    """
    n = y.shape[0]
    start = jnp.maximum(0, n - window_size)
    tail = lax.dynamic_slice(y, (start,), (window_size,))
    wavg = jnp.mean(tail)
    return jnp.full((h,), wavg, dtype=y.dtype)


def _window_average(
    y: jnp.ndarray,
    h: int,
    fitted: bool,
    window_size: int,
) -> Dict[str, jnp.ndarray]:
    """Window average forecast.

    Args:
        y: Time series.
        h: Forecasting horizon.
        fitted: Whether to return fitted values (not implemented).
        window_size: Window size for averaging.

    Returns:
        Dict with 'mean' key containing constant forecast of length h.

    Raises:
        NotImplementedError: If fitted=True.
    """
    if fitted:
        raise NotImplementedError("return fitted")
    if y.size < window_size:
        return {"mean": jnp.full((h,), jnp.nan, dtype=y.dtype)}
    mean = _window_average_core(y, window_size, h)
    return {"mean": mean}


def _chunk_sums(array: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    """Split array into equal chunks and sum each. Incomplete tail discarded.

    Uses jnp.add.reduceat for efficiency.

    Args:
        array: Input array.
        chunk_size: Size of each chunk.

    Returns:
        Array of chunk sums.
    """
    n = array.size
    n_chunks = n // chunk_size
    n_elems = n_chunks * chunk_size
    trimmed = array[:n_elems]
    if n_chunks == 0:
        return jnp.zeros((0,), dtype=array.dtype)
    idx = jnp.arange(0, n_elems, chunk_size)
    return jnp.add.reduceat(trimmed, idx)


def _chunk_forecast(y: jnp.ndarray, aggregation_level: int) -> jnp.ndarray:
    """Compute SES forecast on aggregated (chunked) time series.

    Args:
        y: Input time series.
        aggregation_level: Chunk size for temporal aggregation.

    Returns:
        One-step forecast for the aggregated series.
    """
    lost_remainder_data = len(y) % aggregation_level
    y_cut = y[lost_remainder_data:]
    aggregation_sums = _chunk_sums(y_cut, aggregation_level)
    sums_forecast, _ = _optimized_ses_forecast(aggregation_sums)
    return sums_forecast


# ============================================================
# SECTION 7 — Intermittent Demand Helpers
# ============================================================

@jax.jit
def _demand(x: jnp.ndarray) -> jnp.ndarray:
    """Extract positive (non-zero) elements from array (JIT-compiled).

    Used by Croston-family models for intermittent demand.
    Returns fixed-size array (same size as input) for JIT compatibility.
    Non-zero values are packed at the start, remaining positions filled with NaN.

    Args:
        x: Input array.

    Returns:
        Fixed-size array with non-zero values packed at start, rest NaN.

    Example:
        >>> x = jnp.array([0., 5., 0., 3., 0.])
        >>> _demand(x)
        array([5., 3., nan, nan, nan])
    """
    indices = jnp.where(x > 0, size=x.size, fill_value=-1)[0]
    result = jnp.where(
        indices >= 0,
        jnp.where(indices < x.size, x[jnp.clip(indices, 0, x.size - 1)], jnp.nan),
        jnp.nan
    )
    return result


@jax.jit
def _intervals_c(x: jnp.ndarray) -> jnp.ndarray:
    """Compute intervals between non-zero elements (Croston variant, JIT-compiled).

    Returns fixed-size NaN-padded array for JIT compatibility.
    Used by Croston-family models.

    Args:
        x: Input array.

    Returns:
        Fixed-size array with intervals packed at start, rest NaN.

    Example:
        >>> x = jnp.array([0., 5., 0., 0., 3., 0., 2.])
        >>> _intervals_c(x)
        array([1., 3., 2., nan, nan, nan, nan])
    """
    nonzero_idxs = jnp.where(x != 0, size=x.size, fill_value=-1)[0]
    positions = jnp.where(nonzero_idxs >= 0, nonzero_idxs + 1, -1)
    intervals = jnp.diff(positions, prepend=0)
    valid_mask = positions >= 0
    result = jnp.where(valid_mask, intervals.astype(x.dtype), jnp.nan)
    return result


def _intervals(x: jnp.ndarray) -> jnp.ndarray:
    """Intervals between nonzero elements (IMAPA variant).

    Unlike ``_intervals_c``, returns a compact array of diffs (no NaN padding)
    and prepends the position of the first nonzero element.

    Args:
        x: Input array.

    Returns:
        Float array of inter-arrival intervals.
    """
    idx = jnp.where(x != 0)[0]
    padded = jnp.concatenate([jnp.array([0], dtype=idx.dtype), idx + 1])
    diffs = jnp.diff(padded)
    return diffs.astype(x.dtype)


def _expand_fitted_demand(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Expand demand fitted values back to original series length (JIT-compiled).

    Used by Croston-family models. Uses lax.fori_loop for JIT compatibility.

    Logic:
        - If y[i-1] > 0: Use next fitted value (demand occurred).
        - If y[i-1] == 0 and we've seen demand: Carry forward previous value.
        - If y[i-1] == 0 and no demand yet: Use naive forecast (y[i-1]).

    Args:
        fitted: SES fitted values for demand (length = num_nonzero + 1).
        y: Original time series.

    Returns:
        Fitted values expanded to match y's length.
    """
    n = y.size
    out = jnp.full_like(y, jnp.nan)

    def body_fn(i, state):
        out_arr, fitted_idx = state
        fitted_idx = jnp.where(y[i - 1] > 0, fitted_idx + 1, fitted_idx)
        val = jax.lax.cond(
            y[i - 1] > 0,
            lambda: fitted[fitted_idx],
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],
                lambda: y[i - 1]
            )
        )
        out_arr = out_arr.at[i].set(val)
        return out_arr, fitted_idx

    out, _ = jax.lax.fori_loop(1, n, body_fn, (out, 0))
    return out


@jit
def _expand_fitted_intervals(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Expand interval fitted values back to original series length (JIT-compiled).

    Used by Croston-family models. Uses lax.fori_loop for JIT compatibility.
    Avoids division by zero by replacing zero fitted values with 1.

    Logic:
        - If y[i-1] != 0: Use next fitted value (replace 0 with 1).
        - If y[i-1] == 0 and we've seen intervals: Carry forward previous value.
        - If y[i-1] == 0 and no intervals yet: Use 1.

    Args:
        fitted: SES fitted values for intervals (length = num_nonzero + 1).
        y: Original time series.

    Returns:
        Fitted intervals expanded to match y's length.
    """
    n = y.size
    out = jnp.full_like(y, jnp.nan)

    def body_fn(i, state):
        out_arr, fitted_idx = state
        fitted_idx = jnp.where(
            y[i - 1] != 0,
            fitted_idx + 1,
            fitted_idx
        )
        val = jax.lax.cond(
            y[i - 1] != 0,
            lambda: jnp.where(
                fitted[fitted_idx] == 0,
                1.0,
                fitted[fitted_idx]
            ),
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],
                lambda: 1.0
            )
        )
        out_arr = out_arr.at[i].set(val)
        return out_arr, fitted_idx

    out, _ = jax.lax.fori_loop(1, n, body_fn, (out, 0))
    return out


# ============================================================
# SECTION 8 — Seasonal & Decomposition
# ============================================================

def _seasonal_exponential_smoothing(
    y: jnp.ndarray,
    h: int,
    fitted: bool,
    season_length: int,
    alpha: float,
) -> Dict[str, jnp.ndarray]:
    """Seasonal exponential smoothing forecast.

    Applies SES independently to each seasonal sub-series, then tiles the
    forecasts to cover horizon h.

    Args:
        y: Input time series.
        h: Forecast horizon.
        fitted: Whether to return in-sample fitted values.
        season_length: Seasonal period.
        alpha: Smoothing parameter for SES.

    Returns:
        Dict with 'mean' and optionally 'fitted' keys.
    """
    n = y.size
    if n < season_length:
        return {"mean": jnp.full(h, jnp.nan, dtype=y.dtype)}

    season_vals = jnp.full((season_length,), jnp.nan, dtype=y.dtype)
    fitted_vals = jnp.full_like(y, jnp.nan)

    for i in range(season_length):
        init_idx = i + n % season_length
        x = y[init_idx::season_length]

        forecast, fitted_season = _ses_forecast(x, alpha)

        season_vals = season_vals.at[i].set(forecast)

        for k in range(fitted_season.size):
            fitted_vals = fitted_vals.at[init_idx + k * season_length].set(fitted_season[k])

    out = _repeat_val_seas(season_vals, h)
    fcst = {"mean": out}
    if fitted:
        fcst["fitted"] = fitted_vals
    return fcst


def _seasonal_naive(
    y: jnp.ndarray,
    h: int,
    season_length: int,
    fitted: bool = False,
) -> Dict[str, jnp.ndarray]:
    """JAX implementation of seasonal-naive forecast.

    Repeats the last season_length observations as the forecast.

    Args:
        y: 1-D array-like (length T). Converted to float32.
        h: Forecast horizon (int >= 1).
        season_length: Seasonal period m (int >= 1).
        fitted: If True, also return in-sample fitted values.

    Returns:
        Dict with 'mean' (shape (h,)) and optionally 'fitted' (shape (T,)).

    Raises:
        ValueError: If y is not 1-D, season_length <= 0, T < season_length, or h < 1.
    """
    y_j = jnp.asarray(y, dtype=jnp.float32).squeeze()
    if y_j.ndim != 1:
        raise ValueError("y must be a 1-D array")
    T = y_j.shape[0]
    m = int(season_length)
    if m <= 0:
        raise ValueError("season_length must be a positive integer")
    if T < m:
        raise ValueError(f"Series length T={T} must be at least season_length={m}")
    if not isinstance(h, int) or h < 1:
        raise ValueError("h must be a positive integer")

    last_m = y_j[-m:]
    idx = jnp.arange(h) % m
    mean = last_m[idx]
    out = {"mean": mean}

    if fitted:
        fitted = jnp.full((T,), jnp.nan, dtype=jnp.float32)
        vals = y_j[: T - m]
        positions = jnp.arange(m, T)
        fitted = fitted.at[positions].set(vals)
        out["fitted"] = fitted

    return out


def seasonal_decompose(
    y: jnp.ndarray,
    model: str = "additive",
    period: int = 1,
) -> Dict[str, jnp.ndarray]:
    """Classical seasonal decomposition using centered moving average.

    Uses mode='valid' convolution with NaN-padding and half-weights for even
    periods (proper centered MA), NaN-aware seasonal averaging, and correct
    normalization.

    Args:
        y: Input time series array.
        model: Decomposition type, 'additive' or 'multiplicative'.
        period: Seasonal period length.

    Returns:
        Dict with 'trend', 'seasonal', and 'resid' keys.
    """
    n = len(y)

    # Centered MA filter (even period gets half-weights at endpoints)
    if period % 2 == 0:
        kernel = jnp.concatenate([
            jnp.array([0.5]),
            jnp.ones(period - 1),
            jnp.array([0.5]),
        ]) / period
    else:
        kernel = jnp.ones(period) / period

    # Convolve with 'valid' mode, then NaN-pad edges
    valid = jnp.convolve(y, kernel, mode='valid')
    pad_before = (n - valid.shape[0]) // 2
    pad_after = n - valid.shape[0] - pad_before
    trend = jnp.concatenate([
        jnp.full(pad_before, jnp.nan),
        valid,
        jnp.full(pad_after, jnp.nan),
    ])

    # Detrend
    if model == "additive":
        detrended = y - trend
    else:
        detrended = y / trend

    # NaN-aware seasonal averaging per period position
    n_full = ((n + period - 1) // period) * period
    padded = jnp.concatenate([detrended, jnp.full(n_full - n, jnp.nan)])
    period_avgs = jnp.nanmean(padded.reshape(-1, period), axis=0)

    # Normalize: multiplicative averages to 1.0, additive sums to 0.0
    if model == "additive":
        period_avgs = period_avgs - jnp.nanmean(period_avgs)
    else:
        period_avgs = period_avgs / jnp.nanmean(period_avgs)

    # Tile to full length
    seasonal = jnp.tile(period_avgs, n // period + 1)[:n]

    # Residuals
    if model == "additive":
        resid = y - trend - seasonal
    else:
        resid = y / (trend * seasonal)

    return {"trend": trend, "seasonal": seasonal, "resid": resid}


@_partial(jax.jit, static_argnums=(1, 2))
def _linear_extrapolate_tail(y: jnp.ndarray, tail_window: int, h: int) -> jnp.ndarray:
    """Fit a linear trend to the last ``tail_window`` observations and extrapolate.

    Args:
        y: Input time series.
        tail_window: Number of trailing observations to use (static).
        h: Forecast horizon (static).

    Returns:
        Extrapolated forecast of length h.
    """
    n = y.shape[0]
    start = jnp.maximum(0, n - tail_window)
    seg = jax.lax.dynamic_slice(y, (start,), (tail_window,))
    m = jnp.minimum(tail_window, n)
    t = jnp.arange(tail_window)
    t_mean = jnp.mean(t)
    y_mean = jnp.mean(seg)
    cov = jnp.mean((t - t_mean) * (seg - y_mean))
    var = jnp.mean((t - t_mean) ** 2) + 1e-12
    slope = cov / var
    intercept = y_mean - slope * t_mean
    t_fore = t_mean + (jnp.arange(h) + 1)
    return intercept + slope * t_fore


# ============================================================
# SECTION 9 — IMAPA
# ============================================================

def _repeat_val_(val: jnp.ndarray, h: int) -> jnp.ndarray:
    """Repeat a scalar JAX value h times (internal IMAPA helper).

    Args:
        val: Scalar JAX array.
        h: Number of repetitions.

    Returns:
        Array of length h filled with val.
    """
    return jnp.full((h,), jnp.asarray(val, dtype=val.dtype))


@_partial(jax.jit, static_argnames=("max_k",))
@_partial(jax.jit, static_argnames=("max_k",))
def _imapa_aggregate_jit(y: jnp.ndarray, max_k: int) -> jnp.ndarray:
    """JIT-friendly aggregation loop with padded sums and masked SES.

    For each aggregation level k = 1..max_k, chunks the series, sums chunks,
    fits SES, and stores the per-observation forecast.

    Args:
        y: Input time series.
        max_k: Maximum aggregation level (static).

    Returns:
        Array of per-k forecasts (shape (max_k,)), NaN where no chunks.
    """
    return _imapa_aggregate_body(y, max_k, max_k)


@_partial(jax.jit, static_argnames=("upper_bound",))
def _imapa_aggregate_body(y: jnp.ndarray, max_k, upper_bound: int) -> jnp.ndarray:
    """JIT-compiled core aggregation loop, vmap-compatible.

    Runs SES optimization for each aggregation level k = 1..max_k.
    ``upper_bound`` is static (for array allocation and JIT specialization);
    ``max_k`` may be traced under vmap (controls actual loop iterations).

    Args:
        y: Input time series.
        max_k: Actual max aggregation level (traced under vmap, concrete otherwise).
        upper_bound: Static upper bound for array allocation (must be >= max_k).

    Returns:
        Array of per-k forecasts (shape (upper_bound,)), NaN where unused.
    """
    dtype = y.dtype
    n = y.shape[0]
    forecasts = jnp.full((upper_bound,), jnp.asarray(jnp.nan, dtype=dtype))

    def body(k, forecasts_arr):
        n_chunks = n // k
        lost = n - (n_chunks * k)
        idx = jnp.arange(n)
        valid = idx >= lost
        y_masked = jnp.where(valid, y, jnp.asarray(0.0, dtype=dtype))
        seg_ids = (idx - lost) // k
        seg_ids = jnp.maximum(seg_ids, 0)
        padded = jnp.zeros((n,), dtype=dtype)
        padded = padded.at[seg_ids].add(y_masked)

        def compute_forecast():
            f = _optimized_ses_forecast_masked(padded, n_chunks)
            return f / jnp.asarray(k, dtype=dtype)

        fcast = lax.cond(
            n_chunks == 0,
            lambda: jnp.asarray(jnp.nan, dtype=dtype),
            compute_forecast,
        )
        # Mask iterations beyond actual max_k (original loop runs k=1..max_k inclusive)
        fcast = jnp.where(k <= max_k, fcast, jnp.asarray(jnp.nan, dtype=dtype))
        forecasts_arr = forecasts_arr.at[k - 1].set(fcast)
        return forecasts_arr

    return lax.fori_loop(1, max_k + 1, body, forecasts)


def _imapa(
    y: jnp.ndarray,
    h: int,
    fitted: bool,
) -> Dict[str, jnp.ndarray]:
    """IMAPA forecaster in pure JAX (intermittent demand).

    Detects inter-arrival spacing, computes mean interval as max aggregation
    level K, then for each k = 1..K: chunks, sums, fits SES with golden-section
    alpha optimization, and scales back by 1/k. Averages per-k forecasts for
    the final constant-mean forecast.

    vmap-compatible: all control flow uses JAX primitives (jnp.where, lax.cond).

    Args:
        y: Input time series.
        h: Forecast horizon.
        fitted: Whether to compute in-sample fitted values (O(T^2), expensive).

    Returns:
        Dict with 'mean' (shape (h,)) and optionally 'fitted' (shape (T,)).
    """
    y = ensure_float(y)
    dtype = y.dtype
    all_zeros = jnp.all(y == 0)

    # Compute mean inter-arrival interval directly (vmap-safe, fixed-size ops)
    nonzero_mask = y != 0
    count_nonzero = jnp.sum(nonzero_mask)
    indices = jnp.arange(y.size)
    last_nonzero_pos = jnp.max(jnp.where(nonzero_mask, indices, -1))
    mean_interval = (last_nonzero_pos + 1) / jnp.maximum(count_nonzero, 1)

    max_aggregation_level = jnp.maximum(jnp.rint(mean_interval).astype(jnp.int32), 1)

    # Use static upper bound for array allocation; traced max_k for loop masking
    upper_bound = y.shape[0]  # static under vmap
    forecasts = _imapa_aggregate_body(y, max_aggregation_level, upper_bound)

    # nanmean avoids boolean indexing (vmap-safe)
    forecast = jnp.nanmean(forecasts)
    forecast = jnp.where(jnp.isfinite(forecast), forecast, jnp.asarray(0.0, dtype=dtype))

    # Mask to zero if all-zeros input
    forecast = jnp.where(all_zeros, jnp.asarray(0.0, dtype=dtype), forecast)

    res: Dict = {"mean": _repeat_val_(val=forecast, h=h)}

    if fitted:
        warnings.warn("Computing fitted values for IMAPA is very expensive.")
        n = y.size
        fitted_vals = jnp.empty_like(y)
        fitted_vals = fitted_vals.at[0].set(jnp.asarray(jnp.nan, dtype=dtype))
        for i in range(n - 1):
            sub = y[: i + 1]
            sub_res = _imapa(sub, h=1, fitted=False)
            fitted_vals = fitted_vals.at[i + 1].set(sub_res["mean"][0])
        res["fitted"] = fitted_vals

    return res


# ============================================================
# SECTION 10 — Miscellaneous
# ============================================================

def is_constant(x: jnp.ndarray) -> jnp.ndarray:
    """Check if all elements of an array are equal.

    Args:
        x: Input array.

    Returns:
        Boolean scalar.
    """
    return jnp.all(x[0] == x)


def acf(x: jnp.ndarray, nlags: int) -> jnp.ndarray:
    """Compute autocorrelation function up to ``nlags`` for a 1-D array.

    Equivalent to statsmodels.tsa.stattools.acf(x, nlags=nlags).

    Args:
        x: Input 1-D array.
        nlags: Number of lags to compute.

    Returns:
        Array of ACF values from lag 0 to nlags (length nlags+1).
    """
    x = x - jnp.mean(x)
    n = x.shape[0]
    denom = jnp.dot(x, x)
    acf_vals = jnp.array([jnp.dot(x[: n - lag], x[lag:]) / denom for lag in range(nlags + 1)])
    return acf_vals


@jax.jit
def calculate_information_criteria(
    residuals: jnp.ndarray,
    n_params: int,
    n: int,
) -> Dict[str, jnp.ndarray]:
    """Calculate AIC, BIC, and AICc from residuals (JIT-compiled).

    Args:
        residuals: Model residuals.
        n_params: Number of estimated parameters.
        n: Number of observations.

    Returns:
        Dict with 'loglik', 'aic', 'bic', 'aicc' as JAX arrays.
    """
    sse = jnp.sum(residuals ** 2)
    lik = n * jnp.log(sse + 1e-10)

    aic = lik + 2 * n_params
    bic = lik + jnp.log(n) * n_params
    denom = n - n_params - 1
    aicc = jnp.where(
        denom > 0,
        aic + (2 * n_params * (n_params + 1)) / denom,
        jnp.inf
    )

    return {
        'loglik': -0.5 * lik,
        'aic': aic,
        'bic': bic,
        'aicc': aicc,
    }
    
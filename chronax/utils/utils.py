"""
utils.py — Shared utilities for all Chronax forecasting models.

Sections
--------
1.  Type / Dtype Helpers          ensure_float, calculate_sigma, _jax_norm_ppf, _quantiles
2.  Data Extraction Helpers       extract_demand, extract_probability
3.  Forecast Output Helpers       _repeat_val, _repeat_val_seas, _calculate_intervals,
                                   _add_fitted_pi, _add_fitted_pi_1
4.  Conformal Interval Helpers    _add_conformal_distribution_intervals, _get_conformal_method,
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
11. Adaptive Optimizer Suite      minimize_armijo, bounded_line_minimize,
                                   multistart_argmin, MinimizeState
12. Nelder-Mead Simplex            nelder_mead, NMState
13. Seasonal Period Detection     detect_period, detect_periods,
                                   seasonal_strength

Public API (imported by other modules)
---------------------------------------
ensure_float, calculate_sigma, extract_demand, extract_probability,
_repeat_val, _repeat_val_seas, _quantiles, _calculate_intervals,
_add_fitted_pi, _add_fitted_pi_1,
_add_conformal_distribution_intervals, _get_conformal_method,
_conformal_method, _store_cs, _add_conformal_intervals, _add_predict_conformal_intervals,
_seasonal_naive, _seasonal_exponential_smoothing, _window_average,
_intervals, _intervals_c, _expand_fitted_intervals, _expand_fitted_demand, _imapa,
calculate_information_criteria, is_constant, acf, results,
minimize_armijo, bounded_line_minimize, multistart_argmin, MinimizeState,
nelder_mead, NMState,
detect_period, detect_periods, seasonal_strength
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
from typing import NamedTuple as _NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, lax
from jax.scipy.stats import norm

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

def _add_conformal_distribution_intervals(
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


def _add_conformal_signed_intervals(
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


def _get_conformal_method(method: str) -> Callable:
    """Look up a conformal prediction interval method by name.

    Args:
        method: Method name ('conformal_distribution' or 'conformal_signed').

    Returns:
        The corresponding interval function.

    Raises:
        ValueError: If method is not supported.
    """
    available_methods = {
        "conformal_distribution": _add_conformal_distribution_intervals,
        "conformal_signed": _add_conformal_signed_intervals,
    }
    if method not in available_methods:
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f"please choose one of {', '.join(available_methods.keys())}"
        )
    return available_methods[method]


def _resolve_conformal_params(self):
    """Return a model's conformal config from either attribute spelling.

    ``conformal_params`` is canonical (it is what
    ``BaseForecaster.conformity_scores`` reads); ``prediction_intervals``
    is the StatsForecast-style alias some ctors expose. Models must behave
    the same whichever one the caller populated.
    """
    cp = getattr(self, "conformal_params", None)
    if cp is not None:
        return cp
    return getattr(self, "prediction_intervals", None)


def _conformal_method(self) -> Callable:
    """Retrieve the conformal method from a model's prediction_intervals config.

    Args:
        self: A forecaster instance with prediction_intervals attribute.

    Returns:
        The conformal interval function.
    """
    return _get_conformal_method(self.prediction_intervals.method)


def _store_cs(self, y: jnp.ndarray, X: Optional[jnp.ndarray]) -> None:
    """Compute and store conformal scores on the model instance.

    Args:
        self: A forecaster instance with prediction_intervals and conformity_scores.
        y: Training time series.
        X: Optional exogenous variables.
    """
    if self.prediction_intervals is not None:
        self._cs = self.conformity_scores(y, X)


def _add_conformal_intervals(
    self,
    fcst: dict,
    y: Optional[jnp.ndarray],
    X: Optional[jnp.ndarray],
    level: Optional[List[int]],
) -> dict:
    """Add conformal prediction intervals to a forecast dict.

    If y is provided, computes fresh conformal scores; otherwise uses stored scores.

    Args:
        self: A forecaster instance.
        fcst: Forecast dict to augment.
        y: Training series (None to use stored scores).
        X: Optional exogenous variables.
        level: Confidence levels (0-100).

    Returns:
        Updated forecast dict with interval keys.
    """
    if self.prediction_intervals is not None and level is not None:
        cs = self.conformity_scores(y, X) if y is not None else self._cs
        conformal_fn = _conformal_method(self)
        res = conformal_fn(fcst=fcst, cs=cs, level=level)
        return res
    return fcst


def _add_predict_conformal_intervals(
    self,
    fcst: dict,
    level: Optional[List[int]],
) -> dict:
    """Add conformal intervals for the predict() path (uses stored scores).

    Args:
        self: A fitted forecaster instance.
        fcst: Forecast dict to augment.
        level: Confidence levels (0-100).

    Returns:
        Updated forecast dict with interval keys.
    """
    return _add_conformal_intervals(self, fcst=fcst, y=None, X=None, level=level)


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
    fitted_last = lax.fori_loop(1, n, body_fun, init_forecast)
    # Final one-step-ahead update with the last valid observation x[n_eff-1].
    # The loop above only produced the SES level from x[0..n_eff-2]; without this
    # update it returned the in-sample fitted value, not the proper one-step
    # forecast — which left IMAPA's SES ~10/9 too high, making IMAPA the only
    # intermittent model behind statsforecast (Croston/TSB/ADIDA use the
    # non-masked `_ses_forecast`, which already does this and is at SF parity).
    # statsforecast's `_ses_forecast` ends with the same update.
    # (n_eff is traced -> `x[last_idx]` is a gather; vmap-safe.)
    last_idx = jnp.maximum(n_eff.astype(jnp.int32) - 1, 0)
    forecast = alpha * x[last_idx] + complement * fitted_last
    return forecast


def _ses_associative_op(left, right):
    a_l, b_l = left
    a_r, b_r = right
    return (a_r + b_r * a_l, b_r * b_l)


@jax.jit
def _ses_sse_assoc(alpha: jnp.ndarray, x: jnp.ndarray, n_eff: jnp.ndarray) -> jnp.ndarray:
    """SES SSE via `lax.associative_scan` (template: `chronax/models/garch.py:_compute_sigma2_parallel_q1`).

    Math-equivalent to `_ses_sse_masked` (validated 0.00% relative error
    on synthetic SES data for α ∈ {0.1, 0.3, 0.5, 0.7, 0.9} via the probe
    at `benchmarks/probe_ses_variants.py`). Uses an associative-scan
    primitive that runs in O(log n) wall time on GPU and shares dispatch
    across vmap'd batches — aligned with Chronax's design philosophy.
    """
    x = ensure_float(x)
    dtype = x.dtype
    n = x.shape[0]
    alpha = jnp.asarray(alpha, dtype=dtype)
    beta = jnp.asarray(1.0, dtype=dtype) - alpha
    if n <= 1:
        return jnp.asarray(0.0, dtype=dtype)
    a_terms = alpha * x[:-1]
    b_terms = jnp.full((n - 1,), beta, dtype=dtype)
    A, B = lax.associative_scan(_ses_associative_op, (a_terms, b_terms))
    levels = A + B * x[0]
    err = x[1:] - levels
    t = jnp.arange(1, n, dtype=jnp.int32)
    mask = t < n_eff
    return jnp.sum(jnp.where(mask, err * err, jnp.asarray(0.0, dtype=dtype)))


@jax.jit
def _ses_forecast_last_assoc(
    x: jnp.ndarray, alpha: jnp.ndarray, n_eff: jnp.ndarray
) -> jnp.ndarray:
    """One-step SES forecast via `lax.associative_scan` (assoc-scan twin
    of `_ses_forecast_last_masked`)."""
    x = ensure_float(x)
    dtype = x.dtype
    n = x.shape[0]
    alpha = jnp.asarray(alpha, dtype=dtype)
    beta = jnp.asarray(1.0, dtype=dtype) - alpha
    if n <= 1:
        # Mirror the masked variant: with no recurrence room, the SES
        # forecast collapses to alpha*x[0] + beta*x[0] = x[0]. Static
        # branch on shape, safe inside @jax.jit.
        return x[0] if n == 1 else jnp.asarray(0.0, dtype=dtype)
    a_terms = alpha * x[:-1]
    b_terms = jnp.full((n - 1,), beta, dtype=dtype)
    A, B = lax.associative_scan(_ses_associative_op, (a_terms, b_terms))
    levels = A + B * x[0]
    fitted_last = levels[jnp.maximum(n_eff.astype(jnp.int32) - 2, 0)]
    last_x = x[jnp.maximum(n_eff.astype(jnp.int32) - 1, 0)]
    return alpha * last_x + beta * fitted_last


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


def _trend_strength_masked(x: jnp.ndarray, n_eff: jnp.ndarray) -> jnp.ndarray:
    """Dimensionless trend strength of x[:n_eff].

    Computes OLS slope on the valid prefix, normalised by in-sample
    residual std and scaled by length. Values ~0 indicate stationary data;
    values >>1 indicate strongly trended series (e.g. AirlinePassengers).

    vmap-safe: n_eff may be traced, all ops are masked.
    """
    dtype = x.dtype
    n = x.shape[0]
    t = jnp.arange(n, dtype=dtype)
    valid = (t < n_eff.astype(dtype)).astype(dtype)
    nv = jnp.sum(valid) + jnp.asarray(1e-12, dtype=dtype)

    x_masked = x * valid
    t_mean = jnp.sum(t * valid) / nv
    y_mean = jnp.sum(x_masked) / nv
    tc = (t - t_mean) * valid
    yc = (x_masked - y_mean) * valid

    slope_denom = jnp.sum(tc * tc) + jnp.asarray(1e-12, dtype=dtype)
    slope = jnp.sum(tc * yc) / slope_denom
    intercept = y_mean - slope * t_mean
    resid = (x_masked - (slope * t + intercept)) * valid
    resid_var = jnp.sum(resid * resid) / nv + jnp.asarray(1e-12, dtype=dtype)
    resid_std = jnp.sqrt(resid_var)
    return jnp.abs(slope) * nv / resid_std


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

    # xatol 1e-5 matches SF's scipy minimize_scalar(bounded) default. Tighter
    # tolerances buy decimal digits of SES-alpha resolution SF does not have, at
    # roughly triple the golden-section evals per aggregation level — each eval an
    # O(n) masked SSE scan. IMAPA is this helper's only consumer; ADIDA has its own
    # SES optimizer.
    return lax.cond(
        prefer_fp32,
        lambda: run(jnp.float32, 1e-5),
        lambda: run(jnp.float64, 1e-5),
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


@_partial(jax.jit, static_argnames=("h",))
def _imapa_point_jit(y: jnp.ndarray, h: int) -> jnp.ndarray:
    """JIT-compiled point-forecast core for IMAPA (no fitted-values path).

    Split out from :func:`_imapa` so the hot path can cache its compiled
    XLA kernel across warm calls; the prior implementation paid a 1-2ms
    Python+trace overhead per call (analogous to the ADIDA jit fix at
    `chronax/models/adida.py:_adida_point`).
    """
    y = ensure_float(y)
    dtype = y.dtype
    all_zeros = jnp.all(y == 0)

    nonzero_mask = y != 0
    count_nonzero = jnp.sum(nonzero_mask)
    indices = jnp.arange(y.size)
    last_nonzero_pos = jnp.max(jnp.where(nonzero_mask, indices, -1))
    mean_interval = (last_nonzero_pos + 1) / jnp.maximum(count_nonzero, 1)

    max_aggregation_level = jnp.maximum(jnp.rint(mean_interval).astype(jnp.int32), 1)
    upper_bound = y.shape[0]
    forecasts = _imapa_aggregate_body(y, max_aggregation_level, upper_bound)

    forecast = jnp.nanmean(forecasts)
    forecast = jnp.where(jnp.isfinite(forecast), forecast, jnp.asarray(0.0, dtype=dtype))
    forecast = jnp.where(all_zeros, jnp.asarray(0.0, dtype=dtype), forecast)
    return _repeat_val_(val=forecast, h=h)


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
    
class MinimizeState(_NamedTuple):
    """Return value of :func:`minimize_armijo`.

    Fields
    ------
    x : jnp.ndarray
        Final iterate (shape matches ``x0``).
    f : jnp.ndarray
        Final objective value (scalar).
    g : jnp.ndarray
        Final gradient (shape matches ``x0``).
    iter : jnp.ndarray
        Number of outer iterations taken (``int32`` scalar).
    converged : jnp.ndarray
        Boolean scalar; True iff ``max(|grad|) <= grad_tol`` at exit.
    """
    x: jnp.ndarray
    f: jnp.ndarray
    g: jnp.ndarray
    iter: jnp.ndarray
    converged: jnp.ndarray


def _lbfgs_two_loop(
    g: jnp.ndarray,
    S: jnp.ndarray,
    Y: jnp.ndarray,
    rho: jnp.ndarray,
    count: jnp.ndarray,
) -> jnp.ndarray:
    """L-BFGS two-loop recursion yielding descent direction ``d = -H · g``.

    History is stored with *newest at index 0*: entries ``S[0..count-1]`` are
    valid, ordered newest → oldest. Unused rows (``i >= count``) are masked
    out so the function shape is static.

    Args:
        g: Current gradient, shape ``(n,)``.
        S: History of step vectors ``x_{k+1} - x_k``, shape ``(M, n)``.
        Y: History of gradient differences ``g_{k+1} - g_k``, shape ``(M, n)``.
        rho: History of ``1 / (y · s)``, shape ``(M,)``.
        count: Number of valid history entries (``<= M``).

    Returns:
        Descent direction ``d`` of shape ``(n,)``. Falls back to ``-g`` when
        ``count == 0``.
    """
    M = S.shape[0]

    def bwd_body(i, carry):
        q, alphas = carry
        valid = i < count
        a = jnp.where(valid, rho[i] * jnp.dot(S[i], q), 0.0)
        q_next = jnp.where(valid, q - a * Y[i], q)
        return q_next, alphas.at[i].set(a)

    alphas0 = jnp.zeros(M, dtype=g.dtype)
    q, alphas = lax.fori_loop(0, M, bwd_body, (g, alphas0))

    # Initial Hessian scale γ = (s_newest · y_newest) / (y_newest · y_newest).
    has_pair = count > 0
    numer = jnp.sum(S[0] * Y[0])
    denom = jnp.sum(Y[0] * Y[0])
    gamma = jnp.where(has_pair & (denom > 0), numer / denom, 1.0)
    r = gamma * q

    def fwd_body(k_, r_val):
        # Iterate oldest → newest. When k_ < count, actual history index is
        # count-1-k_; otherwise the masked branch is a no-op.
        i = jnp.clip(count - 1 - k_, 0, M - 1)
        valid = k_ < count
        b = jnp.where(valid, rho[i] * jnp.dot(Y[i], r_val), 0.0)
        r_next = jnp.where(valid, r_val + (alphas[i] - b) * S[i], r_val)
        return r_next

    r = lax.fori_loop(0, M, fwd_body, r)
    return -r


def _armijo_linesearch(
    f_and_g: Callable,
    x: jnp.ndarray,
    f_x: jnp.ndarray,
    g: jnp.ndarray,
    d: jnp.ndarray,
    c1: float,
    shrink: float,
    max_iter: int,
):
    """Backtracking Armijo line search.

    Starts at ``t = min(1, 1/‖d‖∞)`` — the cap keeps the initial step bounded
    when ``d`` is an unscaled steepest-descent direction (e.g., at iter 0
    before the L-BFGS Hessian approximation has any history). Accept the first
    step satisfying ``f(x + t·d) <= f(x) + c1 · t · (g · d)``. Returns the
    accepted ``(t, x_new, f_new, g_new)`` tuple.

    If no ``t`` satisfies the condition within ``max_iter`` backtracks, returns
    the original ``(x, f_x, g)``. The outer L-BFGS loop then exits on the
    ``f_tol`` criterion (|Δf| ≈ 0).
    """
    g_dot_d = jnp.sum(g * d)
    d_max = jnp.max(jnp.abs(d))
    t_init = jnp.minimum(jnp.asarray(1.0, dtype=x.dtype), 1.0 / (d_max + 1e-12))

    def cond_fn(state):
        _t, _x, _f, _g, iter_, done = state
        return (~done) & (iter_ < max_iter)

    def body_fn(state):
        t, x_c, f_c, g_c, iter_, _done = state
        x_try = x + t * d
        f_try, g_try = f_and_g(x_try)
        armijo_ok = f_try <= f_x + c1 * t * g_dot_d
        t_next = jnp.where(armijo_ok, t, t * shrink)
        x_next = jnp.where(armijo_ok, x_try, x_c)
        f_next = jnp.where(armijo_ok, f_try, f_c)
        g_next = jnp.where(armijo_ok, g_try, g_c)
        return t_next, x_next, f_next, g_next, iter_ + 1, armijo_ok

    init = (
        t_init,
        x, f_x, g,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    )
    t, x_new, f_new, g_new, _, _ = lax.while_loop(cond_fn, body_fn, init)
    return t, x_new, f_new, g_new


def minimize_armijo(
    fun: Callable,
    x0: jnp.ndarray,
    *,
    max_iter: int = 200,
    grad_tol: float = 1e-6,
    f_tol: float = 1e-9,
    memory_size: int = 10,
    linesearch_budget: int = 20,
    linesearch_c1: float = 1e-4,
    linesearch_shrink: float = 0.5,
) -> MinimizeState:
    """Adaptive L-BFGS with backtracking Armijo line search.

    Pure-JAX and vmap-safe. The outer loop is ``lax.while_loop`` whose
    predicate is ``(max |grad| > grad_tol) & (|Δf|/(|f|+1) > f_tol) &
    (iter < max_iter)``. Under ``jax.vmap``, converged batch elements do
    near-zero work while the remaining elements finish.

    For bounded problems, reparametrize in caller-space (softplus, sigmoid,
    softmax) before passing the objective — this module is deliberately
    unconstrained so there is a single clean code path.

    Args:
        fun: Scalar objective ``fun(x) -> float``. Must be differentiable.
        x0: Starting point, 1-D array of shape ``(n,)``.
        max_iter: Upper bound on outer iterations.
        grad_tol: Convergence tolerance on ``max(|grad|)``.
        f_tol: Convergence tolerance on relative objective change.
        memory_size: L-BFGS history depth (typical: 5–15).
        linesearch_budget: Max backtracks per outer iter.
        linesearch_c1: Armijo sufficient-decrease constant.
        linesearch_shrink: Backtrack shrink factor (typical: 0.5).

    Returns:
        :class:`MinimizeState` with fields ``(x, f, g, iter, converged)``.
    """
    x0 = jnp.asarray(x0)
    n = x0.shape[0]
    M = memory_size
    dtype = x0.dtype

    f_and_g = jax.value_and_grad(fun)
    f0, g0 = f_and_g(x0)

    if n == 0:
        return MinimizeState(
            x=x0,
            f=f0,
            g=jnp.zeros_like(x0),
            iter=jnp.asarray(0, dtype=jnp.int32),
            converged=jnp.asarray(True),
        )

    S0 = jnp.zeros((M, n), dtype=dtype)
    Y0 = jnp.zeros((M, n), dtype=dtype)
    rho0 = jnp.zeros((M,), dtype=dtype)
    count0 = jnp.asarray(0, dtype=jnp.int32)
    iter0 = jnp.asarray(0, dtype=jnp.int32)
    prev_f0 = jnp.asarray(jnp.inf, dtype=dtype)

    def cond_fn(state):
        _x, f, g, _S, _Y, _rho, _count, iter_, prev_f = state
        grad_inf = jnp.max(jnp.abs(g))
        rel_df = jnp.abs(f - prev_f) / (jnp.abs(f) + 1.0)
        grad_conv = grad_inf <= grad_tol
        f_conv = (iter_ > 0) & (rel_df <= f_tol)
        not_converged = ~(grad_conv | f_conv)
        within_budget = iter_ < max_iter
        return not_converged & within_budget

    def body_fn(state):
        x, f, g, S, Y, rho, count, iter_, _prev_f = state

        d = _lbfgs_two_loop(g, S, Y, rho, count)
        # Safeguard: if direction is not strictly descent (numerical issue),
        # fall back to steepest descent.
        g_dot_d = jnp.sum(g * d)
        d = jnp.where(g_dot_d < 0, d, -g)

        _t, x_new, f_new, g_new = _armijo_linesearch(
            f_and_g, x, f, g, d,
            c1=linesearch_c1,
            shrink=linesearch_shrink,
            max_iter=linesearch_budget,
        )

        s = x_new - x
        y_v = g_new - g
        sy = jnp.sum(s * y_v)
        yy = jnp.sum(y_v * y_v)
        # Curvature condition: skip history update when s·y is non-positive
        # (direction is not helpful) or when ‖y‖² is negligible (no progress).
        curvature_ok = (sy > 1e-10 * jnp.maximum(yy, 1.0)) & (yy > 0)
        rho_new = jnp.where(curvature_ok, 1.0 / jnp.where(sy != 0, sy, 1.0), 0.0)

        def do_update(_):
            S_next = jnp.concatenate([s[None, :], S[:-1]], axis=0)
            Y_next = jnp.concatenate([y_v[None, :], Y[:-1]], axis=0)
            rho_next = jnp.concatenate([rho_new[None], rho[:-1]])
            return S_next, Y_next, rho_next

        def no_update(_):
            return S, Y, rho

        S_next, Y_next, rho_next = lax.cond(curvature_ok, do_update, no_update, operand=None)
        count_next = jnp.where(curvature_ok, jnp.minimum(count + 1, M), count)

        return (
            x_new, f_new, g_new,
            S_next, Y_next, rho_next, count_next,
            iter_ + 1, f,
        )

    init_state = (x0, f0, g0, S0, Y0, rho0, count0, iter0, prev_f0)
    x_f, f_f, g_f, _, _, _, _, iter_f, prev_f_f = lax.while_loop(cond_fn, body_fn, init_state)

    grad_inf = jnp.max(jnp.abs(g_f))
    rel_df = jnp.abs(f_f - prev_f_f) / (jnp.abs(f_f) + 1.0)
    grad_conv = grad_inf <= grad_tol
    f_conv = (iter_f > 0) & (rel_df <= f_tol)
    converged = grad_conv | f_conv
    return MinimizeState(x=x_f, f=f_f, g=g_f, iter=iter_f, converged=converged)


def bounded_line_minimize(
    f: Callable,
    a: float,
    b: float,
    *,
    tol: float = 1e-7,
    max_iter: int = 100,
    dtype: jnp.dtype = jnp.float64,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """1-D bounded minimizer with a sane ``max_iter`` default.

    Thin wrapper around :func:`_golden_bounded_minimize` that defaults to
    ``max_iter=100`` (vs. the legacy 5000). Under ``jax.vmap`` the worst-case
    iteration count syncs across the batch, so the old 5000 ceiling meant a
    single slow series could make the whole batch spin 50× longer than
    necessary — 100 iters is already well past double-precision convergence
    for the 1-D searches Chronax performs (SES α, Box-Cox λ, etc.).

    Args:
        f: 1-D scalar objective ``f(x) -> float``.
        a, b: Lower and upper search bounds.
        tol: Absolute tolerance on the bracket width.
        max_iter: Max iterations (default 100, vs. legacy 5000).
        dtype: Arithmetic dtype; default ``float64``.

    Returns:
        ``(x_star, f_star)`` — the minimizer and its objective value.
    """
    return _golden_bounded_minimize(
        f, a, b, dtype=dtype, xatol=tol, maxiter=max_iter
    )


def multistart_argmin(
    minimizer: Callable[[jnp.ndarray], MinimizeState],
    starts: jnp.ndarray,
) -> MinimizeState:
    """Run ``minimizer`` from each row of ``starts`` and return the best result.

    Replaces Python ``for model_type in ["STM", "OTM", ...]`` loops in
    AutoTheta / AutoTBATS / AutoCES with a single ``vmap``'d minimization
    followed by an ``argmin`` gather.

    Args:
        minimizer: Single-argument minimizer — typically
            ``lambda x0: minimize_armijo(fun, x0, ...)``.
        starts: Array of starting points, shape ``(K, n)``.

    Returns:
        :class:`MinimizeState` for the best (lowest-``f``) start.
    """
    batched = jax.vmap(minimizer)(starts)
    best_idx = jnp.argmin(batched.f)
    return MinimizeState(
        x=batched.x[best_idx],
        f=batched.f[best_idx],
        g=batched.g[best_idx],
        iter=batched.iter[best_idx],
        converged=batched.converged[best_idx],
    )


# ============================================================
# SECTION 12 — Nelder-Mead Simplex (derivative-free)
# ============================================================
#
# Pure-JAX, vmap-safe, JIT-compatible Nelder-Mead simplex minimizer.
# Complements ``minimize_armijo`` — the latter is faster on smooth convex
# problems but gets trapped in local minima on non-smooth objectives
# (e.g. TBATS's log-likelihood with Box-Cox clips and admissibility
# penalties). Nelder-Mead has no gradient; it navigates by reflecting /
# expanding / contracting / shrinking a simplex of (n+1) vertices.
#
# Uses Gao-Han (2012) adaptive coefficients that outperform fixed scalar
# coefficients for n>2. Dispatch between reflect/expand/contract/shrink
# uses ``jnp.where`` (always-evaluate) for the three non-shrink paths and
# ``lax.cond`` for the expensive shrink path (evaluates (n+1) new
# vertices, skipped most iterations in eager mode).


class NMState(_NamedTuple):
    """Return value of :func:`nelder_mead`.

    Fields
    ------
    x : jnp.ndarray
        Best iterate (shape matches ``x0``).
    f : jnp.ndarray
        Objective value at ``x``.
    iter : jnp.ndarray
        Number of outer iterations (int32 scalar).
    converged : jnp.ndarray
        Bool; True iff ``max(f_vals) - min(f_vals) <= f_tol`` at exit.
    """
    x: jnp.ndarray
    f: jnp.ndarray
    iter: jnp.ndarray
    converged: jnp.ndarray


def nelder_mead(
    fun: Callable,
    x0: jnp.ndarray,
    *,
    max_iter: int = 200,
    x_tol: float = 1e-7,
    f_tol: float = 1e-9,
    initial_simplex_size: float = 0.05,
    zero_coord_step: float = 2.5e-4,
) -> NMState:
    """Vmap-safe JIT Nelder-Mead simplex minimizer (derivative-free).

    Args:
        fun: Scalar objective ``fun(x) -> float``. Does NOT need to be
            differentiable — Nelder-Mead is gradient-free.
        x0: Starting point, 1-D array of shape ``(n,)``.
        max_iter: Upper bound on outer iterations.
        x_tol: Converged when simplex diameter (max ‖vertex - best‖)
            drops below this.
        f_tol: Converged when ``max(f_values) - min(f_values) <= f_tol``.
        initial_simplex_size: Relative perturbation for non-zero
            coordinates of ``x0``. Matches scipy's ``nonzdelt``.
        zero_coord_step: Absolute perturbation for zero-valued coords.
            Matches scipy's ``zdelt``.

    Returns:
        :class:`NMState` with fields ``(x, f, iter, converged)``.
    """
    x0 = jnp.asarray(x0)
    n = x0.shape[0]
    dtype = x0.dtype

    # Gao-Han (2012) adaptive coefficients. For n=1 the standard scalars
    # (α=1, γ=2, ρ=0.5, σ=0.5) apply; for n≥2 adaptive versions converge
    # faster on high-dim problems.
    alpha = jnp.asarray(1.0, dtype=dtype)
    gamma = jnp.asarray(1.0 + 2.0 / max(n, 2), dtype=dtype)
    rho = jnp.asarray(0.75 - 1.0 / (2.0 * max(n, 2)), dtype=dtype)
    sigma = jnp.asarray(1.0 - 1.0 / max(n, 2), dtype=dtype)

    # Build initial simplex (n+1 vertices). Row 0 = x0; row i+1 = x0 with
    # coord i perturbed. Scipy-compatible: nonzdelt × x0[i] if non-zero,
    # else zdelt (absolute).
    perturb = jnp.where(
        jnp.abs(x0) > 1e-12,
        jnp.asarray(initial_simplex_size, dtype=dtype) * x0,
        jnp.asarray(zero_coord_step, dtype=dtype),
    )
    simplex0 = jnp.tile(x0[None, :], (n + 1, 1))
    simplex0 = simplex0.at[1:].add(jnp.diag(perturb))
    f0 = jax.vmap(fun)(simplex0)
    iter0 = jnp.asarray(0, dtype=jnp.int32)

    def cond_fn(state):
        simplex, fvals, it = state
        f_range = jnp.max(fvals) - jnp.min(fvals)
        best_idx = jnp.argmin(fvals)
        diam = jnp.max(jnp.linalg.norm(simplex - simplex[best_idx], axis=1))
        not_conv = (f_range > f_tol) | (diam > x_tol)
        return not_conv & (it < max_iter)

    def body_fn(state):
        simplex, fvals, it = state
        worst_idx = jnp.argmax(fvals)
        best_idx = jnp.argmin(fvals)
        f_worst = fvals[worst_idx]
        f_best = fvals[best_idx]
        # 2nd-worst = max(fvals) with worst excluded.
        mask = jnp.arange(n + 1) != worst_idx
        f_2ndworst = jnp.max(jnp.where(mask, fvals, -jnp.inf))

        worst_vertex = simplex[worst_idx]
        best_vertex = simplex[best_idx]
        # Centroid of all except worst (vmap-safe: sum − worst rather than
        # masked slicing, which would create a dynamic shape).
        centroid = (jnp.sum(simplex, axis=0) - worst_vertex) / n

        # Every candidate point this iteration can need is simplex-determined
        # BEFORE any objective evaluation:
        # reflection, expansion, BOTH contraction candidates (the outside/inside
        # base pick needs f_r, so evaluate both and select the VALUE — per-branch
        # arithmetic unchanged), and the n+1 shrink candidates. One vmapped eval
        # replaces 3 sequential evals plus a lax.cond'd shrink vmap — measured
        # 2.4–2.6× per-iteration amortization on the CES objective — and deletes
        # the shrink cond (whose select-executes-both-branches hazard under an
        # outer CV vmap goes with it). The shrink row at best_idx re-evaluates
        # the best vertex, exactly as the old _do_shrink did.
        x_r = centroid + alpha * (centroid - worst_vertex)
        x_e = centroid + gamma * (x_r - centroid)
        x_c_out = centroid + rho * (x_r - centroid)
        x_c_in = centroid + rho * (worst_vertex - centroid)
        shrink_pts = best_vertex + sigma * (simplex - best_vertex)
        cand = jnp.concatenate(
            [jnp.stack([x_r, x_e, x_c_out, x_c_in]), shrink_pts], axis=0)
        f_cand = jax.vmap(fun)(cand)
        f_r, f_e = f_cand[0], f_cand[1]
        # Contraction: outside when reflection beat the worst, else inside.
        is_outside = f_r < f_worst
        x_c = jnp.where(is_outside, x_c_out, x_c_in)
        f_c = jnp.where(is_outside, f_cand[2], f_cand[3])

        # Case dispatch (all booleans):
        case_expand = f_r < f_best
        case_reflect = (f_r >= f_best) & (f_r < f_2ndworst)
        case_contract_accept = (f_r >= f_2ndworst) & (f_c < jnp.minimum(f_r, f_worst))
        case_shrink = (f_r >= f_2ndworst) & (f_c >= jnp.minimum(f_r, f_worst))

        # Expand-case pick the better of x_e and x_r.
        expand_better = f_e < f_r
        x_expand = jnp.where(expand_better, x_e, x_r)
        f_expand = jnp.where(expand_better, f_e, f_r)

        # Non-shrink update: choose replacement vertex.
        x_new = jnp.where(case_expand, x_expand,
                 jnp.where(case_reflect, x_r,
                  jnp.where(case_contract_accept, x_c, worst_vertex)))
        f_new = jnp.where(case_expand, f_expand,
                 jnp.where(case_reflect, f_r,
                  jnp.where(case_contract_accept, f_c, f_worst)))

        simplex_noshrink = simplex.at[worst_idx].set(x_new)
        f_noshrink = fvals.at[worst_idx].set(f_new)

        # Shrink values came from the same batched eval above; a scalar-predicate
        # where replaces the old lax.cond.
        shrink_f = f_cand[4:]
        simplex_next = jnp.where(case_shrink, shrink_pts, simplex_noshrink)
        f_next = jnp.where(case_shrink, shrink_f, f_noshrink)
        return simplex_next, f_next, it + 1

    init_state = (simplex0, f0, iter0)
    simplex_f, fvals_f, iter_f = lax.while_loop(cond_fn, body_fn, init_state)

    best_idx_f = jnp.argmin(fvals_f)
    f_range_f = jnp.max(fvals_f) - jnp.min(fvals_f)
    best_idx_for_diam = jnp.argmin(fvals_f)
    diam_f = jnp.max(jnp.linalg.norm(simplex_f - simplex_f[best_idx_for_diam], axis=1))
    converged = (f_range_f <= f_tol) & (diam_f <= x_tol)
    return NMState(
        x=simplex_f[best_idx_f],
        f=fvals_f[best_idx_f],
        iter=iter_f,
        converged=converged,
    )


# ============================================================
# SECTION 13 — Seasonal Period Detection
# ============================================================
# Pure-JAX, vmap-safe seasonal period detection (moved here from the
# former chronax/utils/period_detection.py so all automatic period
# detection lives in one utilities module). Three helpers callable
# directly by users (``from chronax.utils import detect_period``) and
# consumable inside any model's ``fit`` path:
#   - detect_period      – single dominant period
#   - detect_periods     – top-n distinct periods (for MSTL etc.)
#   - seasonal_strength  – Wang-Smith-Hyndman strength at a given period
# Every public function is pure JAX (jnp.fft, lax.fori_loop, masked
# arithmetic — no np.*, no Python control flow on traced values, no
# data-dependent shapes), so jax.vmap / jax.jit match eager output.
# Caveat: a model that uses the returned period as a *static-shape*
# parameter cannot vary it across a vmap batch without padded state;
# the detection itself is vmap-safe, padding is the model's job.

def adaptive_iterate(step, init, params_of, *, max_steps, rtol=1e-6, patience=2):
    """Run an optimizer ``step`` to convergence-or-cap, returning the best iterate.

    A ``lax.while_loop`` driver that stops when the relative loss improvement
    stays ``< rtol`` for ``patience`` consecutive iterations, or when ``max_steps``
    is reached — auto-adapting the step count to the data instead of running a
    fixed budget. It tracks the *best-loss* iterate seen (not the last), so a step
    that overshoots never degrades the result.

    Intended to replace a fixed ``lax.scan(step, length=N)`` in an optimizer's
    refinement phase. **vmap semantics:** the loop is a ``lax.while_loop`` with all
    control flow on traced scalars via ``jnp`` — so it composes with ``jax.vmap``.
    Called *eagerly* (e.g. one model candidate at a time), it stops at that call's
    own convergence — the real speedup. Called *under vmap* (a batched CV/conformal
    refit), ``lax.while_loop`` runs until the slowest lane's condition is false, i.e.
    to the slowest lane's convergence capped at ``max_steps`` — correct, and never
    more work than the fixed-``max_steps`` scan it replaces.

    Args:
        step: ``carry -> (carry', loss)``. ``loss`` is the objective at the *input*
            ``carry`` (evaluate-current-then-advance, matching optax and the ETS
            "best evaluated point" convention); ``carry'`` is the advanced state.
            ``carry`` is any pytree (params + optimizer state).
        init: initial carry.
        params_of: ``carry -> params`` — extracts the point that ``step`` evaluates,
            so the tracked best pairs each loss with the point it was measured at
            (never a post-update point whose loss is unknown).
        max_steps: hard iteration cap (static int).
        rtol: relative-improvement plateau threshold (static float). Larger ⇒ stops
            sooner; must be small enough that hard problems run to ``max_steps``.
        patience: consecutive plateau iterations required before stopping (static int).

    Returns:
        ``(best_params, best_loss, n_used)`` — ``n_used`` is a traced int32 count of
        steps actually taken (including the initial one).
    """
    max_steps = int(max_steps)
    patience = int(patience)
    # One step up front seeds best/prev at the loss's own dtype — seeding with a
    # python inf would make the while_loop carry f64 and mismatch an f32 loss
    # ("carry input/output types differ"). The
    # first step evaluates `init`, so `init`'s params seed the best iterate.
    oc1, loss0 = step(init)
    c0 = (oc1, params_of(init), loss0, loss0,
          jnp.asarray(0, jnp.int32), jnp.asarray(1, jnp.int32))

    def cond(c):
        *_, plateau, i = c
        return (i < max_steps) & (plateau < patience)

    def body(c):
        oc, best_p, best_l, prev, plateau, i = c
        p_eval = params_of(oc)                      # the point this step evaluates
        oc2, loss = step(oc)                        # loss is measured AT p_eval
        improved = loss < best_l
        best_p2 = jax.tree_util.tree_map(
            lambda new, old: jnp.where(improved, new, old), p_eval, best_p)
        best_l2 = jnp.where(improved, loss, best_l)
        rel = jnp.abs(prev - loss) / (jnp.abs(prev) + 1e-12)
        plateau2 = jnp.where(rel < rtol, plateau + 1, jnp.asarray(0, jnp.int32))
        return (oc2, best_p2, best_l2, loss, plateau2, i + 1)

    _, best_p, best_l, _, _, n_used = lax.while_loop(cond, body, c0)
    return best_p, best_l, n_used


ArrayLike = Union[jnp.ndarray, "np.ndarray"]  # noqa: F821 — np as string


def _acf_and_score(
    y: jnp.ndarray, max_period: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ACF, candidate mask, harmonic-boosted score, and threshold.

    Returns:
        (lag_arr, candidate_mask, score, threshold) where:
        - lag_arr: shape (max_period - 1,) ints, values 2..max_period.
        - candidate_mask: shape (max_period - 1,) bool, True for valid local
          maxima above the noise threshold.
        - score: shape (max_period - 1,) float, ACF at lag plus harmonic
          contribution; -inf at masked-out positions.
        - threshold: float scalar — the noise floor used.

    Static shapes: max_period is a Python int → all output shapes are static.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = y.shape[0]
    M = int(max_period)
    if M < 2:
        # Degenerate — caller treats this as no detection.
        empty = jnp.zeros(0, dtype=jnp.float64)
        return (
            jnp.zeros(0, dtype=jnp.int32),
            jnp.zeros(0, dtype=jnp.bool_),
            empty,
            jnp.asarray(0.0, dtype=jnp.float64),
        )

    # Detrend via first differencing, center.
    dy = jnp.diff(y)
    nd = dy.shape[0]  # static = n - 1
    dy_centered = dy - jnp.mean(dy)

    # Static FFT length: next power of two ≥ 2*nd.
    fft_len = 1
    while fft_len < 2 * max(nd, 1):
        fft_len *= 2

    # ACF via FFT (Wiener-Khinchin), normalised by acf[0].
    fft_vals = jnp.fft.rfft(dy_centered, n=fft_len)
    acf_full = jnp.fft.irfft(fft_vals * jnp.conj(fft_vals), n=fft_len)
    acf_full = acf_full[:nd]
    acf_norm = acf_full / jnp.maximum(jnp.abs(acf_full[0]), 1e-12)

    # Pad to allow static slicing for lag in [2..M] needing left/right neighbours
    # AND harmonic indices up to 3*M.
    needed_len = max(M + 2, 3 * M + 1)
    if needed_len > nd:
        pad = jnp.zeros(needed_len - nd, dtype=acf_norm.dtype)
        acf = jnp.concatenate([acf_norm, pad])
    else:
        acf = acf_norm

    # Slice the lag window: lag k → acf[k], left = acf[k-1], right = acf[k+1].
    acf_at = acf[2 : M + 1]      # k in 2..M
    acf_left = acf[1 : M]
    acf_right = acf[3 : M + 2]

    # Noise-floor threshold (same as original numpy version).
    n_f = jnp.asarray(n, dtype=jnp.float64)
    threshold = jnp.maximum(0.1, 3.0 / jnp.sqrt(n_f))

    # Local-maximum + above-threshold mask.
    is_peak = (acf_at > threshold) & (acf_at >= acf_left) & (acf_at >= acf_right)

    # Lag must have its right-neighbour inside the original (non-padded) ACF —
    # i.e., k + 1 ≤ nd - 1, which is k ≤ nd - 2.
    lag_arr = jnp.arange(2, M + 1, dtype=jnp.int32)
    valid_lag = lag_arr <= (nd - 2)
    candidate_mask = is_peak & valid_lag

    # Harmonic boost: for each lag k, add 0.3 * acf[m·k] when acf[m·k] is real
    # (in-bounds vs original ACF) and ≥ 0.5 * threshold.
    score = acf_at.astype(jnp.float64)
    for mult in (2, 3):
        harm_lag = lag_arr * mult
        harm_in_bounds = harm_lag < nd
        harm_acf = jnp.take(acf, harm_lag, mode="clip")
        harm_above = harm_acf > (threshold * 0.5)
        contrib = jnp.where(harm_in_bounds & harm_above, harm_acf * 0.3, 0.0)
        score = score + contrib

    # Mask out non-candidates so argmax never picks them.
    masked_score = jnp.where(candidate_mask, score, -jnp.inf)
    return lag_arr, candidate_mask, masked_score, threshold


def _seasonal_strength_grid(y: jnp.ndarray, max_period: int) -> jnp.ndarray:
    """Wang-Smith-Hyndman seasonal strength for every period p in [2, M].

    Returns shape ``(M-1,)`` float64 (index j → period j+2). Pure-jnp,
    static shapes (Python loop over the *static* period grid; each P is a
    Python int so ``jnp.arange(P)`` is static) → vmap/jit-safe. Lets strict
    mode read strength at a *traced* candidate period via ``jnp.take``
    (``seasonal_strength`` itself can't — it does ``int(period)``).
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = y.shape[0]
    M = int(max_period)
    t = jnp.arange(n, dtype=jnp.float64)
    t_c = t - jnp.mean(t)
    y_c = y - jnp.mean(y)
    slope = jnp.sum(t_c * y_c) / jnp.maximum(jnp.sum(t_c * t_c), 1e-12)
    intercept = jnp.mean(y) - slope * jnp.mean(t)
    detrended = y - (slope * t + intercept)
    var_det = jnp.maximum(jnp.var(detrended), 1e-12)
    phase_full = jnp.arange(n, dtype=jnp.int32)
    cols = []
    for P in range(2, M + 1):  # static loop over the fixed period grid
        ph = phase_full % P
        one_hot = (ph[:, None] == jnp.arange(P, dtype=jnp.int32)[None, :]).astype(jnp.float64)
        counts = jnp.maximum(one_hot.sum(axis=0), 1.0)
        means = (one_hot * detrended[:, None]).sum(axis=0) / counts
        means = means - jnp.mean(means)
        seasonal = means[ph]
        s = 1.0 - jnp.var(detrended - seasonal) / var_det
        cols.append(jnp.clip(s, 0.0, 1.0))
    return jnp.stack(cols)  # (M-1,)


def detect_period(
    y: ArrayLike,
    *,
    max_period: int = 60,
    fallback: int = 1,
    strict: bool = False,
) -> jnp.ndarray:
    """Detect the dominant seasonal period of a time series.

    Detailed Description:
        Detrends the series via first differencing, computes the ACF via
        FFT (Wiener-Khinchin), and selects the lag k ∈ [2, max_period] whose
        ACF is a local maximum above a noise-floor threshold and which has
        consistent harmonic support (elevated ACF at 2k, 3k). Returns
        ``fallback`` when no candidate clears the threshold or the series
        is too short.

        The implementation is pure JAX — every comparison is a masked
        ``jnp.where`` and every loop is unrolled or expressed via ``lax``
        — so the function is ``jax.vmap`` and ``jax.jit`` safe. Use it
        inside conformal-inference vmap'd fits or batched-forecast pipelines.

    Args:
        y: Univariate time series. Cast to float64 internally.
        max_period: Static upper bound on the searched lag range. The default
            of 60 covers periods up to weekly-of-year (52) and avoids
            SARIMA state-space explosions at very large periods. Must be ≥ 2.
        fallback: Returned when no significant period is detected. Default 1
            (no seasonality).
        strict: Opt-in (default ``False`` → behaviour byte-identical to
            before; AutoARIMA and existing tests unaffected). When ``True``,
            confirm the ACF-selected lag with a period-folded
            Wang-Smith-Hyndman seasonal-strength gate (≥0.15), a
            ≥3-full-cycle requirement, and a fundamental-over-harmonic
            correction (prefer the smallest sub-harmonic ``k//{2,3,4}`` whose
            strength is ≥0.9× the candidate's, which corrects ~2× aliasing).
            This is the canonical mode for ``season_length="auto"``.

    Returns:
        jnp.ndarray: int32 scalar period in [2, max_period] when a strong
        seasonality is found, else ``fallback``. Under ``jax.vmap`` returns
        shape ``(batch,)``.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> y = jnp.asarray([float(i % 12) for i in range(120)])
        >>> int(detect_period(y))
        12

    Notes:
        Caveat for static-shape consumers: the returned int can be used to
        size SARIMA state-spaces or MSTL seasonal buffers eagerly, but vmap'd
        fits across batches with *different* per-element periods need padded
        state on the model side. The utility itself is vmap-safe; padding
        is the model's responsibility.
    """
    M = int(max_period)
    if M < 2:
        return jnp.asarray(fallback, dtype=jnp.int32)

    lag_arr, candidate_mask, masked_score, threshold = _acf_and_score(y, M)

    has_candidate = jnp.any(candidate_mask)
    best_idx = jnp.argmax(masked_score)
    best_lag = lag_arr[best_idx]
    best_score = masked_score[best_idx]

    accept = has_candidate & (best_score > threshold * 1.5)

    if not strict:
        chosen = jnp.where(
            accept, best_lag, jnp.asarray(fallback, dtype=jnp.int32)
        )
        return chosen.astype(jnp.int32)

    # ---- strict: periodogram + folded-strength (validated algorithm) ----
    # ACF-first-difference aliases on near-integer cycles (lynx 19≈2×9.5).
    # The validated detector uses the periodogram peak + period-folded
    # Wang-Hyndman strength, picking the *smallest* strongly-seasonal
    # significant period (= the fundamental). Pure-jnp / static-shape →
    # vmap & jit safe.
    y_arr = jnp.asarray(y, dtype=jnp.float64)
    n = int(y_arr.shape[0])
    periods = jnp.arange(2, M + 1, dtype=jnp.int32)            # (M-1,) static

    # Linear-detrended periodogram power per candidate period.
    t = jnp.arange(n, dtype=jnp.float64)
    t_c = t - jnp.mean(t)
    yc = y_arr - jnp.mean(y_arr)
    slope = jnp.sum(t_c * yc) / jnp.maximum(jnp.sum(t_c * t_c), 1e-12)
    resid = y_arr - (jnp.mean(y_arr) + slope * t_c)
    pw = jnp.abs(jnp.fft.rfft(resid - jnp.mean(resid))) ** 2   # static len n//2+1
    pw = pw.at[0].set(0.0)
    nbins = pw.shape[0]
    bins = jnp.clip(jnp.round(n / periods.astype(jnp.float64)).astype(jnp.int32),
                    1, nbins - 1)
    cand_pw = jnp.take(pw, bins, mode="clip")                  # (M-1,)
    # Noise floor = median spectral power (static shape; robust to the few
    # ~0 bins — no boolean indexing, vmap/jit-safe).
    noise = jnp.median(pw[1:])

    strength = _seasonal_strength_grid(y_arr, M)               # (M-1,)
    n_i = jnp.asarray(n, jnp.int32)
    # ≥5 full cycles (validated recipe: pmax≈n//5) — folding with <5
    # blocks lets noise spuriously "explain" variance at long periods
    # (sunspots→55, lynx→19). us-deaths n≈48 then yields fallback → the
    # documented DOMAIN_M override at the benchmark layer.
    enough = (n_i // jnp.maximum(periods, 1)) >= 5
    significant = cand_pw > (6.0 * jnp.maximum(noise, 1e-30))
    qualify = significant & enough & (strength >= 0.15)

    # Dominant spectral period among qualifying = periodogram argmax power.
    p_pw = jnp.where(qualify, cand_pw, -jnp.inf)
    p_star = jnp.take(periods, jnp.argmax(p_pw))                # traced int32

    def _strength_at(p):
        return jnp.take(strength, jnp.clip(p - 2, 0, M - 2), mode="clip")

    def _qualify_at(p):
        return jnp.take(qualify, jnp.clip(p - 2, 0, M - 2), mode="clip")

    s_star = _strength_at(p_star)
    # Fundamental-over-harmonic: prefer the smallest exact sub-multiple
    # p_star//q (q=2..5) that is itself qualifying with strength ≥0.9·s_star
    # (de-aliases 2×/3× harmonic picks). q last (=5 ⇒ smallest) wins.
    chosen_p = p_star
    for q in (2, 3, 4, 5):
        sub = p_star // q
        is_div = (jnp.mod(p_star, q) == 0) & (sub >= 2)
        ok = is_div & _qualify_at(sub) & (_strength_at(sub) >= 0.9 * s_star)
        chosen_p = jnp.where(ok, sub, chosen_p)

    has_any = jnp.any(qualify)
    chosen = jnp.where(has_any, chosen_p, jnp.asarray(fallback, jnp.int32))
    return chosen.astype(jnp.int32)


def detect_periods(
    y: ArrayLike,
    *,
    n_periods: int = 3,
    max_period: int = 60,
    fallback: int = 1,
) -> jnp.ndarray:
    """Detect the top ``n_periods`` distinct seasonal periods.

    Detailed Description:
        Computes the same FFT-ACF + harmonic score as :func:`detect_period`,
        then iteratively picks the top peak, records it, and zeros out a
        window of ±max(2, lag/5) around it before the next pick. Static
        shapes throughout — ``n_periods`` is a Python int that fixes the
        output length so vmap is safe. Pads with ``fallback`` when fewer
        than ``n_periods`` distinct strong seasonalities exist.

        Used by MSTL where multiple seasonal components matter (e.g. weekly
        + monthly cycles in retail data).

    Args:
        y: Univariate time series.
        n_periods: Static count of periods to return (>= 1). Default 3.
        max_period: Static upper bound on candidate lags. Default 60.
        fallback: Padding value when fewer real periods exist. Default 1.

    Returns:
        jnp.ndarray: int32 array of shape ``(n_periods,)`` sorted ascending.
        Padded entries hold ``fallback``. Under ``jax.vmap`` returns shape
        ``(batch, n_periods)``.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> # Synthetic weekly + monthly: peaks near 7 and 30
        >>> y = jnp.asarray(...)  # see tests
        >>> detect_periods(y, n_periods=2)
        Array([ 7, 30], dtype=int32)

    Notes:
        The "zero-out window" radius scales with the picked lag (±lag/5)
        so that aliases like 2k aren't picked twice. For tightly-clustered
        true periods (e.g. 24 and 28), increase ``n_periods`` and inspect.
    """
    K = int(n_periods)
    M = int(max_period)
    if K < 1:
        return jnp.zeros(0, dtype=jnp.int32)
    if M < 2:
        return jnp.full(K, fallback, dtype=jnp.int32)

    lag_arr, candidate_mask, masked_score, threshold = _acf_and_score(y, M)
    init_score = masked_score
    init_result = jnp.full(K, fallback, dtype=jnp.int32)

    def body(i, state):
        score_state, result_state = state
        idx = jnp.argmax(score_state)
        lag = lag_arr[idx]
        best = score_state[idx]
        chosen = jnp.where(
            best > threshold * 1.5,
            lag,
            jnp.asarray(fallback, dtype=jnp.int32),
        ).astype(jnp.int32)
        new_result = result_state.at[i].set(chosen)
        # Suppress a window around this lag for the next pick (alias prevention).
        window = jnp.maximum(2, lag // 5)
        nearby = jnp.abs(lag_arr - lag) <= window
        new_score = jnp.where(nearby, -jnp.inf, score_state)
        return (new_score, new_result)

    _, result = lax.fori_loop(0, K, body, (init_score, init_result))
    # Sort ascending so MSTL gets shortest period first.
    return jnp.sort(result)


def seasonal_strength(y: ArrayLike, period: int) -> jnp.ndarray:
    """Wang-Smith-Hyndman strength of seasonality at a given period.

    Detailed Description:
        Linearly detrends the series, computes a per-phase mean (the
        seasonal pattern), and reports ``1 - var(remainder) / var(detrended)``
        clipped to ``[0, 1]``. A value near 1 means the series is dominated
        by the seasonal pattern at that period; near 0 means no seasonality.

        Useful for users to validate whether an auto-detected period is
        actually meaningful: detect → check strength → accept or override.

    Args:
        y: Univariate time series.
        period: Static period (≥ 2) to evaluate strength at.

    Returns:
        jnp.ndarray: float64 scalar in ``[0, 1]``.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> y = jnp.sin(jnp.arange(200) * 2 * jnp.pi / 12)
        >>> float(seasonal_strength(y, 12)) > 0.95
        True
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = y.shape[0]
    P = int(period)
    if P < 2:
        return jnp.asarray(0.0, dtype=jnp.float64)

    # Linear detrend.
    t = jnp.arange(n, dtype=jnp.float64)
    t_c = t - jnp.mean(t)
    y_c = y - jnp.mean(y)
    slope = jnp.sum(t_c * y_c) / jnp.maximum(jnp.sum(t_c * t_c), 1e-12)
    intercept = jnp.mean(y) - slope * jnp.mean(t)
    detrended = y - (slope * t + intercept)

    # Per-phase seasonal mean via one-hot-weighted sum (vmap-safe).
    phase = jnp.arange(n, dtype=jnp.int32) % P
    one_hot = (phase[:, None] == jnp.arange(P, dtype=jnp.int32)[None, :]).astype(jnp.float64)
    counts = one_hot.sum(axis=0)
    sums = (one_hot * detrended[:, None]).sum(axis=0)
    seasonal_means = sums / jnp.maximum(counts, 1.0)
    seasonal_means = seasonal_means - jnp.mean(seasonal_means)
    seasonal = seasonal_means[phase]

    remainder = detrended - seasonal
    var_rem = jnp.var(remainder)
    var_det = jnp.var(detrended)
    strength = 1.0 - var_rem / jnp.maximum(var_det, 1e-12)
    return jnp.clip(strength, 0.0, 1.0)

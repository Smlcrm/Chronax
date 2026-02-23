"""
utils.py — Shared utilities for all Chronax forecasting models.

Sections
--------
1.  Type / Dtype Helpers          ensure_float, calculate_sigma, _jax_norm_ppf, _quantiles
2.  Data Extraction Helpers       extract_demand, extract_probability
3.  Forecast Output Helpers       _repeat_val, _repeat_val_seas, _calculate_intervals,
                                   _add_fitted_pi
4.  Conformal Interval Helpers    _add_conformal_distribution_intervals, _get_conformal_method,
                                   _conformal_method, _store_cs, _add_conformal_intervals,
                                   _add_predict_conformal_intervals
5.  SES Core                      _ses_forecast_nan, _ses_sse, _ses_sse_masked, _ses_forecast,
                                   _ses_forecast_last_masked, _golden_bounded_minimize,
                                   _optimized_ses_forecast, _optimized_ses_forecast_masked
6.  Aggregation / Chunking        _window_average_core, _window_average,
                                   _chunk_sums, _chunk_forecast
7.  Intermittent Demand Helpers   _demand, _intervals_c, _intervals,
                                   _expand_fitted_demand, _expand_fitted_intervals
8.  Seasonal & Decomposition      _seasonal_exponential_smoothing, _seasonal_naive,
                                   seasonal_decompose, _linear_extrapolate_tail
9.  IMAPA                         _imapa_aggregate_jit, _imapa
10. Theta Model                   switch_theta, compute_pi_samples, initparamtheta,
                                   optimize_theta_target_fn, thetamodel, forecast_theta,
                                   is_constant, acf, auto_theta, forward_theta
11. Information Criteria          calculate_information_criteria

Public API (imported by other modules)
---------------------------------------
ensure_float, calculate_sigma, extract_demand, extract_probability,
_repeat_val, _repeat_val_seas, _quantiles, _calculate_intervals,
_add_fitted_pi, _add_conformal_distribution_intervals, _get_conformal_method,
_conformal_method, _store_cs, _add_conformal_intervals, _add_predict_conformal_intervals,
_seasonal_naive, _seasonal_exponential_smoothing, _window_average,
_intervals, _intervals_c, _expand_fitted_intervals, _imapa,
auto_theta, forward_theta, forecast_theta, calculate_information_criteria, results
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
import theta_jax as _theta
from jax import lax
from jax.scipy.stats import norm

# Enable float64 precision — required by the golden-section SES optimizer
# and theta model. Note: this affects the entire JAX session.
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

    Used by intermittent demand models (TSB, Croston) to separate demand
    occurrences from no-demand periods.

    Args:
        y: Time series array that may contain zeros.

    Returns:
        Array containing only the positive values of y.

    Example:
        >>> extract_demand(jnp.array([0, 5, 0, 0, 3, 2, 0]))
        Array([5., 3., 2.], dtype=float32)
    """
    return y[y > 0]


def extract_probability(y: jnp.ndarray) -> jnp.ndarray:
    """Convert a time series to a binary demand-occurrence indicator.

    Used by TSB to track the probability of demand at each time step.

    Args:
        y: Time series array.

    Returns:
        Binary array: 1 where demand occurred, 0 elsewhere.

    Example:
        >>> extract_probability(jnp.array([0, 5, 0, 0, 3, 2, 0]))
        Array([0., 1., 0., 0., 1., 1., 0.], dtype=float32)
    """
    return (y != 0).astype(y.dtype)


# ============================================================
# SECTION 3 — Forecast Output Helpers
# ============================================================

@_partial(jax.jit, static_argnums=(1,))
def _repeat_val(val: Union[float, jnp.ndarray], h: int) -> jnp.ndarray:
    """Repeat a scalar value h times to form a flat forecast array.

    JAX equivalent of statsforecast.utils._repeat_val().

    The output dtype is the promoted type of ``val`` and float32 — float32
    inputs stay float32, float64 inputs stay float64, integer inputs promote
    to float32.

    Args:
        val: Scalar value to repeat (Python float or 0-dim JAX array).
        h: Forecast horizon (number of repetitions). Must be static for JIT.

    Returns:
        Array of shape (h,) filled with val, dtype preserved from input.
    """
    return jnp.full(h, val, dtype=jnp.result_type(val, jnp.float32))


@_partial(jax.jit, static_argnums=(1,))
def _repeat_val_seas(season_vals: jnp.ndarray, h: int) -> jnp.ndarray:
    """Tile a seasonal pattern to cover a forecast horizon of length h.

    JAX equivalent of statsforecast.utils._repeat_val_seas().

    Args:
        season_vals: Seasonal pattern of shape (season_length,).
        h: Forecast horizon. Must be static for JIT.

    Returns:
        Array of shape (h,) formed by tiling season_vals.

    Example:
        >>> _repeat_val_seas(jnp.array([10., 20., 30.]), h=7)
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
    """Compute native (non-conformal) prediction intervals using normal quantiles.

    Compatible with SeasonalNaive.predict() and TBATS.

    Args:
        res: Forecast dict containing at least a "mean" key.
        level: List of confidence levels (0-100).
        h: Forecast horizon.
        sigmah: Standard error per step — scalar or array of shape (h,).

    Returns:
        Dict with ``lo-{lv}`` and ``hi-{lv}`` keys for each level.
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

    out: Dict[str, jnp.ndarray] = {}
    for i, lv in enumerate(level[::-1]):
        out[f"lo-{int(lv)}"] = lo[:, len(level) - 1 - i]
    for i, lv in enumerate(level):
        out[f"hi-{int(lv)}"] = hi[:, i]
    return out


def _add_fitted_pi(
    res: dict,
    se: jnp.ndarray,
    level: List[int],
) -> dict:
    """Add native (non-conformal) in-sample prediction intervals to a result dict.

    Args:
        res: Dict containing a "fitted" key with in-sample values of shape (t,).
        se: Standard error array of shape (t,) or scalar.
        level: List of confidence levels (0-100).

    Returns:
        Updated dict with ``fitted-lo-{lv}`` and ``fitted-hi-{lv}`` keys.
    """
    level_sorted = sorted(level)
    quantiles = _quantiles(level=level_sorted)
    lo = res["fitted"].reshape(-1, 1) - quantiles * se.reshape(-1, 1)
    hi = res["fitted"].reshape(-1, 1) + quantiles * se.reshape(-1, 1)
    lo = lo[:, ::-1]
    lo_dict = {f"fitted-lo-{l}": lo[:, i] for i, l in enumerate(reversed(level_sorted))}
    hi_dict = {f"fitted-hi-{l}": hi[:, i] for i, l in enumerate(level_sorted)}
    return {**res, **lo_dict, **hi_dict}


# ============================================================
# SECTION 4 — Conformal Interval Helpers
# ============================================================

def _add_conformal_distribution_intervals(
    fcst: dict,
    cs: jnp.ndarray,
    level: List[Union[float, int]],
) -> dict:
    """Add conformal prediction intervals to a forecast dict.

    Constructs forecast paths from conformity scores and derives quantiles.
    JAX equivalent of statsforecast conformal distribution strategy.

    Args:
        fcst: Forecast dict with at least a "mean" key of shape (h,).
        cs: Conformity scores array of shape (n_windows, h).
        level: Confidence levels (0-100). Will be sorted internally.

    Returns:
        Updated fcst dict with ``lo-{lv}`` and ``hi-{lv}`` keys per level.
    """
    level = sorted(level)
    alphas = jnp.array([100 - lv for lv in level], dtype=jnp.float32)
    cuts_lower = (alphas / 200.0)[::-1]
    cuts_upper = 1.0 - (alphas / 200.0)
    cuts = jnp.concatenate([cuts_lower, cuts_upper])

    mean = fcst["mean"].reshape(1, -1)
    scores = jnp.vstack([mean - cs, mean + cs])
    quantiles = jnp.quantile(scores, cuts, axis=0)

    lo_cols = [f"lo-{lv}" for lv in reversed(level)]
    hi_cols = [f"hi-{lv}" for lv in level]
    for i, col in enumerate(lo_cols + hi_cols):
        fcst[col] = quantiles[i]
    return fcst


def _get_conformal_method(method: str) -> Callable:
    """Return the conformal interval function for the given method name.

    Args:
        method: Name of the conformal method. Currently supports
                ``"conformal_distribution"``.

    Returns:
        Callable that adds conformal intervals to a forecast dict.

    Raises:
        ValueError: If method is not in the available methods.
    """
    available_methods: Dict[str, Callable] = {
        "conformal_distribution": _add_conformal_distribution_intervals,
    }
    if method not in available_methods:
        raise ValueError(
            f"prediction intervals method '{method}' not supported. "
            f"Choose one of: {', '.join(available_methods)}"
        )
    return available_methods[method]


def _conformal_method(self) -> Callable:
    """Retrieve the conformal interval function bound to a forecaster's settings.

    Args:
        self: A BaseForecaster instance with a ``prediction_intervals`` attribute.

    Returns:
        Conformal interval callable for the configured method.
    """
    return _get_conformal_method(self.prediction_intervals.method)


def _store_cs(self, y: jnp.ndarray, X: Optional[jnp.ndarray]) -> None:
    """Compute and store conformity scores on the forecaster instance.

    Sets ``self._cs`` if ``self.prediction_intervals`` is not None.

    Args:
        self: A BaseForecaster instance.
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
    """Attach conformal prediction intervals to a forecast dict.

    Args:
        self: A BaseForecaster instance with a ``prediction_intervals`` attribute.
        fcst: Forecast dict with at least a "mean" key.
        y: Training series used to compute conformity scores, or None to use
           stored scores (``self._cs``).
        X: Optional exogenous variables.
        level: Confidence levels (0-100), or None to skip interval computation.

    Returns:
        Updated fcst dict, unchanged if prediction_intervals is None or level is None.
    """
    if self.prediction_intervals is not None and level is not None:
        cs = self.conformity_scores(y, X) if y is not None else self._cs
        conformal_fn = _conformal_method(self)
        return conformal_fn(fcst=fcst, cs=cs, level=level)
    return fcst


def _add_predict_conformal_intervals(
    self,
    fcst: dict,
    level: Optional[List[int]],
) -> dict:
    """Attach conformal intervals using stored conformity scores (predict path).

    Convenience wrapper around :func:`_add_conformal_intervals` for the
    ``predict()`` method where no new data is available.

    Args:
        self: A BaseForecaster instance.
        fcst: Forecast dict with at least a "mean" key.
        level: Confidence levels (0-100).

    Returns:
        Updated fcst dict with interval keys added.
    """
    return _add_conformal_intervals(self, fcst=fcst, y=None, X=None, level=level)


# ============================================================
# SECTION 5 — SES Core
# ============================================================

@jax.jit
def _ses_forecast_nan(
    x: jnp.ndarray,
    alpha: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SES forecast with NaN handling — skips NaN values in the recursion.

    Useful for padded arrays produced by Croston-family models.

    Args:
        x: Time series, possibly containing NaNs.
        alpha: Smoothing parameter in [0, 1].

    Returns:
        (forecast, fitted_values) where the first valid position in fitted is NaN.
    """
    complement = 1 - alpha
    n = x.size
    fitted = jnp.full_like(x, jnp.nan)

    is_valid = ~jnp.isnan(x)
    first_valid_idx = jnp.argmax(is_valid)
    fitted = fitted.at[first_valid_idx].set(x[first_valid_idx])

    def body_fun(i: int, fitted_arr: jnp.ndarray) -> jnp.ndarray:
        val = x[i]
        prev_fitted = fitted_arr[i - 1]
        new_fitted = jnp.where(
            jnp.isnan(val),
            jnp.nan,
            jnp.where(
                jnp.isnan(prev_fitted),
                val,
                alpha * val + complement * prev_fitted,
            ),
        )
        return fitted_arr.at[i].set(new_fitted)

    fitted = jax.lax.fori_loop(first_valid_idx + 1, n, body_fun, fitted)
    last_valid_idx = n - 1 - jnp.argmax(is_valid[::-1])
    forecast = fitted[last_valid_idx]
    fitted = fitted.at[first_valid_idx].set(jnp.nan)
    return forecast, fitted


@jax.jit
def _ses_sse(alpha: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Residual sum of squares for a Simple Exponential Smoothing fit.

    Fully JIT-compatible via lax.fori_loop.

    Args:
        alpha: Smoothing parameter in [0, 1] as a JAX scalar.
        x: Clean time series of shape (n,).

    Returns:
        Scalar SSE value.
    """
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha
    n = x.shape[0]

    def body_fun(i: int, state: Tuple) -> Tuple:
        forecast, sse = state
        forecast_new = alpha * x[i - 1] + complement * forecast
        err = x[i] - forecast_new
        return forecast_new, sse + err * err

    init_state = (x[0], jnp.asarray(0.0, dtype=dtype))
    _, sse = lax.fori_loop(1, n, body_fun, init_state)
    return sse


@jax.jit
def _ses_sse_masked(
    alpha: jnp.ndarray,
    x: jnp.ndarray,
    n_eff: jnp.ndarray,
) -> jnp.ndarray:
    """SSE for SES over the first n_eff elements of a padded array.

    Args:
        alpha: Smoothing parameter in [0, 1] as a JAX scalar.
        x: Padded time series of shape (n,).
        n_eff: Number of valid elements at the front of x.

    Returns:
        Scalar SSE computed only over the first n_eff positions.
    """
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha
    n = x.shape[0]

    def body_fun(i: int, state: Tuple) -> Tuple:
        forecast, sse = state

        def do_update() -> Tuple:
            forecast_new = alpha * x[i - 1] + complement * forecast
            err = x[i] - forecast_new
            return forecast_new, sse + err * err

        return lax.cond(i < n_eff, do_update, lambda: (forecast, sse))

    init_state = (x[0], jnp.asarray(0.0, dtype=dtype))
    _, sse = lax.fori_loop(1, n, body_fun, init_state)
    return sse


@jax.jit
def _ses_forecast(
    x: jnp.ndarray,
    alpha: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One-step-ahead SES forecast and in-sample fitted values.

    Fully JIT-compiled via lax.fori_loop; handles arbitrary dtypes.

    Args:
        x: Time series of shape (n,).
        alpha: Smoothing parameter in [0, 1] as a JAX scalar.

    Returns:
        (forecast, fitted_values) where fitted_values[0] is NaN.
    """
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha
    n = x.shape[0]

    fitted = jnp.empty_like(x).at[0].set(x[0])

    def body_fun(i: int, carry: Tuple) -> Tuple:
        j, fitted_arr = carry
        next_val = alpha * x[j] + complement * fitted_arr[j]
        return j + 1, fitted_arr.at[i].set(next_val)

    _, fitted = lax.fori_loop(1, n, body_fun, (0, fitted))
    forecast = alpha * x[n - 1] + complement * fitted[n - 1]
    fitted = fitted.at[0].set(jnp.asarray(jnp.nan, dtype=dtype))
    return forecast, fitted


@jax.jit
def _ses_forecast_last_masked(
    x: jnp.ndarray,
    alpha: jnp.ndarray,
    n_eff: jnp.ndarray,
) -> jnp.ndarray:
    """One-step SES forecast over the first n_eff elements of a padded array.

    Args:
        x: Padded time series of shape (n,).
        alpha: Smoothing parameter in [0, 1] as a JAX scalar.
        n_eff: Number of valid elements at the front of x.

    Returns:
        Scalar one-step-ahead forecast.
    """
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha
    n = x.shape[0]

    def body_fun(i: int, forecast: jnp.ndarray) -> jnp.ndarray:
        return lax.cond(
            i < n_eff,
            lambda: alpha * x[i - 1] + complement * forecast,
            lambda: forecast,
        )

    return lax.fori_loop(1, n, body_fun, x[0])


def _golden_bounded_minimize(
    f: Callable,
    a: float,
    b: float,
    dtype=jnp.float64,
    xatol: Optional[float] = None,
    maxiter: int = 1000,
    early_stop_eps: Optional[float] = None,
    early_stop_patience: int = 50,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic golden-section search mirroring SciPy's ``minimize_scalar`` bounded mode.

    All arithmetic is performed in ``dtype`` (use float64 to match SciPy defaults).

    Args:
        f: Scalar objective function to minimise.
        a: Left bound of search interval.
        b: Right bound of search interval.
        dtype: JAX floating dtype for arithmetic precision.
        xatol: Absolute tolerance on x; defaults to 1e-12 (float64) or 1e-7 (float32).
        maxiter: Maximum number of golden-section iterations.
        early_stop_eps: Relative improvement threshold for early stopping.
        early_stop_patience: Number of non-improving iterations before stopping.

    Returns:
        (x_star, f_star) — minimiser and its objective value.
    """
    if xatol is None:
        xatol = 1e-12 if dtype == jnp.float64 else 1e-7
    if early_stop_eps is None:
        early_stop_eps = xatol

    a = jnp.asarray(a, dtype=dtype)
    b = jnp.asarray(b, dtype=dtype)

    invphi = (jnp.sqrt(jnp.asarray(5.0, dtype=dtype)) - 1.0) / 2.0
    invphi2 = 1.0 - invphi
    xatol_arr = jnp.asarray(xatol, dtype=dtype)
    early_eps_arr = jnp.asarray(early_stop_eps, dtype=dtype)

    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    fc = jnp.asarray(f(c), dtype=dtype)
    fd = jnp.asarray(f(d), dtype=dtype)
    it0 = jnp.asarray(0, dtype=jnp.int32)
    no_improve0 = jnp.asarray(0, dtype=jnp.int32)
    best0 = jnp.minimum(fc, fd)

    def cond_fun(state: Tuple) -> bool:
        a_, b_, c_, d_, fc_, fd_, it_, best_, no_improve_ = state
        return (
            (it_ < maxiter)
            & ((d_ - c_) > xatol_arr)
            & (no_improve_ < early_stop_patience)
        )

    def body_fun(state: Tuple) -> Tuple:
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
    x: jnp.ndarray,
    bounds: Tuple[float, float] = (0.1, 0.3),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SES forecast with alpha selected by golden-section minimisation of SSE.

    Precision heuristic: runs in float64 by default for numerical stability,
    but switches to float32 for degenerate (<=2 non-zero) or sign-changing
    float32 inputs to better match NumPy reference paths.

    Args:
        x: Time series of shape (n,).
        bounds: (lower, upper) search bounds for alpha.

    Returns:
        (forecast, fitted_values) using the optimal alpha.
    """
    x = ensure_float(x)
    out_dtype = x.dtype

    nonzero_cnt = jnp.sum(x != 0)
    has_neg = jnp.any(x < 0)
    prefer_fp32 = (out_dtype == jnp.float32) & ((nonzero_cnt <= 2) | has_neg)

    run_dtype = jnp.float32 if prefer_fp32 else jnp.float64
    x_run = x.astype(run_dtype)
    xatol = 1e-8 if run_dtype == jnp.float32 else 1e-13

    alpha_star, _ = _golden_bounded_minimize(
        lambda a: _ses_sse(a, x_run),
        bounds[0],
        bounds[1],
        dtype=run_dtype,
        xatol=xatol,
        maxiter=5000,
    )
    forecast_run, fitted_run = _ses_forecast(x_run, alpha_star)
    return forecast_run.astype(out_dtype), fitted_run.astype(out_dtype)


def _optimized_ses_forecast_masked(
    x: jnp.ndarray,
    n_eff: jnp.ndarray,
    bounds: Tuple[float, float] = (0.1, 0.3),
) -> jnp.ndarray:
    """SES forecast over the first n_eff elements of a padded array.

    Alpha is chosen by golden-section search on SSE, with the same
    float32/float64 precision heuristic as :func:`_optimized_ses_forecast`.

    Args:
        x: Padded time series of shape (n,).
        n_eff: Number of valid elements at the front of x.
        bounds: (lower, upper) search bounds for alpha.

    Returns:
        Scalar one-step-ahead forecast.
    """
    x = ensure_float(x)
    out_dtype = x.dtype

    nonzero_cnt = jnp.sum(x != 0)
    has_neg = jnp.any(x < 0)
    prefer_fp32 = (out_dtype == jnp.float32) & ((nonzero_cnt <= 2) | has_neg)

    def run(dtype: jnp.dtype, xatol: float) -> jnp.ndarray:
        x_run = x.astype(dtype)
        alpha_star, _ = _golden_bounded_minimize(
            lambda a: _ses_sse_masked(a, x_run, n_eff),
            bounds[0],
            bounds[1],
            dtype=dtype,
            xatol=xatol,
            maxiter=5000,
        )
        return _ses_forecast_last_masked(x_run, alpha_star, n_eff).astype(out_dtype)

    return lax.cond(
        prefer_fp32,
        lambda: run(jnp.float32, 1e-8),
        lambda: run(jnp.float64, 1e-13),
    )


# ============================================================
# SECTION 6 — Aggregation / Chunking
# ============================================================

@_partial(jax.jit, static_argnums=(1, 2))
def _window_average_core(
    y: jnp.ndarray,
    window_size: int,
    h: int,
) -> jnp.ndarray:
    """JIT-compiled window average using dynamic_slice with a static window size.

    Args:
        y: Time series of shape (n,).
        window_size: Number of trailing observations to average. Must be static.
        h: Forecast horizon. Must be static.

    Returns:
        Constant-mean forecast of shape (h,).
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
    """Compute a window-average forecast over the last ``window_size`` observations.

    Args:
        y: Time series of shape (n,).
        h: Forecast horizon.
        fitted: If True, raises NotImplementedError (fitted values not yet supported).
        window_size: Number of trailing observations to average.

    Returns:
        Dict with ``"mean"`` key of shape (h,); or all-NaN if n < window_size.
    """
    if fitted:
        raise NotImplementedError("Fitted values not yet supported for window average.")
    if y.size < window_size:
        return {"mean": jnp.full((h,), jnp.nan, dtype=y.dtype)}
    return {"mean": _window_average_core(y, window_size, h)}


def _chunk_sums(array: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    """Split an array into fixed-size chunks and return the sum of each chunk.

    Incomplete trailing elements are discarded. Returns an empty array when
    there are fewer elements than ``chunk_size``.

    Args:
        array: Input 1-D array.
        chunk_size: Size of each chunk.

    Returns:
        Array of shape (n // chunk_size,) containing per-chunk sums.
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
    """Aggregate a series into fixed-size chunks and forecast the next chunk sum.

    Drops any leading remainder so that the series length is divisible by
    ``aggregation_level``, sums each chunk, then applies
    :func:`_optimized_ses_forecast`.

    Args:
        y: Time series of shape (n,).
        aggregation_level: Number of observations per chunk.

    Returns:
        Scalar forecast of the next aggregated chunk sum.
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
    """Pack non-zero elements of x into a fixed-size array (JIT-compatible).

    Used by Croston-family models for intermittent demand.

    Args:
        x: Input array of shape (n,).

    Returns:
        Array of shape (n,) with non-zero values packed at the start,
        remaining positions filled with NaN.

    Example:
        >>> _demand(jnp.array([0., 5., 0., 3., 0.]))
        array([5., 3., nan, nan, nan])
    """
    indices = jnp.where(x > 0, size=x.size, fill_value=-1)[0]
    return jnp.where(
        indices >= 0,
        jnp.where(indices < x.size, x[jnp.clip(indices, 0, x.size - 1)], jnp.nan),
        jnp.nan,
    )


@jax.jit
def _intervals_c(x: jnp.ndarray) -> jnp.ndarray:
    """Compute inter-demand intervals for Croston-family models (JIT-compatible).

    Returns a fixed-size NaN-padded array suitable for use inside JIT kernels.

    Args:
        x: Input time series of shape (n,).

    Returns:
        Array of shape (n,) with inter-arrival intervals packed at the start,
        remaining positions filled with NaN.

    Example:
        >>> _intervals_c(jnp.array([0., 5., 0., 0., 3., 0., 2.]))
        array([1., 3., 2., nan, nan, nan, nan])
    """
    nonzero_idxs = jnp.where(x != 0, size=x.size, fill_value=-1)[0]
    positions = jnp.where(nonzero_idxs >= 0, nonzero_idxs + 1, -1)
    intervals = jnp.diff(positions, prepend=0)
    valid_mask = positions >= 0
    return jnp.where(valid_mask, intervals.astype(x.dtype), jnp.nan)


def _intervals(x: jnp.ndarray) -> jnp.ndarray:
    """Compute inter-demand intervals for ADIDA/IMAPA (variable-length output).

    Returns the differences between indices of non-zero elements. Unlike
    :func:`_intervals_c`, the output length equals the number of non-zero
    elements minus one and is **not** padded — use only outside JIT boundaries.

    Args:
        x: Input time series of shape (n,).

    Returns:
        Array of shape (k-1,) where k is the number of non-zero elements,
        containing the gaps between consecutive non-zero positions.
    """
    idx = jnp.where(x != 0)[0]
    padded = jnp.concatenate([jnp.array([0], dtype=idx.dtype), idx + 1])
    return jnp.diff(padded).astype(x.dtype)


@jax.jit
def _expand_fitted_demand(
    fitted: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """Expand SES demand fitted values back to the original series length.

    Used by Croston-family models to map per-demand-occurrence fitted values
    back onto the full time axis.

    Logic per position i (starting from 1):
      - y[i-1] > 0  → advance fitted index, use new fitted value
      - y[i-1] == 0 and demand seen → carry forward previous value
      - y[i-1] == 0 and no demand yet → use naive value (y[i-1])

    Args:
        fitted: SES fitted values of shape (num_nonzero + 1,).
        y: Original time series of shape (n,).

    Returns:
        Expanded fitted values of shape (n,).
    """
    n = y.size
    out = jnp.full_like(y, jnp.nan)

    def body_fn(i: int, state: Tuple) -> Tuple:
        out_arr, fitted_idx = state
        fitted_idx = jnp.where(y[i - 1] > 0, fitted_idx + 1, fitted_idx)
        val = jax.lax.cond(
            y[i - 1] > 0,
            lambda: fitted[fitted_idx],
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],
                lambda: y[i - 1],
            ),
        )
        return out_arr.at[i].set(val), fitted_idx

    out, _ = jax.lax.fori_loop(1, n, body_fn, (out, 0))
    return out


@jax.jit
def _expand_fitted_intervals(
    fitted: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """Expand SES interval fitted values back to the original series length.

    Used by Croston-family models to map per-demand-occurrence interval
    fitted values back onto the full time axis, avoiding division by zero.

    Logic per position i (starting from 1):
      - y[i-1] != 0 → advance fitted index, use new value (replace 0 with 1)
      - y[i-1] == 0 and intervals seen → carry forward previous value
      - y[i-1] == 0 and no intervals yet → use 1 (avoid division by zero)

    Args:
        fitted: SES fitted values for intervals of shape (num_nonzero + 1,).
        y: Original time series of shape (n,).

    Returns:
        Expanded interval fitted values of shape (n,).
    """
    n = y.size
    out = jnp.full_like(y, jnp.nan)

    def body_fn(i: int, state: Tuple) -> Tuple:
        out_arr, fitted_idx = state
        fitted_idx = jnp.where(y[i - 1] != 0, fitted_idx + 1, fitted_idx)
        val = jax.lax.cond(
            y[i - 1] != 0,
            lambda: jnp.where(fitted[fitted_idx] == 0, 1.0, fitted[fitted_idx]),
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],
                lambda: 1.0,
            ),
        )
        return out_arr.at[i].set(val), fitted_idx

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
    """Seasonal Exponential Smoothing: per-season SES with a shared alpha.

    Iterates over each season index in Python, applies SES to the
    sub-series for that season, then tiles the season forecasts to length h.
    Returns NaN mean if ``n < season_length``.

    Note: This is the Python-loop reference implementation kept for
    compatibility. The fully JIT-compiled vmap version lives in
    ``seasonal_exponential_smoothing.py``.

    Args:
        y: Time series of shape (n,).
        h: Forecast horizon.
        fitted: If True, also return in-sample fitted values.
        season_length: Number of observations per season (e.g. 12 for monthly).
        alpha: Smoothing parameter in [0, 1].

    Returns:
        Dict with ``"mean"`` of shape (h,) and optionally ``"fitted"`` of shape (n,).
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
    fcst: Dict[str, jnp.ndarray] = {"mean": out}
    if fitted:
        fcst["fitted"] = fitted_vals
    return fcst


def _seasonal_naive(
    y: jnp.ndarray,
    h: int,
    season_length: int,
    fitted: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Seasonal-naive forecast: repeat the last full season for h steps.

    JAX implementation equivalent to statsforecast's seasonal naive.

    The output dtype matches the input dtype — float32 inputs produce float32
    output, float64 inputs produce float64 output. Integer inputs are promoted
    to float32 via :func:`ensure_float`.

    Args:
        y: 1-D time series of shape (T,).
        h: Forecast horizon (positive integer).
        season_length: Seasonal period m (positive integer).
        fitted: If True, also return in-sample fitted values.

    Returns:
        Dict with:
          - ``"mean"``: shape (h,), same dtype as y.
          - ``"fitted"`` (if fitted=True): shape (T,), NaN for first m positions.

    Raises:
        ValueError: If inputs are invalid (non-1D, non-positive period, T < m).
    """
    y_j = ensure_float(jnp.asarray(y).squeeze())
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
    out: Dict[str, jnp.ndarray] = {"mean": mean}

    if fitted:
        fitted_arr = jnp.full((T,), jnp.nan, dtype=y_j.dtype)
        vals = y_j[: T - m]
        positions = jnp.arange(m, T)
        fitted_arr = fitted_arr.at[positions].set(vals)
        out["fitted"] = fitted_arr

    return out


def seasonal_decompose(
    y: jnp.ndarray,
    model: str = "additive",
    period: int = 1,
) -> Dict[str, jnp.ndarray]:
    """Seasonal decomposition using a moving-average trend estimate.

    Provides an interface similar to statsmodels' seasonal_decompose.

    Args:
        y: Time series of shape (n,).
        model: ``"additive"`` (y = trend + seasonal + resid) or
               ``"multiplicative"`` (y = trend * seasonal * resid).
        period: Seasonal period.

    Returns:
        Dict with ``"trend"``, ``"seasonal"``, and ``"resid"`` keys.
    """
    n = len(y)
    kernel = jnp.ones(period) / period
    trend = jnp.convolve(y, kernel, mode="same")

    if model == "additive":
        detrended = y - trend
        seasonal = jnp.tile(
            jnp.mean(detrended.reshape(-1, period), axis=0), n // period + 1
        )[:n]
        resid = detrended - seasonal
    else:
        detrended = y / trend
        seasonal = jnp.tile(
            jnp.mean(detrended.reshape(-1, period), axis=0), n // period + 1
        )[:n]
        resid = detrended / seasonal

    return {"trend": trend, "seasonal": seasonal, "resid": resid}


@_partial(jax.jit, static_argnums=(1, 2))
def _linear_extrapolate_tail(
    y: jnp.ndarray,
    tail_window: int,
    h: int,
) -> jnp.ndarray:
    """Fit a linear trend to the trailing window of y and extrapolate h steps.

    JIT-compiled; tail_window and h must be static.

    Args:
        y: Time series of shape (n,).
        tail_window: Number of trailing observations to fit the trend over.
        h: Number of steps to forecast.

    Returns:
        Forecast array of shape (h,).
    """
    n = y.shape[0]
    start = jnp.maximum(0, n - tail_window)
    seg = jax.lax.dynamic_slice(y, (start,), (tail_window,))
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

@partial(jax.jit, static_argnames=("max_k",))
def _imapa_aggregate_jit(y: jnp.ndarray, max_k: int) -> jnp.ndarray:
    """JIT-friendly aggregation loop for IMAPA with padded sums and masked SES.

    For each aggregation level k in 1..max_k:
      1. Segments y into k-sized chunks (dropping remainder).
      2. Sums each chunk to form an aggregated series.
      3. Applies :func:`_optimized_ses_forecast_masked` to forecast the next sum.
      4. Scales the forecast back by 1/k.

    Args:
        y: Time series of shape (n,).
        max_k: Maximum aggregation level. Must be static for JIT.

    Returns:
        Array of shape (max_k,) with per-level forecasts; NaN for k where
        n // k == 0.
    """
    dtype = y.dtype
    n = y.shape[0]
    forecasts = jnp.full((max_k,), jnp.asarray(jnp.nan, dtype=dtype))

    def body(k: int, forecasts_arr: jnp.ndarray) -> jnp.ndarray:
        n_chunks = n // k
        lost = n - (n_chunks * k)
        idx = jnp.arange(n)
        valid = idx >= lost
        y_masked = jnp.where(valid, y, jnp.asarray(0.0, dtype=dtype))
        seg_ids = jnp.maximum((idx - lost) // k, 0)
        padded = jnp.zeros((n,), dtype=dtype).at[seg_ids].add(y_masked)

        fcast = lax.cond(
            n_chunks == 0,
            lambda: jnp.asarray(jnp.nan, dtype=dtype),
            lambda: _optimized_ses_forecast_masked(padded, n_chunks) / jnp.asarray(k, dtype=dtype),
        )
        return forecasts_arr.at[k - 1].set(fcast)

    return lax.fori_loop(1, max_k + 1, body, forecasts)


def _imapa(
    y: jnp.ndarray,
    h: int,
    fitted: bool,
) -> Dict:
    """IMAPA (Intermittent Multiple Aggregation Prediction Algorithm) in pure JAX.

    Steps:
      1. Detect inter-arrival spacing of non-zero observations; compute mean interval.
      2. Use the rounded mean as the maximum aggregation level K.
      3. For k = 1..K: aggregate, fit SES (golden-section alpha search), scale back.
      4. Average per-k forecasts; optionally compute fitted values (O(T²)).

    Note: Fitted-value computation is expensive — it refits on every prefix.

    Args:
        y: Time series of shape (n,).
        h: Forecast horizon.
        fitted: Whether to compute in-sample fitted values (very slow for large T).

    Returns:
        Dict with:
          - ``"mean"``: constant forecast repeated h times
          - ``"fitted"`` (if fitted=True): shape (n,), first element NaN
    """
    if bool(jnp.all(y == 0)):
        out_dtype = y.dtype if y.dtype in (jnp.float32, jnp.float64) else jnp.float32
        res: Dict = {"mean": jnp.zeros((h,), dtype=out_dtype)}
        if fitted:
            f = jnp.zeros_like(ensure_float(y)).astype(out_dtype)
            f = f.at[0].set(jnp.asarray(jnp.nan, dtype=out_dtype))
            res["fitted"] = f
        return res

    y = ensure_float(y)
    dtype = y.dtype

    y_intervals = _intervals(y)
    mean_interval = jnp.mean(y_intervals)
    max_aggregation_level = max(1, int(jnp.rint(mean_interval).item()))

    forecasts = _imapa_aggregate_jit(y, max_aggregation_level)

    finite_mask = jnp.isfinite(forecasts)
    forecast = jnp.where(
        finite_mask.any(),
        jnp.mean(forecasts[finite_mask]),
        jnp.asarray(0.0, dtype=dtype),
    )

    res = {"mean": _repeat_val(val=forecast, h=h)}

    if fitted:
        warnings.warn("Computing fitted values for IMAPA is very expensive (O(T²)).")
        n = y.size
        fitted_vals = jnp.empty_like(y).at[0].set(jnp.asarray(jnp.nan, dtype=dtype))
        for i in range(n - 1):
            sub_res = _imapa(y[: i + 1], h=1, fitted=False)
            fitted_vals = fitted_vals.at[i + 1].set(sub_res["mean"][0])
        res["fitted"] = fitted_vals

    return res


# ============================================================
# SECTION 10 — Theta Model
# ============================================================

# Default optimisation bounds and starting values used by thetamodel().
# These are module-level constants; thetamodel() may override them per-call.
_THETA_INIT_LEVEL: float = 0.5
_THETA_INIT_ALPHA: float = 0.3
_THETA_INIT_THETA: float = 2.0
_THETA_OPT_LEVEL: bool = True
_THETA_OPT_ALPHA: bool = True
_THETA_OPT_THETA: bool = True
_THETA_LOWER: jnp.ndarray = jnp.array([0.0, 0.0, 1.0])
_THETA_UPPER: jnp.ndarray = jnp.array([1.0, 1.0, 10.0])


def switch_theta(model: str) -> _theta.ModelType:
    """Map a theta model name string to its :class:`theta_jax.ModelType` enum value.

    Args:
        model: One of ``"STM"``, ``"OTM"``, ``"DSTM"``, ``"DOTM"``.

    Returns:
        Corresponding :class:`theta_jax.ModelType` enum member.

    Raises:
        ValueError: If model is not one of the four valid strings.
    """
    mapping = {
        "STM": _theta.ModelType.STM,
        "OTM": _theta.ModelType.OTM,
        "DSTM": _theta.ModelType.DSTM,
        "DOTM": _theta.ModelType.DOTM,
    }
    if model not in mapping:
        raise ValueError(f"Invalid theta model type: '{model}'. Choose from {list(mapping)}.")
    return mapping[model]


def compute_pi_samples(
    n: int,
    h: int,
    states: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: float,
    theta: float,
    mean_y: jnp.ndarray,
    seed: int = 0,
    n_samples: int = 200,
) -> jnp.ndarray:
    """Generate Monte-Carlo forecast samples for theta model prediction intervals.

    Args:
        n: Number of observed time steps.
        h: Forecast horizon.
        states: State array of shape (n, 4+) from :func:`thetamodel`.
        sigma: Residual standard deviation.
        alpha: SES smoothing parameter.
        theta: Theta parameter.
        mean_y: Mean of the training series.
        seed: PRNG seed for reproducibility.
        n_samples: Number of Monte-Carlo samples per horizon step.

    Returns:
        Sample matrix of shape (h, n_samples).
    """
    samples = jnp.full((h, n_samples), jnp.nan, dtype=jnp.float32)
    level, _, A, B = states[-1, :4]  # meany from states unused; mean_y arg is used
    smoothed = level
    key = jrandom.PRNGKey(seed)

    def body_fun(i: int, val: Tuple) -> Tuple:
        smoothed, mean_y, A, B, samples, key = val
        mu = smoothed + (1 - 1 / theta) * (
            A * ((1 - alpha) ** i) + B * (1 - (1 - alpha) ** (i + 1)) / alpha
        )
        key, subkey = jrandom.split(key)
        eps = jrandom.normal(subkey, shape=(n_samples,), dtype=jnp.float32) * sigma
        s = mu + eps
        smoothed_new = (alpha * jnp.mean(s) + (1 - alpha) * smoothed).astype(jnp.float32)
        mean_y_new = ((i * mean_y + jnp.mean(s)) / (i + 1)).astype(jnp.float32)
        B_new = (((i - 1) * B + 6 * (jnp.mean(s) - mean_y_new) / (i + 1)) / (i + 2)).astype(jnp.float32)
        A_new = (mean_y_new - B_new * (i + 2) / 2).astype(jnp.float32)
        samples = samples.at[i - n].set(s)
        return smoothed_new, mean_y_new, A_new, B_new, samples, key

    smoothed, mean_y, A, B, samples, key = jax.lax.fori_loop(
        n,
        n + h,
        body_fun,
        (
            jnp.array(smoothed, dtype=jnp.float32),
            jnp.array(mean_y, dtype=jnp.float32),
            jnp.array(A, dtype=jnp.float32),
            jnp.array(B, dtype=jnp.float32),
            samples,
            key,
        ),
    )
    return samples


def initparamtheta(
    initial_smoothed: float,
    alpha: float,
    theta: float,
    y: jnp.ndarray,
    modeltype: _theta.ModelType,
) -> Dict:
    """Initialise theta model parameters and determine which to optimise.

    For STM/DSTM models, theta is fixed at 2.0 (not optimised).
    For OTM/DOTM models, theta is also optimised if not supplied.

    Args:
        initial_smoothed: Initial level; NaN triggers automatic initialisation.
        alpha: Smoothing parameter; NaN triggers automatic initialisation.
        theta: Theta parameter; NaN triggers automatic initialisation.
        y: Training time series.
        modeltype: :class:`theta_jax.ModelType` enum value.

    Returns:
        Dict with keys ``initial_smoothed``, ``alpha``, ``theta`` (values) and
        ``optimize_initial_smoothed``, ``optimize_alpha``, ``optimize_theta`` (flags).
    """
    if modeltype in [_theta.ModelType.STM, _theta.ModelType.DSTM]:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        optimize_alpha = math.isnan(alpha)
        if optimize_alpha:
            alpha = 0.5
        theta = 2.0
        optimize_theta = False
    else:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        optimize_alpha = math.isnan(alpha)
        if optimize_alpha:
            alpha = 0.5
        optimize_theta = math.isnan(theta)
        if optimize_theta:
            theta = 2.0

    return {
        "initial_smoothed": initial_smoothed,
        "optimize_initial_smoothed": optimize_level,
        "alpha": alpha,
        "optimize_alpha": optimize_alpha,
        "theta": theta,
        "optimize_theta": optimize_theta,
    }


def optimize_theta_target_fn(
    init_par: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    init_level: float,
    init_alpha: float,
    init_theta: float,
    opt_level: bool,
    opt_alpha: bool,
    opt_theta: bool,
    y: jnp.ndarray,
    modeltype: _theta.ModelType,
    nmse: int,
) -> "results":
    """Optimise theta model parameters using :func:`theta_jax.minimize`.

    Args:
        init_par: Initial parameter vector (level, alpha, theta).
        lower: Lower bounds array of shape (3,).
        upper: Upper bounds array of shape (3,).
        init_level: Starting level value.
        init_alpha: Starting alpha value.
        init_theta: Starting theta value.
        opt_level: Whether to optimise the level parameter.
        opt_alpha: Whether to optimise alpha.
        opt_theta: Whether to optimise theta.
        y: Training time series.
        modeltype: :class:`theta_jax.ModelType` enum value.
        nmse: Number of steps for mean-square error computation.

    Returns:
        :class:`results` namedtuple with fields ``x``, ``fn``, ``nit``, ``simplex``.
    """
    opt_res = _theta.minimize(
        x0=init_par,
        lower=lower,
        upper=upper,
        init_level=init_level,
        init_alpha=init_alpha,
        init_theta=init_theta,
        opt_level=opt_level,
        opt_alpha=opt_alpha,
        opt_theta=opt_theta,
        y=y,
        model_type=modeltype,
        nmse=nmse,
    )
    x = jax.device_get(opt_res["x"])
    fn = float(jax.device_get(opt_res["fun"]))
    nit = int(opt_res.get("nit", 0))
    return results(x, fn, nit, None)


def thetamodel(
    y: jnp.ndarray,
    m: int,
    modeltype: str,
    initial_smoothed: float,
    alpha: float,
    theta: float,
    nmse: int,
) -> Dict:
    """Fit a Theta model to a time series.

    Args:
        y: Training time series (will be cast to float64).
        m: Seasonal period.
        modeltype: Model variant: ``"STM"``, ``"OTM"``, ``"DSTM"``, or ``"DOTM"``.
        initial_smoothed: Initial level; NaN for automatic initialisation.
        alpha: Smoothing parameter; NaN for automatic optimisation.
        theta: Theta parameter; NaN for automatic optimisation.
        nmse: Number of multi-step-ahead errors used for MSE computation.

    Returns:
        Dict with keys ``mse``, ``amse``, ``fit``, ``residuals``, ``m``,
        ``states``, ``par``, ``n``, ``modeltype``, ``mean_y``.
    """
    y = y.astype(jnp.float64, copy=False)
    model_type = switch_theta(modeltype)

    par = initparamtheta(
        initial_smoothed=initial_smoothed,
        alpha=alpha,
        theta=theta,
        y=y,
        modeltype=model_type,
    )
    optimize_params = {
        key.replace("optimize_", ""): val for key, val in par.items() if "optim" in key
    }
    x0 = jnp.array([par["initial_smoothed"], par["alpha"], par["theta"]], dtype=jnp.float32)

    fred = optimize_theta_target_fn(
        init_par=x0,
        lower=_THETA_LOWER,
        upper=_THETA_UPPER,
        init_level=_THETA_INIT_LEVEL,
        init_alpha=_THETA_INIT_ALPHA,
        init_theta=_THETA_INIT_THETA,
        opt_level=_THETA_OPT_LEVEL,
        opt_alpha=_THETA_OPT_ALPHA,
        opt_theta=_THETA_OPT_THETA,
        y=y,
        modeltype=model_type,
        nmse=nmse,
    )

    if fred is not None:
        fit_par = fred.x
    j = 0
    if optimize_params.get("initial_smoothed", False):
        par["initial_smoothed"] = float(fit_par[j]); j += 1
    if optimize_params.get("alpha", False):
        par["alpha"] = float(fit_par[j]); j += 1
    if optimize_params.get("theta", False):
        par["theta"] = float(fit_par[j]); j += 1

    amse, e, states, mse = _theta.pegels_resid(
        y, model_type, par["initial_smoothed"], par["alpha"], par["theta"], nmse
    )

    return dict(
        mse=mse,
        amse=amse,
        fit=fred,
        residuals=e,
        m=m,
        states=states,
        par=par,
        n=len(y),
        modeltype=modeltype,
        mean_y=jnp.mean(y),
    )


def forecast_theta(
    obj: Dict,
    h: int,
    level: Optional[List[int]] = None,
) -> Dict:
    """Generate h-step-ahead forecasts from a fitted Theta model.

    Args:
        obj: Fitted model dict returned by :func:`thetamodel` or :func:`auto_theta`.
        h: Forecast horizon.
        level: Optional list of confidence levels (0-100) for prediction intervals.

    Returns:
        Dict with ``"mean"`` key and optionally ``"lo-{lv}"``/``"hi-{lv}"`` keys.
    """
    n_obs = obj["n"]
    states = obj["states"]
    if states.ndim == 1 or states.shape[1] != 4:
        states = jnp.zeros((n_obs, 5), dtype=jnp.float32)
        states = states.at[:, 0].set(obj.get("mean_y", 0.0))
        states = states.at[:, 1].set(obj.get("mean_y", 0.0))

    alpha = obj["par"]["alpha"]
    theta = obj["par"]["theta"]
    model_type = switch_theta(obj["modeltype"])

    states, forecast = _theta.forecast(states, h, model_type, alpha, theta)
    res: Dict = {"mean": forecast}

    if level is not None:
        sigma = jnp.std(obj["residuals"][3:], ddof=1)
        mean_y = obj["mean_y"]
        samples = compute_pi_samples(
            n=obj["n"],
            h=h,
            states=states,
            sigma=sigma,
            alpha=alpha,
            theta=theta,
            mean_y=mean_y,
        )
        for lv in level:
            min_q = (100 - lv) / 200
            max_q = min_q + lv / 100
            res[f"lo-{lv}"] = jnp.quantile(samples, min_q, axis=1)
            res[f"hi-{lv}"] = jnp.quantile(samples, max_q, axis=1)

    if obj.get("decompose", False):
        seas_forecast = _repeat_val_seas(obj["seas_forecast"]["mean"], h=h)
        for key in res:
            if obj["decomposition_type"] == "multiplicative":
                res[key] = res[key] * seas_forecast
            else:
                res[key] = res[key] + seas_forecast

    return res


def is_constant(x: jnp.ndarray) -> jnp.ndarray:
    """Return True if all elements of x are equal to x[0].

    Args:
        x: 1-D JAX array.

    Returns:
        Boolean scalar JAX array.
    """
    return jnp.all(x[0] == x)


def acf(x: jnp.ndarray, nlags: int) -> jnp.ndarray:
    """Compute the autocorrelation function up to ``nlags`` lags.

    Equivalent to ``statsmodels.tsa.stattools.acf(x, nlags=nlags)``.

    Args:
        x: 1-D time series array.
        nlags: Number of lags to compute.

    Returns:
        Array of shape (nlags + 1,) with ACF values at lags 0..nlags.
    """
    x = x - jnp.mean(x)
    n = x.shape[0]
    denom = jnp.dot(x, x)
    return jnp.array([jnp.dot(x[: n - lag], x[lag:]) / denom for lag in range(nlags + 1)])


def auto_theta(
    y: jnp.ndarray,
    m: int,
    model: Optional[str] = None,
    initial_smoothed: Optional[float] = None,
    alpha: Optional[float] = None,
    theta: Optional[float] = None,
    nmse: int = 3,
    decomposition_type: str = "multiplicative",
) -> Dict:
    """Automatically select and fit the best Theta model variant.

    Fits all four variants (STM, OTM, DSTM, DOTM) unless ``model`` is
    specified, selects by MSE, and optionally handles seasonal decomposition.

    Args:
        y: Training time series.
        m: Seasonal period.
        model: Fix to a specific variant (``"STM"``, ``"OTM"``, ``"DSTM"``,
               ``"DOTM"``), or None to auto-select.
        initial_smoothed: Initial level; None for automatic initialisation.
        alpha: Smoothing parameter; None for automatic optimisation.
        theta: Theta parameter; None for automatic optimisation.
        nmse: Number of multi-step-ahead errors (1-30).
        decomposition_type: ``"multiplicative"`` or ``"additive"`` seasonal
                            decomposition (applied when ``m >= 4``).

    Returns:
        Fitted model dict from :func:`thetamodel`, possibly with decomposition
        metadata (``"decompose"``, ``"decomposition_type"``, ``"seas_forecast"``).

    Raises:
        ValueError: If nmse is out of range or model is invalid.
        NotImplementedError: If the series is too short for optimisation.
        Exception: If no model could be fitted.
    """
    if initial_smoothed is None:
        initial_smoothed = jnp.nan
    if alpha is None:
        alpha = jnp.nan
    if theta is None:
        theta = jnp.nan
    if nmse < 1 or nmse > 30:
        raise ValueError("nmse out of range (must be 1-30)")

    if is_constant(y):
        thetamodel(
            y=y, m=m, modeltype="STM", nmse=nmse,
            initial_smoothed=jnp.mean(y) / 2, alpha=0.5, theta=2.0,
        )

    decompose = False
    if m >= 4 and len(y) >= 2 * m:
        r = acf(y, nlags=m)[1:]
        stat = jnp.sqrt((1 + 2 * jnp.sum(r[:-1] ** 2)) / len(y))
        decompose = jnp.abs(r[-1]) / stat > norm.ppf(0.95)

    data_positive = min(y) > 0
    seas_forecast = None
    if decompose:
        if decomposition_type == "multiplicative" and not data_positive:
            decomposition_type = "additive"
        y_decompose = seasonal_decompose(y, model=decomposition_type, period=m)["seasonal"]
        if decomposition_type == "multiplicative" and any(y_decompose < 0.01):
            decomposition_type = "additive"
            y_decompose = seasonal_decompose(y, model="additive", period=m)["seasonal"]
        y = y - y_decompose if decomposition_type == "additive" else y / y_decompose
        seas_forecast = _seasonal_naive(y=y_decompose, h=m, season_length=m, fitted=False)

    if model not in [None, "STM", "OTM", "DSTM", "DOTM"]:
        raise ValueError(f"Invalid model type: '{model}'.")

    n = len(y)
    if n <= 3:
        raise NotImplementedError("Series too short for theta model optimisation (n <= 3).")

    modeltypes = [model] if model is not None else ["STM", "OTM", "DSTM", "DOTM"]

    best_ic = jnp.inf
    best_model = None
    for mtype in modeltypes:
        fit = thetamodel(
            y=y, m=m, modeltype=mtype, nmse=nmse,
            initial_smoothed=initial_smoothed, alpha=alpha, theta=theta,
        )
        fit_ic = fit["mse"]
        if not jnp.isnan(fit_ic) and fit_ic < best_ic:
            best_model = fit
            best_ic = fit_ic

    if jnp.isinf(best_ic):
        raise Exception("No theta model variant could be fitted to the data.")

    if decompose:
        if decomposition_type == "multiplicative":
            best_model["residuals"] = best_model["residuals"] * y_decompose
        else:
            best_model["residuals"] = best_model["residuals"] + y_decompose
        best_model["decompose"] = decompose
        best_model["decomposition_type"] = decomposition_type
        best_model["seas_forecast"] = dict(seas_forecast)

    return best_model


def forward_theta(
    fitted_model: Dict,
    y: jnp.ndarray,
) -> Dict:
    """Re-fit a Theta model on new data using the parameters from a prior fit.

    Args:
        fitted_model: Dict returned by :func:`auto_theta` (contains ``"m"``,
                      ``"modeltype"``, ``"par"``).
        y: New training time series.

    Returns:
        New fitted model dict from :func:`auto_theta`.
    """
    return auto_theta(
        y=y,
        m=fitted_model["m"],
        model=fitted_model["modeltype"],
        initial_smoothed=fitted_model["par"]["initial_smoothed"],
        alpha=fitted_model["par"]["alpha"],
        theta=fitted_model["par"]["theta"],
    )


# ============================================================
# SECTION 11 — Information Criteria
# ============================================================

@jax.jit
def calculate_information_criteria(
    residuals: jnp.ndarray,
    n_params: int,
    n: int,
) -> Dict[str, jnp.ndarray]:
    """Compute AIC, BIC, and AICc from model residuals.

    JIT-compiled; returns JAX arrays for use inside traced functions.

    Args:
        residuals: Model residuals of shape (n,).
        n_params: Number of free parameters in the model.
        n: Number of observations.

    Returns:
        Dict with keys ``"loglik"``, ``"aic"``, ``"bic"``, ``"aicc"``.
        ``"aicc"`` is ``inf`` when n - n_params - 1 <= 0.
    """
    sse = jnp.sum(residuals ** 2)
    lik = n * jnp.log(sse + 1e-10)

    aic = lik + 2 * n_params
    bic = lik + jnp.log(n) * n_params
    denom = n - n_params - 1
    aicc = jnp.where(
        denom > 0,
        aic + (2 * n_params * (n_params + 1)) / denom,
        jnp.inf,
    )

    return {
        "loglik": -0.5 * lik,
        "aic": aic,
        "bic": bic,
        "aicc": aicc,
    }

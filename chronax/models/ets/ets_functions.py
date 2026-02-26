# ets.py (JAX version)
"""
High-level ETS (Exponential Smoothing) model API in JAX.

This module wires together:
- low-level ETS kernels and an Optax Adam optimizer (from `_ets`),
- parameter initialization and admissibility checks,
- state initialization via simple regression/Fourier or seasonal decomposition,
- full model selection / fitting (AICc, etc.),
- forecasting and analytical/simulation-based prediction intervals.

The goal is functional parity with NumPy/statsmodels-style ETS flows while
keeping kernels JIT-friendly and deterministic.
"""

__all__ = ['ets_f']

import math
import os
import time
from functools import partial
from typing import Dict, Any, Optional
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrand
from jax import lax

from . import ets_backend as _ets
from chronax.utils import _calculate_intervals, results

# Global variables
_smalno = jnp.finfo(float).eps
_PHI_LOWER = 0.8
_PHI_UPPER = 0.98
_AICC_RATIO_THRESHOLD = 40.0
EPS = 1e-3
EPS_PURE = 2e-2


def _estimate_ets_iterations(
    y: jnp.ndarray,
    m: int,
    error_type: str,
    trend_type: str,
    season_type: str,
    allow_extended: bool = False,
) -> int:
    y = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    if n < 3:
        return 50

    diffs = y[1:] - y[:-1]
    cv = jnp.std(diffs) / jnp.maximum(jnp.abs(jnp.mean(diffs)), 1e-10)
    noise_score = float(jnp.clip(cv / 5.0, 0.0, 1.0))

    t = jnp.arange(n, dtype=jnp.float64)
    y_mean = jnp.mean(y)
    t_mean = jnp.mean(t)
    slope = jnp.sum((t - t_mean) * (y - y_mean)) / jnp.maximum(
        jnp.sum((t - t_mean) ** 2), 1e-10
    )
    y_pred = y_mean + slope * (t - t_mean)
    ss_res = jnp.sum((y - y_pred) ** 2)
    ss_tot = jnp.sum((y - y_mean) ** 2)
    r2 = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-10)
    trend_difficulty = float(1.0 - jnp.clip(r2, 0.0, 1.0))

    if m > 1 and n >= 2 * m:
        y_detrended = y - y_pred
        y_c = y_detrended - jnp.mean(y_detrended)
        var_y = jnp.var(y_detrended)
        n_acf = n - m
        acf_m = jnp.sum(y_c[:n_acf] * y_c[m:]) / (n_acf * jnp.maximum(var_y, 1e-10))
        seasonality_difficulty = float(
            1.0 - jnp.maximum(jnp.clip(acf_m, -1.0, 1.0), 0.0)
        )
    else:
        seasonality_difficulty = 0.0

    model_complexity = 0.0
    if error_type == "M":
        model_complexity += 0.2
    if trend_type == "M":
        model_complexity += 0.15
    if season_type == "M":
        model_complexity += 0.15

    n_seasons = n // m if m > 1 else 0
    if n_seasons < 3:
        coverage_penalty = 0.3
    elif n_seasons < 5:
        coverage_penalty = 0.15
    else:
        coverage_penalty = 0.0

    complexity = (
        0.30 * noise_score
        + 0.25 * trend_difficulty
        + 0.20 * seasonality_difficulty
        + 0.15 * model_complexity
        + 0.10 * coverage_penalty
    )

    # Use a reasonably wide iteration range; `allow_extended=True` lets callers trade
    # cold-start time for extra accuracy on harder series.
    min_iters = 50
    max_iters = 600 if allow_extended else 350
    iterations = int(min_iters + (complexity**2) * (max_iters - min_iters))
    return max(min_iters, min(max_iters, iterations))


# Global cache for expensive strength calculations
_strength_cache: dict[tuple[int, int], tuple[float, float]] = {}


class _TemplateCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, str, str], jnp.ndarray] = {}

    def get_init_state(
        self, y: jnp.ndarray, m: int, trendtype: str, seasontype: str
    ) -> jnp.ndarray:
        try:
            y_hash = hash(jnp.asarray(y).tobytes())
        except Exception:
            y_hash = id(y)
        key = (y_hash, m, trendtype, seasontype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        init_state = _prepare_init_state(y, m, trendtype, seasontype)
        self._cache[key] = init_state
        return init_state


_template_cache = _TemplateCache()


def _get_cached_strengths(y: jnp.ndarray, m: int) -> tuple[float, float]:
    """
    Cached version of trend and seasonal strength calculations.
    Returns (trend_strength, seasonal_strength).
    """
    try:
        y_hash = hash(y.tobytes())
        cache_key = (y_hash, m)
        cached = _strength_cache.get(cache_key)
        if cached is not None:
            return cached
    except Exception:
        cache_key = None

    ts = _trend_strength(y)
    ss = _seasonal_strength(y, m) if m > 1 else 0.0

    if cache_key is not None:
        _strength_cache[cache_key] = (ts, ss)
    return ts, ss


def _aicc(aic: float, n: int, k: int) -> float:
    """
    Small-sample AICc with enhanced penalty for tiny datasets.
    """
    if k <= 0:
        return aic
    if (n / k) < _AICC_RATIO_THRESHOLD:
        denom = n - k - 1
        if denom <= 0:
            return float(jnp.inf)
        base_aicc = aic + 2.0 * k * (k + 1) / denom
    else:
        base_aicc = aic
    if n < 150:
        extra_penalty = k * (150 - n) / 150.0
        base_aicc = base_aicc + extra_penalty
    return base_aicc


def _trend_strength(y: jnp.ndarray) -> float:
    """Calculate trend strength using detrended residuals (FIXED)"""
    y = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    t = jnp.arange(n, dtype=jnp.float64)
    t = t - jnp.mean(t)
    y0 = y - jnp.mean(y)
    denom = jnp.sum(t * t) + 1e-8
    slope = jnp.sum(t * y0) / denom

    # FIXED: Use detrended residual variance
    trend_line = jnp.mean(y) + slope * t
    resid_std = jnp.std(y - trend_line) + 1e-8

    return float(jnp.abs(slope) / resid_std)


def _seasonal_strength(y: jnp.ndarray, m: int) -> float:
    """Calculate seasonal strength using detrended data (FIXED)"""
    y = jnp.asarray(y, dtype=jnp.float64)
    if m <= 1 or y.shape[0] < 2 * m:
        return 0.0

    n = int(y.shape[0])

    # FIXED: Detrend first
    t = jnp.arange(n, dtype=jnp.float64)
    t = t - jnp.mean(t)
    y0 = y - jnp.mean(y)
    slope = jnp.sum(t * y0) / (jnp.sum(t * t) + 1e-8)
    y_detrended = y - (jnp.mean(y) + slope * t)

    # Now measure seasonality on detrended series
    n_periods = n // m
    y_trim = y_detrended[: n_periods * m].reshape(n_periods, m)
    seasonal_means = jnp.mean(y_trim, axis=0)

    return float(jnp.std(seasonal_means) / (jnp.std(y_detrended) + 1e-8))


def _prepare_init_state(
    y: jnp.ndarray, m: int, trendtype: str, seasontype: str
) -> jnp.ndarray:
    """
    Build initial state vector (level [+ trend] [+ season]) and append the
    seasonal balancing term for multiplicative seasonality.
    """
    init_state = initstate(jnp.asarray(y, dtype=jnp.float64), m, trendtype, seasontype)
    if seasontype == "N":
        return init_state
    nstate = int(init_state.shape[0])
    start = 1 + int(trendtype != "N")
    tail = m * (seasontype == "M") - jnp.sum(init_state[start:nstate])
    return jnp.hstack([init_state, jnp.array([tail], dtype=jnp.float64)])


def _compute_ic(lik: float, n: int, k: int) -> tuple[float, float, float]:
    """
    Compute AIC/BIC/AICc from the scalar objective `lik`.
    """
    aic = float(lik) + 2 * k
    bic = float(lik) + math.log(n) * k
    aicc = _aicc(aic=aic, n=n, k=k)
    return aic, bic, float(aicc)


def _infer_season_length(y: jnp.ndarray, max_m: int = 24) -> int:
    """
    Infer a plausible season length from the series using a simple FFT peak.

    Returns 1 if no clear seasonal peak is found.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    if n < 8:
        return 1
    max_m = int(min(max_m, n // 2))
    if max_m <= 1:
        return 1

    y0 = y - jnp.mean(y)
    # FFT over full series; ignore DC component
    fft = jnp.fft.rfft(y0)
    power = jnp.abs(fft) ** 2
    power = power.at[0].set(0.0)

    # Consider only candidate seasonal periods in [2, max_m]
    periods = jnp.arange(2, max_m + 1, dtype=jnp.int32)
    # map period -> frequency bin ~ n/period
    bins = jnp.clip(jnp.round(n / periods).astype(jnp.int32), 1, power.shape[0] - 1)
    cand_power = power[bins]
    idx = int(jnp.argmax(cand_power))
    best_m = int(periods[idx])

    # Require a minimum signal-to-noise for seasonality
    if float(cand_power[idx]) < 0.05 * float(jnp.max(power)):
        return 1
    return best_m


def _choose_unified_components(
    y: jnp.ndarray,
    m: int,
    allow_multiplicative_trend: bool,
) -> tuple[str, str, str]:
    """Unified heuristic for component selection (FIXED THRESHOLDS)"""
    y = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    positive = float(jnp.min(y)) > 0
    cv = float(jnp.std(y) / (jnp.mean(y) + 1e-8))
    trend_strength, seas_strength = _get_cached_strengths(y, m)
    

    cycles = n // max(m, 1)
    if positive and cv > 0.3 and cycles > 10:
        err = "M"
    else:
        err = "A"
    

    season = "N"
    if m > 1 and cycles >= 10:
        if positive and seas_strength > 0.12 and cv > 0.2 and cycles > 10:
            season = "M"
        elif seas_strength > 0.05:  # FIXED: was 0.1
            season = "A"
    

    # FIXED: Lowered threshold from 0.08 to 0.03
    if trend_strength < 0.03 or n < 50:
        trend = "N"
    else:
        trend = "M" if (allow_multiplicative_trend and positive and cv > 0.4 and n > 100) else "A"
    

    return err, trend, season


def _inv_sigmoid_scaled(val: float, low: float, high: float) -> float:
    z = (val - low) / (high - low)
    z = jnp.clip(z, 1e-4, 1.0 - 1e-4)
    return float(jnp.log(z / (1.0 - z)))


def _transform_smoothing_params_unconstrained(
    p: jnp.ndarray,
    opt_names: list[str],
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    pure_sigmoid: bool = False,
) -> tuple[float, float, float, float]:
    idx = 0
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)
    if pure_sigmoid:
        if "alpha" in opt_names:
            alpha = float(jax.nn.sigmoid(p[idx]))
            alpha = float(EPS_PURE + (1.0 - 2.0 * EPS_PURE) * alpha)
            idx += 1
        if "beta" in opt_names:
            beta = float(jax.nn.sigmoid(p[idx]))
            beta = float(alpha * beta)
            idx += 1
        if "gamma" in opt_names:
            gamma = float(jax.nn.sigmoid(p[idx]))
            gamma = float((1.0 - alpha) * gamma)
            idx += 1
        if "phi" in opt_names:
            ph = float(jax.nn.sigmoid(p[idx]))
            ph = float(EPS_PURE + (1.0 - 2.0 * EPS_PURE) * ph)
            phi = float(lower[3] + (upper[3] - lower[3]) * ph)
        return alpha, beta, gamma, phi

    if "alpha" in opt_names:
        a = float(jax.nn.sigmoid(p[idx] * 0.1))
        alpha = float(lower[0] + (upper[0] - lower[0]) * a)
        idx += 1
    if "beta" in opt_names:
        b = float(jax.nn.sigmoid(p[idx]))
        beta = float(alpha * b)
        idx += 1
    if "gamma" in opt_names:
        g = float(jax.nn.sigmoid(p[idx]))
        gamma = float((1.0 - alpha) * g)
        idx += 1
    if "phi" in opt_names:
        ph = float(jax.nn.sigmoid(p[idx]))
        phi = float(lower[3] + (upper[3] - lower[3]) * ph)
    return alpha, beta, gamma, phi


@partial(jax.jit, static_argnames=("m", "error", "trend", "season", "h"))
def _etssimulate_jit(
    x: jnp.ndarray,
    m: int,
    error: _ets.Component,
    trend: _ets.Component,
    season: _ets.Component,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    h: int,
    e: jnp.ndarray,
) -> jnp.ndarray:
    """
    Simulate h-step future sample paths from a given ETS state.

    Parameters
    ----------
    x : jnp.ndarray
        Initial state vector (level [+ trend] [+ m seasonal entries]).
    m : int
        Seasonal period (>=1). If `season != Nothing`, must be <= 24 here.
    error, trend, season : _ets.Component
        ETS structure: additive/multiplicative or absent.
    alpha, beta, gamma, phi : float
        Smoothing parameters (beta/phi used only if trend present; gamma only if seasonal).
    h : int
        Forecast horizon to simulate.
    y : jnp.ndarray
        Preallocated output array of length h; filled with simulated values.
    e : jnp.ndarray
        Innovations (length h) already drawn (e.g., from N(0, sigma)).

    Notes
    -----
    This is a *stateful* helper used by interval simulation for Class 4/5
    models; it mirrors the kernel `_ets.update` and `_ets.forecast` behavior.
    """
    if m > 24 and season != _ets.Component.Nothing:
        return jnp.zeros((h,), dtype=x.dtype)
    elif m < 1:
        m = 1
    dt = x.dtype
    oldb = jnp.asarray(0.0, dtype=dt)
    # Copy initial state components
    l = jnp.asarray(x[0], dtype=dt)
    if trend != _ets.Component.Nothing:
        b = jnp.asarray(x[1], dtype=dt)
    else:
        b = jnp.asarray(0.0, dtype=dt)
    if season != _ets.Component.Nothing:
        # x offset = 1 + (trend != Nothing)
        off = 1 + int(trend != _ets.Component.Nothing)
        s = jnp.asarray(x[off : off + m], dtype=dt)
    else:
        s = jnp.zeros((m,), dtype=dt)

    y = jnp.zeros((h,), dtype=dt)

    def step(i, carry):
        l, b, s, y, alive = carry

        def do_alive(carry_inner):
            l_i, b_i, s_i, y_i, _ = carry_inner
            oldl = l_i
            oldb = b_i
            olds = s_i
            f = _ets.forecast(
                jnp.zeros((1,), dtype=dt),
                oldl,
                oldb,
                olds,
                m,
                trend,
                season,
                jnp.asarray(phi, dtype=dt),
                1,
            )
            invalid = jnp.abs(f[0] - _ets.NA) < _ets.TOL

            def on_invalid(args):
                l_j, b_j, s_j, y_j = args
                y_j = y_j.at[0].set(_ets.NA)
                return l_j, b_j, s_j, y_j, False

            def on_valid(args):
                l_j, b_j, s_j, y_j = args
                if error == _ets.Component.Additive:
                    y_val = f[0] + e[i]
                else:
                    y_val = f[0] * (1.0 + e[i])
                y_j = y_j.at[i].set(y_val)
                l_new, b_new, s_new = _ets.update(
                    s_j,
                    l_j,
                    b_j,
                    oldl,
                    oldb,
                    olds,
                    m,
                    trend,
                    season,
                    int(error.value),
                    jnp.asarray(alpha, dtype=dt),
                    jnp.asarray(beta, dtype=dt),
                    jnp.asarray(gamma, dtype=dt),
                    jnp.asarray(phi, dtype=dt),
                    y_val,
                )
                return l_new, b_new, s_new, y_j, True

            return lax.cond(invalid, on_invalid, on_valid, (l_i, b_i, s_i, y_i))

        def do_dead(carry_inner):
            return carry_inner

        return lax.cond(alive, do_alive, do_dead, carry)

    l, b, s, y, _ = lax.fori_loop(0, h, step, (l, b, s, y, True))
    return y


def etssimulate(
    x: jnp.ndarray,
    m: int,
    error: _ets.Component,
    trend: _ets.Component,
    season: _ets.Component,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    h: int,
    y: jnp.ndarray,
    e: jnp.ndarray,
) -> jnp.ndarray:
    """
    Simulate h-step future sample paths from a given ETS state.
    """
    y = _etssimulate_jit(
        x,
        m,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        h,
        e,
    )
    return y


@partial(jax.jit, static_argnames=("m", "trend", "season", "h"))
def etsforecast(
    x: jnp.ndarray,
    m: int,
    trend: _ets.Component,
    season: _ets.Component,
    phi: float,
    h: int,
    f: jnp.ndarray,
) -> jnp.ndarray:
    """
    Produce h-step-ahead forecasts from a state snapshot.

    Parameters
    ----------
    x : jnp.ndarray
        State vector (level [+ trend] [+ m seasonal]).
    m : int
        Seasonal period (>=1).
    trend, season : _ets.Component
        Structural flags for trend/seasonality.
    phi : float
        Damping parameter (ignored if no trend).
    h : int
        Number of steps to forecast.
    f : jnp.ndarray
        Optional preallocated buffer (length h); created if None/wrong shape.

    Returns
    -------
    jnp.ndarray
        Forecasts of shape (h,).
    """
    if m < 1:
        m = 1
    dt = x.dtype

    l = jnp.asarray(x[0], dtype=dt)
    has_trend = (trend != _ets.Component.Nothing)
    b = jnp.asarray(x[1], dtype=dt) if has_trend else jnp.asarray(0.0, dtype=dt)

    if season != _ets.Component.Nothing:
        start = 1 + int(has_trend)
        s = jnp.asarray(x[start:start + m], dtype=dt)
    else:
        s = jnp.zeros((m,), dtype=dt)

    if (f is None) or (getattr(f, "shape", ()) != (h,)):
        f = jnp.zeros((h,), dtype=dt)

    f = _ets.forecast(
        f=f,
        l=l,
        b=b,
        s=s,
        m=int(m),
        trend=trend,
        season=season,
        phi=jnp.asarray(phi, dtype=dt),
        h=int(h),
    )
    return f


def initparam(
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    trendtype: str,
    seasontype: str,
    damped: bool,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    m: int,
    bounds: str,
):
    """
    Initialize (and lightly sanitize) smoothing parameters and bounds.

    Mirrors R/NumPy-style heuristics to pick starting values when any of
    alpha/beta/gamma/phi are NaN, and adjusts bounds to respect typical
    constraints (e.g., beta <= alpha, gamma <= 1 - alpha).

    Parameters
    ----------
    alpha, beta, gamma, phi : float
        Optional user-provided starting values (use NaN to auto-init).
    trendtype, seasontype : {"N","A","M"}
        Structure flags as strings for convenience.
    damped : bool
        Whether a damped trend is considered.
    lower, upper : jnp.ndarray
        4-element arrays of lower/upper bound suggestions.
    m : int
        Seasonal period.
    bounds : {"both","usual","admissible"}
        Bound mode (admissible relaxes early to allow search to start).

    Returns
    -------
    (dict, jnp.ndarray, jnp.ndarray)
        Dict of possibly-updated {alpha,beta,gamma,phi}, and the (possibly
        clipped) lower/upper arrays actually used.
    """
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    if bounds == "admissible":
        lower = lower.at[:3].set(0.0)
        upper = upper.at[:3].set(1.0)
    elif jnp.any(lower > upper):
        raise Exception("Inconsistent parameter boundaries")

    # select alpha
    if math.isnan(alpha):
        alpha = 0.1 if m > 1 else 0.2
        if alpha > 1 or alpha < 0:
            alpha = float(lower[0] + 2e-3)
    
    # select beta
    if trendtype != "N" and math.isnan(beta):
        upper = upper.at[1].set(float(jnp.minimum(upper[1], alpha)))
        beta = float(lower[1] + 0.1 * (upper[1] - lower[1]))
        if beta < 0 or beta > alpha:
            beta = alpha - 1e-3
    
    # select gamma
    if seasontype != "N" and math.isnan(gamma):
        upper = upper.at[2].set(float(jnp.minimum(upper[2], 1 - alpha)))
        gamma = 0.05
        if gamma < 0 or gamma > 1 - alpha:
            gamma = 1 - alpha - 1e-3
    
    # select phi
    if damped and math.isnan(phi):
        phi = float(lower[3] + 0.99 * (upper[3] - lower[3]))
        if phi < 0 or phi > 1:
            phi = float(upper[3] - 1e-3)
    
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "phi": phi}, lower, upper


def _polyroots_power_basis(coeff_power_inc: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the roots of a polynomial in power basis with increasing coefficients.

    Parameters
    ----------
    coeff_power_inc : jnp.ndarray
        Coefficients [c0, c1, ..., cN] representing P(x)=c0+c1 x+...+cN x^N.

    Returns
    -------
    jnp.ndarray (complex128)
        Eigenvalues of the companion matrix (the polynomial roots).
    """
    c = jnp.asarray(coeff_power_inc, dtype=jnp.float64)
    n = c.shape[0] - 1
    if n <= 0:
        return jnp.array([], dtype=jnp.complex128)
    # Highest degree coefficient
    a_n = c[-1]
    # Handle degenerate
    if jnp.isclose(a_n, 0.0):
        # trim trailing zeros
        idx = jnp.where(jnp.abs(c[::-1]) > 0)[0]
        if idx.size == 0:
            return jnp.array([], dtype=jnp.complex128)
        k = int(idx[0])
        c = c[: c.shape[0] - k]
        n = c.shape[0] - 1
        if n <= 0:
            return jnp.array([], dtype=jnp.complex128)
        a_n = c[-1]
    # normalize to monic
    c_monic = c / a_n
    # Companion matrix for x^n + c_{n-1} x^{n-1} + ... + c0
    # Build with JAX
    C = jnp.zeros((n, n), dtype=jnp.float64)
    C = C.at[1:, :-1].set(jnp.eye(n - 1, dtype=jnp.float64))
    C = C.at[0, :].set(-c_monic[:-1][::-1])
    # eigenvalues
    evals = jnp.linalg.eigvals(C.astype(jnp.complex128))
    return evals


def admissible(alpha: float, beta: float, gamma: float, phi: float, m: int):
    """
    Check ETS smoothing parameters against standard admissibility conditions.

    Includes classical ETS constraints and a seasonal stability check via the
    characteristic polynomial’s roots (|root| <= 1).

    Returns
    -------
    bool
        True if parameter tuple passes admissibility checks.
    """
    # Mirror original admissibility tests, including characteristic-equation root check
    if math.isnan(phi):
        phi = 1.0
    if phi < 0.0 or phi > 1 + 1e-8:
        return False
    if math.isnan(gamma):
        # non-seasonal
        if alpha < 1 - 1 / phi or alpha > 1 + 1 / phi:
            return False
        if not math.isnan(beta):
            if beta < alpha * (phi - 1) or beta > (1 + phi) * (2 - alpha):
                return False
    elif m > 1:  # seasonal model
        if math.isnan(beta):
            beta = 0.0
        if gamma < max(1 - 1 / phi - alpha, 0) or gamma > 1 + 1 / phi - alpha:
            return False
        if alpha < 1 - 1 / phi - gamma * (1 - m + phi + phi * m) / (2 * phi * m):
            return False
        if beta < -(1 - phi) * (gamma / m + alpha):
            return False
        # Characteristic-equation check via companion-matrix roots (JAX)
        P = jnp.full(2 + m - 2 + 2, jnp.nan)
        P = P.at[:2].set(jnp.array([phi * (1 - alpha - gamma), alpha + beta - alpha * phi + gamma - 1.0]))
        mid_len = m - 2
        if mid_len > 0:
            P = P.at[2 : (m - 2 + 2)].set(jnp.repeat(alpha + beta - alpha * phi, mid_len))
        P = P.at[(m - 2 + 2) :].set(jnp.array([alpha + beta - phi, 1.0]))
        roots = _polyroots_power_basis(P)  # complex
        mod = jnp.abs(roots)
        max_mod = float(jnp.max(mod))
        if max_mod > 1 + 1e-10:
            return False
    return True


def check_param(
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    bounds: str,
    m: int,
):
    """
    Validate smoothing parameters against box bounds and (optionally) admissibility.

    Parameters
    ----------
    alpha, beta, gamma, phi : float
        Candidate smoothing parameters (NaN for unused, e.g., when no season).
    lower, upper : jnp.ndarray
        Elementwise lower/upper bounds (length 4).
    bounds : {"both","usual","admissible"}
        If not "admissible", enforce box bounds; if not "usual", enforce ETS admissibility.
    m : int
        Seasonal period.

    Returns
    -------
    bool
        True if parameters are within range and admissible per `bounds`.
    """
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    if bounds != "admissible":
        if not math.isnan(alpha):
            if alpha < float(lower[0]) or alpha > float(upper[0]):
                return False
        if not math.isnan(beta):
            if beta < float(lower[1]) or beta > alpha or beta > float(upper[1]):
                return False
        if not math.isnan(phi):
            if phi < float(lower[3]) or phi > float(upper[3]):
                return False
        if not math.isnan(gamma):
            if gamma < float(lower[2]) or gamma > 1 - alpha or gamma > float(upper[2]):
                return False
    if bounds != "usual":
        if not admissible(alpha, beta, gamma, phi, m):
            return False
    return True


def fourier(x, period, K, h=None):
    """
    Build a simple Fourier design matrix for seasonality.

    Parameters
    ----------
    x : array-like
        Input series (used only for length alignment).
    period : list[int]
        Seasonal periods to include (e.g., [m]).
    K : list[int]
        Number of harmonics per period.
    h : Optional[int]
        If provided, build the matrix for the *future* h steps; else fit window.

    Returns
    -------
    jnp.ndarray
        Matrix with sin/cos columns for selected harmonics, with degenerate
        sinpi=0 columns removed.
    """
    n = len(x)
    if h is None:
        times = jnp.arange(1, n + 1, dtype=jnp.float64)
    else:
        times = jnp.arange(n + 1, n + h + 1, dtype=jnp.float64)

    if len(period) == 0:
        return jnp.zeros((times.shape[0], 0), dtype=jnp.float64)

    period_arr = jnp.asarray(period, dtype=jnp.float64)
    k_arr = jnp.asarray(K, dtype=jnp.int32)
    max_k = int(jnp.max(k_arr)) if k_arr.size > 0 else 0
    if max_k <= 0:
        return jnp.zeros((times.shape[0], 0), dtype=jnp.float64)

    ks = jnp.arange(1, max_k + 1, dtype=jnp.float64)
    vals = ks[None, :] / period_arr[:, None]
    mask = ks[None, :] <= k_arr[:, None]
    vals = jnp.where(mask, vals, jnp.nan)
    p = jnp.unique(vals[~jnp.isnan(vals)])
    k = jnp.abs(2 * p - jnp.round(2 * p)) > _smalno

    angles = 2 * jnp.pi * times[:, None] * p[None, :]
    sin_cols = jnp.sin(angles)
    cos_cols = jnp.cos(angles)
    X = jnp.stack([sin_cols, cos_cols], axis=2).reshape(times.shape[0], -1)

    mask = jnp.stack([k, jnp.ones_like(k, dtype=bool)], axis=1).reshape(-1)
    return X[:, mask]


@partial(jax.jit, static_argnames=("m", "multiplicative"))
def _seasonal_decompose_jax(
    y: jnp.ndarray,
    m: int,
    multiplicative: bool,
) -> Dict[str, jnp.ndarray]:
    """Lightweight JAX-native seasonal decomposition for initialization."""
    y = jnp.asarray(y, dtype=jnp.float64)
    n = y.shape[0]
    m_eff = max(int(m), 1)

    # Centered moving-average trend proxy
    w = jnp.ones((m_eff,), dtype=jnp.float64) / float(m_eff)
    trend = jnp.convolve(y, w, mode="same")
    if (m_eff % 2) == 0:
        trend = jnp.convolve(trend, jnp.array([0.5, 0.5], dtype=jnp.float64), mode="same")

    if multiplicative:
        trend_safe = jnp.where(jnp.abs(trend) > 1e-8, trend, 1e-8)
        detrended = y / trend_safe
    else:
        detrended = y - trend

    n_periods = n // m_eff
    n_full = n_periods * m_eff

    def _seasonal_from_matrix() -> jnp.ndarray:
        mat = detrended[:n_full].reshape((n_periods, m_eff))
        pat = jnp.mean(mat, axis=0)
        rep = jnp.tile(pat, n // m_eff + 1)[:n]
        if multiplicative:
            return rep / jnp.where(jnp.abs(jnp.mean(rep)) > 1e-8, jnp.mean(rep), 1.0)
        return rep - jnp.mean(rep)

    seasonal = lax.cond(
        n_periods >= 1,
        lambda _: _seasonal_from_matrix(),
        lambda _: jnp.ones((n,), dtype=jnp.float64) if multiplicative else jnp.zeros((n,), dtype=jnp.float64),
        operand=None,
    )
    return {"seasonal": seasonal, "trend": trend}


def initstate(y, m, trendtype, seasontype):
    """
    Initialize ETS states (level [+ trend] [+ seasonal]) from data.

    Strategy
    --------
    - If seasonal and `len(y) < 3m`: fit a small Fourier regression to extract
      rough seasonality; else use `seasonal_decompose` (statsmodels).
    - Deseasonalize (if needed), then fit OLS for intercept/slope over the
      first `max(10, 2m)` points to seed level/trend, with multiplicative
      reparametrization safeguards.

    Returns
    -------
    jnp.ndarray
        Concatenated initial state vector.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = y.shape[0]
    if seasontype != "N":
        if n < 4:
            raise ValueError("You've got to be joking (not enough data).")
        elif n < 3 * m:  # small-n fallback: simple seasonal means
            idx = jnp.arange(n, dtype=jnp.int32) % int(m)
            sums = jnp.zeros((m,), dtype=jnp.float64).at[idx].add(y)
            counts = jnp.zeros((m,), dtype=jnp.float64).at[idx].add(1.0)
            pattern = sums / jnp.maximum(counts, 1.0)
            seasonal = jnp.take(pattern, idx)
            if seasontype == "A":
                seasonal = seasonal - jnp.mean(seasonal)
                y_d = {"seasonal": seasonal}
            else:
                if not float(jnp.min(y)) > 0:
                    raise Exception(
                        "Multiplicative seasonality is not appropriate for zero and negative values"
                    )
                seasonal = seasonal / jnp.maximum(jnp.mean(seasonal), 1e-8)
                y_d = {"seasonal": seasonal}
        else:
            y_d = _seasonal_decompose_jax(y, m, seasontype == "M")
        init_seas = y_d["seasonal"][1:m][::-1]
        if n < 5 * m:
            if n < 2 * m:
                shrinkage = 0.5
            else:
                shrinkage = 0.5 + 0.5 * (n / (5.0 * m))
            if seasontype == "A":
                init_seas = init_seas * shrinkage
            else:
                init_seas = 1.0 + (init_seas - 1.0) * shrinkage
        if seasontype == "A":
            y_sa = y - y_d["seasonal"]
        else:
            init_seas = jnp.clip(init_seas, a_min=1e-2)
            if float(jnp.sum(init_seas)) > m:
                init_seas = init_seas / jnp.sum(init_seas + 1e-2)
            y_sa = y / jnp.clip(y_d["seasonal"], a_min=1e-2)
    else:
        m = 1
        init_seas = jnp.array([], dtype=jnp.float64)
        y_sa = y

    maxn = min(max(10, 2 * m), int(y_sa.shape[0]))
    if trendtype == "N":
        l0 = float(jnp.mean(y_sa[:maxn]))
        return jnp.concatenate([jnp.array([l0]), init_seas])
    else:
        X = jnp.full((n, 2), jnp.nan)
        X = X.at[:, 0].set(1.0)
        X = X.at[:, 1].set(jnp.arange(1, n + 1, dtype=jnp.float64))
        (l, b), *_ = jnp.linalg.lstsq(X[:maxn], y_sa[:maxn], rcond=-1.0)
        l = float(l); b = float(b)
        if trendtype == "A":
            l0 = l
            b0 = b
            if abs(l0 + b0) < 1e-8:
                l0 = l0 * (1 + 1e-3)
                b0 = b0 * (1 - 1e-3)
        else:
            l0 = l + b
            if abs(l0) < 1e-8:
                l0 = 1e-7
            b0 = (l + 2 * b) / l0
            div = b0 if not math.isclose(b0, 0.0, abs_tol=1e-8) else 1e-8
            l0 = l0 / div
            if abs(b0) > 1e10:
                b0 = math.copysign(1e10, b0)
            if l0 < 1e-8 or b0 < 1e-8:
                l0 = max(float(y_sa[0]), 1e-3)
                div2 = float(y_sa[0]) if not math.isclose(float(y_sa[0]), 0.0, abs_tol=1e-8) else 1e-8
                b0 = max(float(y_sa[1]) / div2, 1e-3)
        return jnp.concatenate([jnp.array([l0, b0]), init_seas])


def switch(x: str) -> _ets.Component:
    """
    Map string flags {"N","A","M"} to `_ets.Component` enum.
    """
    if x == "N":
        return _ets.Component.Nothing
    if x == "A":
        return _ets.Component.Additive
    if x == "M":
        return _ets.Component.Multiplicative
    raise ValueError(f"Unknown component {x}")


def switch_criterion(x: str) -> _ets.Criterion:
    """
    Map objective string to `_ets.Criterion` enum.

    Accepted values: {"lik","mse","amse","sigma","mae"}.
    """
    if x == "lik":
        return _ets.Criterion.Likelihood
    if x == "mse":
        return _ets.Criterion.MSE
    if x == "amse":
        return _ets.Criterion.AMSE
    if x == "sigma":
        return _ets.Criterion.Sigma
    if x == "mae":
        return _ets.Criterion.MAE
    raise ValueError(f"Unknown crtierion {x}")


def pegelsresid_C(
    y: jnp.ndarray,
    m: int,
    init_state: jnp.ndarray,
    errortype: str,
    trendtype: str,
    seasontype: str,
    damped: bool,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    nmse: int,
):
    """
    Roll out ETS residuals, AMSE, and likelihood from an initialized state.

    This is the high-level wrapper around `_ets.calc_full` that:
    - prepares buffers,
    - enforces structural simplifications (e.g., set beta/gamma/phi when absent),
    - reshapes the packed state history,
    - returns `(amse, residuals, states, lik)`.

    Returns
    -------
    (jnp.ndarray, jnp.ndarray, jnp.ndarray, float)
        AMSE vector (length nmse), residuals (length n), state snapshots
        ((n+1) x n_state), and the likelihood-style scalar.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    p = int(init_state.shape[0])
    n = int(y.shape[0])

    x = jnp.full((p * (n + 1),), jnp.nan)
    x = x.at[:p].set(jnp.asarray(init_state, dtype=jnp.float64))
    e = jnp.full_like(y, jnp.nan)

    if not damped:
        phi = 1.0
    if trendtype == "N":
        beta = 0.0
    if seasontype == "N":
        gamma = 0.0
    amse = jnp.full((nmse,), jnp.nan)

    amse, e, x, lik = _ets.calc_full(
        x,
        e,
        amse,
        nmse,
        y,
        switch(errortype),
        switch(trendtype),
        switch(seasontype),
        float(alpha),
        float(beta),
        float(gamma),
        float(phi),
        int(m),
    )
    xs = x.reshape((n + 1, p))
    if not jnp.isnan(lik):
        if float(jnp.abs(lik + 99999)) < _ets.TOL:
            lik = jnp.nan
    return amse, e, xs, float(lik)


def optimize_ets_target_fn(
    x0,
    par,
    y,
    init_state,
    errortype,
    trendtype,
    seasontype,
    damped,
    par_noopt,
    lowerb,
    upperb,
    opt_crit,
    nmse,
    bounds,
    m,
    pnames,
    pnames2,
    pad_to=None,
    bucket_size=None,
    maxit=1_000,
    optax_steps=300,
    optax_lr=1e-2,
    optax_clip=1.0,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-6,
    adaptive_tol: bool = True,
    is_final_model: bool = False,
    clip_multiplicative_errors: bool = True,
    pure_sigmoid: bool = False,
    opt_init_state: bool = False,
    init_state_opt: jnp.ndarray | None = None,
):
    """
    Build and solve the ETS optimization problem via `_ets.optimize_bfgs_smoothing`.

    Parameters
    ----------
    x0 : array
        Initial vector of free smoothing parameters.
    par : dict
        Only the optimizable parameters (non-NaN) from initparam.
    y : array
        Observations.
    init_state : array
        Fixed initial state vector.
    errortype, trendtype, seasontype : str
        Structure flags ("A","M","N").
    damped : bool
        Whether trend is damped.
    par_noopt : dict
        Original user-provided params (to freeze any non-NaN).
    lowerb, upperb : array
        Box bounds for the *free* prefix of x0 (smoothing params + states).
    opt_crit : str
        Objective ("lik","mse","amse","sigma","mae").
    nmse : int
        Horizon for AMSE.
    bounds : str
        Bound mode ("both","usual","admissible").
    m : int
        Seasonal period.
    pnames, pnames2 : iterable
        Keys for printing/debug purposes.

    Returns
    -------
    results(...)
        A small namedtuple-like object compatible with your test harness.
    """
    par_alpha = par.get("alpha", jnp.nan)
    alpha = par_noopt["alpha"] if math.isnan(par_alpha) else par_alpha
    if math.isnan(alpha):
        raise ValueError("alpha problem!")
    if trendtype != "N":
        par_beta = par.get("beta", jnp.nan)
        beta = par_noopt["beta"] if math.isnan(par_beta) else par_beta
        if math.isnan(beta):
            raise ValueError("beta problem!")
    else:
        beta = jnp.nan
    if seasontype != "N":
        par_gamma = par.get("gamma", jnp.nan)
        gamma = par_noopt["gamma"] if math.isnan(par_gamma) else par_gamma
        if math.isnan(gamma):
            raise ValueError("gamma problem!")
    else:
        m = 1
        gamma = jnp.nan
    if damped:
        par_phi = par.get("phi", jnp.nan)
        phi = par_noopt["phi"] if math.isnan(par_phi) else par_phi
        if math.isnan(phi):
            raise ValueError("phi problem!")
    else:
        phi = jnp.nan

    optAlpha = not math.isnan(float(alpha))
    optBeta = not math.isnan(float(beta))
    optGamma = not math.isnan(float(gamma))
    optPhi = not math.isnan(float(phi))

    if not math.isnan(par_noopt["alpha"]):
        optAlpha = False
    if not math.isnan(par_noopt["beta"]):
        optBeta = False
    if not math.isnan(par_noopt["gamma"]):
        optGamma = False
    if not math.isnan(par_noopt["phi"]):
        optPhi = False

    if not damped:
        phi = 1.0
    if trendtype == "N":
        beta = 0.0
    if seasontype == "N":
        gamma = 0.0

    def _pad_for_jit(y_arr, pad_to_val, bucket_size_val):
        n_obs = int(y_arr.shape[0])
        if pad_to_val is None and bucket_size_val is None:
            return y_arr, n_obs
        if pad_to_val is None:
            if bucket_size_val is None or bucket_size_val <= 0:
                return y_arr, n_obs
            pad_to_val = int(math.ceil(n_obs / bucket_size_val) * bucket_size_val)
        if pad_to_val <= n_obs:
            return y_arr, n_obs
        pad_len = pad_to_val - n_obs
        pad_val = y_arr[-1]
        pad = jnp.full((pad_len,), pad_val, dtype=y_arr.dtype)
        return jnp.concatenate([y_arr, pad]), n_obs

    y_opt, n_obs = _pad_for_jit(jnp.asarray(y, dtype=jnp.float64), pad_to, bucket_size)
    opt_steps = int(maxit) if optax_steps is None else int(optax_steps)
    if opt_init_state:
        if init_state_opt is None:
            raise ValueError("init_state_opt required when opt_init_state=True")
        x0_full = jnp.concatenate([jnp.asarray(x0, dtype=jnp.float64), jnp.asarray(init_state_opt, dtype=jnp.float64)])
        n_state = int(jnp.asarray(init_state_opt).shape[0])
    else:
        x0_full = jnp.asarray(x0, dtype=jnp.float64)
        n_state = 0
    opt_res = _ets.optimize_bfgs_smoothing(
        x0_full,
        y_opt,
        jnp.asarray(init_state, dtype=jnp.float64),
        switch(errortype),
        switch(trendtype),
        switch(seasontype),
        switch_criterion(opt_crit),
        int(nmse),
        int(m),
        n_obs,
        bool(optAlpha),
        bool(optBeta),
        bool(optGamma),
        bool(optPhi),
        float(alpha),
        float(beta),
        float(gamma),
        float(phi),
        jnp.asarray(lowerb, dtype=jnp.float64),
        jnp.asarray(upperb, dtype=jnp.float64),
        opt_steps,
        float(optax_lr),
        float(optax_clip),
        early_stop_patience,
        early_stop_min_delta,
        adaptive_tol,
        is_final_model,
        clip_multiplicative_errors,
        pure_sigmoid,
        opt_init_state,
        n_state,
    )
    return results(
        x=jnp.asarray(opt_res.x),
        fn=jnp.asarray(opt_res.fun),
        nit=int(opt_res.nit),
        simplex=jnp.empty((0,), dtype=jnp.asarray(opt_res.x).dtype),
    )


def etsmodel(
    y: jnp.ndarray,
    m: int,
    errortype: str,
    trendtype: str,
    seasontype: str,
    damped: bool,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    opt_crit: str,
    nmse: int,
    bounds: str,
    maxit: int = 2_000,
    optax_steps: int | None = 300,
    optax_lr: float = 1e-2,
    optax_clip: float = 1.0,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-6,
    adaptive_tol: bool = True,
    is_final_model: bool = False,
    control=None,
    seed=None,
    trace: bool = False,
    pad_to=None,
    bucket_size=None,
    stabilize: bool = True,
    pure_sigmoid: bool = False,
    selection_mode: bool = False,
    init_state_override: jnp.ndarray | None = None,
):
    """
    Fit a *single* ETS structure to the data and return a stats-like result dict.

    This function:
    1) initializes or accepts smoothing params via `initparam`,
    2) checks ranges/admissibility via `check_param`,
    3) builds initial states via `initstate`,
    4) optimizes smoothing+state vector with `_ets.optimize`,
    5) re-runs the rollout to compute likelihood, residuals, fitted values,
    6) computes information criteria (AIC, BIC, AICc), sigma2, and aggregates
       outputs in a dictionary.

    Returns
    -------
    dict
        Modeled fields include:
        - "loglik", "aic", "bic", "aicc", "mse", "amse", "sigma2"
        - "fit" (opt result), "residuals", "fitted"
        - "components" (e.g., "AAd" flags), "m", "nstate", "states"
        - "par" (alpha,beta,gamma,phi + initial states), "n_params"
    """
    if seasontype == "N":
        m = 1

    par_, lower, upper = initparam(
        alpha, beta, gamma, phi, trendtype, seasontype, damped,
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(upper, dtype=jnp.float64),
        m, bounds
    )
    par_noopt = dict(alpha=alpha, beta=beta, gamma=gamma, phi=phi)

    alpha = par_["alpha"] if not math.isnan(par_["alpha"]) else alpha
    beta = par_["beta"] if not math.isnan(par_["beta"]) else beta
    gamma = par_["gamma"] if not math.isnan(par_["gamma"]) else gamma
    phi = par_["phi"] if not math.isnan(par_["phi"]) else phi

    if not check_param(alpha, beta, gamma, phi,
                       jnp.asarray(lower, dtype=jnp.float64),
                       jnp.asarray(upper, dtype=jnp.float64),
                       bounds, m):
        raise Exception("Parameters out of range")

    # initialize state (closed-form)
    if init_state_override is None:
        init_state_full = _template_cache.get_init_state(y, m, trendtype, seasontype)
    else:
        init_state_full = jnp.asarray(init_state_override, dtype=jnp.float64)
    nstate = int(init_state_full.shape[0])
    opt_init_state = bool(pure_sigmoid and (not stabilize))

    opt_names = [
        name
        for name in ["alpha", "beta", "gamma", "phi"]
        if (not math.isnan(par_[name])) and math.isnan(par_noopt[name])
    ]
    par_clean = {name: par_[name] for name in opt_names}
    par_vec = jnp.full((len(opt_names),), jnp.nan, dtype=jnp.float64)
    if len(opt_names) > 0:
        par_vec = par_vec.at[: len(opt_names)].set(jnp.asarray([par_[name] for name in opt_names], dtype=jnp.float64))

    lower_box = jnp.asarray(lower, dtype=jnp.float64)
    upper_box = jnp.asarray(upper, dtype=jnp.float64)

    np_ = int(par_vec.shape[0])
    n_state_opt = int(init_state_full.shape[0]) if opt_init_state else 0
    np_eff = np_ + n_state_opt
    if np_eff >= len(y) - 1:
        return dict(
            aic=jnp.inf,
            bic=jnp.inf,
            aicc=jnp.inf,
            mse=jnp.inf,
            amse=jnp.inf,
            fit=None,
            par=par_vec,
            states=init_state_full,
        )
    if np_ > 0:
        x0_unconstrained = []
        for name in opt_names:
            if name == "alpha":
                if pure_sigmoid:
                    val = float(jnp.clip(par_[name], 1e-6, 1.0 - 1e-6))
                    x0_unconstrained.append(float(jnp.log(val / (1.0 - val))))
                else:
                    x0_unconstrained.append(_inv_sigmoid_scaled(par_[name], float(lower[0]), float(upper[0])))
            elif name == "beta":
                if pure_sigmoid:
                    val = float(jnp.clip(par_[name], 1e-6, 1.0 - 1e-6))
                    x0_unconstrained.append(float(jnp.log(val / (1.0 - val))))
                else:
                    beta_cap = float(jnp.minimum(upper[1], par_["alpha"]))
                    x0_unconstrained.append(_inv_sigmoid_scaled(par_[name], float(lower[1]), beta_cap))
            elif name == "gamma":
                if pure_sigmoid:
                    val = float(jnp.clip(par_[name], 1e-6, 1.0 - 1e-6))
                    x0_unconstrained.append(float(jnp.log(val / (1.0 - val))))
                else:
                    gamma_cap = float(jnp.minimum(upper[2], 1.0 - par_["alpha"]))
                    x0_unconstrained.append(_inv_sigmoid_scaled(par_[name], float(lower[2]), gamma_cap))
            elif name == "phi":
                x0_unconstrained.append(_inv_sigmoid_scaled(par_[name], float(lower[3]), float(upper[3])))
        par_vec = jnp.asarray(x0_unconstrained, dtype=jnp.float64)
        fred = optimize_ets_target_fn(
            x0=par_vec,
            par=par_clean,
            y=jnp.asarray(y, dtype=jnp.float64),
            init_state=init_state_full,
            errortype=errortype,
            trendtype=trendtype,
            seasontype=seasontype,
            damped=damped,
            par_noopt=par_noopt,
            lowerb=lower_box,
            upperb=upper_box,
            opt_crit=opt_crit,
            nmse=nmse,
            bounds=bounds,
            m=m,
            pnames=par_clean.keys(),
            pnames2=par_noopt.keys(),
            pad_to=pad_to,
            bucket_size=bucket_size,
            maxit=maxit,
            optax_steps=optax_steps,
            optax_lr=optax_lr,
            optax_clip=optax_clip,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            adaptive_tol=adaptive_tol,
            is_final_model=is_final_model,
            clip_multiplicative_errors=stabilize,
            pure_sigmoid=pure_sigmoid,
            opt_init_state=opt_init_state,
            init_state_opt=init_state_full if opt_init_state else None,
        )
        fit_par = jnp.asarray(fred.x)
    else:
        fred = results(x=jnp.asarray(par_vec), fn=jnp.nan, nit=0, simplex=jnp.empty((0,), dtype=jnp.float64))
        fit_par = jnp.asarray(par_vec)

    init_state_fit = init_state_full
    nstate = int(init_state_fit.shape[0])

    if np_ > 0:
        if opt_init_state and n_state_opt > 0:
            fit_par_smooth = fit_par[:-n_state_opt]
            init_state_fit = fit_par[-n_state_opt:]
        else:
            fit_par_smooth = fit_par
        alpha, beta, gamma, phi = _transform_smoothing_params_unconstrained(
            fit_par_smooth, opt_names, alpha, beta, gamma, phi, lower, upper, pure_sigmoid
        )
        if not check_param(alpha, beta, gamma, phi,
                           jnp.asarray(lower, dtype=jnp.float64),
                           jnp.asarray(upper, dtype=jnp.float64),
                           bounds, m):
            return dict(
                aic=jnp.inf,
                bic=jnp.inf,
                aicc=jnp.inf,
                mse=jnp.inf,
                amse=jnp.inf,
                fit=None,
                par=fit_par,
                states=init_state_fit,
            )

    if selection_mode:
        e_tmp = jnp.zeros_like(jnp.asarray(y, dtype=jnp.float64))
        amse_tmp = jnp.zeros((nmse,), dtype=jnp.float64)
        n_obs = int(jnp.asarray(y).shape[0])
        e, amse, lik = _ets._calc_roll_nohist(
            init_state_fit,
            e_tmp,
            amse_tmp,
            nmse,
            jnp.asarray(y, dtype=jnp.float64),
            jnp.asarray(n_obs, dtype=jnp.int32),
            switch(errortype),
            switch(trendtype),
            switch(seasontype),
            float(alpha),
            float(beta),
            float(gamma),
            float(phi),
            int(m),
            clip_multiplicative_errors=stabilize,
            fcst_h=1 if opt_crit == "lik" else nmse,
        )
        states = None
    else:
        amse, e, states, lik = pegelsresid_C(
            y=jnp.asarray(y, dtype=jnp.float64),
            m=m,
            init_state=init_state_fit,
            errortype=errortype,
            trendtype=trendtype,
            seasontype=seasontype,
            damped=damped,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi=phi,
            nmse=nmse,
        )
    lik_opt = None
    if opt_crit == "lik" and pure_sigmoid and (not stabilize):
        try:
            lik_opt = float(fred.fn)
        except Exception:
            lik_opt = None
    k = np_ + n_state_opt + 1
    ny = len(y)
    lik_ic = float(lik_opt) if lik_opt is not None else float(lik)
    aic, bic, aicc = _compute_ic(lik_ic, ny, k)

    mse = float(amse[0])
    amse_mean = float(jnp.mean(amse))

    fit_par_full = jnp.concatenate([jnp.array([alpha, beta, gamma, phi], dtype=jnp.float64), init_state_fit])
    if selection_mode:
        return dict(
            loglik=-0.5 * float(lik),
            aic=aic,
            bic=bic,
            aicc=float(aicc),
            mse=mse,
            amse=amse_mean,
            fit=fred,
            residuals=None,
            components=f"{errortype}{trendtype}{seasontype}{'D' if damped else 'N'}",
            m=m,
            nstate=nstate,
            fitted=None,
            states=None,
            par=fit_par_full,
            sigma2=jnp.nan,
            n_params=k,
        )

    if errortype == "A":
        fits = jnp.asarray(y) - e
    else:
        # protect e == -1
        aux_e = jnp.where(e == -1.0, -1 + 1e-3, e)
        fits = jnp.asarray(y) / (1.0 + aux_e)

    sq_e = e * e
    finites = ~jnp.isinf(sq_e)
    sigma2 = float(jnp.sum(sq_e[finites]) / (ny - k - 1))

    return dict(
        loglik=-0.5 * float(lik),
        aic=aic,
        bic=bic,
        aicc=float(aicc),
        mse=mse,
        amse=amse_mean,
        fit=fred,
        residuals=e,
        components=f"{errortype}{trendtype}{seasontype}{'D' if damped else 'N'}",
        m=m,
        nstate=nstate,
        fitted=fits,
        states=states,
        par=fit_par_full,
        sigma2=sigma2,
        n_params=k,
    )


def is_constant(x):
    """
    Quick check for a constant series.

    Returns
    -------
    bool
        True if all entries equal the first entry.
    """
    x = jnp.asarray(x)
    return bool(jnp.all(x[0] == x))


def ets_f(
    y,
    m,
    model="ZZZ",
    damped=None,
    alpha=None,
    beta=None,
    gamma=None,
    phi=None,
    additive_only=None,
    blambda=None,
    biasadj=None,
    lower=None,
    upper=None,
    opt_crit="lik",
    nmse=3,
    bounds="both",
    ic="aicc",
    restrict=True,
    allow_multiplicative_trend=False,
    use_initial_values=False,
    maxit=2_000,
    optax_steps=None,
    optax_lr=1e-2,
    optax_clip=1.0,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-6,
    allow_extended_iterations: bool = False,
    adaptive_tol: bool = True,
    pad_to=None,
    bucket_size=None,
):
    """
    Top-level ETS interface: model selection + fitting.

    Parameters
    ----------
    y : array-like
        Time series (float64).
    m : int
        Seasonal period.
    model : str or dict
        - If str like "ZZZ", expands to grids:
          error in {"A","M"}, trend in {"N","A"[,"M" if allowed]}, season in {"N","A","M"},
          damped in {True, False} if None.
        - If dict (a previously-fitted object), forecast forward using stored params.
    damped : Optional[bool]
        If None, both damped and undamped are tried for trend != "N".
    alpha, beta, gamma, phi : Optional[float]
        Provide to fix values (set the others to NaN to optimize).
    additive_only : Optional[bool]
        If True, forbids multiplicative forms.
    blambda, biasadj : unused
        Not implemented here (Box-Cox/bias adjustment).
    lower, upper : Optional[array]
        Bounds for (alpha, beta, gamma, phi); defaults applied if None.
    opt_crit : {"lik","mse","amse","sigma","mae"}
        Optimization criterion.
    nmse : int
        Horizon for AMSE tracking (1..30).
    bounds : {"both","usual","admissible"}
        Bound/admissibility behavior.
    ic : {"aicc","aic","bic"}
        Information criterion used to pick the best among candidates.
    restrict : bool
        Apply standard ETS combination restrictions (forbid certain mixes).
    allow_multiplicative_trend : bool
        If True and model="ZZZ", includes "M" trend in the grid.
    use_initial_values : bool
        Reserved; not used here.
    maxit : int
        Maximum iterations budget (passed to optimizer).

    Returns
    -------
    dict
        Best fitted model dictionary (see `etsmodel` return schema) with an
        added "method" key like "ETS(A,Ad,M)".

    Notes
    -----
    - When `model` is a dict, this function acts as a *forward* function (no re-fit).
    - When `y` is constant, it falls back to ANN with alpha≈1 to mimic
      standard implementations.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = len(y)
    season_forced = m > 1 and n < 2 * m
    if season_forced:
        m = 1
    auto_model = isinstance(model, str) and ("Z" in model)
    # Parse model string early so errortype/trendtype/seasontype are
    # available for the optax_steps heuristic below.
    # When model is a dict (forward_ets path), these stay None and the
    # isinstance check in the optax_steps block will pick the default branch.
    errortype = trendtype = seasontype = None
    if isinstance(model, str):
        errortype, trendtype, seasontype = model
        if season_forced:
            seasontype = "N"
    if auto_model and m > 1 and (n / max(m, 1)) < 2:
        m_infer = _infer_season_length(y, max_m=min(24, n // 2))
        if m_infer != m and m_infer > 1:
            m = m_infer
    if optax_steps is None:
        if isinstance(errortype, str) and isinstance(trendtype, str) and isinstance(seasontype, str):
            optax_steps = _estimate_ets_iterations(
                y, m, errortype, trendtype, seasontype, allow_extended=allow_extended_iterations
            )
        else:
            # Optimized: Reduced default iterations for faster training
            if n < 100:
                optax_steps = 50
            elif n < 200:
                optax_steps = 75
            elif n < 500:
                optax_steps = 100
            elif n < 2000:
                optax_steps = 120
            else:
                optax_steps = 100

    if alpha is None:
        alpha = jnp.nan
    if beta is None:
        beta = jnp.nan
    if gamma is None:
        gamma = jnp.nan
    if phi is None:
        phi = jnp.nan
    if blambda is not None:
        raise NotImplementedError("`blambda` not None")
    if nmse < 1 or nmse > 30:
        raise ValueError("nmse out of range")
    if bounds == "both":
        bounds = "admissible"
    if lower is None:
        lower = jnp.array([0.0001, 0.0001, 0.0001, _PHI_LOWER], dtype=jnp.float64)
    else:
        lower = jnp.asarray(lower, dtype=jnp.float64)
    if upper is None:
        upper = jnp.array([0.9999, 0.9999, 0.9999, _PHI_UPPER], dtype=jnp.float64)
    else:
        upper = jnp.asarray(upper, dtype=jnp.float64)
    if jnp.any(upper < lower):
        raise ValueError("Lower limits must be less than upper limits")

    timing_enabled = os.environ.get("CHRONAX_ETS_TIMING", "0") == "1"
    t_start = time.perf_counter() if timing_enabled else 0.0

    if is_constant(y):
        return etsmodel(
            y=y,
            m=m,
            errortype="A",
            trendtype="N",
            seasontype="N",
            alpha=0.9999,
            beta=beta,
            gamma=gamma,
            phi=phi,
            damped=False,
            lower=lower,
            upper=upper,
            opt_crit=opt_crit,
            nmse=nmse,
            bounds=bounds,
            maxit=maxit,
            optax_steps=optax_steps,
            optax_lr=optax_lr,
            optax_clip=optax_clip,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            adaptive_tol=adaptive_tol,
            is_final_model=True,
            pad_to=pad_to,
            bucket_size=bucket_size,
        )

    if isinstance(model, dict):
        m = model["m"]
        errortype, trendtype, seasontype = model["components"][:3]
        damped = model["components"][3] != "N"
        alpha, beta, gamma, phi = model["par"][:4]
        init_state = model["par"][4:]
        amse, e, states, lik = pegelsresid_C(
            y=y,
            m=m,
            init_state=init_state,
            errortype=errortype,
            trendtype=trendtype,
            seasontype=seasontype,
            damped=damped,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi=phi,
            nmse=nmse,
        )
        fred = model["fit"]
        nstate = len(init_state)
        np_ = model["n_params"] - 1
        np_ = np_ + 1
        ny = len(y)
        aic = float(lik) + 2 * np_
        bic = float(lik) + math.log(ny) * np_
        aicc = _aicc(aic=aic, n=ny, k=np_)

        mse = float(amse[0])
        amse_mean = float(jnp.mean(amse))

        fit_par = jnp.concatenate([jnp.array([alpha, beta, gamma, phi], dtype=jnp.float64), init_state])
        if errortype == "A":
            fits = y - e
        else:
            aux_e = jnp.where(e == -1.0, -1 + 1e-3, e)
            fits = y / (1 + aux_e)
        sq_e = e * e
        finites = ~jnp.isinf(sq_e)
        sigma2 = float(jnp.sum(sq_e[finites]) / (ny - np_ - 1))

        return dict(
            loglik=-0.5 * float(lik),
            aic=aic,
            bic=bic,
            aicc=float(aicc),
            mse=mse,
            amse=amse_mean,
            fit=fred,
            residuals=e,
            components=f"{errortype}{trendtype}{seasontype}{'D' if damped else 'N'}",
            m=m,
            nstate=nstate,
            fitted=fits,
            states=states,
            par=fit_par,
            sigma2=sigma2,
            n_params=np_,
        )

    # errortype, trendtype, seasontype already parsed above (early parse block)
    if errortype not in ["M", "A", "Z"]:
        raise ValueError("Invalid error type")
    if trendtype not in ["N", "A", "M", "Z"]:
        raise ValueError("Invalid trend type")
    if seasontype not in ["N", "A", "M", "Z"]:
        raise ValueError("Invalid season type")
    if m < 1 or len(y) <= m:
        seasontype = "N"
    if m == 1:
        if seasontype in ("A", "M"):
            raise ValueError("Nonseasonal data")
        else:
            seasontype = "N"
    if restrict:
        if (
            (errortype == "A" and (trendtype == "M" or seasontype == "M"))
            or (errortype == "M" and trendtype == "M" and seasontype == "A")
            or (
                additive_only
                and (errortype == "M" or trendtype == "M" or seasontype == "M")
            )
        ):
            raise ValueError("Forbidden model combination")
    data_positive = float(jnp.min(y)) > 0
    if (not data_positive) and errortype == "M":
        raise ValueError("Inappropriate model for data with negative or zero values")
    if damped is not None:
        if damped and trendtype == "N":
            ValueError("Forbidden model combination")
    n = len(y)
    npars = 2
    if trendtype in ["A", "M"]:
        npars += 2
    if seasontype in ["A", "M"]:
        npars += 2
    if damped is not None:
        npars += int(bool(damped))
    if n <= npars + 4:
        raise NotImplementedError("tiny datasets")

    cycles = n / max(m, 1)
    unify = os.environ.get("CHRONAX_ETS_UNIFIED", "1") == "1"
    if cycles < 10 or n < 100:
        unify = False
    if unify:
        err_u, trend_u, seas_u = _choose_unified_components(
            y, m, allow_multiplicative_trend
        )
        if errortype == "Z":
            errortype = [err_u]
        if trendtype == "Z":
            trendtype = [trend_u]
        if seasontype == "Z":
            seasontype = [seas_u]
    elif auto_model and (errortype == "Z" or trendtype == "Z" or seasontype == "Z"):
        # Reduce candidate grid for small samples to limit JIT churn.
        err_u, trend_u, seas_u = _choose_unified_components(
            y, m, allow_multiplicative_trend
        )
        if errortype == "Z":
            errortype = [err_u]
        if trendtype == "Z":
            trendtype = ["N"] if n < 50 else [trend_u]
        if seasontype == "Z":
            if m <= 1 or cycles < 6:
                seasontype = ["N"]
            elif cycles < 10 and seas_u == "M":
                seasontype = ["A", "N"]
            else:
                seasontype = [seas_u]
    if errortype == "Z":
        errortype = ["A", "M"]
    if trendtype == "Z":
        trendtype = ["N", "A"] + (["M"] if allow_multiplicative_trend else [])
    if seasontype == "Z":
        seasontype = ["N", "A", "M"]
    prefer_damped_small_n = auto_model and n < 200
    force_additive_error = auto_model and n < 80
    if damped is None:
        damped = [True, False]
    else:
        damped = [damped]

    best_ic = jnp.inf
    best = None
    selection_stabilize = not auto_model
    nmse_sel = 1 if opt_crit == "lik" else nmse
    init_state_cache: dict[tuple[str, str], jnp.ndarray] = {}
    no_improve = 0
    # Optimized: More aggressive early stopping in model selection
    max_no_improve = int(os.environ.get("CHRONAX_ETS_NO_IMPROVE_STOP", "1"))
    min_evals = int(os.environ.get("CHRONAX_ETS_MIN_EVALS", "1"))  # Reduced from 2
    evals = 0
    stop_search = False
    candidates = []
    def _maybe_add_candidate(etype, ttype, stype, dtype):
        if force_additive_error and etype == "M":
            return
        if restrict:
            if etype == "M" and ttype == "M" and stype == "A":
                return
            if additive_only and (etype == "M" or ttype == "M" or stype == "M"):
                return
            # Exclude ETS(A,N,M) only (no trend + multiplicative season)
            if etype == "A" and stype == "M" and ttype == "N":
                return
            if (not data_positive) and etype == "M":
                return
            if (not data_positive) and stype == "M":
                return
        if stype != "N" and m == 1:
            return
        if etype == "M" and cycles < 10:
            return
        if stype == "M" and cycles < 10:
            return
        if ttype == "N" and dtype:
            return
        if prefer_damped_small_n and ttype != "N" and (not dtype):
            return
        candidates.append((etype, ttype, stype, dtype))

    if auto_model:
        ts_strength, ss_strength = _get_cached_strengths(y, m)
        candidates.append(("A", "N", "N", False))

        if ts_strength > 0.1:
            _maybe_add_candidate("A", "A", "N", False)
            if n > 50:
                _maybe_add_candidate("A", "A", "N", True)

        if ss_strength > 0.1 and m > 1:
            _maybe_add_candidate("A", "N", "A", False)
            if ts_strength > 0.1:
                _maybe_add_candidate("A", "A", "A", False)

        if data_positive:
            if ts_strength > 0.3:
                _maybe_add_candidate("M", "A", "N", False)
            if ss_strength > 0.3 and m > 1:
                _maybe_add_candidate("M", "N", "M", False)
                if ts_strength > 0.2:
                    _maybe_add_candidate("M", "A", "M", False)

        # Prioritize a richer set of candidates for accuracy; we still cap the
        # total to avoid pathological runtimes on large grids.
        candidates = candidates[:12]

        priority = ["ANN", "AAN", "ANA", "AAA", "MAN", "MNM", "MAM", "AAM", "MMM"]
        priority_index = {k: i for i, k in enumerate(priority)}

        def _rank(c):
            et, tt, st, dt = c
            key = f"{et}{tt}{st}"
            base = priority_index.get(key, len(priority))
            damp_penalty = 0.5 if dt else 0.0
            return (base + damp_penalty, key)

        candidates.sort(key=_rank)
    else:
        for etype in errortype:
            for ttype in trendtype:
                for stype in seasontype:
                    for dtype in damped:
                        _maybe_add_candidate(etype, ttype, stype, dtype)

    if auto_model and len(candidates) > 3:
        # Quick pre-screening pass: use a modest number of steps to rank
        # candidates, then keep only the best few for full optimization.
        quick_steps = 20 if optax_steps is None else min(int(optax_steps), 20)
        quick_scores = []
        for etype, ttype, stype, dtype in candidates:
            init_key = (ttype, stype)
            init_state_override = _template_cache.get_init_state(y, m, ttype, stype)
            quick_fit = etsmodel(
                y,
                m,
                etype,
                ttype,
                stype,
                dtype,
                alpha,
                beta,
                gamma,
                phi,
                lower=lower,
                upper=upper,
                opt_crit=opt_crit,
                nmse=nmse_sel,
                bounds=bounds,
                maxit=maxit,
                optax_steps=quick_steps,
                optax_lr=optax_lr,
                optax_clip=optax_clip,
                early_stop_patience=max(5, early_stop_patience // 2),  # Optimized: More aggressive for quick selection
                early_stop_min_delta=early_stop_min_delta * 2,  # Optimized: Relaxed for quick selection
                adaptive_tol=adaptive_tol,
                is_final_model=False,
                pad_to=pad_to,
                bucket_size=bucket_size,
                stabilize=selection_stabilize,
                pure_sigmoid=auto_model,
                selection_mode=True,
                init_state_override=init_state_override,
            )
            quick_scores.append((quick_fit[ic], (etype, ttype, stype, dtype)))
        quick_scores.sort(key=lambda item: float(item[0]))
        # Keep the top few candidates for final selection to balance speed/accuracy.
        candidates = [c for _, c in quick_scores[:4]]

    t_select_start = time.perf_counter() if timing_enabled else 0.0
    # For non-auto (fixed-spec) models there is only one candidate, so
    # skip the selection_mode=True fast path and go straight to a full fit.
    # selection_mode=True uses _calc_roll_nohist which returns NaN for some
    # model specs (e.g. AAN, AAA), causing the selection to fail.
    use_selection_mode = auto_model
    for etype, ttype, stype, dtype in candidates:
        init_key = (ttype, stype)
        if init_key in init_state_cache:
            init_state_override = init_state_cache[init_key]
        else:
            init_state_override = _template_cache.get_init_state(y, m, ttype, stype)
            init_state_cache[init_key] = init_state_override
        fit = etsmodel(
            y,
            m,
            etype,
            ttype,
            stype,
            dtype,
            alpha,
            beta,
            gamma,
            phi,
            lower=lower,
            upper=upper,
            opt_crit=opt_crit,
            nmse=nmse_sel,
            bounds=bounds,
            maxit=maxit,
            optax_steps=optax_steps,
            optax_lr=optax_lr,
            optax_clip=optax_clip,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            adaptive_tol=adaptive_tol,
            is_final_model=True,
            pad_to=pad_to,
            bucket_size=bucket_size,
            stabilize=selection_stabilize,
            pure_sigmoid=auto_model,
            selection_mode=use_selection_mode,
            init_state_override=init_state_override,
        )
        fit_ic = fit[ic]
        if not math.isnan(float(fit_ic)):
            aicc_delta = float(fit_ic - best_ic)
            if float(fit_ic) < float(best_ic):
                best = fit
                best_ic = fit_ic
                best_e = etype
                best_t = ttype
                best_s = stype
                best_d = dtype
                no_improve = 0
            else:
                no_improve += 1
            evals += 1
            # Selection early-stopping tuned slightly toward accuracy: require
            # more evidence before giving up on exploring candidates.
            if (evals >= min_evals and no_improve >= max_no_improve) or (
                evals >= 4 and aicc_delta > 4.0
            ) or (evals >= 12):
                stop_search = True
                break
    if best is None:
        raise ValueError("No admissible ETS model found")
    t_select_end = time.perf_counter() if timing_enabled else 0.0
    # For auto_model, the selection loop used selection_mode=True (fast path)
    # so we need a final re-fit with selection_mode=False to get full results
    # (fitted values, residuals, states). For non-auto models, the selection
    # loop already used selection_mode=False, so the result is complete.
    if auto_model:
        init_key = (best_t, best_s)
        init_state_override = init_state_cache.get(init_key)
    t_opt_start = time.perf_counter() if timing_enabled else 0.0
    if auto_model:
        best = etsmodel(
            y,
            m,
            best_e,
            best_t,
            best_s,
            best_d,
            alpha,
            beta,
            gamma,
            phi,
            lower=lower,
            upper=upper,
            opt_crit=opt_crit,
            nmse=nmse,
            bounds=bounds,
            maxit=maxit,
            optax_steps=optax_steps,
            optax_lr=optax_lr,
            optax_clip=optax_clip,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            adaptive_tol=adaptive_tol,
            is_final_model=True,
            pad_to=pad_to,
            bucket_size=bucket_size,
            stabilize=True,
            pure_sigmoid=auto_model,
            init_state_override=init_state_override,
        )
    t_opt_end = time.perf_counter() if timing_enabled else 0.0
    if best is None or jnp.isinf(best_ic):
        raise Exception("no model able to be fitted")
    best["method"] = f"ETS({best_e},{best_t}{'d' if best_d else ''},{best_s})"
    if timing_enabled:
        best["_timing"] = {
            "selection_sec": float(t_select_end - t_select_start),
            "final_fit_sec": float(t_opt_end - t_opt_start),
            "total_ets_f_sec": float(time.perf_counter() - t_start),
        }
        print(f"[Chronax ETS timing] {best['_timing']}")
    return best


def pegelsfcast_C(h, obj, npaths=None, level=None, bootstrap=None):
    """
    One-step call to produce the mean forecast path from a fitted model dict.

    Parameters
    ----------
    h : int
        Horizon.
    obj : dict
        Fitted model dictionary from `etsmodel` / `ets_f`.
    npaths, level, bootstrap : unused
        Present for interface parity.

    Returns
    -------
    jnp.ndarray
        Mean forecast of length h.
    """
    states = jnp.asarray(obj["states"][-1, :], dtype=jnp.float64)
    etype, ttype, stype = [switch(comp) for comp in obj["components"][:3]]
    phi = 1.0 if obj["components"][3] == "N" else float(obj["par"][3])
    m = int(obj["m"])

    # allocate and CAPTURE the result of etsforecast
    forecast = jnp.full((h,), jnp.nan, dtype=states.dtype)
    forecast = etsforecast(
        x=states, m=m, trend=ttype, season=stype, phi=phi, h=h, f=forecast
    )
    return forecast


@partial(jax.jit, static_argnames=("h",))
def _compute_sigmah(pf, h, sigma, cvals):
    """
    Helper for multiplicative-error variance recursion used in intervals.

    Parameters
    ----------
    pf : jnp.ndarray
        Point forecasts (length h).
    h : int
        Horizon.
    sigma : float
        Innovation variance estimate (sigma^2).
    cvals : jnp.ndarray
        Coefficients per step capturing linearization terms.

    Returns
    -------
    jnp.ndarray
        sigma_h (length h) used to scale interval widths.
    """
    theta = jnp.zeros((h,), dtype=pf.dtype)
    theta = theta.at[0].set(pf[0] ** 2)

    def body(k, theta_acc):
        sum_val = jnp.dot(cvals[:k] ** 2, theta_acc[:k][::-1])
        return theta_acc.at[k].set(pf[k] ** 2 + sigma * sum_val)

    theta = lax.fori_loop(1, h, body, theta)
    return (1 + sigma) * theta - pf**2


@partial(jax.jit, static_argnames=("h", "season_length", "trend", "damped"))
def _class3models(
    h,
    sigma,
    last_state,
    season_length,
    error,
    trend,
    seasonality,
    damped,
    alpha,
    beta,
    gamma,
    phi,
):
    """
    Analytical variance for Class 3 (multiplicative seasonality) ETS cases.

    Returns
    -------
    jnp.ndarray
        Variance per horizon step (length h).
    """
    damped_val = (damped != "N")
    p = last_state.shape[0]

    H1 = jnp.array([[1, 1]], dtype=jnp.float64) if trend != "N" else jnp.array([[1]], dtype=jnp.float64)
    H2 = jnp.concatenate([jnp.zeros(season_length - 1), jnp.array([1.0])]).reshape(1, season_length)

    if trend == "N":
        F1 = jnp.array(1.0)
        G1 = jnp.array(alpha)
    else:
        f1 = phi if damped_val else 1.0
        F1 = jnp.array([[1.0, 1.0], [0.0, f1]], dtype=jnp.float64)
        G1 = jnp.array([[alpha, alpha], [beta, beta]], dtype=jnp.float64)

    f2_top = jnp.concatenate([jnp.zeros(season_length - 1), jnp.array([1.0])]).reshape(1, season_length)
    f2_bottom = jnp.c_[jnp.identity(season_length - 1), jnp.zeros((season_length - 1, 1))]
    F2 = jnp.r_[f2_top, f2_bottom]

    G2 = jnp.zeros((season_length, season_length))
    G2 = G2.at[0, season_length - 1].set(gamma)

    Mh = jnp.matmul(
        last_state[0 : (p - season_length)].reshape(-1, 1),
        last_state[(p - season_length) : p].reshape(1, season_length),
    )
    Vh = jnp.zeros((Mh.size, Mh.size))
    H21 = jnp.kron(H2, H1)
    F21 = jnp.kron(F2, F1)
    G21 = jnp.kron(G2, G1)
    K = jnp.kron(G2, F1) + jnp.kron(F2, G1)
    mu = jnp.zeros(h, dtype=last_state.dtype)
    var = jnp.zeros(h, dtype=last_state.dtype)
    vecMh = Mh.flatten()
    vecMh_col = vecMh.reshape(vecMh.shape[0], 1)

    def body(i, carry):
        Vh_acc, mu_acc, var_acc = carry
        mu_i = jnp.squeeze(H1 @ (Mh @ H2.T))
        var_i = jnp.squeeze((1 + sigma) * (H21 @ (Vh_acc @ H21.T))) + sigma * (mu_i ** 2)
        mu_acc = mu_acc.at[i].set(mu_i)
        var_acc = var_acc.at[i].set(var_i)
        exp1 = F21 @ (Vh_acc @ F21.T)
        exp2 = F21 @ (Vh_acc @ G21.T)
        exp3 = G21 @ (Vh_acc @ F21.T)
        exp4 = K @ ((Vh_acc + (vecMh * vecMh_col)) @ K.T)
        exp5 = (sigma * G21) @ ((3 * Vh_acc + 2 * vecMh * vecMh_col) @ G21.T)
        Vh_next = exp1 + sigma * (exp2 + exp3 + exp4 + exp5)
        return (Vh_next, mu_acc, var_acc)

    Vh, mu, var = lax.fori_loop(0, h, body, (Vh, mu, var))

    if trend == "N":
        Mh = F1 * (Mh @ F2.T) + G1 * (Mh @ G2.T) * sigma
    else:
        Mh = F1 @ (Mh @ F2.T) + G1 @ (Mh @ G2.T) * sigma

    return var


def _compute_pred_intervals(model: Dict[str, Any], forecasts: Dict[str, jnp.ndarray], h: int, level):
    """
    Compute prediction intervals for ETS forecasts.

    Uses closed-form expressions for Classes 1–3 where available, and falls
    back to simulation for Classes 4–5 (mixtures/multiplicative trickier cases).

    Parameters
    ----------
    model : dict
        Fitted model dictionary (from `etsmodel` / `ets_f`).
    forecasts : dict
        Must include "mean" forecast path.
    h : int
        Horizon.
    level : list[int]
        Confidence levels, e.g., [80, 95].

    Returns
    -------
    dict
        Keys: "lo-<level>", "hi-<level>" arrays aligned with horizon.
    """
    sigma = float(model["sigma2"])
    season_length = int(model["m"])
    pf = forecasts["mean"]

    model_type = model["components"]
    steps = jnp.arange(1, h + 1)
    hm = jnp.floor((h - 1) / season_length)
    last_state = jnp.asarray(model["states"][-1])

    # error, trend, and seasonality type
    error = model_type[0]
    trend = model_type[1]
    seasonality = model_type[2]
    damped = model_type[3]

    # parameters
    alpha = float(model["par"][0])
    beta = float(model["par"][1])
    gamma = float(model["par"][2])
    phi = float(model["par"][3])

    exp1 = alpha**2 + alpha * beta * steps + (1 / 6) * beta**2 * steps * (2 * steps - 1)
    exp2 = (beta * phi * steps) / (1 - phi) ** 2
    exp3 = 2 * alpha * (1 - phi) + beta * phi
    exp4 = (beta * phi * (1 - phi**steps)) / ((1 - phi) ** 2 * (1 - phi**2))
    exp5 = 2 * alpha * (1 - phi**2) + beta * phi * (1 + 2 * phi - phi**steps)

    compute_intervals = True
    # Class 1 models
    if error == "A" and trend == "N" and seasonality == "N" and damped == "N":
        sigmah = 1 + alpha**2 * (steps - 1)
        sigmah = sigma * sigmah

    elif error == "A" and trend == "A" and seasonality == "N" and damped == "N":
        sigmah = 1 + (steps - 1) * exp1
        sigmah = sigma * sigmah

    elif error == "A" and trend == "A" and seasonality == "N" and damped == "D":
        sigmah = 1 + alpha**2 * (steps - 1) + exp2 * exp3 - exp4 * exp5
        sigmah = sigma * sigmah

    elif error == "A" and trend == "N" and seasonality == "A" and damped == "N":
        sigmah = 1 + alpha**2 * (steps - 1) + gamma * hm * (2 * alpha + gamma)
        sigmah = sigma * sigmah

    elif error == "A" and trend == "A" and seasonality == "A" and damped == "N":
        exp6 = 2 * alpha + gamma + beta * season_length * (hm + 1)
        sigmah = 1 + (steps - 1) * exp1 + gamma * hm * exp6
        sigmah = sigma * sigmah

    elif error == "A" and trend == "A" and seasonality == "A" and damped == "D":
        exp7 = (2 * beta * gamma * phi) / ((1 - phi) * (1 - phi**season_length))
        exp8 = hm * (1 - phi**season_length) - (phi**season_length) * (1 - phi ** (season_length * hm))
        sigmah = 1 + alpha**2 * (steps - 1) + exp2 * exp3 - exp4 * exp5 + gamma * hm * (2 * alpha + gamma) + exp7 * exp8
        sigmah = sigma * sigmah

    # Class 2 models
    elif error == "M" and trend == "N" and seasonality == "N" and damped == "N":
        cvals = jnp.full((h,), alpha)
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and trend == "A" and seasonality == "N" and damped == "N":
        cvals = alpha + beta * steps
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and trend == "A" and seasonality == "N" and damped == "D":
        cvals = jnp.full((h,), jnp.nan)
        phi_powers = phi ** jnp.arange(1, h + 1, dtype=jnp.float64)
        cvals = cvals.at[:].set(alpha + beta * jnp.cumsum(phi_powers))
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and trend == "N" and seasonality == "A" and damped == "N":
        dvals = jnp.where(jnp.arange(1, h + 1) % season_length == 0, 1.0, 0.0)
        cvals = alpha + gamma * dvals
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and trend == "A" and seasonality == "A" and damped == "N":
        dvals = jnp.where(jnp.arange(1, h + 1) % season_length == 0, 1.0, 0.0)
        cvals = alpha * beta * steps + gamma * dvals
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and trend == "A" and seasonality == "A" and damped == "D":
        dvals = jnp.where(jnp.arange(1, h + 1) % season_length == 0, 1.0, 0.0)
        phi_powers = phi ** jnp.arange(1, h + 1, dtype=jnp.float64)
        cvals = alpha + beta * jnp.cumsum(phi_powers) + gamma * dvals
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and seasonality == "M":
        sigmah = _class3models(
            h, sigma, last_state, season_length, error, trend, seasonality, damped, alpha, beta, gamma, phi
        )

    else:
        # Classes 4 and 5 models: simulation approach
        compute_intervals = False
        nsim = 5000
        y_path = jnp.zeros((nsim, h), dtype=jnp.float64)

        # Fill NaNs as in original
        beta_sim = 0.0 if math.isnan(beta) else beta
        gamma_sim = 0.0 if math.isnan(gamma) else gamma
        phi_sim = 0.0 if math.isnan(phi) else phi

        key = jrand.PRNGKey(1)
        e = jrand.normal(key, shape=(nsim, h)) * math.sqrt(sigma)

        def run_sim(e_k):
            return _etssimulate_jit(
                last_state,
                season_length,
                switch(error),
                switch(trend),
                switch(seasonality),
                alpha,
                beta_sim,
                gamma_sim,
                phi_sim,
                h,
                e_k,
            )

        y_path = jax.vmap(run_sim)(e)

        lower_q = 0.5 - jnp.asarray(level) / 200.0
        upper_q = 0.5 + jnp.asarray(level) / 200.0
        lower = jnp.quantile(y_path, lower_q.reshape(-1, 1), axis=0)
        upper = jnp.quantile(y_path, upper_q.reshape(-1, 1), axis=0)
        pi = {
            **{f"lo-{int(lv)}": lower[i] for i, lv in enumerate(level)},
            **{f"hi-{int(lv)}": upper[i] for i, lv in enumerate(level)},
        }

    if compute_intervals:
        pi = _calculate_intervals(forecasts, level=level, h=h, sigmah=jnp.sqrt(sigmah))

    return pi


def forecast_ets(obj, h, level=None):
    """
    Convenience wrapper: produce forecasts (and optional PI) from fitted model.

    Parameters
    ----------
    obj : dict
        Fitted model dictionary returned by `ets_f`/`etsmodel`.
    h : int
        Horizon.
    level : Optional[list[int]]
        Confidence levels (e.g., [80, 95]) for prediction intervals.

    Returns
    -------
    dict
        Keys: "mean", "residuals", "fitted", and optionally "lo-XX"/"hi-XX".
    """
    fcst = pegelsfcast_C(h, obj)
    out = {"mean": fcst}
    out["residuals"] = obj["residuals"]
    out["fitted"] = obj["fitted"]
    if level is not None:
        pi = _compute_pred_intervals(model=obj, forecasts=out, level=level, h=h)
        out = {**out, **pi}
    return out


def forward_ets(fitted_model, y):
    """
    Reuse a previously fitted ETS model dictionary to roll forward on new data.

    Parameters
    ----------
    fitted_model : dict
        Output of `ets_f` / `etsmodel` with learned params & states.
    y : array-like
        New time series segment to append/continue the fit.

    Returns
    -------
    dict
        New fitted model dict (same schema), reusing structure & params.
    """
    return ets_f(y=y, m=fitted_model["m"], model=fitted_model)

"""
1. ensure_float() has one signature, a JAX array, and returns one object, a JAX array.
   It is a helper method for to ensure that the datatypes within a given array are float32. 
"""


import jax
from jax import lax
import jax.numpy as jnp
from functools import partial as _partial
from typing import Optional, List, Dict, Union, Tuple
from jax.scipy.special import ndtri  # JAX inverse normal CDF
from collections import namedtuple
from functools import partial
import os

results = namedtuple("results", "x fn nit simplex")

def ensure_float(y: jnp.ndarray) -> jnp.ndarray:
    if not jnp.issubdtype(y.dtype, jnp.floating):
        return y.astype(jnp.float32)
    return y

@jax.jit
def calculate_sigma(residuals: jnp.ndarray, n: int) -> jnp.ndarray:
    """Calculate sigma for residuals using JAX operations.

    Args:
        residuals: Residual values
        n: Number of degrees of freedom

    Returns:
        Sigma value as JAX array
    """
    sigma = jnp.where(
        n > 0,
        jnp.sqrt(jnp.nansum(residuals**2) / n),
        0.0
    )
    return sigma

def _jax_norm_ppf(p):
    """JAX implementation of normal percent point function (inverse CDF).

    Uses Beasley-Springer-Moro approximation for the inverse normal CDF.
    """
    # Clamp p to avoid numerical issues
    p = jnp.clip(p, 1e-10, 1 - 1e-10)

    # For p > 0.5, use symmetry
    sign = jnp.where(p > 0.5, 1.0, -1.0)
    p_adj = jnp.where(p > 0.5, p, 1.0 - p)

    # Beasley-Springer-Moro approximation
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    t = jnp.sqrt(-2 * jnp.log(1 - p_adj))
    z = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)

    return sign * z

def _quantiles(level: list[int | float]) -> jnp.ndarray:
    """
    Convert confidence levels to z-scores using normal inverse CDF.
    JAX equivalent of statsforecast.utils._quantiles()
    
    Args:
        level: List of confidence levels (0-100), e.g., [80, 95]
        
    Returns:
        Array of z-scores corresponding to each level
    """
    level_arr = jnp.atleast_1d(jnp.asarray(level, jnp.float32))
    p = 0.5 + (level_arr / 200.0)
    return jax.vmap(_jax_norm_ppf)(p)

def _calculate_intervals(
    res: dict,
    level: list[int],
    h: int,
    sigmah: jnp.ndarray | float
) -> dict:
    """
    Calculate native (non-conformal) prediction intervals using normal quantiles.
    Compatible with SeasonalNaive.predict() calls.
    """
    # Ensure mean is a JAX array
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
    fitted: jnp.ndarray, 
    sigmah: jnp.ndarray | float, 
    level: list[int]
) -> dict:
    """
    Calculate NATIVE (non-conformal) fitted (in-sample) prediction intervals.
    JAX equivalent of statsforecast.models._add_fitted_pi()
    
    Args:
        fitted: Fitted values of shape (t,)
        sigmah: Standard error for predictions (scalar or array)
        level: Sorted list of confidence levels
        
    Returns:
        Dictionary with 'fitted-lo-XX' and 'fitted-hi-XX' keys for each level
    """
    z = _quantiles(level)
    # Broadcast to shape (t, len(level))
    lo = fitted[:, None] - z[None, :] * sigmah
    hi = fitted[:, None] + z[None, :] * sigmah
    
    out = {}
    for i, lv in enumerate(level[::-1]):
        out[f"fitted-lo-{int(lv)}"] = lo[:, len(level) - 1 - i]
    for i, lv in enumerate(level):
        out[f"fitted-hi-{int(lv)}"] = hi[:, i]
    return out

from functools import partial as _partial

@_partial(jax.jit, static_argnums=(1,))
def _repeat_val_seas(season_vals: jnp.ndarray, h: int) -> jnp.ndarray:
    """
    Tile seasonal values to cover forecast horizon h.
    JAX equivalent of statsforecast.utils._repeat_val_seas()
    
    Args:
        season_vals: Seasonal pattern of shape (season_length,)
        h: Forecast horizon (static - must be known at compile time)
        
    Returns:
        Tiled pattern of length h
        
    Example:
        >>> season_vals = jnp.array([10.0, 20.0, 30.0])
        >>> _repeat_val_seas(season_vals, h=7)
        array([10., 20., 30., 10., 20., 30., 10.])
    """
    import math
    repeats = math.ceil(h / season_vals.size)
    return jnp.tile(season_vals, repeats)[:h]

@_partial(jax.jit, static_argnums=(1,))
def _repeat_val(val: float, h: int) -> jnp.ndarray:
    """
    Repeat scalar value h times.
    JAX equivalent of statsforecast.utils._repeat_val()
    
    Args:
        val: Scalar value to repeat
        h: Number of repetitions (forecast horizon)
        
    Returns:
        Array of length h filled with val
    """
    return jnp.full(h, val, dtype=jnp.float32)
def _seasonal_naive(
    y,
    h: int,
    season_length: int,
    fitted: bool = False,
) -> Dict[str, jnp.ndarray]:
    """
    JAX implementation of seasonal-naive forecast.
    
    Args:
        y: 1-D array-like (length T). Will be converted to jax array (float32).
        h: forecast horizon (int >= 1)
        season_length: seasonal period m (int >= 1)
        fitted: if True, also return in-sample fitted values
        
    Returns:
        dict with keys:
          - "mean": jnp.ndarray shape (h,)
          - optionally "fitted": jnp.ndarray shape (T,)
    """
    # convert input to jax array float32
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

    # last m observations (shape (m,))
    last_m = y_j[-m:]

    # build mean forecast by cycling through last_m
    idx = jnp.arange(h) % m            # shape (h,)
    mean = last_m[idx]                 # shape (h,)

    out = {"mean": mean}

    if fitted:
        # build fitted array: NaN for first m entries, and y[0:T-m] mapped to positions m..T-1
        fitted = jnp.full((T,), jnp.nan, dtype=jnp.float32)
        # values to place: y[0 : T-m]
        vals = y_j[: T - m]
        # scatter assignment into fitted at positions m..T-1
        positions = jnp.arange(m, T)
        fitted = fitted.at[positions].set(vals)
        out["fitted"] = fitted

    return out

def _add_fitted_pi(res, se, level):
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

def _add_conformal_distribution_intervals(
    fcst: dict,
    cs: jnp.ndarray,
    level: list[float] | list[int],
) -> dict:
    """
    Adds conformal intervals to the `fcst` dict based on conformal scores `cs`.
    `level` should be already sorted. This strategy creates forecast paths
    based on errors and calculates quantiles using those paths.
    """
    level = sorted(level)
    alphas = jnp.array([100 - lv for lv in level], dtype=jnp.float32)
    cuts_lower = (alphas / 200.0)[::-1]          # lower cuts reversed
    cuts_upper = 1.0 - (alphas / 200.0)         # upper cuts
    cuts = jnp.concatenate([cuts_lower, cuts_upper])

    mean = fcst["mean"].reshape(1, -1)          # 2D: 1 x horizon
    scores = jnp.vstack([mean - cs, mean + cs]) # shape: 2 x horizon
    quantiles = jnp.quantile(scores, cuts, axis=0)

    # generate column names
    lo_cols = [f"lo-{lv}" for lv in reversed(level)]
    hi_cols = [f"hi-{lv}" for lv in level]
    out_cols = lo_cols + hi_cols

    for i, col in enumerate(out_cols):
        fcst[col] = quantiles[i]

    return fcst

def _get_conformal_method(method: str):
    available_methods = {
        "conformal_distribution": _add_conformal_distribution_intervals,
        # "conformal_error": _add_conformal_error_intervals,
    }
    if method not in available_methods.keys():
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f"please choose one of {', '.join(available_methods.keys())}"
        )
    return available_methods[method]

# Optional: jitted wrapper (uncomment to use)
# _seasonal_naive_jit = jax.jit(_seasonal_naive, static_argnums=(1,2,3))

@jax.jit
def _ses_forecast(x, alpha):
    complement = 1 - alpha
    n = x.size
    fitted = jnp.full_like(x, jnp.nan)
    fitted = fitted.at[0].set(x[0])  # first value

    def body_fun(i, val):
        fitted_arr, j = val
        fitted_arr = fitted_arr.at[i].set(alpha * x[j] + complement * fitted_arr[j])
        j += 1
        return fitted_arr, j

    fitted, _ = jax.lax.fori_loop(1, n, body_fun, (fitted, 0))
    forecast = alpha * x[-1] + complement * fitted[-1]
    fitted = fitted.at[0].set(jnp.nan)  # match original behavior
    return forecast, fitted


def _seasonal_exponential_smoothing(y, h, fitted, season_length, alpha):
    n = y.size
    if n < season_length:
        return {"mean": jnp.full(h, jnp.nan, dtype=y.dtype)}

    season_vals = jnp.full((season_length,), jnp.nan, dtype=y.dtype)
    fitted_vals = jnp.full_like(y, jnp.nan)

    for i in range(season_length):
        init_idx = i + n % season_length
        x = y[init_idx::season_length]  # Python slice, works fine

        forecast, fitted_season = _ses_forecast(x, alpha)

        season_vals = season_vals.at[i].set(forecast)

        for k in range(fitted_season.size):
            fitted_vals = fitted_vals.at[init_idx + k * season_length].set(fitted_season[k])

    out = _repeat_val_seas(season_vals, h)
    fcst = {"mean": out}
    if fitted:
        fcst["fitted"] = fitted_vals
    return fcst

def _conformal_method(self):
        return _get_conformal_method(self.prediction_intervals.method)

def _store_cs(self, y, X):
    if self.prediction_intervals is not None:
        self._cs = self.conformity_scores(y, X)

def _add_conformal_intervals(self, fcst, y, X, level):
    if self.prediction_intervals is not None and level is not None:
        cs = self.conformity_scores(y, X) if y is not None else self._cs
        res = self._conformal_method(fcst=fcst, cs=cs, level=level)
        return res
    return fcst

def _add_predict_conformal_intervals(self, fcst, level):
    return self._add_conformal_intervals(fcst=fcst, y=None, X=None, level=level)




def _calculate_intervals(out, level, h, sigmah):
    # level may be list/tuple/array — keep Python copy for dict keys
    level_list = list(level)

    # Quantiles as JAX array
    z = _quantiles(jnp.asarray(level_list))           # shape: (L,)

    # Build (L, h) matrix of quantiles
    zz = jnp.repeat(z[:, None], h, axis=1)            # shape: (L, h)

    # Ensure (1, h) shapes for broadcasting
    mean_row = out["mean"][None, :]                   # (1, h)
    sigmah_row = sigmah[None, :]                      # (1, h)

    lower = mean_row - zz * sigmah_row                # (L, h)
    upper = mean_row + zz * sigmah_row                # (L, h)

    pred_int = {
        **{f"lo-{lv}": lower[i] for i, lv in enumerate(level_list)},
        **{f"hi-{lv}": upper[i] for i, lv in enumerate(level_list)},
    }
    return pred_int

@jax.jit
def _quantiles(level):
    level = jnp.asarray(level)
    # norm.ppf(0.5 + level/200) -> ndtri in JAX
    z = ndtri(0.5 + level / 200.0)
    return z

def _calculate_sigma(residuals, n):
    if n > 0:
        sigma = jnp.nansum(residuals**2)
        sigma = sigma / n
        sigma = jnp.sqrt(sigma)
    else:
        sigma = 0
    return sigma

def _repeat_val(val: float, h: int) -> jnp.ndarray:
    return jnp.full((h,), jnp.asarray(val))

@partial(jax.jit, static_argnums=(1, 2))
def _window_average_core(y: jnp.ndarray, window_size: int, h: int) -> jnp.ndarray:
    """
    JIT-able core: take the last `window_size` values using dynamic_slice (static size),
    average them, and repeat to length h.
    """
    n = y.shape[0]
    # start = max(0, n - window_size)  (dynamic start is OK; size must be static)
    start = jnp.maximum(0, n - window_size)
    tail = lax.dynamic_slice(y, (start,), (window_size,))
    wavg = jnp.mean(tail)
    return jnp.full((h,), wavg, dtype=y.dtype)

def _window_average(
    y: jnp.ndarray,  # time series
    h: int,          # forecasting horizon
    fitted: bool,    # fitted values
    window_size: int # window size
):
    if fitted:
        raise NotImplementedError("return fitted")
    if y.size < window_size:
        return {"mean": jnp.full((h,), jnp.nan, dtype=y.dtype)}
    # JIT-compiled fast path
    mean = _window_average_core(y, window_size, h)
    return {"mean": mean}

# JAX-only IMAPA with SES + bounded golden-section search
# -------------------------------------------------------
# imapa_jax.py — Pure JAX IMAPA with Brent-bounded SES optimizer (search in float64)

from typing import Tuple, Dict, Any

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
import warnings


# ---------------------------
# Small helpers
# ---------------------------

def _repeat_val_(val: float, h: int, dtype) -> jnp.ndarray:
    return jnp.full((h,), jnp.asarray(val, dtype=dtype))


def _intervals(x: jnp.ndarray) -> jnp.ndarray:
    """Intervals between nonzero elements (match numpy reference)."""
    idx = jnp.where(x != 0)[0]
    padded = jnp.concatenate([jnp.array([0], dtype=idx.dtype), idx + 1])
    diffs = jnp.diff(padded)
    return diffs.astype(x.dtype)


def _chunk_sums(array: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    """Split into equal chunks and sum each chunk. Incomplete tail discarded."""
    n = array.size
    n_chunks = n // chunk_size
    n_elems = n_chunks * chunk_size
    trimmed = array[:n_elems]
    if n_chunks == 0:
        return jnp.zeros((0,), dtype=array.dtype)
    reshaped = trimmed.reshape((n_chunks, chunk_size))
    return reshaped.sum(axis=1)


# ---------------------------
# SES core
# ---------------------------

def _ses_sse(alpha: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Residual sum of squares for simple exponential smoothing."""
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
    forecast, sse = lax.fori_loop(1, n, body_fun, init_state)
    return sse


def _ses_forecast(x: jnp.ndarray, alpha: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One-step ahead forecast and in-sample fitted values for SES."""
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


# ---------------------------
# Golden-section (SciPy "bounded") optimizer
# ---------------------------

def _golden_bounded_minimize(
    f,
    a: float,
    b: float,
    dtype=jnp.float64,
    xatol: float | None = None,
    maxiter: int = 1000,
):
    """
    Deterministic golden-section search matching SciPy's "bounded" behavior.
    All arithmetic in `dtype` (use float64 to mimic SciPy).
    Returns (x*, f(x*)).
    """
    if xatol is None:
        # SciPy's bounded uses absolute tolerance; we use a tight default in float64
        xatol = 1e-12 if dtype == jnp.float64 else 1e-7

    a = jnp.asarray(a, dtype=dtype).item()
    b = jnp.asarray(b, dtype=dtype).item()
    if not (a < b):
        raise ValueError("Bounds must satisfy a < b.")

    invphi = (jnp.sqrt(jnp.asarray(5.0, dtype=dtype)) - 1.0) / 2.0   # ~0.6180339887
    invphi2 = 1.0 - invphi                                           # ~0.3819660113

    # Initial interior points
    h = b - a
    if h <= xatol:
        x = (a + b) / 2.0
        return jnp.asarray(x, dtype=dtype), jnp.asarray(f(jnp.asarray(x, dtype=dtype)), dtype=dtype)

    n = int(jnp.ceil(jnp.log(xatol / h) / jnp.log(invphi))) if h > 0 else 1
    c = a + invphi2 * h
    d = a + invphi * h
    fc = float(f(jnp.asarray(c, dtype=dtype)))
    fd = float(f(jnp.asarray(d, dtype=dtype)))

    it = 0
    while it < maxiter and (d - c) > xatol:
        it += 1
        if fc < fd:
            b, d, fd = d, c, fc
            h = invphi * h
            c = a + invphi2 * h
            fc = float(f(jnp.asarray(c, dtype=dtype)))
        else:
            a, c, fc = c, d, fd
            h = invphi * h
            d = a + invphi * h
            fd = float(f(jnp.asarray(d, dtype=dtype)))

    # Best point is the smaller of c,d (or their function values)
    if fc < fd:
        xstar, fstar = c, fc
    else:
        xstar, fstar = d, fd

    return jnp.asarray(xstar, dtype=dtype), jnp.asarray(fstar, dtype=dtype)


# ---------------------------
# Optimized SES forecast (search in float64 via golden-section)
# ---------------------------
def _optimized_ses_forecast(
    x: jnp.ndarray, bounds: Tuple[float, float] = (0.1, 0.3)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute one-step SES forecast with alpha chosen by golden-section minimization of SSE.

    Heuristic:
      - If input is float32 AND (sequence is degenerate (<=2 nonzero) OR has any negatives),
        run BOTH the optimizer and SES recursion in float32 to match NumPy reference.
      - Otherwise, run both in float64 for numerical stability.
    """
    x = ensure_float(x)
    out_dtype = x.dtype

    # Detect tricky cases that are sensitive to fp32 rounding in NumPy
    nonzero_cnt = jnp.sum(x != 0)
    has_neg = jnp.any(x < 0)
    prefer_fp32 = (out_dtype == jnp.float32) & ((nonzero_cnt <= 2) | has_neg)

    run_dtype = jnp.float32 if prefer_fp32 else jnp.float64
    x_run = x.astype(run_dtype)

    def obj(a):
        return _ses_sse(a, x_run)

    # Slightly looser xatol for fp32 (closer to SciPy bounded behavior in fp32)
    xatol = 1e-8 if run_dtype == jnp.float32 else 1e-13

    alpha_star, _ = _golden_bounded_minimize(
        obj, bounds[0], bounds[1], dtype=run_dtype, xatol=xatol, maxiter=5000
    )

    # Run SES recursion in the same dtype we optimized in (to match NumPy path),
    # then cast outputs back to the original dtype of the series.
    forecast_run, fitted_run = _ses_forecast(x_run, alpha_star)

    forecast = forecast_run.astype(out_dtype)
    fitted = fitted_run.astype(out_dtype)
    return forecast, fitted



# ---------------------------
# IMAPA
# ---------------------------

def _imapa(
    y: jnp.ndarray,
    h: int,
    fitted: bool,
) -> Dict[str, Any]:
    """
    IMAPA forecaster in pure JAX (intermittent demand).

    What it does:
        1) Detects inter-arrival spacing of non-zero observations in `y` and
           computes a mean interval.
        2) Uses that mean (rounded) as the maximum aggregation level K.
        3) For each aggregation level k = 1..K:
            - Drops a short remainder (so length is divisible by k).
            - Chunks and sums the series into length-k blocks.
            - Fits Single Exponential Smoothing (SES) to the aggregated series,
              selecting alpha via a bounded golden-section search on SSE.
            - Scales the one-step SES forecast back by 1/k.
        4) Averages the per-k forecasts to produce a single constant-mean forecast
           of length `h`. Optionally computes in-sample fitted values by
           refitting on prefixes (O(T²) warning).

    How it differs from a typical NumPy reference:
        - Pure JAX: vectorized core math and JIT-friendly loops; no SciPy optimizer.
        - Optimizer: uses a deterministic golden-section search (SciPy-like
          “bounded” behavior) implemented in JAX instead of `scipy.optimize`.
        - Dtype policy: inputs are normalized with `ensure_float` (ints -> fp32).
          SES optimization/recursion may run in fp64 for stability except for
          very sparse or sign-changing fp32 inputs, where we mirror fp32 end-to-end
          to match NumPy paths more closely.
        - Guard rails: handles all-zeros fast path and skips empty chunk sets.

    Limitations:
        - Fitted values: computed by recursive refits on prefixes, which is
          O(T²) and expensive; intended for testing/debugging, not production.
        - JIT boundaries: the outer Python loops over aggregation levels and
          prefix refits are not fully fused; very large T or K can impact speed.
        - Sensitivity in edge cases: extremely short, single-spike, or highly
          negative/alternating sequences can be sensitive to dtype/tolerance;
          the fp32/fp64 heuristic mitigates this but tiny deltas vs NumPy can
          still occur if tolerances are set extremely tight.
        - Assumes non-seasonal SES per aggregation. If strong seasonality exists
          after aggregation, this model intentionally keeps the constant-mean
          IMAPA assumption.

    Returns:
        dict with:
          - "mean": (h,) constant forecast replicated across horizon
          - optionally "fitted": (T,) in-sample values with first element NaN

    """
    # All zeros shortcut
    if bool(jnp.all(y == 0)):
        out_dtype = y.dtype if y.dtype in (jnp.float32, jnp.float64) else jnp.float32
        res = {"mean": jnp.zeros((h,), dtype=out_dtype)}
        if fitted:
            f = jnp.zeros_like(ensure_float(y)).astype(out_dtype)
            f = f.at[0].set(jnp.asarray(jnp.nan, dtype=out_dtype))
            res["fitted"] = f
        return res

    y = ensure_float(y)
    dtype = y.dtype

    y_intervals = _intervals(y)
    mean_interval = jnp.mean(y_intervals)
    max_aggregation_level = int(jnp.rint(mean_interval).item())
    if max_aggregation_level < 1:
        max_aggregation_level = 1

    forecasts = jnp.empty((max_aggregation_level,), dtype=dtype)

    for aggregation_level in range(1, max_aggregation_level + 1):
        lost_remainder_data = int(y.shape[0] % aggregation_level)
        y_cut = y[lost_remainder_data:]
        aggregation_sums = _chunk_sums(y_cut, aggregation_level)
        if aggregation_sums.size == 0:
            # If no chunks, set NaN to skip in mean
            forecasts = forecasts.at[aggregation_level - 1].set(jnp.asarray(jnp.nan, dtype=dtype))
            continue
        fcast, _ = _optimized_ses_forecast(aggregation_sums)
        forecasts = forecasts.at[aggregation_level - 1].set(
            fcast / jnp.asarray(aggregation_level, dtype=dtype)
        )

    # Mean of finite forecasts (there shouldn't be NaNs normally, but guard anyway)
    finite_mask = jnp.isfinite(forecasts)
    forecast = jnp.where(
        finite_mask.any(),
        jnp.mean(forecasts[finite_mask]),
        jnp.asarray(0.0, dtype=dtype),
    )

    res: Dict[str, Any] = {"mean": _repeat_val_(val=forecast, h=h, dtype=dtype)}

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

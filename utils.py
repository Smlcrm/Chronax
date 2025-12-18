"""
1. ensure_float() has one signature, a JAX array, and returns one object, a JAX array.
   It is a helper method for to ensure that the datatypes within a given array are float32. 
"""
import os
from collections import namedtuple
from functools import partial
from typing import Optional, List, Dict, Union, Tuple
import theta_jax as _theta
from jax.scipy.stats import norm
import math
from collections import namedtuple
import jax.random as jrandom

import jax
from jax import jit, lax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jax.scipy.special import ndtri  # JAX inverse normal CDF

results = namedtuple("results", "x fn nit simplex")


import os
from collections import namedtuple
from functools import partial
from typing import Optional, List, Dict, Union, Tuple
import theta_jax as _theta
from jax.scipy.stats import norm
import math
from collections import namedtuple
import jax.random as jrandom

import jax
from jax import jit, lax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jax.scipy.special import ndtri  # JAX inverse normal CDF

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

def extract_demand(y: jnp.ndarray) -> jnp.ndarray:
    """Extract positive (non-zero) demand values from a time series.

    This is used for intermittent demand models like TSB and Croston,
    where we need to separate demand occurrences from no-demand periods.

    Args:
        y: Time series array that may contain zeros

    Returns:
        Array containing only positive values from y

    Example:
        >>> y = jnp.array([0, 5, 0, 0, 3, 2, 0])
        >>> extract_demand(y)
        Array([5., 3., 2.], dtype=float32)
    """
    return y[y > 0]

def extract_probability(y: jnp.ndarray) -> jnp.ndarray:
    """Convert time series to binary probability indicator (1=demand, 0=no demand).

    This is used for intermittent demand models like TSB to track the
    probability of demand occurrence at each time step.

    Args:
        y: Time series array

    Returns:
        Binary array where 1 indicates demand occurred, 0 indicates no demand

    Example:
        >>> y = jnp.array([0, 5, 0, 0, 3, 2, 0])
        >>> extract_probability(y)
        Array([0., 1., 0., 0., 1., 1., 0.], dtype=float32)
    """
    return (y != 0).astype(y.dtype)


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

# def _add_fitted_pi(
def _add_fitted_pi_1(
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
def _ses_forecast_nan(x, alpha):
    """
    Simple Exponential Smoothing forecast with NaN handling.
    
    Skips NaN values in computation - useful for padded arrays from Croston models.
    """
    complement = 1 - alpha
    n = x.size
    fitted = jnp.full_like(x, jnp.nan)
    
    # Find first non-NaN value
    is_valid = ~jnp.isnan(x)
    first_valid_idx = jnp.argmax(is_valid)  # Index of first True (non-NaN)
    first_valid_val = x[first_valid_idx]
    
    # Initialize fitted with first valid value
    fitted = fitted.at[first_valid_idx].set(first_valid_val)
    
    def body_fun(i, fitted_arr):
        # Only update if current value is not NaN
        val = x[i]
        prev_fitted = fitted_arr[i-1]
        
        # Compute new fitted value: use previous fitted if current is NaN
        new_fitted = jnp.where(
            jnp.isnan(val),
            jnp.nan,  # Keep NaN if input is NaN
            jnp.where(
                jnp.isnan(prev_fitted),
                val,  # If no previous fitted, use current value
                alpha * val + complement * prev_fitted
            )
        )
        fitted_arr = fitted_arr.at[i].set(new_fitted)
        return fitted_arr
    
    # Apply SES to all positions after first valid
    fitted = jax.lax.fori_loop(first_valid_idx + 1, n, body_fun, fitted)
    
    # Forecast: find last non-NaN value
    last_valid_idx = n - 1 - jnp.argmax(is_valid[::-1])
    forecast = fitted[last_valid_idx]
    
    # Set first fitted to NaN to match original behavior
    fitted = fitted.at[first_valid_idx].set(jnp.nan)
    
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

@jax.jit
def calculate_information_criteria(
    residuals: jnp.ndarray,
    n_params: int,
    n: int,
) -> Dict[str, jnp.ndarray]:
    """Calculate AIC, BIC, and AICc from residuals (JIT-compiled, returns JAX arrays)."""
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
    
    # Return JAX arrays instead of Python floats for JIT compatibility
    return {
        'loglik': -0.5 * lik,
        'aic': aic,
        'bic': bic,
        'aicc': aicc,
    }


@jax.jit
def _demand(x: jnp.ndarray) -> jnp.ndarray:
    """
    Extract positive (non-zero) elements from array.
    Used by Croston-family models for intermittent demand.
    
    Args:
        x: Input array
        
    Returns:
        Fixed-size array with non-zero values packed at start, rest filled with NaN
        
    Example:
        >>> x = jnp.array([0., 5., 0., 3., 0.])
        >>> _demand(x)
        array([5., 3., nan, nan, nan])
        
    Note:
        Returns fixed-size array (same size as input) for JIT compatibility.
        Non-zero values are packed at the start, remaining positions filled with NaN.
    """
    # Get indices where x > 0, with fill_value for padding
    indices = jnp.where(x > 0, size=x.size, fill_value=-1)[0]
    
    # Create result array: gather values where indices are valid, else NaN
    result = jnp.where(
        indices >= 0,
        jnp.where(indices < x.size, x[jnp.clip(indices, 0, x.size-1)], jnp.nan),
        jnp.nan
    )
    return result


@jax.jit
# def _intervals(x: jnp.ndarray) -> jnp.ndarray:
def _intervals_c(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute intervals between non-zero elements.
    Used by Croston-family models for intermittent demand.
    
    Args:
        x: Input array
        
    Returns:
        Fixed-size array with intervals packed at start, rest filled with NaN
        
    Example:
        >>> x = jnp.array([0., 5., 0., 0., 3., 0., 2.])
        >>> _intervals(x)
        array([1., 3., 2., nan, nan, nan, nan])  # First interval at position 1, then gap of 3, then gap of 2
        
    Note:
        Returns fixed-size array (same size as input) for JIT compatibility.
        Intervals are packed at the start, remaining positions filled with NaN.
    """
    # Get indices of non-zero elements, with fill_value for padding
    nonzero_idxs = jnp.where(x != 0, size=x.size, fill_value=-1)[0]
    
    # Compute positions (1-indexed)
    positions = jnp.where(nonzero_idxs >= 0, nonzero_idxs + 1, -1)
    
    # Compute intervals using diff with prepend
    intervals = jnp.diff(positions, prepend=0)
    
    # Mask out invalid intervals (where positions were -1)
    valid_mask = positions >= 0
    result = jnp.where(valid_mask, intervals.astype(x.dtype), jnp.nan)
    
    return result


def _expand_fitted_demand(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Expand demand fitted values back to original series length.
    Used by Croston-family models.
    
    Args:
        fitted: SES fitted values for demand (length = num_nonzero + 1)
        y: Original time series
        
    Returns:
        Fitted values expanded to match y's length
        
    Logic:
        - If y[i-1] > 0: Use next fitted value (demand occurred)
        - If y[i-1] == 0 and we've seen demand: Carry forward previous value
        - If y[i-1] == 0 and no demand yet: Use naive forecast (y[i-1])
    """
    n = y.size
    out = jnp.full_like(y, jnp.nan)
    
    def body_fn(i, state):
        out_arr, fitted_idx = state
        
        # If previous value was positive, advance fitted index
        fitted_idx = jnp.where(
            y[i - 1] > 0,
            fitted_idx + 1,
            fitted_idx
        )
        
        # Determine output value based on conditions
        val = jax.lax.cond(
            y[i - 1] > 0,
            lambda: fitted[fitted_idx],  # Use new fitted value
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],  # Carry forward previous
                lambda: y[i - 1]  # Use naive (no demand seen yet)
            )
        )
        
        out_arr = out_arr.at[i].set(val)
        return out_arr, fitted_idx
    
    out, _ = jax.lax.fori_loop(1, n, body_fn, (out, 0))
    return out


def _expand_fitted_intervals(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Expand interval fitted values back to original series length.
    Used by Croston-family models.
    
    Args:
        fitted: SES fitted values for intervals (length = num_nonzero + 1)
        y: Original time series
        
    Returns:
        Fitted intervals expanded to match y's length (avoids division by zero)
        
    Logic:
        - If y[i-1] != 0: Use next fitted value, but replace 0 with 1 (avoid div by zero)
        - If y[i-1] == 0 and we've seen intervals: Carry forward previous value
        - If y[i-1] == 0 and no intervals yet: Use 1 (avoid division by zero)
    """
    n = y.size
    out = jnp.full_like(y, jnp.nan)
    
    def body_fn(i, state):
        out_arr, fitted_idx = state
        
        # If previous value was non-zero, advance fitted index
        fitted_idx = jnp.where(
            y[i - 1] != 0,
            fitted_idx + 1,
            fitted_idx
        )
        
        # Determine output value based on conditions
        val = jax.lax.cond(
            y[i - 1] != 0,
            lambda: jnp.where(
                fitted[fitted_idx] == 0,
                1.0,  # Avoid division by zero
                fitted[fitted_idx]
            ),
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],  # Carry forward previous
                lambda: 1.0  # No intervals seen yet, use 1
            )
        )
        
        out_arr = out_arr.at[i].set(val)
        return out_arr, fitted_idx
    
    out, _ = jax.lax.fori_loop(1, n, body_fn, (out, 0))
    return out


@jax.jit
def _demand(x: jnp.ndarray) -> jnp.ndarray:
    """
    Extract positive (non-zero) elements from array.
    Used by Croston-family models for intermittent demand.
    
    Args:
        x: Input array
        
    Returns:
        Fixed-size array with non-zero values packed at start, rest filled with NaN
        
    Example:
        >>> x = jnp.array([0., 5., 0., 3., 0.])
        >>> _demand(x)
        array([5., 3., nan, nan, nan])
        
    Note:
        Returns fixed-size array (same size as input) for JIT compatibility.
        Non-zero values are packed at the start, remaining positions filled with NaN.
    """
    # Get indices where x > 0, with fill_value for padding
    indices = jnp.where(x > 0, size=x.size, fill_value=-1)[0]
    
    # Create result array: gather values where indices are valid, else NaN
    result = jnp.where(
        indices >= 0,
        jnp.where(indices < x.size, x[jnp.clip(indices, 0, x.size-1)], jnp.nan),
        jnp.nan
    )
    return result


@jax.jit
def _intervals(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute intervals between non-zero elements.
    Used by Croston-family models for intermittent demand.
    
    Args:
        x: Input array
        
    Returns:
        Fixed-size array with intervals packed at start, rest filled with NaN
        
    Example:
        >>> x = jnp.array([0., 5., 0., 0., 3., 0., 2.])
        >>> _intervals(x)
        array([1., 3., 2., nan, nan, nan, nan])  # First interval at position 1, then gap of 3, then gap of 2
        
    Note:
        Returns fixed-size array (same size as input) for JIT compatibility.
        Intervals are packed at the start, remaining positions filled with NaN.
    """
    # Get indices of non-zero elements, with fill_value for padding
    nonzero_idxs = jnp.where(x != 0, size=x.size, fill_value=-1)[0]
    
    # Compute positions (1-indexed)
    positions = jnp.where(nonzero_idxs >= 0, nonzero_idxs + 1, -1)
    
    # Compute intervals using diff with prepend
    intervals = jnp.diff(positions, prepend=0)
    
    # Mask out invalid intervals (where positions were -1)
    valid_mask = positions >= 0
    result = jnp.where(valid_mask, intervals.astype(x.dtype), jnp.nan)
    
    return result


def _expand_fitted_demand(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Expand demand fitted values back to original series length.
    Used by Croston-family models.
    
    Args:
        fitted: SES fitted values for demand (length = num_nonzero + 1)
        y: Original time series
        
    Returns:
        Fitted values expanded to match y's length
        
    Logic:
        - If y[i-1] > 0: Use next fitted value (demand occurred)
        - If y[i-1] == 0 and we've seen demand: Carry forward previous value
        - If y[i-1] == 0 and no demand yet: Use naive forecast (y[i-1])
    """
    n = y.size
    out = jnp.full_like(y, jnp.nan)
    
    def body_fn(i, state):
        out_arr, fitted_idx = state
        
        # If previous value was positive, advance fitted index
        fitted_idx = jnp.where(
            y[i - 1] > 0,
            fitted_idx + 1,
            fitted_idx
        )
        
        # Determine output value based on conditions
        val = jax.lax.cond(
            y[i - 1] > 0,
            lambda: fitted[fitted_idx],  # Use new fitted value
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],  # Carry forward previous
                lambda: y[i - 1]  # Use naive (no demand seen yet)
            )
        )
        
        out_arr = out_arr.at[i].set(val)
        return out_arr, fitted_idx
    
    out, _ = jax.lax.fori_loop(1, n, body_fn, (out, 0))
    return out


def _expand_fitted_intervals(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Expand interval fitted values back to original series length.
    Used by Croston-family models.
    
    Args:
        fitted: SES fitted values for intervals (length = num_nonzero + 1)
        y: Original time series
        
    Returns:
        Fitted intervals expanded to match y's length (avoids division by zero)
        
    Logic:
        - If y[i-1] != 0: Use next fitted value, but replace 0 with 1 (avoid div by zero)
        - If y[i-1] == 0 and we've seen intervals: Carry forward previous value
        - If y[i-1] == 0 and no intervals yet: Use 1 (avoid division by zero)
    """
    n = y.size
    out = jnp.full_like(y, jnp.nan)
    
    def body_fn(i, state):
        out_arr, fitted_idx = state
        
        # If previous value was non-zero, advance fitted index
        fitted_idx = jnp.where(
            y[i - 1] != 0,
            fitted_idx + 1,
            fitted_idx
        )
        
        # Determine output value based on conditions
        val = jax.lax.cond(
            y[i - 1] != 0,
            lambda: jnp.where(
                fitted[fitted_idx] == 0,
                1.0,  # Avoid division by zero
                fitted[fitted_idx]
            ),
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],  # Carry forward previous
                lambda: 1.0  # No intervals seen yet, use 1
            )
        )
        
        out_arr = out_arr.at[i].set(val)
        return out_arr, fitted_idx
    
    out, _ = jax.lax.fori_loop(1, n, body_fn, (out, 0))
    return out


@jax.jit
def _demand(x: jnp.ndarray) -> jnp.ndarray:
    """
    Extract positive (non-zero) elements from array.
    Used by Croston-family models for intermittent demand.
    
    Args:
        x: Input array
        
    Returns:
        Fixed-size array with non-zero values packed at start, rest filled with NaN
        
    Example:
        >>> x = jnp.array([0., 5., 0., 3., 0.])
        >>> _demand(x)
        array([5., 3., nan, nan, nan])
        
    Note:
        Returns fixed-size array (same size as input) for JIT compatibility.
        Non-zero values are packed at the start, remaining positions filled with NaN.
    """
    # Get indices where x > 0, with fill_value for padding
    indices = jnp.where(x > 0, size=x.size, fill_value=-1)[0]
    
    # Create result array: gather values where indices are valid, else NaN
    result = jnp.where(
        indices >= 0,
        jnp.where(indices < x.size, x[jnp.clip(indices, 0, x.size-1)], jnp.nan),
        jnp.nan
    )
    return result


@jax.jit
def _intervals(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute intervals between non-zero elements.
    Used by Croston-family models for intermittent demand.
    
    Args:
        x: Input array
        
    Returns:
        Fixed-size array with intervals packed at start, rest filled with NaN
        
    Example:
        >>> x = jnp.array([0., 5., 0., 0., 3., 0., 2.])
        >>> _intervals(x)
        array([1., 3., 2., nan, nan, nan, nan])  # First interval at position 1, then gap of 3, then gap of 2
        
    Note:
        Returns fixed-size array (same size as input) for JIT compatibility.
        Intervals are packed at the start, remaining positions filled with NaN.
    """
    # Get indices of non-zero elements, with fill_value for padding
    nonzero_idxs = jnp.where(x != 0, size=x.size, fill_value=-1)[0]
    
    # Compute positions (1-indexed)
    positions = jnp.where(nonzero_idxs >= 0, nonzero_idxs + 1, -1)
    
    # Compute intervals using diff with prepend
    intervals = jnp.diff(positions, prepend=0)
    
    # Mask out invalid intervals (where positions were -1)
    valid_mask = positions >= 0
    result = jnp.where(valid_mask, intervals.astype(x.dtype), jnp.nan)
    
    return result


def _expand_fitted_demand(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Expand demand fitted values back to original series length.
    Used by Croston-family models.
    
    Args:
        fitted: SES fitted values for demand (length = num_nonzero + 1)
        y: Original time series
        
    Returns:
        Fitted values expanded to match y's length
        
    Logic:
        - If y[i-1] > 0: Use next fitted value (demand occurred)
        - If y[i-1] == 0 and we've seen demand: Carry forward previous value
        - If y[i-1] == 0 and no demand yet: Use naive forecast (y[i-1])
    """
    n = y.size
    out = jnp.full_like(y, jnp.nan)
    
    def body_fn(i, state):
        out_arr, fitted_idx = state
        
        # If previous value was positive, advance fitted index
        fitted_idx = jnp.where(
            y[i - 1] > 0,
            fitted_idx + 1,
            fitted_idx
        )
        
        # Determine output value based on conditions
        val = jax.lax.cond(
            y[i - 1] > 0,
            lambda: fitted[fitted_idx],  # Use new fitted value
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],  # Carry forward previous
                lambda: y[i - 1]  # Use naive (no demand seen yet)
            )
        )
        
        out_arr = out_arr.at[i].set(val)
        return out_arr, fitted_idx
    
    out, _ = jax.lax.fori_loop(1, n, body_fn, (out, 0))
    return out


def _expand_fitted_intervals(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Expand interval fitted values back to original series length.
    Used by Croston-family models.
    
    Args:
        fitted: SES fitted values for intervals (length = num_nonzero + 1)
        y: Original time series
        
    Returns:
        Fitted intervals expanded to match y's length (avoids division by zero)
        
    Logic:
        - If y[i-1] != 0: Use next fitted value, but replace 0 with 1 (avoid div by zero)
        - If y[i-1] == 0 and we've seen intervals: Carry forward previous value
        - If y[i-1] == 0 and no intervals yet: Use 1 (avoid division by zero)
    """
    n = y.size
    out = jnp.full_like(y, jnp.nan)
    
    def body_fn(i, state):
        out_arr, fitted_idx = state
        
        # If previous value was non-zero, advance fitted index
        fitted_idx = jnp.where(
            y[i - 1] != 0,
            fitted_idx + 1,
            fitted_idx
        )
        
        # Determine output value based on conditions
        val = jax.lax.cond(
            y[i - 1] != 0,
            lambda: jnp.where(
                fitted[fitted_idx] == 0,
                1.0,  # Avoid division by zero
                fitted[fitted_idx]
            ),
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],  # Carry forward previous
                lambda: 1.0  # No intervals seen yet, use 1
            )
        )
        
        out_arr = out_arr.at[i].set(val)
        return out_arr, fitted_idx
    
    out, _ = jax.lax.fori_loop(1, n, body_fn, (out, 0))
    return out

def _conformal_method(self):
        return _get_conformal_method(self.prediction_intervals.method)

def _store_cs(self, y, X):
    if self.prediction_intervals is not None:
        self._cs = self.conformity_scores(y, X)

def _add_conformal_intervals(self, fcst, y, X, level):
    if self.prediction_intervals is not None and level is not None:
        cs = self.conformity_scores(y, X) if y is not None else self._cs
        conformal_fn = _conformal_method(self)
        res = conformal_fn(fcst=fcst, cs=cs, level=level)
        return res
    return fcst

def _add_predict_conformal_intervals(self, fcst, level):
    from utils import _add_conformal_intervals
    return _add_conformal_intervals(self,fcst=fcst, y=None, X=None, level=level)

def switch_theta(model: str) -> _theta.ModelType:
    if model == "STM":
        return _theta.ModelType.STM
    if model == "OTM":
        return _theta.ModelType.OTM
    if model == "DSTM":
        return _theta.ModelType.DSTM
    if model == "DOTM":
        return _theta.ModelType.DOTM
    raise ValueError(f"Invalid model type: {model}.")

def compute_pi_samples(n, h, states, sigma, alpha, theta, mean_y, seed=0, n_samples=200):
    """
    Compute forecast samples for conformal intervals in JAX.
    """
    samples = jnp.full((h, n_samples), jnp.nan, dtype=jnp.float32)

    # Unpack last state: level, meany, An, Bn
    level, meany, A, B = states[-1, :4]
    smoothed = level

    # Initialize PRNG key
    key = jrandom.PRNGKey(seed)

    def body_fun(i, val):
        smoothed, mean_y, A, B, samples, key = val
        # deterministic part
        mu = smoothed + (1 - 1 / theta) * (A * ((1 - alpha) ** i) + B * (1 - (1 - alpha) ** (i + 1)) / alpha)

        # random noise
        key, subkey = jrandom.split(key)
        eps = jrandom.normal(subkey, shape=(n_samples,), dtype=jnp.float32) * sigma

        # sample for this step
        s = mu + eps

        # update smoothed, mean, A, B
        smoothed_new = alpha * jnp.mean(s) + (1 - alpha) * smoothed
        mean_y_new = (i * mean_y + jnp.mean(s)) / (i + 1)
        B_new = ((i - 1) * B + 6 * (jnp.mean(s) - mean_y_new) / (i + 1)) / (i + 2)
        A_new = mean_y_new - B_new * (i + 2) / 2

        samples = samples.at[i - n].set(s)
        return smoothed_new, mean_y_new, A_new, B_new, samples, key

    # Loop over forecast horizon
    smoothed, mean_y, A, B, samples, key = jax.lax.fori_loop(
        n, n + h, body_fun, (smoothed, mean_y, A, B, samples, key)
    )

    return samples

def initparamtheta(
    initial_smoothed: float,
    alpha: float,
    theta: float,
    y: jnp.ndarray,
    modeltype: _theta.ModelType,
):
    if modeltype in [_theta.ModelType.STM, _theta.ModelType.DSTM]:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        if math.isnan(alpha):
            alpha = 0.5
            optimize_alpha = True
        else:
            optimize_alpha = False
        theta = 2.0  # no optimize
        optimize_theta = False
    else:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        if math.isnan(alpha):
            alpha = 0.5
            optimize_alpha = True
        else:
            optimize_alpha = False
        if math.isnan(theta):
            theta = 2.0
            optimize_theta = True
        else:
            optimize_theta = False
    return {
        "initial_smoothed": initial_smoothed,
        "optimize_initial_smoothed": optimize_level,
        "alpha": alpha,
        "optimize_alpha": optimize_alpha,
        "theta": theta,
        "optimize_theta": optimize_theta,
    }

def optimize_theta_target_fn(
    init_par,
    lower,
    upper,
    init_level,
    init_alpha,
    init_theta,
    opt_level,
    opt_alpha,
    opt_theta,
    y,
    modeltype,
    nmse,
):
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

    # ✅ Make sure these are NumPy scalars, not JAX DeviceArrays
    x = jax.device_get(opt_res["x"])
    fn = float(jax.device_get(opt_res["fun"]))
    nit = int(opt_res.get("nit", 0))

    results = namedtuple("results", "x fn nit simplex")
    return results(x, fn, nit, None)

# Initializations
init_level = 0.5
init_alpha = 0.3
init_theta = 2.0

opt_level = True
opt_alpha = True
opt_theta = True

lower = jnp.array([0.0, 0.0, 1.0])
upper = jnp.array([1.0, 1.0, 10.0])

# Initial guess vector (x0)
par = jnp.array([init_level, init_alpha, init_theta])

def thetamodel(
    y: jnp.ndarray,
    m: int,
    modeltype: str,
    initial_smoothed: float,
    alpha: float,
    theta: float,
    nmse: int,
):
    y = y.astype(jnp.float64, copy=False)
    model_type = switch_theta(modeltype)
    # initial parameters
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
    # parameter optimization
    fred = optimize_theta_target_fn(
    init_par=x0,
    lower=lower,
    upper=upper,
    init_level=init_level,
    init_alpha=init_alpha,
    init_theta=init_theta,
    opt_level=opt_level,
    opt_alpha=opt_alpha,
    opt_theta=opt_theta,
    y=y,
    modeltype=model_type,
    nmse=nmse,
)

    if fred is not None:
        fit_par = fred.x

    j = 0
    if optimize_params.get("initial_smoothed", False):
        par["initial_smoothed"] = float(fit_par[j])
        j += 1
    if optimize_params.get("alpha", False):
        par["alpha"] = float(fit_par[j])
        j += 1
    if optimize_params.get("theta", False):
        par["theta"] = float(fit_par[j])
        j += 1

    amse, e, states, mse = _theta.pegels_resid(
        y,
        model_type,
        par["initial_smoothed"],
        par["alpha"],
        par["theta"],
        nmse,
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

def forecast_theta(obj, h, level=None):
    # Extract parameters
    n_obs = obj["n"]  # number of observed steps
    # Initialize states if not already correct shape
    states = obj["states"]
    if states.ndim == 1 or states.shape[1] != 4:
        # assume 1D array, reshape/pad to (n_obs, 4)
        states = jnp.zeros((n_obs, 5), dtype=jnp.float32)
        # optionally fill initial level/mean/An/Bn
        states = states.at[:, 0].set(obj.get("mean_y", 0.0))   # level
        states = states.at[:, 1].set(obj.get("mean_y", 0.0))   # mean
        states = states.at[:, 2].set(0.0)                      # An
        states = states.at[:, 3].set(0.0)                      # Bn
        states = states.at[:, 4].set(0.0)     

    alpha = obj["par"]["alpha"]
    theta = obj["par"]["theta"]
    model_type = switch_theta(obj["modeltype"])

    # Call the JAX forecast function
    states, forecast = _theta.forecast(states, h, model_type, alpha, theta)

    # Build result dictionary
    res = {"mean": forecast}

    # Compute prediction intervals if requested
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

    # Recompose if seasonal decomposition was used
    if obj.get("decompose", False):
        seas_forecast = _repeat_val_seas(obj["seas_forecast"]["mean"], h=h)
        for key in res:
            if obj["decomposition_type"] == "multiplicative":
                res[key] = res[key] * seas_forecast
            else:
                res[key] = res[key] + seas_forecast

    return res

def is_constant(x):
    return jnp.all(x[0] == x)

def seasonal_decompose(y: jnp.ndarray, model: str = "additive", period: int = 1):
    """
    Simple seasonal decomposition using moving average.
    Returns a dict with 'trend', 'seasonal', and 'resid' like statsmodels.
    """
    n = len(y)
    # Moving average for trend
    kernel = jnp.ones(period) / period
    trend = jnp.convolve(y, kernel, mode="same")

    if model == "additive":
        detrended = y - trend
        seasonal = jnp.tile(
            jnp.mean(detrended.reshape(-1, period), axis=0), n // period + 1
        )[:n]
        resid = detrended - seasonal
    else:  # multiplicative
        detrended = y / trend
        seasonal = jnp.tile(
            jnp.mean(detrended.reshape(-1, period), axis=0), n // period + 1
        )[:n]
        resid = detrended / seasonal

    return {
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid,
    }

def acf(x: jnp.ndarray, nlags: int) -> jnp.ndarray:
    """
    Compute autocorrelation function up to `nlags` for 1D array x using JAX.
    Equivalent to statsmodels.tsa.stattools.acf(x, nlags=nlags, fft=False).
    """
    x = x - jnp.mean(x)
    n = x.shape[0]
    denom = jnp.dot(x, x)
    acf_vals = jnp.array([jnp.dot(x[: n - lag], x[lag:]) / denom for lag in range(nlags + 1)])
    return acf_vals

def auto_theta(
    y,
    m,
    model=None,
    initial_smoothed=None,
    alpha=None,
    theta=None,
    nmse=3,
    decomposition_type="multiplicative",
):
    # converting params to floats
    # to improve numba compilation
    if initial_smoothed is None:
        initial_smoothed = jnp.nan
    if alpha is None:
        alpha = jnp.nan
    if theta is None:
        theta = jnp.nan
    if nmse < 1 or nmse > 30:
        raise ValueError("nmse out of range")
    # constan values
    if is_constant(y):
        thetamodel(
            y=y,
            m=m,
            modeltype="STM",
            nmse=nmse,
            initial_smoothed=jnp.mean(y) / 2,
            alpha=0.5,
            theta=2.0,
        )
    # seasonal decomposition if needed
    decompose = False
    # seasonal test
    if m >= 4 and len(y) >= 2 * m:
        r = acf(y, nlags=m, fft=False)[1:]
        stat = jnp.sqrt((1 + 2 * jnp.sum(r[:-1] ** 2)) / len(y))
        decompose = jnp.abs(r[-1]) / stat > norm.ppf(0.95)

    data_positive = min(y) > 0
    if decompose:
        # change decomposition type if data is not positive
        if decomposition_type == "multiplicative" and not data_positive:
            decomposition_type = "additive"
        y_decompose = seasonal_decompose(y, model=decomposition_type, period=m).seasonal
        if decomposition_type == "multiplicative" and any(y_decompose < 0.01):
            decomposition_type = "additive"
            y_decompose = seasonal_decompose(y, model="additive", period=m).seasonal
        if decomposition_type == "additive":
            y = y - y_decompose
        else:
            y = y / y_decompose
        seas_forecast = _seasonal_naive(
            y=y_decompose, h=m, season_length=m, fitted=False
        )

    # validate model
    if model not in [None, "STM", "OTM", "DSTM", "DOTM"]:
        raise ValueError(f"Invalid model type: {model}.")

    n = len(y)
    npars = 3
    # non-optimized tiny datasets
    if n <= npars:
        raise NotImplementedError("tiny datasets")
    if model is None:
        modeltype = ["STM", "OTM", "DSTM", "DOTM"]
    else:
        modeltype = [model]

    best_ic = jnp.inf
    for mtype in modeltype:
        fit = thetamodel(
            y=y,
            m=m,
            modeltype=mtype,
            nmse=nmse,
            initial_smoothed=initial_smoothed,
            alpha=alpha,
            theta=theta,
        )
        fit_ic = fit["mse"]
        if not jnp.isnan(fit_ic):
            if fit_ic < best_ic:
                model = fit
                best_ic = fit_ic
    if jnp.isinf(best_ic):
        raise Exception("no model able to be fitted")

    if decompose:
        if decomposition_type == "multiplicative":
            model["residuals"] = model["residuals"] * y_decompose
        else:
            model["residuals"] = model["residuals"] + y_decompose
        model["decompose"] = decompose
        model["decomposition_type"] = decomposition_type
        model["seas_forecast"] = dict(seas_forecast)
    return model


def forward_theta(fitted_model, y):
    m = fitted_model["m"]
    model = fitted_model["modeltype"]
    initial_smoothed = fitted_model["par"]["initial_smoothed"]
    alpha = fitted_model["par"]["alpha"]
    theta = fitted_model["par"]["theta"]
    return auto_theta(
        y=y,
        m=m,
        model=model,
        initial_smoothed=initial_smoothed,
        alpha=alpha,
        theta=theta,
    )
    return _add_conformal_intervals(self, fcst=fcst, y=None, X=None, level=level)

def _intervals(y: jnp.ndarray) -> jnp.ndarray:
    """
    Return intervals between non-zero observations as a float32 array of length len(y).
    The valid intervals are placed at the front of the returned array and the rest
    are padded with 0.0 to ensure a stable shape/dtype for JAX control-flow.
    """
    n = y.size
    nz_idx = jnp.where(y != 0)[0]  # indices of non-zero entries (int32)

    def no_nz():
        # no non-zero values: return zero-padded float32 array length n
        return jnp.zeros((n,), dtype=jnp.float32)

    def some_nz():
        # diffs are integers; cast to float32 and pad with zeros up to length n
        diffs = jnp.diff(nz_idx).astype(jnp.float32)  # shape (k-1,) if k>=1 else (0,)
        k = diffs.size
        pad_len = n - k
        pad = jnp.zeros((pad_len,), dtype=jnp.float32)
        return jnp.concatenate([diffs, pad], axis=0)

    return lax.cond(nz_idx.size == 0, no_nz, some_nz)


def _chunk_sums(array: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    r"""Splits an array into chunks and returns the sum of each chunk.
    Incomplete chunks are discarded"""
    n_chunks = array.size // chunk_size
    n_elems = n_chunks * chunk_size
    return array[:n_elems].reshape(n_chunks, chunk_size).sum(axis=1)

@jit
def _ses_sse(alpha: float, x: jnp.ndarray) -> float:
    r"""Compute the residual sum of squares for a simple exponential smoothing fit.

    Args:
        alpha (float): Smoothing parameter.
        x (numpy.array): Clean time series of shape (n, ).

    Returns:
        sse (float): Residual sum of squares for the fit.
    """
    complement = 1 - alpha
    forecast = x[0]
    sse = 0.0

    for i in range(1, len(x)):
        forecast = alpha * x[i - 1] + complement * forecast
        sse += (x[i] - forecast) ** 2

    return sse

def _optimized_ses_forecast(
    x: jnp.ndarray, bounds: Tuple[float, float] = (0.1, 0.3), n_grid: int = 50
) -> Tuple[float, jnp.ndarray]:
    alphas = jnp.linspace(bounds[0], bounds[1], n_grid)

    def sse_for_alpha(alpha):
        return _ses_sse(alpha, x)

    sses = jax.vmap(sse_for_alpha)(alphas)
    best_idx = jnp.argmin(sses)
    best_alpha = alphas[best_idx]

    forecast, fitted = _ses_forecast(x, best_alpha)
    return forecast, fitted

def _chunk_forecast(y, aggregation_level):
    lost_remainder_data = len(y) % aggregation_level
    y_cut = y[lost_remainder_data:]
    aggregation_sums = _chunk_sums(y_cut, aggregation_level)
    sums_forecast, _ = _optimized_ses_forecast(aggregation_sums)
    return sums_forecast

@jit
# def _expand_fitted_intervals(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
def _expand_fitted_intervals_c(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    out = jnp.empty_like(y)
    # out[0] = jnp.nan
    out = out.at[0].set(jnp.nan)
    fitted_idx = 0
    for i in range(1, y.size):
        if y[i - 1] != 0:
            fitted_idx += 1
            if fitted[fitted_idx] == 0:
                # to avoid division by zero
                out[i] = 1
            else:
                out[i] = fitted[fitted_idx]
        elif fitted_idx > 0:
            # if this entry is zero, the model didn't change
            out[i] = out[i - 1]
        else:
            # if we haven't seen any intervals, use 1 to avoid division by zero
            out[i] = 1
    return out
    from utils import _add_conformal_intervals
    return _add_conformal_intervals(self,fcst=fcst, y=None, X=None, level=level)

def switch_theta(model: str) -> _theta.ModelType:
    if model == "STM":
        return _theta.ModelType.STM
    if model == "OTM":
        return _theta.ModelType.OTM
    if model == "DSTM":
        return _theta.ModelType.DSTM
    if model == "DOTM":
        return _theta.ModelType.DOTM
    raise ValueError(f"Invalid model type: {model}.")

def compute_pi_samples(n, h, states, sigma, alpha, theta, mean_y, seed=0, n_samples=200):
    """
    Compute forecast samples for conformal intervals in JAX.
    """
    samples = jnp.full((h, n_samples), jnp.nan, dtype=jnp.float32)

    # Unpack last state: level, meany, An, Bn
    level, meany, A, B = states[-1, :4]
    smoothed = level

    # Initialize PRNG key
    key = jrandom.PRNGKey(seed)

    def body_fun(i, val):
        smoothed, mean_y, A, B, samples, key = val
        # deterministic part
        mu = smoothed + (1 - 1 / theta) * (A * ((1 - alpha) ** i) + B * (1 - (1 - alpha) ** (i + 1)) / alpha)

        # random noise
        key, subkey = jrandom.split(key)
        eps = jrandom.normal(subkey, shape=(n_samples,), dtype=jnp.float32) * sigma

        # sample for this step
        s = mu + eps

        # update smoothed, mean, A, B
        smoothed_new = alpha * jnp.mean(s) + (1 - alpha) * smoothed
        mean_y_new = (i * mean_y + jnp.mean(s)) / (i + 1)
        B_new = ((i - 1) * B + 6 * (jnp.mean(s) - mean_y_new) / (i + 1)) / (i + 2)
        A_new = mean_y_new - B_new * (i + 2) / 2

        samples = samples.at[i - n].set(s)
        return smoothed_new, mean_y_new, A_new, B_new, samples, key

    # Loop over forecast horizon
    smoothed, mean_y, A, B, samples, key = jax.lax.fori_loop(
        n, n + h, body_fun, (smoothed, mean_y, A, B, samples, key)
    )

    return samples

def initparamtheta(
    initial_smoothed: float,
    alpha: float,
    theta: float,
    y: jnp.ndarray,
    modeltype: _theta.ModelType,
):
    if modeltype in [_theta.ModelType.STM, _theta.ModelType.DSTM]:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        if math.isnan(alpha):
            alpha = 0.5
            optimize_alpha = True
        else:
            optimize_alpha = False
        theta = 2.0  # no optimize
        optimize_theta = False
    else:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        if math.isnan(alpha):
            alpha = 0.5
            optimize_alpha = True
        else:
            optimize_alpha = False
        if math.isnan(theta):
            theta = 2.0
            optimize_theta = True
        else:
            optimize_theta = False
    return {
        "initial_smoothed": initial_smoothed,
        "optimize_initial_smoothed": optimize_level,
        "alpha": alpha,
        "optimize_alpha": optimize_alpha,
        "theta": theta,
        "optimize_theta": optimize_theta,
    }

def optimize_theta_target_fn(
    init_par,
    lower,
    upper,
    init_level,
    init_alpha,
    init_theta,
    opt_level,
    opt_alpha,
    opt_theta,
    y,
    modeltype,
    nmse,
):
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

    # ✅ Make sure these are NumPy scalars, not JAX DeviceArrays
    x = jax.device_get(opt_res["x"])
    fn = float(jax.device_get(opt_res["fun"]))
    nit = int(opt_res.get("nit", 0))

    results = namedtuple("results", "x fn nit simplex")
    return results(x, fn, nit, None)

# Initializations
init_level = 0.5
init_alpha = 0.3
init_theta = 2.0

opt_level = True
opt_alpha = True
opt_theta = True

lower = jnp.array([0.0, 0.0, 1.0])
upper = jnp.array([1.0, 1.0, 10.0])

# Initial guess vector (x0)
par = jnp.array([init_level, init_alpha, init_theta])

def thetamodel(
    y: jnp.ndarray,
    m: int,
    modeltype: str,
    initial_smoothed: float,
    alpha: float,
    theta: float,
    nmse: int,
):
    y = y.astype(jnp.float64, copy=False)
    model_type = switch_theta(modeltype)
    # initial parameters
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
    # parameter optimization
    fred = optimize_theta_target_fn(
    init_par=x0,
    lower=lower,
    upper=upper,
    init_level=init_level,
    init_alpha=init_alpha,
    init_theta=init_theta,
    opt_level=opt_level,
    opt_alpha=opt_alpha,
    opt_theta=opt_theta,
    y=y,
    modeltype=model_type,
    nmse=nmse,
)

    if fred is not None:
        fit_par = fred.x

    j = 0
    if optimize_params.get("initial_smoothed", False):
        par["initial_smoothed"] = float(fit_par[j])
        j += 1
    if optimize_params.get("alpha", False):
        par["alpha"] = float(fit_par[j])
        j += 1
    if optimize_params.get("theta", False):
        par["theta"] = float(fit_par[j])
        j += 1

    amse, e, states, mse = _theta.pegels_resid(
        y,
        model_type,
        par["initial_smoothed"],
        par["alpha"],
        par["theta"],
        nmse,
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

def forecast_theta(obj, h, level=None):
    # Extract parameters
    n_obs = obj["n"]  # number of observed steps
    # Initialize states if not already correct shape
    states = obj["states"]
    if states.ndim == 1 or states.shape[1] != 4:
        # assume 1D array, reshape/pad to (n_obs, 4)
        states = jnp.zeros((n_obs, 5), dtype=jnp.float32)
        # optionally fill initial level/mean/An/Bn
        states = states.at[:, 0].set(obj.get("mean_y", 0.0))   # level
        states = states.at[:, 1].set(obj.get("mean_y", 0.0))   # mean
        states = states.at[:, 2].set(0.0)                      # An
        states = states.at[:, 3].set(0.0)                      # Bn
        states = states.at[:, 4].set(0.0)     

    alpha = obj["par"]["alpha"]
    theta = obj["par"]["theta"]
    model_type = switch_theta(obj["modeltype"])

    # Call the JAX forecast function
    states, forecast = _theta.forecast(states, h, model_type, alpha, theta)

    # Build result dictionary
    res = {"mean": forecast}

    # Compute prediction intervals if requested
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

    # Recompose if seasonal decomposition was used
    if obj.get("decompose", False):
        seas_forecast = _repeat_val_seas(obj["seas_forecast"]["mean"], h=h)
        for key in res:
            if obj["decomposition_type"] == "multiplicative":
                res[key] = res[key] * seas_forecast
            else:
                res[key] = res[key] + seas_forecast

    return res

def is_constant(x):
    return jnp.all(x[0] == x)

def seasonal_decompose(y: jnp.ndarray, model: str = "additive", period: int = 1):
    """
    Simple seasonal decomposition using moving average.
    Returns a dict with 'trend', 'seasonal', and 'resid' like statsmodels.
    """
    n = len(y)
    # Moving average for trend
    kernel = jnp.ones(period) / period
    trend = jnp.convolve(y, kernel, mode="same")

    if model == "additive":
        detrended = y - trend
        seasonal = jnp.tile(
            jnp.mean(detrended.reshape(-1, period), axis=0), n // period + 1
        )[:n]
        resid = detrended - seasonal
    else:  # multiplicative
        detrended = y / trend
        seasonal = jnp.tile(
            jnp.mean(detrended.reshape(-1, period), axis=0), n // period + 1
        )[:n]
        resid = detrended / seasonal

    return {
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid,
    }

def acf(x: jnp.ndarray, nlags: int) -> jnp.ndarray:
    """
    Compute autocorrelation function up to `nlags` for 1D array x using JAX.
    Equivalent to statsmodels.tsa.stattools.acf(x, nlags=nlags, fft=False).
    """
    x = x - jnp.mean(x)
    n = x.shape[0]
    denom = jnp.dot(x, x)
    acf_vals = jnp.array([jnp.dot(x[: n - lag], x[lag:]) / denom for lag in range(nlags + 1)])
    return acf_vals

def auto_theta(
    y,
    m,
    model=None,
    initial_smoothed=None,
    alpha=None,
    theta=None,
    nmse=3,
    decomposition_type="multiplicative",
):
    # converting params to floats
    # to improve numba compilation
    if initial_smoothed is None:
        initial_smoothed = jnp.nan
    if alpha is None:
        alpha = jnp.nan
    if theta is None:
        theta = jnp.nan
    if nmse < 1 or nmse > 30:
        raise ValueError("nmse out of range")
    # constan values
    if is_constant(y):
        thetamodel(
            y=y,
            m=m,
            modeltype="STM",
            nmse=nmse,
            initial_smoothed=jnp.mean(y) / 2,
            alpha=0.5,
            theta=2.0,
        )
    # seasonal decomposition if needed
    decompose = False
    # seasonal test
    if m >= 4 and len(y) >= 2 * m:
        r = acf(y, nlags=m, fft=False)[1:]
        stat = jnp.sqrt((1 + 2 * jnp.sum(r[:-1] ** 2)) / len(y))
        decompose = jnp.abs(r[-1]) / stat > norm.ppf(0.95)

    data_positive = min(y) > 0
    if decompose:
        # change decomposition type if data is not positive
        if decomposition_type == "multiplicative" and not data_positive:
            decomposition_type = "additive"
        y_decompose = seasonal_decompose(y, model=decomposition_type, period=m).seasonal
        if decomposition_type == "multiplicative" and any(y_decompose < 0.01):
            decomposition_type = "additive"
            y_decompose = seasonal_decompose(y, model="additive", period=m).seasonal
        if decomposition_type == "additive":
            y = y - y_decompose
        else:
            y = y / y_decompose
        seas_forecast = _seasonal_naive(
            y=y_decompose, h=m, season_length=m, fitted=False
        )

    # validate model
    if model not in [None, "STM", "OTM", "DSTM", "DOTM"]:
        raise ValueError(f"Invalid model type: {model}.")

    n = len(y)
    npars = 3
    # non-optimized tiny datasets
    if n <= npars:
        raise NotImplementedError("tiny datasets")
    if model is None:
        modeltype = ["STM", "OTM", "DSTM", "DOTM"]
    else:
        modeltype = [model]

    best_ic = jnp.inf
    for mtype in modeltype:
        fit = thetamodel(
            y=y,
            m=m,
            modeltype=mtype,
            nmse=nmse,
            initial_smoothed=initial_smoothed,
            alpha=alpha,
            theta=theta,
        )
        fit_ic = fit["mse"]
        if not jnp.isnan(fit_ic):
            if fit_ic < best_ic:
                model = fit
                best_ic = fit_ic
    if jnp.isinf(best_ic):
        raise Exception("no model able to be fitted")

    if decompose:
        if decomposition_type == "multiplicative":
            model["residuals"] = model["residuals"] * y_decompose
        else:
            model["residuals"] = model["residuals"] + y_decompose
        model["decompose"] = decompose
        model["decomposition_type"] = decomposition_type
        model["seas_forecast"] = dict(seas_forecast)
    return model


def forward_theta(fitted_model, y):
    m = fitted_model["m"]
    model = fitted_model["modeltype"]
    initial_smoothed = fitted_model["par"]["initial_smoothed"]
    alpha = fitted_model["par"]["alpha"]
    theta = fitted_model["par"]["theta"]
    return auto_theta(
        y=y,
        m=m,
        model=model,
        initial_smoothed=initial_smoothed,
        alpha=alpha,
        theta=theta,
    )
    return _add_conformal_intervals(self, fcst=fcst, y=None, X=None, level=level)

def _intervals(y: jnp.ndarray) -> jnp.ndarray:
    """
    Return intervals between non-zero observations as a float32 array of length len(y).
    The valid intervals are placed at the front of the returned array and the rest
    are padded with 0.0 to ensure a stable shape/dtype for JAX control-flow.
    """
    n = y.size
    nz_idx = jnp.where(y != 0)[0]  # indices of non-zero entries (int32)

    def no_nz():
        # no non-zero values: return zero-padded float32 array length n
        return jnp.zeros((n,), dtype=jnp.float32)

    def some_nz():
        # diffs are integers; cast to float32 and pad with zeros up to length n
        diffs = jnp.diff(nz_idx).astype(jnp.float32)  # shape (k-1,) if k>=1 else (0,)
        k = diffs.size
        pad_len = n - k
        pad = jnp.zeros((pad_len,), dtype=jnp.float32)
        return jnp.concatenate([diffs, pad], axis=0)

    return lax.cond(nz_idx.size == 0, no_nz, some_nz)


def _chunk_sums(array: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    r"""Splits an array into chunks and returns the sum of each chunk.
    Incomplete chunks are discarded"""
    n_chunks = array.size // chunk_size
    n_elems = n_chunks * chunk_size
    return array[:n_elems].reshape(n_chunks, chunk_size).sum(axis=1)

@jit
def _ses_sse(alpha: float, x: jnp.ndarray) -> float:
    r"""Compute the residual sum of squares for a simple exponential smoothing fit.

    Args:
        alpha (float): Smoothing parameter.
        x (numpy.array): Clean time series of shape (n, ).

    Returns:
        sse (float): Residual sum of squares for the fit.
    """
    complement = 1 - alpha
    forecast = x[0]
    sse = 0.0

    for i in range(1, len(x)):
        forecast = alpha * x[i - 1] + complement * forecast
        sse += (x[i] - forecast) ** 2

    return sse

def _optimized_ses_forecast(
    x: jnp.ndarray, bounds: Tuple[float, float] = (0.1, 0.3), n_grid: int = 50
) -> Tuple[float, jnp.ndarray]:
    alphas = jnp.linspace(bounds[0], bounds[1], n_grid)

    def sse_for_alpha(alpha):
        return _ses_sse(alpha, x)

    sses = jax.vmap(sse_for_alpha)(alphas)
    best_idx = jnp.argmin(sses)
    best_alpha = alphas[best_idx]

    forecast, fitted = _ses_forecast(x, best_alpha)
    return forecast, fitted

def _chunk_forecast(y, aggregation_level):
    lost_remainder_data = len(y) % aggregation_level
    y_cut = y[lost_remainder_data:]
    aggregation_sums = _chunk_sums(y_cut, aggregation_level)
    sums_forecast, _ = _optimized_ses_forecast(aggregation_sums)
    return sums_forecast

@jit
# def _expand_fitted_intervals(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
def _expand_fitted_intervals_c(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    out = jnp.empty_like(y)
    # out[0] = jnp.nan
    out = out.at[0].set(jnp.nan)
    fitted_idx = 0
    for i in range(1, y.size):
        if y[i - 1] != 0:
            fitted_idx += 1
            if fitted[fitted_idx] == 0:
                # to avoid division by zero
                out[i] = 1
            else:
                out[i] = fitted[fitted_idx]
        elif fitted_idx > 0:
            # if this entry is zero, the model didn't change
            out[i] = out[i - 1]
        else:
            # if we haven't seen any intervals, use 1 to avoid division by zero
            out[i] = 1
    return out
    from utils import _add_conformal_intervals
    return _add_conformal_intervals(self,fcst=fcst, y=None, X=None, level=level)

def switch_theta(model: str) -> _theta.ModelType:
    if model == "STM":
        return _theta.ModelType.STM
    if model == "OTM":
        return _theta.ModelType.OTM
    if model == "DSTM":
        return _theta.ModelType.DSTM
    if model == "DOTM":
        return _theta.ModelType.DOTM
    raise ValueError(f"Invalid model type: {model}.")

def compute_pi_samples(n, h, states, sigma, alpha, theta, mean_y, seed=0, n_samples=200):
    """
    Compute forecast samples for conformal intervals in JAX.
    """
    samples = jnp.full((h, n_samples), jnp.nan, dtype=jnp.float32)

    # Unpack last state: level, meany, An, Bn
    level, meany, A, B = states[-1, :4]
    smoothed = level

    # Initialize PRNG key
    key = jrandom.PRNGKey(seed)

    def body_fun(i, val):
        smoothed, mean_y, A, B, samples, key = val
        # deterministic part
        mu = smoothed + (1 - 1 / theta) * (A * ((1 - alpha) ** i) + B * (1 - (1 - alpha) ** (i + 1)) / alpha)

        # random noise
        key, subkey = jrandom.split(key)
        eps = jrandom.normal(subkey, shape=(n_samples,), dtype=jnp.float32) * sigma

        # sample for this step
        s = mu + eps

        # update smoothed, mean, A, B
        smoothed_new = alpha * jnp.mean(s) + (1 - alpha) * smoothed
        mean_y_new = (i * mean_y + jnp.mean(s)) / (i + 1)
        B_new = ((i - 1) * B + 6 * (jnp.mean(s) - mean_y_new) / (i + 1)) / (i + 2)
        A_new = mean_y_new - B_new * (i + 2) / 2

        samples = samples.at[i - n].set(s)
        return smoothed_new, mean_y_new, A_new, B_new, samples, key

    # Loop over forecast horizon
    smoothed, mean_y, A, B, samples, key = jax.lax.fori_loop(
        n, n + h, body_fun, (smoothed, mean_y, A, B, samples, key)
    )

    return samples

def initparamtheta(
    initial_smoothed: float,
    alpha: float,
    theta: float,
    y: jnp.ndarray,
    modeltype: _theta.ModelType,
):
    if modeltype in [_theta.ModelType.STM, _theta.ModelType.DSTM]:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        if math.isnan(alpha):
            alpha = 0.5
            optimize_alpha = True
        else:
            optimize_alpha = False
        theta = 2.0  # no optimize
        optimize_theta = False
    else:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        if math.isnan(alpha):
            alpha = 0.5
            optimize_alpha = True
        else:
            optimize_alpha = False
        if math.isnan(theta):
            theta = 2.0
            optimize_theta = True
        else:
            optimize_theta = False
    return {
        "initial_smoothed": initial_smoothed,
        "optimize_initial_smoothed": optimize_level,
        "alpha": alpha,
        "optimize_alpha": optimize_alpha,
        "theta": theta,
        "optimize_theta": optimize_theta,
    }

def optimize_theta_target_fn(
    init_par,
    lower,
    upper,
    init_level,
    init_alpha,
    init_theta,
    opt_level,
    opt_alpha,
    opt_theta,
    y,
    modeltype,
    nmse,
):
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

    # ✅ Make sure these are NumPy scalars, not JAX DeviceArrays
    x = jax.device_get(opt_res["x"])
    fn = float(jax.device_get(opt_res["fun"]))
    nit = int(opt_res.get("nit", 0))

    results = namedtuple("results", "x fn nit simplex")
    return results(x, fn, nit, None)

# Initializations
init_level = 0.5
init_alpha = 0.3
init_theta = 2.0

opt_level = True
opt_alpha = True
opt_theta = True

lower = jnp.array([0.0, 0.0, 1.0])
upper = jnp.array([1.0, 1.0, 10.0])

# Initial guess vector (x0)
par = jnp.array([init_level, init_alpha, init_theta])

def thetamodel(
    y: jnp.ndarray,
    m: int,
    modeltype: str,
    initial_smoothed: float,
    alpha: float,
    theta: float,
    nmse: int,
):
    y = y.astype(jnp.float64, copy=False)
    model_type = switch_theta(modeltype)
    # initial parameters
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
    # parameter optimization
    fred = optimize_theta_target_fn(
    init_par=x0,
    lower=lower,
    upper=upper,
    init_level=init_level,
    init_alpha=init_alpha,
    init_theta=init_theta,
    opt_level=opt_level,
    opt_alpha=opt_alpha,
    opt_theta=opt_theta,
    y=y,
    modeltype=model_type,
    nmse=nmse,
)

    if fred is not None:
        fit_par = fred.x

    j = 0
    if optimize_params.get("initial_smoothed", False):
        par["initial_smoothed"] = float(fit_par[j])
        j += 1
    if optimize_params.get("alpha", False):
        par["alpha"] = float(fit_par[j])
        j += 1
    if optimize_params.get("theta", False):
        par["theta"] = float(fit_par[j])
        j += 1

    amse, e, states, mse = _theta.pegels_resid(
        y,
        model_type,
        par["initial_smoothed"],
        par["alpha"],
        par["theta"],
        nmse,
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

def forecast_theta(obj, h, level=None):
    # Extract parameters
    n_obs = obj["n"]  # number of observed steps
    # Initialize states if not already correct shape
    states = obj["states"]
    if states.ndim == 1 or states.shape[1] != 4:
        # assume 1D array, reshape/pad to (n_obs, 4)
        states = jnp.zeros((n_obs, 5), dtype=jnp.float32)
        # optionally fill initial level/mean/An/Bn
        states = states.at[:, 0].set(obj.get("mean_y", 0.0))   # level
        states = states.at[:, 1].set(obj.get("mean_y", 0.0))   # mean
        states = states.at[:, 2].set(0.0)                      # An
        states = states.at[:, 3].set(0.0)                      # Bn
        states = states.at[:, 4].set(0.0)     

    alpha = obj["par"]["alpha"]
    theta = obj["par"]["theta"]
    model_type = switch_theta(obj["modeltype"])

    # Call the JAX forecast function
    states, forecast = _theta.forecast(states, h, model_type, alpha, theta)

    # Build result dictionary
    res = {"mean": forecast}

    # Compute prediction intervals if requested
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

    # Recompose if seasonal decomposition was used
    if obj.get("decompose", False):
        seas_forecast = _repeat_val_seas(obj["seas_forecast"]["mean"], h=h)
        for key in res:
            if obj["decomposition_type"] == "multiplicative":
                res[key] = res[key] * seas_forecast
            else:
                res[key] = res[key] + seas_forecast

    return res

def is_constant(x):
    return jnp.all(x[0] == x)

def seasonal_decompose(y: jnp.ndarray, model: str = "additive", period: int = 1):
    """
    Simple seasonal decomposition using moving average.
    Returns a dict with 'trend', 'seasonal', and 'resid' like statsmodels.
    """
    n = len(y)
    # Moving average for trend
    kernel = jnp.ones(period) / period
    trend = jnp.convolve(y, kernel, mode="same")

    if model == "additive":
        detrended = y - trend
        seasonal = jnp.tile(
            jnp.mean(detrended.reshape(-1, period), axis=0), n // period + 1
        )[:n]
        resid = detrended - seasonal
    else:  # multiplicative
        detrended = y / trend
        seasonal = jnp.tile(
            jnp.mean(detrended.reshape(-1, period), axis=0), n // period + 1
        )[:n]
        resid = detrended / seasonal

    return {
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid,
    }

def acf(x: jnp.ndarray, nlags: int) -> jnp.ndarray:
    """
    Compute autocorrelation function up to `nlags` for 1D array x using JAX.
    Equivalent to statsmodels.tsa.stattools.acf(x, nlags=nlags, fft=False).
    """
    x = x - jnp.mean(x)
    n = x.shape[0]
    denom = jnp.dot(x, x)
    acf_vals = jnp.array([jnp.dot(x[: n - lag], x[lag:]) / denom for lag in range(nlags + 1)])
    return acf_vals

def auto_theta(
    y,
    m,
    model=None,
    initial_smoothed=None,
    alpha=None,
    theta=None,
    nmse=3,
    decomposition_type="multiplicative",
):
    # converting params to floats
    # to improve numba compilation
    if initial_smoothed is None:
        initial_smoothed = jnp.nan
    if alpha is None:
        alpha = jnp.nan
    if theta is None:
        theta = jnp.nan
    if nmse < 1 or nmse > 30:
        raise ValueError("nmse out of range")
    # constan values
    if is_constant(y):
        thetamodel(
            y=y,
            m=m,
            modeltype="STM",
            nmse=nmse,
            initial_smoothed=jnp.mean(y) / 2,
            alpha=0.5,
            theta=2.0,
        )
    # seasonal decomposition if needed
    decompose = False
    # seasonal test
    if m >= 4 and len(y) >= 2 * m:
        r = acf(y, nlags=m, fft=False)[1:]
        stat = jnp.sqrt((1 + 2 * jnp.sum(r[:-1] ** 2)) / len(y))
        decompose = jnp.abs(r[-1]) / stat > norm.ppf(0.95)

    data_positive = min(y) > 0
    if decompose:
        # change decomposition type if data is not positive
        if decomposition_type == "multiplicative" and not data_positive:
            decomposition_type = "additive"
        y_decompose = seasonal_decompose(y, model=decomposition_type, period=m).seasonal
        if decomposition_type == "multiplicative" and any(y_decompose < 0.01):
            decomposition_type = "additive"
            y_decompose = seasonal_decompose(y, model="additive", period=m).seasonal
        if decomposition_type == "additive":
            y = y - y_decompose
        else:
            y = y / y_decompose
        seas_forecast = _seasonal_naive(
            y=y_decompose, h=m, season_length=m, fitted=False
        )

    # validate model
    if model not in [None, "STM", "OTM", "DSTM", "DOTM"]:
        raise ValueError(f"Invalid model type: {model}.")

    n = len(y)
    npars = 3
    # non-optimized tiny datasets
    if n <= npars:
        raise NotImplementedError("tiny datasets")
    if model is None:
        modeltype = ["STM", "OTM", "DSTM", "DOTM"]
    else:
        modeltype = [model]

    best_ic = jnp.inf
    for mtype in modeltype:
        fit = thetamodel(
            y=y,
            m=m,
            modeltype=mtype,
            nmse=nmse,
            initial_smoothed=initial_smoothed,
            alpha=alpha,
            theta=theta,
        )
        fit_ic = fit["mse"]
        if not jnp.isnan(fit_ic):
            if fit_ic < best_ic:
                model = fit
                best_ic = fit_ic
    if jnp.isinf(best_ic):
        raise Exception("no model able to be fitted")

    if decompose:
        if decomposition_type == "multiplicative":
            model["residuals"] = model["residuals"] * y_decompose
        else:
            model["residuals"] = model["residuals"] + y_decompose
        model["decompose"] = decompose
        model["decomposition_type"] = decomposition_type
        model["seas_forecast"] = dict(seas_forecast)
    return model


def forward_theta(fitted_model, y):
    m = fitted_model["m"]
    model = fitted_model["modeltype"]
    initial_smoothed = fitted_model["par"]["initial_smoothed"]
    alpha = fitted_model["par"]["alpha"]
    theta = fitted_model["par"]["theta"]
    return auto_theta(
        y=y,
        m=m,
        model=model,
        initial_smoothed=initial_smoothed,
        alpha=alpha,
        theta=theta,
    )
    return _add_conformal_intervals(self, fcst=fcst, y=None, X=None, level=level)

def _intervals(y: jnp.ndarray) -> jnp.ndarray:
    """
    Return intervals between non-zero observations as a float32 array of length len(y).
    The valid intervals are placed at the front of the returned array and the rest
    are padded with 0.0 to ensure a stable shape/dtype for JAX control-flow.
    """
    n = y.size
    nz_idx = jnp.where(y != 0)[0]  # indices of non-zero entries (int32)

    def no_nz():
        # no non-zero values: return zero-padded float32 array length n
        return jnp.zeros((n,), dtype=jnp.float32)

    def some_nz():
        # diffs are integers; cast to float32 and pad with zeros up to length n
        diffs = jnp.diff(nz_idx).astype(jnp.float32)  # shape (k-1,) if k>=1 else (0,)
        k = diffs.size
        pad_len = n - k
        pad = jnp.zeros((pad_len,), dtype=jnp.float32)
        return jnp.concatenate([diffs, pad], axis=0)

    return lax.cond(nz_idx.size == 0, no_nz, some_nz)


def _chunk_sums(array: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    r"""Splits an array into chunks and returns the sum of each chunk.
    Incomplete chunks are discarded"""
    n_chunks = array.size // chunk_size
    n_elems = n_chunks * chunk_size
    return array[:n_elems].reshape(n_chunks, chunk_size).sum(axis=1)

@jit
def _ses_sse(alpha: float, x: jnp.ndarray) -> float:
    r"""Compute the residual sum of squares for a simple exponential smoothing fit.

    Args:
        alpha (float): Smoothing parameter.
        x (numpy.array): Clean time series of shape (n, ).

    Returns:
        sse (float): Residual sum of squares for the fit.
    """
    complement = 1 - alpha
    forecast = x[0]
    sse = 0.0

    for i in range(1, len(x)):
        forecast = alpha * x[i - 1] + complement * forecast
        sse += (x[i] - forecast) ** 2

    return sse

def _optimized_ses_forecast(
    x: jnp.ndarray, bounds: Tuple[float, float] = (0.1, 0.3), n_grid: int = 50
) -> Tuple[float, jnp.ndarray]:
    alphas = jnp.linspace(bounds[0], bounds[1], n_grid)

    def sse_for_alpha(alpha):
        return _ses_sse(alpha, x)

    sses = jax.vmap(sse_for_alpha)(alphas)
    best_idx = jnp.argmin(sses)
    best_alpha = alphas[best_idx]

    forecast, fitted = _ses_forecast(x, best_alpha)
    return forecast, fitted

def _chunk_forecast(y, aggregation_level):
    lost_remainder_data = len(y) % aggregation_level
    y_cut = y[lost_remainder_data:]
    aggregation_sums = _chunk_sums(y_cut, aggregation_level)
    sums_forecast, _ = _optimized_ses_forecast(aggregation_sums)
    return sums_forecast

@jit
def _expand_fitted_intervals(fitted: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    n = y.size
    out = jnp.full_like(y, jnp.nan)
    
    def body_fn(i, state):
        out_arr, fitted_idx = state
        
        # If previous value was non-zero, advance fitted index
        fitted_idx = jnp.where(
            y[i - 1] != 0,
            fitted_idx + 1,
            fitted_idx
        )
        
        # Determine output value based on conditions
        val = jax.lax.cond(
            y[i - 1] != 0,
            lambda: jnp.where(
                fitted[fitted_idx] == 0,
                1.0,  # Avoid division by zero
                fitted[fitted_idx]
            ),
            lambda: jax.lax.cond(
                fitted_idx > 0,
                lambda: out_arr[i - 1],  # Carry forward previous
                lambda: 1.0  # No intervals seen yet, use 1
            )
        )
        
        out_arr = out_arr.at[i].set(val)
        return out_arr, fitted_idx
    
    out, _ = jax.lax.fori_loop(1, n, body_fn, (out, 0))
    return out


# JAX-only IMAPA with SES + bounded golden-section search
import warnings

jax.config.update("jax_enable_x64", True)


# ---------------------------
# Small helpers
# ---------------------------

@partial(jax.jit, static_argnames=['h'])
def _repeat_val_(val: float, h: int) -> jnp.ndarray:
    return jnp.full((h,), jnp.asarray(val, dtype=val.dtype))

def _intervals(x: jnp.ndarray) -> jnp.ndarray:
    """Intervals between nonzero elements (match numpy reference)."""
    idx = jnp.where(x != 0)[0]
    padded = jnp.concatenate([jnp.array([0], dtype=idx.dtype), idx + 1])
    diffs = jnp.diff(padded)
    return diffs.astype(x.dtype)

@partial(jax.jit, static_argnames=['chunk_size'])
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
@jax.jit
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

@jax.jit
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
) -> Dict:
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
        res = {"mean": jnp.zeros((h,), dtype=out_dtype)}  # Keep this as-is
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
@_partial(jax.jit, static_argnums=(1, 2))
def _linear_extrapolate_tail(y: jnp.ndarray, tail_window: int, h: int) -> jnp.ndarray:
    n = y.shape[0]
    start = jnp.maximum(0, n - tail_window)
    # Use dynamic_slice with STATIC size for JIT compatibility
    # tail_window is static, so we can use it directly
    seg = jax.lax.dynamic_slice(y, (start,), (tail_window,))
    # If n < tail_window, we'll have padded values - need to handle this
    m = jnp.minimum(tail_window, n)
    t = jnp.arange(tail_window)
    t_mean = jnp.mean(t); y_mean = jnp.mean(seg)
    cov = jnp.mean((t - t_mean) * (seg - y_mean))
    var = jnp.mean((t - t_mean) ** 2) + 1e-12
    slope = cov / var
    intercept = y_mean - slope * t_mean
    t_fore = t_mean + (jnp.arange(h) + 1)
    return intercept + slope * t_fore

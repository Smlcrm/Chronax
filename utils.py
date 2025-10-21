"""
1. ensure_float() has one signature, a JAX array, and returns one object, a JAX array.
   It is a helper method for to ensure that the datatypes within a given array are float32. 
"""


import os
from collections import namedtuple
from functools import partial
from typing import Optional, List, Dict, Union, Tuple

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
    out = jnp.empty_like(y)
    out[0] = jnp.nan
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

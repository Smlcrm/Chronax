"""
1. ensure_float() has one signature, a JAX array, and returns one object, a JAX array.
   It is a helper method for to ensure that the datatypes within a given array are float32. 
"""


import jax
import jax.numpy as jnp

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
    level_arr = jnp.asarray(level, jnp.float32)
    p = 0.5 + (level_arr / 200.0)
    return jax.vmap(_jax_norm_ppf)(p)


def _calculate_intervals(
    mean: jnp.ndarray, 
    sigmah: jnp.ndarray | float, 
    level: list[int]
) -> dict:
    """
    Calculate NATIVE (non-conformal) prediction intervals using normal quantiles.
    JAX equivalent of statsforecast.utils._calculate_intervals()
    
    Args:
        mean: Point forecasts of shape (h,)
        sigmah: Standard error for predictions (scalar or array)
        level: Sorted list of confidence levels
        
    Returns:
        Dictionary with 'lo-XX' and 'hi-XX' keys for each level
    """
    z = _quantiles(level)
    # Broadcast to shape (h, len(level))
    lo = mean[:, None] - z[None, :] * sigmah
    hi = mean[:, None] + z[None, :] * sigmah
    
    out = {}
    # Add intervals in correct order (reversed for lo, normal for hi)
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
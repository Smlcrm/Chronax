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
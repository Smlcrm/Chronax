"""Loss functions for DeepAR model (JAX/FLAX implementation)."""

import jax.numpy as jnp


def nll_gaussian(y_true: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Gaussian negative log-likelihood.
    
    Args:
        y_true: Target values, shape (...,)
        mu: Mean predictions, shape (...,)
        sigma: Standard deviation predictions, shape (...,)
    
    Returns:
        NLL per element, shape (...,)
    """
    # Clamp sigma for numerical stability
    sigma = jnp.clip(sigma, 1e-6, 1e6)
    
    # NLL = 0.5 * log(2π) + log(σ) + 0.5 * ((y - μ) / σ)²
    return 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma) + 0.5 * ((y_true - mu) / sigma) ** 2


def nll_gaussian_masked(
    y_true: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Masked Gaussian NLL (for handling variable-length horizons).
    
    Args:
        y_true: Target values, shape [B, H, 1]
        mu: Mean predictions, shape [B, H]
        sigma: Std dev predictions, shape [B, H]
        mask: Binary mask, shape [B, H, 1], where 1 means include in loss
    
    Returns:
        Scalar loss (mean over unmasked elements)
    """
    # Reshape mu, sigma to match y_true
    mu = mu[..., None]
    sigma = sigma[..., None]
    
    nll = nll_gaussian(y_true, mu, sigma)
    masked_nll = nll * mask
    
    # Average over non-masked positions
    num_valid = jnp.sum(mask)
    return jnp.sum(masked_nll) / jnp.maximum(num_valid, 1.0)


def quantile_loss(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    quantile: float,
) -> jnp.ndarray:
    """
    Quantile loss for computing quantiles via quantile regression.
    
    Args:
        y_true: Target values, shape (...,)
        y_pred: Predicted quantile values, shape (...,)
        quantile: Quantile level in (0, 1)
    
    Returns:
        Quantile loss per element
    """
    error = y_true - y_pred
    return jnp.where(
        error >= 0,
        quantile * error,
        (quantile - 1) * error,
    )

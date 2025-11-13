# Part 1: Scale-dependent Erros
import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy


## Mean Absolute Error (MAE)
def mean_absolute_error(y, y_pred):
  diff = jnp.abs(y - y_pred)
  return jnp.mean(diff)

## Mean Squared Error
def mean_squared_error(y, y_pred):
  diff = jnp.power((y - y_pred), 2)
  return jnp.mean(diff)

## Root Mean Squared Error
def root_mean_squared_error(y, y_pred):
  mse = jnp.mean((jnp.power((y - y_pred), 2)))
  return jnp.power(mse, 0.5)

## Bias
def bias (y, y_pred):
  return y_pred - y

## Culmulative Forecasting Errors
def cfe(y, y_pred):
  return jnp.cumsum(y - y_pred)

## Absolute Period in Stock
def pis(y, y_pred):
  return jnp.abs(jnp.cumsum(y - y_pred))

## Scaled Absolute Period in Stock
def spis(y, y_pred):
  pis = jnp.abs(jnp.cumsum(y - y_pred))
  mean = jnp.mean(pis)
  return pis / mean

# Percentage Errors
## Mean Absolute Percentage Error
def mean_absolute_percentage_error(y, y_pred):
  error = jnp.abs(y - y_pred) / (jnp.abs(y) + 1e-8)
  return jnp.mean(error)

## Symmetric Mean Absolute Percentage Error
def symmetric_mean_absolute_percentage_error(y, y_pred):
  error = jnp.abs(y - y_pred) / (jnp.abs(y) + jnp.abs(y_pred))
  return jnp.mean(error)

# Scale-independent Errors
## Mean Absolute Scaled Error
def mean_absolute_scaled_error(y, y_pred, y_seasonal):
  num = jnp.abs(y - y_pred)
  den = mean_absolute_error(y, y_seasonal)
  return jnp.mean(num/den)

## Relative Mean Absolute Error
def relative_mean_absolute_error(y, y_pred, y_base):
  num = jnp.abs(y - y_pred)
  den = mean_absolute_error(y, y_base)
  return jnp.mean(num/den)

## Normalized Deviation
def normalized_deviation(y, y_pred):
  num = jnp.sum(jnp.abs(y - y_pred))
  den = jnp.sum(y)
  return num/den

## Mean Squared Scaled Error
def mean_squared_scaled_error(y, y_pred, y_seasonal):
  num = jnp.power((y - y_pred), 2)
  den = mean_squared_error(y, y_seasonal)
  return jnp.mean(num/den)

## Root Mean Squared Scaled Error
def root_mean_squared_scaled_error(y, y_pred, y_seasonal):
  num = jnp.power((y - y_pred), 2)
  den = mean_squared_error(y, y_seasonal)
  return jnp.mean(jnp.power((num/den), 0.5))

## Quantile Loss
def quantile_loss(y: jnp.ndarray, y_pred: jnp.ndarray, q: float) -> jnp.ndarray:
    """
    Mean Quantile (Pinball) Loss in JAX.
    
    Args:
        y: (N,) True values.
        y_pred: (N,) Predicted quantile values.
        q: Quantile level (e.g., 0.1, 0.5, 0.9).
    
    Returns:
        Scalar mean quantile loss.
    """
    delta = y - y_pred
    loss = jnp.maximum(q * delta, (q - 1) * delta)
    return jnp.mean(loss)


def scaled_quantile_loss(y: jnp.ndarray, y_pred: jnp.ndarray, q: float, y_seasonal: jnp.ndarray,) -> jnp.ndarray:
    """
    Scaled Quantile Loss (SQL) in JAX.
    Similar to quantile_loss, but normalized by in-sample mean absolute error.
    
    Args:
        y: (N,) Test (out-of-sample) actuals.
        y_pred: (N,) Test (out-of-sample) quantile predictions.
        q: Quantile level.
        y_seasonal: (N,) In-sample seasonal baseline values (e.g., y[t] - y[t-season]).
    
    Returns:
        Scalar scaled quantile loss.
    """
    num = quantile_loss(y, y_pred, q)

    den = mean_absolute_error(y, y_seasonal)

    return num / (den + 1e-8)

## Multi-Quantile Loss
def multi_quantile_loss(y: jnp.ndarray, y_pred: jnp.ndarray, quantiles: jnp.ndarray,) -> jnp.ndarray:
    """
    Multi-Quantile Loss (MQL) in JAX (barebones version).

    Args:
        y:         (N, Q) array of true values.
        y_pred:    (N, Q) array of predicted quantiles for each observation.
        quantiles: (Q,) array of quantile levels (e.g., [0.1, 0.5, 0.9]).

    Returns:
        Scalar mean MQL across all samples and quantiles.
    """
    errors = jnp.expand_dims(y, axis=-1) - y_pred              # shape (N, Q)
    loss = jnp.maximum(errors * quantiles, errors * (quantiles - 1))
    return jnp.mean(loss)

## Multi Scaled Quantile Loss
def scaled_multi_quantile_loss(y: jnp.ndarray, y_pred_quantiles: jnp.ndarray, quantiles: jnp.ndarray, y_seasonal: jnp.ndarray) -> jnp.ndarray:
    """
    Scaled Multi-Quantile Loss (SMQL) in JAX.
    Equivalent to the original DataFrame-based `scaled_mqloss` logic.

    Args:
        y_true: (N,) Actual values.
        y_pred_quantiles: (N, Q) Predicted quantiles.
        quantiles: (Q,) Quantile levels.
        seasonality: Seasonal period (e.g., 7 for daily data).

    Returns:
        Scalar SMQL.
    """
    num = multi_quantile_loss(y, y_pred_quantiles, quantiles)
    den = mean_absolute_error(y, y_seasonal)
    return num /(den+ 1e-8)

## Coverage
def coverage(y: jnp.ndarray, y_lo: jnp.ndarray, y_hi: jnp.ndarray) -> jnp.ndarray:
    """
    Computes coverage of prediction intervals in JAX.

    Args:
        y: (N,) True target values.
        y_lo: (N,) Lower prediction interval (e.g., 5th percentile).
        y_hi: (N,) Upper prediction interval (e.g., 95th percentile).

    Returns:
        Scalar coverage rate (fraction of points where y_lo <= y <= y_hi).
    """
    covered = (y >= y_lo) & (y <= y_hi)
    return jnp.mean(covered)

## Calibrartion
def calibration(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Fraction of y that is <= model's predicted quantile (Calibration in JAX).

    Args:
        y: (N,) True target values.
        y_pred: (N,) Predicted quantile values.

    Returns:
        Scalar calibration fraction (between 0 and 1).
    """
    return jnp.mean((y <= y_pred).astype(jnp.float32))

## Scaled CRPS
import jax.numpy as jnp
def scaled_crps(y: jnp.ndarray,y_pred: jnp.ndarray,quantiles: jnp.ndarray,) -> jnp.ndarray:
    """
    Scaled Continuous Ranked Probability Score (Scaled CRPS) in JAX.

    Args:
        y_true: (N,) True observed values.
        y_pred_: (N, Q) Predicted quantile values for each observation.
        quantiles: (Q,) Quantile levels (e.g., [0.1, 0.5, 0.9]).

    Returns:
        Scalar Scaled CRPS value.
    """
    eps = jnp.finfo(jnp.float32).eps
    mql = multi_quantile_loss(y, y_pred, quantiles)
    counts = y.size
    norm = jnp.sum(jnp.abs(y))
    return (2 * mql * counts) / (norm + eps)

## tweedie_deviance

def tweedie_deviance(y: jnp.ndarray, y_pred: jnp.ndarray, power: float) -> jnp.ndarray:
    """
    Compute Tweedie deviance for multiple models in JAX.

    Args:
        y: True values, shape (N,)
        y_pred: Predicted values, shape (N, M)
        power: Tweedie power parameter
               0 = Gaussian (squared error)
               1 = Poisson
               1 < power < 2 = Compound Poisson-Gamma
               2 = Gamma
               >2 = Inverse Gaussian

    Returns:
        Mean deviance per model, shape (M,)
    """
    if power < 0:
        raise ValueError("Power must be non-negative.")

    if y_pred.ndim == y.ndim + 1:
        y_b = jnp.expand_dims(y, axis=-1)
    else:
        y_b = y
    
    # Check positivity constraints
    if power >= 2 and jnp.any(y_b <= 0):
        raise ValueError("For power >= 2, all targets must be strictly positive.")
    if jnp.any(y_pred <= 0):
        raise ValueError("All predictions must be strictly positive for Tweedie deviance.")

    if power == 0:
        dev = (y_pred - y_b) ** 2
    elif power == 1:
        dev = 2 * (xlogy(y_b, y_b / y_pred) - (y_b - y_pred))
    elif power == 2:
        dev = 2 * ((jnp.log(y_pred) - jnp.log(y_b)) + (y_b / y_pred) - 1)
    else:
        y_clip = jnp.clip(y_b, 0)
        dev = 2 * (
            y_clip ** (2 - power) / ((1 - power) * (2 - power))
            - y_clip * (y_pred ** (1 - power)) / (1 - power)
            + y_pred ** (2 - power) / (2 - power)
        )
    return jnp.mean(dev)  # shape (M,)

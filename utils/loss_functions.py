# Part 1: Scale-dependent Erros
import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy


## Mean Absolute Error (MAE)
def mean_absolute_error(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Mean Absolute Error (MAE).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.

  Returns:
      jnp.ndarray: A scalar array representing the mean absolute error.
  """
  diff = jnp.abs(y - y_pred)
  return jnp.mean(diff)

## Mean Squared Error
def mean_squared_error(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Mean Squared Error (MSE).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.

  Returns:
      jnp.ndarray: A scalar array representing the mean squared error.
  """
  diff = jnp.power((y - y_pred), 2)
  return jnp.mean(diff)

## Root Mean Squared Error
def root_mean_squared_error(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Root Mean Squared Error (RMSE).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.

  Returns:
      jnp.ndarray: A scalar array representing the root mean squared error.
  """
  mse = jnp.mean((jnp.power((y - y_pred), 2)))
  return jnp.power(mse, 0.5)

## Bias
def bias(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the point-wise forecasting bias.

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.

  Returns:
      jnp.ndarray: An array of differences (y_pred - y).
  """
  return y_pred - y

## Culmulative Forecasting Errors
def cfe(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """
  Computes Cumulative Forecasting Errors (CFE).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.

  Returns:
      jnp.ndarray: An array representing the cumulative sum of errors over time.
  """
  return jnp.cumsum(y - y_pred)

## Absolute Period in Stock
def pis(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Absolute Period in Stock (PIS) tracking cumulative error magnitude.

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.

  Returns:
      jnp.ndarray: An array of the absolute cumulative sums of forecasting errors.
  """
  return jnp.abs(jnp.cumsum(y - y_pred))

## Scaled Absolute Period in Stock
def spis(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Scaled Absolute Period in Stock (SPIS).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.

  Returns:
      jnp.ndarray: The Absolute Period in Stock scaled by its mean.
  """
  pis = jnp.abs(jnp.cumsum(y - y_pred))
  mean = jnp.mean(pis)
  return pis / mean

# Percentage Errors
## Mean Absolute Percentage Error
def mean_absolute_percentage_error(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Mean Absolute Percentage Error (MAPE).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.

  Returns:
      jnp.ndarray: A scalar array representing the MAPE.
  """
  error = jnp.abs(y - y_pred) / (jnp.abs(y) + 1e-8)
  return jnp.mean(error)

## Symmetric Mean Absolute Percentage Error
def symmetric_mean_absolute_percentage_error(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Symmetric Mean Absolute Percentage Error (sMAPE).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.

  Returns:
      jnp.ndarray: A scalar array representing the sMAPE.
  """
  error = jnp.abs(y - y_pred) / (jnp.abs(y) + jnp.abs(y_pred))
  return jnp.mean(error)

# Scale-independent Errors
## Mean Absolute Scaled Error
def mean_absolute_scaled_error(y: jnp.ndarray, y_pred: jnp.ndarray, y_seasonal: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Mean Absolute Scaled Error (MASE).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.
      y_seasonal (jnp.ndarray): Array of in-sample naive seasonal forecasts.

  Returns:
      jnp.ndarray: A scalar array representing the MASE.
  """
  num = jnp.abs(y - y_pred)
  den = mean_absolute_error(y, y_seasonal)
  return jnp.mean(num/den)

## Relative Mean Absolute Error
def relative_mean_absolute_error(y: jnp.ndarray, y_pred: jnp.ndarray, y_base: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Relative Mean Absolute Error (RMAE).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.
      y_base (jnp.ndarray): Array of predictions from a baseline/benchmark model.

  Returns:
      jnp.ndarray: A scalar array representing the RMAE.
  """
  num = jnp.abs(y - y_pred)
  den = mean_absolute_error(y, y_base)
  return jnp.mean(num/den)

## Normalized Deviation
def normalized_deviation(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Normalized Deviation.

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.

  Returns:
      jnp.ndarray: A scalar array representing the normalized deviation.
  """
  num = jnp.sum(jnp.abs(y - y_pred))
  den = jnp.sum(y)
  return num/den

## Mean Squared Scaled Error
def mean_squared_scaled_error(y: jnp.ndarray, y_pred: jnp.ndarray, y_seasonal: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Mean Squared Scaled Error (MSSE).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.
      y_seasonal (jnp.ndarray): Array of in-sample naive seasonal forecasts.

  Returns:
      jnp.ndarray: A scalar array representing the MSSE.
  """
  num = jnp.power((y - y_pred), 2)
  den = mean_squared_error(y, y_seasonal)
  return jnp.mean(num/den)

## Root Mean Squared Scaled Error
def root_mean_squared_scaled_error(y: jnp.ndarray, y_pred: jnp.ndarray, y_seasonal: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the Root Mean Squared Scaled Error (RMSSE).

  Args:
      y (jnp.ndarray): Array of true observed values.
      y_pred (jnp.ndarray): Array of predicted values.
      y_seasonal (jnp.ndarray): Array of in-sample naive seasonal forecasts.

  Returns:
      jnp.ndarray: A scalar array representing the RMSSE.
  """
  num = jnp.power((y - y_pred), 2)
  den = mean_squared_error(y, y_seasonal)
  return jnp.mean(jnp.power((num/den), 0.5))

## Quantile Loss
def quantile_loss(y: jnp.ndarray, y_pred: jnp.ndarray, q: float) -> jnp.ndarray:
    """
    Mean Quantile (Pinball) Loss in JAX.
    
    Args:
        y (jnp.ndarray): (N,) True values.
        y_pred (jnp.ndarray): (N,) Predicted quantile values.
        q (float): Quantile level (e.g., 0.1, 0.5, 0.9).
    
    Returns:
        jnp.ndarray: Scalar mean quantile loss.
    """
    delta = y - y_pred
    loss = jnp.maximum(q * delta, (q - 1) * delta)
    return jnp.mean(loss)


def scaled_quantile_loss(y: jnp.ndarray, y_pred: jnp.ndarray, q: float, y_seasonal: jnp.ndarray) -> jnp.ndarray:
    """
    Scaled Quantile Loss (SQL) in JAX.
    Similar to quantile_loss, but normalized by in-sample mean absolute error.
    
    Args:
        y (jnp.ndarray): (N,) Test (out-of-sample) actuals.
        y_pred (jnp.ndarray): (N,) Test (out-of-sample) quantile predictions.
        q (float): Quantile level.
        y_seasonal (jnp.ndarray): (N,) In-sample seasonal baseline values (e.g., y[t] - y[t-season]).
    
    Returns:
        jnp.ndarray: Scalar scaled quantile loss.
    """
    num = quantile_loss(y, y_pred, q)

    den = mean_absolute_error(y, y_seasonal)

    return num / (den + 1e-8)

## Multi-Quantile Loss
def multi_quantile_loss(y: jnp.ndarray, y_pred: jnp.ndarray, quantiles: jnp.ndarray) -> jnp.ndarray:
    """
    Multi-Quantile Loss (MQL) in JAX (barebones version).

    Args:
        y (jnp.ndarray): (N,) array of true values.
        y_pred (jnp.ndarray): (N, Q) array of predicted quantiles for each observation.
        quantiles (jnp.ndarray): (Q,) array of quantile levels (e.g., [0.1, 0.5, 0.9]).

    Returns:
        jnp.ndarray: Scalar mean MQL across all samples and quantiles.
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
        y (jnp.ndarray): (N,) Actual values.
        y_pred_quantiles (jnp.ndarray): (N, Q) Predicted quantiles.
        quantiles (jnp.ndarray): (Q,) Quantile levels.
        y_seasonal (jnp.ndarray): (N,) Seasonal baseline values (e.g., naive forecast errors).

    Returns:
        jnp.ndarray: Scalar SMQL.
    """
    num = multi_quantile_loss(y, y_pred_quantiles, quantiles)
    den = mean_absolute_error(y, y_seasonal)
    return num /(den+ 1e-8)

## Coverage
def coverage(y: jnp.ndarray, y_lo: jnp.ndarray, y_hi: jnp.ndarray) -> jnp.ndarray:
    """
    Computes coverage of prediction intervals in JAX.

    Args:
        y (jnp.ndarray): (N,) True target values.
        y_lo (jnp.ndarray): (N,) Lower prediction interval (e.g., 5th percentile).
        y_hi (jnp.ndarray): (N,) Upper prediction interval (e.g., 95th percentile).

    Returns:
        jnp.ndarray: Scalar coverage rate (fraction of points where y_lo <= y <= y_hi).
    """
    covered = (y >= y_lo) & (y <= y_hi)
    return jnp.mean(covered)

## Calibrartion
def calibration(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Fraction of y that is <= model's predicted quantile (Calibration in JAX).

    Args:
        y (jnp.ndarray): (N,) True target values.
        y_pred (jnp.ndarray): (N,) Predicted quantile values.

    Returns:
        jnp.ndarray: Scalar calibration fraction (between 0 and 1).
    """
    return jnp.mean((y <= y_pred).astype(jnp.float32))

## Scaled CRPS
import jax.numpy as jnp
def scaled_crps(y: jnp.ndarray, y_pred: jnp.ndarray, quantiles: jnp.ndarray) -> jnp.ndarray:
    """
    Scaled Continuous Ranked Probability Score (Scaled CRPS) in JAX.

    Args:
        y (jnp.ndarray): (N,) True observed values.
        y_pred (jnp.ndarray): (N, Q) Predicted quantile values for each observation.
        quantiles (jnp.ndarray): (Q,) Quantile levels (e.g., [0.1, 0.5, 0.9]).

    Returns:
        jnp.ndarray: Scalar Scaled CRPS value.
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
        y (jnp.ndarray): True values, shape (N,).
        y_pred (jnp.ndarray): Predicted values, shape (N, M).
        power (float): Tweedie power parameter.
                       0 = Gaussian (squared error)
                       1 = Poisson
                       1 < power < 2 = Compound Poisson-Gamma
                       2 = Gamma
                       >2 = Inverse Gaussian

    Returns:
        jnp.ndarray: Mean deviance (scalar over all elements, though inline comment notes (M,) if un-aggregated).
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
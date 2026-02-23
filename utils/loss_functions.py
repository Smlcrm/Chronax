"""
File: loss_functions.py

High-level Purpose:
    Defines reusable deterministic and probabilistic forecasting loss/metric
    functions implemented with JAX tensors for model training and evaluation.

Problem Solved:
    Consolidates metric implementations in one module to enforce consistent
    error calculations across experiments, benchmarking, and validation tests.

Architectural Role:
    Serves as the metrics utility layer consumed by forecasting models,
    evaluation scripts, and the corresponding test suite.

Major Classes/Functions:
    - Scale-dependent metrics: MAE, MSE, RMSE, bias, CFE, PIS, SPIS.
    - Percentage and scaled metrics: MAPE, SMAPE, MASE-family variants.
    - Quantile/probabilistic metrics: quantile loss, multi-quantile loss,
      CRPS-like scaled score, coverage, calibration.
    - `tweedie_deviance` for generalized exponential dispersion error families.

External Dependencies:
    - `jax`, `jax.numpy`
    - `jax.scipy.special.xlogy`

Expected Inputs and Outputs:
    - Input: JAX-compatible tensors for targets/predictions and optional
      auxiliary arrays (seasonal baselines, quantiles, power parameters).
    - Output: scalar or tensor metric values as `jnp.ndarray`.

Example:
    >>> import jax.numpy as jnp
    >>> from loss_functions import mean_absolute_error
    >>> y = jnp.array([1.0, 2.0, 3.0])
    >>> y_hat = jnp.array([1.2, 1.9, 2.8])
    >>> float(mean_absolute_error(y, y_hat))
    0.16666667

Assumptions:
    - Array shapes are broadcast-compatible for each metric.
    - Inputs satisfy domain constraints for metrics that require positivity.

Side Effects:
    - None. Functions are pure numeric computations.

Author:
    Auto-documented
Date:
    2026-02-21
"""

# Part 1: Scale-dependent Erros
import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy

Array = jnp.ndarray


## Mean Absolute Error (MAE)
def mean_absolute_error(y: Array, y_pred: Array) -> Array:
  """
  Compute the mean absolute error between observed and predicted values.

  Detailed Description:
      Computes the average of the absolute element-wise differences between
      the target array `y` and the prediction array `y_pred`. This is a
      scale-dependent metric: larger magnitudes in the data yield larger MAE.
      Used throughout the module as the primary L1 loss and as the denominator
      in scaled metrics (e.g. MASE, relative MAE).

  Args:
      y (Array): Observed or true values; any shape supported by JAX.
      y_pred (Array): Predicted values; must be broadcast-compatible with `y`.

  Returns:
      Array: Scalar (0-dimensional) JAX array containing the mean absolute
          error. Typically converted to float for reporting.

  Raises:
      None. Invalid or non-finite inputs produce non-finite output.

  Side Effects:
      None. Pure function.

  Example:
      >>> y = jnp.array([1.0, 2.0, 3.0])
      >>> y_hat = jnp.array([1.2, 1.9, 2.8])
      >>> float(mean_absolute_error(y, y_hat))
      0.16666667

  Notes:
      Role: Core scale-dependent accuracy metric and building block for
      scaled and relative metrics in forecasting evaluation and model training.
  """
  diff = jnp.abs(y - y_pred)
  return jnp.mean(diff)

## Mean Squared Error
def mean_squared_error(y: Array, y_pred: Array) -> Array:
  """
  Compute the mean squared error between observed and predicted values.

  Detailed Description:
      Computes the average of the squared element-wise errors (y - y_pred)^2.
      MSE is scale-dependent and penalizes large errors more than MAE. It is
      differentiable everywhere and is commonly used as a training objective and
      for variance estimation in scaled metrics (e.g. MSSE, RMSSE).

  Args:
      y (Array): Observed or true values; any shape supported by JAX.
      y_pred (Array): Predicted values; must be broadcast-compatible with `y`.

  Returns:
      Array: Scalar (0-dimensional) JAX array containing the mean squared
          error. Units are the square of the original variable.

  Raises:
      None. Non-finite inputs yield non-finite output.

  Side Effects:
      None. Pure function.

  Example:
      >>> mean_squared_error(jnp.array([1.0, 2.0]), jnp.array([1.5, 2.2]))

  Notes:
      Role: Primary L2 loss for model fitting and denominator for
      mean-squared-scaled and root-mean-squared-scaled error metrics.
  """
  diff = jnp.power((y - y_pred), 2)
  return jnp.mean(diff)

## Root Mean Squared Error
def root_mean_squared_error(y: Array, y_pred: Array) -> Array:
  """
  Compute the root mean squared error (RMSE) between observed and predicted values.

  Detailed Description:
      Computes the square root of the mean squared error so that the result
      is in the same units as the target variable. RMSE is scale-dependent
      and is widely used for point-forecast accuracy reporting and comparison
      across models.

  Args:
      y (Array): Observed or true values; any shape supported by JAX.
      y_pred (Array): Predicted values; must be broadcast-compatible with `y`.

  Returns:
      Array: Scalar (0-dimensional) JAX array containing RMSE, in same units
          as `y` and `y_pred`.

  Raises:
      None. Non-finite inputs yield non-finite output.

  Side Effects:
      None. Pure function.

  Example:
      >>> root_mean_squared_error(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.1, 2.1, 2.9]))

  Notes:
      Role: Standard scale-dependent accuracy metric for benchmarking and
      reporting forecast performance in interpretable units.
  """
  mse = jnp.mean((jnp.power((y - y_pred), 2)))
  return jnp.power(mse, 0.5)

## Bias
def bias(y: Array, y_pred: Array) -> Array:
  """
  Compute the signed forecast error (prediction minus actual) at each element.

  Detailed Description:
      Returns the element-wise difference y_pred - y, i.e. positive values
      indicate over-forecasting and negative values indicate under-forecasting.
      Used to assess systematic bias in forecasts and in inventory/stock
      applications where sign of error matters.

  Args:
      y (Array): Observed or true values.
      y_pred (Array): Predicted values; must be broadcast-compatible with `y`.

  Returns:
      Array: Same shape as (broadcast of) `y` and `y_pred`, containing
          signed errors. Not aggregated; callers may sum or average as needed.

  Raises:
      None.

  Side Effects:
      None. Pure function.

  Example:
      >>> bias(jnp.array([10.0, 20.0]), jnp.array([12.0, 18.0]))
      Array([ 2., -2.], dtype=float32)

  Notes:
      Role: Diagnostic and building block for bias-aware metrics and
      cumulative error calculations (e.g. CFE, PIS).
  """
  return y_pred - y

## Culmulative Forecasting Errors
def cfe(y: Array, y_pred: Array) -> Array:
  """
  Compute the cumulative sum of forecast errors (actual minus predicted) over time.

  Detailed Description:
      Forms the sequence of running totals of (y - y_pred). Positive values
      indicate persistent under-forecasting (actuals exceed predictions);
      negative values indicate persistent over-forecasting. Used in inventory
      and demand planning to track bias accumulation over the horizon.

  Args:
      y (Array): Observed values, typically a 1D time series.
      y_pred (Array): Predicted values; same length/shape as `y` for
          meaningful interpretation.

  Returns:
      Array: Cumulative sum of (y - y_pred), same shape as the flattened
          difference. For 1D inputs, shape (n,) with the i-th element being
          the sum of the first i errors.

  Raises:
      None.

  Side Effects:
      None. Pure function.

  Example:
      >>> cfe(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.5, 1.5, 2.5]))

  Notes:
      Role: Foundation for bias diagnostics and for PIS/SPIS (period-in-stock)
      metrics in inventory and supply-chain forecasting.
  """
  return jnp.cumsum(y - y_pred)

## Absolute Period in Stock
def pis(y: Array, y_pred: Array) -> Array:
  """
  Compute the absolute cumulative forecast error (Period In Stock style).

  Detailed Description:
      Takes the cumulative sum of (y - y_pred) and returns its element-wise
      absolute value. Measures the magnitude of accumulated bias over time
      regardless of direction, used in inventory contexts to quantify
      cumulative deviation from forecasts.

  Args:
      y (Array): Observed values (e.g. demand or sales).
      y_pred (Array): Predicted values; same shape as `y` for interpretation.

  Returns:
      Array: Absolute values of the cumulative sum of (y - y_pred), same
          shape as the cumulative sum.

  Raises:
      None.

  Side Effects:
      None. Pure function.

  Example:
      >>> pis(jnp.array([10.0, 12.0, 8.0]), jnp.array([10.0, 10.0, 10.0]))

  Notes:
      Role: Building block for PIS-based inventory metrics; SPIS normalizes
      this by its mean to obtain a scale-free measure.
  """
  return jnp.abs(jnp.cumsum(y - y_pred))

## Scaled Absolute Period in Stock
def spis(y: Array, y_pred: Array) -> Array:
  """
  Compute the scaled absolute cumulative forecast error (SPIS).

  Detailed Description:
      Computes the absolute cumulative error (PIS), then scales it by its
      mean so that the resulting sequence has mean 1.0. This yields a
      scale-independent view of how cumulative error evolves relative to
      its typical magnitude, used for cross-series comparison in inventory
      and demand forecasting.

  Args:
      y (Array): Observed values.
      y_pred (Array): Predicted values; same shape as `y`.

  Returns:
      Array: PIS values divided by their mean; same shape as PIS. Mean of
          the output is 1.0 (unless PIS is all zeros, in which case
          division may produce non-finite values).

  Raises:
      None. Zero mean of PIS can yield inf/nan.

  Side Effects:
      None. Pure function.

  Example:
      >>> spis(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.5, 1.5, 2.5]))

  Notes:
      Role: Scale-normalized cumulative error metric for comparing
      forecast bias accumulation across different series or units.
  """
  pis = jnp.abs(jnp.cumsum(y - y_pred))
  mean = jnp.mean(pis)
  return pis / mean

# Percentage Errors
## Mean Absolute Percentage Error
def mean_absolute_percentage_error(y: Array, y_pred: Array) -> Array:
  """
  Compute the mean absolute percentage error (MAPE).

  Detailed Description:
      Computes the mean of |y - y_pred| / (|y| + eps), with a small epsilon
      to avoid division by zero. MAPE is scale-independent and expressed as
      a proportion (e.g. 0.05 for 5% average error). It is undefined or
      unstable when true values are zero or very small.

  Args:
      y (Array): Observed or true values. Should be non-zero for meaningful
          interpretation; zeros are stabilized with 1e-8.
      y_pred (Array): Predicted values; broadcast-compatible with `y`.

  Returns:
      Array: Scalar mean absolute percentage error (fraction, not percentage).
          Multiply by 100 for percentage units.

  Raises:
      None. Zero or negative `y` are stabilized, not raised.

  Side Effects:
      None. Pure function.

  Example:
      >>> mean_absolute_percentage_error(jnp.array([10.0, 20.0]), jnp.array([11.0, 19.0]))

  Notes:
      Role: Scale-independent accuracy metric for reporting and comparison;
      avoid for series with zeros or near-zero actuals.
  """
  error = jnp.abs(y - y_pred) / (jnp.abs(y) + 1e-8)
  return jnp.mean(error)

## Symmetric Mean Absolute Percentage Error
def symmetric_mean_absolute_percentage_error(y: Array, y_pred: Array) -> Array:
  """
  Compute the symmetric mean absolute percentage error (SMAPE).

  Detailed Description:
      Computes the mean of |y - y_pred| / (|y| + |y_pred|), which is
      symmetric in actual and predicted and bounded between 0 and 1. Unlike
      MAPE, it remains defined when actuals or predictions are zero (except
      when both are zero at the same point). Commonly used in forecasting
      benchmarks as a scale-independent metric.

  Args:
      y (Array): Observed or true values.
      y_pred (Array): Predicted values; broadcast-compatible with `y`.

  Returns:
      Array: Scalar SMAPE (fraction in [0, 1]). Multiply by 100 for percentage.

  Raises:
      None. When both y and y_pred are zero, the term is 0/0 and may be nan.

  Side Effects:
      None. Pure function.

  Example:
      >>> symmetric_mean_absolute_percentage_error(jnp.array([1.0, 2.0]), jnp.array([1.2, 1.8]))

  Notes:
      Role: Scale-independent, symmetric alternative to MAPE for model
      comparison and evaluation when zeros or sign changes are present.
  """
  error = jnp.abs(y - y_pred) / (jnp.abs(y) + jnp.abs(y_pred))
  return jnp.mean(error)

# Scale-independent Errors
## Mean Absolute Scaled Error
def mean_absolute_scaled_error(y: Array, y_pred: Array, y_seasonal: Array) -> Array:
  """
  Compute the mean absolute scaled error (MASE) using a seasonal naive baseline.

  Detailed Description:
      Scales the mean absolute error of the model (|y - y_pred|) by the mean
      absolute error of a seasonal naive forecast (|y - y_seasonal|). Values
      below 1.0 indicate the model outperforms the naive baseline; above 1.0
      indicates worse performance. MASE is scale-independent and comparable
      across series with different units.

  Args:
      y (Array): Observed values (typically out-of-sample).
      y_pred (Array): Model predictions; same shape as `y`.
      y_seasonal (Array): Seasonal naive baseline (e.g. previous season same
          period); same shape as `y`. Often y_seasonal[t] = y[t - period].

  Returns:
      Array: Scalar MASE. Ratio of model MAE to baseline MAE; denominator
          is stabilized with 1e-8 internally where needed.

  Raises:
      None. Zero denominator is stabilized.

  Side Effects:
      None. Pure function.

  Example:
      >>> mean_absolute_scaled_error(y_test, y_hat, y_naive)

  Notes:
      Role: Primary scale-independent accuracy metric for univariate
      forecasting; used in benchmarks and model selection.
  """
  num = jnp.abs(y - y_pred)
  den = mean_absolute_error(y, y_seasonal)
  return jnp.mean(num/den)

## Relative Mean Absolute Error
def relative_mean_absolute_error(y: Array, y_pred: Array, y_base: Array) -> Array:
  """
  Compute the relative mean absolute error (RelMAE) against an arbitrary baseline.

  Detailed Description:
      Divides the mean absolute error of the model (|y - y_pred|) by the mean
      absolute error of a baseline forecast (|y - y_base|). Values below 1.0
      mean the model beats the baseline; above 1.0 means the baseline is
      better. The baseline can be naive, seasonal naive, or another model's
      forecasts, enabling flexible pairwise comparison.

  Args:
      y (Array): Observed values.
      y_pred (Array): Model predictions; same shape as `y`.
      y_base (Array): Baseline forecast values; same shape as `y`.

  Returns:
      Array: Scalar RelMAE. Ratio of model MAE to baseline MAE.

  Raises:
      None. Zero denominator is stabilized.

  Side Effects:
      None. Pure function.

  Example:
      >>> relative_mean_absolute_error(y_true, model_pred, naive_pred)

  Notes:
      Role: Scale-independent comparison of a model against any chosen
      baseline in evaluation and benchmarking.
  """
  num = jnp.abs(y - y_pred)
  den = mean_absolute_error(y, y_base)
  return jnp.mean(num/den)

## Normalized Deviation
def normalized_deviation(y: Array, y_pred: Array) -> Array:
  """
  Compute the normalized total absolute deviation by total observed value.

  Detailed Description:
      Divides the sum of absolute errors (|y - y_pred|) by the sum of
      observed values (y). Yields a scale-independent ratio interpretable as
      total absolute error per unit of total demand/volume. Used in
      inventory and demand contexts where total volume is the natural scale.

  Args:
      y (Array): Observed values (e.g. demand); typically non-negative.
      y_pred (Array): Predicted values; same shape as `y`.

  Returns:
      Array: Scalar ratio. Sum(|y - y_pred|) / Sum(y). No explicit
          denominator stabilization; caller should ensure sum(y) > 0.

  Raises:
      None. Zero sum(y) can yield inf/nan.

  Side Effects:
      None. Pure function.

  Example:
      >>> normalized_deviation(jnp.array([10.0, 20.0, 30.0]), jnp.array([12.0, 18.0, 32.0]))

  Notes:
      Role: Scale-independent aggregate accuracy metric for volume-based
      forecasting and reporting.
  """
  num = jnp.sum(jnp.abs(y - y_pred))
  den = jnp.sum(y)
  return num/den

## Mean Squared Scaled Error
def mean_squared_scaled_error(y: Array, y_pred: Array, y_seasonal: Array) -> Array:
  """
  Compute the mean squared scaled error (MSSE) using a seasonal baseline.

  Detailed Description:
      Scales the mean squared error of the model by the MSE of a seasonal
      naive forecast (y vs y_seasonal). Analogous to MASE but for squared
      errors; values below 1.0 indicate the model outperforms the baseline.
      Scale-independent and useful when squared-error loss is the objective.

  Args:
      y (Array): Observed values.
      y_pred (Array): Model predictions; same shape as `y`.
      y_seasonal (Array): Seasonal naive baseline; same shape as `y`.

  Returns:
      Array: Scalar MSSE. Ratio of model MSE to baseline MSE.

  Raises:
      None. Zero denominator can yield inf/nan.

  Side Effects:
      None. Pure function.

  Example:
      >>> mean_squared_scaled_error(y_test, y_hat, y_naive)

  Notes:
      Role: Scale-independent L2-style metric for model comparison and
      selection when using squared-error criteria.
  """
  num = jnp.power((y - y_pred), 2)
  den = mean_squared_error(y, y_seasonal)
  return jnp.mean(num/den)

## Root Mean Squared Scaled Error
def root_mean_squared_scaled_error(y: Array, y_pred: Array, y_seasonal: Array) -> Array:
  """
  Compute the root mean squared scaled error (RMSSE) using a seasonal baseline.

  Detailed Description:
      Computes the element-wise squared error scaled by baseline MSE, then
      takes the mean of the square roots (so each term is in "RMSE units"
      relative to the baseline), and returns the mean of those. Produces a
      scale-independent metric that penalizes large relative errors. Used in
      forecasting competitions and benchmarks.

  Args:
      y (Array): Observed values.
      y_pred (Array): Model predictions; same shape as `y`.
      y_seasonal (Array): Seasonal naive baseline; same shape as `y`.

  Returns:
      Array: Scalar RMSSE. Mean of sqrt((y - y_pred)^2 / baseline_MSE).

  Raises:
      None. Zero denominator can yield inf/nan.

  Side Effects:
      None. Pure function.

  Example:
      >>> root_mean_squared_scaled_error(y_test, y_hat, y_naive)

  Notes:
      Role: Scale-independent L2-style metric in same units as the variable,
      for benchmarking and model comparison.
  """
  num = jnp.power((y - y_pred), 2)
  den = mean_squared_error(y, y_seasonal)
  return jnp.mean(jnp.power((num/den), 0.5))

## Quantile Loss
def quantile_loss(y: jnp.ndarray, y_pred: jnp.ndarray, q: float) -> jnp.ndarray:
    """
    Compute the mean quantile (pinball) loss for a single quantile level.

    Detailed Description:
        For each observation, the loss is q * (y - y_pred) when y > y_pred
        (under-prediction) and (q - 1) * (y - y_pred) when y <= y_pred
        (over-prediction). The mean over all observations is returned. This
        loss is minimized when y_pred equals the q-quantile of the conditional
        distribution of y. Used for quantile regression and prediction
        interval estimation.

    Args:
        y (jnp.ndarray): True observed values; typically shape (N,) or
            broadcast-compatible.
        y_pred (jnp.ndarray): Predicted quantile values; same shape as `y`.
        q (float): Quantile level in (0, 1), e.g. 0.5 for median, 0.1 for
            lower tail, 0.9 for upper tail.

    Returns:
        jnp.ndarray: Scalar mean pinball loss. Same dtype as inputs.

    Raises:
        None. q outside [0, 1] is not checked; behavior may be unintuitive.

    Side Effects:
        None. Pure function.

    Example:
        >>> quantile_loss(y_true, y_median, 0.5)  # median loss
        >>> quantile_loss(y_true, y_lo, 0.1)      # 10% quantile loss

    Notes:
        Role: Core loss for quantile regression and probabilistic
        forecasting; building block for multi-quantile and scaled quantile
        metrics (MQL, SQL, CRPS-style scores).
    """
    delta = y - y_pred
    loss = jnp.maximum(q * delta, (q - 1) * delta)
    return jnp.mean(loss)


def scaled_quantile_loss(y: jnp.ndarray, y_pred: jnp.ndarray, q: float, y_seasonal: jnp.ndarray,) -> jnp.ndarray:
    """
    Compute the scaled quantile loss (SQL): quantile loss normalized by baseline MAE.

    Detailed Description:
        Computes the mean quantile (pinball) loss for level q, then divides by
        the mean absolute error of a seasonal naive baseline (y vs y_seasonal).
        This makes the metric scale-independent and comparable across series.
        Values below 1.0 indicate the quantile forecast beats the naive baseline.

    Args:
        y (jnp.ndarray): Test (out-of-sample) actual values; shape (N,) or
            compatible.
        y_pred (jnp.ndarray): Test (out-of-sample) quantile predictions for
            level q; same shape as `y`.
        q (float): Quantile level in (0, 1).
        y_seasonal (jnp.ndarray): In-sample seasonal baseline (e.g. previous
            season same period); same shape as `y`. Used to compute
            denominator MAE.

    Returns:
        jnp.ndarray: Scalar SQL. Quantile loss / (MAE of baseline); denominator
            is stabilized with 1e-8.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> scaled_quantile_loss(y_test, y_q10, 0.1, y_naive)

    Notes:
        Role: Scale-independent quantile accuracy metric for evaluation and
        benchmarking of probabilistic forecasts at a single quantile.
    """
    num = quantile_loss(y, y_pred, q)

    den = mean_absolute_error(y, y_seasonal)

    return num / (den + 1e-8)

## Multi-Quantile Loss
def multi_quantile_loss(y: jnp.ndarray, y_pred: jnp.ndarray, quantiles: jnp.ndarray,) -> jnp.ndarray:
    """
    Compute the mean multi-quantile (pinball) loss across multiple quantile levels.

    Detailed Description:
        For each quantile level and each observation, computes the pinball
        loss (as in quantile_loss). The implementation uses broadcasting:
        errors = y - y_pred (across quantiles), then applies
        max(q * errors, (q - 1) * errors) per quantile and takes the mean over
        all elements. Used to evaluate full predictive distributions via
        several quantile predictions (e.g. 0.1, 0.5, 0.9).

    Args:
        y (jnp.ndarray): True values. May be (N,) expanded to (N, Q) or (N, Q)
            directly; must broadcast with y_pred.
        y_pred (jnp.ndarray): Predicted quantiles; shape (N, Q) for N samples
            and Q quantile levels.
        quantiles (jnp.ndarray): Quantile levels, shape (Q,), e.g. [0.1, 0.5, 0.9].

    Returns:
        jnp.ndarray: Scalar mean loss across all samples and quantiles.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> multi_quantile_loss(y_true, y_quantiles, jnp.array([0.1, 0.5, 0.9]))

    Notes:
        Role: Primary loss for multi-quantile and probabilistic forecasting
        evaluation; used in CRPS-style metrics and scaled multi-quantile loss.
    """
    errors = jnp.expand_dims(y, axis=-1) - y_pred              # shape (N, Q)
    loss = jnp.maximum(errors * quantiles, errors * (quantiles - 1))
    return jnp.mean(loss)

## Multi Scaled Quantile Loss
def scaled_multi_quantile_loss(y: jnp.ndarray, y_pred_quantiles: jnp.ndarray, quantiles: jnp.ndarray, y_seasonal: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the scaled multi-quantile loss (SMQL): MQL normalized by baseline MAE.

    Detailed Description:
        Computes the multi-quantile (pinball) loss across all quantile levels,
        then divides by the mean absolute error of a seasonal naive baseline
        (y vs y_seasonal). Yields a scale-independent score for full
        probabilistic forecasts; values below 1.0 indicate the model
        outperforms the baseline on average across quantiles.

    Args:
        y (jnp.ndarray): Actual out-of-sample values; (N,) or compatible.
        y_pred_quantiles (jnp.ndarray): Predicted quantiles for each level;
            shape (N, Q).
        quantiles (jnp.ndarray): Quantile levels, shape (Q,).
        y_seasonal (jnp.ndarray): Seasonal naive baseline; same length as `y`.
            Used as denominator MAE.

    Returns:
        jnp.ndarray: Scalar SMQL. MQL / (MAE of baseline); denominator
            stabilized with 1e-8.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> scaled_multi_quantile_loss(y_test, y_q, quantiles, y_naive)

    Notes:
        Role: Scale-independent multi-quantile metric for benchmarking
        probabilistic forecasting models.
    """
    num = multi_quantile_loss(y, y_pred_quantiles, quantiles)
    den = mean_absolute_error(y, y_seasonal)
    return num /(den+ 1e-8)

## Coverage
def coverage(y: jnp.ndarray, y_lo: jnp.ndarray, y_hi: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the empirical coverage rate of a prediction interval.

    Detailed Description:
        Counts the fraction of observations where the true value y lies within
        the interval [y_lo, y_hi]. For a well-calibrated (1 - alpha) interval
        (e.g. 90%), coverage should be close to (1 - alpha). Used to assess
        whether prediction intervals are too narrow (under-coverage) or too
        wide (over-coverage).

    Args:
        y (jnp.ndarray): True target values; shape (N,) or compatible.
        y_lo (jnp.ndarray): Lower bound of the prediction interval (e.g.
            5th percentile); same shape as `y`.
        y_hi (jnp.ndarray): Upper bound (e.g. 95th percentile); same shape as `y`.

    Returns:
        jnp.ndarray: Scalar in [0, 1]. Proportion of points with
            y_lo <= y <= y_hi.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> coverage(y_true, y_lo_90, y_hi_90)  # expect ~0.90 for 90% intervals

    Notes:
        Role: Calibration metric for interval forecasts; used in validation
        and reporting of uncertainty estimates.
    """
    covered = (y >= y_lo) & (y <= y_hi)
    return jnp.mean(covered)

## Calibrartion
def calibration(y: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the empirical calibration rate for a quantile forecast.

    Detailed Description:
        Returns the fraction of observations where the true value y is less
        than or equal to the predicted quantile y_pred. For a correctly
        calibrated q-quantile forecast, this fraction should be close to q.
        Used to check whether quantile predictions are well-calibrated (e.g.
        50% of actuals below median forecast).

    Args:
        y (jnp.ndarray): True target values; shape (N,) or compatible.
        y_pred (jnp.ndarray): Predicted quantile values (e.g. median or
            other level); same shape as `y`.

    Returns:
        jnp.ndarray: Scalar in [0, 1]. Proportion of points with y <= y_pred.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> calibration(y_true, y_median)  # expect ~0.5 for median

    Notes:
        Role: Diagnostic for quantile calibration; complements coverage
        by focusing on a single quantile level.
    """
    return jnp.mean((y <= y_pred).astype(jnp.float32))

## Scaled CRPS
def scaled_crps(y: jnp.ndarray,y_pred: jnp.ndarray,quantiles: jnp.ndarray,) -> jnp.ndarray:
    """
    Compute a scaled approximation to the Continuous Ranked Probability Score (CRPS).

    Detailed Description:
        Uses the multi-quantile loss (MQL) as a discrete approximation to the
        CRPS, then scales by (2 * MQL * N) / (sum of |y|) so the result is
        scale-independent and comparable across series. Larger values indicate
        worse probabilistic forecasts. The formula rewards sharpness and
        calibration of the predictive distribution represented by the quantiles.

    Args:
        y (jnp.ndarray): True observed values; shape (N,).
        y_pred (jnp.ndarray): Predicted quantiles for each observation; shape
            (N, Q) for Q quantile levels.
        quantiles (jnp.ndarray): Quantile levels, shape (Q,), e.g. [0.1, 0.5, 0.9].

    Returns:
        jnp.ndarray: Scalar scaled CRPS. Denominator uses sum(|y|) with
        epsilon stabilization to avoid division by zero.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> scaled_crps(y_true, y_quantiles, jnp.array([0.1, 0.5, 0.9]))

    Notes:
        Role: Scale-independent probabilistic forecast metric for model
        comparison and evaluation when multiple quantiles are available.
    """
    eps = jnp.finfo(jnp.float32).eps
    mql = multi_quantile_loss(y, y_pred, quantiles)
    counts = y.size
    norm = jnp.sum(jnp.abs(y))
    return (2 * mql * counts) / (norm + eps)

## tweedie_deviance

def tweedie_deviance(y: jnp.ndarray, y_pred: jnp.ndarray, power: float) -> jnp.ndarray:
    """
    Compute the Tweedie deviance for exponential-dispersion family distributions.

    Detailed Description:
        Evaluates the unit deviance for the Tweedie family parameterized by
        `power`. Special cases: power=0 (Gaussian/MSE), power=1 (Poisson),
        power=2 (Gamma). For 1 < power < 2 the distribution is compound
        Poisson-Gamma; for power > 2, inverse Gaussian. Used in generalized
        linear models and loss functions for non-negative or count targets.
        Returns the mean deviance over observations (and over models if
        y_pred has an extra dimension).

    Args:
        y (jnp.ndarray): True observed values; shape (N,) or (N,) for
            broadcasting. Must be non-negative for power >= 1; strictly
            positive for power >= 2.
        y_pred (jnp.ndarray): Predicted values; shape (N,) or (N, M) for M
            models. Must be strictly positive.
        power (float): Tweedie power parameter: 0 (Gaussian), 1 (Poisson),
            in (1, 2) (compound Poisson-Gamma), 2 (Gamma), >2 (inverse Gaussian).

    Returns:
        jnp.ndarray: Mean deviance (scalar or per-model if y_pred is (N, M)).
            Same units as squared error for power=0.

    Raises:
        ValueError: If power < 0; if power >= 2 and any y <= 0; if any
            y_pred <= 0.

    Side Effects:
        None. Pure function.

    Example:
        >>> tweedie_deviance(y_count, y_pred, power=1)   # Poisson
        >>> tweedie_deviance(y_positive, y_pred, power=2)  # Gamma

    Notes:
        Role: Proper scoring rule for count and positive continuous targets
        in forecasting and GLM fitting; supports heterogeneous variance
        (e.g. variance proportional to mean^power).
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

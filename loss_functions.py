# Part 1: Scale-dependent Erros
import jax
import jax.numpy as jnp

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
  error = jnp.abs(y - y_pred) / jnp.abs(y)
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
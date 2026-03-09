"""
SeasonalExponentialSmoothing (SES) for seasonal time series in JAX.

Applies Simple Exponential Smoothing independently to each seasonal subseries
(i.e. each phase of the cycle). The forecast for horizon h is a tiled repetition
of per-season SES forecasts, making it suitable for data with stable seasonal
patterns and no clear trend.

Formula (per season s):
    ℓ[t,s] = α · y[t] + (1-α) · ℓ[t-1,s]
    ŷ[t+h, s] = ℓ[T, s]  (flat forecast per season)

Implementation:
    - Pure function kernels for JIT compilation and vmapping
    - `_ses_forecast_masked`: SES on padded fixed-size arrays (JIT-safe)
    - `_seasonal_exponential_smoothing_jit`: vmaps SES over all seasons at once
    - `_seasonal_exponential_smoothing`: dispatcher handling the n < season_length edge case
    - Class: thin orchestrator delegating to the kernels above

Instance Attributes:
    - season_length: number of observations per seasonal cycle
    - alpha: smoothing parameter shared across all seasons (0 ≤ α ≤ 1)
    - alias: model identifier string
    - prediction_intervals: optional ConformalIntervals for conformal prediction
    - conformal_params: alias for prediction_intervals (BaseForecaster compatibility)
    - only_conformal_intervals: always True (no native parametric intervals)

Methods:
    - fit(y, X): fit per-season SES and optionally compute conformity scores
    - predict(h, X, level): h-step ahead forecasts; optional conformal intervals
    - predict_in_sample(): return in-sample fitted values
    - forecast(y, h, X, X_future, level, fitted): stateless fit+predict
"""
from functools import partial
from typing import Optional, List, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from chronax.utils import ConformalIntervals
from chronax.models.base_forecaster import BaseForecaster
from chronax import utils
from chronax.utils import (
    ensure_float,
    _repeat_val_seas,
    _add_conformal_distribution_intervals,
    _get_conformal_method,
    _conformal_method,
    _store_cs,
    _add_conformal_intervals,
    _add_predict_conformal_intervals,
)


# =============================================================================
# PURE FUNCTION KERNELS
# =============================================================================
# Architecture (functional, like auto_arima):
#   - _ses_forecast_masked: SES over first n_eff elements of a padded slice (JIT)
#   - _seasonal_exponential_smoothing_jit: main kernel - vmap over seasons + tile (JIT)
#   - _seasonal_exponential_smoothing: dispatcher (pure, handles n < season_length edge case)
#   - Class: barebones orchestrator, delegates to kernels above
# =============================================================================


@jax.jit
def _ses_forecast_masked(
    x: jnp.ndarray, alpha: jnp.ndarray, n_eff: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SES over first n_eff elements of padded x. Returns (forecast, fitted_padded)."""
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha
    n = x.shape[0]

    def body_fun(i, fitted_arr):
        def update():
            next_val = alpha * x[i - 1] + complement * fitted_arr[i - 1]
            return fitted_arr.at[i].set(next_val)
        return lax.cond(i < n_eff, update, lambda: fitted_arr)

    init_fitted = jnp.empty_like(x).at[0].set(x[0])
    fitted = lax.fori_loop(1, n, body_fun, init_fitted)
    forecast = lax.cond(
        n_eff > 0,
        lambda: alpha * x[n_eff - 1] + complement * fitted[n_eff - 1],
        lambda: jnp.asarray(jnp.nan, dtype=dtype),
    )
    fitted = fitted.at[0].set(jnp.asarray(jnp.nan, dtype=dtype))
    return forecast, fitted


@partial(jax.jit, static_argnums=(1, 2, 3))
def _seasonal_exponential_smoothing_jit(
    y: jnp.ndarray, h: int, fitted: bool, season_length: int, alpha: float
) -> Dict[str, jnp.ndarray]:
    """
    Fully JIT-compiled path: vmap over seasons, no Python loops.
    Used when we can build a padded season matrix (n >= season_length).
    """
    n = y.shape[0]
    max_len = (n + season_length - 1) // season_length

    # Build (season_length, max_len) matrix: row i = season i values, padded with 0
    i_grid = jnp.arange(season_length)[:, None]
    k_grid = jnp.arange(max_len)[None, :]
    indices = (i_grid + n % season_length) + k_grid * season_length
    valid = indices < n
    safe_idx = jnp.minimum(indices, n - 1)
    padded = jnp.where(valid, y[safe_idx], 0.0)
    n_eff = jnp.sum(valid, axis=1).astype(jnp.int32)

    # vmap SES over all seasons in one kernel launch
    alpha_arr = jnp.broadcast_to(alpha, (season_length,))
    forecasts, fitted_rows = jax.vmap(_ses_forecast_masked, in_axes=(0, 0, 0))(
        padded, alpha_arr, n_eff
    )

    out = _repeat_val_seas(forecasts, h)
    fcst: Dict[str, jnp.ndarray] = {"mean": out}

    if fitted:
        # Scatter fitted values back: one .at[].set per season (in JIT loop)
        fitted_vals = jnp.full_like(y, jnp.nan)

        def scatter_body(i, fv):
            init_idx = i + n % season_length
            idx = init_idx + jnp.arange(max_len, dtype=jnp.int32) * season_length
            mask = idx < n
            idx_safe = jnp.where(mask, idx, 0)
            vals = jnp.where(mask, fitted_rows[i], fv[0])
            return fv.at[idx_safe].set(vals)

        fitted_vals = lax.fori_loop(0, season_length, scatter_body, fitted_vals)
        fcst["fitted"] = fitted_vals

    return fcst


def _seasonal_exponential_smoothing(
    y: jnp.ndarray,
    h: int,
    fitted: bool,
    season_length: int,
    alpha: float,
) -> Dict[str, jnp.ndarray]:
    """
    Core pure kernel: seasonal exponential smoothing fit and forecast.
    Math: per-season SES with shared alpha; forecast = tiled season_vals.
    Uses fully JIT-compiled path (vmap over seasons) when n >= season_length.
    """
    n = y.size
    if n < season_length:
        return {"mean": jnp.full(h, jnp.nan, dtype=y.dtype)}

    return _seasonal_exponential_smoothing_jit(y, h, fitted, season_length, alpha)


# =============================================================================
# MODEL CLASS (barebones orchestrator)
# =============================================================================

class SeasonalExponentialSmoothing(BaseForecaster):
    r"""SeasonalExponentialSmoothing model.

    Uses a weighted average of all past observations where the weights decrease exponentially into the past.
    Suitable for data with no clear trend or seasonality.
    Assuming there are $t$ observations and season $s$, the one-step forecast is given by:
    $\hat{y}_{t+1,s} = \alpha y_t + (1-\alpha) \hat{y}_{t-1,s}$

    Notes:
        This method is an extremely simplified of Holt-Winter's method where the trend and level are set to zero.
        And a single seasonal smoothing parameter $\alpha$ is shared across seasons.

    References:
        - [Charles. C. Holt (1957). "Forecasting seasonals and trends by exponentially weighted moving averages", ONR Research Memorandum, Carnegie Institute of Technology 52.](https://www.sciencedirect.com/science/article/abs/pii/S0169207003001134).
        - [Peter R. Winters (1960). "Forecasting sales by exponentially weighted moving averages". Management Science](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.6.3.324).

    Args:
        alpha (float): Smoothing parameter.
        season_length (int): Number of observations per unit of time. Ex: 24 Hourly data.
        alias (str): Custom name of the model.
        prediction_intervals (Optional[ConformalIntervals]): Information to compute conformal prediction intervals. This is required for generating future prediction intervals.
    """

    def __init__(
        self,
        season_length: int,
        alpha: float,
        alias: str = "SeasonalES",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ) -> None:
        self.season_length = season_length
        self.alpha = alpha
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.conformal_params = prediction_intervals
        self.only_conformal_intervals = True

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "SeasonalExponentialSmoothing":
        r"""Fit the SeasonalExponentialSmoothing model.

        Applies per-season SES to the input series and stores the fitted
        seasonal pattern. If `prediction_intervals` is configured, conformity
        scores are also computed and cached for use in `predict()`.

        Args:
            y (jnp.ndarray): Clean time series of shape (t,).
            X (Optional[jnp.ndarray]): Exogenous variables (unused; included for API compatibility). Default is None.

        Returns:
            SeasonalExponentialSmoothing: Self (fitted model instance).
        """
        y = ensure_float(y)
        mod = _seasonal_exponential_smoothing(
            y=y,
            season_length=self.season_length,
            alpha=self.alpha,
            fitted=True,
            h=self.season_length,
        )
        self.model_ = mod
        _store_cs(self, y=y, X=X)
        return self

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ) -> Dict[str, jnp.ndarray]:
        r"""Generate h-step ahead forecasts using the fitted model.

        Tiles the stored per-season SES forecasts to cover the requested
        horizon. Optionally adds conformal prediction intervals.

        Args:
            h (int): Forecast horizon (number of steps ahead).
            X (Optional[jnp.ndarray]): Exogenous variables (unused; included for API compatibility). Default is None.
            level (Optional[List[int]]): Confidence levels (0--100) for prediction intervals, e.g. [80, 95]. Requires ``prediction_intervals`` to be set. Default is None.

        Returns:
            Dict[str, jnp.ndarray]: Dictionary containing ``"mean"`` (point forecasts of shape (h,)) and optionally ``"lo-{l}"`` / ``"hi-{l}"`` (conformal interval bounds for each level l, only present when level is not None).

        Raises:
            Exception: If level is requested but ``prediction_intervals`` is None.
        """
        mean = _repeat_val_seas(self.model_["mean"], h=h)
        res = {"mean": mean}
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = _add_predict_conformal_intervals(self, res, level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")
        return res

    def predict_in_sample(self) -> Dict[str, jnp.ndarray]:
        r"""Return in-sample fitted values from the last fit() call.

        Returns:
            Dict[str, jnp.ndarray]: Dictionary containing:
                - "fitted": In-sample predictions of shape (t,).
        """
        res = {"fitted": self.model_["fitted"]}
        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        r"""Memory-efficient stateless fit+predict in one call.

        Fits the model on `y` and immediately generates forecasts without
        storing any model state. Useful for cross-validation loops or
        one-shot forecasting.

        Args:
            y (jnp.ndarray): Clean time series of shape (t,).
            h (int): Forecast horizon (number of steps ahead).
            X (Optional[jnp.ndarray]): In-sample exogenous variables (unused; included for API compatibility). Default is None.
            X_future (Optional[jnp.ndarray]): Future exogenous variables (unused; included for API compatibility). Default is None.
            level (Optional[List[int]]): Confidence levels (0--100) for prediction intervals, e.g. [80, 95]. Requires ``prediction_intervals`` to be set. Default is None.
            fitted (bool): Whether to include in-sample fitted values in the output. Default is False.

        Returns:
            Dict[str, jnp.ndarray]: Dictionary containing ``"mean"`` (point forecasts of shape (h,)), ``"fitted"`` (in-sample fitted values of shape (t,), only if fitted=True), and optionally ``"lo-{l}"`` / ``"hi-{l}"`` (conformal interval bounds for each level l, only present when level is not None).

        Raises:
            Exception: If level is requested but ``prediction_intervals`` is None.
        """
        y = ensure_float(y)
        res = _seasonal_exponential_smoothing(
            y=y, h=h, fitted=fitted, alpha=self.alpha, season_length=self.season_length
        )
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = _add_conformal_intervals(self, fcst=res, y=y, X=X, level=level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")
        return res

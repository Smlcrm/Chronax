import jax
import jax.numpy as jnp
from jax import jit
from typing import Optional, List, Dict, Union

from conformal_intervals import ConformalIntervals
from base_forecaster import BaseForecaster

from utils import (
    _seasonal_naive,
    _repeat_val_seas,
    ensure_float,
    calculate_sigma,
    _calculate_intervals,
    _quantiles,
    _add_fitted_pi,
    _add_conformal_distribution_intervals,
    _get_conformal_method,
)

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
        prediction_intervals (Optional[ConformalIntervals]): Information to compute conformal prediction intervals.
            This is required for generating future prediction intervals.
    """

    def __init__(
        self,
        season_length: int,
        alpha: float,
        alias: str = "SeasonalES",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        self.season_length = season_length
        self.alpha = alpha
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.only_conformal_intervals = True

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ):
        r"""Fit the SeasonalExponentialSmoothing model.

        Fit an SeasonalExponentialSmoothing to a time series (numpy array) `y`
        and optionally exogenous variables (numpy array) `X`.

        Args:
            y (numpy.array): Clean time series of shape (t, ).
            X (array-like): Optional exogenous of shape (t, n_x).

        Returns:
            SeasonalExponentialSmoothing: SeasonalExponentialSmoothing fitted model.
        """
        y = ensure_float(y)
        mod = _seasonal_exponential_smoothing(
            y=y,
            season_length=self.season_length,
            alpha=self.alpha,
            fitted=True,
            h=self.season_length,
        )
        self.model_ = dict(mod)
        self._store_cs(y=y, X=X)
        return self

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ):
        r"""Predict with fitted SeasonalExponentialSmoothing.

        Args:
            h (int): Forecast horizon.
            X (array-like): Optional insample exogenous of shape (t, n_x).
            level (List[float]): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        mean = _repeat_val_seas(self.model_["mean"], h=h)
        res = {"mean": mean}
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = self._add_predict_conformal_intervals(res, level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")
        return res

    def predict_in_sample(self):
        r"""Access fitted SeasonalExponentialSmoothing insample predictions.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions.
        """
        res = {"fitted": self.model_["fitted"]}
        return res

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        r"""Memory Efficient SeasonalExponentialSmoothing predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like): Optional insample exogenous of shape (t, n_x).
            X_future (array-like): Optional exogenous of shape (h, n_x).
            level (List[float]): Confidence levels (0-100) for prediction intervals.
            fitted (bool): Whether or not returns insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        y = ensure_float(y)
        res = _seasonal_exponential_smoothing(
            y=y, h=h, fitted=fitted, alpha=self.alpha, season_length=self.season_length
        )
        res = dict(res)
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")
        return res


def _seasonal_ses_optimized(
    y: np.ndarray,  # time series
    h: int,  # forecasting horizon
    fitted: bool,  # fitted values
    season_length: int,  # season length
):
    n = y.size
    if n < season_length:
        return {"mean": np.full(h, np.nan, dtype=y.dtype)}
    season_vals = np.empty(season_length, dtype=y.dtype)
    fitted_vals = np.full_like(y, np.nan)
    for i in range(season_length):
        init_idx = i + n % season_length
        season_vals[i], fitted_vals[init_idx::season_length] = _optimized_ses_forecast(
            y[init_idx::season_length], (0.01, 0.99)
        )
    out = _repeat_val_seas(season_vals=season_vals, h=h)
    fcst = {"mean": out}
    if fitted:
        fcst["fitted"] = fitted_vals
    return fcst
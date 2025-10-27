import jax
import jax.numpy as jnp
from jax import jit
from jax import vmap
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
    _store_cs,
    _add_fitted_pi,
    _add_conformal_distribution_intervals,
    _get_conformal_method,
    _seasonal_exponential_smoothing,
    _ses_forecast,
    _add_predict_conformal_intervals,
    _add_conformal_intervals
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
        self.conformal_params = prediction_intervals
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
            y (jnp.ndarray): Clean time series of shape (t, ).
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
        # self._store_cs(y=y, X=X)
        _store_cs(self, y=y, X=X)
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
            # res = self._add_predict_conformal_intervals(res, level)
            res = _add_predict_conformal_intervals(self,res, level)

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
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        r"""Memory Efficient SeasonalExponentialSmoothing predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (jnp.ndarray): Clean time series of shape (n, ).
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
            # res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
            res = _add_conformal_intervals(self, fcst=res, y=y, X=X, level=level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")
        return res

# def test():
#     y = jnp.arange(36.0)

#     pi = ConformalIntervals(h=12, n_windows=2)
#     model = SeasonalExponentialSmoothing(season_length=12, alpha=0.5, prediction_intervals=pi)
#     fitted_model = model.fit(y)

#     result = fitted_model.predict(h=12, level=(60,75))
#     forecast = fitted_model.forecast(y, h=12, level=[80, 95])
    
#     assert "mean" in result, "Missing mean forecast"

#     assert len(result["mean"]) == 12, "Forecast length mismatch"

#     for lvl in [60, 75]:
#         if f"lo-{lvl}" in result:
#             assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"
#         else:
#             print(f"Warning: Interval {lvl}% not computed due to missing `prediction_intervals`")

#     for lvl in [80, 95]:
#         assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
#         assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"
#         assert len(forecast[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
#         assert len(forecast[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

# if __name__ == "__main__":
#     test()
#     print("Test passed!")
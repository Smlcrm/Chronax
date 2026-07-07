
"""Seasonal Naive forecaster implementation."""

import jax.numpy as jnp
from typing import Dict, List, Optional
from chronax.utils import (
    _seasonal_naive,
    _repeat_val_seas,
    ensure_float,
    calculate_sigma,
    _calculate_intervals,
    _store_cs,
    _add_conformal_intervals,
    _add_predict_conformal_intervals,
    _add_fitted_pi,
)
from chronax.utils import ConformalIntervals
from chronax.models.base_forecaster import BaseForecaster

ForecastDict = Dict[str, jnp.ndarray]


class SeasonalNaive(BaseForecaster):
    def __init__(
        self,
        season_length: int,
        alias: str = "SeasonalNaive",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ) -> None:
        """Seasonal naive model.

        A method similar to the naive, but uses the last known observation of the same period (e.g. the same month of the previous year) in order to capture seasonal variations.

        References:
            - [Rob J. Hyndman and George Athanasopoulos (2018). "forecasting principles and practice, Simple Methods"](https://otexts.com/fpp3/simple-methods.html#seasonal-na%C3%AFve-method).

        Args:
            season_length (int): Number of observations per unit of time. Ex: 24 Hourly data.
            alias (str): Custom name of the model.
            prediction_intervals (Optional[ConformalIntervals]): Information to compute conformal prediction intervals.
                By default, the model will compute the native prediction
                intervals.
        """
        self.season_length = season_length
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        # BaseForecaster.conformity_scores reads conformal_params (see
        # seasonal_exponential_smoothing.py for the same aliasing).
        self.conformal_params = prediction_intervals

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "SeasonalNaive":
        """Fit the SeasonalNaive model.

        Fit an SeasonalNaive to a time series (numpy array) `y`.

        Args:
            y (jnp.ndarray): Clean time series of shape (t, ).
            X (array-like): Optional exogenous of shape (t, n_x).

        Returns:
            self: SeasonalNaive fitted model.
        """
        y = ensure_float(y)
        mod: ForecastDict = dict(
            _seasonal_naive(
                y=y,
                season_length=self.season_length,
                h=self.season_length,
                fitted=True,
            )
        )
        residuals = y - mod["fitted"]
        mod["sigma"] = calculate_sigma(residuals, len(y) - self.season_length)
        self.model_ = mod
        _store_cs(self, y=y, X=X)
        return self

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ) -> ForecastDict:
        """Predict with fitted SeasonalNaive.

        Args:
            h (int): Forecast horizon.
            X (array-like): Optional exogenous of shape (h, n_x).
            level (List[float]): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and
                `level_*` for probabilistic predictions.
        """
        del X
        mean = _repeat_val_seas(season_vals=self.model_["mean"], h=h)
        res: ForecastDict = {"mean": mean}

        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = _add_predict_conformal_intervals(self, res, level)
        else:
            k = jnp.floor(jnp.arange(h) / self.season_length)
            sigma = self.model_["sigma"]
            sigmah = sigma * jnp.sqrt(k + 1)
            pred_int = _calculate_intervals(res, level, h, sigmah)
            res = {**res, **pred_int}
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None) -> ForecastDict:
        """Access fitted SeasonalNaive in-sample predictions.

        Args:
            level (List[float]): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions and
                `level_*` for probabilistic predictions.
        """
        res: ForecastDict = {"fitted": self.model_["fitted"]}
        if level is not None:
            level = sorted(level)
            res = _add_fitted_pi(res=res, se=self.model_["sigma"], level=level)
        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> ForecastDict:
        """Memory Efficient SeasonalNaive predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (jnp.ndarray): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like): Optional in-sample exogenous of shape (t, n_x).
            X_future (array-like): Optional exogenous of shape (h, n_x).
            level (List[float]): Confidence levels (0-100) for prediction intervals.
            fitted (bool): Whether or not to return in-sample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and
                `level_*` for probabilistic predictions.
        """
        del X_future
        y = ensure_float(y)
        out = _seasonal_naive(
            y=y,
            h=h,
            fitted=fitted or (level is not None),
            season_length=self.season_length,
        )
        res: ForecastDict = {"mean": out["mean"]}
        if fitted:
            res["fitted"] = out["fitted"]
        if level is not None:
            level = sorted(level)
            if self.prediction_intervals is not None:
                res = _add_conformal_intervals(self, fcst=res, y=y, X=X, level=level)
            else:
                k = jnp.floor(jnp.arange(h) / self.season_length)
                residuals = y - out["fitted"]
                sigma = calculate_sigma(residuals, len(y) - self.season_length)
                sigmah = sigma * jnp.sqrt(k + 1)
                pred_int = _calculate_intervals(out, level, h, sigmah)
                res = {**res, **pred_int}
            if fitted:
                residuals = y - out["fitted"]
                sigma = calculate_sigma(residuals, len(y) - self.season_length)
                res = _add_fitted_pi(res=res, se=sigma, level=level)
        return res

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> ForecastDict:
        """Apply the fitted model to a new or updated series.

        Args:
            y (jnp.ndarray): Clean time series of shape (n,).
            h (int): Forecast horizon.
            X (array-like): Optional in-sample exogenous of shape (t, n_x).
            X_future (array-like): Optional exogenous of shape (h, n_x).
            level (List[float]): Confidence levels (0-100) for prediction intervals.
            fitted (bool): Whether or not to return in-sample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        y = ensure_float(y)
        res = self.forecast(
            y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted
        )
        return res

# # Test Cases
# def test():
#     y = jnp.arange(24.0)

#     model = SeasonalNaive(season_length=12)
#     fitted_model = model.fit(y)

#     result = fitted_model.predict(h=12, level=(60,75))
#     forecast = fitted_model.forecast(y, h=12, level=[80, 95])
    
#     assert "mean" in result, "Missing mean forecast"

#     assert len(result["mean"]) == 12, "Forecast length mismatch"

#     for lvl in [60, 75]:
#         assert f"lo-{lvl}" in result, f"Missing lower bound for {lvl}% interval"
#         assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"
#         assert f"lo-{lvl}" in result, f"Missing lower bound for {lvl}% interval"
#         assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"

#     for lvl in [80, 95]:
#         assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
#         assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"
#         assert len(forecast[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
#         assert len(forecast[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

# if __name__ == "__main__":
#     test()
#     print("Test passed!")

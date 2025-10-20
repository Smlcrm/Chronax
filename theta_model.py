import jax
import jax.numpy as jnp
from jax import jit
from utils import (
    ensure_float,
    _add_fitted_pi,
    auto_theta,
    forward_theta,
    forecast_theta,
    _store_cs,
    _add_conformal_intervals,
    _add_predict_conformal_intervals,
    _conformal_method,
)
from base_forecaster import BaseForecaster
from typing import Optional, List
from conformal_intervals import ConformalIntervals

class AutoTheta(BaseForecaster):
    r"""AutoTheta model.

    Automatically selects the best Theta (Standard Theta Model ('STM'),
    Optimized Theta Model ('OTM'), Dynamic Standard Theta Model ('DSTM'),
    Dynamic Optimized Theta Model ('DOTM')) model using mse.

    Args:
        season_length (int, default=1): Number of observations per unit of time. Ex: 24 Hourly data.
        decomposition_type (str, default="multiplicative"): Sesonal decomposition type, 'multiplicative' (default) or 'additive'.
        model (Optional[str], optional): Controlling Theta Model. By default searchs the best model.
        alias (str, default="AutoTheta"): Custom name of the model.
        prediction_intervals (Optional[ConformalIntervals], optional): Information to compute conformal prediction intervals.
            By default, the model will compute the native prediction intervals.

    References:
        - [Jose A. Fiorucci, Tiago R. Pellegrini, Francisco Louzada, Fotios Petropoulos, Anne B. Koehler (2016). "Models for optimising the theta method and their relationship to state space models". International Journal of Forecasting](https://www.sciencedirect.com/science/article/pii/S0169207016300243)
    """
    
    def __init__(
    self,
    season_length: int = 1,
    decomposition_type: str = "multiplicative",
    model: Optional[str] = None,
    alias: str = "AutoTheta",
    prediction_intervals: Optional[ConformalIntervals] = None,
    conformal_params: Optional[ConformalIntervals] = None,
    ):
        self.season_length = season_length
        self.decomposition_type = decomposition_type
        self.model = model
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        if conformal_params is None:
            self.conformal_params = ConformalIntervals()
        else:
            if not isinstance(conformal_params, ConformalIntervals):
                raise TypeError(
                    "conformal_params must be a ConformalIntervals object."
                )
            self.conformal_params = conformal_params


    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ):
        r"""Fit the AutoTheta model.

        Fit an AutoTheta model to a time series (numpy array) `y`
        and optionally exogenous variables (numpy array) `X`.

        Args:
            y (numpy.array): Clean time series of shape (t, ).
            X (array-like, optional): Optional exogenous of shape (t, n_x).

        Returns:
            AutoTheta: AutoTheta fitted model.
        """
        y = ensure_float(y)
        self.model_ = auto_theta(
            y=y,
            m=self.season_length,
            model=self.model,
            decomposition_type=self.decomposition_type,
        )
        self.model_["fitted"] = y - self.model_["residuals"]
        _store_cs(self, y, X)
        return self

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ):
        r"""Predict with fitted AutoTheta.

        Args:
            h (int): Forecast horizon.
            X (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        fcst = forecast_theta(self.model_, h=h, level=level)
        if self.prediction_intervals is not None and level is not None:
            fcst = _add_predict_conformal_intervals(self, fcst, level)
        return fcst

    def predict_in_sample(self, level: Optional[List[int]] = None):
        r"""Access fitted AutoTheta insample predictions.

        Args:
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions and `level_*` for probabilistic predictions.
        """
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            se = jnp.std(self.model_["residuals"][3:], ddof=1)
            res = _add_fitted_pi(res=res, se=se, level=level)
        return res

    def forecast(
    self,
    y: jnp.ndarray,
    h: int,
    X: Optional[jnp.ndarray] = None,
    X_future: Optional[jnp.ndarray] = None,
    level: Optional[List[int]] = None,
    fitted: bool = False,):
        y = ensure_float(y)
        mod = auto_theta(
            y=y,
            m=self.season_length,
            model=self.model,
            decomposition_type=self.decomposition_type,
        )
        res = forecast_theta(mod, h, level=level)
        
        # Use the utils function directly
        if self.prediction_intervals is not None and level is not None:
            res = _add_conformal_intervals(self, fcst=res, y=y, X=X, level=level)
        
        if fitted:
            res["fitted"] = y - mod["residuals"]
        
        if level is not None and fitted:
            se = jnp.std(mod["residuals"][3:], ddof=1)
            res = _add_fitted_pi(res=res, se=se, level=level)
        
        return res

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        r"""Apply fitted AutoTheta to a new time series.

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like, optional): Optional insample exogenous of shape (t, n_x).
            X_future (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.
            fitted (bool, default=False): Whether or not to return insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        y = ensure_float(y)
        mod = forward_theta(self.model_, y=y)
        res = forecast_theta(mod, h, level=level)
        if self.prediction_intervals is not None:
            res = _add_conformal_intervals(self, fcst=res, y=y, X=X, level=level)
        if fitted:
            res["fitted"] = y - mod["residuals"]
        if level is not None and fitted:
            # add prediction intervals for fitted values
            se = jnp.std(mod["residuals"][3:], ddof=1)
            res = _add_fitted_pi(res=res, se=se, level=level)
        return res

class Theta(AutoTheta):
    r"""Standard Theta Method.

    References:
        - [Jose A. Fiorucci, Tiago R. Pellegrini, Francisco Louzada, Fotios Petropoulos, Anne B. Koehler (2016). "Models for optimising the theta method and their relationship to state space models". International Journal of Forecasting](https://www.sciencedirect.com/science/article/pii/S0169207016300243)

    Args:
        season_length (int): Number of observations per unit of time. Ex: 24 Hourly data. Default 1.
        decomposition_type (str): Sesonal decomposition type, 'multiplicative' (default) or 'additive'. Default 'multiplicative'.
        alias (str): Custom name of the model. Default 'Theta'.
        prediction_intervals (Optional[ConformalIntervals]): Information to compute conformal prediction intervals.
            By default, the model will compute the native prediction intervals. Default None.
    """

    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: str = "multiplicative",
        alias: str = "Theta",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        super().__init__(
            season_length=season_length,
            model="STM",
            decomposition_type=decomposition_type,
            alias=alias,
            prediction_intervals=prediction_intervals,
        )

# Test Cases
def test_autotheta():
    # simple increasing series
    y = jnp.arange(24.0)

    ci = ConformalIntervals(method="conformal_distribution")
    model = Theta(prediction_intervals=ci)

    # --- Fit the model ---
    fitted_model = model.fit(y)

    # --- Predict using fitted model ---
    result = fitted_model.predict(h=12, level=[60, 75])

    # --- Forecast directly from raw series ---
    forecast = fitted_model.forecast(y, h=12, level=[80, 95])

    # ---- Assertions ----
    assert "mean" in result, "Missing mean forecast"
    assert len(result["mean"]) == 12, "Forecast length mismatch"

    for lvl in [60, 75]:
        assert f"lo-{lvl}" in result, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"
        assert len(result[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
        assert len(result[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

    for lvl in [80, 95]:
        assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"
        assert len(forecast[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
        assert len(forecast[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

    print("AutoTheta test passed!")


def test_theta():
    # Basic variant using the fixed STM model
    y = jnp.arange(24.0)

    ci = ConformalIntervals(method="conformal_distribution")
    model = Theta(prediction_intervals=ci)

    fitted_model = model.fit(y)
    result = fitted_model.predict(h=6, level=[70, 90])
    forecast = fitted_model.forecast(y, h=6, level=[90, 95])

    assert "mean" in result, "Missing mean forecast"
    assert len(result["mean"]) == 6, "Forecast length mismatch"

    for lvl in [70, 90]:
        assert f"lo-{lvl}" in result, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"

    for lvl in [90, 95]:
        assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"

    print("Theta test passed!")


if __name__ == "__main__":
    test_autotheta()
    test_theta()

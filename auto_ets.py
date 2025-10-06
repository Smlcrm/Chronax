from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
from jax.scipy.special import ndtri  # JAX inverse normal CDF

from conformal_intervals import (
    ConformalIntervals,
)

from utils import ensure_float, _calculate_sigma, _quantiles

from base_forecaster import BaseForecaster

from ets_v2 import ets_f, forecast_ets, forward_ets

_PHI_LOWER = 0.8
_PHI_UPPER = 0.98



def _add_fitted_pi(res, se, level):
    level = sorted(level)
    level = jnp.asarray(level)
    quantiles = _quantiles(level=level)
    lo = res["fitted"].reshape(-1, 1) - quantiles * se.reshape(-1, 1)
    hi = res["fitted"].reshape(-1, 1) + quantiles * se.reshape(-1, 1)
    lo = lo[:, ::-1]
    lo = {f"fitted-lo-{l}": lo[:, i] for i, l in enumerate(reversed(level))}
    hi = {f"fitted-hi-{l}": hi[:, i] for i, l in enumerate(level)}
    res = {**res, **lo, **hi}
    return res

class AutoETS(BaseForecaster):
    r"""Automatic Exponential Smoothing model.

    Automatically selects the best ETS (Error, Trend, Seasonality)
    model using an information criterion. Default is Akaike Information Criterion (AICc), while particular models are estimated using maximum likelihood.
    The state-space equations can be determined based on their $M$ multiplicative, $A$ additive,
    $Z$ optimized or $N$ ommited components. The `model` string parameter defines the ETS equations:
    E in [$M, A, Z$], T in [$N, A, M, Z$], and S in [$N, A, M, Z$].

    For example when model='ANN' (additive error, no trend, and no seasonality), ETS will
    explore only a simple exponential smoothing.

    If the component is selected as 'Z', it operates as a placeholder to ask the AutoETS model
    to figure out the best parameter.

    Args:
        season_length (int, default=1): Number of observations per unit of time. Ex: 24 Hourly data.
        model (str, default="ZZZ"): Controlling state-space-equations.
        damped (bool, optional): A parameter that 'dampens' the trend.
        phi (float, optional): Smoothing parameter for trend damping. Only used when `damped=True`.
        alias (str, default="AutoETS"): Custom name of the model.
        prediction_intervals (Optional[ConformalIntervals], optional): Information to compute conformal prediction intervals.
            By default, the model will compute the native prediction intervals.

    Notes:
        This implementation is a mirror of Hyndman's [forecast::ets](https://github.com/robjhyndman/forecast).

    References:
        - [Rob J. Hyndman, Yeasmin Khandakar (2008). "Automatic Time Series Forecasting: The forecast package for R"](https://www.jstatsoft.org/article/view/v027i03).
        - [Hyndman, Rob, et al (2008). "Forecasting with exponential smoothing: the state space approach"](https://robjhyndman.com/expsmooth/).
    """

    def __init__(
        self,
        season_length: int = 1,
        model: str = "ZZZ",
        damped: Optional[bool] = None,
        phi: Optional[float] = None,
        alias: str = "AutoETS",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        self.season_length = season_length
        self.model = model
        self.damped = damped
        if phi is not None:
            if not isinstance(phi, float):
                raise ValueError("phi must be `None` or float.")
            if not _PHI_LOWER <= phi <= _PHI_UPPER:
                raise ValueError(f"Valid range for phi is [{_PHI_LOWER}, {_PHI_UPPER}]")
        self.phi = phi
        self.alias = alias
        self.conformal_params = prediction_intervals

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ):
        r"""Fit the Exponential Smoothing model.

        Fit an Exponential Smoothing model to a time series (numpy array) `y`
        and optionally exogenous variables (numpy array) `X`.

        Args:
            y (numpy.array): Clean time series of shape (t, ).
            X (array-like, optional): Optional exogenous of shape (t, n_x).

        Returns:
            AutoETS: Exponential Smoothing fitted model.
        """
        print("Fitting ETS model. This may take a while...")
        y = ensure_float(y)
        self.model_ = ets_f(
            y, m=self.season_length, model=self.model, damped=self.damped, phi=self.phi
        )
        self.model_["actual_residuals"] = y - self.model_["fitted"]
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None
        return self

    def predict(
        self, h: int, X: Optional[jnp.ndarray] = None, level: Optional[List[int]] = None
    ):
        r"""Predict with fitted Exponential Smoothing.

        Args:
            h (int): Forecast horizon.
            X (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        print("Predicting with ETS model.")
        fcst = forecast_ets(self.model_, h=h, level=level)
        res = {"mean": fcst["mean"]}
        if level is None:
            return res
        level = sorted(level)
        if self.conformal_params is not None:
            if self._cs is None:
                raise ValueError(
                    "Conformity scores not cached. Fit the model with conformal_params set, "
                    "or use forecast(y, ...) which recomputes them."
                )
            print("Using conformal prediction intervals.", self._cs)
            return self.add_confidence_intervals(
                fcst=res,
                cs=self._cs,
                level=level,
                method=self.conformal_params.method,
            )

        # Native ETS intervals
        res.update({f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level)})
        res.update({f"hi-{l}": fcst[f"hi-{l}"] for l in level})
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None):
        r"""Access fitted Exponential Smoothing insample predictions.

        Args:
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions and `level_*` for probabilistic predictions.
        """
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            residuals = self.model_["actual_residuals"]
            se = _calculate_sigma(residuals, len(residuals) - self.model_["n_params"])
            res = _add_fitted_pi(res=res, se=se, level=level)
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
        r"""Memory Efficient Exponential Smoothing predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like, optional): Optional insample exogenpus of shape (t, n_x).
            X_future (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.
            fitted (bool, default=False): Whether or not returns insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        y = ensure_float(y)
        mod = ets_f(
            y, m=self.season_length, model=self.model, damped=self.damped, phi=self.phi
        )
        fcst = forecast_ets(mod, h=h, level=level)
        keys = ["mean"]
        if fitted:
            keys.append("fitted")
        res = {key: fcst[key] for key in keys}

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
            else:
                res = {
                    **res,
                    **{f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level)},
                    **{f"hi-{l}": fcst[f"hi-{l}"] for l in level},
                }
            if fitted:
                # add prediction intervals for fitted values
                se = _calculate_sigma(y - mod["fitted"], len(y) - mod["n_params"])
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
        r"""Apply fitted Exponential Smoothing model to a new time series.

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like, optional): Optional insample exogenpus of shape (t, n_x).
            X_future (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels for prediction intervals.
            fitted (bool, default=False): Whether or not to return insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        y = ensure_float(y)
        mod = forward_ets(self.model_, y=y)
        fcst = forecast_ets(mod, h=h, level=level)
        keys = ["mean"]
        if fitted:
            keys.append("fitted")
        res = {key: fcst[key] for key in keys}

        if level is None:
            return res

        level = sorted(level)
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=X)  # recompute on the new series
            res = self.add_confidence_intervals(
                fcst=res, cs=cs, level=level, method=self.conformal_params.method
            )
        else:
            res.update({f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level)})
            res.update({f"hi-{l}": fcst[f"hi-{l}"] for l in level})

            if fitted:
                se = _calculate_sigma(y - mod["fitted"], len(y) - int(mod["n_params"]))
                res = _add_fitted_pi(res=res, se=se, level=level)
        return res

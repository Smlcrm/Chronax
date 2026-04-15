"""
AutoTheta and Theta — JAX-accelerated Theta method classes.

``AutoTheta`` automatically selects the best Theta model variant
(STM, OTM, DSTM, DOTM) by fitting all four and choosing by MSE.
``Theta`` is a convenience wrapper that always uses STM.

Both support additive and multiplicative seasonal decomposition,
Monte Carlo prediction intervals, and conformal prediction intervals.
The computational engine (state initialization, optimization, forecasting,
Monte Carlo sampling) lives in ``theta_model.py``.

References:
    Jose A. Fiorucci et al. (2016). "Models for optimising the theta method and
    their relationship to state space models". International Journal of Forecasting.
"""
import jax.numpy as jnp
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals
from chronax.models.theta.theta_model import (
    _auto_theta,
    _forecast_from_model,
    _forward_theta,
)
from chronax.utils import (
    ensure_float,
    _add_fitted_pi,
    _store_cs,
    _add_conformal_intervals,
    _add_predict_conformal_intervals,
)

__all__ = ['AutoTheta', 'Theta']


class AutoTheta(BaseForecaster):
    r"""AutoTheta model.

    Automatically selects the best Theta model variant (STM, OTM, DSTM, DOTM)
    using MSE.

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per unit of time.
    decomposition_type : str, default 'multiplicative'
        Seasonal decomposition type: 'multiplicative' or 'additive'.
    model : str or None, default None
        Controlling theta model variant. None searches the best model.
    alias : str, default 'AutoTheta'
        Custom name of the model.
    prediction_intervals : ConformalIntervals or None, default None
        Configuration for conformal prediction intervals.
    conformal_params : ConformalIntervals or None, default None
        Parameters for conformal prediction intervals.
    n_samples : int, default 200
        Number of Monte Carlo samples for prediction intervals.
    """

    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: str = "multiplicative",
        model: str | None = None,
        alias: str = "AutoTheta",
        prediction_intervals: ConformalIntervals | None = None,
        conformal_params: ConformalIntervals | None = None,
        n_samples: int = 200,
    ):
        self.season_length = season_length
        self.decomposition_type = decomposition_type
        self.model = model
        self.alias = alias
        self.n_samples = n_samples
        if conformal_params is not None and not isinstance(conformal_params, ConformalIntervals):
            raise TypeError("conformal_params must be a ConformalIntervals object.")
        if prediction_intervals is not None and not isinstance(prediction_intervals, ConformalIntervals):
            raise TypeError("prediction_intervals must be a ConformalIntervals object.")

        effective_cfg = conformal_params
        if effective_cfg is None:
            effective_cfg = prediction_intervals
        if effective_cfg is None:
            effective_cfg = ConformalIntervals()

        self.conformal_params = effective_cfg
        self.prediction_intervals = effective_cfg

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "AutoTheta":
        r"""Fit the AutoTheta model.

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (t,).
        X : jnp.ndarray or None, default None
            Optional exogenous of shape (t, n_x).

        Returns
        -------
        AutoTheta
            Fitted model instance.
        """
        y = ensure_float(y)
        self.model_ = _auto_theta(
            y=y,
            m=self.season_length,
            model=self.model,
            decomposition_type=self.decomposition_type,
        )
        if jnp.isnan(self.model_["mse"]):
            raise Exception("No model able to be fitted")
        self.model_["fitted"] = y - self.model_["residuals"]
        _store_cs(self, y, X)
        return self

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list | None = None,
    ) -> dict:
        r"""Predict with fitted AutoTheta.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Optional exogenous of shape (h, n_x).
        level : list of float or None, default None
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        dict
            Keys: 'mean' and optionally 'lo-{lv}', 'hi-{lv}'.
        """
        fcst = _forecast_from_model(self.model_, h=h, level=level, n_samples=self.n_samples)
        if self.prediction_intervals is not None and level is not None:
            fcst = _add_predict_conformal_intervals(self, fcst, level)
        return fcst

    def predict_in_sample(self, level: list | None = None) -> dict:
        r"""Access fitted AutoTheta in-sample predictions.

        Parameters
        ----------
        level : list of float or None, default None
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        dict
            Keys: 'fitted' and optionally 'lo-{lv}', 'hi-{lv}'.
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
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list | None = None,
        fitted: bool = False,
    ) -> dict:
        r"""Memory-efficient AutoTheta predictions.

        Fits and forecasts without storing model state.

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (t,).
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Optional exogenous of shape (t, n_x).
        X_future : jnp.ndarray or None, default None
            Optional future exogenous of shape (h, n_x).
        level : list of float or None, default None
            Confidence levels (0-100) for prediction intervals.
        fitted : bool, default False
            Whether to return in-sample predictions.

        Returns
        -------
        dict
            Keys: 'mean', and optionally 'fitted', 'lo-{lv}', 'hi-{lv}'.
        """
        y = ensure_float(y)
        mod = _auto_theta(
            y=y,
            m=self.season_length,
            model=self.model,
            decomposition_type=self.decomposition_type,
        )
        res = _forecast_from_model(mod, h, level=level, n_samples=self.n_samples)

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
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list | None = None,
        fitted: bool = False,
    ) -> dict:
        r"""Apply fitted AutoTheta model to a new time series.

        Uses the model type and parameters from the original fit.

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (n,).
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Optional exogenous of shape (n, n_x).
        X_future : jnp.ndarray or None, default None
            Optional future exogenous of shape (h, n_x).
        level : list of float or None, default None
            Confidence levels (0-100) for prediction intervals.
        fitted : bool, default False
            Whether to return in-sample predictions.

        Returns
        -------
        dict
            Keys: 'mean', and optionally 'fitted', 'lo-{lv}', 'hi-{lv}'.
        """
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        y = ensure_float(y)
        mod = _forward_theta(self.model_, y=y)
        res = _forecast_from_model(mod, h, level=level, n_samples=self.n_samples)
        if self.prediction_intervals is not None and level is not None:
            res = _add_conformal_intervals(self, fcst=res, y=y, X=X, level=level)
        if fitted:
            res["fitted"] = y - mod["residuals"]
        if level is not None and fitted:
            se = jnp.std(mod["residuals"][3:], ddof=1)
            res = _add_fitted_pi(res=res, se=se, level=level)
        return res


class Theta(AutoTheta):
    r"""Standard Theta Method (STM).

    A simplified version of AutoTheta that always uses the Standard Theta Model.

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per unit of time.
    decomposition_type : str, default 'multiplicative'
        Seasonal decomposition type: 'multiplicative' or 'additive'.
    alias : str, default 'Theta'
        Custom name of the model.
    prediction_intervals : ConformalIntervals or None, default None
        Configuration for conformal prediction intervals.
    n_samples : int, default 200
        Number of Monte Carlo samples for prediction intervals.
    """

    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: str = "multiplicative",
        alias: str = "Theta",
        prediction_intervals: ConformalIntervals | None = None,
        n_samples: int = 200,
    ):
        super().__init__(
            season_length=season_length,
            model="STM",
            decomposition_type=decomposition_type,
            alias=alias,
            prediction_intervals=prediction_intervals,
            n_samples=n_samples,
        )

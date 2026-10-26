"""
AutoTheta and Theta: JAX-accelerated Theta method classes.

``AutoTheta`` fits the four Theta model variants, the standard Theta method
(STM), the optimized Theta method (OTM), the dynamic standard Theta method
(DSTM) and the dynamic optimized Theta method (DOTM), and keeps the one with
the lowest mean squared error (MSE). ``Theta`` is a convenience wrapper that
always uses STM.

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

    Fits the Theta model variants (STM, OTM, DSTM, DOTM) and keeps the one
    with the lowest MSE. When ``season_length >= 4``, the series has at least
    two full seasons and it tests as seasonal, the series is deseasonalised
    before fitting and the forecast is reseasonalised. Prediction intervals come from Monte Carlo simulation of
    the fitted model, or from conformal scores when ``prediction_intervals``
    is set. The model ignores exogenous regressors.

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per unit of time.
    decomposition_type : str, default 'multiplicative'
        Seasonal decomposition type: 'multiplicative' or 'additive'.
    model : str or None, default None
        Theta variant to fit: 'STM', 'OTM', 'DSTM' or 'DOTM'. None fits all
        four and keeps the one with the lowest MSE.
    alias : str, default 'AutoTheta'
        Custom name of the model.
    prediction_intervals : ConformalIntervals or None, default None
        When set, ``fit`` computes conformal scores, and ``predict`` and
        ``forecast`` return conformal intervals in place of the Monte Carlo
        intervals. The window layout comes from ``conformal_params``.
    conformal_params : ConformalIntervals or None, default None
        Window configuration for the base-class ``conformity_scores`` path.
        None uses ``ConformalIntervals()`` with its defaults.
    n_samples : int, default 200
        Number of Monte Carlo sample paths for prediction intervals.

    Raises
    ------
    TypeError
        If ``conformal_params`` is not a ``ConformalIntervals`` instance.
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
        self.prediction_intervals = prediction_intervals
        self.n_samples = n_samples
        if conformal_params is None:
            self.conformal_params = ConformalIntervals()
        else:
            if not isinstance(conformal_params, ConformalIntervals):
                raise TypeError("conformal_params must be a ConformalIntervals object.")
            self.conformal_params = conformal_params

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "AutoTheta":
        r"""Fit the AutoTheta model.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate time series of shape ``(T,)``.
        X : jnp.ndarray or None, default None
            Ignored by the model. Forwarded only to the conformal
            cross-validation splitter when ``prediction_intervals`` is set.

        Returns
        -------
        AutoTheta
            Fitted model instance (``self``).

        Raises
        ------
        ValueError
            If ``model`` is not one of 'STM', 'OTM', 'DSTM', 'DOTM'.
        NotImplementedError
            If the series has 3 or fewer observations.
        Exception
            If no variant produces a finite MSE.
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
        r"""Predict with the fitted AutoTheta model.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Ignored.
        level : list of int or None, default None
            Confidence levels in (0, 100), e.g. ``[80, 95]``.

        Returns
        -------
        dict
            ``'mean'`` of shape ``(h,)``, plus ``'lo-{lv}'`` and
            ``'hi-{lv}'`` of shape ``(h,)`` per level when ``level`` is given
            (Monte Carlo intervals, or conformal intervals when
            ``prediction_intervals`` is set).

        Raises
        ------
        ValueError
            If called before ``fit``.
        """
        self._require_fitted()
        fcst = _forecast_from_model(self.model_, h=h, level=level, n_samples=self.n_samples)
        if self.prediction_intervals is not None and level is not None:
            fcst = _add_predict_conformal_intervals(self, fcst, level)
        return fcst

    def predict_in_sample(self, level: list | None = None) -> dict:
        r"""Return in-sample fitted values of the fitted AutoTheta model.

        Requires a prior ``fit``.

        Parameters
        ----------
        level : list of int or None, default None
            Confidence levels in (0, 100), e.g. ``[80, 95]``.

        Returns
        -------
        dict
            ``'fitted'`` of shape ``(T,)``, plus ``'fitted-lo-{lv}'`` and
            ``'fitted-hi-{lv}'`` per level when ``level`` is given.
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
        r"""Stateless fit and forecast in one call.

        Fits on ``y`` and forecasts without storing model state on ``self``.
        This is the method the base-class ``conformity_scores`` maps over
        cross-validation windows.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate time series of shape ``(T,)``.
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Ignored by the model. Forwarded only to the conformal
            cross-validation splitter when ``prediction_intervals`` is set.
        X_future : jnp.ndarray or None, default None
            Ignored.
        level : list of int or None, default None
            Confidence levels in (0, 100), e.g. ``[80, 95]``.
        fitted : bool, default False
            If True, also return in-sample fitted values.

        Returns
        -------
        dict
            ``'mean'`` of shape ``(h,)``; ``'lo-{lv}'`` and ``'hi-{lv}'`` per
            level when ``level`` is given; ``'fitted'`` of shape ``(T,)``
            when ``fitted`` is True; ``'fitted-lo-{lv}'`` and
            ``'fitted-hi-{lv}'`` when both are given.

        Raises
        ------
        ValueError
            If ``model`` is not one of 'STM', 'OTM', 'DSTM', 'DOTM'.
        NotImplementedError
            If the series has 3 or fewer observations.
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
        r"""Apply the fitted AutoTheta model to a new time series.

        Re-fits on ``y`` with the model variant and parameters of the
        original fit, then forecasts.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate time series of shape ``(T,)``.
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Ignored by the model. Forwarded only to the conformal
            cross-validation splitter when ``prediction_intervals`` is set.
        X_future : jnp.ndarray or None, default None
            Ignored.
        level : list of int or None, default None
            Confidence levels in (0, 100), e.g. ``[80, 95]``.
        fitted : bool, default False
            If True, also return in-sample fitted values.

        Returns
        -------
        dict
            Same keys as :meth:`forecast`.

        Raises
        ------
        Exception
            If called before ``fit``.
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
    r"""Standard Theta method (STM).

    :class:`AutoTheta` with ``model='STM'`` fixed. ``fit``, ``predict``,
    ``predict_in_sample``, ``forecast`` and ``forward`` are inherited
    unchanged; ``conformal_params`` is always ``ConformalIntervals()``.

    Parameters
    ----------
    season_length : int, default 1
        Number of observations per unit of time.
    decomposition_type : str, default 'multiplicative'
        Seasonal decomposition type: 'multiplicative' or 'additive'.
    alias : str, default 'Theta'
        Custom name of the model.
    prediction_intervals : ConformalIntervals or None, default None
        When set, intervals are conformal instead of Monte Carlo (see
        :class:`AutoTheta`).
    n_samples : int, default 200
        Number of Monte Carlo sample paths for prediction intervals.
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

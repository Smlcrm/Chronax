
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
    """Seasonal naive model.

    Each forecast equals the last observation of the same seasonal position
    (for example the same month of the previous year). Native intervals use the
    residual standard error scaled by ``sqrt(k + 1)``, where ``k`` is the
    number of completed seasons in the horizon. Forecast values are float32.

    References:
        - [Rob J. Hyndman and George Athanasopoulos (2018). "forecasting principles and practice, Simple Methods"](https://otexts.com/fpp3/simple-methods.html#seasonal-na%C3%AFve-method).

    Args:
        season_length (int): Number of observations per seasonal cycle, e.g.
            24 for hourly data with a daily cycle.
        alias (str): Custom name of the model. Defaults to
            ``"SeasonalNaive"``.
        prediction_intervals (Optional[ConformalIntervals]): Configuration for
            conformal prediction intervals. When ``None``, the model computes
            native intervals. Defaults to ``None``.
    """

    def __init__(
        self,
        season_length: int,
        alias: str = "SeasonalNaive",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ) -> None:
        self.season_length = season_length
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        # BaseForecaster.conformity_scores reads conformal_params.
        self.conformal_params = prediction_intervals

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "SeasonalNaive":
        """Fit the SeasonalNaive model.

        Stores the last season, the in-sample fitted values and the residual
        standard error in ``model_``. When ``prediction_intervals`` is set,
        conformity scores are computed and cached for :meth:`predict`.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (Optional[jnp.ndarray]): Exogenous regressors, shape ``(T, F)``.
                Only forwarded to ``conformity_scores``. Defaults to ``None``.

        Returns:
            SeasonalNaive: ``self``.

        Raises:
            ValueError: If ``y`` is not 1-D, has fewer than ``season_length``
                observations, or ``season_length <= 0``; or if
                ``prediction_intervals`` is set and the series is too short
                for 2 cross-validation (CV) windows.
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
        """Predict with the fitted SeasonalNaive model.

        Args:
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (Optional[List[int]]): Confidence levels in (0, 100), e.g.
                ``[80, 95]``. Defaults to ``None``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``) plus ``"lo-{lv}"``/``"hi-{lv}"``
            per level when ``level`` is given (conformal when
            ``prediction_intervals`` is set, native otherwise).

        Raises:
            ValueError: If called before ``fit``.
        """
        self._require_fitted()
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
        """Return the in-sample fitted values of the fitted model.

        Args:
            level (Optional[List[int]]): Confidence levels in (0, 100).
                Defaults to ``None``.

        Returns:
            dict: ``"fitted"`` (shape ``(T,)``, NaN for the first season)
            plus ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"`` per level when
            ``level`` is given.
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
        """Stateless fit+predict in one call.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): In-sample exogenous regressors, shape
                ``(T, F)``. Only forwarded to ``conformity_scores``. Defaults
                to ``None``.
            X_future (Optional[jnp.ndarray]): Future exogenous regressors,
                shape ``(h, F)``. Ignored. Defaults to ``None``.
            level (Optional[List[int]]): Confidence levels in (0, 100).
                Defaults to ``None``.
            fitted (bool): Whether to return in-sample fitted values under
                ``"fitted"``. Defaults to ``False``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``); ``"fitted"`` (shape ``(T,)``)
            when ``fitted``; ``"lo-{lv}"``/``"hi-{lv}"`` per level when
            ``level`` is given, plus ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"``
            when both are set.

        Raises:
            ValueError: If ``y`` is not 1-D, has fewer than ``season_length``
                observations, or ``h < 1``; or if the conformal path finds
                the series too short for 2 CV windows.
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
        """Apply the model to a new series; delegates to :meth:`forecast`.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): In-sample exogenous regressors, shape
                ``(T, F)``. Defaults to ``None``.
            X_future (Optional[jnp.ndarray]): Future exogenous regressors,
                shape ``(h, F)``. Ignored. Defaults to ``None``.
            level (Optional[List[int]]): Confidence levels in (0, 100).
                Defaults to ``None``.
            fitted (bool): Whether to return in-sample fitted values. Defaults
                to ``False``.

        Returns:
            dict: Same keys as :meth:`forecast`.
        """
        y = ensure_float(y)
        res = self.forecast(
            y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted
        )
        return res

"""
HistoricAverage forecaster: every forecast equals the mean of the series.

The jitted ``_historic_average`` computes the mean forecast and, optionally,
the fitted values (all equal to the mean). Native intervals use the residual
standard error scaled by ``sqrt(1 + 1/n)``, constant over the horizon;
conformal intervals come from ``BaseForecaster`` when ``conformal_params`` is
set. Forecast values are float32.
"""

import jax
import jax.numpy as jnp

from chronax import utils
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals


# ---------- Core: Single jitted function does everything ----------

from functools import partial

@partial(jax.jit, static_argnums=(1, 2))
def _historic_average(y: jnp.ndarray, h: int, fitted: bool = False) -> dict:
    """Core HistoricAverage computation (pure, jitted).

    Args:
        y (jnp.ndarray): Univariate series, shape ``(T,)``.
        h (int): Forecast horizon (static).
        fitted (bool): Whether to return fitted values (static).

    Returns:
        dict: ``"mean"`` (shape ``(h,)``, float32) and optionally ``"fitted"``
        (shape ``(T,)``).
    """
    mean_val = jnp.nanmean(y)
    fcst = {"mean": jnp.full((h,), mean_val, dtype=jnp.float32)}
    if fitted:
        fcst["fitted"] = jnp.full_like(y, mean_val)
    return fcst


# ---------- Model class ----------

class HistoricAverage(BaseForecaster):
    """HistoricAverage model.

    Every forecast equals the mean of all observations. Native intervals use
    ``sigma_h = sigma * sqrt(1 + 1/n)`` with normal quantiles.

    Args:
        alias (str): Custom name of the model. Defaults to
            ``"HistoricAverage"``.
        conformal_params (ConformalIntervals | None): Configuration for
            conformal prediction intervals. When ``None``, native intervals
            are used. Defaults to ``None``.
    """

    uses_exog = False

    def __init__(self, alias: str = "HistoricAverage", conformal_params: ConformalIntervals | None = None) -> None:
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = {}

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "HistoricAverage":
        r"""Fit the HistoricAverage model.

        Stores the mean, the fitted values, the residual standard error and
        the series length in ``model_``. When ``conformal_params`` is set,
        conformity scores are computed and cached for :meth:`predict`.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (jnp.ndarray | None): Exogenous regressors, shape ``(T, F)``.
                Ignored. Defaults to ``None``.

        Returns:
            HistoricAverage: ``self``.

        Raises:
            ValueError: If ``conformal_params`` is set and the series is too
                short for 2 cross-validation (CV) windows.
        """
        y = utils.ensure_float(y)
        mod = _historic_average(y, h=1, fitted=True)
        residuals = y - mod["fitted"]
        
        self.model_ = {
            "mean": mod["mean"],
            "fitted": mod["fitted"],
            "sigma": utils.calculate_sigma(residuals, len(y) - 1),
            "n": len(y),
        }
        # Score the real series, not the fitted values: the fitted values are a
        # constant series whose CV residuals are ~0 (zero-width intervals).
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y)
        return self

    def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int] | None = None) -> dict:
        r"""Predict with the fitted HistoricAverage model.

        Args:
            h (int): Forecast horizon.
            X (jnp.ndarray | None): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (list[int] | None): Confidence levels in (0, 100), e.g.
                ``[80, 95]``. Defaults to ``None``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``, float32, all equal to the mean)
            plus ``"lo-{lv}"``/``"hi-{lv}"`` per level when ``level`` is given.

        Raises:
            ValueError: If called before ``fit``.
        """
        self._require_fitted()
        mean = jnp.full((h,), self.model_["mean"][0], dtype=jnp.float32)
        res = {"mean": mean}
        
        if level is not None:
            level = sorted(level)
            sigma, n = self.model_["sigma"], self.model_["n"]
            sigmah = sigma * jnp.sqrt(1.0 + (1.0 / n))
            
            if self.conformal_params is not None:
                res = self.add_confidence_intervals(
                    res, self._cs, level, self.conformal_params.method
                )
            else:
                res = {**res, **utils._calculate_intervals(res, level, h, sigmah)}
        return res

    def predict_in_sample(self, level: list[int] | None = None) -> dict:
        r"""Return the in-sample fitted values of the fitted model.

        All fitted values equal the mean. Fitted intervals use
        ``sigma * sqrt(1 + 1/n)`` with normal quantiles.

        Args:
            level (list[int] | None): Confidence levels in (0, 100). Defaults
                to ``None``.

        Returns:
            dict: ``"fitted"`` (shape ``(T,)``) plus
            ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"`` per level when ``level``
            is given.
        """
        res = {"fitted": self.model_["fitted"]}
        
        if level is not None:
            sigmah = self.model_["sigma"] * jnp.sqrt(1.0 + (1.0 / self.model_["n"]))
            res = utils._add_fitted_pi(res, sigmah, sorted(level))
        
        return res

    def forecast(
        self, y: jnp.ndarray, h: int, X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None, level: list[int] | None = None, fitted: bool = False
    ) -> dict:
        r"""Stateless fit+predict in one call.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon.
            X (jnp.ndarray | None): In-sample exogenous regressors, shape
                ``(T, F)``. Ignored. Defaults to ``None``.
            X_future (jnp.ndarray | None): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (list[int] | None): Confidence levels in (0, 100). Defaults
                to ``None``.
            fitted (bool): Whether to return in-sample fitted values under
                ``"fitted"``. Defaults to ``False``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``, float32); ``"fitted"`` (shape
            ``(T,)``) when ``fitted``; ``"lo-{lv}"``/``"hi-{lv}"`` per level
            when ``level`` is given, plus ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"``
            when both are set.

        Raises:
            ValueError: If the conformal path finds the series too short for
                2 CV windows.
        """
        y = utils.ensure_float(y)
        out = _historic_average(y, h, fitted or (level is not None))
        res = {"mean": out["mean"]}
        
        if fitted:
            res["fitted"] = out["fitted"]
        
        if level is not None:
            level = sorted(level)
            residuals = y - out["fitted"]
            sigma = utils.calculate_sigma(residuals, len(y) - 1)
            sigmah = sigma * jnp.sqrt(1.0 + (1.0 / len(y)))
            
            if self.conformal_params is not None:
                res = self.add_confidence_intervals(
                    res, self.conformity_scores(y, X=None), level,
                    self.conformal_params.method
                )
            else:
                res = {**res, **utils._calculate_intervals(out,level, h, sigmah)}
            
            if fitted:
                res = utils._add_fitted_pi(res, sigmah, level)
        
        return res

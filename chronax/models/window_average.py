r"""WindowAverage forecasting model using JAX.

This module implements a WindowAverage forecasting model that predicts future values
as the mean of the last k observations, where k is the window_size parameter.
Supports conformal prediction intervals via BaseForecaster.

The WindowAverage method is a simple but effective forecasting technique that computes
the average of the most recent observations within a specified window. Wider windows
capture global trends, while narrower windows reveal local trends.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax

from chronax.utils import ConformalIntervals

from chronax.models.base_forecaster import BaseForecaster

from chronax.utils import _repeat_val, _window_average, ensure_float


class WindowAverage(BaseForecaster):
    r"""WindowAverage model.

    Forecasts the average of the last $k$ observations, with $k$ the window
    length. Wider windows capture global trends, narrower windows local ones.
    Intervals are conformal only. Fitted values are not supported: there is no
    ``predict_in_sample`` and ``forecast(fitted=True)`` raises.

    References:
        - [Rob J. Hyndman and George Athanasopoulos (2018). "forecasting principles and practice, Simple Methods"](https://otexts.com/fpp3/simple-methods.html).

    Args:
        window_size (int): Number of trailing observations to average.
        alias (str): Custom name of the model. Defaults to
            ``"WindowAverage"``.
        conformal_params (Optional[ConformalIntervals]): Configuration for
            conformal prediction intervals; required for ``level``. Defaults
            to ``None``.
    """

    def __init__(self, window_size: int, alias: str = "WindowAverage",
        conformal_params: Optional[ConformalIntervals] = None) -> None:
        self.window_size = window_size
        self.alias = alias
        self.conformal_params = conformal_params
        self.only_conformal_intervals = True

    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "WindowAverage":
        r"""Fit the WindowAverage model.

        Stores the window mean in ``model_``. When ``conformal_params`` is
        set, conformity scores are computed and cached for :meth:`predict`.
        A series shorter than ``window_size`` yields a NaN mean.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (Optional[jnp.ndarray]): Exogenous regressors, shape ``(T, F)``.
                Only forwarded to ``conformity_scores``. Defaults to ``None``.

        Returns:
            WindowAverage: ``self``.

        Raises:
            ValueError: If ``conformal_params`` is set and the series is too
                short for 2 cross-validation (CV) windows.
        """

        y = ensure_float(y)
        mod = _window_average(y=y, h=1, window_size=self.window_size, fitted=False)
        self.model_ = dict(mod)
        # Cache conformity scores for predict(); the base class does not persist _cs.
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None
        return self
    

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ) -> Dict[str, jnp.ndarray]:
        r"""Predict with the fitted WindowAverage model.

        Args:
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (Optional[List[int]]): Confidence levels in (0, 100), e.g.
                ``[80, 95]``. Requires ``conformal_params``. Defaults to
                ``None``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``) plus ``"lo-{lv}"``/``"hi-{lv}"``
            per level when ``level`` is given.

        Raises:
            ValueError: If called before ``fit``; if ``level`` is given
                without ``conformal_params``; or if no conformity scores are
                cached.
        """
        self._require_fitted()
        mean = _repeat_val(self.model_["mean"][0], h=h) 
        res = {"mean": mean}
        if level is None:
            return res
        level = sorted(level)
        if self.conformal_params is None:
            raise ValueError("You must pass `conformal_params` to compute intervals.")

        if self._cs is None:
            raise ValueError(
                "Conformity scores are not available. Fit the model first (fit(...)) "
                "with `conformal_params` set so predict() can use cached scores."
            )

        return BaseForecaster.add_confidence_intervals(
            fcst=res,
            cs=self._cs,
            level=list(level),
            method=self.conformal_params.method,
        )

    
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        r"""Stateless fit+predict in one call.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): In-sample exogenous regressors, shape
                ``(T, F)``. Ignored. Defaults to ``None``.
            X_future (Optional[jnp.ndarray]): Future exogenous regressors,
                shape ``(h, F)``. Ignored. Defaults to ``None``.
            level (Optional[List[int]]): Confidence levels in (0, 100).
                Requires ``conformal_params``. Defaults to ``None``.
            fitted (bool): Must be ``False``; fitted values are not
                supported. Defaults to ``False``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``, NaN when ``T < window_size``)
            plus ``"lo-{lv}"``/``"hi-{lv}"`` per level when ``level`` is
            given.

        Raises:
            NotImplementedError: If ``fitted`` is ``True``.
            Exception: If ``level`` is given without ``conformal_params``.
            ValueError: If ``level`` is given and the series is too short for
                2 CV windows.
        """
        y = ensure_float(y)
        res = _window_average(y=y, h=h, fitted=fitted, window_size=self.window_size)
        res = dict(res)
        if level is None:
            return res
        level = sorted(level)
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=None)
            res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
        else:
            raise Exception("You must pass `conformal_params` to compute them.")
        return res
    

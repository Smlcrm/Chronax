"""
Random walk with drift: the naive forecast plus a linear drift.

Port of statsforecast's ``RandomWalkWithDrift``. The forecast is
``y[T] + h * slope`` with ``slope = (y[T] - y[1]) / (T - 1)``. ``_rwd_core``
computes the forecasts, fitted values, slope and residual standard error.
Native intervals use the drift-adjusted variance ``h * (1 + h / (n - 1))``;
conformal intervals come from ``BaseForecaster`` when ``conformal_params`` is
set.
"""

import jax
import jax.numpy as jnp
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals
from chronax import utils

class RandomWalkWithDrift(BaseForecaster):
    """Random walk with drift model.

    Args:
        alias (str): Custom name of the model. Defaults to ``"RWD"``.
        conformal_params (ConformalIntervals | None): Configuration for
            conformal prediction intervals. When ``None``, native intervals
            (drift-adjusted residual standard error, normal quantiles) are
            used. Defaults to ``None``.

    Note:
        Two signature quirks relative to ``BaseForecaster``: ``forecast``
        takes ``(h, y, ...)`` in that order, and ``predict`` has no ``X``
        parameter. Call ``forecast`` with keyword arguments.
    """

    def __init__(
        self,
        alias: str = "RWD",
        conformal_params: ConformalIntervals | None = None,
    ):
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = None

    @staticmethod
    def _rwd_core(y: jnp.ndarray, h: int) -> dict:
        """Core random walk with drift computation.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon.

        Returns:
            dict: ``"mean"``, ``"fitted"``, ``"slope"``, ``"last_y"``,
            ``"sigma"`` and ``"n"``.
        """
        slope = (y[-1] - y[0]) / (y.size - 1)

        # y_T + k * slope for k = 1, ..., h
        forecast_steps = jnp.arange(1, h + 1, dtype=y.dtype)
        forecasts = y[-1] + slope * forecast_steps

        # fitted[t] = y[t-1] + slope; the first value is NaN
        fitted_vals = jnp.concatenate([
            jnp.array([jnp.nan]),
            slope + y[:-1]
        ])

        residuals = y - fitted_vals
        sigma = utils.calculate_sigma(residuals, len(residuals) - 1)

        dictionary = {
            'mean': forecasts,
            'fitted': fitted_vals,
            'slope': slope,
            'last_y': y[-1],
            'sigma': sigma,
            'n': len(y)
        }
        return dictionary

    _rwd_core_jit = jax.jit(_rwd_core.__func__, static_argnums=(1,))

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None):
        """Fit the RandomWalkWithDrift model.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (jnp.ndarray | None): Exogenous regressors, shape ``(T, F)``.
                Ignored. Defaults to ``None``.

        Returns:
            RandomWalkWithDrift: ``self``.
        """
        y = utils.ensure_float(y)
        mod = RandomWalkWithDrift._rwd_core_jit(y, h=1)
        mod['y_train'] = y  # Store for conformal prediction
        self.model_ = mod
        return self

    def predict(self, h: int, level: list[int] | None = None):
        """Predict with the fitted RandomWalkWithDrift model.

        This method has no ``X`` parameter.

        Args:
            h (int): Forecast horizon.
            level (list[int] | None): Confidence levels in (0, 100), e.g.
                ``[80, 95]``. Defaults to ``None``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``) plus ``"lo-{lv}"``/``"hi-{lv}"``
            per level when ``level`` is given.

        Raises:
            ValueError: If called before ``fit``, or if the conformal path
                finds the training series too short for 2 cross-validation
                (CV) windows.
        """
        self._require_fitted()
        forecast_steps = jnp.arange(1, h + 1, dtype=self.model_["last_y"].dtype)
        mean = self.model_["last_y"] + self.model_["slope"] * forecast_steps
        res = {"mean": mean}

        if level is None:
            return res

        level = sorted(level)
        if self.conformal_params is not None:
            # Use base class conformal prediction
            cs = self.conformity_scores(y=self.model_['y_train'], X=None)
            res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
        else:
            # Native prediction intervals for random walk with drift
            steps = jnp.arange(1, h + 1)
            sigma = self.model_["sigma"]
            # Drift-adjusted variance: steps * (1 + steps / (n-1))
            sigmah = sigma * jnp.sqrt(steps * (1 + steps / (self.model_["n"] - 1)))
            pred_int = self._calculate_rwd_intervals(res, level, h, sigmah)
            res = {**res, **pred_int}
        return res

    def predict_in_sample(self, level: list[int] | None = None):
        """Return the in-sample fitted values of the fitted model.

        Args:
            level (list[int] | None): Confidence levels in (0, 100). Defaults
                to ``None``.

        Returns:
            dict: ``"fitted"`` (shape ``(T,)``, first value NaN) plus
            ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"`` per level when ``level``
            is given.
        """
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            level = sorted(level)
            res = self._add_rwd_fitted_intervals(res=res, se=self.model_["sigma"], level=level)
        return res

    def forecast(
        self,
        h: int,
        y: jnp.ndarray,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int] | None = None,
        fitted: bool = False,
    ):
        """Stateless fit+predict in one call.

        Takes ``h`` before ``y``, the reverse of the ``BaseForecaster``
        signature; call with keyword arguments.

        Args:
            h (int): Forecast horizon.
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (jnp.ndarray | None): In-sample exogenous regressors, shape
                ``(T, F)``. Ignored. Defaults to ``None``.
            X_future (jnp.ndarray | None): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (list[int] | None): Confidence levels in (0, 100). Defaults
                to ``None``.
            fitted (bool): Whether to return in-sample fitted values under
                ``"fitted"``. Defaults to ``False``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``); ``"fitted"`` (shape ``(T,)``)
            when ``fitted``; ``"lo-{lv}"``/``"hi-{lv}"`` per level when
            ``level`` is given, plus ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"``
            when both are set.

        Raises:
            ValueError: If the conformal path finds the series too short for
                2 CV windows.
        """
        y = utils.ensure_float(y)
        out = RandomWalkWithDrift._rwd_core_jit(y=y, h=h)
        res = {"mean": out["mean"]}

        if fitted:
            res["fitted"] = out["fitted"]

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                # Use base class conformal prediction
                cs = self.conformity_scores(y=y, X=None)
                res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
            else:
                # Native prediction intervals for random walk with drift
                steps = jnp.arange(1, h + 1)
                sigma = out["sigma"]
                # Drift-adjusted variance: steps * (1 + steps / (n-1))
                sigmah = sigma * jnp.sqrt(steps * (1 + steps / (len(y) - 1)))
                pred_int = self._calculate_rwd_intervals(res, level, h, sigmah)
                res = {**res, **pred_int}
            if fitted:
                sigma = out["sigma"]
                res = self._add_rwd_fitted_intervals(res=res, se=sigma, level=level)
        return res

    def _calculate_rwd_intervals(self, res, level, h, sigmah):
        """Calculate native prediction intervals for random walk with drift using JAX operations."""
        z_scores = jnp.array([utils._jax_norm_ppf(0.5 + lv / 200) for lv in level])
        mean = res["mean"]

        z_lower = z_scores[::-1]  # lower bounds use reversed level order
        z_upper = z_scores

        # mean (h,) x z_scores (n_levels,) -> (h, n_levels)
        lower_bounds = mean[:, None] - z_lower * sigmah[:, None]
        upper_bounds = mean[:, None] + z_upper * sigmah[:, None]

        intervals = {}
        for i, lv in enumerate(reversed(level)):
            intervals[f"lo-{lv}"] = lower_bounds[:, i]

        # Upper bounds
        for i, lv in enumerate(level):
            intervals[f"hi-{lv}"] = upper_bounds[:, i]

        return intervals

    def _add_rwd_fitted_intervals(self, res, se, level):
        """Add fitted prediction intervals for random walk with drift."""
        fitted = res["fitted"]

        # For fitted intervals, use constant standard error
        for lv in level:
            z = utils._jax_norm_ppf(0.5 + lv / 200)
            res[f"fitted-lo-{lv}"] = fitted - z * se
            res[f"fitted-hi-{lv}"] = fitted + z * se

        return res
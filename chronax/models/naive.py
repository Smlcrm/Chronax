"""
Naive forecaster: every forecast equals the last observed value.

Port of statsforecast's ``Naive``. ``_naive_core`` computes the forecasts, the
one-step fitted values and the residual standard error. Native intervals use
the residual standard error with a normal assumption and grow as ``sqrt(h)``;
conformal intervals come from ``BaseForecaster`` when ``conformal_params`` is
set.
"""

import jax
import jax.numpy as jnp
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals
from chronax import utils

class Naive(BaseForecaster):
    """Naive model: every forecast equals the last observed value.

    Args:
        alias (str): Custom name of the model. Defaults to ``"Naive"``.
        conformal_params (ConformalIntervals | None): Configuration for
            conformal prediction intervals. When ``None``, native intervals
            (residual standard error, normal quantiles) are used. Defaults to
            ``None``.

    Note:
        ``forecast`` and ``forward`` take ``(h, y, ...)`` in that order, the
        reverse of the ``BaseForecaster`` signature. Call them with keyword
        arguments.
    """

    def __init__(
        self,
        alias: str = "Naive",
        conformal_params: ConformalIntervals | None = None,
    ):
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = None

    @staticmethod
    def _naive_core(y: jnp.ndarray, h: int) -> dict:
        last_value = y[-1]
        forecasts = jnp.full(h, last_value, dtype=y.dtype)
        fitted_vals = jnp.concatenate([
            jnp.array([jnp.nan]),
            y[:-1]
        ])
        residuals = y - fitted_vals
        sigma = utils.calculate_sigma(residuals, len(residuals) - 1)
        dictionary = {'mean': forecasts, 'fitted': fitted_vals, 'sigma': sigma, 'last_y': y[-1]}
        return dictionary

    _naive_core_jit = jax.jit(_naive_core.__func__, static_argnums=(1,))

    def fit(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray | None = None,
    ):
        """Fit the Naive model.

        Stores the last value, the one-step fitted values, the residual
        standard error and the training series in ``model_``.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (jnp.ndarray | None): Exogenous regressors, shape ``(T, F)``.
                Ignored. Defaults to ``None``.

        Returns:
            Naive: ``self``.
        """
        y = utils.ensure_float(y)
        mod = Naive._naive_core_jit(y, h=1)
        mod['y_train'] = y  # Store for conformal prediction
        self.model_ = mod
        return self

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int] | None = None,
    ):
        """Predict with the fitted Naive model.

        Args:
            h (int): Forecast horizon.
            X (jnp.ndarray | None): Future exogenous regressors, shape
                ``(h, F)``. Only forwarded to ``conformity_scores``; otherwise
                ignored. Defaults to ``None``.
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
        mean = jnp.full(h, self.model_["last_y"], dtype=self.model_["last_y"].dtype)
        res = {"mean": mean}

        if level is None:
            return res

        level = sorted(level)
        if self.conformal_params is not None:
            # Use base class conformal prediction
            cs = self.conformity_scores(y=self.model_['y_train'], X=X)
            res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
        else:
            # Native prediction intervals for naive model
            steps = jnp.arange(1, h + 1)
            sigma = self.model_["sigma"]
            sigmah = sigma * jnp.sqrt(steps)
            pred_int = self._calculate_naive_intervals(res, level, h, sigmah)
            res = {**res, **pred_int}
        return res

    def predict_in_sample(self, level: list[int] | None = None):
        """Return the in-sample fitted values of the fitted Naive model.

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
            res = self._add_naive_fitted_intervals(res=res, se=self.model_["sigma"], level=level)
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
                ``(T, F)``. Only forwarded to ``conformity_scores``. Defaults
                to ``None``.
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
        out = Naive._naive_core_jit(y=y, h=h)
        res = {"mean": out["mean"]}

        if fitted:
            res["fitted"] = out["fitted"]

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                # Use base class conformal prediction
                cs = self.conformity_scores(y=y, X=X)
                res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
            else:
                # Native prediction intervals for naive model
                steps = jnp.arange(1, h + 1)
                sigma = out["sigma"]
                sigmah = sigma * jnp.sqrt(steps)
                pred_int = self._calculate_naive_intervals(res, level, h, sigmah)
                res = {**res, **pred_int}
            if fitted:
                sigma = out["sigma"]
                res = self._add_naive_fitted_intervals(res=res, se=sigma, level=level)
        return res

    def forward(
        self,
        h: int,
        y: jnp.ndarray,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int] | None = None,
        fitted: bool = False,
    ):
        """Apply the model to a new series; delegates to :meth:`forecast`.

        Takes ``h`` before ``y``; call with keyword arguments.

        Args:
            h (int): Forecast horizon.
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (jnp.ndarray | None): In-sample exogenous regressors, shape
                ``(T, F)``. Defaults to ``None``.
            X_future (jnp.ndarray | None): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (list[int] | None): Confidence levels in (0, 100). Defaults
                to ``None``.
            fitted (bool): Whether to return in-sample fitted values. Defaults
                to ``False``.

        Returns:
            dict: Same keys as :meth:`forecast`.
        """
        y = utils.ensure_float(y)
        res = self.forecast(
            h=h, y=y, X=X, X_future=X_future, level=level, fitted=fitted
        )
        return res

    def _calculate_naive_intervals(self, res, level, h, sigmah):
        """Calculate native prediction intervals for naive model using JAX operations."""
        level = sorted(level)
        z_scores = jnp.array([utils._jax_norm_ppf(0.5 + lv / 200) for lv in level])

        mean = res["mean"]
        intervals = {}

        # Lower bounds in reverse level order (statsforecast key convention)
        for i, lv in enumerate(reversed(level)):
            z = z_scores[len(level) - 1 - i]
            intervals[f"lo-{lv}"] = mean - z * sigmah

        # Upper bounds
        for i, lv in enumerate(level):
            z = z_scores[i]
            intervals[f"hi-{lv}"] = mean + z * sigmah

        return intervals

    def _add_naive_fitted_intervals(self, res, se, level):
        """Add fitted prediction intervals for naive model."""
        level = sorted(level)
        fitted = res["fitted"]

        # For fitted intervals, use constant standard error
        for lv in level:
            z = utils._jax_norm_ppf(0.5 + lv / 200)
            res[f"fitted-lo-{lv}"] = fitted - z * se
            res[f"fitted-hi-{lv}"] = fitted + z * se

        return res
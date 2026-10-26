"""
The naive class implements statsforecast's naive forecasting model, in which any forecast is equal to the previously observed value.

This JAX implementation provides complete compatibility with statsforecast's Naive class, including:
- fit(): Trains the model and stores parameters
- predict(): Makes forecasts using fitted model (stateful) with confidence intervals
- predict_in_sample(): Returns fitted values from training with confidence intervals
- forecast(): Stateless prediction (fit + predict in one step) with confidence intervals
- forward(): Applies fitted model to new/updated time series

Core computation is handled by _naive_core() which computes forecasts, fitted values, and residual statistics.

Confidence Intervals:
- Native intervals: Uses residual standard error with normal distribution assumption
- Conformal intervals: Uses base_forecaster's conformal prediction methods when conformal_params is provided

Instance Attributes:
1. All instance attributes in the parent class base_forecaster
2. model_: Dict containing fitted parameters (last_y, sigma, fitted values)

Class Attributes:
- Inherits from base_forecaster

Methods:
- All statsforecast Naive methods implemented with JAX operations
- Full confidence interval support (both native and conformal)
- Proper integration with base_forecaster's conformal prediction framework
"""

import jax
import jax.numpy as jnp
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals
from chronax import utils

class Naive(BaseForecaster):
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

    # JIT-compiled version for performance
    _naive_core_jit = jax.jit(_naive_core.__func__, static_argnums=(1,))

    def fit(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray | None = None,
    ):
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
        """Predict with fitted Naive.

        Args:
            h: Forecast horizon.
            X: Optional exogenous of shape (h, n_x).
            level: Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        self._require_fitted()
        # Use stored last_y to create forecasts
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
        """Access fitted Naive insample predictions.

        Args:
            level: Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions.
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
        """Memory Efficient Naive predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            h: Forecast horizon.
            y: Clean time series of shape (n,).
            X: Optional insample exogenous of shape (t, n_x).
            X_future: Optional exogenous of shape (h, n_x).
            level: Confidence levels (0-100) for prediction intervals.
            fitted: Whether or not to return insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
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
        """Apply fitted model to an new/updated series.

        Args:
            h: Forecast horizon.
            y: Clean time series of shape (n,).
            X: Optional insample exogenous of shape (t, n_x).
            X_future: Optional exogenous of shape (h, n_x).
            level: Confidence levels (0-100) for prediction intervals.
            fitted: Whether or not to return insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
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

        # Lower bounds (in reverse order to match statsforecast convention)
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
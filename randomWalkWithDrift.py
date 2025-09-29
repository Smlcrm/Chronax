"""
The RandomWalkWithDrift class implements statsforecast's random walk with drift forecasting model.

A variation of the naive method that allows forecasts to change over time by extrapolating
a linear trend between the first and last observations.

This JAX implementation provides complete compatibility with statsforecast's RandomWalkWithDrift class, including:
- fit(): Trains the model and stores parameters (slope, last_y, sigma, fitted values)
- predict(): Makes forecasts using fitted model (stateful) with confidence intervals
- predict_in_sample(): Returns fitted values from training with confidence intervals
- forecast(): Stateless prediction (fit + predict in one step) with confidence intervals

Core computation is handled by _rwd_core() which computes forecasts, fitted values, slope, and residual statistics.

Confidence Intervals:
- Native intervals: Uses residual standard error with drift-adjusted variance formula
- Conformal intervals: Uses base_forecaster's conformal prediction methods when conformal_params is provided

Mathematical Formula:
ŷ[t+h] = y[T] + h * slope, where slope = (y[T] - y[1]) / (T-1)

Instance Attributes:
1. All instance attributes in the parent class base_forecaster
2. model_: Dict containing fitted parameters (slope, last_y, sigma, fitted values, n)

Class Attributes:
- Inherits from base_forecaster

Methods:
- All statsforecast RandomWalkWithDrift methods implemented with JAX operations
- Full confidence interval support (both native and conformal)
- Proper integration with base_forecaster's conformal prediction framework
"""

import jax.numpy as jnp
from base_forecaster import base_forecaster
from conformal_intervals import conformal_intervals
import utils

def _jax_norm_ppf(p):
    """JAX implementation of normal percent point function (inverse CDF).

    Uses Beasley-Springer-Moro approximation for the inverse normal CDF.
    """
    # Clamp p to avoid numerical issues
    p = jnp.clip(p, 1e-10, 1 - 1e-10)

    # For p > 0.5, use symmetry
    sign = jnp.where(p > 0.5, 1.0, -1.0)
    p_adj = jnp.where(p > 0.5, p, 1.0 - p)

    # Beasley-Springer-Moro approximation
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    t = jnp.sqrt(-2 * jnp.log(1 - p_adj))
    z = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)

    return sign * z

class RandomWalkWithDrift(base_forecaster):
    def __init__(
        self,
        alias: str = "RWD",
        conformal_params: conformal_intervals | None = None,
    ):
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = None

    @staticmethod
    def _rwd_core(y: jnp.ndarray, h: int) -> dict:
        """Core Random Walk with Drift computation.

        Args:
            y: Time series array
            h: Forecast horizon

        Returns:
            dict: Contains mean forecasts, fitted values, slope, last_y, and sigma
        """
        # Calculate slope as linear trend between first and last observation
        slope = (y[-1] - y[0]) / (y.size - 1)

        # Generate forecasts: y_T + h * slope for h = 1, 2, ..., H
        forecast_steps = jnp.arange(1, h + 1, dtype=y.dtype)
        forecasts = y[-1] + slope * forecast_steps

        # Calculate fitted values: slope + y[t-1] for t = 2, ..., T
        fitted_vals = jnp.concatenate([
            jnp.array([jnp.nan]),  # First fitted value is NaN
            slope + y[:-1]         # Remaining fitted values
        ])

        # Calculate residuals and sigma
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

    def fit(self, y: jnp.ndarray):
        """Fit the RandomWalkWithDrift model.

        Args:
            y: Clean time series of shape (t,)

        Returns:
            self: Fitted RandomWalkWithDrift model
        """
        y = utils.ensure_float(y)
        mod = RandomWalkWithDrift._rwd_core(y, h=1)
        self.model_ = mod
        return self

    def predict(self, h: int, level: list[int] | None = None):
        """Predict with fitted RandomWalkWithDrift.

        Args:
            h: Forecast horizon
            level: Confidence levels (0-100) for prediction intervals

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions
        """
        # Generate forecasts using stored slope and last_y
        forecast_steps = jnp.arange(1, h + 1, dtype=self.model_["last_y"].dtype)
        mean = self.model_["last_y"] + self.model_["slope"] * forecast_steps
        res = {"mean": mean}

        if level is None:
            return res

        level = sorted(level)
        if self.conformal_params is not None:
            # Use base class conformal prediction
            cs = self.conformity_scores(y=None, X=None)  # Uses stored conformity scores
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
        """Access fitted RandomWalkWithDrift insample predictions.

        Args:
            level: Confidence levels (0-100) for prediction intervals

        Returns:
            dict: Dictionary with entries `fitted` for point predictions
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
        level: list[int] | None = None,
        fitted: bool = False,
    ):
        """Memory Efficient RandomWalkWithDrift predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            h: Forecast horizon
            y: Clean time series of shape (n,)
            level: Confidence levels (0-100) for prediction intervals
            fitted: Whether or not to return insample predictions

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions
        """
        y = utils.ensure_float(y)
        out = RandomWalkWithDrift._rwd_core(y=y, h=h)
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
        z_scores = jnp.array([_jax_norm_ppf(0.5 + lv / 200) for lv in level])

        mean = res["mean"]

        # Calculate intervals: mean ± z_score * sigmah
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

    def _add_rwd_fitted_intervals(self, res, se, level):
        """Add fitted prediction intervals for random walk with drift."""
        fitted = res["fitted"]

        # For fitted intervals, use constant standard error
        for lv in level:
            z = _jax_norm_ppf(0.5 + lv / 200)
            res[f"fitted-lo-{lv}"] = fitted - z * se
            res[f"fitted-hi-{lv}"] = fitted + z * se

        return res
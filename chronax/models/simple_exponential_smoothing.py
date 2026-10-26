"""
Simple Exponential Smoothing (SES) implementation using JAX.

SES is a weighted average forecasting method where more recent observations receive exponentially higher weights.
The forecast is a flat line at the final smoothed level.

Formula:
    ℓ[t] = α·y[t] + (1-α)·ℓ[t-1]
    ŷ[t+h] = ℓ[T] for all h ≥ 1

Where:
    - ℓ[t] is the smoothed level at time t
    - α ∈ [0,1] is the smoothing parameter (higher α = more weight on recent observations)
    - y[t] is the observed value at time t
    - T is the final observation index

Implementation:
    - The core ``_ses`` is JIT-compiled with static ``h`` and ``fitted``
    - Prediction intervals are conformal only, via BaseForecaster
"""

import jax
import jax.numpy as jnp
from functools import partial as _partial

from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals
from chronax import utils


# ============================================================================
# Core JAX Functions
# ============================================================================


@_partial(jax.jit, static_argnums=(2, 3))
def _ses(y: jnp.ndarray, alpha: float, h: int, fitted: bool) -> dict:
    """Simple exponential smoothing core, equivalent to statsforecast's ``_ses``.

    Args:
        y (jnp.ndarray): Univariate series, shape ``(T,)``.
        alpha (float): Smoothing parameter in [0, 1].
        h (int): Forecast horizon (static).
        fitted (bool): Whether to include fitted values (static).

    Returns:
        dict: ``"mean"`` (shape ``(h,)``) and optionally ``"fitted"`` (shape
        ``(T,)``).
    """
    fcst, fitted_vals = utils._ses_forecast(y, alpha)
    out = {"mean": utils._repeat_val(val=fcst, h=h)}
    if fitted:
        out["fitted"] = fitted_vals
    return out


# ============================================================================
# Model Class
# ============================================================================

class SimpleExponentialSmoothing(BaseForecaster):
    """Simple exponential smoothing (SES) model.

    Weighted average of past observations with exponentially decreasing
    weights, ``level[t] = alpha * y[t] + (1 - alpha) * level[t-1]``; the
    forecast is flat at the final level. Intervals are conformal only.

    Args:
        alpha (float): Smoothing parameter in [0, 1].
        alias (str): Custom name of the model. Defaults to ``"SES"``.
        conformal_params (ConformalIntervals | None): Configuration for
            conformal prediction intervals. Defaults to ``None``.
        prediction_intervals (ConformalIntervals | None): Alias for
            ``conformal_params`` (statsforecast naming); takes precedence when
            both are given. Defaults to ``None``.

    Raises:
        ValueError: If ``alpha`` is outside [0, 1].
    """
    
    def __init__(
        self,
        alpha: float,
        alias: str = "SES",
        conformal_params: ConformalIntervals | None = None,
        prediction_intervals: ConformalIntervals | None = None,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        self.alpha = alpha
        self.alias = alias
        # `prediction_intervals` is the statsforecast-compatible alias for
        # `conformal_params`; either name configures the conformal intervals.
        ci = prediction_intervals if prediction_intervals is not None else conformal_params
        self.prediction_intervals = ci
        self.conformal_params = ci
        self.model_ = {}
    
    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "SimpleExponentialSmoothing":
        r"""Fit the SimpleExponentialSmoothing model.

        Runs SES on the full series to produce the fitted values and the final
        level (the flat forecast). When ``conformal_params`` is set, conformity
        scores are computed and cached under ``model_["_cs"]``.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (jnp.ndarray | None): Exogenous regressors, shape ``(T, F)``.
                Only forwarded to ``conformity_scores``. Defaults to ``None``.

        Returns:
            SimpleExponentialSmoothing: ``self``.

        Raises:
            ValueError: If ``conformal_params`` is set and the series is too
                short for 2 cross-validation (CV) windows.
        """
        y = utils.ensure_float(y)
        mod = _ses(y=y, alpha=self.alpha, h=1, fitted=True)
        self.model_ = dict(mod)
        
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=X)
            self.model_["_cs"] = cs
        
        return self
    
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
    ) -> dict:
        r"""Stateless fit+predict in one call.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon.
            X (jnp.ndarray | None): In-sample exogenous regressors, shape
                ``(T, F)``. Ignored. Defaults to ``None``.
            X_future (jnp.ndarray | None): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (list[int | float] | None): Confidence levels in (0, 100),
                e.g. ``[80, 95]``. Requires ``conformal_params``. Defaults to
                ``None``.
            fitted (bool): Accepted for ``BaseForecaster`` interface parity;
                ignored, no ``"fitted"`` key is returned. Defaults to
                ``False``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``, all equal to the final level)
            plus ``"lo-{lv}"``/``"hi-{lv}"`` per level when ``level`` is given.

        Raises:
            Exception: If ``level`` is given without ``conformal_params``.
            ValueError: If ``level`` is given and the series is too short for
                2 CV windows.
        """
        y = utils.ensure_float(y)
        mod = _ses(y=y, alpha=self.alpha, h=h, fitted=False)
        res = {"mean": mod["mean"]}
        if level is None:
            # The conformity_scores CV path never passes `level` and returns here.
            return res
        # A requested `level` is honoured or raises; it is never silently dropped.
        if self.conformal_params is None:
            raise Exception("You must pass `conformal_params` to compute them.")
        cs = self.conformity_scores(y=y, X=None)
        return self.add_confidence_intervals(
            res, cs, sorted(level), self.conformal_params.method
        )
    
    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int] | None = None,
    ) -> dict:
        r"""Predict with the fitted SimpleExponentialSmoothing model.

        All forecasts equal the final level. Conformal intervals use the
        scores cached at ``fit``.

        Args:
            h (int): Forecast horizon.
            X (jnp.ndarray | None): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (list[int] | None): Confidence levels in (0, 100), e.g.
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
        mean = utils._repeat_val(val=self.model_["mean"][0], h=h)
        res = {"mean": mean}
        
        if level is None:
            return res
        
        if self.conformal_params is None:
            raise ValueError(
                "You must pass `conformal_params` to compute prediction intervals."
            )
        
        cs = self.model_.get("_cs")
        if cs is None:
            raise ValueError("Model must be fitted before computing intervals.")
        
        res = self.add_confidence_intervals(
            fcst=res, cs=cs, level=level, method=self.conformal_params.method
        )
        return res
    
    def predict_in_sample(self) -> dict:
        r"""Return the in-sample fitted values of the fitted model.

        Returns:
            dict: ``"fitted"`` (shape ``(T,)``); the first value is NaN.

        Raises:
            ValueError: If called before ``fit``.
        """
        if "fitted" not in self.model_:
            raise ValueError("Model must be fitted first.")
        return {"fitted": self.model_["fitted"]}

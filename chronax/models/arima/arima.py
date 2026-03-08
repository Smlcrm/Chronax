"""
File: arima.py

High-level Purpose:
    Provides a production ARIMA forecaster wrapper with a stable class interface
    for fitting and forecasting univariate time-series with optional exogenous
    regressors.

Problem Solved:
    Encapsulates ARIMA estimation and forecasting mechanics so downstream
    forecasting pipelines can call a single estimator object without handling
    optimization internals directly.

Architectural Role:
    Sits at the model-interface layer of the forecasting system, delegating
    low-level estimation and Kalman/ARIMA math to `auto_arima.py` while
    exposing a `BaseForecaster` compatible API for training, forecasting, and
    interval prediction.

Major Classes/Functions:
    - `ARIMA`: Fixed-order ARIMA estimator with fit/forecast/predict paths.

External Dependencies:
    - `numpy`
    - `jax`, `jax.numpy`
    - Internal modules: `auto_arima`, `base_forecaster`, `utils`

Expected Inputs and Outputs:
    - Input: historical series (`jnp.ndarray`-compatible), forecast horizon, and
      optional exogenous regressor matrices.
    - Output: dictionaries containing mean forecasts and optional prediction
      intervals.

Example:
    >>> import jax.numpy as jnp
    >>> from arima import ARIMA
    >>> model = ARIMA(order=(1, 1, 1), seasonal_order=(0, 0, 0), period=1)
    >>> model.fit(jnp.array([1.0, 2.0, 2.5, 3.0]))
    >>> preds = model.predict(h=3)
    >>> preds["mean"].shape
    (3,)

Assumptions:
    - Input series are numeric and can be converted to float64.
    - The underlying optimizer converges to a finite solution for selected
      model orders.

Side Effects:
    - Mutates estimator instance state (fitted model, cached training stats).

Author:
    Auto-documented
Date:
    2026-02-21
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from .auto_arima import (
    arima_fit,
    _fit_model_bfgs,
    _objective_css,
    _objective_ml,
    _forecast_from_params,
    _reconstruct_forecast,
    predict_arima,
    _aa_standardize,
    _aa_denormalize
)
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import _quantiles

Array = jnp.ndarray

class ARIMA(BaseForecaster):
    """
    ARIMA

    Description:
        Represents a fixed-order ARIMA forecaster that standardizes training
        data for optimization stability and returns forecasts in the original
        value scale.

    Attributes:
        uses_exog (bool): Indicates support for exogenous regressors.
        order (tuple[int, int, int]): Non-seasonal ARIMA order `(p, d, q)`.
        seasonal_order (tuple[int, int, int]): Seasonal ARIMA order `(P, D, Q)`.
        period (int): Seasonal period length.
        include_mean (bool): Whether to include intercept/drift term.
        method (str): Optimization method selector (for example `CSS`, `ML`).
        alias (str): Display name for external reporting.
        ``model_`` (dict[str, Any] | None): Fitted model payload after ``fit``.
        standardize (bool): Whether to standardize data before fitting.
        _y_mean (jnp.ndarray | None): Cached training mean for de-normalization.
        _y_std (jnp.ndarray | None): Cached training std for de-normalization.
        _delta (jnp.ndarray): Differencing polynomial coefficients.
        _arma (tuple[int, ...]): Expanded ARMA metadata tuple.
        _narma (int): Number of ARMA parameters.
        _ncxreg (int): Number of deterministic coefficients.
        _n_exog (int): Number of exogenous regressors.

    Args:
        order (tuple[int, int, int]): Non-seasonal ARIMA order.
        seasonal_order (tuple[int, int, int]): Seasonal ARIMA order.
        period (int): Seasonal frequency.
        include_mean (bool): Include intercept/drift component.
        method (str): Optimization strategy label.
        alias (str): Friendly estimator alias.
        standardize (bool): Standardize input series before fit.

    Methods:
        fit(): Estimate model parameters from training data.
        forecast(): Run one-shot fit and produce horizon forecasts.
        predict(): Forecast from an already-fitted model.

    Returns:
        Produces forecasts and optional prediction intervals through dictionary
        outputs keyed by `mean`, `lo-<level>`, and `hi-<level>`.

    Example:
        >>> import jax.numpy as jnp
        >>> model = ARIMA(order=(1, 1, 1))
        >>> model.fit(jnp.array([1.0, 1.5, 2.0, 2.2]))
        >>> model.predict(h=2)["mean"]

    Notes:
        This class is stateful and not thread-safe for concurrent mutation.
    """
    uses_exog: bool = True
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        period: int = 1,
        include_mean: bool = True,
        method: str = "CSS",
        alias: str = "ARIMA",
        standardize: bool = True,
    ) -> None:
        """
        Initialize a fixed-order ARIMA estimator.

        Detailed Description:
            Stores configuration for ARIMA estimation and precomputes static
            differencing metadata used during optimization and forecasting.

        Args:
            order (tuple[int, int, int]): Non-seasonal order `(p, d, q)`.
            seasonal_order (tuple[int, int, int]): Seasonal order `(P, D, Q)`.
            period (int): Seasonal cycle length.
            include_mean (bool): Whether to include deterministic mean/drift.
            method (str): Objective method strategy.
            alias (str): User-facing model name.
            standardize (bool): Enables series standardization.

        Returns:
            None: Constructor initializes estimator state.

        Raises:
            ValueError: Propagated by downstream array operations if invalid
                seasonal period/order combinations are provided.

        Side Effects:
            Populates model configuration and cached polynomial metadata.

        Example:
            >>> model = ARIMA(order=(1, 1, 1), seasonal_order=(0, 0, 0), period=1)

        Notes:
            Parameter validation is largely deferred to fitting kernels.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.period = period
        self.include_mean = include_mean
        self.method = method
        self.alias = alias
        self.model_: dict[str, Any] | None = None
        self.standardize = standardize
        self._y_mean: jnp.ndarray | None = None
        self._y_std: jnp.ndarray | None = None

        # Pre-compute and cache the delta polynomial (depends only on order/period)
        p, d, q = order
        P, D, Q = seasonal_order
        delta = jnp.array([1.0], dtype=jnp.float64)
        for _ in range(d):
            delta = jnp.convolve(delta, jnp.array([1.0, -1.0]))
        for _ in range(D):
            seas_diff = jnp.concatenate([jnp.array([1.0]), jnp.zeros(period - 1), jnp.array([-1.0])])
            delta = jnp.convolve(delta, seas_diff)
        self._delta: jnp.ndarray = -delta[1:]
        self._arma: tuple[int, ...] = (p, q, P, Q, period, d, D)
        self._narma: int = p + q + P + Q
        n_exog = 0
        self._ncxreg: int = n_exog + (1 if include_mean else 0)
        self._n_exog: int = n_exog

    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "ARIMA":
        """
        Fit ARIMA parameters on a training series.

        Detailed Description:
            Converts inputs to float64 JAX arrays, optionally standardizes the
            series, delegates parameter optimization to `arima_fit`, and stores
            fitted state for subsequent `predict` calls.

        Args:
            y (jnp.ndarray): Training target series.
            X (jnp.ndarray | None, optional): Optional exogenous matrix aligned with ``y``.

        Returns:
            ARIMA: The fitted estimator instance.

        Raises:
            RuntimeError: Propagated from fitting internals if optimization
                fails irrecoverably.

        Side Effects:
            Mutates ``model_``, ``y_train_``, and cached normalization statistics.

        Example:
            >>> model = ARIMA(order=(1, 1, 1))
            >>> _ = model.fit(jnp.array([10.0, 11.0, 13.0, 12.0]))

        Notes:
            Forecast reconstruction assumes the same differencing specification
            that was used during fitting.
        """
        y_jax = jnp.asarray(np.array(y), dtype=jnp.float64)
        if X is not None:
            X = jnp.asarray(X, dtype=jnp.float64)

        # Standardize training series if enabled
        if self.standardize:
            y_fit, self._y_mean, self._y_std = _aa_standardize(y_jax)
        else:
            y_fit = y_jax
            self._y_mean = jnp.array(0.0, dtype=jnp.float64)
            self._y_std = jnp.array(1.0, dtype=jnp.float64)

        self.y_train_ = y_fit

        seasonal = {'order': self.seasonal_order, 'period': self.period}
        
        self.model_ = arima_fit(
            y_fit, order=self.order, seasonal=seasonal,
            include_mean=self.include_mean, method=self.method, xreg=X
        )
        
        p, d, q = self.order
        P, D, Q = self.seasonal_order
        self.model_['arma'] = (p, q, P, Q, self.period, d, D)
        
        return self

    def forecast(self, h: int, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        """
        Fit and forecast in one call.

        Detailed Description:
            Runs a fast one-shot ARIMA fit followed by a fused forecast kernel.
            This pathway skips information-criteria computation and is intended
            for low-overhead repeated forecasting from raw history.

        Args:
            h (int): Forecast horizon.
            y (jnp.ndarray): Source training series.
            X (jnp.ndarray | None, optional): Exogenous regressors. The fast path currently does not use exogenous variables.

        Returns:
            dict[str, jnp.ndarray]: Forecast dictionary containing `mean`.

        Raises:
            RuntimeError: Propagated if optimizer objective diverges.

        Side Effects:
            None on persistent fitted state; runs transient optimization only.

        Example:
            >>> model = ARIMA(order=(1, 0, 1))
            >>> out = model.forecast(h=3, y=jnp.array([1.0, 2.0, 2.2, 3.0]))
            >>> out["mean"].shape
            (3,)

        Notes:
            Uses cached polynomial metadata from constructor configuration.
        """
        y_jax = jnp.asarray(np.array(y), dtype=jnp.float64)

        # Standardize input series for fast one-shot fit if enabled
        if self.standardize:
            y_fit, f_mean, f_std = _aa_standardize(y_jax)
        else:
            y_fit = y_jax
            f_mean = jnp.array(0.0, dtype=jnp.float64)
            f_std = jnp.array(1.0, dtype=jnp.float64)
        p, d, q = self.order
        P, D, Q = self.seasonal_order
        
        # --- BFGS optimization (same as arima_fit, but uses cached delta) ---
        use_drift = self.include_mean and (d + D) == 1
        init_params = jnp.zeros(self._narma + self._ncxreg, dtype=jnp.float64) + 1e-3
        
        if self.include_mean and (d + D) == 0:
            init_params = init_params.at[self._narma + self._n_exog].set(jnp.nanmean(y_fit))
        
        if use_drift:
            if D == 1 and self.period > 1:
                dx = y_fit[self.period:] - y_fit[:-self.period]
            else:
                dx = y_fit[1:] - y_fit[:-1]
            init_params = init_params.at[self._narma + self._n_exog].set(jnp.nanmean(dx))

        method = self.method.upper()
        current_params = init_params
        maxiter = 50  # Reduced: CSS converges fast for low-order models
        
        if "CSS" in method:
            css_iter = maxiter // 2 if method == "CSS-ML" else maxiter
            current_params = _fit_model_bfgs(
                current_params, y_fit, None, self._delta, _objective_css,
                self._arma, self._ncxreg, self._n_exog, self.include_mean, css_iter
            )
        if "ML" in method:
            ml_iter = maxiter // 2 if method == "CSS-ML" else maxiter
            current_params = _fit_model_bfgs(
                current_params, y_fit, None, self._delta, _objective_ml,
                self._arma, self._ncxreg, self._n_exog, self.include_mean, ml_iter
            )

        # --- Fused forecast: single XLA dispatch for params → forecast ---
        raw_fc = _forecast_from_params(
            current_params, y_fit, self._delta,
            self._arma, self._ncxreg, self._n_exog, self.include_mean, h
        )

        # --- Reconstruction using delta polynomial inverse filter ---
        fc_norm = _reconstruct_forecast(raw_fc, y_fit, self._arma, h)
        fc = _aa_denormalize(fc_norm, f_mean, f_std)

        return {"mean": fc}

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: int | tuple[int, ...] | None = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Generate forecasts from a fitted ARIMA model.

        Detailed Description:
            Uses stored fitted parameters to produce horizon forecasts and, when
            requested, interval bounds computed from forecast standard errors.

        Args:
            h (int): Number of future steps to predict.
            X (jnp.ndarray | None, optional): Optional exogenous future matrix.
            level (int | tuple[int, ...] | None, optional): Confidence levels for interval generation.

        Returns:
            dict[str, jnp.ndarray]: Dictionary containing `mean` and optional
            lower/upper interval keys.

        Raises:
            RuntimeError: If called before `fit`.

        Side Effects:
            None. Uses cached fitted state without mutating model parameters.

        Example:
            >>> model = ARIMA(order=(1, 1, 1)).fit(jnp.array([1.0, 2.0, 3.0, 4.0]))
            >>> model.predict(h=2, level=95)["mean"].shape
            (2,)

        Notes:
            Interval widths for integrated models use cumulative uncertainty.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        if X is not None:
            X = jnp.asarray(X, dtype=jnp.float64)
        
        preds = predict_arima(self.model_, n_ahead=h, newxreg=X, se_fit=(level is not None))
        
        if isinstance(preds, tuple):
            mean_pred, se_pred = preds
        else:
            mean_pred, se_pred = preds, None

        # Reconstruct in standardized space
        fc_norm = _reconstruct_forecast(mean_pred, self.y_train_, self.model_["arma"], h)
        # Map back to original scale if standardization was used
        if self.standardize:
            mean_orig = _aa_denormalize(fc_norm, self._y_mean, self._y_std)
        else:
            mean_orig = fc_norm

        if se_pred is not None:
            p, q, P, Q, m, d, D = self.model_["arma"]
            if d + D > 0:
                se_scaled = jnp.sqrt(jnp.cumsum(se_pred**2))
            else:
                se_scaled = se_pred
            se_orig = se_scaled * (self._y_std if self.standardize else 1.0)
        else:
            se_orig = None

        result = {}
        result["mean"] = mean_orig
            
        if level is not None and se_orig is not None:
            if isinstance(level, int): level = (level,)
            z_scores = _quantiles(level)
            for i, lv in enumerate(level):
                q = z_scores[i]
                result[f"lo-{lv}"] = result["mean"] - q * se_orig
                result[f"hi-{lv}"] = result["mean"] + q * se_orig
            
        return result
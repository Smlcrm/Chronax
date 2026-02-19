from __future__ import annotations
import warnings
from typing import Dict, Optional, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp

from auto_arima import (
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
from base_forecaster import BaseForecaster
from utils import _quantiles

Array = jnp.ndarray

class ARIMA(BaseForecaster):
    """
    Fixed ARIMA model wrapper.

    Internally standardizes the series (zero mean, unit variance)
    for faster and more stable optimization, and always returns
    forecasts in the original scale.
    """
    uses_exog = True
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        period: int = 1,
        include_mean: bool = True,
        method: str = "CSS",
        alias: str = "ARIMA",
        standardize: bool = True,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.period = period
        self.include_mean = include_mean
        self.method = method
        self.alias = alias
        self.model_ = None
        self.standardize = standardize
        self._y_mean = None
        self._y_std = None

        # Pre-compute and cache the delta polynomial (depends only on order/period)
        p, d, q = order
        P, D, Q = seasonal_order
        delta = jnp.array([1.0], dtype=jnp.float64)
        for _ in range(d):
            delta = jnp.convolve(delta, jnp.array([1.0, -1.0]))
        for _ in range(D):
            seas_diff = jnp.concatenate([jnp.array([1.0]), jnp.zeros(period - 1), jnp.array([-1.0])])
            delta = jnp.convolve(delta, seas_diff)
        self._delta = -delta[1:]
        self._arma = (p, q, P, Q, period, d, D)
        self._narma = p + q + P + Q
        n_exog = 0
        self._ncxreg = n_exog + (1 if include_mean else 0)
        self._n_exog = n_exog

    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "ARIMA":
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
        Fast fit-and-predict in one shot. Uses fused JIT kernel to minimize
        Python-to-XLA dispatch overhead. No metric computation (AIC, BIC, etc.).
        
        Args:
            h: Forecast horizon
            y: Training series
            X: Exogenous regressors (not supported in fast path)
            
        Returns:
            Dict with 'mean' key containing forecast array
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

    def predict(self, h: int, X: Optional[jnp.ndarray] = None, level=None) -> Dict[str, jnp.ndarray]:
        if self.model_ is None: raise RuntimeError("Model not fitted.")
        if X is not None: X = jnp.asarray(X, dtype=jnp.float64)
        
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
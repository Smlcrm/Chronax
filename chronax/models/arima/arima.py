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
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp

from .auto_arima import (
    arima_fit,
    _fit_model_scan,
    _objective_css,
    _objective_ml,
    _objective_ml_ss,
    _use_steadystate,
    _forecast_from_params,
    predict_arima,
    _aa_standardize,
    _aa_denormalize
)
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import _quantiles
from chronax.utils.conformal_intervals import ConformalIntervals

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
    # Exogenous support is not wired end-to-end on the stateless/CV paths
    # (X is silently ignored there), so it is not advertised; passing X
    # raises instead of silently dropping it.
    uses_exog: bool = False

    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        period: int = 1,
        include_mean: bool = True,
        method: str = "CSS",
        alias: str = "ARIMA",
        standardize: bool = True,
        conformal_params: Optional[ConformalIntervals] = None,
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
            conformal_params (ConformalIntervals | None): Conformal prediction
                configuration; when set, ``fit`` caches conformity scores and
                interval requests use conformal (not analytic) bounds.

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
        self.conformal_params = conformal_params
        self._cs: jnp.ndarray | None = None
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
        # No host round-trip: np.array(y) here would concretize traced inputs and
        # break conformity_scores' vmapped forecast->fit path.
        y_jax = jnp.asarray(y, dtype=jnp.float64)
        if X is not None:
            raise ValueError(
                "ARIMA does not currently support exogenous regressors "
                "end-to-end; fit accepts only y (use arima_fit(xreg=...) for "
                "the low-level exogenous API)."
            )

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

        # Pre-compute and cache conformity scores on the training series for
        # predict() intervals (sibling convention: AutoCES/WindowAverage/SES).
        # Safe without a clone: forecast() below writes nothing to self.
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y_jax, X=X)
        else:
            self._cs = None

        return self

    def forecast(self, h: int, y: jnp.ndarray, X: Optional[jnp.ndarray] = None, X_future: Optional[jnp.ndarray] = None, level: Optional[list] = None, fitted: bool = False) -> Dict[str, jnp.ndarray]:
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
            X_future (jnp.ndarray | None, optional): Future exogenous regressors (unused; included for BaseForecaster compliance). Default is None.
            level (list | None, optional): Confidence levels (0-100) for conformal
                prediction intervals. Requires ``conformal_params``. Default is None.
            fitted (bool, optional): Whether to return fitted values (unused; the
                fast path computes forecasts only). Default is False.

        Returns:
            dict[str, jnp.ndarray]: Forecast dictionary containing `mean`, plus
            `lo-{l}`/`hi-{l}` conformal bounds when `level` is given.

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
        if X is not None or X_future is not None:
            raise ValueError(
                "ARIMA does not currently support exogenous regressors "
                "end-to-end; forecast accepts only y."
            )
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        # Standardize input series for fast one-shot fit if enabled
        if self.standardize:
            y_fit, f_mean, f_std = _aa_standardize(y_jax)
        else:
            y_fit = y_jax
            f_mean = jnp.array(0.0, dtype=jnp.float64)
            f_std = jnp.array(1.0, dtype=jnp.float64)
        p, d, q = self.order
        P, D, Q = self.seasonal_order
        
        # --- Scan optimization (same objectives as arima_fit, cached delta) ---
        use_drift = self.include_mean and (d + D) == 1
        init_params = jnp.zeros(self._narma + self._ncxreg, dtype=jnp.float64) + 1e-3
        
        if self.include_mean and (d + D) == 0:
            init_params = init_params.at[self._narma + self._n_exog].set(jnp.nanmean(y_fit))
        
        if use_drift:
            if D == 1 and self.period > 1:
                dx = y_fit[self.period:] - y_fit[:-self.period]
                per_step = self.period
            else:
                dx = y_fit[1:] - y_fit[:-1]
                per_step = 1
            init_params = init_params.at[self._narma + self._n_exog].set(jnp.nanmean(dx) / per_step)

        method = self.method.upper()
        current_params = init_params
        # Each phase gets the full budget (matches arima_fit); the optimizer's
        # convergence gate exits early on easy fits, so the cap only matters
        # on hard surfaces.
        maxiter = 100

        # _fit_model_scan (not jax.scipy BFGS): this path runs under
        # conformity_scores' vmap; the scan optimizer is batch-stable by
        # construction and robust to BFGS's inconsistent line-search-failure
        # returns.
        if "CSS" in method:
            current_params = _fit_model_scan(
                current_params, y_fit, None, self._delta, _objective_css,
                self._arma, self._ncxreg, self._n_exog, self.include_mean, maxiter
            )
        if "ML" in method:
            # Long non-seasonal series use the steady-state likelihood (frozen-gain
            # tail, exact after burn-in) to skip the full-series covariance recursion
            # on every optimizer evaluation.
            ml_obj = _objective_ml_ss if _use_steadystate(y_fit.shape[0], self._arma) else _objective_ml
            current_params = _fit_model_scan(
                current_params, y_fit, None, self._delta, ml_obj,
                self._arma, self._ncxreg, self._n_exog, self.include_mean, maxiter
            )

        # Closed-form mean/drift pin (same rescue as arima_fit, so the fast
        # path and fit()+predict() agree on the deterministic component).
        if self.include_mean:
            _mi = self._narma + self._n_exog
            if use_drift:
                current_params = current_params.at[_mi].set(jnp.nanmean(dx) / per_step)
            elif (d + D) == 0:
                pass  # in-objective clip already bounds the mean
            else:
                current_params = current_params.at[_mi].set(0.0)

        # --- Fused forecast: single XLA dispatch for params → forecast ---
        raw_fc, raw_se = _forecast_from_params(
            current_params, y_fit, self._delta,
            self._arma, self._ncxreg, self._n_exog, self.include_mean, h
        )

        # raw_fc is on the training-series scale: the state space's
        # differencing rows integrate internally.
        fc = _aa_denormalize(raw_fc, f_mean, f_std)

        res = {"mean": fc}
        if level is None:
            return res

        level = sorted(level)
        if self.conformal_params is None:
            # Analytic z-score fallback, mirroring predict()'s behaviour when
            # no conformal configuration exists.
            se_orig = raw_se * (f_std if self.standardize else jnp.array(1.0, dtype=jnp.float64))
            z_scores = _quantiles(level)
            for i, lv in enumerate(level):
                res[f"lo-{lv}"] = res["mean"] - z_scores[i] * se_orig
                res[f"hi-{lv}"] = res["mean"] + z_scores[i] * se_orig
            return res
        if h != self.conformal_params.h:
            raise ValueError(
                f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                "conformity scores cover exactly conformal_params.h steps."
            )
        cs = self.conformity_scores(y=y_jax, X=X)
        return self.add_confidence_intervals(res, cs, level, self.conformal_params.method)

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
            raise ValueError(
                "ARIMA does not currently support exogenous regressors "
                "end-to-end; predict accepts only h."
            )

        # Conformal intervals take over whenever conformal_params is set;
        # analytic z-score intervals remain the fallback. Standard errors are
        # only computed when the analytic path will actually use them.
        use_conformal = level is not None and self.conformal_params is not None
        preds = predict_arima(
            self.model_, n_ahead=h, newxreg=X,
            se_fit=(level is not None and not use_conformal),
        )
        
        if isinstance(preds, tuple):
            mean_pred, se_pred = preds
        else:
            mean_pred, se_pred = preds, None

        # mean_pred is on the training-series scale (integration lives in the
        # state space); map back to the original scale if standardization was
        # used.
        if self.standardize:
            mean_orig = _aa_denormalize(mean_pred, self._y_mean, self._y_std)
        else:
            mean_orig = mean_pred

        if se_pred is not None:
            # se_pred is the integrated-series forecast SE (the state space
            # bakes differencing into T/Z); only rescale to original units.
            se_orig = se_pred * (self._y_std if self.standardize else 1.0)
        else:
            se_orig = None

        result = {}
        result["mean"] = mean_orig

        if use_conformal:
            level = sorted([level] if isinstance(level, int) else list(level))
            if self._cs is None:
                raise ValueError(
                    "Conformity scores are not available. Fit the model (fit(...)) "
                    "with `conformal_params` set so predict() can use cached scores."
                )
            if h != self.conformal_params.h:
                raise ValueError(
                    f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                    "conformity scores cover exactly conformal_params.h steps."
                )
            return self.add_confidence_intervals(result, self._cs, level, self.conformal_params.method)

        if level is not None and se_orig is not None:
            if isinstance(level, int): level = (level,)
            z_scores = _quantiles(level)
            for i, lv in enumerate(level):
                q = z_scores[i]
                result[f"lo-{lv}"] = result["mean"] - q * se_orig
                result[f"hi-{lv}"] = result["mean"] + q * se_orig
            
        return result
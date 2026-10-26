"""
Croston's Classic method for intermittent demand forecasting.

Designed for sparse time series with many zeros and occasional non-zero demands.
Decomposes series into demand size and inter-demand intervals, forecasting each separately.

Formula:
    ŷ[t+h] = demand_forecast / interval_forecast

Where:
    - demand_forecast: simple exponential smoothing (SES, α=0.1) on non-zero values
    - interval_forecast: SES (α=0.1) on time gaps between non-zero values
    - α = 0.1 is Croston's original fixed smoothing parameter

Implementation:
    - ``jax.lax.cond`` selects the naive fallback when no positive demand exists
    - JIT-compiled with static ``h`` and ``fitted``
    - Fitted values are expanded back to the original series length
    - Forecast intervals are conformal only; fitted intervals use the residual
      standard error
"""

import jax
import jax.numpy as jnp
from functools import partial as _partial
from typing import Optional, List, Dict
from chronax import utils
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals,_demand, _intervals_c, _ses_forecast_nan, _repeat_val, _expand_fitted_demand, _expand_fitted_intervals


# ============================================================================
# CORE CROSTON CLASSIC LOGIC
# ============================================================================

@_partial(jax.jit, static_argnums=(1, 2))
def _croston_classic(
    y: jnp.ndarray,
    h: int,
    fitted: bool = False,
) -> dict:
    """Croston's Classic method for intermittent demand.

    1. Extract the demand sizes (positive values).
    2. Extract the intervals between demands.
    3. Forecast both with SES at α=0.1.
    4. Combine as ``demand_forecast / interval_forecast``.

    Args:
        y (jnp.ndarray): Univariate series, shape ``(T,)``.
        h (int): Forecast horizon (static).
        fitted (bool): Whether to return in-sample fitted values (static).

    Returns:
        dict: ``"mean"`` (shape ``(h,)``) and optionally ``"fitted"`` (shape
        ``(T,)``).

    Example:
        >>> y = jnp.array([0., 5., 0., 0., 3., 0., 2., 0., 0.])
        >>> result = _croston_classic(y, h=3, fitted=False)
        >>> result['mean'].shape
        (3,)

    Note:
        Demand and interval arrays are fixed-size and NaN-padded so the
        function traces under jit and vmap.
    """
    y = utils.ensure_float(y)
    alpha = 0.1  # Croston's fixed smoothing parameter

    yd = utils._demand(y)

    def no_demand_forecast():
        """Naive fallback when the series has no positive demand."""
        last_val = y[-1]
        mean = utils._repeat_val(last_val, h)
        out = {"mean": mean}
        if fitted:
            # Naive fitted: y[t] = y[t-1]
            out["fitted"] = jnp.concatenate([jnp.array([jnp.nan]), y[:-1]])
        return out

    def croston_forecast():
        """Standard Croston algorithm when demand exists."""
        ydp, ydf = utils._ses_forecast_nan(yd, alpha)

        yi = utils._intervals_c(y)

        yip, yif = utils._ses_forecast_nan(yi, alpha)

        # demand / interval; a zero interval forecast falls back to the demand
        mean_val = jnp.where(
            yip != 0.0,
            ydp / yip,
            ydp
        )
        
        out = {"mean": utils._repeat_val(mean_val, h)}
        
        if fitted:
            # Expand the compact fitted arrays (plus the final forecast) back
            # to the original series length.
            ydf_expanded = utils._expand_fitted_demand(
                jnp.append(ydf, ydp), y
            )
            yif_expanded = utils._expand_fitted_intervals(
                jnp.append(yif, yip), y
            )
            out["fitted"] = ydf_expanded / yif_expanded

        return out

    # No-demand predicate: `y <= 0` also covers series with negative values.
    # `yd` is fixed-size and NaN-padded, so `yd.size` is never 0.
    return jax.lax.cond(
        jnp.all(y <= 0),
        no_demand_forecast,
        croston_forecast
    )


# ============================================================================
# CROSTON CLASSIC MODEL CLASS
# ============================================================================

class CrostonClassic(BaseForecaster):
    """Croston's Classic method for intermittent demand time series.

    Forecasts demand size and inter-demand interval separately with SES at
    the fixed α=0.1 and returns their ratio as a flat forecast. Forecast
    intervals are conformal only; fitted intervals use the residual standard
    error.

    Args:
        alias (str): Custom name of the model. Defaults to
            ``"CrostonClassic"``.
        conformal_params (Optional[ConformalIntervals]): Configuration for
            conformal prediction intervals; required for ``level``. Defaults
            to ``None``.

    Example:
        >>> from chronax.utils import ConformalIntervals
        >>> ci = ConformalIntervals(h=3, n_windows=10)
        >>> model = CrostonClassic(conformal_params=ci)
        >>> y = jnp.array([0., 5., 0., 0., 3., 0., 2., 0., 0., 4., 0., 0., 3.])
        >>> model.fit(y)
        >>> forecast = model.predict(h=3, level=[80, 95])
    """
    
    def __init__(
        self,
        alias: str = "CrostonClassic",
        conformal_params: Optional[ConformalIntervals] = None,
    ) -> None:
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = {}
        self._cs = None
    
    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "CrostonClassic":
        """Fit the CrostonClassic model.

        Stores the forecast, fitted values and residual standard error in
        ``model_``. When ``conformal_params`` is set, conformity scores are
        computed and cached for :meth:`predict`.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (Optional[jnp.ndarray]): Exogenous regressors, shape ``(T, F)``.
                Only forwarded to ``conformity_scores``. Defaults to ``None``.

        Returns:
            CrostonClassic: ``self``.

        Raises:
            ValueError: If ``conformal_params`` is set and the series is too
                short for 2 cross-validation (CV) windows.
        """
        y = utils.ensure_float(y)

        self.model_ = _croston_classic(y=y, h=1, fitted=True)

        residuals = y - self.model_["fitted"]
        self.model_["sigma"] = utils.calculate_sigma(residuals, y.size)

        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y, X)
        
        return self
    
    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Predict with the fitted CrostonClassic model.

        Args:
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (Optional[List[int]]): Confidence levels in (0, 100), e.g.
                ``[80, 95]``. Requires ``conformal_params``. Defaults to
                ``None``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``, flat) plus
            ``"lo-{lv}"``/``"hi-{lv}"`` per level when ``level`` is given.

        Raises:
            ValueError: If called before ``fit``, or if ``level`` is given
                without ``conformal_params``.
        """
        self._require_fitted()
        mean = utils._repeat_val(val=self.model_["mean"][0], h=h)
        res = {"mean": mean}

        if level is not None:
            if self.conformal_params is None:
                raise ValueError(
                    "You must instantiate the class with `conformal_params` "
                    "to calculate prediction intervals"
                )
            level = sorted(level)
            res = self.add_confidence_intervals(
                fcst=res,
                cs=self._cs,
                level=level,
                method=self.conformal_params.method
            )
        
        return res
    
    def predict_in_sample(
        self,
        level: Optional[List[int]] = None
    ) -> Dict[str, jnp.ndarray]:
        """Return the in-sample fitted values of the fitted model.

        Args:
            level (Optional[List[int]]): Confidence levels in (0, 100).
                Defaults to ``None``.

        Returns:
            dict: ``"fitted"`` (shape ``(T,)``) plus
            ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"`` per level when ``level``
            is given (residual standard error, normal quantiles).
        """
        res = {"fitted": self.model_["fitted"]}

        if level is not None:
            level = sorted(level)
            res = utils._add_fitted_pi(res, self.model_["sigma"], level)
        
        return res
    
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """Stateless fit+predict in one call.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): In-sample exogenous regressors, shape
                ``(T, F)``. Only forwarded to ``conformity_scores``. Defaults
                to ``None``.
            X_future (Optional[jnp.ndarray]): Future exogenous regressors,
                shape ``(h, F)``. Ignored. Defaults to ``None``.
            level (Optional[List[int]]): Confidence levels in (0, 100).
                Requires ``conformal_params``. Defaults to ``None``.
            fitted (bool): Whether to return in-sample fitted values under
                ``"fitted"``. Defaults to ``False``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``); ``"fitted"`` (shape ``(T,)``)
            when ``fitted``; ``"lo-{lv}"``/``"hi-{lv}"`` per level when
            ``level`` is given, plus ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"``
            when both are set.

        Raises:
            ValueError: If ``level`` is given without ``conformal_params``, or
                if the series is too short for 2 CV windows.
        """
        y = utils.ensure_float(y)

        res = _croston_classic(y=y, h=h, fitted=fitted)

        if level is not None:
            if self.conformal_params is None:
                raise ValueError(
                    "You must instantiate the class with `conformal_params` "
                    "to calculate prediction intervals"
                )
            level = sorted(level)

            res = self.add_confidence_intervals(
                fcst=dict(res),
                cs=self.conformity_scores(y, X),
                level=level,
                method=self.conformal_params.method
            )

        if fitted and level is not None:
            sigma = utils.calculate_sigma(y - res["fitted"], y.size)
            res = {**res, **utils._add_fitted_pi_1(
                fitted=res["fitted"],
                sigmah=sigma,
                level=level
            )}
        
        return res

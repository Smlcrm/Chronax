"""
SeasonalExponentialSmoothing for seasonal time series in JAX.

Applies simple exponential smoothing (SES) independently to each seasonal
subseries (each phase of the cycle). The forecast tiles the per-season SES
forecasts over the horizon, which suits data with a stable seasonal pattern
and no clear trend.

Formula (per season s):
    ℓ[t,s] = α · y[t] + (1-α) · ℓ[t-1,s]
    ŷ[t+h, s] = ℓ[T, s]  (flat forecast per season)

Implementation:
    - `_ses_forecast_masked`: SES on padded fixed-size arrays (JIT-safe)
    - `_seasonal_exponential_smoothing_jit`: vmaps SES over all seasons at once
    - `_seasonal_exponential_smoothing`: dispatcher for the n < season_length case
    - Intervals are conformal only (``only_conformal_intervals = True``)
"""
from functools import partial
from typing import Optional, List, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from chronax.utils import ConformalIntervals
from chronax.models.base_forecaster import BaseForecaster
from chronax import utils
from chronax.utils import (
    ensure_float,
    _repeat_val_seas,
    _add_conformal_distribution_intervals,
    _get_conformal_method,
    _conformal_method,
    _store_cs,
    _add_conformal_intervals,
    _add_predict_conformal_intervals,
)


# =============================================================================
# Pure function kernels
# =============================================================================


@jax.jit
def _ses_forecast_masked(
    x: jnp.ndarray, alpha: jnp.ndarray, n_eff: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SES over first n_eff elements of padded x. Returns (forecast, fitted_padded)."""
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha
    n = x.shape[0]

    def body_fun(i, fitted_arr):
        def update():
            next_val = alpha * x[i - 1] + complement * fitted_arr[i - 1]
            return fitted_arr.at[i].set(next_val)
        return lax.cond(i < n_eff, update, lambda: fitted_arr)

    init_fitted = jnp.empty_like(x).at[0].set(x[0])
    fitted = lax.fori_loop(1, n, body_fun, init_fitted)
    forecast = lax.cond(
        n_eff > 0,
        lambda: alpha * x[n_eff - 1] + complement * fitted[n_eff - 1],
        lambda: jnp.asarray(jnp.nan, dtype=dtype),
    )
    fitted = fitted.at[0].set(jnp.asarray(jnp.nan, dtype=dtype))
    return forecast, fitted


@partial(jax.jit, static_argnums=(1, 2, 3))
def _seasonal_exponential_smoothing_jit(
    y: jnp.ndarray, h: int, fitted: bool, season_length: int, alpha: float
) -> Dict[str, jnp.ndarray]:
    """JIT-compiled path: vmap SES over the padded season matrix (n >= season_length)."""
    n = y.shape[0]
    max_len = (n + season_length - 1) // season_length

    # Build (season_length, max_len) matrix: row i = season i values, padded with 0
    i_grid = jnp.arange(season_length)[:, None]
    k_grid = jnp.arange(max_len)[None, :]
    indices = (i_grid + n % season_length) + k_grid * season_length
    valid = indices < n
    safe_idx = jnp.minimum(indices, n - 1)
    padded = jnp.where(valid, y[safe_idx], 0.0)
    n_eff = jnp.sum(valid, axis=1).astype(jnp.int32)

    # vmap SES over all seasons in one kernel launch
    alpha_arr = jnp.broadcast_to(alpha, (season_length,))
    forecasts, fitted_rows = jax.vmap(_ses_forecast_masked, in_axes=(0, 0, 0))(
        padded, alpha_arr, n_eff
    )

    out = _repeat_val_seas(forecasts, h)
    fcst: Dict[str, jnp.ndarray] = {"mean": out}

    if fitted:
        # Scatter fitted values back: one .at[].set per season (in JIT loop)
        fitted_vals = jnp.full_like(y, jnp.nan)

        def scatter_body(i, fv):
            init_idx = i + n % season_length
            idx = init_idx + jnp.arange(max_len, dtype=jnp.int32) * season_length
            mask = idx < n
            idx_safe = jnp.where(mask, idx, 0)
            vals = jnp.where(mask, fitted_rows[i], fv[0])
            return fv.at[idx_safe].set(vals)

        fitted_vals = lax.fori_loop(0, season_length, scatter_body, fitted_vals)
        fcst["fitted"] = fitted_vals

    return fcst


def _seasonal_exponential_smoothing(
    y: jnp.ndarray,
    h: int,
    fitted: bool,
    season_length: int,
    alpha: float,
) -> Dict[str, jnp.ndarray]:
    """Per-season SES with a shared alpha; the forecast tiles the season values.

    Returns ``{"mean": NaN}`` without ``"fitted"`` when ``n < season_length``.
    """
    n = y.size
    if n < season_length:
        return {"mean": jnp.full(h, jnp.nan, dtype=y.dtype)}

    return _seasonal_exponential_smoothing_jit(y, h, fitted, season_length, alpha)


# =============================================================================
# Model class
# =============================================================================

class SeasonalExponentialSmoothing(BaseForecaster):
    r"""SeasonalExponentialSmoothing model.

    Applies SES to each seasonal subseries with one shared smoothing
    parameter. With $t$ observations and season $s$, the one-step forecast is
    $\hat{y}_{t+1,s} = \alpha y_t + (1-\alpha) \hat{y}_{t-1,s}$. The method is
    Holt-Winters with the level and trend removed. Intervals are conformal
    only.

    References:
        - [Charles. C. Holt (1957). "Forecasting seasonals and trends by exponentially weighted moving averages", ONR Research Memorandum, Carnegie Institute of Technology 52.](https://www.sciencedirect.com/science/article/abs/pii/S0169207003001134).
        - [Peter R. Winters (1960). "Forecasting sales by exponentially weighted moving averages". Management Science](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.6.3.324).

    Args:
        season_length (int): Number of observations per seasonal cycle, e.g.
            24 for hourly data with a daily cycle.
        alpha (float): Smoothing parameter shared across seasons.
        alias (str): Custom name of the model. Defaults to ``"SeasonalES"``.
        prediction_intervals (Optional[ConformalIntervals]): Configuration for
            conformal prediction intervals; required for ``level``. Defaults
            to ``None``.
    """

    def __init__(
        self,
        season_length: int,
        alpha: float,
        alias: str = "SeasonalES",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ) -> None:
        self.season_length = season_length
        self.alpha = alpha
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.conformal_params = prediction_intervals
        self.only_conformal_intervals = True

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "SeasonalExponentialSmoothing":
        r"""Fit the SeasonalExponentialSmoothing model.

        Applies per-season SES and stores the per-season forecasts and the
        fitted values in ``model_``. When ``prediction_intervals`` is set,
        conformity scores are computed and cached for :meth:`predict`. A
        series shorter than ``season_length`` yields a NaN mean and no
        ``"fitted"`` entry.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (Optional[jnp.ndarray]): Exogenous regressors, shape ``(T, F)``.
                Only forwarded to ``conformity_scores``. Defaults to ``None``.

        Returns:
            SeasonalExponentialSmoothing: ``self``.

        Raises:
            ValueError: If ``prediction_intervals`` is set and the series is
                too short for 2 cross-validation (CV) windows.
        """
        y = ensure_float(y)
        mod = _seasonal_exponential_smoothing(
            y=y,
            season_length=self.season_length,
            alpha=self.alpha,
            fitted=True,
            h=self.season_length,
        )
        self.model_ = mod
        _store_cs(self, y=y, X=X)
        return self

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ) -> Dict[str, jnp.ndarray]:
        r"""Predict with the fitted SeasonalExponentialSmoothing model.

        Tiles the stored per-season forecasts over the horizon.

        Args:
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (Optional[List[int]]): Confidence levels in (0, 100), e.g.
                ``[80, 95]``. Requires ``prediction_intervals``. Defaults to
                ``None``.

        Returns:
            Dict[str, jnp.ndarray]: ``"mean"`` (shape ``(h,)``) plus
            ``"lo-{lv}"``/``"hi-{lv}"`` per level when ``level`` is given.

        Raises:
            ValueError: If called before ``fit``.
            Exception: If ``level`` is given without ``prediction_intervals``.
        """
        self._require_fitted()
        mean = _repeat_val_seas(self.model_["mean"], h=h)
        res = {"mean": mean}
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = _add_predict_conformal_intervals(self, res, level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")
        return res

    def predict_in_sample(self) -> Dict[str, jnp.ndarray]:
        r"""Return the in-sample fitted values of the fitted model.

        Returns:
            Dict[str, jnp.ndarray]: ``"fitted"`` (shape ``(T,)``).
        """
        res = {"fitted": self.model_["fitted"]}
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
        r"""Stateless fit+predict in one call.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): In-sample exogenous regressors, shape
                ``(T, F)``. Only forwarded to ``conformity_scores``. Defaults
                to ``None``.
            X_future (Optional[jnp.ndarray]): Future exogenous regressors,
                shape ``(h, F)``. Ignored. Defaults to ``None``.
            level (Optional[List[int]]): Confidence levels in (0, 100).
                Requires ``prediction_intervals``. Defaults to ``None``.
            fitted (bool): Whether to return in-sample fitted values under
                ``"fitted"``. Defaults to ``False``.

        Returns:
            Dict[str, jnp.ndarray]: ``"mean"`` (shape ``(h,)``, NaN when
            ``T < season_length``); ``"fitted"`` (shape ``(T,)``) when
            ``fitted``; ``"lo-{lv}"``/``"hi-{lv}"`` per level when ``level``
            is given.

        Raises:
            Exception: If ``level`` is given without ``prediction_intervals``.
            ValueError: If ``level`` is given and the series is too short for
                2 CV windows.
        """
        y = ensure_float(y)
        res = _seasonal_exponential_smoothing(
            y=y, h=h, fitted=fitted, alpha=self.alpha, season_length=self.season_length
        )
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = _add_conformal_intervals(self, fcst=res, y=y, X=X, level=level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")
        return res

"""
SeasonalExponentialSmoothing model with self-contained pure function kernels.
All math/logic lives in pure functions; the class is a barebones orchestrator.
"""
import math
from functools import partial
from typing import Optional, List, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from conformal_intervals import ConformalIntervals
from base_forecaster import BaseForecaster


# =============================================================================
# PURE FUNCTION KERNELS
# =============================================================================
# Architecture (functional, like auto_arima):
#   - _ses_forecast: core SES for a single season slice (JIT)
#   - _seasonal_exponential_smoothing: main kernel - per-season SES + tile (pure)
#   - _repeat_val_seas: forecast tiling (JIT)
#   - Class: barebones orchestrator, delegates to kernels above
# =============================================================================

def ensure_float(y: jnp.ndarray) -> jnp.ndarray:
    if not jnp.issubdtype(y.dtype, jnp.floating):
        return y.astype(jnp.float32)
    return y


@partial(jax.jit, static_argnums=(1,))
def _repeat_val_seas(season_vals: jnp.ndarray, h: int) -> jnp.ndarray:
    """
    Tile seasonal values to cover forecast horizon h.
    JAX equivalent of statsforecast.utils._repeat_val_seas()
    """
    repeats = math.ceil(h / season_vals.size)
    return jnp.tile(season_vals, repeats)[:h]


@jax.jit
def _ses_forecast(x: jnp.ndarray, alpha: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One-step ahead forecast and in-sample fitted values for SES."""
    x = ensure_float(x)
    dtype = x.dtype
    alpha = jnp.asarray(alpha, dtype=dtype)
    complement = jnp.asarray(1.0, dtype=dtype) - alpha

    n = x.shape[0]
    fitted = jnp.empty_like(x)
    fitted = fitted.at[0].set(x[0])

    def body_fun(i, carry):
        j, fitted_arr = carry
        next_val = alpha * x[j] + complement * fitted_arr[j]
        fitted_arr = fitted_arr.at[i].set(next_val)
        return (j + 1, fitted_arr)

    _, fitted = lax.fori_loop(1, n, body_fun, (0, fitted))
    forecast = alpha * x[n - 1] + complement * fitted[n - 1]
    fitted = fitted.at[0].set(jnp.asarray(jnp.nan, dtype=dtype))
    return forecast, fitted


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
    """
    Fully JIT-compiled path: vmap over seasons, no Python loops.
    Used when we can build a padded season matrix (n >= season_length).
    """
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
    """
    Core pure kernel: seasonal exponential smoothing fit and forecast.
    Math: per-season SES with shared alpha; forecast = tiled season_vals.
    Uses fully JIT-compiled path (vmap over seasons) when n >= season_length.
    """
    n = y.size
    if n < season_length:
        return {"mean": jnp.full(h, jnp.nan, dtype=y.dtype)}

    return _seasonal_exponential_smoothing_jit(y, h, fitted, season_length, alpha)


def _add_conformal_distribution_intervals(
    fcst: dict,
    cs: jnp.ndarray,
    level: list,
) -> dict:
    """
    Adds conformal intervals to the `fcst` dict based on conformal scores `cs`.
    `level` should be already sorted.
    """
    level = sorted(level)
    alphas = jnp.array([100 - lv for lv in level], dtype=jnp.float32)
    cuts_lower = (alphas / 200.0)[::-1]
    cuts_upper = 1.0 - (alphas / 200.0)
    cuts = jnp.concatenate([cuts_lower, cuts_upper])

    mean = fcst["mean"].reshape(1, -1)
    scores = jnp.vstack([mean - cs, mean + cs])
    quantiles = jnp.quantile(scores, cuts, axis=0)

    lo_cols = [f"lo-{lv}" for lv in reversed(level)]
    hi_cols = [f"hi-{lv}" for lv in level]
    out_cols = lo_cols + hi_cols

    for i, col in enumerate(out_cols):
        fcst[col] = quantiles[i]

    return fcst


def _get_conformal_method(method: str):
    available_methods = {
        "conformal_distribution": _add_conformal_distribution_intervals,
    }
    if method not in available_methods:
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f"please choose one of {', '.join(available_methods)}"
        )
    return available_methods[method]


def _conformal_method(self):
    return _get_conformal_method(self.prediction_intervals.method)


def _store_cs(self, y, X):
    if self.prediction_intervals is not None:
        self._cs = self.conformity_scores(y, X)


def _add_conformal_intervals(self, fcst, y, X, level):
    if self.prediction_intervals is not None and level is not None:
        cs = self.conformity_scores(y, X) if y is not None else self._cs
        conformal_fn = _conformal_method(self)
        res = conformal_fn(fcst=fcst, cs=cs, level=level)
        return res
    return fcst


def _add_predict_conformal_intervals(self, fcst, level):
    return _add_conformal_intervals(self, fcst=fcst, y=None, X=None, level=level)


# =============================================================================
# MODEL CLASS (barebones orchestrator)
# =============================================================================

class SeasonalExponentialSmoothing(BaseForecaster):
    r"""SeasonalExponentialSmoothing model.

    Uses a weighted average of all past observations where the weights decrease exponentially into the past.
    Suitable for data with no clear trend or seasonality.
    Assuming there are $t$ observations and season $s$, the one-step forecast is given by:
    $\hat{y}_{t+1,s} = \alpha y_t + (1-\alpha) \hat{y}_{t-1,s}$

    Notes:
        This method is an extremely simplified of Holt-Winter's method where the trend and level are set to zero.
        And a single seasonal smoothing parameter $\alpha$ is shared across seasons.

    References:
        - [Charles. C. Holt (1957). "Forecasting seasonals and trends by exponentially weighted moving averages", ONR Research Memorandum, Carnegie Institute of Technology 52.](https://www.sciencedirect.com/science/article/abs/pii/S0169207003001134).
        - [Peter R. Winters (1960). "Forecasting sales by exponentially weighted moving averages". Management Science](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.6.3.324).

    Args:
        alpha (float): Smoothing parameter.
        season_length (int): Number of observations per unit of time. Ex: 24 Hourly data.
        alias (str): Custom name of the model.
        prediction_intervals (Optional[ConformalIntervals]): Information to compute conformal prediction intervals.
            This is required for generating future prediction intervals.
    """

    def __init__(
        self,
        season_length: int,
        alpha: float,
        alias: str = "SeasonalES",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
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
    ):
        r"""Fit the SeasonalExponentialSmoothing model."""
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
    ):
        r"""Predict with fitted SeasonalExponentialSmoothing."""
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

    def predict_in_sample(self):
        r"""Access fitted SeasonalExponentialSmoothing insample predictions."""
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
    ):
        r"""Memory Efficient SeasonalExponentialSmoothing predictions."""
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

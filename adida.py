import jax.numpy as jnp
from jax import jit, lax
from functools import lru_cache
from utils import (
    ensure_float,
    calculate_sigma,
    _add_fitted_pi,
    _add_conformal_intervals,
    _add_predict_conformal_intervals,
    _intervals,
    _expand_fitted_intervals,
    _store_cs,
)
from base_forecaster import BaseForecaster
from typing import Optional, List
from conformal_intervals import ConformalIntervals
import warnings

_ALPHA_LOWER = 0.1
_ALPHA_UPPER = 0.3
_GOLDEN_RATIO = 0.6180339887498949
_ALPHA_SEARCH_ITERS = 8

_JIT_MIN_CHUNKS = 2


def _interval_mean(y: jnp.ndarray) -> jnp.ndarray:
    nonzero_mask = y != 0
    nonzero_count = jnp.count_nonzero(nonzero_mask)
    if nonzero_count == 0:
        return jnp.array(1.0, dtype=y.dtype)

    # mean(diff(nonzero_idxs + 1, prepend=0)) == (last_nonzero_idx + 1) / count_nonzero
    last_nonzero_idx = jnp.max(
        jnp.where(nonzero_mask, jnp.arange(y.shape[0], dtype=jnp.int32), 0)
    )
    return (last_nonzero_idx.astype(y.dtype) + 1.0) / nonzero_count.astype(y.dtype)


def _ses_sse(alpha: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    complement = 1.0 - alpha
    init_carry = (x[0], jnp.array(0.0, dtype=x.dtype))

    def body(i: int, carry):
        forecast, sse = carry
        forecast = alpha * x[i - 1] + complement * forecast
        err = x[i] - forecast
        return forecast, sse + err * err

    _, sse = lax.fori_loop(1, x.shape[0], body, init_carry)
    return sse


def _ses_forecast(alpha: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    complement = 1.0 - alpha

    def body(i: int, forecast: jnp.ndarray):
        return alpha * x[i - 1] + complement * forecast

    fitted_last = lax.fori_loop(1, x.shape[0], body, x[0])
    return alpha * x[-1] + complement * fitted_last


def _optimized_ses_forecast(x: jnp.ndarray) -> jnp.ndarray:
    left = jnp.array(_ALPHA_LOWER, dtype=x.dtype)
    right = jnp.array(_ALPHA_UPPER, dtype=x.dtype)
    ratio = jnp.array(_GOLDEN_RATIO, dtype=x.dtype)
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc = _ses_sse(c, x)
    fd = _ses_sse(d, x)

    def step(_: int, state):
        left, right, c, d, fc, fd = state

        def keep_left(curr):
            left, right, c, d, fc, fd = curr
            right = d
            d = c
            fd = fc
            c = right - ratio * (right - left)
            fc = _ses_sse(c, x)
            return left, right, c, d, fc, fd

        def keep_right(curr):
            left, right, c, d, fc, fd = curr
            left = c
            c = d
            fc = fd
            d = left + ratio * (right - left)
            fd = _ses_sse(d, x)
            return left, right, c, d, fc, fd

        return lax.cond(fc <= fd, keep_left, keep_right, state)

    left, right, _, _, _, _ = lax.fori_loop(
        0, _ALPHA_SEARCH_ITERS, step, (left, right, c, d, fc, fd)
    )
    alpha = 0.5 * (left + right)
    return _ses_forecast(alpha, x)


def _chunk_forecast_impl(series: jnp.ndarray, aggregation_level: int):
    lost_remainder_data = series.shape[0] % aggregation_level
    y_cut = series[lost_remainder_data:]
    n_chunks = y_cut.shape[0] // aggregation_level

    if n_chunks == 0:
        return jnp.array(0.0, dtype=series.dtype)

    aggregation_sums = y_cut[: n_chunks * aggregation_level].reshape(
        (n_chunks, aggregation_level)
    ).sum(axis=1)

    if n_chunks == 1:
        return aggregation_sums[0]

    return _optimized_ses_forecast(aggregation_sums)


@lru_cache(maxsize=256)
def _get_chunk_forecast_runner(aggregation_level: int):
    aggregation_level = max(1, int(aggregation_level))

    @jit
    def _run(series: jnp.ndarray):
        return _chunk_forecast_impl(series, aggregation_level)

    return _run

def _chunk_forecast_fast(y: jnp.ndarray, aggregation_level: int):
    aggregation_level = max(1, int(aggregation_level))
    n_chunks = y.shape[0] // aggregation_level

    # Only the degenerate case stays eager; warm path should hit cached JIT.
    if n_chunks < _JIT_MIN_CHUNKS:
        return _chunk_forecast_impl(y, aggregation_level)

    return _get_chunk_forecast_runner(aggregation_level)(y)


def _repeat_val_jax(val: jnp.ndarray, h: int):
    return jnp.full((h,), val)


def _adida_point(
    y: jnp.ndarray,  # time series
    h: int,  # forecasting horizon
):
    mean_interval = _interval_mean(y)
    aggregation_level = max(1, int(jnp.round(mean_interval).item()))
    sums_forecast = _chunk_forecast_fast(y, aggregation_level)
    forecast = sums_forecast / aggregation_level
    return {"mean": _repeat_val_jax(val=forecast, h=h)}


def _adida(
    y: jnp.ndarray,  # time series
    h: int,  # forecasting horizon
    fitted: bool,  # fitted values
):
    if not fitted:
        return _adida_point(y=y, h=h)

    if (y == 0).all():
        res = {"mean": jnp.zeros(h, dtype=y.dtype)}
        fitted_vals = jnp.zeros_like(y)
        if fitted_vals.size > 0:
            fitted_vals = fitted_vals.at[0].set(jnp.nan)
        res["fitted"] = fitted_vals
        return res

    res = _adida_point(y=y, h=h)
    warnings.warn("Computing fitted values for ADIDA is very expensive")

    y_intervals = _intervals(y)
    fitted_aggregation_levels = jnp.round(
        y_intervals.cumsum() / jnp.arange(1, y_intervals.size + 1)
    )
    fitted_aggregation_levels = _expand_fitted_intervals(
        jnp.append(jnp.nan, fitted_aggregation_levels), y
    )[1:].astype(jnp.int32)

    sums_fitted = []
    for i, agg_lvl in enumerate(fitted_aggregation_levels):
        agg_lvl_int = max(1, int(agg_lvl))
        sums_fitted.append(_chunk_forecast_fast(y[: i + 1], agg_lvl_int))
    sums_fitted = jnp.asarray(sums_fitted, dtype=y.dtype)

    res["fitted"] = jnp.append(jnp.nan, sums_fitted / fitted_aggregation_levels)
    return res


def _fit_adida(model: "ADIDA", y: jnp.ndarray, X: Optional[jnp.ndarray]):
    y = ensure_float(y)
    model._y = y
    model.model_ = _adida_point(y=y, h=1)
    if model.prediction_intervals is not None:
        _store_cs(model, y=y, X=X)
    return model


def _point_predict(model: "ADIDA", h: int):
    return {"mean": _repeat_val_jax(val=model.model_["mean"][0], h=h)}


def _predict_without_intervals(
    model: "ADIDA",
    h: int,
    X: Optional[jnp.ndarray],
    level: Optional[List[int]],
):
    del X, level
    return _point_predict(model, h)


def _predict_with_intervals(
    model: "ADIDA",
    h: int,
    X: Optional[jnp.ndarray],
    level: Optional[List[int]],
):
    del X
    res = _point_predict(model, h)
    if level is None:
        return res
    return _add_predict_conformal_intervals(model, res, sorted(level))


def _forecast_core(y: jnp.ndarray, h: int, fitted: bool):
    y = ensure_float(y)
    res = _adida(y=y, h=h, fitted=True) if fitted else _adida_point(y=y, h=h)
    return y, res


def _forecast_without_intervals(
    model: "ADIDA",
    y: jnp.ndarray,
    h: int,
    X: Optional[jnp.ndarray],
    X_future: Optional[jnp.ndarray],
    level: Optional[List[int]],
    fitted: bool,
):
    del model, X, X_future, level
    _, res = _forecast_core(y=y, h=h, fitted=fitted)
    return res


def _forecast_with_intervals(
    model: "ADIDA",
    y: jnp.ndarray,
    h: int,
    X: Optional[jnp.ndarray],
    X_future: Optional[jnp.ndarray],
    level: Optional[List[int]],
    fitted: bool,
):
    del X_future
    y, res = _forecast_core(y=y, h=h, fitted=fitted)
    if level is None:
        return res
    level = sorted(level)
    res = _add_conformal_intervals(model, fcst=res, y=y, X=X, level=level)
    if fitted:
        sigma = calculate_sigma(y - res["fitted"], y.size)
        res = _add_fitted_pi(res=res, se=sigma, level=level)
    return res


def _predict_in_sample_adida(model: "ADIDA", level: Optional[List[int]] = None):
    fitted = _adida(y=model._y, h=1, fitted=True)["fitted"]
    res = {"fitted": fitted}
    if level is not None:
        sigma = calculate_sigma(model._y - fitted, model._y.size)
        res = _add_fitted_pi(res=res, se=sigma, level=level)
    return res


class ADIDA(BaseForecaster):
    def __init__(
        self,
        alias: str = "ADIDA",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        """ADIDA model.

        Aggregate-Dissagregate Intermittent Demand Approach: Uses temporal aggregation to reduce the
        number of zero observations. Once the data has been agregated, it uses the optimized SES to
        generate the forecasts at the new level. It then breaks down the forecast to the original
        level using equal weights.

        ADIDA specializes on sparse or intermittent series are series with very few non-zero observations.
        They are notoriously hard to forecast, and so, different methods have been developed
        especifically for them.

        References:
            - [Nikolopoulos, K., Syntetos, A. A., Boylan, J. E., Petropoulos, F., & Assimakopoulos, V. (2011). An aggregate–disaggregate intermittent demand approach (ADIDA) to forecasting: an empirical proposition and analysis. Journal of the Operational Research Society, 62(3), 544-554.](https://researchportal.bath.ac.uk/en/publications/an-aggregate-disaggregate-intermittent-demand-approach-adida-to-f).

        Args:
            alias (str, optional): Custom name of the model. Defaults to "ADIDA".
            prediction_intervals (Optional[ConformalIntervals], optional): Information to compute conformal prediction intervals.
                By default, the model will compute the native prediction intervals. Defaults to None.
        """
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.only_conformal_intervals = True
        self.conformal_params = prediction_intervals
        self._predict_impl = (
            _predict_with_intervals
            if prediction_intervals is not None
            else _predict_without_intervals
        )
        self._forecast_impl = (
            _forecast_with_intervals
            if prediction_intervals is not None
            else _forecast_without_intervals
        )

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ):
        """Fit the ADIDA model.

        Fit an ADIDA to a time series (numpy array) `y`.

        Args:
            y (np.ndarray): Clean time series of shape (t, ).
            X (Optional[np.ndarray], optional): Optional exogenous variables. Defaults to None.

        Returns:
            ADIDA: ADIDA fitted model.
        """
        return _fit_adida(self, y, X)

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ):
        """Predict with fitted ADIDA.

        Args:
            h (int): Forecast horizon.
            X (Optional[np.ndarray], optional): Optional exogenous of shape (h, n_x). Defaults to None.
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        return self._predict_impl(self, h, X, level)

    def predict_in_sample(self, level: Optional[List[int]] = None):
        """Access fitted ADIDA insample predictions.

        Args:
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions and `level_*` for probabilistic predictions.
        """
        return _predict_in_sample_adida(self, level)

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        """Memory Efficient ADIDA predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (np.ndarray): Clean time series of shape (n,).
            h (int): Forecast horizon.
            X (Optional[np.ndarray], optional): Optional insample exogenous of shape (t, n_x). Defaults to None.
            X_future (Optional[np.ndarray], optional): Optional exogenous of shape (h, n_x). Defaults to None.
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.
            fitted (bool, optional): Whether or not to return insample predictions. Defaults to False.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        return self._forecast_impl(self, y, h, X, X_future, level, fitted)
    
# # Test Cases
# def test():
#     # simple increasing series
#     y = jnp.arange(24.0)

#     ci = ConformalIntervals(method="conformal_distribution")
#     model = ADIDA(prediction_intervals=ci)

#     fitted_model = model.fit(y)

#     result = fitted_model.predict(h=12, level=[60, 75])
#     forecast = fitted_model.forecast(y, h=12, level=[80, 95])

#     # ---- Assertions ----
#     assert "mean" in result, "Missing mean forecast"
#     assert len(result["mean"]) == 12, "Forecast length mismatch"

#     for lvl in [60, 75]:
#         assert f"lo-{lvl}" in result, f"Missing lower bound for {lvl}% interval"
#         assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"
#         assert len(result[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
#         assert len(result[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

#     for lvl in [80, 95]:
#         assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
#         assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"
#         assert len(forecast[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
#         assert len(forecast[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

#     print("Test passed!")


# if __name__ == "__main__":
#     test()

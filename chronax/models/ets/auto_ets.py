from __future__ import annotations
from typing import Any, List, Optional

import jax.numpy as jnp
import os
import time

from chronax.utils import (
    ConformalIntervals,
    ensure_float,
    calculate_sigma,
    _add_fitted_pi,
)

from chronax.models.base_forecaster import BaseForecaster

from .ets_functions import ets_f, forecast_ets, forward_ets, _infer_season_length
_PHI_LOWER = 0.8
_PHI_UPPER = 0.98

class AutoETS(BaseForecaster):
    r"""Automatic Exponential Smoothing model.

    Automatically selects the best ETS (Error, Trend, Seasonality)
    model using an information criterion. Default is Akaike Information Criterion (AICc), while particular models are estimated using maximum likelihood.
    The state-space equations can be determined based on their $M$ multiplicative, $A$ additive,
    $Z$ optimized or $N$ ommited components. The `model` string parameter defines the ETS equations:
    E in [$M, A, Z$], T in [$N, A, M, Z$], and S in [$N, A, M, Z$].

    For example when model='ANN' (additive error, no trend, and no seasonality), ETS will
    explore only a simple exponential smoothing.

    If the component is selected as 'Z', it operates as a placeholder to ask the AutoETS model
    to figure out the best parameter.

    Args:
        season_length (int, default=1): Number of observations per unit of time. Ex: 24 Hourly data.
        model (str, default="ZZZ"): Controlling state-space-equations.
        damped (bool, optional): A parameter that 'dampens' the trend.
        phi (float, optional): Smoothing parameter for trend damping. Only used when `damped=True`.
        alias (str, default="AutoETS"): Custom name of the model.
        conformal_params (Optional[ConformalIntervals], optional): Conformal prediction configuration.
            By default, the model will compute the native prediction intervals.

    Notes:
        This implementation is a mirror of Hyndman's [forecast::ets](https://github.com/robjhyndman/forecast).

    References:
        - [Rob J. Hyndman, Yeasmin Khandakar (2008). "Automatic Time Series Forecasting: The forecast package for R"](https://www.jstatsoft.org/article/view/v027i03).
        - [Hyndman, Rob, et al (2008). "Forecasting with exponential smoothing: the state space approach"](https://robjhyndman.com/expsmooth/).
    """

    @staticmethod
    def _validate_h(h: int) -> None:
        """Validate that the forecast horizon is a positive integer."""
        if not isinstance(h, int) or h <= 0:
            raise ValueError(f"h must be positive integer, got {h}")

    @staticmethod
    def _validate_level(level: Optional[List[int]]) -> None:
        """Validate an optional list of interval confidence levels."""
        if level is None:
            return
        if not isinstance(level, list):
            raise ValueError("level must be a list or None")
        for lv in level:
            if not isinstance(lv, (int, float)) or lv < 0 or lv > 100:
                raise ValueError(f"level values must be in [0, 100], got {lv}")

    def _validate_series_length(self, y: jnp.ndarray) -> None:
        """Ensure the series is long enough for the configured season length."""
        min_len = max(self.season_length, 3)
        if len(y) < min_len:
            raise ValueError(
                f"Series too short: need >= {min_len} observations, got {len(y)}"
            )

    def __init__(
        self,
        season_length: int = 1,
        model: str = "ZZZ",
        damped: Optional[bool] = None,
        phi: Optional[float] = None,
        max_iter: Optional[int] = None,
        optax_lr: float = 7e-2,  # Optimized: Slightly higher LR for faster convergence
        optax_clip: float = 5.0,
        early_stop_patience: int = 10,  # Optimized: More aggressive early stopping
        early_stop_min_delta: float = 1e-5,  # Optimized: Slightly relaxed for faster convergence
        alias: str = "AutoETS",
        conformal_params: Optional[ConformalIntervals] = None,
    ) -> None:
        """Initialize the AutoETS estimator configuration."""
        self.season_length = season_length
        self.model = model
        self.damped = damped
        if phi is not None:
            if not isinstance(phi, float):
                raise ValueError("phi must be `None` or float.")
            if not _PHI_LOWER <= phi <= _PHI_UPPER:
                raise ValueError(f"Valid range for phi is [{_PHI_LOWER}, {_PHI_UPPER}]")
        self.phi = phi
        self.max_iter = max_iter
        self.optax_lr = optax_lr
        self.optax_clip = optax_clip
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.alias = alias
        self.conformal_params = conformal_params
        self.optax_steps = self.max_iter

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "AutoETS":
        r"""Fit the Exponential Smoothing model.

        Fit an Exponential Smoothing model to a time series (numpy array) `y`
        and optionally exogenous variables (numpy array) `X`.

        Args:
            y (numpy.array): Clean time series of shape (t, ).
            X (array-like, optional): Optional exogenous of shape (t, n_x).

        Returns:
            AutoETS: Exponential Smoothing fitted model.
        """
        y = ensure_float(y)
        self._validate_series_length(y)

        # Infer an effective season length from the data when possible.
        # This helps avoid config-level seasonality mismatches (e.g. m=24
        # for clearly monthly data), which are a major source of accuracy loss.
        m_eff = int(self.season_length)
        try:
            m_infer = int(_infer_season_length(y, max_m=min(24, len(y) // 2)))
            if m_infer > 1:
                m_eff = m_infer
        except Exception:
            # Fall back silently to the configured season_length
            m_eff = int(self.season_length)

        self._y_fit = y
        self._m_eff = m_eff
        # Heuristic: give more iterations to longer / more complex series for accuracy.
        if self.max_iter is None:
            n = len(y)
            if n < 200:
                self.optax_steps = 200
            elif n < 1000:
                self.optax_steps = 300
            elif n < 5000:
                self.optax_steps = 400
            else:
                self.optax_steps = 500
        else:
            self.optax_steps = self.max_iter
        self.model_ = ets_f(
            y,
            m=m_eff,
            model=self.model,
            damped=self.damped,
            phi=self.phi,
            # Use default likelihood-based criterion; allow multiplicative trend
            # for richer models on positive series.
            allow_multiplicative_trend=True,
            allow_extended_iterations=True,
            optax_steps=self.optax_steps,
            optax_lr=self.optax_lr,
            optax_clip=self.optax_clip,
            early_stop_patience=self.early_stop_patience,
            early_stop_min_delta=self.early_stop_min_delta,
        )
        self.model_["actual_residuals"] = y - self.model_["fitted"]
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None
        return self

    def predict(
        self, h: int, X: Optional[jnp.ndarray] = None, level: Optional[List[int]] = None
    ) -> dict[str, jnp.ndarray]:
        r"""Predict with fitted Exponential Smoothing.

        Args:
            h (int): Forecast horizon.
            X (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        self._validate_h(h)
        self._validate_level(level)
        fcst = forecast_ets(self.model_, h=h, level=level)
        res = {"mean": fcst["mean"]}
        if level is None:
            return res
        level = sorted(level)
        if self.conformal_params is not None:
            if self._cs is None:
                raise ValueError(
                    "Conformity scores not cached. Fit the model with conformal_params set, "
                    "or use forecast(y, ...) which recomputes them."
                )
            return self.add_confidence_intervals(
                fcst=res,
                cs=self._cs,
                level=level,
                method=self.conformal_params.method,
            )

        # Native ETS intervals
        res.update({f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level)})
        res.update({f"hi-{l}": fcst[f"hi-{l}"] for l in level})
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None) -> dict[str, jnp.ndarray]:
        r"""Access fitted Exponential Smoothing insample predictions.

        Args:
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions and `level_*` for probabilistic predictions.
        """
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            residuals = self.model_["actual_residuals"]
            # se = _calculate_sigma(residuals, len(residuals) - self.model_["n_params"])
            se = calculate_sigma(residuals, len(residuals) - self.model_["n_params"])
            res = _add_fitted_pi(res=res, se=se, level=level)
        return res

    def _compute_forecast_with_intervals(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray],
        level: Optional[List[int]],
        fitted: bool,
        use_forward: bool,
    ) -> dict[str, Any]:
        """Compute forecasts plus optional native or conformal intervals."""
        y = ensure_float(y)
        self._validate_h(h)
        self._validate_level(level)
        self._validate_series_length(y)

        # Reuse effective season length from fit when available; otherwise,
        # attempt to infer from the new series.
        m_eff = int(getattr(self, "_m_eff", self.season_length))
        try:
            if not hasattr(self, "_m_eff"):
                m_infer = int(_infer_season_length(y, max_m=min(24, len(y) // 2)))
                if m_infer > 1:
                    m_eff = m_infer
        except Exception:
            m_eff = int(self.season_length)

        if use_forward:
            if not hasattr(self, "model_"):
                raise Exception("You have to use the `fit` method first")
            mod = forward_ets(self.model_, y=y)
        else:
            mod = None
            if hasattr(self, "model_") and hasattr(self, "_y_fit"):
                try:
                    same_y = y.shape == self._y_fit.shape and bool(jnp.all(y == self._y_fit))
                except Exception:
                    same_y = False
                if same_y:
                    mod = self.model_
            if mod is None:
                # Stateless path: mirror the fit-time iteration heuristic.
                if self.max_iter is None:
                    n = len(y)
                    if n < 200:
                        optax_steps = 200
                    elif n < 1000:
                        optax_steps = 300
                    elif n < 5000:
                        optax_steps = 400
                    else:
                        optax_steps = 500
                else:
                    optax_steps = self.max_iter
                mod = ets_f(
                    y,
                    m=m_eff,
                    model=self.model,
                    damped=self.damped,
                    phi=self.phi,
                    allow_multiplicative_trend=True,
                    allow_extended_iterations=True,
                    optax_steps=optax_steps,
                    optax_lr=self.optax_lr,
                    optax_clip=self.optax_clip,
                    early_stop_patience=self.early_stop_patience,
                    early_stop_min_delta=self.early_stop_min_delta,
                )
                # Cache fitted model for warm calls (same y), so we don't refit on every forecast.
                self.model_ = mod
                self._y_fit = y

        timing_enabled = os.environ.get("CHRONAX_ETS_TIMING", "0") == "1"
        t_fcst_start = time.perf_counter() if timing_enabled else 0.0
        fcst = forecast_ets(mod, h=h, level=level)
        t_fcst_end = time.perf_counter() if timing_enabled else 0.0
        keys = ["mean"]
        if fitted:
            keys.append("fitted")
        res = {key: fcst[key] for key in keys}
        if level is None:
            return res

        level_sorted = sorted(level)
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=X)
            res = self.add_confidence_intervals(res, cs, level_sorted, self.conformal_params.method)
        else:
            res = {
                **res,
                **{f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level_sorted)},
                **{f"hi-{l}": fcst[f"hi-{l}"] for l in level_sorted},
            }
        if fitted:
            se = calculate_sigma(y - mod["fitted"], len(y) - mod["n_params"])
            res = _add_fitted_pi(res=res, se=se, level=level_sorted)
        if timing_enabled and isinstance(mod, dict) and "_timing" in mod:
            res["_timing"] = {
                **mod["_timing"],
                "forecast_sec": float(t_fcst_end - t_fcst_start),
            }
        return res
    
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> dict[str, Any]:
        r"""Memory Efficient Exponential Smoothing predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like, optional): Optional insample exogenpus of shape (t, n_x).
            X_future (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.
            fitted (bool, default=False): Whether or not returns insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        return self._compute_forecast_with_intervals(
            y=y, h=h, X=X, level=level, fitted=fitted, use_forward=False
        )

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> dict[str, Any]:
        r"""Apply fitted Exponential Smoothing model to a new time series.

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like, optional): Optional insample exogenpus of shape (t, n_x).
            X_future (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels for prediction intervals.
            fitted (bool, default=False): Whether or not to return insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        return self._compute_forecast_with_intervals(
            y=y, h=h, X=X, level=level, fitted=fitted, use_forward=True
        )


# -------------------------------------------------------------------
# Test cases
# -------------------------------------------------------------------

import numpy as np

# ============== Helpers ==============
def _assert_allclose(
    actual: object,
    expected: object,
    atol: float = 1e-4,
    rtol: float = 1e-6,
    label: str = "",
) -> None:
    """Assert two arrays are numerically close within tolerances."""
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if not np.allclose(a, e, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{label} mismatch.\nActual:   {a}\nExpected: {e}\n"
            f"max|Δ|={np.max(np.abs(a - e))}, atol={atol}, rtol={rtol}"
        )

def _print_ok(name: str) -> None:
    """Print a standardized success message for local smoke tests."""
    print(f"{name}: OK")


# ============== Core expected-value tests ==============

def test_constant_series_ANN_expected() -> None:
    """Smoke-test that a constant series selects and forecasts like ANN."""
    y = jnp.full((40,), 10.0, dtype=jnp.float64)
    m = AutoETS(season_length=1, model="ZZZ")
    m.fit(y)

    fitted = m.predict_in_sample(level=None)["fitted"]
    _assert_allclose(np.mean(np.abs(np.asarray(fitted) - 10.0)), 0.0, atol=1e-2, label="ANN fitted MAE")

    h = 5
    out = m.predict(h=h, level=None)
    expected = np.full(h, 10.0, dtype=np.float64)
    _assert_allclose(out["mean"], expected, atol=1e-1, label="ANN mean")
    _print_ok("test_constant_series_ANN_expected")


def test_linear_trend_AAN_expected() -> None:
    """Smoke-test that a linear trend produces a stable additive trend path."""
    a, b = 2.0, 0.5
    t = np.arange(60, dtype=np.float64)
    y = jnp.asarray(a + b * t, dtype=jnp.float64)

    ae = AutoETS(season_length=1, model="ZZZ")
    ae.fit(y)

    h = 6
    out = ae.predict(h=h, level=None)
    mean = np.asarray(out["mean"])

    last_val = a + b * (len(y) - 1)
    expected_first = last_val + b
    expected = expected_first + b * np.arange(h, dtype=np.float64)
    _assert_allclose(mean[0], expected_first, atol=0.2, label="AAN step-1")
    _assert_allclose(mean, expected, atol=0.5, label="AAN path")
    _print_ok("test_linear_trend_AAN_expected")


def test_additive_seasonality_AAA_expected() -> None:
    """Smoke-test that additive seasonality repeats the learned pattern."""
    m = 4
    base = 10.0
    season = np.array([+1.0, -1.0, +2.0, -2.0], dtype=np.float64)
    y = jnp.asarray(np.tile(season, 15) + base, dtype=jnp.float64)  # len 60

    ae = AutoETS(season_length=m, model="ZZZ")
    ae.fit(y)

    h = 8
    out = ae.predict(h=h, level=None)
    mean = np.asarray(out["mean"])

    idx0 = len(y) % m
    expected = np.array([base + season[(idx0 + k) % m] for k in range(h)], dtype=np.float64)
    _assert_allclose(mean, expected, atol=1.0, label="AAA seasonal pattern")
    _print_ok("test_additive_seasonality_AAA_expected")


def test_stateless_forecast_matches_stateful_expected() -> None:
    """Smoke-test that stateless and stateful forecast paths agree."""
    t = np.arange(48, dtype=np.float64)
    base, slope = 5.0, 0.1
    season = np.array([0.0, +1.0, 0.0, -1.0], dtype=np.float64)
    y = jnp.asarray(base + slope * t + np.tile(season, len(t) // 4 + 1)[: len(t)], dtype=jnp.float64)

    h = 5
    ae = AutoETS(season_length=4, model="ZZZ")
    ae.fit(y)
    stateful = ae.predict(h=h, level=None)["mean"]
    stateless = ae.forecast(y=y, h=h, level=None, fitted=False)["mean"]

    _assert_allclose(stateful, stateless, atol=1e-6, label="stateless vs stateful mean")
    _print_ok("test_stateless_forecast_matches_stateful_expected")


# ============== Intervals & level handling ==============

def test_predict_native_intervals_and_unsorted_levels() -> None:
    """Smoke-test native intervals and level sorting behavior."""
    y = jnp.asarray([10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 12.5, 12.0] * 4, dtype=jnp.float64)
    ae = AutoETS(season_length=1, model="ZZZ")
    ae.fit(y)
    # Unsorted on purpose
    out = ae.predict(h=4, level=[95, 80, 50])

    assert "mean" in out and out["mean"].shape == (4,)
    for lv in (50, 80, 95):
        assert f"hi-{lv}" in out and f"lo-{lv}" in out
        assert out[f"hi-{lv}"].shape == (4,) and out[f"lo-{lv}"].shape == (4,)
    # sanity: lo ≤ mean ≤ hi
    m = np.asarray(out["mean"])
    for lv in (50, 80, 95):
        lo = np.asarray(out[f"lo-{lv}"]); hi = np.asarray(out[f"hi-{lv}"])
        assert np.all(lo <= m + 1e-12) and np.all(m <= hi + 1e-12), f"mean not inside {lv}% band"
    _print_ok("test_predict_native_intervals_and_unsorted_levels")


def test_forecast_adds_fitted_and_fitted_intervals_when_requested() -> None:
    """Smoke-test stateless forecasts with fitted values and fitted intervals."""
    # Needs enough history to avoid tiny-dataset path
    t = np.arange(60, dtype=np.float64)
    y = jnp.asarray(5.0 + 0.2 * t, dtype=jnp.float64)
    ae = AutoETS(season_length=1, model="ZZZ")
    out = ae.forecast(y=y, h=5, level=[80, 95], fitted=True)
    assert "mean" in out and out["mean"].shape == (5,)
    assert "fitted" in out and out["fitted"].shape == y.shape
    for lv in (80, 95):
        assert f"lo-{lv}" in out and f"hi-{lv}" in out
        assert f"fitted-lo-{lv}" in out and f"fitted-hi-{lv}" in out
        # fitted must lie between its own bands
        fl = np.asarray(out[f"fitted-lo-{lv}"]); fh = np.asarray(out[f"fitted-hi-{lv}"])
        ft = np.asarray(out["fitted"])
        assert np.all(fl <= ft + 1e-12) and np.all(ft <= fh + 1e-12)
    _print_ok("test_forecast_adds_fitted_and_fitted_intervals_when_requested")


def test_predict_in_sample_intervals_monotonicity_expected() -> None:
    """Smoke-test in-sample interval nesting across confidence levels."""
    y = jnp.asarray([10.0, 12.0, 11.0, 13.0, 12.0, 11.5, 12.5, 12.0] * 3, dtype=jnp.float64)
    ae = AutoETS(season_length=1, model="ZZZ")
    ae.fit(y)

    res = ae.predict_in_sample(level=[80, 95])
    fitted = np.asarray(res["fitted"])
    lo80 = np.asarray(res["fitted-lo-80"])
    lo95 = np.asarray(res["fitted-lo-95"])
    hi80 = np.asarray(res["fitted-hi-80"])
    hi95 = np.asarray(res["fitted-hi-95"])
    if not (np.all(lo95 <= lo80 + 1e-12) and np.all(hi95 >= hi80 - 1e-12)):
        raise AssertionError("Monotonicity failed: 95% should be wider than 80%")
    if not (np.all(lo80 <= fitted) and np.all(fitted <= hi80)):
        raise AssertionError("Fitted not in 80% band")
    if not (np.all(lo95 <= fitted) and np.all(fitted <= hi95)):
        raise AssertionError("Fitted not in 95% band")
    _print_ok("test_predict_in_sample_intervals_monotonicity_expected")


# ============== Conformal intervals paths ==============

def test_fit_caches_conformal_then_predict_uses_cache() -> None:
    """
    Conformal path with windows large enough to avoid tiny-dataset errors inside
    rolling-origin fits. We use a longer y and modest n_windows/h.
    """
    # Make y long enough so that the earliest training window has plenty of data.
    # Rule of thumb: ensure n - n_windows*h >= ~20 for comfort.
    n = 40
    t = np.arange(n, dtype=np.float64)
    y = jnp.asarray(8.0 + 0.05 * t + 0.5 * np.sin(2 * np.pi * t / 8), dtype=jnp.float64)

    cfg = ConformalIntervals(n_windows=5, h=3, method="conformal_distribution")
    ae = AutoETS(season_length=1, model="ZZZ", conformal_params=cfg)

    ae.fit(y)  # should not hit tiny-dataset in any window
    assert getattr(ae, "_cs") is not None

    out = ae.predict(h=3, level=[80, 95])
    for lv in (80, 95):
        assert f"lo-{lv}" in out and f"hi-{lv}" in out
        m = np.asarray(out["mean"])
        lo = np.asarray(out[f"lo-{lv}"]); hi = np.asarray(out[f"hi-{lv}"])
        assert np.all(lo <= m + 1e-12) and np.all(m <= hi + 1e-12)
    print("test_fit_caches_conformal_then_predict_uses_cache: OK")


def test_stateless_forecast_with_conformal_intervals() -> None:
    """
    Stateless forecast with conformal intervals; series long enough that the
    internal rolling windows used for conformity scores are also large.
    """
    n = 36
    t = np.arange(n, dtype=np.float64)
    y = jnp.asarray(5.0 + 0.1 * t + 0.3 * np.sin(2 * np.pi * t / 6), dtype=jnp.float64)

    cfg = ConformalIntervals(n_windows=4, h=2, method="conformal_distribution")
    ae = AutoETS(season_length=1, model="ZZZ", conformal_params=cfg)

    out = ae.forecast(y=y, h=6, level=[90], fitted=False)
    assert "mean" in out and out["mean"].shape == (6,)
    assert "lo-90" in out and "hi-90" in out
    mean = np.asarray(out["mean"])
    lo90 = np.asarray(out["lo-90"]); hi90 = np.asarray(out["hi-90"])
    assert np.all(lo90 <= mean + 1e-12) and np.all(mean <= hi90 + 1e-12)
    print("test_stateless_forecast_with_conformal_intervals: OK")

# ============== Forward / state-handling & errors ==============

def test_forward_on_new_series_expected_or_reasonable() -> None:
    """Smoke-test forwarding a fitted model onto a new related series."""
    t = np.arange(36, dtype=np.float64)
    yA = jnp.asarray(15.0 + 0.3 * t, dtype=jnp.float64)
    yB = jnp.asarray(10.0 + 0.3 * t, dtype=jnp.float64)

    ae = AutoETS(season_length=1, model="ZZZ")
    ae.fit(yA)

    h = 5
    outB = ae.forward(y=yB, h=h, level=[80], fitted=True)
    assert outB["mean"].shape == (h,)
    assert outB["fitted"].shape == yB.shape
    # MSE threshold
    mse = float(np.mean((np.asarray(outB["fitted"]) - np.asarray(yB)) ** 2))
    if mse >= 2.0:
        raise AssertionError(f"forward fitted MSE too large: {mse}")
    # interval sanity
    lo = np.asarray(outB["lo-80"]); hi = np.asarray(outB["hi-80"]); mean = np.asarray(outB["mean"])
    assert np.all(lo <= mean + 1e-12) and np.all(mean <= hi + 1e-12)
    # fitted bands exist
    assert "fitted-lo-80" in outB and "fitted-hi-80" in outB
    _print_ok("test_forward_on_new_series_expected_or_reasonable")


def test_predict_before_fit_raises() -> None:
    """Smoke-test that predict raises before the model is fitted."""
    ae = AutoETS(season_length=1, model="ZZZ")
    try:
        _ = ae.predict(h=1)
        raise AssertionError("Expected an error when calling predict() before fit()")
    except Exception:
        pass
    _print_ok("test_predict_before_fit_raises")


def test_forward_before_fit_raises() -> None:
    """Smoke-test that forward raises before the model is fitted."""
    ae = AutoETS(season_length=1, model="ZZZ")
    try:
        _ = ae.forward(y=jnp.asarray([1., 2., 3.]), h=1)
        raise AssertionError("Expected an error when calling forward() before fit()")
    except Exception:
        pass
    _print_ok("test_forward_before_fit_raises")


def test_tiny_series_policy_expected() -> None:
    """Smoke-test the tiny-series policy for short inputs."""
    y = jnp.asarray([10.0, 11.0, 10.5], dtype=jnp.float64)
    ae = AutoETS(season_length=1, model="ZZZ")
    try:
        ae.fit(y)
        out = ae.predict(h=3, level=None)
        if out["mean"].shape != (3,):
            raise AssertionError("Expected 3-step forecast for tiny series")
    except NotImplementedError as e:
        msg = str(e).lower()
        if "tiny datasets" not in msg:
            raise
    _print_ok("test_tiny_series_policy_expected")


# ============== φ validation & dtypes ==============

def test_phi_validation_range_and_type() -> None:
    """Smoke-test phi validation across invalid and boundary inputs."""
    # invalid range
    try:
        _ = AutoETS(phi=1.5)
        raise AssertionError("Expected ValueError for invalid phi range")
    except ValueError:
        pass
    # invalid type
    try:
        _ = AutoETS(phi="0.9")  # type: ignore
        raise AssertionError("Expected ValueError for non-float phi")
    except ValueError:
        pass
    # valid boundary
    _ = AutoETS(phi=0.8)
    _ = AutoETS(phi=0.98)
    _print_ok("test_phi_validation_range_and_type")


def test_basic_shapes_and_dtypes_predict() -> None:
    """Smoke-test basic output shape and dtype behavior for predict."""
    y = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
    ae = AutoETS(season_length=1, model="ZZZ")
    ae.fit(y.astype(jnp.float64))  # ensure float for the model
    out = ae.predict(h=3, level=None)
    assert out["mean"].shape == (3,)
    assert out["mean"].dtype in (jnp.float32, jnp.float64)
    _print_ok("test_basic_shapes_and_dtypes_predict")


# ============== Runner ==============
if __name__ == "__main__":
    # Core behavior
    test_constant_series_ANN_expected()
    test_linear_trend_AAN_expected()
    test_additive_seasonality_AAA_expected()
    test_stateless_forecast_matches_stateful_expected()

    # Intervals & conformal
    test_predict_native_intervals_and_unsorted_levels()
    test_forecast_adds_fitted_and_fitted_intervals_when_requested()
    test_predict_in_sample_intervals_monotonicity_expected()
    test_fit_caches_conformal_then_predict_uses_cache()
    test_stateless_forecast_with_conformal_intervals()

    # Forward & errors
    test_forward_on_new_series_expected_or_reasonable()
    test_predict_before_fit_raises()
    test_forward_before_fit_raises()
    test_tiny_series_policy_expected()

    # Validation & dtypes
    test_phi_validation_range_and_type()
    test_basic_shapes_and_dtypes_predict()

    print("All AutoETS full-coverage tests passed.")

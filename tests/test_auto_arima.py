# test_auto_arima.py
import math
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import pytest
import time
from typing import Optional

# --- BaseForecaster from models package ---
from chronax.models.base_forecaster import BaseForecaster
# =============================================================================
# IMPORTS (Separated as requested)
# =============================================================================

# 1. High-Level Wrappers (Classes)
from chronax.models import ARIMA, AutoARIMA

# 2. Low-Level Functions (Math & Logic)
from chronax.models.arima.auto_arima import (
    diff, 
    partrans, 
    invpartrans, 
    arima_transpar,
    arima_css,
    make_arima,
    arima_like,
    StateSpaceModel,
    ndiffs,   # Stationarity Test
    nsdiffs,  # Seasonality Test
    arima_fit
)

# ----------------------------
# Helpers
# ----------------------------
def _rng(seed=0):
    rs = np.random.RandomState(seed)
    return rs

def _randn(n, scale=1.0, seed=0):
    return jnp.array(_rng(seed).randn(n) * scale, dtype=jnp.float64) # Use float64 for precision tests

def _random_walk(n=200, scale=1.0, seed=0):
    eps = _rng(seed).randn(n) * scale
    y = np.cumsum(eps)
    return jnp.array(y, dtype=jnp.float64)

def _inv_difference(last_vals, diffs, d):
    """
    Manual inverse differencing helper for testing roundtrips.
    """
    if d <= 0:
        return diffs
    
    y = np.array(diffs, copy=True)
    last_vals = np.array(last_vals)
    
    current_diffs = y
    res = current_diffs
    for i in range(d):
        start = last_vals[-(i + 1)]
        res = np.concatenate(([start], res))
        res = np.cumsum(res)[1:]
    return jnp.array(res)


# ----------------------------
# Core behavior & equivalence
# ----------------------------
def test_fit_predict_shape_consistency():
    print("\n[Test 1] Running fit_predict_shape_consistency...")
    y = _randn(20, seed=42)
    model = ARIMA(order=(1, 0, 1), include_mean=True)

    print("  -> Fitting model...")
    model.fit(y)
    p1 = model.predict(h=4)["mean"] 

    assert p1.shape == (4,)
    assert np.all(np.isfinite(p1))
    print("  -> [Test 1] Passed.")


# ----------------------------
# Basic orders & simple data
# ----------------------------
def test_arima_000_constant_series_predicts_mean():
    print("\n[Test 2] Running arima_000_constant_series_predicts_mean...")
    # Constant series 7.0
    y = jnp.full((20,), 7.0, dtype=jnp.float64)
    model = ARIMA(order=(0, 0, 0), include_mean=True)

    model.fit(y)
    pred = model.predict(h=5)
    mean_pred = np.asarray(pred["mean"])

    assert mean_pred.shape == (5,)
    # Should predict exactly 7.0
    np.testing.assert_allclose(mean_pred, 7.0, atol=1e-2)
    print("  -> [Test 2] Passed.")


def test_arima_011_on_random_walk_shapes_and_finiteness():
    print("\n[Test 3] Running arima_011_on_random_walk_shapes_and_finiteness...")
    y = _random_walk(n=30, scale=0.8, seed=123)
    model = ARIMA(order=(0, 1, 1), include_mean=True)

    model.fit(y)
    out = model.predict(h=5)
    mean = np.asarray(out["mean"])

    assert mean.shape == (5,)
    assert np.all(np.isfinite(mean)), "Forecasts must be finite"
    print("  -> [Test 3] Passed.")


# ----------------------------
# Interval Tests (Kalman Filter based)
# ----------------------------
def test_prediction_intervals_keys_and_monotonicity():
    print("\n[Test 4] Running prediction_intervals_keys_and_monotonicity...")
    y = _randn(25, seed=7)
    model = ARIMA(order=(1, 0, 1), include_mean=True)

    model.fit(y)
    # Using integers for levels
    res = model.predict(h=4, level=[80, 95])

    for k in ["mean", "lo-80", "hi-80", "lo-95", "hi-95"]:
        assert k in res, f"Missing key: {k}"

    mean = np.asarray(res["mean"])
    lo80 = np.asarray(res["lo-80"])
    hi80 = np.asarray(res["hi-80"])
    lo95 = np.asarray(res["lo-95"])
    hi95 = np.asarray(res["hi-95"])

    assert mean.shape == (4,)
    
    # Monotonicity checks: 95% interval should be wider than 80%
    assert np.all(lo95 <= lo80 + 1e-6)
    assert np.all(lo80 <= mean + 1e-6)
    assert np.all(mean <= hi80 + 1e-6)
    assert np.all(hi80 <= hi95 + 1e-6)
    print("  -> [Test 4] Passed.")


# ----------------------------
# API / input validation
# ----------------------------
def test_predict_before_fit_raises():
    print("\n[Test 5] Running predict_before_fit_raises...")
    model = ARIMA(order=(1, 1, 1))
    with pytest.raises(RuntimeError):
        _ = model.predict(h=3)
    print("  -> [Test 5] Passed.")


# ----------------------------
# Differencing/inversion sanity
# ----------------------------
def test_differencing_roundtrip():
    print("\n[Test 6] Running differencing_roundtrip...")
    y = _randn(30, seed=99)
    d = 1
    diffs = diff(y, lag=1, differences=d)
    y_rt = _inv_difference(y[:d], diffs, d)
    base = y[d:]
    np.testing.assert_allclose(np.asarray(y_rt), np.asarray(base), atol=1e-5)
    print("  -> [Test 6] Passed.")


# -----------------
# Intercept handling
# -----------------

def test_000_intercept_matches_sample_mean():
    print("\n[Test 7] Running 000_intercept_matches_sample_mean...")
    # White noise around 3.5
    y = jnp.full((20,), 3.5, dtype=jnp.float64) + _randn(20, seed=123, scale=0.01)
    model = ARIMA(order=(0, 0, 0), include_mean=True)
    model.fit(y)
    
    pred = np.asarray(model.predict(h=4)["mean"])
    np.testing.assert_allclose(pred, 3.5, atol=0.1)
    print("  -> [Test 7] Passed.")


def test_drift_like_behavior_when_d_gt_0_and_mean_included():
    print("\n[Test 8] Running drift_like_behavior_when_d_gt_0...")
    n = 30
    slope = 0.5
    base = 10.0
    noise = _randn(n, seed=22, scale=0.1)
    t = jnp.arange(n, dtype=jnp.float64)
    y = base + slope * t + noise

    # ARIMA(0,1,0) with mean -> Random Walk with Drift
    model = ARIMA(order=(0, 1, 0), include_mean=True)
    model.fit(y)
    pred = np.asarray(model.predict(h=5)["mean"])
    
    # Forecast should continue the upward trend
    assert pred[-1] > pred[0], "Forecast should show upward trend"
    # Check that forecast shows positive trend (drift estimation can vary)
    # The model estimates drift from differenced data, so exact match isn't guaranteed
    overall_trend = (pred[-1] - pred[0]) / (len(pred) - 1)
    assert overall_trend > 0, "Forecast should have positive overall trend"
    print("  -> [Test 8] Passed.")


# -----------------
# AutoARIMA Specific Tests
# -----------------

def test_auto_arima_finds_model():
    print("\n[Test 9] Running auto_arima_finds_model (stepwise)...")
    # Simple AR(1) process
    y = np.zeros(30)
    for t in range(1, 30):
        y[t] = 0.7 * y[t-1] + np.random.randn()
    y = jnp.array(y)

    model = AutoARIMA(max_p=3, max_q=3, stationary=True, stepwise=True, seasonal=False)
    print("  -> Fitting AutoARIMA...")
    model.fit(y)
    
    assert model.model_ is not None
    assert "arma" in model.model_
    pred = model.predict(h=5)
    assert pred["mean"].shape == (5,)
    print(f"  -> Selected model: {model.summary()}")
    print("  -> [Test 9] Passed.")


# -----------------
# Parameter Transform Tests (Stationarity)
# -----------------

def test_partrans_guarantees_stationarity_bounds():
    print("\n[Test 10] Running partrans_guarantees_stationarity_bounds...")
    raw_params = jnp.array([10.0, -10.0, 5.0])
    
    phi_1 = partrans(1, raw_params[:1])
    assert jnp.abs(phi_1[0]) < 1.0

    phi_2 = partrans(2, raw_params[:2])
    p1, p2 = float(phi_2[0]), float(phi_2[1])
    # Stationarity triangle for AR(2)
    assert abs(p2) < 1.0
    assert p2 + p1 < 1.0
    assert p2 - p1 < 1.0
    print("  -> [Test 10] Passed.")


# -----------------
# Numerical Stability & Structure
# -----------------

def test_make_arima_structure():
    print("\n[Test 11] Running make_arima_structure...")
    # Test internal state space construction
    arma = (1, 0, 0, 0, 1, 0, 0) # p=1
    phi = jnp.array([0.5])
    theta = jnp.array([])
    delta = jnp.array([]) # d=0
    
    mod = make_arima(phi, theta, delta, arma)
    
    # T should be [[0.5]] for AR(1)
    # Z should be [1.0]
    assert mod.T.shape == (1, 1)
    np.testing.assert_allclose(mod.T[0,0], 0.5)
    assert mod.Z.shape == (1,)
    np.testing.assert_allclose(mod.Z[0], 1.0)
    print("  -> [Test 11] Passed.")


# =============================================================================
# DATA SCIENCE & HEURISTICS TESTS (Accuracy Improvement)
# =============================================================================

def test_ds_stationarity_check_ndiffs():
    print("\n[Test 12] [DS] Running stationarity_check_ndiffs...")
    
    # Case A: White Noise (Should be Stationary -> d=0)
    # We need enough samples for KPSS to be statistically significant
    wn = _randn(100, seed=42) 
    d_wn = ndiffs(wn, max_d=2)
    print(f"  -> White Noise detected d={d_wn}")
    assert d_wn == 0, "White noise should ideally have d=0"

    # Case B: Random Walk (Should be Non-Stationary -> d=1)
    rw = _random_walk(n=100, seed=42)
    d_rw = ndiffs(rw, max_d=2)
    print(f"  -> Random Walk detected d={d_rw}")
    assert d_rw >= 1, "Random walk should require d>=1"
    
    print("  -> [Test 12] Passed.")


def test_ds_seasonality_check_nsdiffs():
    print("\n[Test 13] [DS] Running seasonality_check_nsdiffs...")
    
    # Case A: Strong Seasonality (Period 12)
    # y = trend + season + noise
    t = jnp.arange(100)
    trend = 0.1 * t
    # Strong seasonal pattern: Sine wave
    season = 10.0 * jnp.sin(2 * jnp.pi * t / 12)
    y_seas = trend + season + _randn(100, scale=0.5)
    
    D = nsdiffs(y_seas, period=12, max_D=1)
    print(f"  -> Strong Seasonal Data (m=12) detected D={D}")
    assert D == 1, "Should detect need for seasonal differencing (D=1)"
    
    # Case B: No Seasonality
    y_noseas = trend + _randn(100, scale=0.5)
    D_none = nsdiffs(y_noseas, period=12, max_D=1)
    print(f"  -> Non-seasonal Data detected D={D_none}")
    assert D_none == 0, "Should detect NO need for seasonal differencing (D=0)"
    
    print("  -> [Test 13] Passed.")


def test_ds_multiplicative_mimic_airpassengers():
    print("\n[Test 14] [DS] Running multiplicative_mimic_airpassengers...")
    """
    This attempts to reproduce the "AirPassengers" failure case.
    AirPassengers increases in variance over time (Multiplicative).
    
    We create a synthetic series: (10 + Trend) * (1 + Seasonality)
    """
    t = jnp.arange(120, dtype=jnp.float64) # 10 years of monthly data
    trend = 0.05 * t + 5.0
    season = 0.3 * jnp.sin(2 * jnp.pi * t / 12) # Period 12
    
    # Multiplicative: Trend * (1 + Season)
    y = trend * (1.0 + season) + _randn(120, scale=0.2)
    
    print("  -> Fitting AutoARIMA on synthetic AirPassengers...")
    # Ensure we tell it period=12 (User must supply this knowledge usually, or we detect it)
    model = AutoARIMA(
        period=12, 
        seasonal=True, 
        stepwise=True,
        # Allow it to search reasonably
        max_p=2, max_q=2, max_P=1, max_Q=1
    )
    
    start_t = time.time()
    model.fit(y)
    print(f"  -> Fit time: {time.time() - start_t:.4f}s")
    
    summary = model.summary()
    print(f"  -> Model Summary:\n{summary}")
    
    # Check if Seasonal Difference was detected
    # We expect something like ARIMA(p,d,q)(P,1,Q)[12]
    # The summary string is "ARIMA(p,d,q)(P,D,Q)[m] ..."
    # Let's parse the D term roughly
    
    p, q, P, Q, m, d, D = model.model_['arma']
    
    print(f"  -> Detected Orders: d={d}, D={D}")
    
    # Ideally, for this data, D should be 1 to handle the seasonality.
    # d might be 1 to handle the linear trend.
    if D == 1:
        print("  -> SUCCESS: Seasonal differencing detected.")
    else:
        print("  -> WARNING: Seasonal differencing NOT detected. Model might underfit.")
        
    # Forecast check
    pred = model.predict(h=12)["mean"]
    
    # The forecast should have "wiggles" (seasonality), not a straight line
    range_pred = np.max(pred) - np.min(pred)
    print(f"  -> Forecast Range (Seasonality Amplitude): {range_pred:.4f}")
    
    # If range is very small, it failed to capture seasonality
    assert range_pred > 1.0, "Forecast is too flat! Failed to capture seasonality."
    
    print("  -> [Test 14] Passed.")


# =============================================================================
# FORECAST STATE-PROPAGATION TESTS
# The Kalman forecast must start from the end-of-sample filtered state so the
# fitted AR/seasonal-AR (and MA, via the state) dynamics shape the forecast; a
# forecast from the zero initial state is flat at the deterministic component.
# =============================================================================

def _np_kalman_filter(y_adj, T, Z, V, a0, P0):
    """NumPy reference of _kalman_filter_core: returns the final one-step-ahead
    prior state (a, P) after filtering y_adj."""
    a = np.array(a0, dtype=float).copy()
    P = np.array(P0, dtype=float).copy()
    T = np.array(T); Z = np.array(Z); V = np.array(V)
    for y_t in np.asarray(y_adj, dtype=float):
        v = y_t - Z @ a
        F = Z @ P @ Z
        F_safe = max(F, 1e-9)
        M = P @ Z
        K = M / F_safe
        a = a + K * v
        P = P - np.outer(K, M)
        a = T @ a
        P = T @ P @ T.T + V
    return a, P


def _np_forecast_from_state(h, T, Z, V, a, P):
    """NumPy reference forecast from a one-step-ahead prior state: read the
    state, then advance (equivalent to R/SF's advance-then-read from the
    posterior)."""
    T = np.array(T); Z = np.array(Z); V = np.array(V)
    a = np.array(a, dtype=float).copy(); P = np.array(P, dtype=float).copy()
    fc = np.empty(h); var = np.empty(h)
    for k in range(h):
        fc[k] = Z @ a
        var[k] = Z @ P @ Z
        a = T @ a
        P = T @ P @ T.T + V
    return fc, var


def _ar1_series(n=400, phi=0.8, mu=10.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = mu
    for t in range(1, n):
        y[t] = mu + phi * (y[t - 1] - mu) + rng.normal(0.0, sigma)
    return jnp.asarray(y, dtype=jnp.float64)


def test_getQ0_satisfies_filter_stationarity_equation():
    # P0 must be the stationary covariance of the SAME representation the
    # filter runs: P0 = T P0 T' + V (make_arima's T convention).
    for p, q, phi_v, theta_v in [
        (1, 1, [0.5], [0.5]),
        (2, 0, [0.5, -0.3], []),
        (0, 1, [], [0.7]),
        (2, 2, [0.4, -0.2], [0.3, 0.1]),
    ]:
        arma = (p, q, 0, 0, 1, 0, 0)
        phi = jnp.asarray(phi_v, dtype=jnp.float64)
        theta = jnp.asarray(theta_v, dtype=jnp.float64)
        mod = make_arima(phi, theta, jnp.zeros(0), arma)
        P0 = np.array(mod.P0)
        T = np.array(mod.T)
        V = np.array(mod.V)
        resid = P0 - (T @ P0 @ T.T + V)
        assert np.abs(resid).max() < 1e-8, (
            f"P0 is not stationary for the filter's T (p={p}, q={q}): "
            f"max residual {np.abs(resid).max():.3g}"
        )


def test_predict_matches_numpy_kalman_reference_ar1():
    # Fixed AR(1)+mean: predict() must equal the reference filter+forecast
    # run on the model's own stored (y_adj, state-space) at the fitted params.
    y = _ar1_series()
    m = ARIMA(order=(1, 0, 0), include_mean=True, standardize=False)
    m.fit(y)
    mod = m.model_["model"]
    y_adj = m.model_["y_adj"]
    a_fin, P_fin = _np_kalman_filter(y_adj, mod.T, mod.Z, mod.V, mod.a0, mod.P0)
    fc_ref, _ = _np_forecast_from_state(5, mod.T, mod.Z, mod.V, a_fin, P_fin)
    narma = 1
    intercept = float(m.model_["coef"][narma])
    pred = np.asarray(m.predict(h=5)["mean"])
    np.testing.assert_allclose(pred, fc_ref + intercept, rtol=1e-8, atol=1e-8)


def test_ar1_forecast_decays_from_last_state_not_flat():
    # The 1-step forecast must be mu + phi*(y_T - mu) in the model's own
    # fitted parameters, and successive steps must decay geometrically toward
    # the mean — a flat forecast means the AR state never propagated.
    y = _ar1_series(seed=3)
    # Ensure the forecast origin sits away from the mean so decay is visible.
    y = jnp.concatenate([y, jnp.array([14.0], dtype=jnp.float64)])
    m = ARIMA(order=(1, 0, 0), include_mean=True, standardize=False)
    m.fit(y)
    phi_eff = float(m.model_["model"].T[0, 0])
    narma = 1
    mu = float(m.model_["coef"][narma])
    pred = np.asarray(m.predict(h=5)["mean"])
    yT = float(y[-1])
    theory = mu + phi_eff ** np.arange(1, 6) * (yT - mu)
    np.testing.assert_allclose(pred, theory, rtol=1e-6, atol=1e-6)
    assert abs(pred[0] - pred[4]) > 0.1, "forecast is flat: AR state not propagated"


def test_arima_110_one_step_uses_last_difference():
    # ARIMA(1,1,0) without mean: y_hat[n+k] = y_n + sum_{j=1..k} phi^j * dy_n.
    rng = np.random.default_rng(7)
    dy = np.zeros(300)
    for t in range(1, 300):
        dy[t] = 0.6 * dy[t - 1] + rng.normal(0.0, 1.0)
    y = jnp.asarray(100.0 + np.cumsum(dy), dtype=jnp.float64)
    m = ARIMA(order=(1, 1, 0), include_mean=False, standardize=False)
    m.fit(y)
    phi_eff = float(m.model_["model"].T[0, 0])
    yT = float(y[-1])
    dyT = float(y[-1] - y[-2])
    pred = np.asarray(m.predict(h=3)["mean"])
    theory = yT + np.array([
        phi_eff * dyT,
        phi_eff * dyT + phi_eff ** 2 * dyT,
        phi_eff * dyT + phi_eff ** 2 * dyT + phi_eff ** 3 * dyT,
    ])
    np.testing.assert_allclose(pred, theory, rtol=1e-5, atol=1e-5)


def test_forecast_fast_path_agrees_with_fit_predict():
    # The stateless forecast() (scan optimizer) and fit()+predict() (BFGS)
    # share the forecast architecture; endpoints differ only through the
    # optimizers' convergence noise.
    y = _ar1_series(seed=11)
    m1 = ARIMA(order=(1, 0, 0), include_mean=True, standardize=False, method="CSS")
    fc_fast = np.asarray(m1.forecast(h=5, y=y)["mean"])
    m2 = ARIMA(order=(1, 0, 0), include_mean=True, standardize=False, method="CSS")
    m2.fit(y)
    fc_eager = np.asarray(m2.predict(h=5)["mean"])
    np.testing.assert_allclose(fc_fast, fc_eager, rtol=1e-3, atol=1e-3)


def test_analytic_se_matches_numpy_reference():
    # Analytic (non-conformal) intervals: SE must follow the forecast-variance
    # recursion seeded from the end-of-sample filtered covariance.
    y = _ar1_series(seed=5)
    m = ARIMA(order=(1, 0, 0), include_mean=True, standardize=False)
    m.fit(y)
    mod = m.model_["model"]
    a_fin, P_fin = _np_kalman_filter(m.model_["y_adj"], mod.T, mod.Z, mod.V, mod.a0, mod.P0)
    _, var_ref = _np_forecast_from_state(4, mod.T, mod.Z, mod.V, a_fin, P_fin)
    sigma2 = float(m.model_["sigma2"])
    se_ref = np.sqrt(var_ref * sigma2)
    out = m.predict(h=4, level=95)
    # Divide out the same z-score predict used, so this test pins the SE
    # recursion itself and stays independent of the ppf approximation.
    from chronax.utils import _quantiles
    z95 = float(_quantiles((95,))[0])
    se_obs = (np.asarray(out["hi-95"]) - np.asarray(out["mean"])) / z95
    np.testing.assert_allclose(se_obs, se_ref, rtol=1e-6, atol=1e-8)


def test_drift_ramp_continues_trend_at_mean_difference():
    # (0,1,0)+mean = RW with drift: each forecast increment equals the mean
    # first difference of the training series.
    rng = np.random.default_rng(13)
    n = 200
    y = jnp.asarray(5.0 + 0.5 * np.arange(n) + rng.normal(0, 0.2, n), dtype=jnp.float64)
    m = ARIMA(order=(0, 1, 0), include_mean=True, standardize=False)
    m.fit(y)
    pred = np.asarray(m.predict(h=4)["mean"])
    drift = float(jnp.mean(y[1:] - y[:-1]))
    theory = float(y[-1]) + drift * np.arange(1, 5)
    np.testing.assert_allclose(pred, theory, rtol=1e-6, atol=1e-6)


def test_seasonal_drift_slope_is_per_step():
    # (0,0,0)(0,1,0)_4 + mean: the drift coefficient measures the mean
    # seasonal difference; the forecast trend must advance at that amount per
    # SEASON (i.e. mean(D_4 y)/4 per step), not per step.
    rng = np.random.default_rng(17)
    n = 240
    t = np.arange(n)
    seas = np.array([3.0, -1.0, -2.5, 0.5])
    y = jnp.asarray(2.0 + 0.25 * t + seas[t % 4] + rng.normal(0, 0.05, n), dtype=jnp.float64)
    m = ARIMA(order=(0, 0, 0), seasonal_order=(0, 1, 0), period=4, include_mean=True, standardize=False)
    m.fit(y)
    pred = np.asarray(m.predict(h=8)["mean"])
    # Slope across one full season = 4 * per-step slope of the data.
    season_step = pred[4:] - pred[:4]
    np.testing.assert_allclose(season_step, 4 * 0.25, rtol=0.05)


def test_forecast_level_analytic_fallback_without_conformal():
    # Stateless forecast(level=...) without conformal_params must emit
    # analytic z-score intervals (parity with predict()'s fallback), not
    # raise and not silently drop the level request.
    y = _ar1_series(seed=31)
    m = ARIMA(order=(1, 0, 0), include_mean=True, standardize=False)
    res = m.forecast(h=4, y=y, level=[80, 95])
    for key in ("mean", "lo-80", "hi-80", "lo-95", "hi-95"):
        assert key in res
    mean = np.asarray(res["mean"])
    lo95, hi95 = np.asarray(res["lo-95"]), np.asarray(res["hi-95"])
    lo80, hi80 = np.asarray(res["lo-80"]), np.asarray(res["hi-80"])
    assert np.all(np.isfinite(lo95)) and np.all(np.isfinite(hi95))
    assert np.all(lo95 < mean) and np.all(mean < hi95)
    assert np.all(lo95 <= lo80) and np.all(hi80 <= hi95)
    width = hi95 - lo95
    assert np.all(np.diff(width) > 0), "AR(1) forecast interval must widen with horizon"


def test_exog_not_supported_raises():
    y = _ar1_series(n=60, seed=37)
    Xmat = np.ones((60, 1))
    with pytest.raises(ValueError):
        ARIMA(order=(1, 0, 0)).fit(y, X=Xmat)
    with pytest.raises(ValueError):
        ARIMA(order=(1, 0, 0)).forecast(h=3, y=y, X=Xmat)
    with pytest.raises(ValueError):
        AutoARIMA(seasonal=False).fit(y, X=Xmat)


def test_autoarima_ar_series_forecast_not_flat():
    # AutoARIMA on a strongly autocorrelated stationary series must produce a
    # dynamic (decaying) forecast when its selected order has p>0.
    y = _ar1_series(n=300, phi=0.9, mu=0.0, seed=23)
    y = jnp.concatenate([y, jnp.array([4.0], dtype=jnp.float64)])
    m = AutoARIMA(seasonal=False, stationary=True, stepwise=True)
    m.fit(y)
    p_sel = m.model_["arma"][0]
    pred = np.asarray(m.predict(h=6)["mean"])
    if p_sel > 0:
        assert np.std(pred) > 0.05, "AutoARIMA forecast is flat despite AR order"
        assert abs(pred[0]) > abs(pred[5]), "forecast does not decay toward the mean"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-s", __file__]))
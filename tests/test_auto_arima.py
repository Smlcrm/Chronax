# test_auto_arima.py
import math
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import pytest
import time
from typing import Optional

# --- Mock BaseForecaster for standalone testing if not present ---
try:
    from base_forecaster import BaseForecaster
except ImportError:
    class BaseForecaster:
        def fit(self, y, X=None): raise NotImplementedError
        def predict(self, h, X=None, level=None): raise NotImplementedError

# =============================================================================
# IMPORTS (Separated as requested)
# =============================================================================

# 1. High-Level Wrappers (Classes)
from auto_arima_wrap import (
    ARIMA, 
    AutoARIMA
)

# 2. Low-Level Functions (Math & Logic)
from auto_arima_functions import (
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
    assert pred[-1] > pred[0]
    # Slope should be roughly 0.5
    estimated_slope = pred[-1] - pred[-2]
    np.testing.assert_allclose(estimated_slope, 0.5, atol=0.2)
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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-s", __file__]))
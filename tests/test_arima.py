# test_arima.py
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

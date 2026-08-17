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
    arima_fit,
    _kalman_filter_core,
    _kalman_filter_steadystate,
    _objective_ml,
    _objective_ml_ss,
    _use_steadystate,
    _unpack_and_adjust_jit,
    _STEADYSTATE_BURNIN,
    _STEADYSTATE_MIN_N,
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


# =============================================================================
# Steady-state Kalman (Rung 2) — large-n eager ML path
# =============================================================================
def _ss_long_series(n=3000, seed=7):
    """A long non-seasonal ARIMA(1,1,1)-like series (n > _STEADYSTATE_MIN_N)."""
    rs = np.random.RandomState(seed)
    e = rs.randn(n + 1)
    d = np.zeros(n)                      # ARMA(1,1) on the differences
    for t in range(1, n):
        d[t] = 0.6 * d[t - 1] + e[t] + 0.3 * e[t - 1]
    return np.cumsum(d) * 5.0 + 100.0    # integrate -> d=1


def _ss_fitted_mod(n=3000):
    """Fit ARIMA(1,1,1) on a long series; return (mod, y_adj) for the filter referee."""
    y = jnp.asarray(_ss_long_series(n), dtype=jnp.float64)
    m = ARIMA(order=(1, 1, 1), method="CSS-ML")
    m.fit(y)
    md = m.model_
    params, arma, delta = md["coef"], md["arma"], md["delta"]
    narma = sum(arma[:4]); ncxreg = params.shape[0] - narma; n_exog = ncxreg - 1
    y_adj, phi, theta, _ = _unpack_and_adjust_jit(params, y, None, arma, ncxreg, n_exog, True)
    return make_arima(phi, theta, delta, arma), y_adj, (params, arma, delta, ncxreg, n_exog)


def _nll(ssq, sumlog, nu):
    snu = max(float(nu), 1.0); sq = max(float(ssq), 1e-8)
    return 0.5 * (snu * float(jnp.log(sq / snu)) + float(sumlog))


def test_steadystate_matches_full_filter():
    """The frozen-gain filter matches the full filter's NLL + end state to ~machine precision
    (the gain converges far inside the burn-in). Gate on the NLL the optimizer consumes."""
    mod, y_adj, _ = _ss_fitted_mod()
    (a1, _, ssq1, sl1, nu1), _, _ = _kalman_filter_core(y_adj, mod)
    a2, _, ssq2, sl2, nu2 = _kalman_filter_steadystate(y_adj, mod, _STEADYSTATE_BURNIN)
    nll1, nll2 = _nll(ssq1, sl1, nu1), _nll(ssq2, sl2, nu2)
    assert abs(nll2 - nll1) / abs(nll1) < 1e-9            # objective value (the gate)
    assert jnp.allclose(a2, a1, rtol=1e-9, atol=1e-8)    # end state (feeds the forecast)
    assert int(nu2) == int(nu1)


def test_objective_ml_ss_grad_matches_full():
    """Reverse-differentiability gate: value_and_grad of the steady-state objective RUNS
    (no while_loop) and matches the full objective at a realistic AND a displaced point."""
    import jax
    _, y_adj, (params, arma, delta, ncxreg, n_exog) = _ss_fitted_mod()
    full = lambda q: _objective_ml(q, y_adj, None, delta, arma, ncxreg, n_exog, True)
    ss = lambda q: _objective_ml_ss(q, y_adj, None, delta, arma, ncxreg, n_exog, True)
    narma = sum(arma[:4])
    for p in [params, params.at[:narma].add(0.1)]:
        vf, gf = jax.value_and_grad(full)(p)
        vs, gs = jax.value_and_grad(ss)(p)
        assert abs(float(vs) - float(vf)) / abs(float(vf)) < 1e-9
        assert jnp.allclose(gs, gf, rtol=1e-6, atol=1e-7)


def test_use_steadystate_gate():
    """Routing predicate: long, non-seasonal, d+D<=1 only. Short/seasonal/over-differenced fall back."""
    ns = (1, 1, 0, 0, 1, 1, 0)                 # non-seasonal ARIMA(1,1,1), d=1
    assert _use_steadystate(_STEADYSTATE_MIN_N + 1, ns) is True
    assert _use_steadystate(_STEADYSTATE_MIN_N - 1, ns) is False       # short series
    assert _use_steadystate(9999, (1, 1, 0, 2, 52, 1, 0)) is False     # seasonal MA (Q>0)
    assert _use_steadystate(9999, (1, 1, 1, 0, 12, 1, 0)) is False     # seasonal AR (P>0)
    assert _use_steadystate(9999, (1, 1, 0, 0, 12, 1, 1)) is False     # seasonal diff (D>0)
    assert _use_steadystate(9999, (0, 1, 0, 0, 1, 2, 0)) is False      # over-differenced (d=2)


def test_forecast_largen_is_finite_and_deterministic():
    """forecast() on a long series (steady-state branch) is finite and repeatable."""
    y = jnp.asarray(_ss_long_series(3000), dtype=jnp.float64)
    m = ARIMA(order=(1, 1, 1), method="CSS-ML")
    f1 = np.asarray(m.forecast(h=24, y=y)["mean"])
    f2 = np.asarray(m.forecast(h=24, y=y)["mean"])
    assert np.all(np.isfinite(f1))
    assert np.array_equal(f1, f2)


def test_overdifferenced_largen_forecast_uses_full_filter():
    """Over-differenced (d>=2) long series are NOT routed to steady-state — their MA roots reach
    the invertibility boundary where the frozen gain never converges — so they forecast through
    the exact full filter. (Regression: the pre-fix routing lacked the d+D<=1 guard.)"""
    y = jnp.asarray(_ss_long_series(3000), dtype=jnp.float64)
    assert _use_steadystate(len(y), (0, 1, 0, 0, 1, 2, 0)) is False   # d=2 excluded by construction
    fc = np.asarray(ARIMA(order=(0, 2, 1), method="CSS-ML").forecast(h=24, y=y)["mean"])
    assert np.all(np.isfinite(fc))


def test_arima_fit_steadystate_flag(monkeypatch):
    """arima_fit(steadystate=False) must NOT enter the frozen-gain path (AutoARIMA's search
    needs the exact objective); steadystate=True on a long non-seasonal fit must. Monkeypatch
    the SS filter to raise so the guard is proven live, not vacuous. jax.clear_caches() before
    the True arm defeats the jit-cache confound (a cached _fit_model_bfgs would never re-trace
    the patched fn)."""
    import jax
    import chronax.models.arima.auto_arima as aa
    y = jnp.asarray(_ss_long_series(3000), dtype=jnp.float64)

    def _boom(*a, **k):
        raise RuntimeError("steady-state entered")
    monkeypatch.setattr(aa, "_kalman_filter_steadystate", _boom)

    jax.clear_caches()
    r = aa.arima_fit(y, order=(1, 1, 1), method="ML", steadystate=False)  # full filter, no SS
    assert bool(r["success"])
    jax.clear_caches()
    with pytest.raises(RuntimeError):                                     # SS live under the flag
        aa.arima_fit(y, order=(1, 1, 1), method="ML", steadystate=True)

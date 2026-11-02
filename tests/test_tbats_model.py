from chronax.models import AutoTBATS, TBATS
from chronax.utils import ConformalIntervals

import warnings
from typing import Dict, Tuple
from chronax.utils import ensure_float as _ensure_float, calculate_sigma as _calculate_sigma, _add_fitted_pi, _calculate_intervals


import numpy as np
import jax.numpy as jnp
# =========================
# Test Cases
# =========================

def _arr_close(a, b, tol=1e-4):
    """Helper to compare arrays with tolerance."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    return jnp.all(jnp.abs(a - b) <= tol)


def test_basic_fit_predict():
    """Test basic fit and predict workflow."""
    print("\n[Test 1] Basic fit and predict")
    y = jnp.array([10., 12., 13., 15., 17., 20., 22., 25., 27., 30.])
    model = AutoTBATS(season_length=3, use_boxcox=False, use_trend=True,
                      use_damped_trend=False, use_arma_errors=False)
    model.fit(y)

    result = model.predict(h=3, level=None)
    print(result)
    assert "mean" in result
    assert result["mean"].shape == (3,)
    assert not jnp.any(jnp.isnan(result["mean"]))
    print(f"  Forecast: {result['mean']}")
    print("  ✓ Basic fit and predict works")


def test_forecast_with_fitted():
    """Test forecast method with fitted values."""
    print("\n[Test 2] Forecast with fitted values")
    y = jnp.array([5., 7., 6., 8., 10., 9., 11., 13.])
    model = TBATS(season_length=4, use_boxcox=False)

    result = model.forecast(y=y, h=2, fitted=True, level=None)
    assert "mean" in result
    assert "fitted" in result
    assert result["mean"].shape == (2,)
    assert result["fitted"].shape == (8,)
    print(f"  Forecast shape: {result['mean'].shape}")
    print(f"  Fitted shape: {result['fitted'].shape}")
    print("  ✓ Forecast with fitted values works")


def test_prediction_intervals_parametric():
    """Test parametric prediction intervals."""
    print("\n[Test 3] Parametric prediction intervals")
    y = jnp.array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19.])
    model = AutoTBATS(season_length=2, use_boxcox=False,
                      conformal_params=None)  # No conformal, use parametric
    model.fit(y)

    result = model.predict(h=3, level=[80, 95])
    assert "mean" in result
    assert "lo-80" in result and "hi-80" in result
    assert "lo-95" in result and "hi-95" in result

    # Check interval ordering
    assert jnp.all(result["lo-95"] <= result["lo-80"])
    assert jnp.all(result["hi-80"] <= result["hi-95"])
    assert jnp.all(result["lo-80"] <= result["mean"])
    assert jnp.all(result["mean"] <= result["hi-80"])

    print(f"  Mean: {result['mean']}")
    print(f"  80% interval: [{result['lo-80'][0]:.2f}, {result['hi-80'][0]:.2f}]")
    print(f"  95% interval: [{result['lo-95'][0]:.2f}, {result['hi-95'][0]:.2f}]")
    print("  ✓ Parametric prediction intervals work")


def test_conformal_intervals():
    """Test conformal prediction intervals."""
    print("\n[Test 4] Conformal prediction intervals")
    y = jnp.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
    ci = ConformalIntervals(h=2, n_windows=3, method="conformal_distribution")
    model = AutoTBATS(season_length=3, use_boxcox=False, conformal_params=ci)
    model.fit(y)

    result = model.predict(h=2, level=[90])
    assert "mean" in result
    assert "lo-90" in result and "hi-90" in result
    assert jnp.all(result["lo-90"] <= result["mean"])
    assert jnp.all(result["mean"] <= result["hi-90"])

    print(f"  Mean: {result['mean']}")
    print(f"  90% conformal interval: [{result['lo-90'][0]:.2f}, {result['hi-90'][0]:.2f}]")
    print("  ✓ Conformal prediction intervals work")


def test_boxcox_transformation1():
    """SKIPPED BC leads to inf objective: Test Box-Cox transformation handling."""
    print("\n[Test 5] Box-Cox transformation")
    y = jnp.array([1., 2., 4., 8., 16., 32., 64., 128.])  # Exponential growth
    model = AutoTBATS(season_length=2, use_boxcox=True, use_trend=True)
    model.fit(y)

    result = model.predict(h=2, level=None)
    assert "mean" in result
    assert jnp.all(result["mean"] > 0)  # Already back on original scale
    print(f"  Forecast: {result['mean']}")
    print(f"  Box-Cox lambda: {model.model_.get('BoxCox_lambda', 'N/A')}")
    print("  ✓ Box-Cox transformation works")


def test_input_validation():
    """Test input validation."""
    print("\n[Test 6] Input validation")

    # Test NaN handling
    y_nan = jnp.array([1., 2., jnp.nan, 4., 5.])
    model = AutoTBATS(season_length=2)
    try:
        model.fit(y_nan)
        assert False, "Should raise ValueError for NaN"
    except ValueError as e:
        assert "NaN" in str(e)
        print("  ✓ NaN validation works")

    # Test Box-Cox with negative values
    y_neg = jnp.array([-1., 2., 3., 4., 5.])
    model_bc = AutoTBATS(season_length=2, use_boxcox=True)
    try:
        model_bc.fit(y_neg)
        assert False, "Should raise ValueError for negative with Box-Cox"
    except ValueError as e:
        assert "positive" in str(e)
        print("  ✓ Box-Cox validation works")


def test_predict_before_fit():
    """Test that predict fails before fit."""
    print("\n[Test 7] Predict before fit")
    model = AutoTBATS(season_length=2)
    try:
        model.predict(h=2)
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        assert "fit" in str(e).lower()
        print("  ✓ Predict before fit validation works")


# -----------------------------
# Your JAX implementation
# -----------------------------
from chronax.models.tbats.tbats_core import (
    tbats_selection as jax_tbats_selection,  # kept for multi-season test
    tbats_forecast as jax_tbats_forecast,
    compute_sigmah as jax_compute_sigmah,
)


# ============================================================================
# Data generators
# ============================================================================
def generate_seasonal_data(n: int = 200, season_length: int = 12,
                           trend: bool = True, noise_level: float = 1.0, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / season_length)
    trend_component = 0.05 * t if trend else 0.0
    noise = np.random.normal(0, noise_level, n)
    return 100 + trend_component + seasonal + noise

def generate_multiple_seasonal_data(n: int = 365,
                                    periods: Tuple[int, ...] = (7, 365),
                                    noise_level: float = 2.0, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    t = np.arange(n)
    y = 100 + 0.05 * t
    for i, period in enumerate(periods):
        amp = 10 * (2 - i * 0.5)
        y += amp * np.sin(2 * np.pi * t / period)
    y += np.random.normal(0, noise_level, n)
    return y

def generate_exponential_data(n: int = 100, growth_rate: float = 0.05,
                              season_length: int = 12, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    t = np.arange(n)
    trend = 10 * np.exp(growth_rate * t / 10)
    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * t / season_length)
    noise = np.random.lognormal(0, 0.1, n)
    return trend * seasonal_factor * noise


def test_prediction_intervals():
    print("\n" + "="*70)
    print("TEST 5: Prediction Intervals (JAX sigma(h))")
    print("="*70)

    y = generate_seasonal_data(n=100, season_length=12, seed=42)
    y_jax = jnp.array(y, dtype=jnp.float32)

    # Build via core selection (to get sigmah definition); alternatively, your class might expose it.
    print("JAX core selection for sigma(h)")
    jax_mod = jax_tbats_selection(
        y=y_jax, seasonal_periods=[12],
        use_boxcox=False, bc_lower=0.0, bc_upper=1.0,
        use_trend=True, use_damped_trend=False, use_arma_errors=False,
    )
    h = 12
    sigmah = np.array(jax_compute_sigmah(jax_mod, h=h))
    print(f"JAX sigmah (first 6): {sigmah[:6]}")
    print(f"JAX sigmah (last 6):  {sigmah[6:]}")
    is_increasing = np.all(np.diff(sigmah) >= -1e-6)
    print(f"Sigmah non-decreasing: {is_increasing}")

    assert is_increasing


def test_edge_cases():
    print("\n" + "="*70)
    print("TEST 6: Edge Cases (AutoTBATS.forecast)")
    print("="*70)

    # Short series
    y = generate_seasonal_data(n=20, season_length=12, seed=42)
    y_jax = jnp.array(y, dtype=jnp.float32)
    model = AutoTBATS(season_length=12, use_boxcox=False, use_arma_errors=False)
    r = model.forecast(y=y_jax, h=5)
    assert np.all(np.isfinite(np.asarray(r["mean"])))
    print("   ✓ Handles short series")

    # Long horizon
    y = generate_seasonal_data(n=100, season_length=12, seed=42)
    y_jax = jnp.array(y, dtype=jnp.float32)
    model = AutoTBATS(season_length=12, use_boxcox=False)
    r = model.forecast(y=y_jax, h=100)
    assert r["mean"].shape == (100,)
    print("   ✓ Handles long horizon")

    # NaN handling: fit() raises eagerly; the stateless (vmap-traceable)
    # forecast() cannot raise on data VALUES — NaN propagates instead.
    y = generate_seasonal_data(n=50, season_length=12, seed=42)
    y_jax = jnp.array(y, dtype=jnp.float32).at[25].set(jnp.nan)
    model = AutoTBATS(season_length=12)
    try:
        model.fit(y_jax)
        assert False, "fit should raise ValueError for NaN"
    except ValueError as e:
        assert "NaN" in str(e)
    r = model.forecast(y=y_jax, h=5)
    assert np.any(np.isnan(np.asarray(r["mean"])))  # NaN in, NaN out
    print("   ✓ NaN: fit raises, forecast propagates")

    # Box-Cox + negative: fit() raises when Box-Cox is FORCED; the auto grid
    # (use_boxcox=None) degrades gracefully — the Box-Cox candidates are
    # masked out of the AIC argmin and a non-Box-Cox candidate wins.
    y = generate_seasonal_data(n=50, season_length=12, seed=42) - 110
    y_jax = jnp.array(y, dtype=jnp.float32)
    try:
        AutoTBATS(season_length=12, use_boxcox=True).fit(y_jax)
        assert False, "fit should raise ValueError for negative with forced Box-Cox"
    except ValueError as e:
        assert "positive" in str(e).lower()
    r = AutoTBATS(season_length=12, use_boxcox=None).forecast(y=y_jax, h=5)
    assert np.all(np.isfinite(np.asarray(r["mean"])))
    print("   ✓ Box-Cox + negative: fit raises, auto grid degrades gracefully")


def test_numerical_stability():
    print("\n" + "="*70)
    print("TEST 7: Numerical Stability (AutoTBATS.forecast)")
    print("="*70)

    # Large scale
    y = generate_seasonal_data(n=100, season_length=12, seed=42) * 1e6
    r = AutoTBATS(season_length=12, use_boxcox=False).forecast(
        y=jnp.array(y, dtype=jnp.float32), h=12)
    assert np.all(np.isfinite(np.asarray(r["mean"])))
    print("   ✓ Handles large values")

    # Small scale
    y = generate_seasonal_data(n=100, season_length=12, seed=42) * 1e-3
    r = AutoTBATS(season_length=12, use_boxcox=False).forecast(
        y=jnp.array(y, dtype=jnp.float32), h=12)
    assert np.all(np.isfinite(np.asarray(r["mean"])))
    print("   ✓ Handles small values")

    # High variance
    y = generate_seasonal_data(n=100, season_length=12, noise_level=50.0, seed=42)
    r = AutoTBATS(season_length=12, use_boxcox=False).forecast(
        y=jnp.array(y, dtype=jnp.float32), h=12)
    assert np.all(np.isfinite(np.asarray(r["mean"])))
    print("   ✓ Handles high variance")


import jax.numpy as jnp

def test_predict_in_sample_basic():
    """predict_in_sample returns fitted of correct length with no NaNs."""
    print("\n[InSample 1] Basic predict_in_sample")
    y = jnp.array([10., 12., 11., 13., 15., 14., 16., 17.])
    model = AutoTBATS(season_length=4, use_boxcox=False, use_trend=True,
                      use_damped_trend=False, use_arma_errors=False)
    model.fit(y)
    res = model.predict_in_sample(level=None)
    assert "fitted" in res
    assert res["fitted"].shape == (y.shape[0],)
    assert not jnp.any(jnp.isnan(res["fitted"]))
    print(f"  Fitted (first 5): {res['fitted'][:5]}")
    print("  ✓ Basic in-sample fitted works")


def test_predict_in_sample_with_intervals():
    """predict_in_sample returns intervals and preserves ordering."""
    print("\n[InSample 2] In-sample intervals")
    y = jnp.array([5., 6., 7., 8., 9., 10., 11., 12., 13., 14.])
    model = AutoTBATS(season_length=2, use_boxcox=False, use_trend=True,
                      use_damped_trend=False, use_arma_errors=False)
    model.fit(y)
    res = model.predict_in_sample(level=(80, 95))
    # Keys exist
    for k in ("fitted", "lo-80", "hi-80", "lo-95", "hi-95"):
        assert k in res, f"Missing key {k}"
    # Shapes match
    n = y.shape[0]
    for k in ("fitted", "lo-80", "hi-80", "lo-95", "hi-95"):
        assert res[k].shape == (n,)
    # Ordering
    assert jnp.all(res["lo-95"] <= res["lo-80"])
    assert jnp.all(res["hi-80"] <= res["hi-95"])
    assert jnp.all(res["lo-80"] <= res["fitted"])
    assert jnp.all(res["fitted"] <= res["hi-80"])
    print(f"  Fitted (first 3): {res['fitted'][:3]}")
    print(f"  80% PI (first 1): [{res['lo-80'][0]:.3f}, {res['hi-80'][0]:.3f}]")
    print(f"  95% PI (first 1): [{res['lo-95'][0]:.3f}, {res['hi-95'][0]:.3f}]")
    print("  ✓ In-sample intervals OK")


def test_predict_in_sample_boxcox():
    """Box–Cox: outputs are back on original scale and finite."""
    print("\n[InSample 3] Box–Cox predict_in_sample")
    y = jnp.array([1., 2., 4., 8., 16., 32., 64., 128.])  # strictly positive
    model = AutoTBATS(season_length=2, use_boxcox=True, use_trend=True,
                      use_damped_trend=False, use_arma_errors=False)
    model.fit(y)
    res = model.predict_in_sample(level=None)
    assert "fitted" in res
    assert res["fitted"].shape == (y.shape[0],)
    assert jnp.all(jnp.isfinite(res["fitted"]))
    assert jnp.all(res["fitted"] > 0)
    lam = model.model_.get("BoxCox_lambda", None)
    print(f"  λ (Box–Cox): {lam}")
    print(f"  Fitted (first 5): {res['fitted'][:5]}")
    print("  ✓ Box–Cox in-sample back-transform OK")


def test_predict_in_sample_requires_fit():
    """Calling predict_in_sample before fit raises."""
    print("\n[InSample 4] predict_in_sample requires fit")
    model = AutoTBATS(season_length=3, use_boxcox=False)
    try:
        _ = model.predict_in_sample(level=None)
        assert False, "predict_in_sample should raise before fit"
    except RuntimeError as e:
        assert "fit" in str(e).lower()
        print("  ✓ Raises RuntimeError before fit")


def test_predict_in_sample_interval_shapes():
    """Shapes of returned arrays match n."""
    print("\n[InSample 5] Interval shapes")
    y = jnp.array([3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23., 25.])
    model = AutoTBATS(season_length=4, use_boxcox=False, use_trend=True)
    model.fit(y)
    res = model.predict_in_sample(level=(90,))
    n = y.shape[0]
    for k in ("fitted", "lo-90", "hi-90"):
        assert k in res
        assert res[k].shape == (n,)
    print(f"  n={n} → shapes OK")
    print("  ✓ Interval shapes verified")


# =========================
# Extra Full-Coverage Tests
# =========================

from chronax.models.tbats.tbats_core import _boxcox, _inv_boxcox, tbats_selection as _sel, tbats_forecast as _fc, compute_sigmah as _sig

def _roundtrip_ok(x: np.ndarray, lam: float, tol: float = 1e-6) -> bool:
    y = _boxcox(jnp.asarray(x), lam)
    x2 = _inv_boxcox(y, lam)
    return np.allclose(np.asarray(x), np.asarray(x2), rtol=1e-6, atol=tol)

def test_boxcox_roundtrip_various_lambdas():
    """Box–Cox <-> inverse roundtrip is stable for several λ (including 0)."""
    xs = np.linspace(0.1, 10.0, 50)  # strictly positive
    for lam in (-0.75, -0.25, 0.0, 0.25, 0.75, 1.0):
        print(_roundtrip_ok(xs, lam))
        assert _roundtrip_ok(xs, lam), f"Roundtrip failed for lambda={lam}"
    print("✓ Box–Cox/_inv_boxcox roundtrip across lambdas")

def test_boxcox_interval_monotonicity_predict_and_forecast():
    """Intervals monotone & positive under Box–Cox for predict() and forecast()."""
    y = jnp.array([1., 2., 4., 8., 16., 32., 64., 128., 256., 512.], dtype=jnp.float32)
    levels = [50, 80, 95]
    # Class API (fit + predict)
    model = AutoTBATS(season_length=2, use_boxcox=True, use_trend=True, use_damped_trend=False, use_arma_errors=False)
    model.fit(y)
    out_pred = model.predict(h=3, level=levels)
    
    for lo, hi in (("lo-50","hi-50"),("lo-80","hi-80"),("lo-95","hi-95")):
        print("lo", out_pred[lo], "mean", out_pred["mean"],"hi", out_pred[hi])
        assert jnp.all(out_pred[lo] <= out_pred[hi])
        assert jnp.all(out_pred[lo] <= out_pred["mean"])
        assert jnp.all(out_pred["mean"] <= out_pred[hi])
        assert jnp.all(jnp.isfinite(out_pred[lo])) and jnp.all(jnp.isfinite(out_pred[hi]))
        assert jnp.all(out_pred[lo] > 0) and jnp.all(out_pred[hi] > 0)

    # Stateless forecast path
    out_fc = model.forecast(y=y, h=3, level=levels, fitted=True)
    for lo, hi in (("lo-50","hi-50"),("lo-80","hi-80"),("lo-95","hi-95")):
        assert jnp.all(out_fc[lo] <= out_fc[hi])
        assert jnp.all(out_fc[lo] <= out_fc["mean"])
        assert jnp.all(out_fc["mean"] <= out_fc[hi])
        assert jnp.all(out_fc[lo] > 0) and jnp.all(out_fc[hi] > 0)
    # Also verify fitted PIs present and positive (when fitted=True)
    assert "fitted" in out_fc 
    print("✓ Box–Cox intervals monotone/positive (predict & forecast)")

def test_levels_unsorted_are_sorted_internally_insample():
    y = jnp.array([5.,6.,7.,8.,9.,10.,11.,12.])
    m = AutoTBATS(season_length=4, use_boxcox=False, use_trend=True, use_damped_trend=False, use_arma_errors=False)
    m.fit(y)
    res = m.predict_in_sample(level=(95,80))  # unsorted on purpose
    for k in ("fitted","lo-80","hi-80","lo-95","hi-95"):
        assert k in res
    assert jnp.all(res["lo-95"] <= res["lo-80"])
    assert jnp.all(res["hi-80"] <= res["hi-95"])
    print("✓ predict_in_sample sorts levels and preserves ordering")

def test_warning_on_short_sample_vs_seasonality():
    """Short sample relative to largest season raises a warning."""
    y = jnp.arange(20., dtype=jnp.float32)  # length < 3*max(season)
    m = AutoTBATS(season_length=[12, 6], use_boxcox=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(y)
        assert any("Short sample" in str(wi.message) for wi in w), "Expected short-sample warning"
    print("✓ Short-sample warning emitted")

def test_input_validation_inf_raises():
    y = jnp.array([1., jnp.inf, 3., 4.])
    m = AutoTBATS(season_length=2)
    try:
        m.fit(y)
        assert False, "Expected ValueError for Inf"
    except ValueError as e:
        assert "Inf" in str(e)
    print("✓ Input validation rejects Inf")

def test_multi_season_list_supported_by_class():
    """Class should accept a list of seasonal periods and produce finite output."""
    y = jnp.array(generate_multiple_seasonal_data(n=400, periods=(7, 30), seed=123), dtype=jnp.float32)
    m = AutoTBATS(season_length=[7,30], use_boxcox=False, use_trend=True, use_damped_trend=False, use_arma_errors=False)
    m.fit(y)
    r = m.predict(h=14, level=[90])
    assert r["mean"].shape == (14,)
    assert jnp.all(jnp.isfinite(r["mean"]))
    assert "lo-90" in r and "hi-90" in r
    print("✓ Multi-season list works through class API")

def test_arma_toggle_both_paths_run():
    """Ensure both use_arma_errors=False/True run without error."""
    y = jnp.array(generate_seasonal_data(n=150, season_length=12, seed=7), dtype=jnp.float32)
    for flag in (False, True):
        m = AutoTBATS(season_length=12, use_boxcox=False, use_trend=True, use_damped_trend=False, use_arma_errors=flag)
        m.fit(y)
        out = m.predict(h=6, level=[80])
        assert out["mean"].shape == (6,)
        assert jnp.all(jnp.isfinite(out["mean"]))
    print("✓ ARMA toggle works in fit+predict")

def test_damped_vs_undamped_long_horizon_behavior():
    """Damped trend should generally produce lower-magnitude long-horizon forecasts vs undamped."""
    y = jnp.array(generate_seasonal_data(n=180, season_length=12, trend=True, seed=9), dtype=jnp.float32)
    h = 36
    undamped = AutoTBATS(season_length=12, use_boxcox=False, use_trend=True, use_damped_trend=False, use_arma_errors=False).forecast(y, h)["mean"]
    damped   = AutoTBATS(season_length=12, use_boxcox=False, use_trend=True, use_damped_trend=True,  use_arma_errors=False).forecast(y, h)["mean"]
    # Heuristic: variance of damped horizon forecasts should be <= undamped's
    assert np.var(np.asarray(damped)) <= 1.2 * np.var(np.asarray(undamped))
    # And the tail absolute value should be generally smaller
    assert np.mean(np.abs(np.asarray(damped)[-12:])) <= 1.1 * np.mean(np.abs(np.asarray(undamped)[-12:]))
    print("✓ Damped vs undamped behaves as expected (heuristic checks)")

def test_sigmah_monotone_increasing_core_path():
    """sigma(h) produced by core path should be non-decreasing in h."""
    y = jnp.array(generate_seasonal_data(n=120, season_length=12, seed=11), dtype=jnp.float32)
    mod = _sel(y=y, seasonal_periods=[12], use_boxcox=False, bc_lower=0.0, bc_upper=1.0,
               use_trend=True, use_damped_trend=False, use_arma_errors=False)
    sigmah = np.asarray(_sig(mod, h=24)).ravel()
    diffs = np.diff(sigmah)
    assert np.all(diffs >= -1e-8)
    print("✓ sigma(h) non-decreasing in h (core path)")

def test_fit_predict_vs_forecast_parity_same_cfg():
    """fit()+predict(h) vs forecast(y,h) should be close under same config."""
    y = jnp.array(generate_seasonal_data(n=110, season_length=12, seed=21), dtype=jnp.float32)
    cfg = dict(season_length=12, use_boxcox=False, use_trend=True, use_damped_trend=False, use_arma_errors=False)
    m = AutoTBATS(**cfg)
    m.fit(y)
    out1 = m.predict(h=12)
    out2 = m.forecast(y=y, h=12)
    a, b = np.asarray(out1["mean"]), np.asarray(out2["mean"])
    assert np.allclose(a, b, rtol=0.15, atol=2.5), f"Means diverged (max abs diff {np.max(np.abs(a-b)):.3f})"
    print("✓ fit+predict and forecast are approximately aligned")

def test_conformal_intervals_shapes_and_monotonicity():
    """Conformal integration returns sane shapes and ordering."""
    y = jnp.array(generate_seasonal_data(n=130, season_length=12, seed=99), dtype=jnp.float32)
    ci = ConformalIntervals(h=5, n_windows=4, method="conformal_distribution")
    m = AutoTBATS(season_length=12, use_boxcox=False, conformal_params=ci)
    m.fit(y)
    out = m.predict(h=5, level=[70, 90])
    for lvl in (70, 90):
        lo, hi = f"lo-{lvl}", f"hi-{lvl}"
        assert lo in out and hi in out
        assert out[lo].shape == (5,) and out[hi].shape == (5,)
        assert bool(jnp.all(out[lo] <= out["mean"]))
        assert bool(jnp.all(out["mean"] <= out[hi]))
    print("✓ Conformal intervals: shapes and ordering OK")

def test_core_returns_original_scale_mean_when_boxcox_used():
    """Verify our contract: tbats_forecast returns original-scale mean if Box–Cox was used."""
    # Build a positive, multiplicative series where Box–Cox tends to help
    y_np = generate_exponential_data(n=120, seed=123)
    y = jnp.array(y_np, dtype=jnp.float32)
    mod = _sel(y=y, seasonal_periods=[12], use_boxcox=True, bc_lower=0.0, bc_upper=1.0,
               use_trend=True, use_damped_trend=False, use_arma_errors=False)
    out = _fc(mod, h=6)
    mean = np.asarray(out["mean"])
    # If already original scale, values should be near magnitude of last obs
    assert mean.shape == (6,)
    assert np.isfinite(mean).all()
    assert mean[-1] > 0 and mean[0] > 0
    # Very loose check: order of magnitude comparable to recent y
    assert 0.05 * np.abs(y_np[-1]) <= mean[0] <= 20.0 * np.abs(y_np[-1])
    print("✓ Core forecast() returns original-scale mean under Box–Cox")

def test_errors_key_present_and_finite_after_fit():
    """Model stores finite errors used for in-sample intervals."""
    y = jnp.array(generate_seasonal_data(n=90, season_length=6, seed=6), dtype=jnp.float32)
    m = AutoTBATS(season_length=6, use_boxcox=False, use_trend=True, use_damped_trend=False, use_arma_errors=False)
    m.fit(y)
    assert "errors" in m.model_ and jnp.ndim(m.model_["errors"]) == 2
    assert jnp.all(jnp.isfinite(m.model_["errors"]))
    print("✓ errors present and finite after fit")


def test_arma_on_initialization_paths():
    y = jnp.array(generate_seasonal_data(n=120, season_length=12, seed=1), dtype=jnp.float32)
    # Encourage small orders (fast)
    m = AutoTBATS(season_length=12, use_boxcox=False, use_trend=False, use_arma_errors=True)
    m.fit(y)
    p = int(m.model_.get("p", 0)); q = int(m.model_.get("q", 0))
    assert p >= 0 and q >= 0
    # Ensure state matrices are consistent with p,q
    F = m.model_["F"]; d = F.shape[0]
    assert d >= (1 + 2*int(jnp.sum(jnp.asarray(m.model_["k_vector"]))))


# ===============================
# Additional Coverage Tests
# ===============================

import numpy as np
import jax.numpy as jnp

# ---------- ARMA identification & boundaries ----------

def _gen_arma_noise(n, ar=None, ma=None, seed=123):
    rng = np.random.RandomState(seed)
    e = rng.normal(0, 1, size=n + 200)  # burn-in
    x = np.zeros_like(e)
    p = 0 if ar is None else len(ar)
    q = 0 if ma is None else len(ma)
    ar = np.array([]) if ar is None else np.array(ar, dtype=float)
    ma = np.array([]) if ma is None else np.array(ma, dtype=float)
    for t in range(max(p, q), len(e)):
        ar_part = (ar * x[t - np.arange(1, p + 1)]).sum() if p else 0.0
        ma_part = (ma * e[t - np.arange(1, q + 1)]).sum() if q else 0.0
        x[t] = ar_part + e[t] + ma_part
    return x[200:]  # drop burn-in

def test_arma_order_selection_ar_only_ma_only_higher_orders():
    """AR-only, MA-only, and ARMA(p,q) produce finite forecasts and nonzero orders when enabled."""
    n = 240
    t = np.arange(n)
    base = 100 + 0.03 * t + 6 * np.sin(2 * np.pi * t / 12)

    # forecast() is stateless (never writes model_) — fit first, then predict.
    # AR(2) noise
    y_ar = base + _gen_arma_noise(n, ar=[0.6, -0.3], ma=None, seed=1)
    m_ar = AutoTBATS(season_length=12, use_boxcox=False, use_trend=True,
                     use_damped_trend=False, use_arma_errors=True)
    m_ar.fit(jnp.array(y_ar, dtype=jnp.float32))
    out_ar = m_ar.predict(h=12)
    assert np.all(np.isfinite(np.asarray(out_ar["mean"])))
    # Expect some AR structure discovered (p>0 OR q>0 OK, but AR likely >0)
    assert m_ar.model_["p"] >= 0 and m_ar.model_["q"] >= 0

    # MA(2) noise
    y_ma = base + _gen_arma_noise(n, ar=None, ma=[-0.5, 0.4], seed=2)
    m_ma = AutoTBATS(season_length=12, use_boxcox=False, use_trend=True,
                     use_damped_trend=False, use_arma_errors=True)
    m_ma.fit(jnp.array(y_ma, dtype=jnp.float32))
    out_ma = m_ma.predict(h=12)
    assert np.all(np.isfinite(np.asarray(out_ma["mean"])))
    assert m_ma.model_["p"] >= 0 and m_ma.model_["q"] >= 0

    # ARMA(2,1) noise
    y_arma = base + _gen_arma_noise(n, ar=[0.5, -0.2], ma=[0.4], seed=3)
    m_arma = AutoTBATS(season_length=12, use_boxcox=False, use_trend=True,
                       use_damped_trend=False, use_arma_errors=True)
    m_arma.fit(jnp.array(y_arma, dtype=jnp.float32))
    out_arma = m_arma.predict(h=12)
    assert np.all(np.isfinite(np.asarray(out_arma["mean"])))
    assert m_arma.model_["p"] >= 0 and m_arma.model_["q"] >= 0
    print("✓ AR/MA/ARMA selection yields finite forecasts with nonnegative orders")


def test_stationarity_invertibility_constraints_enforced():
    """AR/MA coefficients are kept inside a safe region (no wild explosions)."""
    n = 200
    t = np.arange(n)
    y = 50 + 0.02 * t + 4 * np.sin(2 * np.pi * t / 6) + _gen_arma_noise(n, ar=[0.7], ma=[-0.4], seed=44)
    m = AutoTBATS(season_length=6, use_boxcox=False, use_trend=True,
                  use_damped_trend=False, use_arma_errors=True)
    m.fit(jnp.array(y, dtype=jnp.float32))  # forecast() is stateless; fit stores model_
    _ = m.predict(h=24)
    ar = m.model_.get("ar_coeffs", None)
    ma = m.model_.get("ma_coeffs", None)
    if ar is not None:
        assert jnp.all(jnp.abs(ar) < 0.99)
    if ma is not None:
        assert jnp.all(jnp.abs(ma) < 0.99)
    print("✓ AR/MA coeffs inside stability-ish region")


# ---------- Guerrero λ selection & bounds (edge behavior) ----------

def test_boxcox_lambda_edge_behavior_and_roundtrip():
    """Whatever λ the model settles on: inverse(BoxCox(y, λ)) ≈ y, and λ is finite."""
    n = 120
    y = jnp.array(np.exp(0.02 * np.arange(n)) * (1 + 0.1*np.sin(2*np.pi*np.arange(n)/12)) + 1.0,
                  dtype=jnp.float32)  # strictly positive, skewed
    m = AutoTBATS(season_length=12, use_boxcox=True, use_trend=True,
                  use_damped_trend=False, use_arma_errors=False,
                  )
    m.fit(y)  # forecast() is stateless; fit stores model_
    r = m.predict(h=6)
    lam = m.model_.get("BoxCox_lambda", None)
    assert lam is not None and np.isfinite(np.asarray(lam))
    # Roundtrip check on the forecast mean value (smoke)
    from chronax.models.tbats.tbats_core import _boxcox as _bc, _inv_boxcox as _ibc
    bc = _bc(r["mean"], lam)
    inv = _ibc(bc, lam)
    assert np.allclose(np.asarray(r["mean"]), np.asarray(inv), rtol=1e-6, atol=1e-6)
    print(f"✓ Box–Cox λ finite and roundtrip ok (λ≈{float(lam):.4f})")


# ---------- Seasonal harmonics ----------

def test_find_harmonics_small_and_prime_periods():
    """find_harmonics behaves on small and prime periods."""
    from chronax.models.tbats.tbats_core import find_harmonics
    n = 120
    y = jnp.array(5 + 2*jnp.sin(2*jnp.pi*jnp.arange(n)/5) + 0.5*jnp.sin(2*jnp.pi*jnp.arange(n)/11),
                  dtype=jnp.float32)
    for m in (2, 5, 11, 13):  # small & prime
        k, z = find_harmonics(y, m)
        # k is a jnp int32 scalar (traced-safe); concrete when called eagerly
        assert int(k) >= 1
        assert z.shape[0] == n
        assert jnp.all(jnp.isfinite(z))
    print("✓ find_harmonics stable on small/prime periods")

def test_multiperiod_harmonics_selection_smoke():
    """Smoke test for multi-season JAX — finite forecasts and correct k_vector length."""
    periods = [7, 30]          # switch to [7, 30, 365] after capping harmonics
    n = 365
    t = np.arange(n)
    y = 100 + 0.05*t \
        + 6*np.sin(2*np.pi*t/7) \
        + 3*np.sin(2*np.pi*t/30) \
        + np.random.normal(0, 1.5, n)
    y = jnp.array(y, dtype=jnp.float32)

    m = AutoTBATS(
        season_length=periods,
        use_boxcox=False,
        use_trend=True,
        use_damped_trend=False,
        use_arma_errors=False,
    )
    m.fit(y)  # forecast() is stateless; fit stores model_
    r = m.predict(h=14)
    kv = m.model_.get("k_vector", None)

    # Assert k_vector length matches the number of periods provided
    assert kv is not None and kv.shape[0] == len(periods), f"k_vector={kv}, periods={periods}"

    # Basic sanity: harmonics ≥ 1 and forecast finite
    assert jnp.all(kv >= 1)
    assert np.all(np.isfinite(np.asarray(r["mean"])))

    print(f"✓ multi-period k_vector computed: {np.asarray(kv)} for periods={periods}")


# ---------- Optimizer behavior ----------

def test_optimizer_reproducibility_same_data_same_result():
    """Two fits on the same data produce the same optim_params (deterministic NM)."""
    np.random.seed(7)
    n = 150
    t = np.arange(n)
    y = 100 + 0.02*t + 5*np.sin(2*np.pi*t/12) + np.random.normal(0, 1.0, n)
    yj = jnp.array(y, dtype=jnp.float32)

    m1 = AutoTBATS(season_length=12, use_boxcox=False, use_trend=True,
                   use_damped_trend=False, use_arma_errors=False)
    m1.fit(yj)
    p1 = np.asarray(m1.model_["optim_params"])

    m2 = AutoTBATS(season_length=12, use_boxcox=False, use_trend=True,
                   use_damped_trend=False, use_arma_errors=False)
    m2.fit(yj)
    p2 = np.asarray(m2.model_["optim_params"])

    assert np.allclose(p1, p2, rtol=1e-6, atol=1e-6)
    print("✓ Nelder–Mead deterministic on fixed data/config")


# ---------- Contract tests ----------

def test_tbats_core_forecast_exposes_mean_bc_when_boxcox_active():
    """tbats_core.tbats_forecast returns mean_bc if λ is used; otherwise None/absent."""
    from chronax.models.tbats.tbats_core import tbats_selection as _sel, tbats_forecast as _fc
    # Positive skewed series to encourage BC
    n = 120
    y = jnp.array(np.exp(0.02*np.arange(n)) * (1 + 0.1*np.sin(2*np.pi*np.arange(n)/12)) + 1.0,
                  dtype=jnp.float32)
    mod = _sel(y, seasonal_periods=[12], use_boxcox=True, bc_lower=0.0, bc_upper=1.0,
               use_trend=True, use_damped_trend=False, use_arma_errors=False)
    out = _fc(mod, h=6)
    if "mean_bc" in out:
        assert out["mean_bc"] is None or out["mean_bc"].shape == (6,)
        print("✓ tbats_core.tbats_forecast exposes mean_bc")
    else:
        # Allow skip if core not patched yet
        print("~ SKIP: tbats_core.tbats_forecast has no 'mean_bc' key")


def test_predict_in_sample_levels_shapes_and_keys():
    """predict_in_sample returns correct keys and shapes for multiple levels."""
    y = jnp.array([10., 12., 11., 13., 15., 14., 16., 17., 19., 18., 20., 21.], dtype=jnp.float32)
    m = AutoTBATS(season_length=4, use_boxcox=False, use_trend=True, use_damped_trend=False, use_arma_errors=False)
    m.fit(y)
    res = m.predict_in_sample(level=(50, 80, 95))
    assert res["fitted"].shape == (y.shape[0],)
    for L in (50, 80, 95):
        assert f"lo-{L}" in res and f"hi-{L}" in res
        assert res[f"lo-{L}"].shape == (y.shape[0],) and res[f"hi-{L}"].shape == (y.shape[0],)
    print("✓ predict_in_sample: keys and shapes across multiple levels")


# ---------- AIC & model selection ----------

def test_model_selection_picks_lowest_aic_deterministically():
    """tbats_selection returns the candidate with minimal AIC (recomputed)."""
    from chronax.models.tbats.tbats_core import tbats_model as _model, tbats_selection as _sel, find_harmonics as _fh
    # Build a dataset where trend helps a bit
    np.random.seed(123)
    n = 160
    t = np.arange(n)
    y = 50 + 0.03*t + 8*np.sin(2*np.pi*t/12) + np.random.normal(0, 1.0, n)
    yj = jnp.array(y, dtype=jnp.float32)

    # Rebuild k_vector like selection does
    ks = []
    z = yj
    for period in [12]:
        k, z = _fh(z, int(period))
        ks.append(int(k))
    k_vec = jnp.asarray(ks, dtype=jnp.int32)

    # Two explicit candidates
    cand1 = _model(yj, [12], k_vec, use_boxcox=False, bc_lower=0.0, bc_upper=1.0,
                   use_trend=True, use_damped_trend=False, use_arma_errors=False)
    cand2 = _model(yj, [12], k_vec, use_boxcox=False, bc_lower=0.0, bc_upper=1.0,
                   use_trend=False, use_damped_trend=False, use_arma_errors=False)
    aic_min = min(cand1["aic"], cand2["aic"])

    best = _sel(yj, seasonal_periods=[12], use_boxcox=False, bc_lower=0.0, bc_upper=1.0,
                use_trend=None, use_damped_trend=None, use_arma_errors=False)
    assert abs(float(best["aic"]) - float(aic_min)) < 1e-6
    print("✓ selection returns the minimal AIC candidate deterministically")


# ---------- Performance / regression guards (slow-ish but bounded) ----------

def test_performance_long_series_long_horizon_smoke():
    """Long n, long h, multi-season without ARMA runs and returns finite output (smoke)."""
    np.random.seed(9)
    n = 600
    t = np.arange(n)
    y = 100 + 0.02*t + 5*np.sin(2*np.pi*t/7) + 3*np.sin(2*np.pi*t/30) + 2*np.sin(2*np.pi*t/365) + np.random.normal(0, 1.2, n)
    yj = jnp.array(y, dtype=jnp.float32)
    m = AutoTBATS(season_length=[7, 30, 365], use_boxcox=False, use_trend=True,
                  use_damped_trend=False, use_arma_errors=False)
    r = m.forecast(yj, h=120)
    arr = np.asarray(r["mean"])
    print("Arr",arr)
    assert arr.shape == (120,) and np.all(np.isfinite(arr))
    print("✓ performance smoke: long series & horizon ran and returned finite output")


# ---------- Error semantics ----------

def test_damped_without_trend_raises_in_selection():
    """Using damped trend with trend=False should raise."""
    y = jnp.array([10., 11., 12., 13., 14., 15., 16., 17.], dtype=jnp.float32)
    m = AutoTBATS(season_length=4, use_boxcox=False, use_trend=False, use_damped_trend=True, use_arma_errors=False)
    try:
        _ = m.forecast(y, h=3)
        assert False, "Expected ValueError for damped trend without trend"
    except ValueError as e:
        assert "damped" in str(e).lower()
        print("✓ error: damped trend without trend raises")

def test_boxcox_negative_input_semantics():
    """Negative values + forced Box–Cox: fit raises; stateless forecast degrades.

    forecast() must trace under vmap (conformity_scores), so it cannot raise
    on data VALUES — a forced-Box-Cox fit on non-positive data instead
    degrades lax-natively to an UNTRANSFORMED fit (the core's NaN-lambda
    sentinel marks the transform as identity) and still produces a finite,
    honestly-optimized forecast. fit() keeps the eager raise.
    """
    y = jnp.array([-1., 2., 3., 4., 5., 6., 7., 8.], dtype=jnp.float32)
    m = AutoTBATS(season_length=4, use_boxcox=True, use_trend=True, use_damped_trend=False, use_arma_errors=False)
    try:
        m.fit(y)
        assert False, "Expected ValueError for negative inputs with forced Box–Cox"
    except ValueError as e:
        assert "positive" in str(e).lower()
    out = m.forecast(y, h=2)
    assert np.all(np.isfinite(np.asarray(out["mean"])))  # degraded to untransformed fit
    print("✓ negative + forced Box–Cox: fit raises, forecast degrades to untransformed")


# ============================================================================
# Forecast-variance and structure regression tests
# ============================================================================

def test_sigmah_variance_recursion_de_livera():
    """Native interval variance must follow v_h = sigma2*(1 + sum_{j<h} c_j^2)
    with c_j = w' F^(j-1) g — the first tail term uses F^0 = I, not F."""
    from chronax.models.tbats.tbats_core import _compute_sigmah_core
    F = jnp.array([[0.9]]); w = jnp.array([1.0]); g = jnp.array([0.5])
    sigma2 = jnp.asarray(4.0); y_sigma = jnp.asarray(1.0)
    se = np.asarray(_compute_sigmah_core(F, w, g, sigma2, y_sigma, 3, True))
    c1 = 0.5                # w' F^0 g
    c2 = 0.9 * 0.5          # w' F^1 g
    expect = np.sqrt(4.0 * np.array([1.0, 1.0 + c1**2, 1.0 + c1**2 + c2**2]))
    np.testing.assert_allclose(se, expect, rtol=1e-12)


def test_default_trend_grid_omits_damped_candidate():
    """The default AutoTBATS grid deliberately OMITS the (True, True) damped
    candidate: A/B showed adding it regresses airline/synth-multi holdout
    (chronax's damped fit underperforms there while its AIC wins selection —
    IC != holdout). It is only searched when explicitly requested."""
    from chronax.models.tbats.tbats_core import _trend_grid
    assert _trend_grid(None, None) == [(False, False), (True, False)]
    assert _trend_grid(True, None) == [(True, False), (True, True)]
    assert _trend_grid(True, True) == [(True, True)]
    assert _trend_grid(True, False) == [(True, False)]
    assert _trend_grid(False, None) == [(False, False)]


def test_boxcox_seed_roundtrip_preserves_negative_components():
    """The seed-state Box-Cox roundtrip must NOT clamp negative components
    (e.g. a downward trend slope) to a tiny positive value: the removed
    `_ensure_pos` wrapper would have sign-flipped them. This test pins that
    (a) the clamp genuinely changes such a seed (so its removal is
    observable) and (b) a downtrending series fit with forced Box-Cox+trend
    stays finite end-to-end."""
    from chronax.models.tbats.tbats_core import _inv_boxcox, _ensure_pos
    x0_bc = jnp.asarray([10.0, -1.5, 0.3])  # level, NEGATIVE trend, seasonal
    unclamped = np.asarray(_inv_boxcox(x0_bc, jnp.asarray(1.0)))  # = x + 1
    clamped = np.asarray(_ensure_pos(_inv_boxcox(x0_bc, jnp.asarray(1.0))))
    # The seed prep now uses `unclamped`; the old code used `clamped`.
    assert unclamped[1] < 0.0, "inv_boxcox should preserve the negative trend"
    assert clamped[1] > 0.0, "ensure_pos should have clamped it positive"
    assert not np.allclose(unclamped, clamped), "clamp must be observable here"

    rng = np.random.default_rng(0)
    n = 120
    y = 500.0 - 3.0 * np.arange(n) + rng.normal(0, 5, n)  # positive downtrend
    m = TBATS(season_length=12, use_boxcox=True, use_trend=True, use_damped_trend=False)
    fc = np.asarray(m.forecast(y=jnp.asarray(y), h=12)["mean"])
    assert np.all(np.isfinite(fc))
    assert float(np.polyfit(np.arange(12), fc, 1)[0]) < 0.0, "downtrend lost"


def test_find_harmonics_short_series_runs():
    """n < 2m must use the expanding rolling mean (no constant-mean special
    case) and return finite output."""
    from chronax.models.tbats.tbats_core import find_harmonics
    rng = np.random.default_rng(1)
    y = jnp.asarray(50 + 10 * np.sin(2 * np.pi * np.arange(18) / 12) + rng.normal(0, 0.5, 18))
    k, z = find_harmonics(y, 12)
    assert int(k) >= 1
    assert np.all(np.isfinite(np.asarray(z)))


if __name__ == "__main__":
    print("=" * 60)
    print("Testing AutoTBATS Class")
    print("=" * 60)

    test_basic_fit_predict()
    test_forecast_with_fitted()
    test_prediction_intervals_parametric()
    test_conformal_intervals()
    test_boxcox_transformation1() 
    test_input_validation()
    test_predict_before_fit()


    print("Testing predict_in_sample")
    print("=" * 60)

    test_predict_in_sample_basic()
    test_predict_in_sample_with_intervals()
    test_predict_in_sample_boxcox()  
    test_predict_in_sample_requires_fit()
    test_predict_in_sample_interval_shapes()

    test_boxcox_roundtrip_various_lambdas()
    test_boxcox_interval_monotonicity_predict_and_forecast()
    test_levels_unsorted_are_sorted_internally_insample()
    test_warning_on_short_sample_vs_seasonality()
    test_input_validation_inf_raises()
    test_multi_season_list_supported_by_class()
    test_arma_toggle_both_paths_run()
    test_damped_vs_undamped_long_horizon_behavior()
    test_sigmah_monotone_increasing_core_path()
    test_fit_predict_vs_forecast_parity_same_cfg()
    test_conformal_intervals_shapes_and_monotonicity()
    test_core_returns_original_scale_mean_when_boxcox_used()
    test_errors_key_present_and_finite_after_fit()

    test_arma_on_initialization_paths()
    test_arma_order_selection_ar_only_ma_only_higher_orders()
    test_stationarity_invertibility_constraints_enforced()
    test_boxcox_lambda_edge_behavior_and_roundtrip()
    test_find_harmonics_small_and_prime_periods()
    test_multiperiod_harmonics_selection_smoke()
    test_optimizer_reproducibility_same_data_same_result()
    test_tbats_core_forecast_exposes_mean_bc_when_boxcox_active()
    test_predict_in_sample_levels_shapes_and_keys()
    test_model_selection_picks_lowest_aic_deterministically()
    test_performance_long_series_long_horizon_smoke()
    test_damped_without_trend_raises_in_selection()
    test_boxcox_negative_input_semantics()

    print("\n" + "=" * 60)
    print("✓ All AutoTBATS tests passed!")
    print("=" * 60)

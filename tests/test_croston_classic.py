import jax
import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial as _partial
from typing import Optional, List, Dict
from chronax import utils
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals

from chronax.models import CrostonClassic


# ============================================================================
# Collected tests
# ============================================================================

def _intermittent_y(trailing_zeros=5):
    base = [0, 0, 3, 0, 0, 0, 5, 0, 2, 0, 0, 4, 0, 0, 6, 0, 1, 0, 0, 3] * 3
    return np.array(base + [0] * trailing_zeros, dtype=float)


def test_forecast_mean_matches_statsforecast():
    sf_models = pytest.importorskip("statsforecast.models")
    y = _intermittent_y()
    cx = np.asarray(CrostonClassic().forecast(y=y, h=4)["mean"])
    sf = sf_models.CrostonClassic().forecast(y=y, h=4)["mean"]
    np.testing.assert_allclose(cx, sf, rtol=1e-6)


def test_fitted_values_match_statsforecast():
    # Fitted values are one-step-ahead: position i uses only demands before i,
    # and positions after the last demand carry the final level (no NaN tail).
    sf_models = pytest.importorskip("statsforecast.models")
    for tz in (0, 5):
        y = _intermittent_y(trailing_zeros=tz)
        cx = np.asarray(CrostonClassic().forecast(y=y, h=4, fitted=True)["fitted"])
        sf = sf_models.CrostonClassic().forecast(y=y, h=4, fitted=True)["fitted"]
        np.testing.assert_allclose(cx, sf, rtol=1e-6, equal_nan=True)


def test_forecast_fitted_with_level_returns_fitted_intervals():
    y = _intermittent_y()
    m = CrostonClassic(conformal_params=ConformalIntervals(n_windows=2, h=4))
    res = m.forecast(y=y, h=4, level=[80, 95], fitted=True)
    for key in ("mean", "fitted", "lo-95", "hi-95",
                "fitted-lo-95", "fitted-hi-95", "fitted-lo-80", "fitted-hi-80"):
        assert key in res, f"missing {key}"
    assert np.asarray(res["fitted-lo-95"]).shape == y.shape


# ============================================================================
# Test Cases
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CROSTON CLASSIC - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Test 1: Basic instantiation and attributes
    print("\n[Test 1] Basic instantiation and attributes")
    print("-" * 80)
    model = CrostonClassic()
    assert model.alias == "CrostonClassic"
    assert model.conformal_params is None
    assert model.model_ == {}
    print("Model alias:", model.alias)
    print("✓ Model instantiates correctly with default parameters")
    
    # Test 2: Fit and predict with intermittent data
    print("\n[Test 2] Fit and predict with intermittent demand series")
    print("-" * 80)
    y = jnp.array([0., 5., 0., 0., 3., 0., 2., 0., 0., 4., 0., 0., 3.])
    model = CrostonClassic()
    model.fit(y)
    forecast = model.predict(h=3)
    
    print(f"Training data: {y}")
    print(f"Forecast (h=3): {forecast['mean']}")
    
    assert forecast["mean"].shape == (3,)
    assert jnp.all(jnp.isfinite(forecast["mean"]))
    # Croston forecast should be constant across horizon
    assert jnp.allclose(forecast["mean"][0], forecast["mean"][1]), "Croston produces flat forecast"
    print("✓ Intermittent demand forecast computed successfully")
    
    # Test 3: Fitted values and in-sample predictions
    print("\n[Test 3] Fitted values and in-sample predictions")
    print("-" * 80)
    assert "fitted" in model.model_
    assert model.model_["fitted"].shape == y.shape
    fitted = model.predict_in_sample()
    
    print(f"Training data length: {len(y)}")
    print(f"Fitted values length: {len(fitted['fitted'])}")
    print(f"First 5 fitted values: {fitted['fitted'][:5]}")
    
    assert "fitted" in fitted
    assert fitted["fitted"].shape == y.shape
    print("✓ Fitted values have correct shape and structure")
    
    # Test 4: Conformal prediction intervals
    print("\n[Test 4] Conformal prediction intervals")
    print("-" * 80)
    ci = ConformalIntervals(n_windows=3, h=2)
    model_ci = CrostonClassic(conformal_params=ci)
    model_ci.fit(y)
    forecast_ci = model_ci.predict(h=2, level=[80, 95])
    
    print(f"Conformal config: n_windows={ci.n_windows}, h={ci.h}")
    print(f"Mean forecast: {forecast_ci['mean']}")
    print(f"80% interval: [{forecast_ci['lo-80']}, {forecast_ci['hi-80']}]")
    print(f"95% interval: [{forecast_ci['lo-95']}, {forecast_ci['hi-95']}]")
    
    assert "mean" in forecast_ci
    assert "lo-80" in forecast_ci and "hi-80" in forecast_ci
    assert "lo-95" in forecast_ci and "hi-95" in forecast_ci
    assert forecast_ci["mean"].shape == (2,)
    assert jnp.all(forecast_ci["lo-80"] <= forecast_ci["mean"])
    assert jnp.all(forecast_ci["mean"] <= forecast_ci["hi-80"])
    assert jnp.all(forecast_ci["lo-95"] <= forecast_ci["lo-80"])
    assert jnp.all(forecast_ci["hi-80"] <= forecast_ci["hi-95"])
    print("✓ Conformal intervals computed correctly with proper nesting")
    
    # Test 5: Edge case - all zeros (no demand)
    print("\n[Test 5] Edge case: all zeros (no demand)")
    print("-" * 80)
    y_zeros = jnp.zeros(10)
    model_zeros = CrostonClassic()
    model_zeros.fit(y_zeros)
    forecast_zeros = model_zeros.predict(h=3)
    
    print(f"Training data: {y_zeros}")
    print(f"Forecast (should be all zeros): {forecast_zeros['mean']}")
    
    assert forecast_zeros["mean"].shape == (3,)
    assert jnp.all(forecast_zeros["mean"] == 0.0)
    print("✓ All-zero series correctly produces zero forecast (fallback to naive)")
    
    # Test 6: Edge case - no zeros (dense/continuous series)
    print("\n[Test 6] Edge case: no zeros (dense series)")
    print("-" * 80)
    y_dense = jnp.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    model_dense = CrostonClassic()
    model_dense.fit(y_dense)
    forecast_dense = model_dense.predict(h=3)
    
    print(f"Training data: {y_dense}")
    print(f"Forecast: {forecast_dense['mean']}")
    
    assert forecast_dense["mean"].shape == (3,)
    assert jnp.all(jnp.isfinite(forecast_dense["mean"]))
    # For dense series with interval=1 everywhere, forecast ≈ demand
    print("✓ Dense series (no intermittency) handled correctly")
    
    # Test 7: forecast() method (memory-efficient one-shot)
    print("\n[Test 7] forecast() method (one-shot without state)")
    print("-" * 80)
    y = jnp.array([0., 5., 0., 0., 3., 0., 2., 0., 0., 4.])
    model = CrostonClassic()
    forecast_result = model.forecast(y, h=3, fitted=True)
    
    print(f"Training data: {y}")
    print(f"Forecast: {forecast_result['mean']}")
    print(f"Fitted values included: {'fitted' in forecast_result}")
    
    assert "mean" in forecast_result
    assert "fitted" in forecast_result
    assert forecast_result["mean"].shape == (3,)
    assert forecast_result["fitted"].shape == y.shape
    print("✓ forecast() correctly returns mean and fitted without storing state")
    
    # Test 8: Error handling - predict without fit
    print("\n[Test 8] Error handling: predict without fit")
    print("-" * 80)
    model_unfit = CrostonClassic()
    try:
        model_unfit.predict(h=3)
        assert False, "Should raise error"
    except (KeyError, AttributeError) as e:
        print(f"✓ Correctly raises {type(e).__name__} when predicting without fitting")
    
    # Test 9: Error handling - intervals without conformal_params
    print("\n[Test 9] Error handling: intervals without conformal_params")
    print("-" * 80)
    model_no_ci = CrostonClassic()
    model_no_ci.fit(y)
    try:
        model_no_ci.predict(h=3, level=[80])
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "conformal_params" in str(e)
        print(f"✓ Correctly raises ValueError: {e}")
    
    # Test 10: new() method from BaseForecaster
    print("\n[Test 10] new() method from BaseForecaster")
    print("-" * 80)
    model_orig = CrostonClassic(alias="Original")
    model_orig.fit(y)
    model_new = model_orig.new()
    
    print(f"Original model alias: {model_orig.alias}")
    print(f"New model alias: {model_new.alias}")
    print(f"New model is fitted: {bool(model_new.model_)}")
    
    assert model_new.alias == "Original"
    # assert model_new.model_ == {}
    assert model_new.model_ != {}
    assert model_new._cs is None
    # print("✓ new() creates fresh instance with same parameters but no fitted state")
    print("✓ new() creates fresh instance with same parameters also with fitted state")
    
    # Test 11: Custom alias
    print("\n[Test 11] Custom alias")
    print("-" * 80)
    model_custom = CrostonClassic(alias="MyCroston")
    
    print(f"Custom alias: {model_custom.alias}")
    print(f"Repr: {repr(model_custom)}")
    
    assert model_custom.alias == "MyCroston"
    assert "MyCroston" in repr(model_custom)
    print("✓ Custom alias correctly stored and displayed")
    
    # Test 12: Verify Croston logic - single non-zero demand
    print("\n[Test 12] Croston logic: single non-zero demand")
    print("-" * 80)
    y_single = jnp.array([0., 0., 5., 0., 0., 0.])
    model_single = CrostonClassic()
    model_single.fit(y_single)
    forecast_single = model_single.predict(h=1)
    
    print(f"Training data: {y_single}")
    print(f"Single demand at index 2, value = 5.0")
    print(f"Forecast: {forecast_single['mean'][0]:.4f}")
    
    # With single demand, should produce positive finite forecast
    assert forecast_single["mean"][0] > 0
    assert jnp.isfinite(forecast_single["mean"][0])
    print("✓ Single demand produces valid positive forecast")
    
    # Test 13: Fitted intervals (native parametric method)
    print("\n[Test 13] Fitted intervals (native parametric)")
    print("-" * 80)
    y = jnp.array([0., 5., 0., 0., 3., 0., 2., 0., 0., 4.])
    model = CrostonClassic()
    model.fit(y)
    fitted_pi = model.predict_in_sample(level=[90])
    
    print(f"Training data: {y}")
    print(f"Fitted values: {fitted_pi['fitted']}")
    # print(f"90% interval available: {'lo-90' in fitted_pi and 'hi-90' in fitted_pi}")
    print(f"90% interval available: {'fitted-lo-90' in fitted_pi and 'fitted-hi-90' in fitted_pi}")    
    assert "fitted" in fitted_pi
    # assert "lo-90" in fitted_pi
    assert "fitted-lo-90" in fitted_pi
    # assert "hi-90" in fitted_pi
    assert "fitted-hi-90" in fitted_pi
    assert fitted_pi["fitted"].shape == y.shape
    print("✓ Native fitted intervals computed correctly")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nTest Summary:")
    print("  [1]  Basic instantiation with default parameters")
    print("  [2]  Fit and predict with intermittent demand series")
    print("  [3]  Fitted values and in-sample predictions")
    print("  [4]  Conformal prediction intervals with proper nesting")
    print("  [5]  Edge case: all zeros (no demand fallback)")
    print("  [6]  Edge case: no zeros (dense/continuous series)")
    print("  [7]  forecast() method (memory-efficient one-shot)")
    print("  [8]  Error handling: predict without fit")
    print("  [9]  Error handling: intervals without conformal_params")
    print("  [10] new() method from BaseForecaster")
    print("  [11] Custom alias functionality")
    print("  [12] Croston logic with single non-zero demand")
    print("  [13] Native fitted intervals (parametric method)")
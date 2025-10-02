"""
The HistoricAverage class implements a forecasting model where every forecast is
the historical mean of the observed series. This JAX implementation mirrors the
capabilities and conventions used across the codebase and is designed to be
fast, simple, and interoperable with conformal prediction utilities.

This implementation provides:
- fit(): Trains the model and stores parameters
- predict(): Makes forecasts using the fitted model (stateful), with optional confidence intervals
- predict_in_sample(): Returns in-sample fitted values with optional confidence intervals
- forecast(): Stateless prediction (fit + predict in a single call), with optional confidence intervals

Core computation is handled by the jitted function `_historic_average`, which
computes the mean-based forecasts and, optionally, the fitted values.

Confidence Intervals
- Native intervals: Uses the residual standard error with the normal
  approximation. For horizon h, HistoricAverage uses a constant standard error
  scaled by sqrt(1 + 1/n) (matching the implementation in this module).
- Conformal intervals: When `conformal_params` is provided, intervals can be
  computed via the BaseForecaster's conformal framework using
  `add_confidence_intervals`.

Instance Attributes
1. alias: model name, declared on initialization
2. conformal_params: optional conformal_intervals object to enable conformal interval computation
3. model_: dictionary storing fitted artifacts, e.g. mean, fitted values, sigma, n

Class Attributes
- uses_exog: whether the model supports exogenous variables (False for HistoricAverage)

Methods
- fit(y, X=None): stores mean, fitted values, sigma, and n
- predict(h, X=None, level=None): forecasts h steps ahead with optional intervals
- predict_in_sample(level=None): returns fitted values with optional intervals
- forecast(y, h, X=None, X_future=None, level=None, fitted=False): memory-efficient
  predictions without storing state; optionally returns fitted values and/or
  intervals

This model is intended as a simple and strong baseline: it is robust,
computationally light, and provides both native and conformal uncertainty
quantification paths.
"""

import jax
import jax.numpy as jnp

import utils
from base_forecaster import BaseForecaster
from conformal_intervals import ConformalIntervals


# ---------- Core: Single jitted function does everything ----------

from functools import partial

@partial(jax.jit, static_argnums=(1, 2))
def _historic_average(y: jnp.ndarray, h: int, fitted: bool = False) -> dict:
    """
    Core HistoricAverage computation (pure, jitted, reusable).
    
    Args:
        y: Time series
        h: Forecast horizon (static - must be known at compile time)
        fitted: Whether to return fitted values (static - must be known at compile time)
        
    Returns:
        Dict with 'mean' and optionally 'fitted'
    """
    mean_val = jnp.nanmean(y)
    fcst = {"mean": jnp.full((h,), mean_val, dtype=jnp.float32)}
    if fitted:
        fcst["fitted"] = jnp.full_like(y, mean_val)
    return fcst


# ---------- Model class ----------

class HistoricAverage(BaseForecaster):
    """
    Historic Average forecasting in JAX.
    
    Forecast = mean of all historical observations.
    Native intervals: σ_h = σ * sqrt(1 + 1/n)
    """
    
    uses_exog = False

    def __init__(self, alias: str = "HistoricAverage", conformal_params: ConformalIntervals | None = None):
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = {}

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "HistoricAverage":
        y = utils.ensure_float(y)
        mod = _historic_average(y, h=1, fitted=True)
        residuals = y - mod["fitted"]
        
        self.model_ = {
            "mean": mod["mean"],
            "fitted": mod["fitted"],
            "sigma": utils.calculate_sigma(residuals, len(y) - 1),
            "n": len(y),
        }
        
        # Fast conformity: vmap over dynamic slices
        @jax.jit
        def forecast_fn(y_full, X_full, te, h):
            return jnp.full((h,), jnp.nanmean(y_full[:te]), dtype=jnp.float32)
        
        self.forecast_fn = forecast_fn
        return self

    def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int] | None = None) -> dict:
        mean = jnp.full((h,), self.model_["mean"][0], dtype=jnp.float32)
        res = {"mean": mean}
        
        if level is not None:
            level = sorted(level)
            sigma, n = self.model_["sigma"], self.model_["n"]
            sigmah = sigma * jnp.sqrt(1.0 + (1.0 / n))
            
            if self.conformal_params is not None:
                res = self.add_confidence_intervals(
                    res, self.conformity_scores(self.model_["fitted"], X=None), level, "conformal_distribution"
                )
            else:
                res = {**res, **utils._calculate_intervals(mean, sigmah, level)}
        
        return res

    def predict_in_sample(self, level: list[int] | None = None) -> dict:
        res = {"fitted": self.model_["fitted"]}
        
        if level is not None:
            sigmah = self.model_["sigma"] * jnp.sqrt(1.0 + (1.0 / self.model_["n"]))
            res = {**res, **utils._add_fitted_pi(res["fitted"], sigmah, sorted(level))}
        
        return res

    def forecast(
        self, y: jnp.ndarray, h: int, X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None, level: list[int] | None = None, fitted: bool = False
    ) -> dict:
        y = utils.ensure_float(y)
        out = _historic_average(y, h, fitted or (level is not None))
        res = {"mean": out["mean"]}
        
        if fitted:
            res["fitted"] = out["fitted"]
        
        if level is not None:
            level = sorted(level)
            residuals = y - out["fitted"]
            sigma = utils.calculate_sigma(residuals, len(y) - 1)
            sigmah = sigma * jnp.sqrt(1.0 + (1.0 / len(y)))
            
            if self.conformal_params is not None:
                res = self.add_confidence_intervals(
                    res, self.conformity_scores(y, X=None), level, "conformal_distribution"
                )
            else:
                res = {**res, **utils._calculate_intervals(out["mean"], sigmah, level)}
            
            if fitted:
                res = {**res, **utils._add_fitted_pi(out["fitted"], sigmah, level)}
        
        return res


# ---------- Test ----------

if __name__ == "__main__":
    print("=" * 60)
    print("Testing HistoricAverage Model")
    print("=" * 60)
    
    # Test 1: Basic instantiation and repr
    print("\n[Test 1] Instantiation and __repr__")
    model = HistoricAverage()
    print(f"  Model alias: {model.alias}")
    print(f"  Model repr: {repr(model)}")
    assert model.alias == "HistoricAverage", "Default alias should be 'HistoricAverage'"
    assert repr(model) == "HistoricAverage", "repr should return alias"
    print("  ✓ Instantiation and repr work correctly")
    
    # Test 2: Fit and basic predict
    print("\n[Test 2] Fit and predict (no intervals)")
    y_train = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    model.fit(y_train)
    
    # Check model_ attributes
    assert "mean" in model.model_, "model_ should have 'mean' key"
    assert "fitted" in model.model_, "model_ should have 'fitted' key"
    assert "sigma" in model.model_, "model_ should have 'sigma' key"
    assert "n" in model.model_, "model_ should have 'n' key"
    assert model.model_["n"] == 10, "n should equal length of training data"
    
    # Check mean is correct (average of 1..10 is 5.5)
    expected_mean = jnp.mean(y_train)
    assert jnp.allclose(model.model_["mean"][0], expected_mean), f"Mean should be {expected_mean}"
    print(f"  Fitted mean: {model.model_['mean'][0]:.2f}")
    print(f"  Fitted sigma: {model.model_['sigma']:.4f}")
    
    # Predict
    h = 5
    preds = model.predict(h=h)
    assert "mean" in preds, "Predictions should have 'mean' key"
    assert preds["mean"].shape == (h,), f"Mean should have shape ({h},)"
    assert jnp.allclose(preds["mean"], expected_mean), "All predictions should equal historical mean"
    print(f"  Predictions (h={h}): {preds['mean']}")
    print("  ✓ Fit and predict work correctly")
    
    # Test 3: Predict with native intervals
    print("\n[Test 3] Predict with native intervals")
    preds_with_intervals = model.predict(h=3, level=[80, 95])
    
    # Check interval keys
    expected_keys = ["mean", "lo-80", "hi-80", "lo-95", "hi-95"]
    for key in expected_keys:
        assert key in preds_with_intervals, f"Should have '{key}' key"
        assert preds_with_intervals[key].shape == (3,), f"{key} should have shape (3,)"
    
    # Check intervals are monotonic: lo-95 <= lo-80 <= mean <= hi-80 <= hi-95
    mean = preds_with_intervals["mean"]
    lo_80 = preds_with_intervals["lo-80"]
    hi_80 = preds_with_intervals["hi-80"]
    lo_95 = preds_with_intervals["lo-95"]
    hi_95 = preds_with_intervals["hi-95"]
    
    assert jnp.all(lo_95 <= lo_80), "lo-95 should be <= lo-80"
    assert jnp.all(lo_80 <= mean), "lo-80 should be <= mean"
    assert jnp.all(mean <= hi_80), "mean should be <= hi-80"
    assert jnp.all(hi_80 <= hi_95), "hi-80 should be <= hi-95"
    
    print(f"  Mean: {mean[0]:.2f}")
    print(f"  80% interval: [{lo_80[0]:.2f}, {hi_80[0]:.2f}]")
    print(f"  95% interval: [{lo_95[0]:.2f}, {hi_95[0]:.2f}]")
    print("  ✓ Native intervals work correctly")
    
    # Test 4: predict_in_sample
    print("\n[Test 4] predict_in_sample")
    fitted_preds = model.predict_in_sample()
    assert "fitted" in fitted_preds, "Should have 'fitted' key"
    assert fitted_preds["fitted"].shape == y_train.shape, "Fitted should match training data shape"
    # All fitted values should equal the mean (that's how HistoricAverage works)
    assert jnp.allclose(fitted_preds["fitted"], expected_mean), "All fitted values should equal mean"
    print(f"  Fitted values (first 5): {fitted_preds['fitted'][:5]}")
    
    # With intervals
    fitted_with_intervals = model.predict_in_sample(level=[90])
    assert "fitted-lo-90" in fitted_with_intervals, "Should have fitted-lo-90"
    assert "fitted-hi-90" in fitted_with_intervals, "Should have fitted-hi-90"
    print(f"  90% fitted interval at first point: [{fitted_with_intervals['fitted-lo-90'][0]:.2f}, {fitted_with_intervals['fitted-hi-90'][0]:.2f}]")
    print("  ✓ predict_in_sample works correctly")
    
    # Test 5: Stateless forecast
    print("\n[Test 5] Stateless forecast (no intervals)")
    y_new = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
    forecast_res = HistoricAverage().forecast(y=y_new, h=3)
    assert "mean" in forecast_res, "Forecast should have 'mean' key"
    expected_new_mean = jnp.mean(y_new)
    assert jnp.allclose(forecast_res["mean"], expected_new_mean), f"Mean should be {expected_new_mean}"
    print(f"  Forecast mean: {forecast_res['mean'][0]:.2f} (expected: {expected_new_mean:.2f})")
    print("  ✓ Stateless forecast works correctly")
    
    # Test 6: Forecast with intervals and fitted values
    print("\n[Test 6] Forecast with intervals and fitted values")
    forecast_full = HistoricAverage().forecast(y=y_train, h=4, level=[80], fitted=True)
    
    assert "mean" in forecast_full, "Should have 'mean' key"
    assert "fitted" in forecast_full, "Should have 'fitted' key"
    assert "lo-80" in forecast_full, "Should have 'lo-80' key"
    assert "hi-80" in forecast_full, "Should have 'hi-80' key"
    assert "fitted-lo-80" in forecast_full, "Should have 'fitted-lo-80' key"
    assert "fitted-hi-80" in forecast_full, "Should have 'fitted-hi-80' key"
    
    assert forecast_full["mean"].shape == (4,), "Mean should have shape (4,)"
    assert forecast_full["fitted"].shape == y_train.shape, "Fitted should match input shape"
    
    print(f"  Forecast mean: {forecast_full['mean'][0]:.2f}")
    print(f"  Forecast 80% interval: [{forecast_full['lo-80'][0]:.2f}, {forecast_full['hi-80'][0]:.2f}]")
    print(f"  Fitted 80% interval (first point): [{forecast_full['fitted-lo-80'][0]:.2f}, {forecast_full['fitted-hi-80'][0]:.2f}]")
    print("  ✓ Forecast with intervals and fitted values works correctly")
    
    # Test 7: Edge case - small dataset
    print("\n[Test 7] Edge case - small dataset")
    y_small = jnp.array([5.0, 10.0])
    model_small = HistoricAverage().fit(y_small)
    preds_small = model_small.predict(h=2)
    expected_small_mean = jnp.mean(y_small)
    assert jnp.allclose(preds_small["mean"], expected_small_mean), "Should handle small datasets"
    print(f"  Small dataset mean: {expected_small_mean:.2f}")
    print(f"  Predictions: {preds_small['mean']}")
    print("  ✓ Small dataset handled correctly")
    
    # Test 8: new() method (shallow copy)
    print("\n[Test 8] new() method (shallow copy)")
    model_original = HistoricAverage(alias="Original")
    model_original.fit(y_train)
    model_copy = model_original.new()
    
    assert model_copy is not model_original, "new() should return a different object"
    assert model_copy.alias == model_original.alias, "Alias should be copied"
    assert model_copy.model_ is model_original.model_, "model_ should be shallow-copied (same reference)"
    
    # Modify copy's model_ and verify it affects original (shallow copy behavior)
    original_mean = model_original.model_["mean"][0]
    model_copy.model_["test_key"] = "test_value"
    assert "test_key" in model_original.model_, "Shallow copy shares nested references"
    
    print(f"  Original alias: {model_original.alias}")
    print(f"  Copy alias: {model_copy.alias}")
    print(f"  Shallow copy verified: nested dict shared")
    print("  ✓ new() method works correctly")
    
    # Test 9: Custom alias
    print("\n[Test 9] Custom alias")
    custom_model = HistoricAverage(alias="MyCustomHistAvg")
    assert custom_model.alias == "MyCustomHistAvg", "Should accept custom alias"
    assert repr(custom_model) == "MyCustomHistAvg", "repr should use custom alias"
    print(f"  Custom alias: {custom_model.alias}")
    print("  ✓ Custom alias works correctly")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ All HistoricAverage tests passed successfully!")
    print("=" * 60)
"""
Croston's Classic method for intermittent demand forecasting.

Designed for sparse time series with many zeros and occasional non-zero demands.
Decomposes series into demand size and inter-demand intervals, forecasting each separately.

Formula:
    ŷ[t+h] = demand_forecast / interval_forecast

Where:
    - demand_forecast: SES(α=0.1) on non-zero values
    - interval_forecast: SES(α=0.1) on time gaps between non-zero values
    - α = 0.1 is Croston's original fixed smoothing parameter

Implementation:
    - Uses jax.lax.cond for efficient no-demand fallback
    - JIT-compiled with static_argnums for h and fitted parameters
    - Expands fitted values back to original series length
    - Falls back to naive forecast when no non-zero demands exist

Instance Attributes:
    - alias: model identifier string
    - conformal_params: optional ConformalIntervals object for prediction intervals
    - model_: dict storing fitted values and conformity scores post-fit
    - _cs: conformity scores for conformal prediction

Methods:
    - fit(y, X): fits the model and computes conformity scores if conformal_params is set
    - forecast(y, h, X, X_future, level, fitted): pure forecast function with optional intervals
    - predict(h, X, level): generates h-step ahead predictions with optional conformal intervals
    - predict_in_sample(level): returns fitted values with optional native intervals
"""

import jax
import jax.numpy as jnp
from functools import partial as _partial
from typing import Optional, List, Dict
import utils
from base_forecaster import BaseForecaster
from conformal_intervals import ConformalIntervals


# ============================================================================
# CORE CROSTON CLASSIC LOGIC
# ============================================================================

@_partial(jax.jit, static_argnums=(1, 2))
def _croston_classic(
    y: jnp.ndarray,
    h: int,
    fitted: bool = False,
) -> dict:
    """
    Croston's Classic method for intermittent demand forecasting.
    
    Core function implementing Croston's algorithm:
    1. Extract demand (non-zero values)
    2. Extract intervals (time between non-zero values)
    3. Forecast both using SES with α=0.1
    4. Combine as: forecast = demand_forecast / interval_forecast
    
    Args:
        y: Time series (1D array)
        h: Forecast horizon (static)
        fitted: Whether to return in-sample fitted values (static)
        
    Returns:
        Dictionary with 'mean' and optionally 'fitted'
        
    Example:
        >>> y = jnp.array([0., 5., 0., 0., 3., 0., 2., 0., 0.])
        >>> result = _croston_classic(y, h=3, fitted=False)
        >>> result['mean'].shape
        (3,)
        
    Note:
        JIT-compiled using fixed-size NaN-padded arrays for intermittent demand.
    """
    y = utils.ensure_float(y)
    alpha = 0.1  # Fixed smoothing parameter for Croston Classic
    
    # Extract demand (non-zero values)
    yd = utils._demand(y)
    
    # Handle no-demand case: fall back to naive forecast
    def no_demand_forecast():
        """Fallback when series has no non-zero values."""
        last_val = y[-1]
        mean = utils._repeat_val(last_val, h)
        out = {"mean": mean}
        if fitted:
            # Naive fitted: y[t] = y[t-1]
            out["fitted"] = jnp.concatenate([jnp.array([jnp.nan]), y[:-1]])
        return out
    
    # Normal Croston forecast
    def croston_forecast():
        """Standard Croston algorithm when demand exists."""
        # Forecast demand using SES
        ydp, ydf = utils._ses_forecast(yd, alpha)
        
        # Extract intervals between non-zero values
        yi = utils._intervals(y)
        
        # Forecast intervals using SES
        yip, yif = utils._ses_forecast(yi, alpha)
        
        # Compute mean forecast: demand / interval
        # If interval forecast is 0, just use demand to avoid division by zero
        mean_val = jnp.where(
            yip != 0.0,
            ydp / yip,
            ydp
        )
        
        out = {"mean": utils._repeat_val(mean_val, h)}
        
        if fitted:
            # Expand fitted values back to original series length
            # Append forecast to fitted arrays for expansion
            ydf_expanded = utils._expand_fitted_demand(
                jnp.append(ydf, ydp), y
            )
            yif_expanded = utils._expand_fitted_intervals(
                jnp.append(yif, yip), y
            )
            # Combine: fitted = demand_fitted / interval_fitted
            out["fitted"] = ydf_expanded / yif_expanded
        
        return out
    
    # Use lax.cond to handle no-demand case efficiently
    return jax.lax.cond(
        yd.size == 0,
        no_demand_forecast,
        croston_forecast
    )


# ============================================================================
# CROSTON CLASSIC MODEL CLASS
# ============================================================================

class CrostonClassic(BaseForecaster):
    """
    Croston's Classic method for intermittent demand time series.
    
    Suitable for series with many zero values and occasional non-zero demand.
    Uses SES (α=0.1) to forecast both demand size and inter-demand intervals.
    
    **Key Features:**
    - Handles sparse/intermittent data (many zeros)
    - Fixed smoothing parameter α=0.1 (Croston's original specification)
    - Decomposes series into demand size and demand intervals
    - Conformal prediction intervals supported
    
    **Note:** Only conformal intervals are supported (no native parametric intervals).
    
    Args:
        alias: Model name (default: "CrostonClassic")
        conformal_params: Configuration for conformal prediction intervals
        
    Example:
        >>> from conformal_intervals import ConformalIntervals
        >>> ci = ConformalIntervals(h=3, n_windows=10)
        >>> model = CrostonClassic(conformal_params=ci)
        >>> y = jnp.array([0., 5., 0., 0., 3., 0., 2., 0., 0., 4., 0., 0., 3.])
        >>> model.fit(y)
        >>> forecast = model.predict(h=3, level=[80, 95])
    """
    
    def __init__(
        self,
        alias: str = "CrostonClassic",
        conformal_params: Optional[ConformalIntervals] = None,
    ):
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = {}
        self._cs = None
    
    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "CrostonClassic":
        """
        Fit Croston Classic model to time series.
        
        Args:
            y: Time series of shape (t,)
            X: Unused (no exogenous support)
            
        Returns:
            Self (fitted model)
        """
        y = utils.ensure_float(y)
        
        # Fit model and compute fitted values
        self.model_ = _croston_classic(y=y, h=1, fitted=True)
        
        # Calculate residual standard error
        residuals = y - self.model_["fitted"]
        self.model_["sigma"] = utils.calculate_sigma(residuals, y.size)
        
        # Store conformity scores if conformal intervals requested
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y, X)
        
        return self
    
    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Generate forecasts using fitted model.
        
        Args:
            h: Forecast horizon
            X: Unused (no exogenous support)
            level: Confidence levels for prediction intervals (0-100)
            
        Returns:
            Dictionary with 'mean' and optional interval keys ('lo-XX', 'hi-XX')
        """
        # Point forecast (constant across horizon for Croston Classic)
        mean = utils._repeat_val(val=self.model_["mean"][0], h=h)
        res = {"mean": mean}
        
        # Add conformal prediction intervals if requested
        if level is not None:
            if self.conformal_params is None:
                raise ValueError(
                    "You must instantiate the class with `conformal_params` "
                    "to calculate prediction intervals"
                )
            level = sorted(level)
            res = self.add_confidence_intervals(
                fcst=res,
                cs=self._cs,
                level=level,
                method=self.conformal_params.method
            )
        
        return res
    
    def predict_in_sample(
        self,
        level: Optional[List[int]] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Access fitted (in-sample) predictions.
        
        Args:
            level: Confidence levels for fitted intervals (0-100)
            
        Returns:
            Dictionary with 'fitted' and optional interval keys
            
        Note:
            Native (parametric) fitted intervals are supported using residual std error.
        """
        res = {"fitted": self.model_["fitted"]}
        
        # Add native (parametric) fitted intervals if requested
        if level is not None:
            level = sorted(level)
            res = {**res, **utils._add_fitted_pi(
                fitted=self.model_["fitted"],
                sigmah=self.model_["sigma"],
                level=level
            )}
        
        return res
    
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """
        Memory-efficient forecast without storing model state.
        
        Equivalent to fit().predict() but avoids object storage overhead.
        Useful for one-shot forecasting or cross-validation loops.
        
        Args:
            y: Time series of shape (t,)
            h: Forecast horizon
            X: Unused (no exogenous support)
            X_future: Unused (no exogenous support)
            level: Confidence levels for prediction intervals (0-100)
            fitted: Whether to return in-sample fitted values
            
        Returns:
            Dictionary with 'mean', optional 'fitted', and interval keys
        """
        y = utils.ensure_float(y)
        
        # Compute forecast using core function
        res = _croston_classic(y=y, h=h, fitted=fitted)
        
        # Add conformal prediction intervals if requested
        if level is not None:
            if self.conformal_params is None:
                raise ValueError(
                    "You must instantiate the class with `conformal_params` "
                    "to calculate prediction intervals"
                )
            level = sorted(level)
            
            # Add conformal intervals for forecast
            res = self.add_confidence_intervals(
                fcst=dict(res),
                cs=self.conformity_scores(y, X),
                level=level,
                method=self.conformal_params.method
            )
        
        # Add native fitted intervals if requested
        if fitted and level is not None:
            sigma = utils.calculate_sigma(y - res["fitted"], y.size)
            res = {**res, **utils._add_fitted_pi(
                fitted=res["fitted"],
                sigmah=sigma,
                level=level
            )}
        
        return res


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
    assert model_new.model_ == {}
    assert model_new._cs is None
    print("✓ new() creates fresh instance with same parameters but no fitted state")
    
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
    print(f"90% interval available: {'lo-90' in fitted_pi and 'hi-90' in fitted_pi}")
    
    assert "fitted" in fitted_pi
    assert "lo-90" in fitted_pi
    assert "hi-90" in fitted_pi
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
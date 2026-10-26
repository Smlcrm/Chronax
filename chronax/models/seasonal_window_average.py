"""
The SeasonalWindowAverage class implements a forecasting model where predictions
are computed by averaging the last window_size complete seasonal cycles for each
position in the seasonal period. This JAX implementation is designed for time
series with strong, stable seasonal patterns.

This implementation provides:
- fit(): Trains the model and stores the averaged seasonal pattern
- predict(): Makes forecasts using the fitted model (stateful), with optional conformal intervals
- forecast(): Stateless prediction (fit + predict in a single call), with optional conformal intervals

Core computation is handled by the JIT-compiled function `_seasonal_window_average`,
which extracts the last window_size seasonal cycles, reshapes them into a matrix,
and averages across cycles to produce a repeating seasonal pattern for forecasting.

Confidence Intervals
- Native intervals: NOT SUPPORTED. This model does not provide parametric/native
  prediction intervals due to its non-parametric, pattern-averaging nature.
- Conformal intervals: When `conformal_params` is provided, intervals are computed
  via the BaseForecaster's conformal framework using `add_confidence_intervals`.
  The `only_conformal_intervals` flag is always True for this model.

Instance Attributes
1. alias: model name, declared on initialization (default: "SeasWA")
2. conformal_params: optional ConformalIntervals object to enable conformal intervals
3. season_length: number of observations per seasonal cycle (e.g., 24 for hourly/daily)
4. window_size: number of recent complete cycles to average (e.g., 7 for weekly)
5. only_conformal_intervals: flag indicating no native intervals exist (always True)
6. ``model_``: dictionary storing the fitted seasonal pattern of length season_length

Class Attributes
- uses_exog: whether the model supports exogenous variables (False for SeasonalWindowAverage)

Methods
- fit(y, X=None): computes and stores the average seasonal pattern from the last
  window_size cycles
- predict(h, X=None, level=None): forecasts h steps ahead by tiling the stored
  seasonal pattern; optionally adds conformal intervals
- predict_in_sample(level=None): NOT IMPLEMENTED (raises NotImplementedError)
- forecast(y, h, X=None, X_future=None, level=None, fitted=False): memory-efficient
  predictions without storing state; raises NotImplementedError if fitted=True

Key Limitations
- Requires at least season_length × window_size observations
- Does NOT support fitted values (predict_in_sample and fitted=True)
- Does NOT support native/parametric intervals (conformal only)

This model is best suited for time series with strong, stable seasonal patterns
where recent cycles provide reliable indicators. It is computationally efficient
(fully JIT-compiled) and provides distribution-free uncertainty via conformal prediction.
"""

import jax
import jax.numpy as jnp
from jax import lax

from chronax import utils
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals


# ---------- Core: Single jitted function ----------

from functools import partial

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _seasonal_window_average(
    y: jnp.ndarray, 
    h: int, 
    fitted: bool, 
    season_length: int, 
    window_size: int
) -> dict:
    """
    Core SeasonalWindowAverage computation (pure, jitted, reusable).
    
    Computes forecast by averaging the last window_size complete seasonal cycles.
    For each position in the seasonal period, takes mean of last window_size values.
    
    Args:
        y: Time series of shape (t,)
        h: Forecast horizon
        fitted: Whether to return fitted values (NOT IMPLEMENTED - raises error)
        season_length: Number of observations per seasonal period (e.g., 24 for hourly with daily seasonality)
        window_size: Number of most recent seasonal cycles to average (e.g., 7 for last week)
        
    Returns:
        Dictionary with 'mean' key containing forecasts of shape (h,)
        Returns NaN forecast if insufficient data (< season_length * window_size observations)
        
    Example:
        For hourly data with daily seasonality (season_length=24) and window_size=7:
        - Takes last 7*24=168 observations
        - Reshapes to (7, 24)
        - Averages across 7 days to get 24-hour seasonal pattern
        - Tiles pattern to cover horizon h
    """
    # Fitted values not supported for this model
    if fitted:
        raise NotImplementedError("fitted values not supported for SeasonalWindowAverage")
    
    min_samples = season_length * window_size  # This is static (known at compile time)
    
    # Always pad y with min_samples NaNs at the start to ensure we can always
    # slice min_samples elements. This is necessary because lax.cond traces both 
    # branches at compile time, and the reshape operation needs a fixed-size array.
    # Using static min_samples (not y.size) ensures the shape is known at compile time.
    y_padded = jnp.concatenate([jnp.full(min_samples, jnp.nan, dtype=y.dtype), y])
    
    # Take the last min_samples elements (always guaranteed to have correct size)
    y_window = y_padded[-min_samples:]
    
    def sufficient_data(_):
        """Compute seasonal forecast when we have enough data."""
        # Reshape to (window_size, season_length) and average across windows (axis=0)
        # This gives us the average value for each position in the seasonal period
        season_avgs = y_window.reshape(window_size, season_length).mean(axis=0)
        
        # Tile seasonal pattern to cover horizon h
        return utils._repeat_val_seas(season_avgs, h)
    
    def insufficient_data(_):
        """Return NaN when we don't have enough data."""
        # dtype must match the sufficient_data branch exactly: lax.cond requires
        # identical abstract values (shape AND dtype) from both branches, and the
        # sufficient branch inherits y's dtype (float64 when x64 is enabled).
        return jnp.full(h, jnp.nan, dtype=y.dtype)
    
    # Use lax.cond for true conditional branching (only one branch executes)
    # This is required for JIT compilation with traced values (y.size is not known at compile time)
    mean = lax.cond(
        y.size >= min_samples,  # condition: do we have enough data?
        sufficient_data,         # true branch: compute seasonal forecast
        insufficient_data,       # false branch: return NaN
        None                     # operand (unused but required by lax.cond signature)
    )
    
    return {"mean": mean}


# ---------- Model class ----------

class SeasonalWindowAverage(BaseForecaster):
    """
    SeasonalWindowAverage forecasting model in JAX.
    
    Uses the average of the last k observations of each seasonal period as forecast,
    where k = window_size.
    
    Key limitations:
    - ONLY supports conformal prediction intervals (no native/parametric intervals)
    - Does NOT support predict_in_sample (fitted values)
    - Requires at least season_length * window_size observations
    
    Attributes:
        season_length: Number of observations per seasonal period
        window_size: Number of seasonal cycles to average
        alias: Model name
        prediction_intervals: ConformalIntervals object (REQUIRED for intervals)
        only_conformal_intervals: Flag indicating no native intervals (always True)
        ``model_``: Dictionary storing fitted seasonal pattern
    
    Example:
        >>> # Hourly data with daily seasonality, averaging last 7 days
        >>> from chronax.utils import ConformalIntervals
        >>> ci = ConformalIntervals(h=24, n_windows=10)
        >>> model = SeasonalWindowAverage(season_length=24, window_size=7, prediction_intervals=ci)
        >>> model.fit(y_train)
        >>> forecasts = model.predict(h=24, level=[80, 95])
    """
    
    uses_exog = False  # This model does not support exogenous variables

    def __init__(
        self,
        season_length: int,
        window_size: int,
        alias: str = "SeasWA",
        conformal_params: ConformalIntervals | None = None,
    ) -> None:
        """
        Initialize SeasonalWindowAverage model.
        
        Args:
            season_length: Number of observations per seasonal period 
                          (e.g., 24 for hourly data with daily seasonality)
            window_size: Number of most recent seasonal cycles to average 
                        (e.g., 7 to average the same hour over last 7 days)
            alias: Custom model name
            conformal_params: conformal_intervals object (REQUIRED for computing intervals)
        """
        self.season_length = season_length
        self.window_size = window_size
        self.alias = alias
        self.conformal_params = conformal_params
        self.only_conformal_intervals = True  # No native intervals available for this model
        self.model_ = {}

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "SeasonalWindowAverage":
        """
        Fit SeasonalWindowAverage model to time series y.
        
        Computes and stores the seasonal pattern (averages for each position in season).
        Also caches conformity scores on the training series when `conformal_params` is set.
        
        Args:
            y: Time series of shape (t,)
            X: Ignored (no exogenous support)
            
        Returns:
            self: Fitted model
        """
        y = utils.ensure_float(y)

        # Compute and store seasonal pattern (will be tiled in predict)
        mod = _seasonal_window_average(
            y,
            h=self.season_length,  # compute one full seasonal cycle
            fitted=False,
            season_length=self.season_length,
            window_size=self.window_size
        )
        self.model_ = mod
        # Pre-compute and cache conformity scores on the TRAINING series for
        # predict() intervals (the fitted seasonal pattern alone is too short
        # and is not the series the CV residuals must come from).
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None
        return self

    def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int] | None = None) -> dict:
        """
        Generate h-step ahead forecasts using fitted model.
        
        Args:
            h: Forecast horizon (number of steps ahead)
            X: Ignored (no exogenous support)
            level: Confidence levels (0-100) for prediction intervals (e.g., [80, 95]). Requires prediction_intervals to be set.
            
        Returns:
            Dictionary with keys:
            - 'mean': Point forecasts of shape (h,)
            - 'lo-XX': Lower bounds at XX% level (if level specified)
            - 'hi-XX': Upper bounds at XX% level (if level specified)
            
        Raises:
            Exception: If level is requested but prediction_intervals is None
        """
        self._require_fitted()
        # Tile stored seasonal pattern to cover horizon h
        mean = utils._repeat_val_seas(self.model_["mean"], h)
        res = {"mean": mean}
        
        if level is None:
            return res
        
        # Add conformal prediction intervals
        level = sorted(level)
        if self.conformal_params is None:
            raise Exception("You must pass `conformal_params` to compute them.")
        if getattr(self, "_cs", None) is None:
            raise ValueError(
                "Conformity scores are not available. Fit the model first (fit(...)) "
                "with `conformal_params` set so predict() can use cached scores."
            )
        if h != self.conformal_params.h:
            raise ValueError(
                f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                "conformity scores cover exactly conformal_params.h steps."
            )

        res = self.add_confidence_intervals(
            res,
            self._cs,  # cached at fit() on the training series
            level,
            self.conformal_params.method,
        )

        return res

    def predict_in_sample(self, level: list[int] | None = None) -> dict:
        """
        Access fitted in-sample predictions (NOT IMPLEMENTED).
        
        SeasonalWindowAverage does not support fitted values computation.
        
        Args:
            level: Confidence levels (0-100) for prediction intervals
            
        Raises:
            NotImplementedError: This method is not supported for SeasonalWindowAverage
        """
        raise NotImplementedError("predict_in_sample not supported for SeasonalWindowAverage")

    def forecast(
        self, 
        y: jnp.ndarray, 
        h: int, 
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None, 
        level: list[int] | None = None, 
        fitted: bool = False
    ) -> dict:
        """
        Memory-efficient SeasonalWindowAverage predictions.
        
        This method avoids memory burden from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.
        
        Args:
            y: Time series of shape (t,)
            h: Forecast horizon
            X: Ignored (no exogenous support)
            X_future: Ignored (no exogenous support)
            level: Confidence levels (0-100) for prediction intervals
            fitted: Whether to return in-sample predictions (NOT SUPPORTED - will raise error if True)
            
        Returns:
            Dictionary with 'mean' and optional 'lo-XX'/'hi-XX' interval keys
            
        Raises:
            Exception: If level is requested but prediction_intervals is None
            NotImplementedError: If fitted=True (not supported)
        """
        y = utils.ensure_float(y)
        
        # Compute forecast directly without storing fitted model
        res = dict(_seasonal_window_average(y, h, fitted, self.season_length, self.window_size))
        
        if level is None:
            return res
        
        # Add conformal prediction intervals
        level = sorted(level)
        if self.conformal_params is None:
            raise Exception("You must pass `conformal_params` to compute them.")
        if h != self.conformal_params.h:
            raise ValueError(
                f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                "conformity scores cover exactly conformal_params.h steps."
            )

        res = self.add_confidence_intervals(
            res,
            self.conformity_scores(y, X=None),  # compute conformity scores on-the-fly
            level,
            self.conformal_params.method,
        )

        return res


# ---------- Test ----------

# if __name__ == "__main__":
#     print("=" * 60)
#     print("Testing SeasonalWindowAverage Model")
#     print("=" * 60)
    
#     # Test 1: Basic instantiation and repr
#     print("\n[Test 1] Instantiation and __repr__")
#     model = SeasonalWindowAverage(season_length=7, window_size=2)
#     print(f"  Model alias: {model.alias}")
#     print(f"  Model repr: {repr(model)}")
#     print(f"  Season length: {model.season_length}")
#     print(f"  Window size: {model.window_size}")
#     assert model.alias == "SeasWA", "Default alias should be 'SeasWA'"
#     assert repr(model) == "SeasWA", "repr should return alias"
#     assert model.only_conformal_intervals == True, "Should only support conformal intervals"
#     print("  ✓ Instantiation and repr work correctly")
    
#     # Test 2: Fit and basic predict
#     print("\n[Test 2] Fit and predict (no intervals)")
#     # Create seasonal data: pattern [1, 2, 3, 4, 5, 6, 7] repeated 3 times
#     season_pattern = jnp.array([1., 2., 3., 4., 5., 6., 7.])
#     y_train = jnp.tile(season_pattern, 3)  # 21 observations (3 complete cycles)
    
#     model = SeasonalWindowAverage(season_length=7, window_size=2)
#     model.fit(y_train)
    
#     # Check model_ attributes
#     assert "mean" in model.model_, "model_ should have 'mean' key"
#     assert model.model_["mean"].shape == (7,), "Stored pattern should have shape (season_length,)"
    
#     # Predict for h=14 (2 complete cycles)
#     h = 14
#     preds = model.predict(h=h)
#     assert "mean" in preds, "Predictions should have 'mean' key"
#     assert preds["mean"].shape == (h,), f"Mean should have shape ({h},)"
    
#     # First 7 predictions should match stored seasonal pattern
#     assert jnp.allclose(preds["mean"][:7], model.model_["mean"]), "First cycle should match stored pattern"
#     assert jnp.allclose(preds["mean"][7:14], model.model_["mean"]), "Second cycle should match stored pattern"
    
#     print(f"  Stored seasonal pattern: {model.model_['mean']}")
#     print(f"  Predictions (first 7): {preds['mean'][:7]}")
#     print(f"  Predictions (next 7): {preds['mean'][7:14]}")
#     print("  ✓ Fit and predict work correctly")
    
#     # Test 3: Stateless forecast
#     print("\n[Test 3] Stateless forecast (no intervals)")
#     # Test with hourly data (season_length=24, window_size=7 for last week)
#     season_length = 24
#     window_size = 7
    
#     # Create data: 10 complete days
#     daily_pattern = jnp.arange(1., 25.)  # 1 to 24
#     y_multi_season = jnp.tile(daily_pattern, 10)  # 240 observations
    
#     forecast_res = SeasonalWindowAverage(season_length=season_length, window_size=window_size).forecast(
#         y=y_multi_season, h=48
#     )
    
#     assert "mean" in forecast_res, "Forecast should have 'mean' key"
#     assert forecast_res["mean"].shape == (48,), "Should forecast 48 hours"
    
#     # Check that forecast repeats the seasonal pattern
#     assert jnp.allclose(forecast_res["mean"][:24], forecast_res["mean"][24:48]), "Should repeat 24-hour pattern"
    
#     print(f"  Forecast for h=48 (2 days)")
#     print(f"  First day pattern (hours 0-5): {forecast_res['mean'][:6]}")
#     print(f"  Second day pattern (hours 0-5): {forecast_res['mean'][24:30]}")
#     print("  ✓ Stateless forecast works correctly")
    
#     # Test 4: Edge case - insufficient data (returns NaN)
#     # NOTE: Skipped because lax.cond traces both branches, causing reshape errors
#     # when y_window.size < season_length * window_size, even though that branch won't execute
#     print("\n[Test 4] Edge case - insufficient data (SKIPPED)")
#     print("  ⚠ Skipped: lax.cond traces both branches, causing reshape incompatibility")
#     print("  ✓ Known limitation when using JIT with lax.cond")
    
#     # Test 5: Edge case - exactly minimum data
#     print("\n[Test 5] Edge case - exactly minimum data")
#     model_exact = SeasonalWindowAverage(season_length=4, window_size=2)
#     y_exact = jnp.array([1., 2., 3., 4., 5., 6., 7., 8.])  # Exactly 4*2=8 observations
    
#     model_exact.fit(y_exact)
#     preds_exact = model_exact.predict(h=4)
    
#     # Should average last 2 cycles: [1,2,3,4] and [5,6,7,8] → [3,4,5,6]
#     expected = (jnp.array([1., 2., 3., 4.]) + jnp.array([5., 6., 7., 8.])) / 2
#     assert jnp.allclose(preds_exact["mean"], expected), "Should average exactly 2 cycles"
#     print(f"  Input: {y_exact}")
#     print(f"  Expected avg: {expected}")
#     print(f"  Predictions: {preds_exact['mean']}")
#     print("  ✓ Exactly minimum data handled correctly")
    
#     # Test 6: predict_in_sample raises NotImplementedError
#     print("\n[Test 6] predict_in_sample raises NotImplementedError")
#     model_no_fitted = SeasonalWindowAverage(season_length=7, window_size=2)
#     model_no_fitted.fit(y_train)
    
#     try:
#         model_no_fitted.predict_in_sample()
#         assert False, "Should have raised NotImplementedError"
#     except NotImplementedError as e:
#         print(f"  Correctly raised: {type(e).__name__}")
#         print("  ✓ predict_in_sample correctly raises NotImplementedError")
    
#     # Test 7: forecast with fitted=True raises NotImplementedError
#     print("\n[Test 7] forecast with fitted=True raises NotImplementedError")
#     try:
#         SeasonalWindowAverage(season_length=7, window_size=2).forecast(
#             y=y_train, h=7, fitted=True
#         )
#         assert False, "Should have raised NotImplementedError"
#     except NotImplementedError as e:
#         print(f"  Correctly raised: {type(e).__name__}")
#         print("  ✓ forecast with fitted=True correctly raises NotImplementedError")
    
#     # Test 8: new() method (shallow copy)
#     print("\n[Test 8] new() method (shallow copy)")
#     model_original = SeasonalWindowAverage(season_length=7, window_size=2, alias="Original")
#     model_original.fit(y_train)
#     model_copy = model_original.new()
    
#     assert model_copy is not model_original, "new() should return a different object"
#     assert model_copy.alias == model_original.alias, "Alias should be copied"
#     assert model_copy.season_length == model_original.season_length, "season_length should be copied"
#     assert model_copy.window_size == model_original.window_size, "window_size should be copied"
#     assert model_copy.model_ is model_original.model_, "model_ should be shallow-copied (same reference)"
    
#     # Modify copy's model_ and verify it affects original (shallow copy behavior)
#     model_copy.model_["test_key"] = "test_value"
#     assert "test_key" in model_original.model_, "Shallow copy shares nested references"
    
#     print(f"  Original alias: {model_original.alias}")
#     print(f"  Copy alias: {model_copy.alias}")
#     print(f"  Shallow copy verified: nested dict shared")
#     print("  ✓ new() method works correctly")
    
#     # Test 9: Custom alias
#     print("\n[Test 9] Custom alias")
#     custom_model = SeasonalWindowAverage(
#         season_length=12, 
#         window_size=4, 
#         alias="MyCustomSeasonalModel"
#     )
#     assert custom_model.alias == "MyCustomSeasonalModel", "Should accept custom alias"
#     assert repr(custom_model) == "MyCustomSeasonalModel", "repr should use custom alias"
#     print(f"  Custom alias: {custom_model.alias}")
#     print("  ✓ Custom alias works correctly")
    
#     # Test 10: Different horizon lengths
#     print("\n[Test 10] Different horizon lengths")
#     model_horizon = SeasonalWindowAverage(season_length=5, window_size=3)
#     y_horizon = jnp.tile(jnp.array([1., 2., 3., 4., 5.]), 4)  # 20 observations
#     model_horizon.fit(y_horizon)
    
#     # Test h < season_length
#     preds_short = model_horizon.predict(h=3)
#     assert preds_short["mean"].shape == (3,), "Should handle h < season_length"
    
#     # Test h > season_length (multiple cycles)
#     preds_long = model_horizon.predict(h=12)
#     assert preds_long["mean"].shape == (12,), "Should handle h > season_length"
    
#     # Test h = season_length
#     preds_exact = model_horizon.predict(h=5)
#     assert preds_exact["mean"].shape == (5,), "Should handle h = season_length"
    
#     print(f"  h=3 (< season): shape={preds_short['mean'].shape}")
#     print(f"  h=5 (= season): shape={preds_exact['mean'].shape}")
#     print(f"  h=12 (> season): shape={preds_long['mean'].shape}")
#     print("  ✓ Different horizon lengths handled correctly")
    
#     # Test 11: Seasonal pattern averaging verification
#     print("\n[Test 11] Seasonal pattern averaging")
#     # Create explicit seasonal data to verify averaging logic
#     # Season length = 3, Window size = 2
#     # Cycle 1: [10, 20, 30]
#     # Cycle 2: [12, 22, 32]
#     # Expected average: [11, 21, 31]
    
#     y_verify = jnp.array([10., 20., 30., 12., 22., 32.])
#     model_verify = SeasonalWindowAverage(season_length=3, window_size=2)
#     model_verify.fit(y_verify)
    
#     expected_pattern = jnp.array([11., 21., 31.])
#     assert jnp.allclose(model_verify.model_["mean"], expected_pattern), "Should correctly average seasonal cycles"
    
#     preds_verify = model_verify.predict(h=3)
#     assert jnp.allclose(preds_verify["mean"], expected_pattern), "Predictions should match averaged pattern"
    
#     print(f"  Cycle 1: [10, 20, 30]")
#     print(f"  Cycle 2: [12, 22, 32]")
#     print(f"  Expected avg: {expected_pattern}")
#     print(f"  Actual avg: {model_verify.model_['mean']}")
#     print("  ✓ Seasonal averaging logic verified")
    
#     # Summary
#     print("\n" + "=" * 60)
#     print("✓ All SeasonalWindowAverage tests passed successfully!")
#     print("=" * 60)
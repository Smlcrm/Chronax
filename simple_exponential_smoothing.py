"""
Simple Exponential Smoothing (SES) implementation using JAX.

SES is a weighted average forecasting method where more recent observations receive exponentially higher weights.
The forecast is a flat line at the final smoothed level.

Formula:
    ℓ[t] = α·y[t] + (1-α)·ℓ[t-1]
    ŷ[t+h] = ℓ[T] for all h ≥ 1

Where:
    - ℓ[t] is the smoothed level at time t
    - α ∈ [0,1] is the smoothing parameter (higher α = more weight on recent observations)
    - y[t] is the observed value at time t
    - T is the final observation index

Implementation:
    - Uses jax.lax.fori_loop for efficient recursive computation of fitted values
    - JIT-compiled with static_argnums for h and fitted parameters
    - Integrates with BaseForecaster for conformal prediction intervals

Instance Attributes:
    - alpha: smoothing parameter (0 ≤ α ≤ 1)
    - alias: model identifier string
    - conformal_params: optional ConformalIntervals object for prediction intervals
    - model_: dict storing fitted values and conformity scores post-fit

Methods:
    - fit(y, X): fits the model and computes conformity scores if conformal_params is set
    - forecast(y, h, X, X_future): pure forecast function for conformity score computation
    - predict(h, X, level): generates h-step ahead predictions with optional conformal intervals
    - predict_in_sample(): returns fitted values from training data
"""

import jax
import jax.numpy as jnp
from functools import partial as _partial

from base_forecaster import BaseForecaster
from conformal_intervals import ConformalIntervals
import utils


# ============================================================================
# Core JAX Functions
# ============================================================================


@_partial(jax.jit, static_argnums=(2, 3))
def _ses(y: jnp.ndarray, alpha: float, h: int, fitted: bool) -> dict:
    """
    Simple Exponential Smoothing core logic.
    JAX equivalent of statsforecast.models._ses()
    
    Args:
        y: Time series, shape (n,)
        alpha: Smoothing parameter in [0,1]
        h: Forecast horizon (static)
        fitted: Include fitted values (static)
        
    Returns:
        Dict with "mean" and optionally "fitted"
    """
    fcst, fitted_vals = utils._ses_forecast(y, alpha)
    out = {"mean": utils._repeat_val(val=fcst, h=h)}
    if fitted:
        out["fitted"] = fitted_vals
    return out


# ============================================================================
# Model Class
# ============================================================================

class SimpleExponentialSmoothing(BaseForecaster):
    """
    JAX-optimized Simple Exponential Smoothing.
    
    Weighted average of past observations with exponentially decreasing weights.
    Formula: ŷ[t+1] = α*y[t] + (1-α)*ŷ[t]
    
    Args:
        alpha: Smoothing parameter in [0,1]
        alias: Model name
        conformal_params: ConformalIntervals for prediction intervals
    """
    
    def __init__(
        self,
        alpha: float,
        alias: str = "SES",
        conformal_params: ConformalIntervals | None = None,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        self.alpha = alpha
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = {}
    
    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "SimpleExponentialSmoothing":
        r"""Fit the SimpleExponentialSmoothing model.

        Runs SES on the full series to produce fitted values and the final
        smoothed level (used as the flat forecast). If `conformal_params` is
        configured, conformity scores are computed and cached for predict().

        Args:
            y (jnp.ndarray): Clean time series of shape (t,).
            X (jnp.ndarray | None): Exogenous variables (unused; included for
                API compatibility). Default is None.

        Returns:
            SimpleExponentialSmoothing: Self (fitted model instance).
        """
        y = utils.ensure_float(y)
        mod = _ses(y=y, alpha=self.alpha, h=1, fitted=True)
        self.model_ = dict(mod)
        
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=X)
            self.model_["_cs"] = cs
        
        return self
    
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
    ) -> dict:
        r"""Memory-efficient stateless fit+predict in one call.

        Fits SES on `y` and immediately generates h-step ahead point forecasts
        without storing any model state. Used internally by BaseForecaster for
        conformity score computation in cross-validation windows.

        Args:
            y (jnp.ndarray): Clean time series of shape (t,).
            h (int): Forecast horizon (number of steps ahead).
            X (jnp.ndarray | None): In-sample exogenous variables (unused;
                included for API compatibility). Default is None.
            X_future (jnp.ndarray | None): Future exogenous variables (unused;
                included for API compatibility). Default is None.

        Returns:
            dict: Dictionary containing:
                - "mean": Point forecasts of shape (h,), all equal to the final smoothed level.
        """
        y = utils.ensure_float(y)
        mod = _ses(y=y, alpha=self.alpha, h=h, fitted=False)
        return {"mean": mod["mean"]}
    
    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int] | None = None,
    ) -> dict:
        r"""Generate h-step ahead forecasts using the fitted model.

        All h forecasts equal the final smoothed level ℓ[T]. Optionally adds
        conformal prediction intervals using cached conformity scores from fit().

        Args:
            h (int): Forecast horizon (number of steps ahead).
            X (jnp.ndarray | None): Exogenous variables (unused; included for
                API compatibility). Default is None.
            level (list[int] | None): Confidence levels (0–100) for prediction
                intervals, e.g. [80, 95]. Requires `conformal_params` to be set.
                Default is None.

        Returns:
            dict: Dictionary containing:
                - "mean": Point forecasts of shape (h,).
                - "lo-{l}" / "hi-{l}": Conformal interval bounds for each level l
                  (only present when level is not None).

        Raises:
            ValueError: If level is requested but `conformal_params` is None.
            ValueError: If level is requested but the model has not been fitted yet.
        """
        mean = utils._repeat_val(val=self.model_["mean"][0], h=h)
        res = {"mean": mean}
        
        if level is None:
            return res
        
        if self.conformal_params is None:
            raise ValueError(
                "You must pass `conformal_params` to compute prediction intervals."
            )
        
        cs = self.model_.get("_cs")
        if cs is None:
            raise ValueError("Model must be fitted before computing intervals.")
        
        res = self.add_confidence_intervals(
            fcst=res, cs=cs, level=level, method="conformal_distribution"
        )
        return res
    
    def predict_in_sample(self) -> dict:
        r"""Return in-sample fitted values from the last fit() call.

        Returns:
            dict: Dictionary containing:
                - "fitted": In-sample smoothed predictions of shape (t,).
                  The first value is NaN (no prior level available at t=0).

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if "fitted" not in self.model_:
            raise ValueError("Model must be fitted first.")
        return {"fitted": self.model_["fitted"]}


# ============================================================================
# Test Cases
# ============================================================================

# if __name__ == "__main__":
#     print("=" * 80)
#     print("SIMPLE EXPONENTIAL SMOOTHING - COMPREHENSIVE TEST SUITE")
#     print("=" * 80)
    
#     # Test 1: Basic functionality with alpha=0.5
#     print("\n[Test 1] Basic SES with alpha=0.5")
#     print("-" * 80)
#     y = jnp.array([10.0, 12.0, 13.0, 15.0, 14.0, 16.0])
#     model = SimpleExponentialSmoothing(alpha=0.5)
#     model.fit(y)
#     pred = model.predict(h=3)
    
#     print(f"Training data: {y}")
#     print(f"Predictions (h=3): {pred['mean']}")
#     print(f"Fitted values: {model.model_['fitted']}")
    
#     # Verify flat forecast
#     assert jnp.allclose(pred['mean'][0], pred['mean'][1]), "SES should produce flat forecast"
#     assert jnp.allclose(pred['mean'][1], pred['mean'][2]), "SES should produce flat forecast"
#     print("✓ Flat forecast verified")
    
#     # Verify first fitted value is NaN
#     assert jnp.isnan(model.model_['fitted'][0]), "First fitted value should be NaN"
#     print("✓ First fitted value is NaN as expected")
    
    
#     # Test 2: High alpha (α=0.9) - more weight on recent observations
#     print("\n[Test 2] High alpha (α=0.9) - recent observations dominate")
#     print("-" * 80)
#     y = jnp.array([5.0, 5.0, 5.0, 5.0, 10.0])
#     model_high = SimpleExponentialSmoothing(alpha=0.9)
#     model_high.fit(y)
#     pred_high = model_high.predict(h=1)
    
#     print(f"Training data: {y}")
#     print(f"Prediction with α=0.9: {pred_high['mean'][0]:.4f}")
    
#     # With high alpha, forecast should be close to last observation
#     assert pred_high['mean'][0] > 9.0, "High alpha should weight recent observation heavily"
#     print("✓ High alpha correctly weights recent observations")
    
    
#     # Test 3: Low alpha (α=0.1) - more smoothing
#     print("\n[Test 3] Low alpha (α=0.1) - heavy smoothing")
#     print("-" * 80)
#     model_low = SimpleExponentialSmoothing(alpha=0.1)
#     model_low.fit(y)
#     pred_low = model_low.predict(h=1)
    
#     print(f"Training data: {y}")
#     print(f"Prediction with α=0.1: {pred_low['mean'][0]:.4f}")
    
#     # With low alpha, forecast should be smoother (closer to historical average)
#     assert pred_low['mean'][0] < pred_high['mean'][0], "Low alpha should smooth more than high alpha"
#     print("✓ Low alpha produces more smoothed forecast")
    
    
#     # Test 4: Conformal prediction intervals (SKIPPED - requires base_forecaster fix)
#     print("\n[Test 4] Conformal prediction intervals")
#     print("-" * 80)
#     print("⚠ SKIPPED: Requires base_forecaster.py to use lax.dynamic_slice for JAX compatibility")
#     print("  This is a known limitation with dynamic slicing in vmapped functions.")
    
    
#     # Test 5: Edge case - alpha=0 (no update)
#     print("\n[Test 5] Edge case: alpha=0 (pure inertia)")
#     print("-" * 80)
#     y = jnp.array([5.0, 10.0, 15.0, 20.0])
#     model_zero = SimpleExponentialSmoothing(alpha=0.0)
#     model_zero.fit(y)
#     pred_zero = model_zero.predict(h=2)
    
#     print(f"Training data: {y}")
#     print(f"Prediction with α=0: {pred_zero['mean'][0]:.4f}")
#     print(f"Expected (first observation): {y[0]:.4f}")
    
#     # With alpha=0, all fitted values should equal first observation
#     assert jnp.allclose(pred_zero['mean'][0], y[0], atol=1e-5), "Alpha=0 should maintain initial level"
#     print("✓ Alpha=0 correctly maintains initial level")
    
    
#     # Test 6: Edge case - alpha=1 (naive forecast)
#     print("\n[Test 6] Edge case: alpha=1 (naive/last value)")
#     print("-" * 80)
#     y = jnp.array([5.0, 10.0, 15.0, 20.0])
#     model_one = SimpleExponentialSmoothing(alpha=1.0)
#     model_one.fit(y)
#     pred_one = model_one.predict(h=2)
    
#     print(f"Training data: {y}")
#     print(f"Prediction with α=1: {pred_one['mean'][0]:.4f}")
#     print(f"Expected (last observation): {y[-1]:.4f}")
    
#     # With alpha=1, forecast should equal last observation
#     assert jnp.allclose(pred_one['mean'][0], y[-1], atol=1e-5), "Alpha=1 should forecast last value"
#     print("✓ Alpha=1 correctly forecasts last observation")
    
    
#     # Test 7: Validation - invalid alpha
#     print("\n[Test 7] Validation: invalid alpha values")
#     print("-" * 80)
#     try:
#         SimpleExponentialSmoothing(alpha=1.5)
#         assert False, "Should raise ValueError for alpha > 1"
#     except ValueError as e:
#         print(f"✓ Correctly rejected alpha=1.5: {e}")
    
#     try:
#         SimpleExponentialSmoothing(alpha=-0.1)
#         assert False, "Should raise ValueError for alpha < 0"
#     except ValueError as e:
#         print(f"✓ Correctly rejected alpha=-0.1: {e}")
    
    
#     # Test 8: predict_in_sample functionality
#     print("\n[Test 8] In-sample fitted values")
#     print("-" * 80)
#     y = jnp.array([10.0, 11.0, 12.0, 13.0, 14.0])
#     model = SimpleExponentialSmoothing(alpha=0.5)
#     model.fit(y)
#     in_sample = model.predict_in_sample()
    
#     print(f"Training data: {y}")
#     print(f"Fitted values: {in_sample['fitted']}")
    
#     assert len(in_sample['fitted']) == len(y), "Fitted values should match training data length"
#     assert jnp.isnan(in_sample['fitted'][0]), "First fitted value should be NaN"
#     print("✓ In-sample predictions retrieved correctly")
    
    
#     # Test 9: Manual SES calculation verification
#     print("\n[Test 9] Manual calculation verification")
#     print("-" * 80)
#     y = jnp.array([10.0, 12.0, 14.0])
#     alpha = 0.5
#     model = SimpleExponentialSmoothing(alpha=alpha)
#     model.fit(y)
    
#     # Manual calculation:
#     # fitted[0] = NaN
#     # fitted[1] = 0.5 * 10 + 0.5 * 10 = 10.0
#     # fitted[2] = 0.5 * 12 + 0.5 * 10 = 11.0
#     # forecast = 0.5 * 14 + 0.5 * 11 = 12.5
    
#     expected_fitted = jnp.array([jnp.nan, 10.0, 11.0])
#     expected_forecast = 12.5
    
#     print(f"Training data: {y}")
#     print(f"Alpha: {alpha}")
#     print(f"Expected fitted: {expected_fitted}")
#     print(f"Actual fitted: {model.model_['fitted']}")
#     print(f"Expected forecast: {expected_forecast}")
#     print(f"Actual forecast: {model.predict(h=1)['mean'][0]}")
    
#     assert jnp.allclose(model.model_['fitted'][1:], expected_fitted[1:], atol=1e-5), \
#         "Fitted values don't match manual calculation"
#     assert jnp.allclose(model.predict(h=1)['mean'][0], expected_forecast, atol=1e-5), \
#         "Forecast doesn't match manual calculation"
#     print("✓ Manual calculation matches JAX implementation")
    
    
#     print("\n" + "=" * 80)
#     print("ALL TESTS PASSED ✓")
#     print("=" * 80)
#     print("\nTest Summary:")
#     print("  [1] Basic SES functionality with flat forecast")
#     print("  [2] High alpha (0.9) weights recent observations heavily")
#     print("  [3] Low alpha (0.1) produces smoothed forecasts")
#     print("  [4] Conformal prediction intervals (SKIPPED - base_forecaster needs lax.dynamic_slice)")
#     print("  [5] Alpha=0 edge case (pure inertia)")
#     print("  [6] Alpha=1 edge case (naive forecast)")
#     print("  [7] Input validation for invalid alpha values")
#     print("  [8] In-sample fitted values retrieval")
#     print("  [9] Manual calculation verification against JAX implementation")
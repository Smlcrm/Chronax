import jax
import jax.numpy as jnp
from jax import lax
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals
from chronax import utils

__all__ = ['TSB']

class TSB(BaseForecaster):
    uses_exog = False

    def __init__(
        self,
        alpha_d: float,
        alpha_p: float,
        alias: str = "TSB",
        conformal_params: ConformalIntervals | None = None,
    ):
        self._validate_parameters(alpha_d, alpha_p)
        self.alpha_d = alpha_d
        self.alpha_p = alpha_p
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = None

    @staticmethod
    def _validate_parameters(alpha_d: float, alpha_p: float) -> None:
        if not isinstance(alpha_d, (int, float)) or alpha_d <= 0 or alpha_d >= 1:
            raise ValueError(f"alpha_d must be in range (0, 1), got {alpha_d}")
        if not isinstance(alpha_p, (int, float)) or alpha_p <= 0 or alpha_p >= 1:
            raise ValueError(f"alpha_p must be in range (0, 1), got {alpha_p}")

    @staticmethod
    def _tsb_core(y: jnp.ndarray, h: int, alpha_d: float, alpha_p: float, fitted: bool = True) -> dict:
        """TSB algorithm: exponential smoothing on demand probability and magnitude."""
        n = len(y)
        prob_indicators = utils.extract_probability(y)

        # Demand level initializes at the FIRST positive demand (the SES over
        # compact demands starts there); a fixed placeholder init would leave
        # a (1-alpha_d)^cnt * (init - d1) residue in the final level.
        demand_init = jnp.where(jnp.any(y > 0), y[jnp.argmax(y > 0)], 0.0)
        prob_init = prob_indicators[0]

        def tsb_step(carry, inputs):
            demand_level, prob_level = carry
            y_t, prob_indicator = inputs

            new_prob = prob_level + alpha_p * (prob_indicator - prob_level)
            new_demand = jnp.where(y_t > 0, demand_level + alpha_d * (y_t - demand_level), demand_level)

            return (new_demand, new_prob), (prob_level, demand_level)

        (final_demand, final_prob), (prob_fitted, demand_fitted) = lax.scan(
            tsb_step, (demand_init, prob_init), (y, prob_indicators)
        )

        # First fitted value is undefined (no prior observation) — NaN, which
        # the nansum inside calculate_sigma excludes; denominator is n.
        fitted_vals = prob_fitted * demand_fitted
        fitted_vals = fitted_vals.at[0].set(jnp.nan)
        residuals = y - fitted_vals
        sigma = utils.calculate_sigma(residuals, n)
        forecasts = jnp.full(h, final_prob * final_demand, dtype=y.dtype)

        result = {
            'mean': forecasts,
            'demand_level': final_demand,
            'prob_level': final_prob,
            'sigma': sigma,
        }
        if fitted:
            result['fitted'] = fitted_vals
        return result

    _tsb_core_jit = jax.jit(_tsb_core.__func__, static_argnums=(1, 2, 3, 4))

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None):
        y = utils.ensure_float(y)
        mod = TSB._tsb_core_jit(y, h=1, alpha_d=self.alpha_d, alpha_p=self.alpha_p, fitted=True)
        mod['y_train'] = y
        self.model_ = mod
        return self

    def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int] | None = None):
        if self.model_ is None:
            raise ValueError("Model must be fitted before calling predict()")

        mean = jnp.full(h, self.model_['prob_level'] * self.model_['demand_level'], dtype=jnp.float32)
        res = {"mean": mean}

        if level is not None:
            level = sorted(level)
            if self.conformal_params is None:
                raise ValueError(
                    "You must instantiate the class with `conformal_params` "
                    "to calculate prediction intervals"
                )
            cs = self.conformity_scores(y=self.model_['y_train'], X=X)
            res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
        return res

    def predict_in_sample(self, level: list[int] | None = None):
        if self.model_ is None:
            raise ValueError("Model must be fitted before calling predict_in_sample()")

        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            res = self._add_fitted_intervals(res, self.model_["sigma"], level)
        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int] | None = None,
        fitted: bool = False,
    ):
        y = utils.ensure_float(y)
        out = TSB._tsb_core_jit(y=y, h=h, alpha_d=self.alpha_d, alpha_p=self.alpha_p, fitted=fitted)
        res = {"mean": out["mean"]}

        if fitted:
            res["fitted"] = out["fitted"]

        if level is not None:
            level = sorted(level)
            if self.conformal_params is None:
                raise ValueError(
                    "You must instantiate the class with `conformal_params` "
                    "to calculate prediction intervals"
                )
            cs = self.conformity_scores(y=y, X=X)
            res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
            if fitted:
                res = self._add_fitted_intervals(res, out["sigma"], level)
        return res

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int] | None = None,
        fitted: bool = False,
    ):
        return self.forecast(y=utils.ensure_float(y), h=h, X=X, X_future=X_future, level=level, fitted=fitted)

    def _add_fitted_intervals(self, res: dict, sigma: float, level: list[int]) -> dict:
        """Add constant-width intervals to fitted values."""
        for lv in sorted(level):
            z = utils._jax_norm_ppf(0.5 + lv / 200)
            res[f"fitted-lo-{lv}"] = res["fitted"] - z * sigma
            res[f"fitted-hi-{lv}"] = res["fitted"] + z * sigma
        return res


# test cases 

if __name__ == '__main__':

    import numpy as np

    passed = 0
    failed = 0

    # Test 1: Basic fit/predict workflow with intermittent demand
    print("\n[Test 1] Basic fit/predict with intermittent demand")
    try:
        y = jnp.array([0, 5, 0, 0, 3, 0, 2, 0, 0, 4, 0, 0, 3], dtype=jnp.float32)
        model = TSB(alpha_d=0.8, alpha_p=0.9)
        model.fit(y)
        result = model.predict(h=5)

        assert 'mean' in result
        assert len(result['mean']) == 5
        assert jnp.all(result['mean'] > 0)  # Forecasts should be positive
        assert jnp.all(jnp.isfinite(result['mean']))  # No NaN or inf

        print(f"Fitted model successfully")
        print(f"Mean forecast: {float(result['mean'][0]):.4f}")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 2: Prediction intervals (native/analytical)
    print("\n[Test 2] Native prediction intervals")
    try:
        y = jnp.array([0, 5, 0, 0, 3, 0, 2, 0, 0, 4, 0, 0, 3], dtype=jnp.float32)
        model = TSB(alpha_d=0.8, alpha_p=0.9)
        model.fit(y)
        result = model.predict(h=5, level=[80, 95])

        assert 'lo-80' in result
        assert 'hi-80' in result
        assert 'lo-95' in result
        assert 'hi-95' in result

        # Check interval ordering at each forecast horizon
        for i in range(5):
            assert result['lo-95'][i] < result['lo-80'][i]
            assert result['lo-80'][i] < result['mean'][i]
            assert result['mean'][i] < result['hi-80'][i]
            assert result['hi-80'][i] < result['hi-95'][i]

        # Check intervals widen over time
        assert result['hi-95'][-1] - result['lo-95'][-1] > result['hi-95'][0] - result['lo-95'][0]

        print(f"Intervals generated correctly")
        print(f"95% CI at h=1: [{float(result['lo-95'][0]):.4f}, {float(result['hi-95'][0]):.4f}]")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 3: Conformal prediction intervals
    print("\n[Test 3] Conformal prediction intervals")
    try:
        y = jnp.array([0, 5, 0, 0, 3, 0, 2, 0, 0, 4, 0, 0, 3, 0, 5, 0, 0, 2], dtype=jnp.float32)
        conformal_config = ConformalIntervals(n_windows=3, h=2)
        model = TSB(alpha_d=0.8, alpha_p=0.9, conformal_params=conformal_config)
        result = model.forecast(y=y, h=3, level=[80, 95])

        assert 'lo-80' in result
        assert 'hi-95' in result
        print(f"Conformal intervals generated")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 4: predict_in_sample with fitted intervals
    print("\n[Test 4] predict_in_sample with intervals")
    try:
        y = jnp.array([0, 5, 0, 0, 3, 0, 2, 0, 0, 4], dtype=jnp.float32)
        model = TSB(alpha_d=0.8, alpha_p=0.9)
        model.fit(y)
        result = model.predict_in_sample(level=[95])

        assert 'fitted' in result
        assert 'fitted-lo-95' in result
        assert 'fitted-hi-95' in result
        assert len(result['fitted']) == len(y)

        print(f"Fitted values with intervals returned")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 5: forecast() method (stateless)
    print("\n[Test 5] forecast() stateless method")
    try:
        y = jnp.array([0, 5, 0, 0, 3, 0, 2, 0, 0, 4], dtype=jnp.float32)
        model = TSB(alpha_d=0.8, alpha_p=0.9)
        result = model.forecast(y=y, h=3, level=[80], fitted=True)

        assert 'mean' in result
        assert 'fitted' in result
        assert 'lo-80' in result
        assert len(result['mean']) == 3
        assert len(result['fitted']) == len(y)

        print(f"Stateless forecast with fitted values")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 6: forward() method (apply to new data)
    print("\n[Test 6] forward() method")
    try:
        y1 = jnp.array([0, 5, 0, 0, 3, 0, 2], dtype=jnp.float32)
        y2 = jnp.array([0, 4, 0, 0, 6, 0, 1, 0, 3], dtype=jnp.float32)
        model = TSB(alpha_d=0.8, alpha_p=0.9)
        model.fit(y1)
        result = model.forward(y=y2, h=3)

        assert 'mean' in result
        assert len(result['mean']) == 3

        print(f"forward() applied to new data")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 7: All-zero time series edge case
    print("\n[Test 7] All-zero time series")
    try:
        y = jnp.zeros(10, dtype=jnp.float32)
        model = TSB(alpha_d=0.8, alpha_p=0.9)
        model.fit(y)
        result = model.predict(h=3)

        assert jnp.all(result['mean'] == 0)
        print(f"All-zero series handled correctly")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 8: No-zero time series (continuous demand)
    print("\n[Test 8] No-zero time series (continuous demand)")
    try:
        y = jnp.array([5, 3, 4, 6, 2, 4, 5, 3, 4], dtype=jnp.float32)
        model = TSB(alpha_d=0.8, alpha_p=0.9)
        result = model.forecast(y=y, h=3)

        assert jnp.all(result['mean'] > 0)
        assert jnp.all(jnp.isfinite(result['mean']))

        print(f"Continuous demand handled correctly")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 9: Parameter validation
    print("\n[Test 9] Parameter validation")
    try:
        # Test invalid alpha_d
        try:
            model = TSB(alpha_d=1.5, alpha_p=0.5)
            print(f"  ✗ Failed to reject alpha_d > 1")
            failed += 1
        except ValueError:
            print(f"Rejected alpha_d > 1")

        # Test invalid alpha_p
        try:
            model = TSB(alpha_d=0.5, alpha_p=0.0)
            print(f"Failed to reject alpha_p = 0")
            failed += 1
        except ValueError:
            print(f"Rejected alpha_p = 0")

        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 10: Different smoothing parameters
    print("\n[Test 10] Different smoothing parameter combinations")
    try:
        y = jnp.array([0, 5, 0, 0, 3, 0, 2, 0, 0, 4], dtype=jnp.float32)

        # High smoothing (responsive)
        model1 = TSB(alpha_d=0.9, alpha_p=0.9)
        result1 = model1.forecast(y=y, h=1)

        # Low smoothing (stable)
        model2 = TSB(alpha_d=0.1, alpha_p=0.1)
        result2 = model2.forecast(y=y, h=1)

        # Forecasts should differ
        assert result1['mean'][0] != result2['mean'][0]

        print(f"High smoothing forecast: {float(result1['mean'][0]):.4f}")
        print(f"Low smoothing forecast: {float(result2['mean'][0]):.4f}")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 11: Single non-zero value
    print("\n[Test 11] Single non-zero value")
    try:
        y = jnp.array([0, 0, 5, 0, 0, 0], dtype=jnp.float32)
        model = TSB(alpha_d=0.8, alpha_p=0.9)
        result = model.forecast(y=y, h=2)

        assert jnp.all(jnp.isfinite(result['mean']))
        assert jnp.all(result['mean'] >= 0)

        print(f"Single non-zero value handled")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # Test 12: Long intermittent series
    print("\n[Test 12] Long intermittent series")
    try:
        # Create a realistic intermittent demand pattern
        np.random.seed(42)
        n = 100
        prob = 0.3
        y_np = np.where(
            np.random.rand(n) < prob,
            np.random.poisson(5, size=n),
            0
        ).astype(np.float32)
        y = jnp.array(y_np)

        model = TSB(alpha_d=0.7, alpha_p=0.8)
        model.fit(y)
        result = model.predict(h=10, level=[80, 95])

        assert jnp.all(jnp.isfinite(result['mean']))
        assert 'lo-80' in result
        assert 'hi-95' in result

        print(f"Long series processed successfully")
        print(f"Mean forecast: {float(result['mean'][0]):.4f}")
        passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        failed += 1

    # summary
    print(f"Test Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("tests passed!")
    else:
        print(f"{failed} test(s) failed")

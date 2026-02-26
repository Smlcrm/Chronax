import jax.numpy as jnp
from chronax.utils import ConformalIntervals
from chronax.models.base_forecaster import BaseForecaster
from chronax.models import HoltWinters

def test_holt_winters():
    """Comprehensive test suite for Holt-Winters model."""
    print("="*60)
    print("Running Holt-Winters model tests...")
    print("="*60)
    passed = 0
    failed = 0

    # Test 1: Basic fit/predict (AAA Model)
    try:
        print("\nTest 1: Basic fit/predict (AAA Model)")
        # Create seasonal data: 3 years of monthly data with trend and seasonality
        import numpy as np
        np.random.seed(42)
        t = np.arange(36)
        trend = 100 + 2 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        y = jnp.array(trend + seasonal + np.random.normal(0, 2, 36), dtype=jnp.float32)

        model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
        model.fit(y)
        result = model.predict(h=12)

        assert 'mean' in result, "Result should contain 'mean' key"
        assert len(result['mean']) == 12, "Forecast length should be 12"
        assert jnp.all(jnp.isfinite(result['mean'])), "Forecasts should be finite"

        print(f"  Fitted alpha: {model.model_['alpha']:.4f}")
        print(f"  Fitted beta: {model.model_['beta']:.4f}")
        print(f"  Fitted gamma: {model.model_['gamma']:.4f}")
        print(f"  First 3 forecasts: {result['mean'][:3]}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 2: MAM Model (Multiplicative error, Additive trend, Multiplicative seasonality)
    try:
        print("\nTest 2: MAM Model")
        import numpy as np
        np.random.seed(42)
        t = np.arange(36)
        trend = 100 + 2 * t
        seasonal = 1 + 0.1 * np.sin(2 * np.pi * t / 12)
        y = jnp.array(trend * seasonal, dtype=jnp.float32)

        model = HoltWinters(season_length=12, error_type='M', season_type='M', damped=False)
        model.fit(y)
        result = model.predict(h=12)

        assert 'mean' in result, "Result should contain 'mean' key"
        assert len(result['mean']) == 12, "Forecast length should be 12"

        print(f"  Fitted alpha: {model.model_['alpha']:.4f}")
        print(f"  Fitted beta: {model.model_['beta']:.4f}")
        print(f"  Fitted gamma: {model.model_['gamma']:.4f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 3: AAM Model (Additive error, Additive trend, Multiplicative seasonality)
    try:
        print("\nTest 3: AAM Model")
        import numpy as np
        np.random.seed(42)
        t = np.arange(36)
        trend = 100 + 2 * t
        seasonal = 1 + 0.1 * np.sin(2 * np.pi * t / 12)
        y = jnp.array(trend * seasonal, dtype=jnp.float32)

        model = HoltWinters(season_length=12, error_type='A', season_type='M', damped=False)
        model.fit(y)
        result = model.predict(h=12)

        assert 'mean' in result, "Should have forecasts"
        assert len(result['mean']) == 12, "Forecast length should be 12"

        print(f"  Model fitted successfully")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 4: MAA Model (Multiplicative error, Additive trend, Additive seasonality)
    try:
        print("\nTest 4: MAA Model")
        import numpy as np
        np.random.seed(42)
        t = np.arange(36)
        trend = 100 + 2 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        y = jnp.array(trend + seasonal, dtype=jnp.float32)

        model = HoltWinters(season_length=12, error_type='M', season_type='A', damped=False)
        model.fit(y)
        result = model.predict(h=12)

        assert 'mean' in result, "Should have forecasts"
        assert len(result['mean']) == 12, "Forecast length should be 12"

        print(f"  Model fitted successfully")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 5: Damped Trend
    try:
        print("\nTest 5: Damped Trend")
        import numpy as np
        np.random.seed(42)
        t = np.arange(36)
        trend = 100 + 2 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        y = jnp.array(trend + seasonal, dtype=jnp.float32)

        # Non-damped model
        model_nodamp = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
        model_nodamp.fit(y)
        result_nodamp = model_nodamp.predict(h=24)

        # Damped model
        model_damp = HoltWinters(season_length=12, error_type='A', season_type='A', damped=True, phi=0.9)
        model_damp.fit(y)
        result_damp = model_damp.predict(h=24)

        # At long horizons, damped should be lower (due to dampening trend)
        last_forecast_damp = result_damp['mean'][-1]
        last_forecast_nodamp = result_nodamp['mean'][-1]

        assert last_forecast_damp < last_forecast_nodamp, \
            "Damped forecast should be lower at long horizons"

        print(f"  Non-damped h=24: {last_forecast_nodamp:.2f}")
        print(f"  Damped h=24: {last_forecast_damp:.2f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 6: Prediction Intervals
    try:
        print("\nTest 6: Prediction Intervals")
        import numpy as np
        np.random.seed(42)
        t = np.arange(36)
        trend = 100 + 2 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        y = jnp.array(trend + seasonal, dtype=jnp.float32)

        model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
        model.fit(y)
        result = model.predict(h=12, level=[80, 95])

        assert 'lo-80' in result, "Should have lo-80"
        assert 'hi-80' in result, "Should have hi-80"
        assert 'lo-95' in result, "Should have lo-95"
        assert 'hi-95' in result, "Should have hi-95"

        # Check ordering for first forecast
        assert result['lo-95'][0] < result['lo-80'][0], "lo-95 < lo-80"
        assert result['lo-80'][0] < result['mean'][0], "lo-80 < mean"
        assert result['mean'][0] < result['hi-80'][0], "mean < hi-80"
        assert result['hi-80'][0] < result['hi-95'][0], "hi-80 < hi-95"

        print(f"  h=1: [{result['lo-95'][0]:.2f}, {result['mean'][0]:.2f}, {result['hi-95'][0]:.2f}]")
        print(f"  h=12: [{result['lo-95'][11]:.2f}, {result['mean'][11]:.2f}, {result['hi-95'][11]:.2f}]")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 7: Different Season Lengths
    try:
        print("\nTest 7: Different Season Lengths")

        # Quarterly data (season_length=4)
        import numpy as np
        np.random.seed(42)
        t = np.arange(28)  # 7 years quarterly
        trend = 100 + 2 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 4)
        y_quarterly = jnp.array(trend + seasonal, dtype=jnp.float32)

        model_q = HoltWinters(season_length=4, error_type='A', season_type='A')
        model_q.fit(y_quarterly)
        result_q = model_q.predict(h=4)
        assert len(result_q['mean']) == 4, "Should forecast 4 quarters"

        # Weekly data (season_length=7)
        t = np.arange(35)  # 5 weeks
        trend = 100 + t
        seasonal = 3 * np.sin(2 * np.pi * t / 7)
        y_weekly = jnp.array(trend + seasonal, dtype=jnp.float32)

        model_w = HoltWinters(season_length=7, error_type='A', season_type='A')
        model_w.fit(y_weekly)
        result_w = model_w.predict(h=7)
        assert len(result_w['mean']) == 7, "Should forecast 7 days"

        print(f"  Quarterly (season=4): ✓")
        print(f"  Weekly (season=7): ✓")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 8: predict_in_sample()
    try:
        print("\nTest 8: predict_in_sample()")
        import numpy as np
        np.random.seed(42)
        t = np.arange(36)
        trend = 100 + 2 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        y = jnp.array(trend + seasonal, dtype=jnp.float32)

        model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
        model.fit(y)
        result = model.predict_in_sample(level=[95])

        assert 'fitted' in result, "Should have fitted values"
        assert len(result['fitted']) == len(y), "Fitted should match training length"
        assert 'fitted-lo-95' in result, "Should have fitted intervals"
        assert 'fitted-hi-95' in result, "Should have fitted intervals"

        print(f"  Fitted length: {len(result['fitted'])}")
        print(f"  First fitted value: {result['fitted'][0]:.2f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 9: forecast() Method
    try:
        print("\nTest 9: forecast() Method (Stateless)")
        import numpy as np
        np.random.seed(42)
        t = np.arange(36)
        trend = 100 + 2 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        y = jnp.array(trend + seasonal, dtype=jnp.float32)

        model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
        # Don't call fit()
        result = model.forecast(y, h=12, fitted=True, level=[95])

        assert 'mean' in result, "Should have forecasts"
        assert 'fitted' in result, "Should have fitted values with fitted=True"
        assert len(result['mean']) == 12, "Forecast length should be 12"
        assert len(result['fitted']) == len(y), "Fitted length should match y"

        print(f"  Forecast h=1: {result['mean'][0]:.2f}")
        print(f"  Includes fitted values and intervals")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 10: forward() Method
    try:
        print("\nTest 10: forward() Method")
        import numpy as np
        np.random.seed(42)
        t = np.arange(36)
        trend = 100 + 2 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        y1 = jnp.array(trend + seasonal, dtype=jnp.float32)

        # Different scale
        trend2 = 200 + 3 * t
        seasonal2 = 15 * np.sin(2 * np.pi * t / 12)
        y2 = jnp.array(trend2 + seasonal2, dtype=jnp.float32)

        model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
        model.fit(y1)

        # Apply to new series
        result = model.forward(y2, h=6)

        assert 'mean' in result, "Should have forecasts"
        assert len(result['mean']) == 6, "Forecast length should be 6"
        # Forecasts should follow y2 scale, not y1
        assert result['mean'][0] > 200, "Forecast should follow y2 scale"

        print(f"  y1 range: [{float(y1.min()):.1f}, {float(y1.max()):.1f}]")
        print(f"  y2 range: [{float(y2.min()):.1f}, {float(y2.max()):.1f}]")
        print(f"  Forward forecast: {result['mean'][0]:.2f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 11: Edge Cases
    try:
        print("\nTest 11: Edge Cases")

        # Test minimum data length (should work with season_length observations)
        import numpy as np
        y_min = jnp.array(np.arange(12, dtype=np.float32) + 100)
        model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
        model.fit(y_min)
        result = model.predict(h=4)
        assert len(result['mean']) == 4, "Should work with season_length observations"

        # Test that insufficient data fails
        y_fail = jnp.array(np.arange(10, dtype=np.float32))
        model2 = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
        try:
            model2.fit(y_fail)
            raise AssertionError("Should raise ValueError for len(y) < season_length")
        except ValueError:
            pass  # Expected

        print(f"  Minimum length (season_length=12): {len(y_min)} ✓")
        print(f"  Rejects insufficient data ✓")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Summary
    print("\n" + "="*60)
    print(f"Tests passed: {passed}/{passed+failed}")
    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"{failed} test(s) failed.")
    print("="*60)


if __name__ == '__main__':
    test_holt_winters()
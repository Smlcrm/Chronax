import jax.numpy as jnp
from chronax.models import Holt
from chronax.utils import ConformalIntervals

def test_holt():
    """Comprehensive test suite for Holt model."""
    print("="*60)
    print("Running Holt model tests...")
    print("="*60)
    passed = 0
    failed = 0

    # Test 1: Basic fit/predict (Additive Error)
    try:
        print("\nTest 1: Basic fit/predict (Additive Error)")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        model = Holt(error_type='A', damped=False)
        model.fit(y)
        result = model.predict(h=5)

        assert 'mean' in result, "Result should contain 'mean' key"
        assert len(result['mean']) == 5, "Forecast length should be 5"
        assert jnp.all(jnp.isfinite(result['mean'])), "Forecasts should be finite"

        print(f"  Fitted alpha: {model.model_['alpha']:.4f}")
        print(f"  Fitted beta: {model.model_['beta']:.4f}")
        print(f"  First 3 forecasts: {result['mean'][:3]}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 2: Multiplicative Error
    try:
        print("\nTest 2: Multiplicative Error")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        model = Holt(error_type='M', damped=False)
        model.fit(y)
        result = model.predict(h=5)

        assert 'mean' in result, "Result should contain 'mean' key"
        assert len(result['mean']) == 5, "Forecast length should be 5"

        print(f"  Fitted alpha: {model.model_['alpha']:.4f}")
        print(f"  Fitted beta: {model.model_['beta']:.4f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 3: Damped Trend
    try:
        print("\nTest 3: Damped Trend")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])

        # Non-damped model
        model_nodamp = Holt(error_type='A', damped=False)
        model_nodamp.fit(y)
        result_nodamp = model_nodamp.predict(h=10)

        # Damped model
        model_damp = Holt(error_type='A', damped=True, phi=0.9)
        model_damp.fit(y)
        result_damp = model_damp.predict(h=10)

        # At longer horizons, damped should be less than non-damped
        last_forecast_damp = result_damp['mean'][-1]
        last_forecast_nodamp = result_nodamp['mean'][-1]

        assert last_forecast_damp < last_forecast_nodamp, \
            "Damped forecast should be lower at long horizons"

        print(f"  Non-damped h=10: {last_forecast_nodamp:.2f}")
        print(f"  Damped h=10: {last_forecast_damp:.2f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 4: Prediction Intervals (Native)
    try:
        print("\nTest 4: Prediction Intervals (Native)")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        model = Holt(error_type='A', damped=False)
        model.fit(y)
        result = model.predict(h=5, level=[80, 95])

        assert 'lo-80' in result, "Should have lo-80"
        assert 'hi-80' in result, "Should have hi-80"
        assert 'lo-95' in result, "Should have lo-95"
        assert 'hi-95' in result, "Should have hi-95"

        # Check ordering: lo-95 < lo-80 < mean < hi-80 < hi-95
        for i in range(5):
            assert result['lo-95'][i] < result['lo-80'][i], "lo-95 < lo-80"
            assert result['lo-80'][i] < result['mean'][i], "lo-80 < mean"
            assert result['mean'][i] < result['hi-80'][i], "mean < hi-80"
            assert result['hi-80'][i] < result['hi-95'][i], "hi-80 < hi-95"

        print(f"  h=1: [{result['lo-95'][0]:.2f}, {result['mean'][0]:.2f}, {result['hi-95'][0]:.2f}]")
        print(f"  h=5: [{result['lo-95'][4]:.2f}, {result['mean'][4]:.2f}, {result['hi-95'][4]:.2f}]")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 5: Conformal Intervals
    try:
        print("\nTest 5: Conformal Intervals")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0, 48.0, 57.0])
        conformal = ConformalIntervals(n_windows=3, h=2)
        model = Holt(error_type='A', damped=False, conformal_params=conformal)
        model.fit(y)
        result = model.predict(h=2, level=[95])

        assert 'lo-95' in result, "Should have conformal lo-95"
        assert 'hi-95' in result, "Should have conformal hi-95"

        print(f"  Conformal interval h=1: [{result['lo-95'][0]:.2f}, {result['hi-95'][0]:.2f}]")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 6: predict_in_sample()
    try:
        print("\nTest 6: predict_in_sample()")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        model = Holt(error_type='A', damped=False)
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

    # Test 7: forecast() Method
    try:
        print("\nTest 7: forecast() Method (Stateless)")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        model = Holt(error_type='A', damped=False)
        # Don't call fit()
        result = model.forecast(y, h=5, fitted=True, level=[95])

        assert 'mean' in result, "Should have forecasts"
        assert 'fitted' in result, "Should have fitted values with fitted=True"
        assert len(result['mean']) == 5, "Forecast length should be 5"
        assert len(result['fitted']) == len(y), "Fitted length should match y"

        print(f"  Forecast h=1: {result['mean'][0]:.2f}")
        print(f"  Includes fitted values and intervals")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 8: forward() Method
    try:
        print("\nTest 8: forward() Method")
        y1 = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        y2 = jnp.array([100.0, 102.0, 105.0, 108.0, 112.0, 117.0, 123.0, 130.0])

        model = Holt(error_type='A', damped=False)
        model.fit(y1)

        # Apply to new series
        result = model.forward(y2, h=3)

        assert 'mean' in result, "Should have forecasts"
        assert len(result['mean']) == 3, "Forecast length should be 3"
        # Forecasts should be in range of y2, not y1
        assert result['mean'][0] > 100, "Forecast should follow y2 scale"

        print(f"  y1 range: [{float(y1.min()):.1f}, {float(y1.max()):.1f}]")
        print(f"  y2 range: [{float(y2.min()):.1f}, {float(y2.max()):.1f}]")
        print(f"  Forward forecast: {result['mean'][0]:.2f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 9: Edge Cases
    try:
        print("\nTest 9: Edge Cases")

        # Test minimum data length (should work with 2 observations)
        y_min = jnp.array([10.0, 12.0])
        model = Holt(error_type='A', damped=False)
        model.fit(y_min)
        result = model.predict(h=2)
        assert len(result['mean']) == 2, "Should work with 2 observations"

        # Test that 1 observation fails
        y_fail = jnp.array([10.0])
        model2 = Holt(error_type='A', damped=False)
        try:
            model2.fit(y_fail)
            raise AssertionError("Should raise ValueError for len(y) < 2")
        except ValueError:
            pass  # Expected

        print(f"  Minimum length (2): {len(y_min)} ✓")
        print(f"  Rejects length 1 ✓")
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
    test_holt()

def test_m_error_native_interval_width_is_relative_scale():
    # M-error native widths use the relative-residual sigma once multiplied
    # by |mean|; an absolute-residual sigma double-scales by the series level.
    import numpy as np
    import jax.numpy as jnp
    from chronax.models import Holt
    rng = np.random.default_rng(0)
    y = jnp.asarray(500.0 + np.cumsum(rng.normal(0.5, 1.0, 150)), dtype=jnp.float64)
    m = Holt(error_type='M')
    res = m.forecast(y=y, h=5, level=[95])
    width = np.asarray(res["hi-95"]) - np.asarray(res["lo-95"])
    mean = np.asarray(res["mean"])
    # Sane relative width: a few percent of the level, never level-squared.
    assert np.all(width > 0)
    assert np.all(width < 0.5 * np.abs(mean)), (
        f"M-error interval width {width[0]:.1f} vs mean {mean[0]:.1f} — "
        "absolute-sigma double scaling"
    )


def test_fixed_phi_damps_the_recursion_and_the_forecast():
    import numpy as np
    import jax.numpy as jnp
    from chronax.models import Holt
    rng = np.random.default_rng(1)
    y = jnp.asarray(10.0 + 0.8 * np.arange(120) + rng.normal(0, 0.5, 120), dtype=jnp.float64)
    m = Holt(damped=True, phi=0.8)
    res = m.forecast(y=y, h=6)
    fc = np.asarray(res["mean"])
    inc = np.diff(fc)
    ratios = inc[1:] / inc[:-1]
    np.testing.assert_allclose(ratios, 0.8, rtol=1e-5)


def test_multiplicative_error_rejects_nonpositive_data():
    import numpy as np
    import jax.numpy as jnp
    import pytest
    from chronax.models import Holt, HoltWinters
    y = jnp.asarray(np.array([1.0, 2.0, -1.0, 3.0] * 10))
    with pytest.raises(ValueError):
        Holt(error_type='M').fit(y)
    with pytest.raises(ValueError):
        HoltWinters(season_length=4, error_type='M').fit(y)

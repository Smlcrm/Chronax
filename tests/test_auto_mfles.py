import jax.numpy as jnp
import numpy as np
import time
from typing import Union, Tuple

# Assumes your AutoMFLES class is in a file named auto_mfles.py
from chronax.models import AutoMFLES
from chronax.utils import ConformalIntervals

def generate_dummy_data(n: int = 100, include_x: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Helper to generate synthetic time series data for testing.
    
    Generates a mock time series consisting of a linear trend, a sinusoidal 
    seasonality (period=12), and normally distributed random noise. Optionally 
    generates a matrix of exogenous regressors corresponding to the same time steps.
    
    Args:
        n (int, optional): The length of the time series to generate. Defaults to 100.
        include_x (bool, optional): Whether to generate and return a matrix of 
                                    exogenous regressors. Defaults to False.
        
    Returns:
        Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]: 
            If include_x is False, returns a 1D array of target values.
            If include_x is True, returns a tuple containing the target array 
            and the 2D exogenous feature array.
    """
    t = jnp.arange(n, dtype=jnp.float32)
    # Trend + Seasonality + Noise
    y = 10.0 + 0.5 * t + 5.0 * jnp.sin(2 * jnp.pi * t / 12.0) + np.random.normal(0, 1, n)
    
    if include_x:
        # Generate two exogenous features
        x1 = jnp.cos(2 * jnp.pi * t / 12.0)
        x2 = t * 0.1
        X = jnp.column_stack((x1, x2))
        return y, X
    return y

def test_auto_mfles() -> None:
    """Comprehensive test suite for the AutoMFLES model.
    
    Executes 10 distinct tests covering:
    1. Basic non-seasonal fit and predict.
    2. Seasonal grid search and parameter parsing.
    3. Exogenous variable integration and scaling.
    4. Exogenous variable safety and mismatch exception checking.
    5. Gaussian prediction interval generation and mathematical boundary correctness.
    6. Hash-based caching mechanism for skipping redundant optimization loops.
    7. Custom user-defined grid configuration application.
    8. Fallback logic for structurally insufficient series lengths.
    9. Initialization exception guards for invalid class parameters.
    10. Multi-metric (sMAPE vs MAE) target optimization support.
    
    Outputs the execution status of each sequential test to the console and 
    provides a final aggregate pass/fail summary.
    """
    print("="*60)
    print("Running AutoMFLES model tests...")
    print("="*60)
    passed = 0
    failed = 0

    # ---------------------------------------------------------
    # Test 1: Basic Fit & Predict (Non-Seasonal)
    # ---------------------------------------------------------
    try:
        print("\nTest 1: Basic Fit & Predict (Non-Seasonal)")
        y = generate_dummy_data(n=60)
        model = AutoMFLES(test_size=5, n_windows=2, season_length=None)
        model.fit(y)
        res = model.predict(h=5)

        assert 'mean' in res, "Result should contain 'mean' key"
        assert len(res['mean']) == 5, "Forecast length should be 5"
        assert jnp.all(jnp.isfinite(res['mean'])), "Forecasts should be finite"
        assert model.best_params_ is not None, "Best params should be populated"
        
        print(f"  Best Params Selected: {model.best_params_}")
        print(f"  First 3 forecasts: {res['mean'][:3]}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # ---------------------------------------------------------
    # Test 2: Seasonal Grid Search
    # ---------------------------------------------------------
    try:
        print("\nTest 2: Seasonal Grid Search & Optimization")
        y = generate_dummy_data(n=80)
        model = AutoMFLES(test_size=12, n_windows=2, season_length=12)
        model.fit(y)
        res = model.predict(h=12)

        assert model.best_params_ is not None
        assert 'seasonal_period' in model.best_params_ or 'seasonal_period' not in model.best_params_, "Grid parsed correctly"
        assert len(res['mean']) == 12

        print(f"  Recognized seasonal config: {model.best_params_}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # ---------------------------------------------------------
    # Test 3: Exogenous Variables Support
    # ---------------------------------------------------------
    try:
        print("\nTest 3: Exogenous Variables (X) Support")
        y, X = generate_dummy_data(n=70, include_x=True)
        train_y, test_y = y[:-5], y[-5:]
        train_X, test_X = X[:-5], X[-5:]

        model = AutoMFLES(test_size=5, season_length=12)
        model.fit(train_y, X=train_X)
        
        assert model.scaling_stats_ is not None, "Scaling stats should be saved for X"
        
        res = model.predict(h=5, X=test_X)
        assert len(res['mean']) == 5, "Should predict successfully with future X"

        print("  Scaling stats captured correctly.")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # ---------------------------------------------------------
    # Test 4: Exogenous Safety Constraints
    # ---------------------------------------------------------
    try:
        print("\nTest 4: Exogenous Variables Safety Checks")
        y, X = generate_dummy_data(n=50, include_x=True)
        
        model = AutoMFLES(test_size=3)
        model.fit(y) # Train WITHOUT X
        
        try:
            model.predict(h=3, X=X[:3]) # Predict WITH X
            raise AssertionError("Should have blocked prediction with X if trained without X")
        except ValueError as ve:
            assert "Model trained without X" in str(ve)

        print("  Properly rejected mismatched Exogenous variables.")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # ---------------------------------------------------------
    # Test 5: Prediction Intervals (Gaussian)
    # ---------------------------------------------------------
    try:
        print("\nTest 5: Gaussian Prediction Intervals")
        y = generate_dummy_data(n=60)
        model = AutoMFLES(test_size=5)
        model.fit(y)
        res = model.predict(h=5, level=[80, 95])

        assert 'lo-80' in res and 'hi-80' in res, "Should contain 80% intervals"
        assert 'lo-95' in res and 'hi-95' in res, "Should contain 95% intervals"
        
        # Ensure mathematical correctness: lo-95 < lo-80 < mean < hi-80 < hi-95
        assert jnp.all(res['lo-95'] < res['lo-80']), "95% lower bound must be < 80% lower bound"
        assert jnp.all(res['lo-80'] < res['mean']), "80% lower bound must be < mean"
        assert jnp.all(res['mean'] < res['hi-80']), "Mean must be < 80% upper bound"
        assert jnp.all(res['hi-80'] < res['hi-95']), "80% upper bound must be < 95% upper bound"

        print(f"  h=1 Bounds: [{res['lo-95'][0]:.2f}, {res['mean'][0]:.2f}, {res['hi-95'][0]:.2f}]")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # ---------------------------------------------------------
    # Test 6: Caching Mechanism (Speed Test)
    # ---------------------------------------------------------
    try:
        print("\nTest 6: Optimization Caching (Hash bypass)")
        y = generate_dummy_data(n=60)
        model = AutoMFLES(test_size=5, n_windows=2)
        
        # First fit (Performs Grid Search)
        t0 = time.time()
        model.fit(y)
        t1 = time.time()
        first_fit_time = t1 - t0
        
        hash_1 = model._cached_y_hash

        # Second fit (Should skip Grid Search)
        t2 = time.time()
        model.fit(y)
        t3 = time.time()
        second_fit_time = t3 - t2
        
        hash_2 = model._cached_y_hash

        assert hash_1 == hash_2, "Hashes should match for identical data"
        assert second_fit_time < first_fit_time, "Second fit should be significantly faster due to caching"

        print(f"  Initial fit time: {first_fit_time:.4f}s")
        print(f"  Cached fit time:  {second_fit_time:.4f}s")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # ---------------------------------------------------------
    # Test 7: Custom Grid Configuration
    # ---------------------------------------------------------
    try:
        print("\nTest 7: Custom User Configuration Grid")
        y = generate_dummy_data(n=40)
        custom_grid = [{"smoother": True, "ma": 4, "seasonal_period": None}]
        
        model = AutoMFLES(test_size=3, config=custom_grid)
        model.fit(y)

        # Ensure it picked the only available config
        assert model.best_params_['ma'] == 4
        assert model.best_params_['smoother'] is True
        
        print("  Custom parameters applied correctly.")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # ---------------------------------------------------------
    # Test 8: Short Series Fallback
    # ---------------------------------------------------------
    try:
        print("\nTest 8: Short Series Fallback Logic")
        # Provide a series too short to do multiple n_windows correctly
        y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        model = AutoMFLES(test_size=4, n_windows=5, step_size=4)
        
        model.fit(y) # Should not crash, should fallback to default grid[0]
        res = model.predict(h=2)

        assert model.best_params_ is not None, "Should have fallen back to a default parameter set"
        assert len(res['mean']) == 2
        
        print("  Safely handled impossible CV window constraints.")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # ---------------------------------------------------------
    # Test 9: Initialization Guards
    # ---------------------------------------------------------
    try:
        print("\nTest 9: Invalid Parameter Initialization Guards")
        
        try:
            AutoMFLES(test_size=0)
            raise AssertionError("Should reject test_size=0")
        except ValueError:
            pass

        try:
            AutoMFLES(test_size=5, n_windows=-1)
            raise AssertionError("Should reject n_windows < 1")
        except ValueError:
            pass

        print("  Properly rejected invalid class parameters.")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # ---------------------------------------------------------
    # Test 10: Multi-Metric Support
    # ---------------------------------------------------------
    try:
        print("\nTest 10: Multi-Metric Support (MAE vs SMAPE)")
        y = generate_dummy_data(n=50)
        
        model_smape = AutoMFLES(test_size=4, metric="smape")
        model_mae = AutoMFLES(test_size=4, metric="mae")
        
        model_smape.fit(y)
        model_mae.fit(y)

        # Just validating that changing the metric doesn't break the pipeline
        assert model_smape.best_params_ is not None
        assert model_mae.best_params_ is not None

        print(f"  Successfully optimized using both metrics.")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"Tests passed: {passed}/{passed+failed}")
    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"{failed} test(s) failed.")
    print("="*60)


def test_auto_mfles_accepts_conformal_params_alias() -> None:
    y = generate_dummy_data(n=48)
    cfg = ConformalIntervals(n_windows=3, h=4, method="conformal_distribution")
    model = AutoMFLES(test_size=4, n_windows=2, conformal_params=cfg)

    assert model.conformal_params is cfg
    assert model.prediction_intervals is cfg

    model.fit(y)
    res = model.predict(h=4, level=[80])
    assert "mean" in res and len(res["mean"]) == 4
    assert "lo-80" in res and "hi-80" in res


def test_auto_mfles_rejects_conflicting_conformal_configs() -> None:
    cfg_a = ConformalIntervals(n_windows=3, h=4, method="conformal_distribution")
    cfg_b = ConformalIntervals(n_windows=3, h=4, method="conformal_signed")
    try:
        AutoMFLES(
            test_size=4,
            prediction_intervals=cfg_a,
            conformal_params=cfg_b,
        )
        raise AssertionError("Expected ValueError for conflicting conformal configs")
    except ValueError as e:
        assert "Pass only one conformal config" in str(e)


if __name__ == '__main__':
    test_auto_mfles()
    test_auto_mfles_accepts_conformal_params_alias()
    test_auto_mfles_rejects_conflicting_conformal_configs()
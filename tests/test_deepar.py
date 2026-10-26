"""
Test suite for DeepAR JAX/FLAX implementation.

Run with: python -m pytest test_deepar.py
Or directly: python test_deepar.py
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from chronax.models.deepar import (
    DeepAR_EncDec,
    train_model,
    forecast_mc,
    quantiles,
    DeepARForecaster,
    create_batch,
    nll_gaussian,
)


# =============================================================================
# Test 1: Basic shape checking
# =============================================================================


def test_1_basic_shapes():
    """Test 1: Verify model output shapes are correct."""
    print("\n=== Test 1: Basic Shape Checking ===")
    
    model = DeepAR_EncDec(hidden=32, dropout_rate=0.0)
    
    # Setup
    T = 50
    d_f = 3
    d_s = 2
    
    y_hist = jnp.ones((T,), dtype=jnp.float32)
    x_f_all = jnp.ones((T, d_f), dtype=jnp.float32)
    x_static = jnp.ones((d_s,), dtype=jnp.float32)
    
    key = random.PRNGKey(0)
    params = model.init(
        {'params': key, 'dropout': key},
        y_seq=y_hist,
        futr_exog=x_f_all,
        x_static=x_static,
        training=False,
    )
    
    # Test encoder output shape
    hT, cT = model.apply(params, y_hist=y_hist, futr_exog=x_f_all, x_static=x_static, method=DeepAR_EncDec.encode)
    assert hT.shape == (1, 32), f"Expected h shape (1, 32), got {hT.shape}"
    assert cT.shape == (1, 32), f"Expected c shape (1, 32), got {cT.shape}"
    print(f"[SUCCESS] Encoder output shapes: h={hT.shape}, c={cT.shape}")
    
    # Test one_step output
    mu, sigma, h_new, c_new = model.apply(
        params,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.ones(d_f, dtype=jnp.float32),
        x_static,
        hT,
        cT,
        True,
        method=DeepAR_EncDec.one_step
    )
    assert mu.shape == (), f"Expected scalar mu, got {mu.shape}"
    assert sigma.shape == (), f"Expected scalar sigma, got {sigma.shape}"
    assert sigma > 0, f"Sigma should be positive, got {sigma}"
    print(f"[SUCCESS] One-step output shapes: mu={mu.shape}, sigma={sigma.shape}, σ={float(sigma):.4f}")
    
    # Test model.__call__ output
    mu_seq, sigma_seq = model.apply(
        params,
        y_seq=y_hist,
        futr_exog=x_f_all,
        x_static=x_static,
        training=False,
        rngs={'dropout': key}
    )
    assert mu_seq.shape == (T,), f"Expected mu shape ({T},), got {mu_seq.shape}"
    assert sigma_seq.shape == (T,), f"Expected sigma shape ({T},), got {sigma_seq.shape}"
    assert jnp.all(sigma_seq > 0), "All sigmas should be positive"
    print(f"[SUCCESS] Training roll output shapes: mu={mu_seq.shape}, sigma={sigma_seq.shape}")
    
    print("✅ Test 1 PASSED\n")


# =============================================================================
# Test 2: Synthetic sine wave training
# =============================================================================


def test_2_synthetic_sine_wave():
    """Test 2: Train on synthetic sine wave and verify loss decreases."""
    print("\n=== Test 2: Synthetic Sine Wave Training ===")
    
    # Generate synthetic data
    T = 100
    t = jnp.linspace(0, 4 * jnp.pi, T)
    y_true = jnp.sin(t) + 0.1 * random.normal(random.PRNGKey(1), (T,))
    y_hist = jnp.array(y_true, dtype=jnp.float32)
    
    print(f"Training on {T} points of noisy sine wave...")
    
    # Train model (returns (model, params, losses, scaler) since the
    # accuracy fix that normalizes y internally)
    model, params, losses, scaler = train_model(
        y_hist,
        x_f_all=None,
        x_static=None,
        hidden=32,
        lr=1e-3,
        steps=200,
        dropout=0.0,
        verbose=False
    )
    
    # Check loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Improvement: {improvement*100:.1f}%")
    
    assert final_loss < initial_loss, "Loss should decrease during training"
    assert improvement > 0.05, f"Expected >5% improvement, got {improvement*100:.1f}%"
    print("[SUCCESS] Loss decreased during training")
    
    # Test forecasting
    H = 20
    paths = forecast_mc(
        params,
        model,
        y_hist,
        x_f_future=None,
        x_static=None,
        H=H,
        N=100,
        seed=42,
        scaler=scaler,
    )
    
    assert paths.shape == (100, H), f"Expected paths shape (100, {H}), got {paths.shape}"
    print(f"[SUCCESS] Generated {paths.shape[0]} forecast paths of length {paths.shape[1]}")
    
    # Check quantiles are ordered
    q10, q50, q90 = quantiles(paths, qs=(0.1, 0.5, 0.9))
    assert jnp.all(q10 <= q50), "10th percentile should be <= median"
    assert jnp.all(q50 <= q90), "Median should be <= 90th percentile"
    print("[SUCCESS] Forecast quantiles properly ordered")
    
    print("✅ Test 2 PASSED\n")


# =============================================================================
# Test 3: With covariates
# =============================================================================


def test_3_with_covariates():
    """Test 3: Train with future and static covariates."""
    print("\n=== Test 3: Training with Covariates ===")
    
    T = 80
    d_f = 2
    d_s = 1
    
    # Generate synthetic data with trend
    t = jnp.linspace(0, 10, T)
    trend = 0.1 * t
    seasonal = jnp.sin(2 * jnp.pi * t / 10)
    noise = 0.2 * random.normal(random.PRNGKey(2), (T,))
    y_hist = jnp.array(trend + seasonal + noise, dtype=jnp.float32)
    
    # Create covariates
    x_f_all = jnp.array(random.normal(random.PRNGKey(3), (T, d_f)), dtype=jnp.float32)
    x_static = jnp.array([1.5], dtype=jnp.float32)
    
    print(f"Training with future covariates (d_f={d_f}) and static covariates (d_s={d_s})...")
    
    model, params, losses, scaler = train_model(
        y_hist,
        x_f_all=x_f_all,
        x_static=x_static,
        hidden=32,
        lr=1e-3,
        steps=150,
        dropout=0.1,
        verbose=False
    )
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss should decrease with covariates"
    print("[SUCCESS] Training converged with covariates")
    
    # Forecast with future covariates
    H = 15
    x_f_future = jnp.array(random.normal(random.PRNGKey(4), (H, d_f)), dtype=jnp.float32)
    
    paths = forecast_mc(
        params,
        model,
        y_hist,
        x_f_hist=x_f_all,
        x_f_future=x_f_future,
        x_static=x_static,
        H=H,
        N=50,
        seed=123,
        scaler=scaler,
    )
    
    assert paths.shape == (50, H), f"Expected (50, {H}), got {paths.shape}"
    print(f"[SUCCESS] Forecasted with covariates: {paths.shape}")
    
    # Check variance across paths
    path_std = jnp.std(paths, axis=0)
    assert jnp.all(path_std > 0), "Paths should have non-zero variance"
    print(f"[SUCCESS] Forecast uncertainty (mean σ={jnp.mean(path_std):.3f})")
    
    print("✅ Test 3 PASSED\n")


# =============================================================================
# Test 4: Data pipeline
# =============================================================================


def test_4_data_pipeline():
    """Test 4: Data batch creation and alignment."""
    print("\n=== Test 4: Data Pipeline ===")
    
    # Create synthetic batch
    y_series = [
        jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32),
        jnp.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float32),
    ]
    
    batch = create_batch(
        y_series=y_series,
        input_size=3,
        h=2,
    )
    
    assert batch["insample_y"].shape == (2, 3, 1), f"Expected (2, 3, 1), got {batch['insample_y'].shape}"
    assert batch["outsample_y"].shape == (2, 2, 1), f"Expected (2, 2, 1), got {batch['outsample_y'].shape}"
    assert batch["available_mask"].shape == (2, 3, 1), f"Expected (2, 3, 1), got {batch['available_mask'].shape}"
    assert batch["sample_mask"].shape == (2, 2, 1), f"Expected (2, 2, 1), got {batch['sample_mask'].shape}"
    
    print(f"[SUCCESS] Batch shapes correct:")
    print(f"  insample_y: {batch['insample_y'].shape}")
    print(f"  outsample_y: {batch['outsample_y'].shape}")
    print(f"  available_mask: {batch['available_mask'].shape}")
    print(f"  sample_mask: {batch['sample_mask'].shape}")
    
    # Verify masks are binary
    assert jnp.all((batch["available_mask"] == 0) | (batch["available_mask"] == 1))
    assert jnp.all((batch["sample_mask"] == 0) | (batch["sample_mask"] == 1))
    print("[SUCCESS] Masks are binary (0 or 1)")
    
    print("✅ Test 4 PASSED\n")


# =============================================================================
# Test 5: High-level forecaster API
# =============================================================================


def test_5_forecaster_api():
    """Test 5: DeepARForecaster high-level API."""
    print("\n=== Test 5: DeepARForecaster API ===")
    
    # Generate synthetic data
    T = 100
    H = 20
    y_series = [jnp.sin(jnp.arange(T, dtype=jnp.float32) * 0.1) + 0.1 * random.normal(random.PRNGKey(i), (T,)) for i in range(3)]
    
    # Initialize forecaster
    forecaster = DeepARForecaster(h=H, hidden_size=32, seed=42)
    
    print("Training forecaster...")
    forecaster.fit(
        y_series=y_series,
        input_size=50,
        num_steps=100,
        batch_size=1,
        verbose=False,
    )
    print("[SUCCESS] Model fitted successfully")
    
    # Test prediction
    y_test = y_series[0]
    forecast = forecaster.forecast(y_test)
    
    assert "median" in forecast, "Forecast should have 'median' key"
    assert "lower" in forecast, "Forecast should have 'lower' key"
    assert "upper" in forecast, "Forecast should have 'upper' key"
    
    assert forecast["median"].shape == (H,), f"Expected shape ({H},), got {forecast['median'].shape}"
    assert forecast["lower"].shape == (H,), f"Expected shape ({H},), got {forecast['lower'].shape}"
    assert forecast["upper"].shape == (H,), f"Expected shape ({H},), got {forecast['upper'].shape}"
    
    # Verify upper > median > lower
    assert jnp.all(forecast["upper"] >= forecast["median"]), "Upper bound should be >= median"
    assert jnp.all(forecast["median"] >= forecast["lower"]), "Median should be >= lower bound"
    
    print(f"[SUCCESS] Forecast shapes: median={forecast['median'].shape}")
    print(f"[SUCCESS] Forecast bounds valid: lower < median < upper")
    
    print("✅ Test 5 PASSED\n")


# =============================================================================
# Test 6: Numerical stability
# =============================================================================


def test_6_numerical_stability():
    """Test 6: Verify NLL computation is numerically stable."""
    print("\n=== Test 6: Numerical Stability ===")
    
    # Test with various sigma values
    y_test = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    mu_test = jnp.array([1.1, 2.1, 2.9], dtype=jnp.float32)
    
    # Test with normal sigma
    sigma_normal = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
    from chronax.models.deepar.loss import nll_gaussian
    nll_normal = nll_gaussian(y_test, mu_test, sigma_normal)
    assert jnp.all(jnp.isfinite(nll_normal)), "NLL should be finite"
    assert jnp.all(nll_normal >= 0), "NLL should be non-negative"
    print(f"[SUCCESS] NLL with normal sigma: mean={jnp.mean(nll_normal):.3f}")
    
    # Test with very small sigma (should be clipped)
    sigma_tiny = jnp.array([1e-10, 1e-10, 1e-10], dtype=jnp.float32)
    nll_tiny = nll_gaussian(y_test, mu_test, sigma_tiny)
    assert jnp.all(jnp.isfinite(nll_tiny)), "NLL should be finite even with tiny sigma"
    print(f"[SUCCESS] NLL with tiny sigma (clipped): mean={jnp.mean(nll_tiny):.3f}")
    
    # Test with very large sigma
    sigma_large = jnp.array([1e6, 1e6, 1e6], dtype=jnp.float32)
    nll_large = nll_gaussian(y_test, mu_test, sigma_large)
    assert jnp.all(jnp.isfinite(nll_large)), "NLL should be finite with large sigma"
    print(f"[SUCCESS] NLL with large sigma: mean={jnp.mean(nll_large):.3f}")
    
    print("✅ Test 6 PASSED\n")


# =============================================================================
# Test 7: JAX vmap and JIT compilation
# =============================================================================


def test_7_jax_compilation():
    """Test 7: Verify JAX vmap and JIT compilation work."""
    print("\n=== Test 7: JAX Compilation Features ===")
    
    model = DeepAR_EncDec(hidden=16, dropout_rate=0.0)
    
    T = 30
    H = 10
    
    y_hist = jnp.ones((T,), dtype=jnp.float32)
    
    key = random.PRNGKey(0)
    params = model.init(
        {'params': key, 'dropout': key},
        y_seq=y_hist,
        futr_exog=None,
        x_static=None,
        training=False,
    )
    
    # Test JIT compilation on training_roll
    @jax.jit
    def jitted_forward(y):
        mu, sigma = model.apply(
            params,
            y_seq=y,
            futr_exog=None,
            x_static=None,
            training=False,
        )
        return mu, sigma
    
    mu, sigma = jitted_forward(y_hist)
    assert mu.shape == (T,), f"Expected shape ({T},), got {mu.shape}"
    print(f"[SUCCESS] JIT compilation successful: output shape {mu.shape}")
    
    # Call again to verify cache hit
    mu2, sigma2 = jitted_forward(y_hist)
    assert jnp.allclose(mu, mu2), "JIT output should be deterministic"
    print(f"[SUCCESS] JIT determinism verified")
    
    # Test vmap over random keys for ensemble
    def sample_path(key):
        return random.normal(key, (H,))
    
    base_key = random.PRNGKey(42)
    path_keys = random.split(base_key, 100)
    
    @jax.jit
    def batch_sample(keys):
        return jax.vmap(sample_path)(keys)
    
    paths = batch_sample(path_keys)
    assert paths.shape == (100, H), f"Expected shape (100, {H}), got {paths.shape}"
    print(f"[SUCCESS] vmap over random keys successful: output shape {paths.shape}")
    
    print("✅ Test 7 PASSED\n")


# =============================================================================
# Main test runner
# =============================================================================


def run_all_tests(verbose=True):
    """Run all tests."""
    print("=" * 70)
    print("DeepAR JAX/FLAX Test Suite")
    print("=" * 70)
    
    tests = [
        test_1_basic_shapes,
        test_2_synthetic_sine_wave,
        test_3_with_covariates,
        test_4_data_pipeline,
        test_5_forecaster_api,
        test_6_numerical_stability,
        test_7_jax_compilation,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {test_fn.__name__} FAILED: {e}\n")
            failed += 1
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

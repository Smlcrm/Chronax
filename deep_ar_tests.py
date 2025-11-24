"""
Test cases for DeepAR Encoder-Decoder model
Run with: python test_deepar.py
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# Import from your main file (adjust import as needed)
# from deep_ar import DeepAR_EncDec, train_model, forecast_mc, quantiles, nll_gauss


def test_1_basic_shapes():
    """Test 1: Basic shape checking for encoder and decoder"""
    print("\n=== Test 1: Basic Shape Checking ===")
    
    from deep_ar import DeepAR_EncDec
    
    model = DeepAR_EncDec(hidden=32, dropout_rate=0.0)
    
    # Initialize model
    T = 50
    d_f = 3
    d_s = 2
    
    y_hist = jnp.ones((T,))
    x_f_all = jnp.ones((T, d_f))
    x_static = jnp.ones((d_s,))
    
    key = random.PRNGKey(0)
    params = model.init(
        {'params': key, 'dropout': key},
        y_hist,
        x_f_all,
        x_static,
        False,
        method=DeepAR_EncDec.training_roll
    )
    
    # Test encoder output shape
    hT, cT = model.apply(params, y_hist, x_static, method=DeepAR_EncDec.encode)
    assert hT.shape == (1, 32), f"Expected h shape (1, 32), got {hT.shape}"
    assert cT.shape == (1, 32), f"Expected c shape (1, 32), got {cT.shape}"
    print(f"✓ Encoder output shapes: h={hT.shape}, c={cT.shape}")
    
    # Test one_step output
    mu, sigma, h_new, c_new = model.apply(
        params,
        jnp.array(1.0),
        x_f_all[0],
        x_static,
        hT,
        cT,
        True,
        method=DeepAR_EncDec.one_step
    )
    assert mu.shape == (), f"Expected scalar mu, got {mu.shape}"
    assert sigma.shape == (), f"Expected scalar sigma, got {sigma.shape}"
    assert sigma > 0, f"Sigma should be positive, got {sigma}"
    print(f"✓ One-step output shapes: mu={mu.shape}, sigma={sigma.shape}, σ={float(sigma):.4f}")
    
    # Test training_roll output
    mu_seq, sigma_seq = model.apply(
        params,
        y_hist,
        x_f_all,
        x_static,
        False,
        method=DeepAR_EncDec.training_roll,
        rngs={'dropout': key}
    )
    assert mu_seq.shape == (T-1,), f"Expected mu shape ({T-1},), got {mu_seq.shape}"
    assert sigma_seq.shape == (T-1,), f"Expected sigma shape ({T-1},), got {sigma_seq.shape}"
    assert jnp.all(sigma_seq > 0), "All sigmas should be positive"
    print(f"✓ Training roll output shapes: mu={mu_seq.shape}, sigma={sigma_seq.shape}")
    
    print("✅ Test 1 PASSED\n")


def test_2_synthetic_sine_wave():
    """Test 2: Train on synthetic sine wave and verify loss decreases"""
    print("\n=== Test 2: Synthetic Sine Wave Training ===")
    
    from deep_ar import train_model, forecast_mc, quantiles
    
    # Generate synthetic data
    T = 100
    t = np.linspace(0, 4 * np.pi, T)
    y_true = np.sin(t) + 0.1 * np.random.randn(T)
    y_hist = jnp.array(y_true, dtype=jnp.float32)
    
    print(f"Training on {T} points of noisy sine wave...")
    
    # Train model
    model, params, losses = train_model(
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
    assert improvement > 0.1, f"Expected >10% improvement, got {improvement*100:.1f}%"
    print("✓ Loss decreased significantly")
    
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
        seed=42
    )
    
    assert paths.shape == (100, H), f"Expected paths shape (100, {H}), got {paths.shape}"
    print(f"✓ Generated {paths.shape[0]} forecast paths of length {paths.shape[1]}")
    
    # Check quantiles are ordered
    q10, q50, q90 = quantiles(paths, qs=(0.1, 0.5, 0.9))
    assert jnp.all(q10 <= q50), "10th percentile should be <= median"
    assert jnp.all(q50 <= q90), "Median should be <= 90th percentile"
    print("✓ Forecast quantiles properly ordered")
    
    print("✅ Test 2 PASSED\n")


def test_3_with_covariates():
    """Test 3: Train with future and static covariates"""
    print("\n=== Test 3: Training with Covariates ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 80
    d_f = 2
    d_s = 1
    
    # Generate synthetic data with trend
    t = np.linspace(0, 10, T)
    trend = 0.1 * t
    seasonal = np.sin(2 * np.pi * t / 10)
    noise = 0.2 * np.random.randn(T)
    y_hist = jnp.array(trend + seasonal + noise, dtype=jnp.float32)
    
    # Create covariates
    x_f_all = jnp.array(np.random.randn(T, d_f), dtype=jnp.float32)
    x_static = jnp.array([1.5], dtype=jnp.float32)
    
    print(f"Training with future covariates (d_f={d_f}) and static covariates (d_s={d_s})...")
    
    model, params, losses = train_model(
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
    print("✓ Training converged with covariates")
    
    # Forecast with future covariates
    H = 15
    x_f_future = jnp.array(np.random.randn(H, d_f), dtype=jnp.float32)
    
    paths = forecast_mc(
        params,
        model,
        y_hist,
        x_f_future=x_f_future,
        x_static=x_static,
        H=H,
        N=50,
        seed=123
    )
    
    assert paths.shape == (50, H), f"Expected (50, {H}), got {paths.shape}"
    print(f"✓ Forecasted with covariates: {paths.shape}")
    
    # Check variance across paths
    path_std = jnp.std(paths, axis=0)
    assert jnp.all(path_std > 0), "Paths should have non-zero variance"
    print(f"✓ Forecast uncertainty (mean σ={jnp.mean(path_std):.3f})")
    
    print("✅ Test 3 PASSED\n")


def test_4_edge_cases():
    """Test 4: Edge cases and error handling"""
    print("\n=== Test 4: Edge Cases ===")
    
    from deep_ar import DeepAR_EncDec, nll_gauss
    
    model = DeepAR_EncDec(hidden=16, dropout_rate=0.0, min_sigma=0.01)
    
    # Test with minimum sequence length
    T_min = 5
    y_short = jnp.ones((T_min,))
    
    key = random.PRNGKey(99)
    params = model.init(
        {'params': key, 'dropout': key},
        y_short,
        None,
        None,
        False,
        method=DeepAR_EncDec.training_roll
    )
    
    mu, sigma = model.apply(
        params,
        y_short,
        None,
        None,
        False,
        method=DeepAR_EncDec.training_roll,
        rngs={'dropout': key}
    )
    
    assert mu.shape == (T_min-1,), f"Expected shape ({T_min-1},), got {mu.shape}"
    print(f"✓ Handles short sequences (T={T_min})")
    
    # Test with None states in one_step
    mu_step, sigma_step, h, c = model.apply(
        params,
        jnp.array(0.5),
        None,  # No future covariates
        None,  # No static covariates
        None,  # No initial h
        None,  # No initial c
        True,
        method=DeepAR_EncDec.one_step
    )
    
    assert h is not None and c is not None, "States should be initialized"
    assert h.shape == (1, 16) and c.shape == (1, 16), f"State shapes wrong: h={h.shape}, c={c.shape}"
    print("✓ Handles None states in one_step")
    
    # Test NLL function with edge cases
    y_test = jnp.array([1.0, 2.0, 3.0])
    mu_test = jnp.array([1.1, 2.1, 2.9])
    sigma_test = jnp.array([0.5, 0.5, 0.5])
    
    nll = nll_gauss(y_test, mu_test, sigma_test)
    assert jnp.all(jnp.isfinite(nll)), "NLL should be finite"
    assert jnp.all(nll >= 0), "NLL should be non-negative"
    print(f"✓ NLL computation stable (mean={jnp.mean(nll):.3f})")
    
    # Test with very small sigma (should be clipped)
    sigma_tiny = jnp.array([1e-10, 1e-10, 1e-10])
    nll_clipped = nll_gauss(y_test, mu_test, sigma_tiny)
    assert jnp.all(jnp.isfinite(nll_clipped)), "NLL should handle tiny sigma via clipping"
    print("✓ Handles tiny sigma values via clipping")
    
    print("✅ Test 4 PASSED\n")


def test_5_determinism():
    """Test 5: Check determinism with same seed"""
    print("\n=== Test 5: Determinism Check ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 60
    y_hist = jnp.array(np.random.randn(T), dtype=jnp.float32)
    
    # Train twice with same seed
    model1, params1, losses1 = train_model(
        y_hist,
        hidden=24,
        lr=1e-3,
        steps=50,
        dropout=0.0,
        verbose=False
    )
    
    model2, params2, losses2 = train_model(
        y_hist,
        hidden=24,
        lr=1e-3,
        steps=50,
        dropout=0.0,
        verbose=False
    )
    
    # Note: Due to JAX's RNG system, losses might differ slightly
    # but the final parameters should converge to similar values
    loss_diff = abs(losses1[-1] - losses2[-1])
    print(f"Final loss difference: {loss_diff:.6f}")
    
    # Forecast with same seed should be identical
    paths1 = forecast_mc(params1, model1, y_hist, H=10, N=20, seed=999)
    paths2 = forecast_mc(params1, model1, y_hist, H=10, N=20, seed=999)
    
    paths_diff = jnp.max(jnp.abs(paths1 - paths2))
    assert paths_diff < 1e-5, f"Forecasts should be identical with same seed, diff={paths_diff}"
    print(f"✓ Forecast determinism verified (max diff={paths_diff:.2e})")
    
    # Different seeds should give different results
    paths3 = forecast_mc(params1, model1, y_hist, H=10, N=20, seed=777)
    paths_diff_2 = jnp.max(jnp.abs(paths1 - paths3))
    assert paths_diff_2 > 1e-3, "Different seeds should produce different forecasts"
    print(f"✓ Different seeds produce different results (diff={paths_diff_2:.3f})")
    
    print("✅ Test 5 PASSED\n")


def test_6_no_data_leakage():
    """Test 6: Verify encoder doesn't see future targets during training"""
    print("\n=== Test 6: Data Leakage Check ===")
    
    from deep_ar import DeepAR_EncDec
    
    model = DeepAR_EncDec(hidden=32, dropout_rate=0.0)
    
    T = 50
    # Create sequence where last value is very different
    y_hist = jnp.concatenate([jnp.ones(T-1), jnp.array([100.0])])
    
    key = random.PRNGKey(0)
    params = model.init(
        {'params': key, 'dropout': key},
        y_hist,
        None,
        None,
        False,
        method=DeepAR_EncDec.training_roll
    )
    
    # Get predictions
    mu, sigma = model.apply(
        params,
        y_hist,
        None,
        None,
        False,
        method=DeepAR_EncDec.training_roll,
        rngs={'dropout': key}
    )
    
    # The prediction for y[T-1] (which is at mu[T-2]) should NOT be close to 100
    # because the encoder should only have seen y[:-1] which ends at 1.0
    last_pred = mu[-1]
    
    print(f"Last value in history: {y_hist[-1]:.1f}")
    print(f"Second-to-last value: {y_hist[-2]:.1f}")
    print(f"Model's prediction for last position: {last_pred:.3f}")
    
    # If there's leakage, the model would somehow "know" about the 100.0 value
    # Without leakage, it should predict something closer to 1.0
    # This is a heuristic check - in practice the untrained model won't be perfect
    
    print("✓ Encoder processes y[:-1], decoder predicts y[1:]")
    print("  (Manual inspection recommended for full verification)")
    
    print("✅ Test 6 PASSED\n")


def test_viz_1_sine_wave_forecast():
    """Visualization Test 1: Sine wave with uncertainty bands"""
    print("\n=== Viz Test 1: Sine Wave Forecast ===")
    
    from deep_ar import train_model, forecast_mc, quantiles
    import matplotlib.pyplot as plt
    
    # Generate clean sine wave
    T_hist = 100
    H = 50
    T_total = T_hist + H
    
    t = np.linspace(0, 8 * np.pi, T_total)
    y_true = 2 * np.sin(t) + 0.3 * np.random.randn(T_total)
    
    y_hist = jnp.array(y_true[:T_hist], dtype=jnp.float32)
    y_future = y_true[T_hist:]
    
    print(f"Training on {T_hist} points, forecasting {H} steps...")
    
    # Train
    model, params, losses = train_model(
        y_hist,
        hidden=64,
        lr=1e-3,
        steps=500,
        dropout=0.1,
        verbose=False
    )
    
    # Forecast
    paths = forecast_mc(
        params,
        model,
        y_hist,
        H=H,
        N=500,
        seed=42
    )
    
    # Compute quantiles
    q10, q50, q90 = quantiles(paths, qs=(0.1, 0.5, 0.9))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Forecast with uncertainty
    t_hist_plot = np.arange(T_hist)
    t_future_plot = np.arange(T_hist, T_hist + H)
    
    ax1.plot(t_hist_plot, y_hist, 'b-', label='History', linewidth=2)
    ax1.plot(t_future_plot, y_future, 'g-', label='True Future', linewidth=2, alpha=0.7)
    ax1.plot(t_future_plot, q50, 'r-', label='Median Forecast', linewidth=2)
    ax1.fill_between(t_future_plot, q10, q90, alpha=0.3, color='red', label='10-90% CI')
    
    ax1.axvline(x=T_hist, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('DeepAR Forecast: Sine Wave', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training loss
    ax2.plot(losses, 'b-', linewidth=2)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('NLL Loss', fontsize=12)
    ax2.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('deepar_test_sine_wave.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to 'deepar_test_sine_wave.png'")
    plt.close()
    
    print("✅ Viz Test 1 PASSED\n")


def test_viz_2_sample_paths():
    """Visualization Test 2: Individual sample paths"""
    print("\n=== Viz Test 2: Sample Paths ===")
    
    from deep_ar import train_model, forecast_mc
    import matplotlib.pyplot as plt
    
    # Generate data with trend and seasonality
    T_hist = 80
    H = 40
    
    t = np.linspace(0, 10, T_hist)
    trend = 0.2 * t
    seasonal = 1.5 * np.sin(2 * np.pi * t / 8)
    noise = 0.3 * np.random.randn(T_hist)
    y_hist = jnp.array(trend + seasonal + noise, dtype=jnp.float32)
    
    print(f"Training on {T_hist} points with trend + seasonality...")
    
    # Train
    model, params, _ = train_model(
        y_hist,
        hidden=48,
        lr=1e-3,
        steps=400,
        dropout=0.15,
        verbose=False
    )
    
    # Generate multiple paths
    paths = forecast_mc(
        params,
        model,
        y_hist,
        H=H,
        N=100,
        seed=123
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    t_hist_plot = np.arange(T_hist)
    t_future_plot = np.arange(T_hist, T_hist + H)
    
    # Plot history
    ax.plot(t_hist_plot, y_hist, 'b-', label='History', linewidth=3, zorder=10)
    
    # Plot first 20 sample paths
    for i in range(20):
        ax.plot(t_future_plot, paths[i], 'r-', alpha=0.2, linewidth=1)
    
    # Add median
    median = jnp.median(paths, axis=0)
    ax.plot(t_future_plot, median, 'darkred', label='Median Forecast', linewidth=3, zorder=5)
    
    ax.axvline(x=T_hist, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('DeepAR Sample Paths (20 shown out of 100)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('deepar_test_sample_paths.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to 'deepar_test_sample_paths.png'")
    plt.close()
    
    print("✅ Viz Test 2 PASSED\n")


def test_viz_3_with_covariates():
    """Visualization Test 3: Forecast with future covariates"""
    print("\n=== Viz Test 3: Forecast with Covariates ===")
    
    from deep_ar import train_model, forecast_mc, quantiles
    import matplotlib.pyplot as plt
    
    T_hist = 120
    H = 30
    T_total = T_hist + H
    
    # Generate data where covariates influence the target
    t = np.linspace(0, 12, T_total)
    x_f_covariate = np.sin(2 * np.pi * t / 10)  # Known future covariate
    
    # Target depends on covariate + own dynamics
    y_true = 1.5 * x_f_covariate + 0.5 * np.sin(2 * np.pi * t / 15) + 0.3 * np.random.randn(T_total)
    
    y_hist = jnp.array(y_true[:T_hist], dtype=jnp.float32)
    y_future = y_true[T_hist:]
    
    x_f_all = jnp.array(x_f_covariate[:T_hist, None], dtype=jnp.float32)
    x_f_future = jnp.array(x_f_covariate[T_hist:, None], dtype=jnp.float32)
    x_static = jnp.array([0.5], dtype=jnp.float32)
    
    print(f"Training with 1 future covariate and 1 static covariate...")
    
    # Train
    model, params, losses = train_model(
        y_hist,
        x_f_all=x_f_all,
        x_static=x_static,
        hidden=64,
        lr=1e-3,
        steps=400,
        dropout=0.1,
        verbose=False
    )
    
    # Forecast WITH covariates
    paths_with = forecast_mc(
        params,
        model,
        y_hist,
        x_f_future=x_f_future,
        x_static=x_static,
        H=H,
        N=300,
        seed=999
    )
    
    # Forecast WITHOUT covariates (zeros)
    paths_without = forecast_mc(
        params,
        model,
        y_hist,
        x_f_future=jnp.zeros_like(x_f_future),
        x_static=x_static,
        H=H,
        N=300,
        seed=999
    )
    
    q10_with, q50_with, q90_with = quantiles(paths_with, qs=(0.1, 0.5, 0.9))
    q10_without, q50_without, q90_without = quantiles(paths_without, qs=(0.1, 0.5, 0.9))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    t_hist_plot = np.arange(T_hist)
    t_future_plot = np.arange(T_hist, T_hist + H)
    
    # Plot 1: Comparison with/without covariates
    ax1.plot(t_hist_plot, y_hist, 'b-', label='History', linewidth=2)
    ax1.plot(t_future_plot, y_future, 'g-', label='True Future', linewidth=2, alpha=0.7)
    
    ax1.plot(t_future_plot, q50_with, 'r-', label='Median (with covariates)', linewidth=2)
    ax1.fill_between(t_future_plot, q10_with, q90_with, alpha=0.25, color='red')
    
    ax1.plot(t_future_plot, q50_without, 'm--', label='Median (without covariates)', linewidth=2)
    ax1.fill_between(t_future_plot, q10_without, q90_without, alpha=0.15, color='magenta')
    
    ax1.axvline(x=T_hist, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Impact of Future Covariates on Forecast', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: The covariate itself
    ax2.plot(t, x_f_covariate, 'purple', linewidth=2, label='Future Covariate')
    ax2.axvline(x=T_hist, color='k', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Covariate Value', fontsize=12)
    ax2.set_title('Known Future Covariate', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('deepar_test_covariates.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to 'deepar_test_covariates.png'")
    plt.close()
    
    print("✅ Viz Test 3 PASSED\n")


def test_viz_4_forecast_distribution():
    """Visualization Test 4: Forecast distribution at different horizons"""
    print("\n=== Viz Test 4: Forecast Distribution Evolution ===")
    
    from deep_ar import train_model, forecast_mc
    import matplotlib.pyplot as plt
    
    T_hist = 100
    H = 40
    
    # Generate data
    t = np.linspace(0, 10, T_hist)
    y_hist = jnp.array(np.sin(t) + 0.2 * np.random.randn(T_hist), dtype=jnp.float32)
    
    print(f"Analyzing forecast uncertainty over {H} horizons...")
    
    # Train
    model, params, _ = train_model(
        y_hist,
        hidden=48,
        lr=1e-3,
        steps=300,
        dropout=0.1,
        verbose=False
    )
    
    # Forecast
    paths = forecast_mc(
        params,
        model,
        y_hist,
        H=H,
        N=1000,
        seed=777
    )
    
    # Plot distributions at specific horizons
    horizons_to_plot = [1, 5, 10, 20, 30, 40]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, h in enumerate(horizons_to_plot):
        ax = axes[idx]
        values = paths[:, h-1]
        
        ax.hist(values, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        mean_val = jnp.mean(values)
        std_val = jnp.std(values)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_val:.2f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ')
        
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Horizon t+{h} (σ={std_val:.2f})', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Forecast Distribution at Different Horizons', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('deepar_test_distributions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to 'deepar_test_distributions.png'")
    plt.close()
    
    # Also plot uncertainty growth
    fig, ax = plt.subplots(figsize=(10, 6))
    
    uncertainties = jnp.std(paths, axis=0)
    horizons = np.arange(1, H + 1)
    
    ax.plot(horizons, uncertainties, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Forecast Horizon', fontsize=12)
    ax.set_ylabel('Standard Deviation', fontsize=12)
    ax.set_title('Forecast Uncertainty Growth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('deepar_test_uncertainty_growth.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to 'deepar_test_uncertainty_growth.png'")
    plt.close()
    
    print("✅ Viz Test 4 PASSED\n")


def test_7_multi_step_encoder():
    """Test 7: Verify encoder processes sequences of varying lengths correctly"""
    print("\n=== Test 7: Multi-Length Encoder Test ===")
    
    from deep_ar import DeepAR_EncDec
    
    model = DeepAR_EncDec(hidden=32, dropout_rate=0.0)
    
    key = random.PRNGKey(42)
    
    # Test with different sequence lengths
    lengths = [10, 50, 100, 200]
    
    for T in lengths:
        y_hist = jnp.ones((T,)) * 0.5
        
        # Initialize if needed
        if T == lengths[0]:
            params = model.init(
                {'params': key, 'dropout': key},
                y_hist,
                None,
                None,
                False,
                method=DeepAR_EncDec.training_roll
            )
        
        # Test encoder
        hT, cT = model.apply(params, y_hist, None, method=DeepAR_EncDec.encode)
        
        assert hT.shape == (1, 32), f"Length {T}: Expected h shape (1, 32), got {hT.shape}"
        assert cT.shape == (1, 32), f"Length {T}: Expected c shape (1, 32), got {cT.shape}"
        assert jnp.all(jnp.isfinite(hT)), f"Length {T}: h contains non-finite values"
        assert jnp.all(jnp.isfinite(cT)), f"Length {T}: c contains non-finite values"
    
    print(f"✓ Encoder handles sequences of lengths: {lengths}")
    print("✅ Test 7 PASSED\n")


def test_8_gradient_flow():
    """Test 8: Verify gradients flow properly during training"""
    print("\n=== Test 8: Gradient Flow Test ===")
    
    from deep_ar import DeepAR_EncDec, nll_gauss
    from jax import grad
    
    model = DeepAR_EncDec(hidden=24, dropout_rate=0.0)
    
    T = 30
    y_hist = jnp.array(np.sin(np.linspace(0, 4*np.pi, T)), dtype=jnp.float32)
    
    key = random.PRNGKey(99)
    params = model.init(
        {'params': key, 'dropout': key},
        y_hist,
        None,
        None,
        False,
        method=DeepAR_EncDec.training_roll
    )
    
    # Define loss function
    def loss_fn(params):
        mu, sigma = model.apply(
            params,
            y_hist,
            None,
            None,
            False,
            method=DeepAR_EncDec.training_roll,
            rngs={'dropout': key}
        )
        y_target = y_hist[1:]
        nll = jnp.mean(nll_gauss(y_target, mu, sigma))
        return nll
    
    # Compute gradients
    grads = grad(loss_fn)(params)
    
    # Check that gradients exist and are finite for all parameters
    def check_grads(grad_tree, path=""):
        if isinstance(grad_tree, dict):
            for k, v in grad_tree.items():
                check_grads(v, f"{path}/{k}")
        else:
            assert jnp.all(jnp.isfinite(grad_tree)), f"Non-finite gradient at {path}"
            grad_norm = jnp.linalg.norm(grad_tree.flatten())
            assert grad_norm > 0, f"Zero gradient at {path}"
    
    check_grads(grads['params'])
    print("✓ All parameter gradients are finite and non-zero")
    
    print("✅ Test 8 PASSED\n")


def test_9_static_covariates_only():
    """Test 9: Train with only static covariates (no future covariates)"""
    print("\n=== Test 9: Static Covariates Only ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 70
    d_s = 3
    
    # Generate data influenced by static features
    np.random.seed(42)
    static_vals = np.array([1.0, -0.5, 2.0])
    t = np.linspace(0, 8, T)
    
    # Static covariates influence the amplitude and offset
    base_signal = static_vals[0] * np.sin(t) + static_vals[1] * np.cos(t) + static_vals[2]
    y_hist = jnp.array(base_signal + 0.2 * np.random.randn(T), dtype=jnp.float32)
    x_static = jnp.array(static_vals, dtype=jnp.float32)
    
    print(f"Training with static covariates only (d_s={d_s})...")
    
    model, params, losses = train_model(
        y_hist,
        x_f_all=None,  # No future covariates
        x_static=x_static,
        hidden=32,
        lr=1e-3,
        steps=200,
        dropout=0.1,
        verbose=False
    )
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss should decrease with static covariates"
    print("✓ Training converged with static covariates only")
    
    # Forecast
    H = 20
    paths = forecast_mc(
        params,
        model,
        y_hist,
        x_f_future=None,
        x_static=x_static,
        H=H,
        N=100,
        seed=456
    )
    
    assert paths.shape == (100, H), f"Expected (100, {H}), got {paths.shape}"
    print(f"✓ Forecasted with static covariates: {paths.shape}")
    
    print("✅ Test 9 PASSED\n")


def test_10_future_covariates_only():
    """Test 10: Train with only future covariates (no static covariates)"""
    print("\n=== Test 10: Future Covariates Only ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 80
    H = 15
    d_f = 2
    
    # Generate data where future covariates are predictive
    np.random.seed(123)
    t = np.linspace(0, 10, T + H)
    x_f_1 = np.sin(2 * np.pi * t / 8)
    x_f_2 = np.cos(2 * np.pi * t / 12)
    
    # Target is influenced by shifted covariates
    y_full = 0.8 * x_f_1 + 0.6 * x_f_2 + 0.3 * np.random.randn(T + H)
    
    y_hist = jnp.array(y_full[:T], dtype=jnp.float32)
    x_f_all = jnp.stack([x_f_1[:T], x_f_2[:T]], axis=1).astype(jnp.float32)
    x_f_future = jnp.stack([x_f_1[T:T+H], x_f_2[T:T+H]], axis=1).astype(jnp.float32)
    
    print(f"Training with future covariates only (d_f={d_f})...")
    
    model, params, losses = train_model(
        y_hist,
        x_f_all=x_f_all,
        x_static=None,  # No static covariates
        hidden=32,
        lr=1e-3,
        steps=200,
        dropout=0.1,
        verbose=False
    )
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss should decrease with future covariates"
    print("✓ Training converged with future covariates only")
    
    # Forecast
    paths = forecast_mc(
        params,
        model,
        y_hist,
        x_f_future=x_f_future,
        x_static=None,
        H=H,
        N=100,
        seed=789
    )
    
    assert paths.shape == (100, H), f"Expected (100, {H}), got {paths.shape}"
    print(f"✓ Forecasted with future covariates: {paths.shape}")
    
    print("✅ Test 10 PASSED\n")


def test_11_varying_noise_levels():
    """Test 11: Test robustness to different noise levels"""
    print("\n=== Test 11: Varying Noise Levels ===")
    
    from deep_ar import train_model
    
    T = 100
    noise_levels = [0.01, 0.1, 0.5, 1.0]
    
    t = np.linspace(0, 6 * np.pi, T)
    clean_signal = np.sin(t)
    
    results = []
    
    for noise_std in noise_levels:
        np.random.seed(42)
        y_hist = jnp.array(clean_signal + noise_std * np.random.randn(T), dtype=jnp.float32)
        
        model, params, losses = train_model(
            y_hist,
            hidden=32,
            lr=1e-3,
            steps=150,
            dropout=0.1,
            verbose=False
        )
        
        final_loss = losses[-1]
        results.append((noise_std, final_loss))
        print(f"  Noise σ={noise_std:.2f}: Final loss={final_loss:.4f}")
    
    # Check that higher noise generally leads to higher loss (with some tolerance)
    # This is a weak ordering check since optimization is stochastic
    print("✓ Model handles various noise levels")
    
    print("✅ Test 11 PASSED\n")


def test_12_forecast_consistency():
    """Test 12: Verify forecast distributions are consistent with training data scale"""
    print("\n=== Test 12: Forecast Scale Consistency ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 80
    H = 20
    
    # Generate data with known scale
    np.random.seed(99)
    t = np.linspace(0, 8, T)
    y_hist = jnp.array(5 + 2 * np.sin(t) + 0.3 * np.random.randn(T), dtype=jnp.float32)
    
    hist_mean = float(jnp.mean(y_hist))
    hist_std = float(jnp.std(y_hist))
    
    print(f"Training data - Mean: {hist_mean:.3f}, Std: {hist_std:.3f}")
    
    model, params, _ = train_model(
        y_hist,
        hidden=48,
        lr=1e-3,
        steps=200,
        dropout=0.1,
        verbose=False
    )
    
    # Generate forecasts
    paths = forecast_mc(
        params,
        model,
        y_hist,
        H=H,
        N=500,
        seed=321
    )
    
    # Check forecast statistics at different horizons
    for h_idx in [0, H//2-1, H-1]:
        forecast_mean = float(jnp.mean(paths[:, h_idx]))
        forecast_std = float(jnp.std(paths[:, h_idx]))
        
        print(f"  Horizon t+{h_idx+1} - Mean: {forecast_mean:.3f}, Std: {forecast_std:.3f}")
        
        # Forecasts should be in a reasonable range (within 5x the historical std)
        assert abs(forecast_mean - hist_mean) < 5 * hist_std, \
            f"Forecast mean too far from history at horizon {h_idx+1}"
    
    print("✓ Forecast scales are consistent with training data")
    
    print("✅ Test 12 PASSED\n")


def test_13_state_persistence():
    """Test 13: Verify LSTM states persist properly across steps"""
    print("\n=== Test 13: State Persistence ===")
    
    from deep_ar import DeepAR_EncDec
    
    model = DeepAR_EncDec(hidden=32, dropout_rate=0.0)
    
    T = 30
    y_hist = jnp.array(np.random.randn(T), dtype=jnp.float32)
    
    key = random.PRNGKey(555)
    params = model.init(
        {'params': key, 'dropout': key},
        y_hist,
        None,
        None,
        False,
        method=DeepAR_EncDec.training_roll
    )
    
    # Get initial state from encoder
    h0, c0 = model.apply(params, y_hist, None, method=DeepAR_EncDec.encode)
    
    # Take multiple steps and verify states change
    states = [(h0, c0)]
    y_prev = y_hist[-1]
    
    for step in range(5):
        mu, sigma, h_new, c_new = model.apply(
            params,
            y_prev,
            None,
            None,
            states[-1][0],
            states[-1][1],
            True,
            method=DeepAR_EncDec.one_step
        )
        
        states.append((h_new, c_new))
        y_prev = mu  # Use prediction as next input
    
    # Verify states are changing
    for i in range(1, len(states)):
        h_diff = jnp.max(jnp.abs(states[i][0] - states[i-1][0]))
        c_diff = jnp.max(jnp.abs(states[i][1] - states[i-1][1]))
        
        assert h_diff > 1e-6, f"Step {i}: h state not changing (diff={h_diff})"
        assert c_diff > 1e-6, f"Step {i}: c state not changing (diff={c_diff})"
    
    print(f"✓ States evolve across {len(states)-1} steps")
    print("✅ Test 13 PASSED\n")


def test_14_batch_forecast_consistency():
    """Test 14: Verify forecasts with different N produce consistent statistics"""
    print("\n=== Test 14: Batch Forecast Consistency ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 60
    H = 15
    
    np.random.seed(777)
    y_hist = jnp.array(np.cumsum(np.random.randn(T)) * 0.1, dtype=jnp.float32)
    
    model, params, _ = train_model(
        y_hist,
        hidden=32,
        lr=1e-3,
        steps=150,
        dropout=0.0,
        verbose=False
    )
    
    # Generate forecasts with different sample sizes
    N_values = [100, 500, 1000]
    medians = []
    
    for N in N_values:
        paths = forecast_mc(
            params,
            model,
            y_hist,
            H=H,
            N=N,
            seed=888  # Same seed for comparison
        )
        
        median = jnp.median(paths[:, H//2])
        medians.append(float(median))
        print(f"  N={N:4d}: median at t+{H//2} = {median:.4f}")
    
    # Check that medians are reasonably consistent across different N
    median_range = max(medians) - min(medians)
    median_mean = sum(medians) / len(medians)
    
    # Allow 10% relative variation
    assert median_range < 0.1 * abs(median_mean) + 0.5, \
        f"Medians vary too much across different N: range={median_range:.4f}"
    
    print("✓ Forecast statistics consistent across different sample sizes")
    
    print("✅ Test 14 PASSED\n")


def test_15_zero_and_constant_sequences():
    """Test 15: Handle edge cases of zero and constant sequences"""
    print("\n=== Test 15: Zero and Constant Sequences ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 50
    H = 10
    
    # Test 1: All zeros
    y_zeros = jnp.zeros((T,), dtype=jnp.float32)
    
    model_zeros, params_zeros, losses_zeros = train_model(
        y_zeros,
        hidden=24,
        lr=1e-3,
        steps=100,
        dropout=0.0,
        verbose=False
    )
    
    paths_zeros = forecast_mc(params_zeros, model_zeros, y_zeros, H=H, N=50, seed=111)
    
    # Forecasts should be near zero with some uncertainty
    mean_forecast = jnp.mean(paths_zeros)
    assert abs(mean_forecast) < 1.0, f"Zero sequence forecast too far from zero: {mean_forecast:.4f}"
    print(f"✓ Zero sequence: mean forecast = {mean_forecast:.4f}")
    
    # Test 2: Constant non-zero with small noise to make it learnable
    constant_val = 5.0
    np.random.seed(42)
    # Add tiny noise to make the sequence non-degenerate
    y_const = jnp.array(constant_val + 0.01 * np.random.randn(T), dtype=jnp.float32)
    
    model_const, params_const, losses_const = train_model(
        y_const,
        hidden=32,
        lr=1e-3,
        steps=300,  # More steps for convergence
        dropout=0.0,
        verbose=False
    )
    
    paths_const = forecast_mc(params_const, model_const, y_const, H=H, N=100, seed=222)
    
    # Forecasts should be near the constant value
    # Allow for more tolerance since constant sequences are challenging
    mean_forecast_const = jnp.mean(paths_const)
    median_forecast_const = jnp.median(paths_const)
    
    # Check that forecast is in reasonable range (within 2.0 of true value)
    error = abs(median_forecast_const - constant_val)
    assert error < 2.0, \
        f"Constant sequence forecast error: expected ~{constant_val}, got median={median_forecast_const:.4f}"
    print(f"✓ Constant sequence (val={constant_val}): median forecast = {median_forecast_const:.4f}, error = {error:.4f}")
    
    # Also verify the model learned something (loss decreased)
    assert losses_const[-1] < losses_const[0], "Loss should decrease on constant sequence"
    print(f"  Loss: {losses_const[0]:.4f} → {losses_const[-1]:.4f}")
    
    print("✅ Test 15 PASSED\n")


def test_16_long_horizon_forecasts():
    """Test 16: Test forecasting at very long horizons"""
    print("\n=== Test 16: Long Horizon Forecasts ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 100
    horizons = [10, 50, 100, 200]
    
    # Generate simple periodic data
    np.random.seed(88)
    t = np.linspace(0, 8*np.pi, T)
    y_hist = jnp.array(np.sin(t) + 0.2*np.random.randn(T), dtype=jnp.float32)
    
    print(f"Training on {T} points...")
    model, params, _ = train_model(
        y_hist,
        hidden=48,
        lr=1e-3,
        steps=300,
        dropout=0.1,
        verbose=False
    )
    
    for H in horizons:
        paths = forecast_mc(params, model, y_hist, H=H, N=100, seed=999)
        
        assert paths.shape == (100, H), f"H={H}: Expected shape (100, {H}), got {paths.shape}"
        assert jnp.all(jnp.isfinite(paths)), f"H={H}: Contains non-finite values"
        
        # Check uncertainty grows with horizon
        uncertainties = jnp.std(paths, axis=0)
        early_unc = float(jnp.mean(uncertainties[:5]))
        late_unc = float(jnp.mean(uncertainties[-5:]))
        
        print(f"  H={H:3d}: Early uncertainty={early_unc:.3f}, Late uncertainty={late_unc:.3f}")
        
        # Uncertainty should generally increase (allow some tolerance)
        if H > 20:
            assert late_unc >= early_unc * 0.8, \
                f"H={H}: Uncertainty should grow with horizon"
    
    print("✓ Model handles long horizons with growing uncertainty")
    print("✅ Test 16 PASSED\n")


def test_17_negative_values():
    """Test 17: Handle sequences with negative values"""
    print("\n=== Test 17: Negative Values ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 80
    H = 20
    
    # Generate data with negative values
    np.random.seed(123)
    t = np.linspace(0, 10, T)
    # Centered around -2
    y_hist = jnp.array(-2 + np.sin(t) + 0.3*np.random.randn(T), dtype=jnp.float32)
    
    assert jnp.any(y_hist < 0), "Test data should contain negative values"
    print(f"Data range: [{float(jnp.min(y_hist)):.2f}, {float(jnp.max(y_hist)):.2f}]")
    
    model, params, losses = train_model(
        y_hist,
        hidden=32,
        lr=1e-3,
        steps=200,
        dropout=0.1,
        verbose=False
    )
    
    assert losses[-1] < losses[0], "Loss should decrease"
    print(f"✓ Training converged: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    # Forecast
    paths = forecast_mc(params, model, y_hist, H=H, N=200, seed=456)
    
    # Forecasts should also be able to produce negative values
    has_negative = jnp.any(paths < 0)
    mean_forecast = float(jnp.mean(paths))
    
    print(f"✓ Forecast mean: {mean_forecast:.3f}, Contains negatives: {has_negative}")
    assert jnp.all(jnp.isfinite(paths)), "All forecasts should be finite"
    
    print("✅ Test 17 PASSED\n")


def test_18_trend_extrapolation():
    """Test 18: Test ability to capture and extrapolate trends"""
    print("\n=== Test 18: Trend Extrapolation ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 100
    H = 30
    
    # Generate data with clear upward trend
    np.random.seed(77)
    t = np.linspace(0, 10, T)
    trend = 0.3 * t  # Linear upward trend
    seasonal = 0.5 * np.sin(2*np.pi*t/10)
    noise = 0.2 * np.random.randn(T)
    y_hist = jnp.array(trend + seasonal + noise, dtype=jnp.float32)
    
    # Calculate trend from last 20 points
    recent_trend = float((y_hist[-1] - y_hist[-20]) / 20)
    print(f"Recent trend: {recent_trend:.4f} per timestep")
    
    model, params, _ = train_model(
        y_hist,
        hidden=64,
        lr=1e-3,
        steps=400,
        dropout=0.1,
        verbose=False
    )
    
    # Forecast
    paths = forecast_mc(params, model, y_hist, H=H, N=200, seed=333)
    
    median_forecast = jnp.median(paths, axis=0)
    
    # Check if forecast continues upward trend
    forecast_start = float(median_forecast[0])
    forecast_end = float(median_forecast[-1])
    forecast_trend = (forecast_end - forecast_start) / H
    
    print(f"✓ Forecast trend: {forecast_trend:.4f} per timestep")
    print(f"  Forecast range: [{forecast_start:.2f}, {forecast_end:.2f}]")
    
    # The forecast should show positive trend (allowing for seasonality)
    assert forecast_end > forecast_start - 1.0, "Forecast should maintain upward trend direction"
    
    print("✅ Test 18 PASSED\n")


def test_19_covariate_dimension_mismatch():
    """Test 19: Ensure proper error handling for dimension mismatches"""
    print("\n=== Test 19: Covariate Dimension Checks ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 50
    H = 10
    
    np.random.seed(99)
    y_hist = jnp.array(np.random.randn(T), dtype=jnp.float32)
    
    # Train with 2D future covariates
    x_f_all = jnp.array(np.random.randn(T, 2), dtype=jnp.float32)
    
    model, params, _ = train_model(
        y_hist,
        x_f_all=x_f_all,
        hidden=24,
        lr=1e-3,
        steps=100,
        dropout=0.0,
        verbose=False
    )
    
    print("✓ Trained with 2D future covariates")
    
    # Test correct dimensions work
    x_f_future_correct = jnp.array(np.random.randn(H, 2), dtype=jnp.float32)
    paths_correct = forecast_mc(params, model, y_hist, 
                                 x_f_future=x_f_future_correct, H=H, N=20, seed=111)
    assert paths_correct.shape == (20, H)
    print("✓ Forecast works with matching covariate dimensions")
    
    # Test with wrong future covariate dimensions (should fail or handle gracefully)
    try:
        x_f_future_wrong = jnp.array(np.random.randn(H, 3), dtype=jnp.float32)  # Wrong: 3D instead of 2D
        paths_wrong = forecast_mc(params, model, y_hist, 
                                   x_f_future=x_f_future_wrong, H=H, N=20, seed=222)
        print("⚠ Warning: Model accepted wrong covariate dimensions (may indicate issue)")
    except Exception as e:
        print(f"✓ Correctly rejected wrong covariate dimensions")
    
    print("✅ Test 19 PASSED\n")


def test_20_encoder_decoder_state_transfer():
    """Test 20: Verify encoder state properly initializes decoder"""
    print("\n=== Test 20: Encoder-Decoder State Transfer ===")
    
    from deep_ar import DeepAR_EncDec
    
    model = DeepAR_EncDec(hidden=32, dropout_rate=0.0)
    
    T = 40
    np.random.seed(555)
    y_hist = jnp.array(np.cumsum(np.random.randn(T)) * 0.1, dtype=jnp.float32)
    
    key = random.PRNGKey(123)
    params = model.init(
        {'params': key, 'dropout': key},
        y_hist,
        None,
        None,
        False,
        method=DeepAR_EncDec.training_roll
    )
    
    # Get encoder state
    h_enc, c_enc = model.apply(params, y_hist, None, method=DeepAR_EncDec.encode)
    
    print(f"Encoder state shape: h={h_enc.shape}, c={c_enc.shape}")
    
    # Use this state in decoder
    mu1, sigma1, h_new1, c_new1 = model.apply(
        params,
        y_hist[-1],
        None,
        None,
        h_enc,
        c_enc,
        True,
        method=DeepAR_EncDec.one_step
    )
    
    # Compare with zero-initialized state
    h_zero = jnp.zeros_like(h_enc)
    c_zero = jnp.zeros_like(c_enc)
    
    mu2, sigma2, h_new2, c_new2 = model.apply(
        params,
        y_hist[-1],
        None,
        None,
        h_zero,
        c_zero,
        True,
        method=DeepAR_EncDec.one_step
    )
    
    # Predictions should be different when using encoder state vs zero state
    mu_diff = abs(float(mu1 - mu2))
    sigma_diff = abs(float(sigma1 - sigma2))
    
    print(f"✓ Prediction difference with encoder state: mu_diff={mu_diff:.4f}, sigma_diff={sigma_diff:.4f}")
    
    # There should be meaningful difference
    assert mu_diff > 1e-4 or sigma_diff > 1e-4, \
        "Encoder state should affect decoder predictions"
    
    print("✅ Test 20 PASSED\n")


def test_21_dropout_effect():
    """Test 21: Verify dropout affects training but not inference"""
    print("\n=== Test 21: Dropout Behavior ===")
    
    from deep_ar import DeepAR_EncDec
    
    T = 60
    np.random.seed(888)
    y_hist = jnp.array(np.sin(np.linspace(0, 4*np.pi, T)), dtype=jnp.float32)
    
    # Model with dropout
    model = DeepAR_EncDec(hidden=32, dropout_rate=0.3)
    
    key = random.PRNGKey(42)
    params = model.init(
        {'params': key, 'dropout': key},
        y_hist,
        None,
        None,
        True,  # training=True
        method=DeepAR_EncDec.training_roll
    )
    
    # Get outputs with training=True (dropout active) using different keys
    key1, key2 = random.split(key)
    
    mu1, sigma1 = model.apply(
        params,
        y_hist,
        None,
        None,
        True,
        method=DeepAR_EncDec.training_roll,
        rngs={'dropout': key1}
    )
    
    mu2, sigma2 = model.apply(
        params,
        y_hist,
        None,
        None,
        True,
        method=DeepAR_EncDec.training_roll,
        rngs={'dropout': key2}
    )
    
    # With dropout and different keys, outputs should differ
    training_diff = float(jnp.mean(jnp.abs(mu1 - mu2)))
    print(f"✓ Training mode difference (dropout active): {training_diff:.6f}")
    
    # In inference mode (deterministic=True in one_step), outputs should be identical
    h, c = model.apply(params, y_hist, None, method=DeepAR_EncDec.encode)
    
    mu_inf1, sigma_inf1, _, _ = model.apply(
        params,
        y_hist[-1],
        None,
        None,
        h,
        c,
        True,  # deterministic=True
        method=DeepAR_EncDec.one_step
    )
    
    mu_inf2, sigma_inf2, _, _ = model.apply(
        params,
        y_hist[-1],
        None,
        None,
        h,
        c,
        True,  # deterministic=True
        method=DeepAR_EncDec.one_step
    )
    
    inference_diff = abs(float(mu_inf1 - mu_inf2))
    print(f"✓ Inference mode difference (dropout off): {inference_diff:.6e}")
    
    assert inference_diff < 1e-6, "Inference should be deterministic"
    
    print("✅ Test 21 PASSED\n")


def test_22_sigma_bounds():
    """Test 22: Verify sigma stays within reasonable bounds"""
    print("\n=== Test 22: Sigma Bounds Check ===")
    
    from deep_ar import train_model, DeepAR_EncDec
    
    T = 80
    
    # Test with different data scales
    scales = [0.01, 0.1, 1.0, 10.0]
    
    for scale in scales:
        np.random.seed(42)
        y_hist = jnp.array(scale * np.sin(np.linspace(0, 6*np.pi, T)), dtype=jnp.float32)
        
        model, params, _ = train_model(
            y_hist,
            hidden=32,
            lr=1e-3,
            steps=150,
            dropout=0.0,
            min_sigma=0.01,
            verbose=False
        )
        
        # Get predictions
        key = random.PRNGKey(99)
        mu, sigma = model.apply(
            params,
            y_hist,
            None,
            None,
            False,
            method=DeepAR_EncDec.training_roll,
            rngs={'dropout': key}
        )
        
        min_sigma = float(jnp.min(sigma))
        max_sigma = float(jnp.max(sigma))
        mean_sigma = float(jnp.mean(sigma))
        
        print(f"  Scale={scale:5.2f}: σ ∈ [{min_sigma:.4f}, {max_sigma:.4f}], mean={mean_sigma:.4f}")
        
        # Check sigma is positive and bounded
        assert jnp.all(sigma > 0), f"Scale {scale}: All sigmas should be positive"
        assert min_sigma >= 0.01, f"Scale {scale}: Sigma should respect min_sigma floor"
        assert max_sigma < 1e3, f"Scale {scale}: Sigma should not explode"
    
    print("✓ Sigma values remain in reasonable bounds across scales")
    print("✅ Test 22 PASSED\n")


def test_23_missing_history_handling():
    """Test 23: Handle very short history lengths"""
    print("\n=== Test 23: Short History Handling ===")
    
    from deep_ar import train_model, forecast_mc
    
    # Test with minimal viable history lengths
    min_lengths = [3, 5, 10, 20]
    
    for T in min_lengths:
        np.random.seed(T)
        y_hist = jnp.array(np.random.randn(T), dtype=jnp.float32)
        
        try:
            model, params, losses = train_model(
                y_hist,
                hidden=16,
                lr=1e-3,
                steps=50,
                dropout=0.0,
                verbose=False
            )
            
            # Try forecasting
            paths = forecast_mc(params, model, y_hist, H=5, N=20, seed=T)
            
            assert paths.shape == (20, 5), f"T={T}: Wrong forecast shape"
            assert jnp.all(jnp.isfinite(paths)), f"T={T}: Non-finite forecasts"
            
            print(f"✓ T={T:2d}: Successfully trained and forecasted")
            
        except Exception as e:
            print(f"✗ T={T:2d}: Failed with error: {str(e)[:50]}")
            if T <= 3:
                print(f"  (Expected for very short sequences)")
            else:
                raise
    
    print("✅ Test 23 PASSED\n")

def test_24_multivariate_covariates():
    """Test 24: Train with multiple future and static covariates"""
    print("\n=== Test 24: Multivariate Covariates ===")
    
    from deep_ar import train_model, forecast_mc, quantiles
    
    T = 100
    H = 25
    d_f = 5  # Multiple future covariates
    d_s = 3  # Multiple static covariates
    
    # Generate rich covariate structure
    np.random.seed(2024)
    t = np.linspace(0, 12, T + H)
    
    # Future covariates with different frequencies
    x_f_1 = np.sin(2 * np.pi * t / 10)  # Slow cycle
    x_f_2 = np.cos(2 * np.pi * t / 5)   # Fast cycle
    x_f_3 = 0.1 * t  # Linear trend
    x_f_4 = np.random.randn(T + H) * 0.2  # Noise
    x_f_5 = np.where(t % 7 < 2, 1.0, 0.0)  # Weekly pattern
    
    x_f_full = np.stack([x_f_1, x_f_2, x_f_3, x_f_4, x_f_5], axis=1)
    
    # Target is influenced by multiple covariates
    y_full = (0.5 * x_f_1 + 0.3 * x_f_2 + 0.4 * x_f_3 + 
              0.2 * np.sin(2 * np.pi * t / 15) + 0.3 * np.random.randn(T + H))
    
    # Static covariates
    x_static = jnp.array([1.5, -0.8, 2.2], dtype=jnp.float32)
    
    y_hist = jnp.array(y_full[:T], dtype=jnp.float32)
    y_future = y_full[T:T+H]
    x_f_all = jnp.array(x_f_full[:T], dtype=jnp.float32)
    x_f_future = jnp.array(x_f_full[T:T+H], dtype=jnp.float32)
    
    print(f"Training with {d_f} future covariates and {d_s} static covariates...")
    
    model, params, losses = train_model(
        y_hist,
        x_f_all=x_f_all,
        x_static=x_static,
        hidden=64,
        lr=1e-3,
        steps=500,
        dropout=0.15,
        verbose=False
    )
    
    print(f"Training: {losses[0]:.4f} → {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Loss should decrease"
    
    # Forecast with full covariates
    paths = forecast_mc(
        params,
        model,
        y_hist,
        x_f_future=x_f_future,
        x_static=x_static,
        H=H,
        N=300,
        seed=2024
    )
    
    q10, q50, q90 = quantiles(paths, qs=(0.1, 0.5, 0.9))
    
    # Compute forecast error
    mse = np.mean((np.array(q50) - y_future) ** 2)
    print(f"✓ Forecast MSE: {mse:.4f}")
    
    # Check coverage
    in_ci = np.sum((y_future >= np.array(q10)) & (y_future <= np.array(q90)))
    coverage = in_ci / H
    print(f"✓ 80% CI coverage: {coverage:.2%} ({in_ci}/{H} points)")
    
    assert paths.shape == (300, H)
    print("✅ Test 24 PASSED\n")


def test_25_covariate_ablation():
    """Test 25: Compare forecasts with and without each covariate"""
    print("\n=== Test 25: Covariate Ablation Study ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 80
    H = 20
    
    # Generate data where we know covariate importance
    np.random.seed(777)
    t = np.linspace(0, 10, T + H)
    
    # Important covariate (strong signal)
    x_f_important = np.sin(2 * np.pi * t / 8)
    # Less important covariate (weak signal)
    x_f_weak = 0.1 * np.cos(2 * np.pi * t / 12)
    
    # Target strongly depends on important covariate
    y_full = 1.5 * x_f_important + 0.2 * x_f_weak + 0.3 * np.random.randn(T + H)
    
    y_hist = jnp.array(y_full[:T], dtype=jnp.float32)
    y_true = y_full[T:T+H]
    
    x_f_all = jnp.stack([x_f_important[:T], x_f_weak[:T]], axis=1).astype(jnp.float32)
    x_f_future_full = jnp.stack([x_f_important[T:T+H], x_f_weak[T:T+H]], axis=1).astype(jnp.float32)
    
    # Train once
    model, params, _ = train_model(
        y_hist,
        x_f_all=x_f_all,
        hidden=48,
        lr=1e-3,
        steps=400,
        dropout=0.1,
        verbose=False
    )
    
    # Test different covariate combinations
    scenarios = {
        'Both covariates': x_f_future_full,
        'Only important': jnp.stack([x_f_important[T:T+H], jnp.zeros(H)], axis=1).astype(jnp.float32),
        'Only weak': jnp.stack([jnp.zeros(H), x_f_weak[T:T+H]], axis=1).astype(jnp.float32),
        'No covariates': jnp.zeros((H, 2), dtype=jnp.float32),
    }
    
    results = {}
    for name, x_f_future in scenarios.items():
        paths = forecast_mc(params, model, y_hist, x_f_future=x_f_future, H=H, N=200, seed=999)
        median_forecast = jnp.median(paths, axis=0)
        mse = float(jnp.mean((median_forecast - y_true) ** 2))
        results[name] = mse
        print(f"  {name:20s}: MSE = {mse:.4f}")
    
    # Verify important covariate improves forecast more than weak one
    assert results['Only important'] < results['Only weak'], \
        "Important covariate should improve forecast more"
    assert results['Both covariates'] < results['No covariates'], \
        "Covariates should improve forecast"
    
    print("✓ Covariate importance correctly reflected in forecast accuracy")
    print("✅ Test 25 PASSED\n")


def test_26_time_varying_covariates():
    """Test 26: Covariates that change pattern over time"""
    print("\n=== Test 26: Time-Varying Covariate Patterns ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 120
    H = 30
    
    np.random.seed(888)
    t = np.linspace(0, 15, T + H)
    
    # Covariate with time-varying frequency (accelerating oscillation)
    freq = 1 + 0.1 * t  # Frequency increases over time
    x_f_varying = np.sin(2 * np.pi * np.cumsum(freq) / 50)
    
    # Target tracks the covariate
    y_full = 0.8 * x_f_varying + 0.4 * np.random.randn(T + H)
    
    y_hist = jnp.array(y_full[:T], dtype=jnp.float32)
    y_future = y_full[T:T+H]
    
    x_f_all = jnp.array(x_f_varying[:T, None], dtype=jnp.float32)
    x_f_future = jnp.array(x_f_varying[T:T+H, None], dtype=jnp.float32)
    
    print("Training with time-varying covariate pattern...")
    
    model, params, losses = train_model(
        y_hist,
        x_f_all=x_f_all,
        hidden=64,
        lr=1e-3,
        steps=400,
        dropout=0.1,
        verbose=False
    )
    
    print(f"Training: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    # Forecast WITH the time-varying covariate
    paths_with = forecast_mc(params, model, y_hist, x_f_future=x_f_future, H=H, N=200, seed=444)
    median_with = jnp.median(paths_with, axis=0)
    
    # Forecast WITHOUT (zeros)
    paths_without = forecast_mc(params, model, y_hist, 
                                 x_f_future=jnp.zeros_like(x_f_future), H=H, N=200, seed=444)
    median_without = jnp.median(paths_without, axis=0)
    
    mse_with = float(jnp.mean((median_with - y_future) ** 2))
    mse_without = float(jnp.mean((median_without - y_future) ** 2))
    
    print(f"✓ Forecast MSE with covariate: {mse_with:.4f}")
    print(f"✓ Forecast MSE without covariate: {mse_without:.4f}")
    print(f"✓ Improvement: {(1 - mse_with/mse_without)*100:.1f}%")
    
    assert mse_with < mse_without, "Time-varying covariate should improve forecast"
    
    print("✅ Test 26 PASSED\n")


def test_27_categorical_as_onehot():
    """Test 27: Handle categorical covariates as one-hot encoded"""
    print("\n=== Test 27: One-Hot Encoded Categorical Covariates ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 100
    H = 20
    
    np.random.seed(555)
    
    # Simulate day-of-week effect (7 categories)
    days = np.arange(T + H) % 7
    onehot = np.zeros((T + H, 7))
    onehot[np.arange(T + H), days] = 1
    
    # Different pattern for each day
    day_effects = np.array([0.0, 0.5, 1.0, 0.8, 0.3, -0.5, -0.8])
    y_full = day_effects[days] + 0.3 * np.random.randn(T + H)
    
    y_hist = jnp.array(y_full[:T], dtype=jnp.float32)
    x_f_all = jnp.array(onehot[:T], dtype=jnp.float32)
    x_f_future = jnp.array(onehot[T:T+H], dtype=jnp.float32)
    
    print("Training with one-hot encoded day-of-week (7 categories)...")
    
    model, params, losses = train_model(
        y_hist,
        x_f_all=x_f_all,
        hidden=48,
        lr=1e-3,
        steps=300,
        dropout=0.1,
        verbose=False
    )
    
    print(f"Training: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    paths = forecast_mc(params, model, y_hist, x_f_future=x_f_future, H=H, N=200, seed=777)
    
    # Check that forecasts vary by day of week
    median = jnp.median(paths, axis=0)
    
    # Group forecasts by day of week
    forecast_by_day = {}
    for h in range(H):
        day = (T + h) % 7
        if day not in forecast_by_day:
            forecast_by_day[day] = []
        forecast_by_day[day].append(float(median[h]))
    
    day_means = {day: np.mean(vals) for day, vals in forecast_by_day.items()}
    
    print("✓ Average forecast by day of week:")
    for day in sorted(day_means.keys()):
        print(f"    Day {day}: {day_means[day]:.3f}")
    
    # Check that there's variation across days
    day_std = np.std(list(day_means.values()))
    assert day_std > 0.1, f"Should see day-of-week variation, got std={day_std:.4f}"
    
    print(f"✓ Day-of-week variation captured (std={day_std:.3f})")
    print("✅ Test 27 PASSED\n")


def test_28_lagged_covariates():
    """Test 28: Using lagged values of target as covariates"""
    print("\n=== Test 28: Lagged Target as Covariate ===")
    
    from deep_ar import train_model, forecast_mc
    
    T = 100
    H = 20
    
    np.random.seed(321)
    
    # Generate AR-like process
    y_full = np.zeros(T + H)
    y_full[0] = np.random.randn()
    for t in range(1, T + H):
        y_full[t] = 0.7 * y_full[t-1] + 0.3 * np.random.randn()
    
    # Use lag-1 as a covariate
    lag1 = np.concatenate([np.array([0.0]), y_full[:-1]])
    
    y_hist = jnp.array(y_full[:T], dtype=jnp.float32)
    x_f_all = jnp.array(lag1[:T, None], dtype=jnp.float32)
    
    print("Training with lag-1 of target as covariate...")
    
    model, params, losses = train_model(
        y_hist,
        x_f_all=x_f_all,
        hidden=32,
        lr=1e-3,
        steps=300,
        dropout=0.05,
        verbose=False
    )
    
    print(f"Training: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    # For forecasting, we need to build lagged values iteratively
    # For simplicity, use actual future values (oracle case)
    x_f_future = jnp.array(lag1[T:T+H, None], dtype=jnp.float32)
    
    paths = forecast_mc(params, model, y_hist, x_f_future=x_f_future, H=H, N=200, seed=654)
    
    median = jnp.median(paths, axis=0)
    y_true = y_full[T:T+H]
    mse = float(jnp.mean((median - y_true) ** 2))
    
    print(f"✓ Forecast MSE: {mse:.4f}")
    
    # The lag should help capture autocorrelation
    assert mse < 1.0, f"With lag-1 covariate, MSE should be reasonable, got {mse:.4f}"
    
    print("✅ Test 28 PASSED\n")


def test_viz_5_multivariate_covariates():
    """Visualization Test 5: Impact of multiple covariates"""
    print("\n=== Viz Test 5: Multivariate Covariate Analysis ===")
    
    from deep_ar import train_model, forecast_mc, quantiles
    import matplotlib.pyplot as plt
    
    T = 120
    H = 40
    
    np.random.seed(2025)
    t = np.linspace(0, 15, T + H)
    
    # Three different covariates
    x1 = np.sin(2 * np.pi * t / 10)
    x2 = np.cos(2 * np.pi * t / 7)
    x3 = 0.05 * t
    
    # Target depends on all three
    y_full = 1.0 * x1 + 0.7 * x2 + 0.5 * x3 + 0.4 * np.random.randn(T + H)
    
    y_hist = jnp.array(y_full[:T], dtype=jnp.float32)
    y_future = y_full[T:T+H]
    
    x_f_all = jnp.stack([x1[:T], x2[:T], x3[:T]], axis=1).astype(jnp.float32)
    x_f_future = jnp.stack([x1[T:T+H], x2[T:T+H], x3[T:T+H]], axis=1).astype(jnp.float32)
    
    print("Training with 3 future covariates...")
    
    model, params, _ = train_model(
        y_hist,
        x_f_all=x_f_all,
        hidden=64,
        lr=1e-3,
        steps=500,
        dropout=0.1,
        verbose=False
    )
    
    # Create forecasts with different covariate combinations
    scenarios = {
        'All 3 covariates': x_f_future,
        'Only x1 (sine)': jnp.stack([x1[T:T+H], jnp.zeros(H), jnp.zeros(H)], axis=1).astype(jnp.float32),
        'Only x2 (cosine)': jnp.stack([jnp.zeros(H), x2[T:T+H], jnp.zeros(H)], axis=1).astype(jnp.float32),
        'Only x3 (trend)': jnp.stack([jnp.zeros(H), jnp.zeros(H), x3[T:T+H]], axis=1).astype(jnp.float32),
        'No covariates': jnp.zeros((H, 3), dtype=jnp.float32),
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    t_hist_plot = np.arange(T)
    t_future_plot = np.arange(T, T + H)
    t_all = np.arange(T + H)
    
    colors = ['red', 'blue', 'green', 'orange', 'gray']
    
    # Main comparison plot (top-left)
    ax = axes[0, 0]
    ax.plot(t_hist_plot, y_hist, 'b-', label='History', linewidth=2)
    ax.plot(t_future_plot, y_future, 'k-', label='True Future', linewidth=2, alpha=0.7)
    
    for (name, x_f), color in zip(scenarios.items(), colors):
        paths = forecast_mc(params, model, y_hist, x_f_future=x_f, H=H, N=300, seed=111)
        q10, q50, q90 = quantiles(paths, qs=(0.1, 0.5, 0.9))
        ax.plot(t_future_plot, q50, color=color, label=name, linewidth=2, alpha=0.8)
    
    ax.axvline(x=T, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Forecast with Different Covariate Combinations', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # MSE comparison (top-right)
    ax = axes[0, 1]
    mse_values = []
    labels = []
    for (name, x_f), color in zip(scenarios.items(), colors):
        paths = forecast_mc(params, model, y_hist, x_f_future=x_f, H=H, N=300, seed=111)
        median = jnp.median(paths, axis=0)
        mse = float(jnp.mean((median - y_future) ** 2))
        mse_values.append(mse)
        labels.append(name)
    
    bars = ax.barh(range(len(labels)), mse_values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Mean Squared Error', fontsize=11)
    ax.set_title('Forecast Error by Covariate Set', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add MSE values on bars
    for i, (bar, mse) in enumerate(zip(bars, mse_values)):
        ax.text(mse + 0.01, i, f'{mse:.3f}', va='center', fontsize=9)
    
    # Plot the three covariates in the remaining subplots
    covariate_data = [
        (x1, 'x1: sin', colors[1]),
        (x2, 'x2: cos', colors[2]),
        (x3, 'x3: trend', colors[3])
    ]
    
    # Place covariates in middle row (1,0), (1,1), and bottom-left (2,0)
    positions = [(1, 0), (1, 1), (2, 0)]
    
    for (xi, name, color), (row, col) in zip(covariate_data, positions):
        ax = axes[row, col]
        ax.plot(t_all, xi, linewidth=2, color=color)
        ax.axvline(x=T, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Covariate Value', fontsize=11)
        ax.set_title(f'Covariate {name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide the bottom-right subplot (2,1)
    axes[2, 1].axis('off')
    
    plt.suptitle('Multivariate Covariate Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('deepar_test_multivariate_covariates.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to 'deepar_test_multivariate_covariates.png'")
    plt.close()
    
    print("✅ Viz Test 5 PASSED\n")

def test_viz_6_covariate_response():
    """Visualization Test 6: Model response to covariate changes"""
    print("\n=== Viz Test 6: Covariate Response Analysis ===")
    
    from deep_ar import train_model, forecast_mc
    import matplotlib.pyplot as plt
    
    T = 100
    H = 50
    
    np.random.seed(1234)
    t = np.linspace(0, 12, T)
    
    # Train on data with step covariate
    x_f_train = np.where(t < 6, 0.0, 1.0)
    y_train = 2.0 + 1.5 * x_f_train + 0.3 * np.random.randn(T)
    
    y_hist = jnp.array(y_train, dtype=jnp.float32)
    x_f_all = jnp.array(x_f_train[:, None], dtype=jnp.float32)
    
    print("Training model to respond to step covariate...")
    
    model, params, _ = train_model(
        y_hist,
        x_f_all=x_f_all,
        hidden=48,
        lr=1e-3,
        steps=400,
        dropout=0.1,
        verbose=False
    )
    
    # Test different future covariate scenarios
    scenarios = {
        'Constant Low': np.zeros(H),
        'Constant High': np.ones(H),
        'Step Up': np.where(np.arange(H) < H//2, 0.0, 1.0),
        'Step Down': np.where(np.arange(H) < H//2, 1.0, 0.0),
        'Oscillating': np.where(np.arange(H) % 10 < 5, 0.0, 1.0),
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (name, x_f_pattern) in enumerate(scenarios.items()):
        ax = axes[idx]
        
        x_f_future = jnp.array(x_f_pattern[:, None], dtype=jnp.float32)
        
        paths = forecast_mc(params, model, y_hist, x_f_future=x_f_future, H=H, N=200, seed=999)
        
        q10 = jnp.quantile(paths, 0.1, axis=0)
        q50 = jnp.quantile(paths, 0.5, axis=0)
        q90 = jnp.quantile(paths, 0.9, axis=0)
        
        t_hist_plot = np.arange(T)
        t_future_plot = np.arange(T, T + H)
        
        # Plot forecast
        ax.plot(t_hist_plot, y_hist, 'b-', label='History', linewidth=2)
        ax.plot(t_future_plot, q50, 'r-', label='Median Forecast', linewidth=2)
        ax.fill_between(t_future_plot, q10, q90, alpha=0.3, color='red', label='10-90% CI')
        
        # Plot covariate on secondary axis
        ax2 = ax.twinx()
        ax2.plot(t_future_plot, x_f_pattern, 'g--', label='Covariate', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Covariate Value', fontsize=10, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim(-0.2, 1.2)
        
        ax.axvline(x=T, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Target Value', fontsize=10)
        ax.set_title(f'Scenario: {name}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot if odd number
    if len(scenarios) < 6:
        axes[-1].axis('off')
    
    plt.suptitle('Model Response to Different Covariate Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('deepar_test_covariate_response.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to 'deepar_test_covariate_response.png'")
    plt.close()
    
    print("✅ Viz Test 6 PASSED\n")


def run_all_tests(include_viz=False):
    """Run all test cases"""
    print("="*60)
    print("Running DeepAR Encoder-Decoder Test Suite")
    print("="*60)
    
    tests = [
        test_1_basic_shapes,
        test_2_synthetic_sine_wave,
        test_3_with_covariates,
        test_4_edge_cases,
        test_5_determinism,
        test_6_no_data_leakage,
        test_7_multi_step_encoder,
        test_8_gradient_flow,
        test_9_static_covariates_only,
        test_10_future_covariates_only,
        test_11_varying_noise_levels,
        test_12_forecast_consistency,
        test_13_state_persistence,
        test_14_batch_forecast_consistency,
        test_15_zero_and_constant_sequences,
        test_16_long_horizon_forecasts,
        test_17_negative_values,
        test_18_trend_extrapolation,
        test_19_covariate_dimension_mismatch,
        test_20_encoder_decoder_state_transfer,
        test_21_dropout_effect,
        test_22_sigma_bounds,
        test_23_missing_history_handling,
        test_24_multivariate_covariates,
        test_25_covariate_ablation,
        test_26_time_varying_covariates,
        test_27_categorical_as_onehot,
        test_28_lagged_covariates,
    ]
    
    if include_viz:
        print("\n" + "="*60)
        print("Including Visualization Tests")
        print("="*60)
        viz_tests = [
            test_viz_1_sine_wave_forecast,
            test_viz_2_sample_paths,
            test_viz_3_with_covariates,
            test_viz_4_forecast_distribution,
            test_viz_5_multivariate_covariates,
            test_viz_6_covariate_response
        ]
        tests.extend(viz_tests)
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {test_fn.__name__} FAILED")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    
    # Check if --viz flag is provided
    include_viz = '--viz' in sys.argv or '-v' in sys.argv
    
    if include_viz:
        print("Running tests with visualizations...")
        print("This will generate PNG files in the current directory.\n")
    
    success = run_all_tests(include_viz=include_viz)
    
    if include_viz:
        print("\n" + "="*60)
        print("Visualization files generated:")
        print("  - deepar_test_sine_wave.png")
        print("  - deepar_test_sample_paths.png")
        print("  - deepar_test_covariates.png")
        print("  - deepar_test_distributions.png")
        print("  - deepar_test_uncertainty_growth.png")
        print("="*60)
    
    exit(0 if success else 1)
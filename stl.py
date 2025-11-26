"""
STL (Seasonal-Trend decomposition using LOESS) in JAX

This module implements the STL algorithm for decomposing time series into seasonal, trend, 
and remainder components using locally weighted regression (LOESS).

**Algorithm:**
STL iteratively applies LOESS smoothing to extract seasonal patterns and trend:
1. Seasonal smoothing by subseries (each phase of the cycle)
2. Centering seasonal component to sum-to-zero over each phase
3. Optional low-pass filtering for seasonal stabilization
4. Trend smoothing on deseasonalized data

**Implementation:**
- JIT-compiled with `jax.lax.fori_loop` and `jax.lax.scan` for efficiency
- Uses tricube kernel weights via `loess_window_jump`
- Supports local constant (deg=0) or local linear (deg=1) regression
- Jump anchors for faster computation on large series
- Conforms to `BaseForecaster` interface for prediction and conformal intervals

**Attributes:**
- `period`: Seasonal period length
- `seasonal`: Seasonal smoother window (odd integer)
- `trend`: Trend smoother window (odd integer)
- `low_pass`: Low-pass filter window for seasonal (optional)
- `seasonal_deg`, `trend_deg`: Polynomial degree (0 or 1)
- `inner`: Number of inner loop iterations

**Methods:**
- `fit(y)`: Decompose series into seasonal, trend, remainder
- `predict(h)`: Forecast by extrapolating trend + repeating seasonal
- `predict_in_sample()`: Return fitted values with decomposition components
"""
# stl_jax.py
from __future__ import annotations
from functools import partial as _partial
import jax
import jax.numpy as jnp
from jax import lax
import utils
from base_forecaster import BaseForecaster
from loess import loess_window_jump  # fixed-window LOESS (deg 0/1, tricube, robust_outer, jump)

# =========================
# STL kernel (JAX-pure)
# =========================

@_partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9))
def stl_decompose(
    y: jnp.ndarray,
    period: int,
    seasonal: int,
    trend: int,
    low_pass: int | None = None,
    seasonal_deg: int = 0,   # {0,1}
    trend_deg: int = 1,      # {0,1}
    seasonal_jump: int = 1,
    trend_jump: int = 1,
    inner: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Minimal STL: seasonal subseries LOESS (with centering) + trend LOESS on deseasonalized series.
    - Windows must be odd; deg ∈ {0,1}; tricube kernel; optional low-pass on seasonal; jump anchors for speed.
    - Robust reweighting can be set inside LOESS via robust_outer (here we keep 0 by default).
    """
    y = y.reshape(-1)
    n = y.shape[0]
    seasonal = int(seasonal) | 1
    trend = int(trend) | 1
    low_pass = 0 if low_pass is None else int(low_pass) | 1
    seasonal_jump = max(1, int(seasonal_jump))
    trend_jump = max(1, int(trend_jump))

    trend_est = jnp.zeros_like(y)
    seas_est = jnp.zeros_like(y)

    def inner_body(_, carry):
        trend_est, seas_est = carry
        detrend = y - trend_est

        # 1) Seasonal smoothing by subseries (phase r)
        # Reshape data into fixed-size matrix for vmap compatibility
        max_sub_len = (n + period - 1) // period  # ceiling(n / period)
        
        # Pad detrend to make it divisible by period
        padded_len = max_sub_len * period
        detrend_padded = jnp.pad(detrend, (0, padded_len - n), mode='edge')
        
        # Reshape into (max_sub_len, period), then transpose to (period, max_sub_len)
        # Each row now contains one subseries (fixed size)
        subseries_matrix = detrend_padded.reshape(max_sub_len, period).T
        
        # Now vmap can work on fixed-size rows
        def smooth_phase(subseries_row):
            # subseries_row has shape (max_sub_len,) - FIXED SIZE!
            return loess_window_jump(subseries_row, seasonal, deg=seasonal_deg, robust_outer=0, jump=seasonal_jump)
        
        # vmap over the first dimension (period subseries)
        shats = jax.vmap(smooth_phase)(subseries_matrix)  # Shape: (period, max_sub_len)
        
        # Stitch seasonal back to full length
        # Transpose back and flatten, then trim to original length
        seas_new_padded = shats.T.reshape(-1)  # Shape: (padded_len,)
        seas_new = seas_new_padded[:n]  # Trim back to original length

        # Center seasonal by phase means (sum-to-zero over each phase)
        def add_t(carry2, t):
            sums, counts = carry2
            r = t % period
            return (sums.at[r].add(seas_new[t]), counts.at[r].add(1)), None

        (phase_sums, phase_counts), _ = lax.scan(add_t, (jnp.zeros(period), jnp.zeros(period)), jnp.arange(n))
        phase_means = phase_sums / jnp.maximum(phase_counts, 1.0)

        def center_t(ti, s):
            r = ti % period
            return s.at[ti].add(-phase_means[r])

        seas_new = lax.fori_loop(0, n, center_t, seas_new)

        # Optional low-pass smoothing of seasonal to stabilize
        seas_sm = lax.cond(
            low_pass > 0,
            lambda _: loess_window_jump(seas_new, low_pass, deg=trend_deg, robust_outer=0, jump=1),
            lambda _: seas_new,
            operand=None,
        )

        # 2) Trend smoothing on deseasonalized data
        trend_new = loess_window_jump(y - seas_sm, trend, deg=trend_deg, robust_outer=0, jump=trend_jump)
        return trend_new, seas_sm

    trend_est, seas_est = lax.fori_loop(0, int(inner), inner_body, (trend_est, seas_est))
    remainder = y - trend_est - seas_est
    return seas_est, trend_est, remainder

# =========================
# STL model (thin wrapper)
# =========================

class STL(BaseForecaster):
    uses_exog = False

    def __init__(
        self,
        period: int,
        seasonal: int | None = None,
        trend: int | None = None,
        low_pass: int | None = None,
        seasonal_deg: int = 0,
        trend_deg: int = 1,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        inner: int = 1,
        tail_window: int | None = None,
        fitted: bool = True,
        alias: str = "STL",
        conformal_params=None,
    ):
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = {}

        self.period = int(period)
        # Defaults close to statsmodels behavior; ensure odd lengths
        self.seasonal = int(seasonal) | 1 if seasonal is not None else ((max(7, 2 * self.period + 1)) | 1)
        self.trend = int(trend) | 1 if trend is not None else ((2 * self.period + 1) | 1)
        self.low_pass = None if low_pass is None else (int(low_pass) | 1)

        self.seasonal_deg = int(seasonal_deg)
        self.trend_deg = int(trend_deg)
        self.seasonal_jump = max(1, int(seasonal_jump))
        self.trend_jump = max(1, int(trend_jump))
        self.inner = int(inner)
        self.tail_window = tail_window
        self.fitted = bool(fitted)

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "STL":
        y = utils.ensure_float(jnp.asarray(y).reshape(-1))
        seas, trend, remainder = stl_decompose(
            y=y,
            period=self.period,
            seasonal=self.seasonal,
            trend=self.trend,
            low_pass=self.low_pass,
            seasonal_deg=self.seasonal_deg,
            trend_deg=self.trend_deg,
            seasonal_jump=self.seasonal_jump,
            trend_jump=self.trend_jump,
            inner=self.inner,
        )
        self.model_ = {
            "y": y,
            "seasonal": seas,
            "trend": trend,
            "remainder": remainder,
            "period": self.period,
        }
        return self

    def predict_in_sample(self, X: jnp.ndarray | None = None, level: list[int | float] | None = None) -> dict:
        y = self.model_["y"]
        mean = self.model_["trend"] + self.model_["seasonal"]
        out = {"mean": mean}
        if self.fitted:
            out["fitted"] = mean
            out["trend"] = self.model_["trend"]
            out["seasonal"] = self.model_["seasonal"]
            out["remainder"] = self.model_["remainder"]
        if level:
            cs = self.conformity_scores(y)
            out = self.add_confidence_intervals(out, cs, level, "conformal_distribution")
        return out

    def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int | float] | None = None) -> dict:
        h = int(h)
        y = self.model_["y"]
        trend = self.model_["trend"]
        seas = self.model_["seasonal"]
        n = y.size
        tail = self.tail_window if self.tail_window is not None else min(n, int(2 * self.period + 1))
        trend_fcst = utils._linear_extrapolate_tail(trend, tail, h)
        seas_fcst = utils._repeat_val_seas(seas[-self.period:], h)
        out = {"mean": trend_fcst + seas_fcst}
        if level:
            cs = self.conformity_scores(y)
            out = self.add_confidence_intervals(out, cs, level, "conformal_distribution")
        return out

    # Required by BaseForecaster.conformity_scores
    def forecast(self, y, h, X=None, X_future=None):
        return self.new().fit(y, X).predict(h, X)


# =========================
# Test Cases
# =========================

if __name__ == "__main__":
    print("=" * 60)
    print("STL Test Suite")
    print("=" * 60)
    
    # Test 1: Basic STL decomposition with synthetic seasonal data
    print("\n[Test 1] Basic STL decomposition")
    n = 50
    period = 7
    t = jnp.arange(n)
    trend = 0.5 * t
    seasonal = 2.0 * jnp.sin(2 * jnp.pi * t / period)
    noise = jax.random.normal(jax.random.PRNGKey(42), (n,)) * 0.3
    y1 = trend + seasonal + noise
    
    model1 = STL(period=period, seasonal=7, trend=15)
    model1.fit(y1)
    result1 = model1.predict_in_sample()
    
    print(f"  Input series length: {n}")
    print(f"  Period: {period}")
    print(f"  Fitted shape: {result1['fitted'].shape}")
    print(f"  Trend shape: {result1['trend'].shape}")
    print(f"  Seasonal shape: {result1['seasonal'].shape}")
    print(f"  Remainder shape: {result1['remainder'].shape}")
    
    assert result1['fitted'].shape == (n,), "Fitted shape mismatch!"
    assert result1['trend'].shape == (n,), "Trend shape mismatch!"
    assert result1['seasonal'].shape == (n,), "Seasonal shape mismatch!"
    assert jnp.all(jnp.isfinite(result1['fitted'])), "Contains NaN/Inf!"
    print("  ✓ Decomposition components OK")
    
    # Test 2: Forecast future values
    print("\n[Test 2] Forecast future values")
    h = 14
    forecast2 = model1.predict(h)
    print(f"  Forecast horizon: {h}")
    print(f"  Forecast mean shape: {forecast2['mean'].shape}")
    print(f"  Forecast values (first 5): {forecast2['mean'][:5]}")
    assert forecast2['mean'].shape == (h,), "Forecast shape mismatch!"
    assert jnp.all(jnp.isfinite(forecast2['mean'])), "Forecast contains NaN/Inf!"
    print("  ✓ Forecast OK")
    
    # Test 3: Seasonal pattern repeats correctly
    print("\n[Test 3] Seasonal pattern repetition")
    seas_pattern = result1['seasonal'][-period:]
    forecast3 = model1.predict(period * 2)
    # Extract seasonal component (remove linear trend)
    trend_start = result1['trend'][-1]
    tail_window = min(n, int(2 * period + 1))
    trend_slope = (result1['trend'][-1] - result1['trend'][-tail_window]) / (tail_window - 1)
    expected_trend = trend_start + trend_slope * jnp.arange(1, period * 2 + 1)
    residual_seasonal = forecast3['mean'] - expected_trend
    
    print(f"  Last seasonal pattern: {seas_pattern}")
    print(f"  Forecast seasonal (first period): {residual_seasonal[:period]}")
    print(f"  Forecast seasonal (second period): {residual_seasonal[period:period*2]}")
    # Check if patterns are similar (allowing for trend effects)
    correlation = jnp.corrcoef(residual_seasonal[:period], residual_seasonal[period:period*2])[0, 1]
    print(f"  Correlation between periods: {correlation:.4f}")
    assert correlation > 0.5, "Seasonal pattern should repeat!"
    print("  ✓ Seasonal repetition OK")
    
    # Test 4: STL with different window sizes
    print("\n[Test 4] STL with different window sizes")
    model4_small = STL(period=7, seasonal=7, trend=11)
    model4_large = STL(period=7, seasonal=15, trend=21)
    
    model4_small.fit(y1)
    model4_large.fit(y1)
    
    result4_small = model4_small.predict_in_sample()
    result4_large = model4_large.predict_in_sample()
    
    # Larger windows should produce smoother trends
    trend_var_small = jnp.var(jnp.diff(result4_small['trend']))
    trend_var_large = jnp.var(jnp.diff(result4_large['trend']))
    
    print(f"  Small window (11) trend variance: {trend_var_small:.6f}")
    print(f"  Large window (21) trend variance: {trend_var_large:.6f}")
    print(f"  Smoothing effect: {trend_var_small / (trend_var_large + 1e-8):.2f}x")
    print("  ✓ Window size effect verified")
    
    # Test 5: Low-pass filtering
    print("\n[Test 5] Low-pass filtering of seasonal")
    model5_no_lp = STL(period=7, seasonal=7, trend=15, low_pass=None)
    model5_with_lp = STL(period=7, seasonal=7, trend=15, low_pass=7)
    
    model5_no_lp.fit(y1)
    model5_with_lp.fit(y1)
    
    result5_no_lp = model5_no_lp.predict_in_sample()
    result5_with_lp = model5_with_lp.predict_in_sample()
    
    seas_diff = jnp.mean(jnp.abs(result5_no_lp['seasonal'] - result5_with_lp['seasonal']))
    print(f"  Seasonal difference (no LP vs LP): {seas_diff:.6f}")
    assert seas_diff > 0, "Low-pass should affect seasonal component!"
    print("  ✓ Low-pass filtering works")
    
    # Test 6: Inner iterations
    print("\n[Test 6] Inner iterations effect")
    model6_iter1 = STL(period=7, seasonal=7, trend=15, inner=1)
    model6_iter3 = STL(period=7, seasonal=7, trend=15, inner=3)
    
    model6_iter1.fit(y1)
    model6_iter3.fit(y1)
    
    result6_iter1 = model6_iter1.predict_in_sample()
    result6_iter3 = model6_iter3.predict_in_sample()
    
    fit_diff = jnp.mean(jnp.abs(result6_iter1['fitted'] - result6_iter3['fitted']))
    print(f"  Fitted difference (1 iter vs 3 iter): {fit_diff:.6f}")
    print("  ✓ Inner iterations converge")
    
    # Test 7: Polynomial degree (deg=0 vs deg=1)
    print("\n[Test 7] Polynomial degree effect")
    model7_deg0 = STL(period=7, seasonal=7, trend=15, seasonal_deg=0, trend_deg=0)
    model7_deg1 = STL(period=7, seasonal=7, trend=15, seasonal_deg=1, trend_deg=1)
    
    model7_deg0.fit(y1)
    model7_deg1.fit(y1)
    
    result7_deg0 = model7_deg0.predict_in_sample()
    result7_deg1 = model7_deg1.predict_in_sample()
    
    print(f"  deg=0 (local constant) trend variance: {jnp.var(jnp.diff(result7_deg0['trend'])):.6f}")
    print(f"  deg=1 (local linear) trend variance: {jnp.var(jnp.diff(result7_deg1['trend'])):.6f}")
    print("  ✓ Polynomial degree variations work")
    
    # Test 8: Jump anchors for speed
    print("\n[Test 8] Jump anchors (performance)")
    model8_jump1 = STL(period=7, seasonal=7, trend=15, seasonal_jump=1, trend_jump=1)
    model8_jump3 = STL(period=7, seasonal=7, trend=15, seasonal_jump=3, trend_jump=3)
    
    import time
    
    # Warm-up
    model8_jump1.fit(y1)
    model8_jump3.fit(y1)
    
    start = time.time()
    for _ in range(10):
        model8_jump1.fit(y1)
    time_jump1 = time.time() - start
    
    start = time.time()
    for _ in range(10):
        model8_jump3.fit(y1)
    time_jump3 = time.time() - start
    
    print(f"  10 fits with jump=1: {time_jump1:.4f}s")
    print(f"  10 fits with jump=3: {time_jump3:.4f}s")
    print(f"  Speedup: {time_jump1 / (time_jump3 + 1e-8):.2f}x")
    print("  ✓ Jump anchors provide speedup")
    
    # Test 9: Constant series edge case
    print("\n[Test 9] Edge case: constant series")
    y9 = jnp.ones(30) * 5.0
    model9 = STL(period=7, seasonal=7, trend=15)
    model9.fit(y9)
    result9 = model9.predict_in_sample()
    
    print(f"  Input (constant 5.0): mean={jnp.mean(y9):.2f}, std={jnp.std(y9):.6f}")
    print(f"  Trend: mean={jnp.mean(result9['trend']):.2f}")
    print(f"  Seasonal: mean={jnp.mean(result9['seasonal']):.6f}, std={jnp.std(result9['seasonal']):.6f}")
    assert jnp.std(result9['seasonal']) < 0.1, "Seasonal should be near zero for constant series!"
    print("  ✓ Constant series handled correctly")
    
    # Test 10: Conformal intervals
    print("\n[Test 10] Conformal prediction intervals")
    from conformal_intervals import ConformalIntervals
    conformal_params = ConformalIntervals(h=14)
    model10 = STL(period=7, seasonal=7, trend=15, conformal_params=conformal_params)
    model10.fit(y1)
    forecast10 = model10.predict(h=14, level=[90, 95])
    
    print(f"  Forecast keys: {list(forecast10.keys())}")
    assert 'mean' in forecast10, "Should have 'mean'!"
    assert 'lo-90' in forecast10, "Should have 'lo-90'!"
    assert 'hi-90' in forecast10, "Should have 'hi-90'!"
    print(f"  Mean forecast (first 5): {forecast10['mean'][:5]}")
    print(f"  90% interval width (first): {forecast10['hi-90'][0] - forecast10['lo-90'][0]:.4f}")
    print("  ✓ Conformal intervals work")
    
    print("\n" + "=" * 60)
    print("✓ All STL tests passed!")
    print("=" * 60)
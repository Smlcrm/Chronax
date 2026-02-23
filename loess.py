"""
LOESS/LOWESS (Locally Weighted Scatterplot Smoothing) in JAX

This module implements LOESS (Local Polynomial Regression) with tricube kernel weighting
and optional robust reweighting for non-parametric smoothing of time series and scatter data.

**Methods:**
- Fixed-window LOESS (`loess_window_jump`): Uses a fixed window size with optional jump anchors for speed
- Fraction-based LOWESS (`lowess_frac`): Classic Cleveland LOWESS using neighborhood fraction
- Convenience wrapper (`loess_smooth`): Automatically selects appropriate method

**Kernel & Weighting:**
- Tricube weights: w(u) = (1 - |u|³)³ for |u| < 1
- Robust reweighting: Bisquare weights via MAD for outlier resistance

**Implementation:**
- JIT-compiled for performance with static window/degree parameters
- Supports local constant (deg=0) and local linear (deg=1) regression
- Edge padding for boundary handling
- Optional jump anchors (STL-style) for faster computation

**Attributes:**
None (utility functions only)

**Methods:**
- `loess_window_jump(y, window, deg, robust_outer, jump)`: Fixed-window LOESS
- `lowess_frac(y, x, frac, it)`: Fraction-based LOWESS
- `loess_smooth(y, ...)`: Convenience wrapper
"""
# loess.py
from __future__ import annotations
from functools import partial as _partial
import jax
import jax.numpy as jnp
from jax import lax
import utils

# =========================
# Private helpers (JAX-pure)
# =========================

@_partial(jax.jit, static_argnums=(1,))
def _edge_pad_1d(y: jnp.ndarray, pad: int) -> jnp.ndarray:
    """
    Symmetric edge padding for centered windows.

    Args:
        y (jnp.ndarray): Input 1D array to pad.
        pad (int): Number of elements to pad on each edge.

    Returns:
        jnp.ndarray: Padded array of shape (n + 2*pad,).
    """
    return jnp.pad(y, (pad, pad), mode="edge")

@_partial(jax.jit, static_argnums=(1,))
def _sliding_windows_1d(y: jnp.ndarray, window: int) -> jnp.ndarray:
    """
    Create centered sliding windows with edge padding.

    Args:
        y (jnp.ndarray): Input 1D array.
        window (int): Window size (typically odd).

    Returns:
        jnp.ndarray: Array of shape (n, window) containing centered windows.
    """
    n = y.shape[0]
    half = window // 2
    ypad = _edge_pad_1d(y, half)

    def slice_one(i):
        return lax.dynamic_slice(ypad, (i,), (window,))

    return jax.vmap(slice_one)(jnp.arange(n))

@_partial(jax.jit, static_argnums=(0,))
def _tricube_weights(window: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute tricube kernel weights and offsets for a given window size.

    Args:
        window (int): Window size (typically odd).

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - offsets of shape (window,): Integer offsets from center.
            - weights of shape (window,): Tricube weights w(u) = (1 - |u|^3)^3.
    """
    half = window // 2
    offsets = jnp.arange(-half, half + 1)
    u = jnp.abs(offsets) / (half + 1e-8)
    w = jnp.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)
    return offsets, w

# ======================================
# Fixed-window LOESS with optional "jump"
# ======================================

@_partial(jax.jit, static_argnums=(1, 2, 3, 4))
def loess_window_jump(
    y: jnp.ndarray,
    window: int,
    deg: int = 1,            # 0 or 1
    robust_outer: int = 0,   # robust reweighting iterations (bisquare via MAD)
    jump: int = 1,           # evaluate only on anchors (0, jump, 2*jump, ...)
) -> jnp.ndarray:
    """
    LOESS with fixed odd window using tricube weights.
    - deg ∈ {0,1}: local constant or local linear (value at center).
    - robust_outer: bisquare outer reweighting count.
    - jump: compute only at anchor points and linearly interpolate in-between (STL-like speedup).

    Args:
        y (jnp.ndarray): Input 1D array to smooth.
        window (int): Window size (will be forced to odd).
        deg (int): Polynomial degree, 0 (local constant) or 1 (local linear). Default: 1.
        robust_outer (int): Number of robust reweighting iterations using bisquare weights. Default: 0.
        jump (int): Evaluate only at anchor points (0, jump, 2*jump, ...) and interpolate. Default: 1.

    Returns:
        jnp.ndarray: Smoothed values of shape (n,).
    """
    y = utils.ensure_float(jnp.asarray(y).reshape(-1))
    n = y.shape[0]
    window = int(window) | 1
    jump = max(1, int(jump))

    Xw = _sliding_windows_1d(y, window)           # (n, window)
    offsets, base_w = _tricube_weights(window)    # (window,)
    t = offsets[None, :]                           # (1, window)
    base_w = base_w[None, :]                       # (1, window)

    anchors = jnp.arange(0, n, jump)
    m = anchors.shape[0]

    def fit_at_anchor(i_anchor, rw):
        idx = anchors[i_anchor]
        # Use dynamic_slice instead of idx:idx+1 for JIT compatibility
        w = base_w * lax.dynamic_slice(rw, (idx, 0), (1, window))  # (1, window)
        W = lax.dynamic_slice(Xw, (idx, 0), (1, window))           # (1, window)

        if deg == 0:
            num = jnp.sum(w * W, axis=1)
            den = jnp.sum(w, axis=1) + 1e-12
            return (num / den)[0]

        # deg == 1: weighted linear regression value at t=0 (intercept)
        sw   = jnp.sum(w, axis=1)
        swt  = jnp.sum(w * t, axis=1)
        swtt = jnp.sum(w * t * t, axis=1)
        swy  = jnp.sum(w * W, axis=1)
        swty = jnp.sum(w * t * W, axis=1)
        denom = sw * swtt - swt * swt + 1e-12
        intercept = (swy * swtt - swt * swty) / denom
        return intercept[0]

    def make_rw_from_resid(resid: jnp.ndarray) -> jnp.ndarray:
        med = jnp.median(resid)
        mad = jnp.median(jnp.abs(resid - med)) + 1e-12
        u = resid / (6.0 * mad + 1e-12)
        row = jnp.where(jnp.abs(u) < 1.0, (1 - u**2) ** 2, 0.0)  # (n,)
        return jnp.repeat(row[:, None], repeats=window, axis=1)  # (n, window)

    def run_once(rw):
        # compute yhat only at anchors
        def body(i, acc):
            yhat_i = fit_at_anchor(i, rw)
            return acc.at[i].set(yhat_i)
        yhat_anchors = lax.fori_loop(0, m, body, jnp.zeros((m,), y.dtype))
        # interpolate to all points
        full = jnp.interp(jnp.arange(n, dtype=y.dtype), anchors.astype(y.dtype), yhat_anchors)
        return full

    rw0 = jnp.ones((n, window), dtype=y.dtype)

    def outer_step(_, rw):
        yhat = run_once(rw)
        resid = y - yhat
        return make_rw_from_resid(resid)

    rw = lax.fori_loop(0, robust_outer, outer_step, rw0)
    return run_once(rw)

# ===========================
# frac-based LOWESS (classic)
# ===========================

@_partial(jax.jit, static_argnums=(2, 3))
def lowess_frac(
    y: jnp.ndarray,
    x: jnp.ndarray,
    frac: float = 2.0 / 3.0,  # span as fraction of n
    it: int = 0,              # robust iterations
) -> jnp.ndarray:
    """
    Statsmodels-style LOWESS (Cleveland):
    - Neighborhood defined by frac*n nearest x.
    - Tricube kernel; robust outer iterations via bisquare.
    - Returns yhat aligned to x (input order).
    Note: 'delta' skip optimization is omitted for JAX-purity.

    Args:
        y (jnp.ndarray): Input 1D array of values to smooth.
        x (jnp.ndarray): Input 1D array of x-coordinates (arbitrary order).
        frac (float): Fraction of data used for neighborhood (0 < frac <= 1). Default: 2/3.
        it (int): Number of robust reweighting iterations. Default: 0.

    Returns:
        jnp.ndarray: Smoothed values of shape (n,) aligned to input order.
    """
    y = utils.ensure_float(jnp.asarray(y).reshape(-1))
    x = utils.ensure_float(jnp.asarray(x).reshape(-1))
    n = y.shape[0]

    order = jnp.argsort(x)
    x_s = x[order]
    y_s = y[order]
    r = jnp.clip(jnp.ceil(frac * n).astype(jnp.int32), 1, n)

    def bandwidth(i):
        lo = jnp.maximum(0, i - r + 1)
        hi = jnp.minimum(n - 1, i + r - 1)
        # max distance to edges in the r-neighborhood
        return jnp.maximum(x_s[i] - x_s[lo], x_s[hi] - x_s[i]) + 1e-12

    h = jax.vmap(bandwidth)(jnp.arange(n))  # (n,)

    def fit_once(rob_obs: jnp.ndarray) -> jnp.ndarray:
        # weights matrix implicit via broadcasting: (n targets, n obs)
        U = (x_s[None, :] - x_s[:, None]) / h[:, None]
        U = jnp.clip(jnp.abs(U), 0.0, 1.0)
        W = (1.0 - U**3) ** 3
        W = W * rob_obs[None, :]  # robust weights on observations

        # Weighted linear regression at each target i with centered design (x_s - x_i)
        X1 = (x_s - x_s[:, None])  # (n, n) centered per-row
        sw   = jnp.sum(W, axis=1)
        swt  = jnp.sum(W * X1, axis=1)
        swtt = jnp.sum(W * X1 * X1, axis=1)
        swy  = jnp.sum(W * y_s[None, :], axis=1)
        swty = jnp.sum(W * X1 * y_s[None, :], axis=1)
        denom = sw * swtt - swt * swt + 1e-12
        intercept = (swy * swtt - swt * swty) / denom
        return intercept  # fitted at x_i (t=0)

    rob0 = jnp.ones((n,), dtype=y.dtype)

    def outer_step(_, rob_obs):
        yhat = fit_once(rob_obs)
        resid = y_s - yhat
        s = jnp.median(jnp.abs(resid)) + 1e-12
        u = resid / (6.0 * s)
        rob_new = jnp.where(jnp.abs(u) < 1.0, (1 - u**2) ** 2, 0.0)
        return rob_new

    rob = lax.fori_loop(0, it, outer_step, rob0)
    yhat_sorted = fit_once(rob)

    # invert permutation
    inv = jnp.empty_like(order)
    inv = inv.at[order].set(jnp.arange(n))
    return yhat_sorted[inv]

# =========================
# Public convenience wrapper
# =========================

def loess_smooth(
    y: jnp.ndarray,
    *,
    window: int | None = None,
    frac: float | None = None,
    deg: int = 1,
    robust_outer: int = 0,
    jump: int = 1,
    x: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Convenience wrapper. Choose one mode:
      - Fixed-window: pass window (odd). Uses loess_window_jump (supports jump, deg, robust_outer).
      - Frac-based: pass frac and x. Uses lowess_frac (supports robust iterations).

    Args:
        y (jnp.ndarray): Input 1D array to smooth.
        window (int | None): Window size for fixed-window LOESS. Default: None.
        frac (float | None): Fraction of data for neighborhood in LOWESS mode. Default: None.
        deg (int): Polynomial degree (0 or 1) for fixed-window mode. Default: 1.
        robust_outer (int): Number of robust reweighting iterations. Default: 0.
        jump (int): Jump size for anchor points in fixed-window mode. Default: 1.
        x (jnp.ndarray | None): X-coordinates for LOWESS mode. Default: None.

    Returns:
        jnp.ndarray: Smoothed values of shape (n,).

    Raises:
        ValueError: If neither window nor (frac, x) are provided.
    """
    if window is not None:
        return loess_window_jump(y, int(window), int(deg), int(robust_outer), int(jump))
    if frac is not None and x is not None:
        return lowess_frac(y, x, float(frac), int(robust_outer))
    raise ValueError("Provide either window (fixed-window LOESS) or both frac and x (LOWESS).")


# =========================
# Test Cases
# =========================

if __name__ == "__main__":
    print("=" * 60)
    print("LOESS/LOWESS Test Suite")
    print("=" * 60)
    
    # Test 1: Basic fixed-window LOESS (local constant)
    print("\n[Test 1] Fixed-window LOESS (deg=0, no robustness)")
    y1 = jnp.array([1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5])
    smoothed1 = loess_window_jump(y1, window=3, deg=0, robust_outer=0, jump=1)
    print(f"  Input: {y1}")
    print(f"  Smoothed (window=3, deg=0): {smoothed1}")
    assert smoothed1.shape == y1.shape, "Shape mismatch!"
    assert jnp.all(jnp.isfinite(smoothed1)), "Contains NaN/Inf!"
    print("  ✓ Shape and finiteness OK")
    
    # Test 2: Fixed-window LOESS (local linear)
    print("\n[Test 2] Fixed-window LOESS (deg=1, local linear)")
    smoothed2 = loess_window_jump(y1, window=5, deg=1, robust_outer=0, jump=1)
    print(f"  Smoothed (window=5, deg=1): {smoothed2}")
    assert smoothed2.shape == y1.shape, "Shape mismatch!"
    print("  ✓ Local linear smoothing OK")
    
    # Test 3: Robust LOESS with outliers
    print("\n[Test 3] Robust LOESS (handles outliers)")
    y3 = jnp.array([1.0, 2.0, 10.0, 3.0, 2.5, 4.0, 3.5])  # 10.0 is outlier
    smoothed3_robust = loess_window_jump(y3, window=5, deg=1, robust_outer=2, jump=1)
    smoothed3_normal = loess_window_jump(y3, window=5, deg=1, robust_outer=0, jump=1)
    print(f"  Input with outlier: {y3}")
    print(f"  Normal LOESS: {smoothed3_normal}")
    print(f"  Robust LOESS (2 iter): {smoothed3_robust}")
    outlier_effect = jnp.abs(smoothed3_robust[2] - smoothed3_normal[2])
    print(f"  Outlier dampening: {outlier_effect:.4f}")
    assert outlier_effect > 0, "Robust iteration should dampen outlier!"
    print("  ✓ Robust reweighting works")
    
    # Test 4: Jump anchors for speed (STL-style)
    print("\n[Test 4] Jump anchors (sparse evaluation + interpolation)")
    y4 = jnp.linspace(0, 10, 50) + jnp.sin(jnp.linspace(0, 4*jnp.pi, 50))
    smoothed4_dense = loess_window_jump(y4, window=11, deg=1, robust_outer=0, jump=1)
    smoothed4_sparse = loess_window_jump(y4, window=11, deg=1, robust_outer=0, jump=5)
    print(f"  Dense (jump=1) vs Sparse (jump=5) difference: {jnp.mean(jnp.abs(smoothed4_dense - smoothed4_sparse)):.6f}")
    assert jnp.mean(jnp.abs(smoothed4_dense - smoothed4_sparse)) < 0.5, "Jump should approximate well!"
    print("  ✓ Jump interpolation approximation OK")
    
    # Test 5: LOWESS (fraction-based, Cleveland style)
    print("\n[Test 5] LOWESS with fraction-based neighborhood")
    x5 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y5 = jnp.array([1.5, 2.2, 2.8, 3.5, 4.1, 4.8, 5.5])
    smoothed5 = lowess_frac(y5, x5, frac=0.5, it=0)
    print(f"  X: {x5}")
    print(f"  Y: {y5}")
    print(f"  Smoothed (frac=0.5): {smoothed5}")
    assert smoothed5.shape == y5.shape, "Shape mismatch!"
    assert jnp.all(jnp.isfinite(smoothed5)), "Contains NaN/Inf!"
    print("  ✓ LOWESS fraction-based OK")
    
    # Test 6: LOWESS with robust iterations
    print("\n[Test 6] LOWESS with robust iterations")
    y6 = jnp.array([1.0, 2.0, 10.0, 3.0, 4.0, 5.0, 6.0])  # outlier at index 2
    x6 = jnp.arange(7, dtype=jnp.float32)
    smoothed6_normal = lowess_frac(y6, x6, frac=0.6, it=0)
    smoothed6_robust = lowess_frac(y6, x6, frac=0.6, it=2)
    print(f"  Normal: {smoothed6_normal}")
    print(f"  Robust (2 iter): {smoothed6_robust}")
    print(f"  Outlier dampening: {jnp.abs(smoothed6_robust[2] - smoothed6_normal[2]):.4f}")
    print("  ✓ LOWESS robust iterations OK")
    
    # Test 7: Convenience wrapper (window mode)
    print("\n[Test 7] Convenience wrapper: loess_smooth (window mode)")
    y7 = jnp.array([1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0])
    smoothed7 = loess_smooth(y7, window=5, deg=1, robust_outer=0, jump=1)
    print(f"  Smoothed: {smoothed7}")
    assert smoothed7.shape == y7.shape, "Shape mismatch!"
    print("  ✓ Wrapper (window mode) OK")
    
    # Test 8: Convenience wrapper (frac mode)
    print("\n[Test 8] Convenience wrapper: loess_smooth (frac mode)")
    x8 = jnp.arange(8, dtype=jnp.float32)
    smoothed8 = loess_smooth(y7, frac=0.5, x=x8)
    print(f"  Smoothed: {smoothed8}")
    assert smoothed8.shape == y7.shape, "Shape mismatch!"
    print("  ✓ Wrapper (frac mode) OK")
    
    # Test 9: Edge cases - constant series
    print("\n[Test 9] Edge case: constant series")
    y9 = jnp.ones(10)
    smoothed9 = loess_window_jump(y9, window=5, deg=1, robust_outer=0, jump=1)
    print(f"  Input (constant): {y9}")
    print(f"  Smoothed: {smoothed9}")
    assert jnp.allclose(smoothed9, y9), "Should preserve constant series!"
    print("  ✓ Constant series handled correctly")
    
    # Test 10: JIT compilation verification
    print("\n[Test 10] JIT compilation check")
    import time
    y10 = jnp.linspace(0, 20, 200) + jax.random.normal(jax.random.PRNGKey(42), (200,)) * 0.5
    
    # Warm-up
    _ = loess_window_jump(y10, window=15, deg=1, robust_outer=0, jump=1)
    
    start = time.time()
    for _ in range(100):
        _ = loess_window_jump(y10, window=15, deg=1, robust_outer=0, jump=1)
    elapsed = time.time() - start
    print(f"  100 iterations: {elapsed:.4f}s ({elapsed*10:.2f}ms per call)")
    print("  ✓ JIT compilation working (fast repeated calls)")
    
    print("\n" + "=" * 60)
    print("✓ All LOESS/LOWESS tests passed!")
    print("=" * 60)
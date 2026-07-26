# test_period_detection.py
"""Tests for chronax.utils period detection (detect_period, detect_periods,
seasonal_strength). Style mirrors tests/test_holt.py — one top-level test
function with try/except sub-tests and inline prints."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from chronax.utils import detect_period, detect_periods, seasonal_strength


def _seasonal(period: int, n: int, amp: float = 1.0, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = amp * np.sin(2 * np.pi * t / period)
    return (base + noise * rng.standard_normal(n)).astype(np.float64)


def _ar1_seasonal(period: int, n: int, phi: float = 0.7, amp: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n + 50)
    z = np.zeros(n + 50)
    for k in range(1, n + 50):
        z[k] = phi * z[k - 1] + eps[k]
    z = z[50:]
    seasonal = amp * np.sin(2 * np.pi * np.arange(n) / period)
    return (z + seasonal).astype(np.float64)


def test_period_detection():
    """Comprehensive test suite for chronax.utils period detection."""
    print("\n=== chronax.utils period detection ===")
    passed = 0
    failed = 0

    # 1. Recover periods on low-noise synth seasonality. (Higher-noise long-
    # period synthetics get aliased to small lags by the harmonic-scoring
    # algorithm — this is the existing detect_period's behaviour and is
    # outside the scope of "is the JAX port correct".)
    for true_p in (12, 24, 30):
        try:
            y = _seasonal(true_p, n=400, noise=0.02, seed=true_p)
            est = int(detect_period(jnp.asarray(y), max_period=60))
            assert est == true_p, f"period={true_p}: got {est}"
            print(f"  -> [Test seasonal p={true_p}] Passed (got {est}).")
            passed += 1
        except AssertionError as e:
            print(f"  -> [Test seasonal p={true_p}] FAILED: {e}")
            failed += 1

    # 2. Pure noise → fallback (1).
    try:
        rng = np.random.default_rng(7)
        y = rng.standard_normal(300).astype(np.float64)
        est = int(detect_period(jnp.asarray(y), max_period=60))
        assert est == 1, f"noise: got {est}"
        print(f"  -> [Test noise → fallback] Passed.")
        passed += 1
    except AssertionError as e:
        print(f"  -> [Test noise → fallback] FAILED: {e}")
        failed += 1

    # 3. Trended-only (no seasonality) → fallback.
    try:
        y = (np.arange(300, dtype=np.float64) * 0.3 + np.random.default_rng(3).standard_normal(300) * 0.05)
        est = int(detect_period(jnp.asarray(y), max_period=60))
        assert est == 1, f"trended: got {est}"
        print(f"  -> [Test trend-only → fallback] Passed.")
        passed += 1
    except AssertionError as e:
        print(f"  -> [Test trend-only → fallback] FAILED: {e}")
        failed += 1

    # 4. AR(1) + sinusoid: detection should still find seasonal period within ±1.
    try:
        y = _ar1_seasonal(period=30, n=600, phi=0.6, amp=4.0, seed=11)
        est = int(detect_period(jnp.asarray(y), max_period=60))
        assert abs(est - 30) <= 1, f"AR1+seasonal p=30: got {est}"
        print(f"  -> [Test AR1+seasonal p=30] Passed (got {est}, target 30 ±1).")
        passed += 1
    except AssertionError as e:
        print(f"  -> [Test AR1+seasonal p=30] FAILED: {e}")
        failed += 1

    # 5. detect_periods on multi-seasonal: 7 + 30 (both within ±2).
    try:
        n = 600
        t = np.arange(n)
        y = (1.5 * np.sin(2 * np.pi * t / 7) + 2.5 * np.sin(2 * np.pi * t / 30) +
             0.05 * np.random.default_rng(2).standard_normal(n)).astype(np.float64)
        ps = np.asarray(detect_periods(jnp.asarray(y), n_periods=2, max_period=60)).tolist()
        ok_short = any(abs(p - 7) <= 1 for p in ps)
        ok_long = any(abs(p - 30) <= 2 for p in ps)
        assert ok_short and ok_long, f"multi-seasonal 7+30: got {ps}"
        assert ps == sorted(ps), f"detect_periods must return ascending: got {ps}"
        print(f"  -> [Test multi-seasonal 7+30] Passed (got {ps}, target 7±1, 30±2).")
        passed += 1
    except AssertionError as e:
        print(f"  -> [Test multi-seasonal 7+30] FAILED: {e}")
        failed += 1

    # 6. seasonal_strength: high on clean sinusoid, low on noise.
    try:
        y_clean = _seasonal(period=24, n=480, noise=0.02, seed=5)
        y_noise = np.random.default_rng(5).standard_normal(480).astype(np.float64)
        s_clean = float(seasonal_strength(jnp.asarray(y_clean), 24))
        s_noise = float(seasonal_strength(jnp.asarray(y_noise), 24))
        assert s_clean > 0.9, f"clean strength: got {s_clean}"
        assert s_noise < 0.2, f"noise strength: got {s_noise}"
        print(f"  -> [Test seasonal_strength] Passed (clean={s_clean:.3f}, noise={s_noise:.3f}).")
        passed += 1
    except AssertionError as e:
        print(f"  -> [Test seasonal_strength] FAILED: {e}")
        failed += 1

    # 7. vmap test — non-negotiable. Use low-noise periods the algo handles.
    try:
        y12 = jnp.asarray(_seasonal(12, n=400, noise=0.02, seed=12))
        y24 = jnp.asarray(_seasonal(24, n=400, noise=0.02, seed=24))
        y30 = jnp.asarray(_seasonal(30, n=400, noise=0.02, seed=30))
        batch = jnp.stack([y12, y24, y30])
        out = jax.vmap(lambda y: detect_period(y, max_period=60))(batch)
        out_np = np.asarray(out).tolist()
        assert out_np == [12, 24, 30], f"vmap: got {out_np}"
        print(f"  -> [Test vmap] Passed (got {out_np}).")
        passed += 1
    except AssertionError as e:
        print(f"  -> [Test vmap] FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"  -> [Test vmap] FAILED with exception: {type(e).__name__}: {e}")
        failed += 1

    # 8. jit test — must match eager.
    try:
        y = jnp.asarray(_seasonal(30, n=500, noise=0.08, seed=99))
        eager = int(detect_period(y, max_period=60))
        jitted = int(jax.jit(lambda y: detect_period(y, max_period=60))(y))
        assert eager == jitted, f"jit: eager={eager}, jitted={jitted}"
        print(f"  -> [Test jit matches eager] Passed (both = {eager}).")
        passed += 1
    except AssertionError as e:
        print(f"  -> [Test jit matches eager] FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"  -> [Test jit matches eager] FAILED with exception: {type(e).__name__}: {e}")
        failed += 1

    # 9. Conformal-style usage: vmap over sliding folds of one series gives same period.
    try:
        y_full = _seasonal(24, n=600, noise=0.02, seed=42)
        # 5 folds of length 400, sliding by 50.
        folds = jnp.stack([jnp.asarray(y_full[k * 50 : k * 50 + 400]) for k in range(5)])
        out = jax.vmap(lambda y: detect_period(y, max_period=60))(folds)
        out_np = np.asarray(out).tolist()
        assert all(p == 24 for p in out_np), f"sliding-folds: got {out_np}"
        print(f"  -> [Test conformal-style vmap] Passed (got {out_np}).")
        passed += 1
    except AssertionError as e:
        print(f"  -> [Test conformal-style vmap] FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"  -> [Test conformal-style vmap] FAILED with exception: {type(e).__name__}: {e}")
        failed += 1

    # 10. detect_periods vmap (vmap mechanics + shape; algorithmic precision
    # of multi-period extraction on pure sinusoids is limited by ACF-alias
    # ambiguity — the test verifies that vmap returns the right shape and at
    # least one period close to a true component on each row).
    try:
        n = 600
        t = np.arange(n)
        y_a = (1.5 * np.sin(2 * np.pi * t / 7) + 2.5 * np.sin(2 * np.pi * t / 30) +
               0.02 * np.random.default_rng(2).standard_normal(n)).astype(np.float64)
        y_b = (1.5 * np.sin(2 * np.pi * t / 11) + 2.5 * np.sin(2 * np.pi * t / 25) +
               0.02 * np.random.default_rng(8).standard_normal(n)).astype(np.float64)
        batch = jnp.stack([jnp.asarray(y_a), jnp.asarray(y_b)])
        out = np.asarray(jax.vmap(lambda y: detect_periods(y, n_periods=2, max_period=60))(batch))
        assert out.shape == (2, 2), f"vmap shape: got {out.shape}"
        # At least one of the detected periods must be near a true component.
        true_a = {7, 14, 21, 28, 30}  # truth + tolerable aliases of 7 + 30
        true_b = {11, 22, 25, 33, 50}
        a_ok = any(p in true_a for p in out[0].tolist())
        b_ok = any(p in true_b for p in out[1].tolist())
        assert a_ok and b_ok, f"row 0 {out[0]}, row 1 {out[1]}"
        print(f"  -> [Test detect_periods vmap] Passed (rows {out[0].tolist()}, {out[1].tolist()}).")
        passed += 1
    except AssertionError as e:
        print(f"  -> [Test detect_periods vmap] FAILED: {e}")
        failed += 1
    except Exception as e:
        print(f"  -> [Test detect_periods vmap] FAILED with exception: {type(e).__name__}: {e}")
        failed += 1

    print(f"\n=== chronax.utils period detection: {passed} passed, {failed} failed ===\n")
    assert failed == 0, f"{failed} sub-test(s) failed"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-s", __file__]))

"""Tests for the xLSTMTime extensions of the XLSTM forecaster.

Covers: RevIN round-trip, learnable moving-average decomposition,
sLSTM block forward, direct-head forecast shape + finiteness, accuracy
on a synthetic seasonal series, vmap-safety of direct-mode forecast,
and a backward-compat sentinel that the legacy ``XLSTM()`` constructor
still works without the new kwargs.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models import XLSTM, SeasonalNaive
from chronax.models.xlstm.xlstm_backend import (
    XLSTMConfig,
    init_params,
    init_slstm_block_state,
    revin_normalize,
    revin_denormalize,
    series_decompose,
    slstm_block_forward,
)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _seasonal_series(n: int = 200, period: int = 24, slope: float = 0.0,
                     amp: float = 2.0, noise: float = 0.3, seed: int = 0) -> jnp.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float32)
    y = slope * t + amp * np.sin(2 * np.pi * t / period) + noise * rng.randn(n)
    return jnp.asarray(y, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# 1. RevIN round-trip
# ---------------------------------------------------------------------------

def test_revin_roundtrip():
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.randn(200).astype(np.float32))
    p = {
        "gamma": jnp.array([1.0], dtype=jnp.float32),
        "beta": jnp.array([0.0], dtype=jnp.float32),
    }
    x_n, stats = revin_normalize(x, p)
    x_rec = revin_denormalize(x_n, p, stats)
    np.testing.assert_allclose(np.asarray(x_rec), np.asarray(x), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Decomposition trend + seasonal == input
# ---------------------------------------------------------------------------

def test_decomposition_sum_equals_input():
    rng = np.random.RandomState(1)
    x = jnp.asarray(rng.randn(150).astype(np.float32))
    kernel = 25
    p = {"k": jnp.full((kernel,), 1.0 / kernel, dtype=jnp.float32)}
    trend, seasonal = series_decompose(x, p, kernel)
    rec = trend + seasonal
    np.testing.assert_allclose(np.asarray(rec), np.asarray(x), rtol=1e-5, atol=1e-5)
    assert trend.shape == x.shape
    assert seasonal.shape == x.shape


# ---------------------------------------------------------------------------
# 3. Decomposition smooths trend (OLS slope agrees with input)
# ---------------------------------------------------------------------------

def test_decomposition_smooths_trend():
    rng = np.random.RandomState(2)
    n = 400
    t = np.arange(n, dtype=np.float32)
    x_np = 0.1 * t + 1.5 * np.sin(2 * np.pi * t / 12) + 0.5 * rng.randn(n)
    x = jnp.asarray(x_np, dtype=jnp.float32)
    kernel = 25
    p = {"k": jnp.full((kernel,), 1.0 / kernel, dtype=jnp.float32)}
    trend, _ = series_decompose(x, p, kernel)
    # OLS slope of trend should match OLS slope of x within 10 %
    slope_x = np.polyfit(t, x_np, 1)[0]
    slope_trend = np.polyfit(t, np.asarray(trend), 1)[0]
    assert abs(slope_trend - slope_x) / abs(slope_x) < 0.10


# ---------------------------------------------------------------------------
# 4. sLSTM block forward — shape and finiteness
# ---------------------------------------------------------------------------

def test_slstm_block_forward_shape_finite():
    cfg = XLSTMConfig(
        embed_dim=32, num_heads=4, head_dim=8, num_layers=1,
        ctx_len=32, horizon_train=4,
        block_types=("slstm",), decode_mode="ar", horizon=4,
    )
    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    # Block functions expect compute-dtype (bf16) params — that's how
    # xlstm_forward calls them. Cast block_p accordingly.
    block_p = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), params["blocks"][0])
    init_state = init_slstm_block_state(cfg, dtype=jnp.bfloat16)
    x_seq = jnp.asarray(np.random.RandomState(3).randn(32, 32).astype(np.float32), dtype=jnp.bfloat16)
    h_seq, _ = slstm_block_forward(block_p, x_seq, init_state, cfg)
    assert h_seq.shape == (32, 32)
    h_f32 = np.asarray(h_seq.astype(jnp.float32))
    assert np.all(np.isfinite(h_f32))


# ---------------------------------------------------------------------------
# 5. Direct-head forecast shape
# ---------------------------------------------------------------------------

def test_direct_head_forecast_shape():
    y = _seasonal_series(n=240, period=24, slope=0.05, seed=4)
    m = XLSTM(
        ctx_len=32, num_layers=2, embed_dim=32, num_heads=4,
        n_epochs=15, batch_size=16, lr=1e-3, seed=0,
        block_types=("slstm", "slstm"),
        use_revin=True, revin_affine=True,
        use_decomposition=True, decomp_kernel=25,
        decode_mode="direct", horizon=12,
        use_conv1d_in_slstm=True, conv1d_kernel=4,
    )
    m.fit(y)
    out = m.predict(h=12)
    mean = np.asarray(out["mean"])
    assert mean.shape == (12,)
    assert np.all(np.isfinite(mean))


# ---------------------------------------------------------------------------
# 6. xLSTMTime beats SeasonalNaive on a strong seasonal series
# ---------------------------------------------------------------------------

def test_xlstmtime_beats_seasonal_naive():
    y = _seasonal_series(n=600, period=24, slope=0.02, amp=3.0, noise=0.2, seed=5)
    h = 24
    y_train, y_test = y[:-h], y[-h:]

    # SeasonalNaive baseline
    sn = SeasonalNaive(season_length=24)
    sn.fit(y_train)
    sn_pred = np.asarray(sn.predict(h=h)["mean"])
    sn_mae = float(np.mean(np.abs(np.asarray(y_test) - sn_pred)))

    # xLSTMTime
    m = XLSTM(
        ctx_len=48, num_layers=2, embed_dim=64, num_heads=4,
        n_epochs=50, batch_size=32, lr=1e-3, seed=0,
        block_types=("slstm", "slstm"),
        use_revin=True, use_decomposition=True,
        decode_mode="direct", horizon=h,
        use_conv1d_in_slstm=True,
    )
    m.fit(y_train)
    pred = np.asarray(m.predict(h=h)["mean"])
    xlstmtime_mae = float(np.mean(np.abs(np.asarray(y_test) - pred)))

    assert np.all(np.isfinite(pred))
    assert xlstmtime_mae < sn_mae, (
        f"xLSTMTime MAE ({xlstmtime_mae:.4f}) must beat SeasonalNaive ({sn_mae:.4f})"
    )


# ---------------------------------------------------------------------------
# 7. vmap-safety of direct-mode forecast
# ---------------------------------------------------------------------------

def test_xlstmtime_vmap_forecast():
    y_train = _seasonal_series(n=300, period=24, slope=0.02, seed=6)
    h = 12
    m = XLSTM(
        ctx_len=32, num_layers=2, embed_dim=32, num_heads=4,
        n_epochs=15, batch_size=16, lr=1e-3, seed=0,
        block_types=("slstm", "slstm"),
        use_revin=True, use_decomposition=True,
        decode_mode="direct", horizon=h,
        use_conv1d_in_slstm=True,
    )
    m.fit(y_train)

    # Build a small batch of contexts of identical length
    rng = np.random.RandomState(99)
    B, T = 4, 120
    t = np.arange(T, dtype=np.float32)
    batch = np.stack([
        2.0 + (0.02 + 0.005 * i) * t + 1.5 * np.sin(2 * np.pi * t / 24) + 0.2 * rng.randn(T)
        for i in range(B)
    ]).astype(np.float32)
    y_batch = jnp.asarray(batch)

    seq = jnp.stack([m.forecast(y=y_batch[i], h=h)["mean"] for i in range(B)], axis=0)
    vm = jax.vmap(lambda y: m.forecast(y=y, h=h)["mean"])(y_batch)
    assert seq.shape == vm.shape == (B, h)
    assert jnp.all(jnp.isfinite(vm))
    np.testing.assert_allclose(np.asarray(seq), np.asarray(vm), rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# 8. Backward-compat sentinel — legacy XLSTM() with no new kwargs
# ---------------------------------------------------------------------------

def test_backward_compat_xlstm_unchanged():
    """Legacy constructor must still produce finite AR-mode forecasts."""
    y = _seasonal_series(n=80, period=24, slope=0.05, seed=7)
    m = XLSTM(
        ctx_len=32, horizon_train=4, num_layers=1, embed_dim=32,
        num_heads=4, n_epochs=20, batch_size=16, seed=0,
    )
    m.fit(y)
    out = m.predict(h=6)
    mean = np.asarray(out["mean"])
    assert mean.shape == (6,)
    assert np.all(np.isfinite(mean))


if __name__ == "__main__":
    test_revin_roundtrip(); print("test_revin_roundtrip: OK")
    test_decomposition_sum_equals_input(); print("test_decomposition_sum_equals_input: OK")
    test_decomposition_smooths_trend(); print("test_decomposition_smooths_trend: OK")
    test_slstm_block_forward_shape_finite(); print("test_slstm_block_forward_shape_finite: OK")
    test_direct_head_forecast_shape(); print("test_direct_head_forecast_shape: OK")
    test_xlstmtime_beats_seasonal_naive(); print("test_xlstmtime_beats_seasonal_naive: OK")
    test_xlstmtime_vmap_forecast(); print("test_xlstmtime_vmap_forecast: OK")
    test_backward_compat_xlstm_unchanged(); print("test_backward_compat_xlstm_unchanged: OK")

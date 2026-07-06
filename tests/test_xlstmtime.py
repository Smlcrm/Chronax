"""Tests for the XLSTM forecaster and its xLSTM/xLSTMTime backend.

Two groups:

A. **xLSTMTime extensions** (via the full :class:`XLSTM` forecaster): RevIN
   round-trip, learnable moving-average decomposition, sLSTM block forward,
   direct-head forecast shape, accuracy on a synthetic seasonal series,
   vmap-safety of direct-mode forecast, and a backward-compat sentinel that the
   legacy ``XLSTM()`` constructor still works without the new kwargs.

B. **FlashRNN-aligned backend optimization** (arXiv 2412.07752): the depthwise
   causal conv, the input-hoisting refactor (vectorized ``*_block_forward`` must
   equal the per-step ``*_block_step`` AR-decode path), and the paper-faithful
   sLSTM cell semantics (forget gate exp|sigmoid x stabilizer per_head|per_cell)
   validated against an independent NumPy reference of Eq. 12-15, plus vmap-safety.
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
    block_init_state,
    revin_normalize,
    revin_denormalize,
    series_decompose,
    slstm_block_forward,
    slstm_block_step,
    mlstm_block_forward,
    mlstm_block_step,
    slstm_recurrence_step,
    _causal_conv1d,
)


# ===========================================================================
# A. xLSTMTime extensions (full XLSTM forecaster)
# ===========================================================================

def _seasonal_series(n: int = 200, period: int = 24, slope: float = 0.0,
                     amp: float = 2.0, noise: float = 0.3, seed: int = 0) -> jnp.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float32)
    y = slope * t + amp * np.sin(2 * np.pi * t / period) + noise * rng.randn(n)
    return jnp.asarray(y, dtype=jnp.float32)


# --- 1. RevIN round-trip ---------------------------------------------------

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


# --- 2. Decomposition trend + seasonal == input ----------------------------

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


# --- 3. Decomposition smooths trend (OLS slope agrees with input) ----------

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


# --- 4. sLSTM block forward — shape and finiteness -------------------------

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


# --- 5. Direct-head forecast shape -----------------------------------------

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


# --- 6. xLSTMTime beats SeasonalNaive on a strong seasonal series ----------

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


# --- 7. vmap-safety of direct-mode forecast --------------------------------

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


# --- 8. Backward-compat sentinel — legacy XLSTM() with no new kwargs --------

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


# ===========================================================================
# B. FlashRNN-aligned backend optimization (arXiv 2412.07752)
# ===========================================================================

# --- B0. Depthwise causal Conv1D -------------------------------------------

def _ref_causal_depthwise_conv(x, W, b, kernel):
    """Independent reference: out[t,d] = sum_k xp[t+k,d] * W[d,k] + b[d],
    where xp is x left-padded with (kernel-1) zeros (causal)."""
    T, D = x.shape
    xp = np.concatenate([np.zeros((kernel - 1, D), x.dtype), x], axis=0)
    out = np.zeros((T, D), np.float64)
    for t in range(T):
        for k in range(kernel):
            out[t] += xp[t + k] * W[:, k]
    return out + b


def test_causal_conv1d_matches_reference():
    T, D, K = 20, 8, 4
    rng = np.random.RandomState(0)
    x = rng.randn(T, D).astype(np.float32)
    W = rng.randn(D, K).astype(np.float32)
    b = rng.randn(D).astype(np.float32)
    got = np.asarray(_causal_conv1d(jnp.asarray(x), jnp.asarray(W), jnp.asarray(b), K))
    ref = _ref_causal_depthwise_conv(x, W, b, K)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_causal_conv1d_is_strictly_causal():
    T, D, K = 16, 4, 4
    rng = np.random.RandomState(1)
    x = rng.randn(T, D).astype(np.float32)
    W = rng.randn(D, K).astype(np.float32)
    b = np.zeros(D, np.float32)
    base = np.asarray(_causal_conv1d(jnp.asarray(x), jnp.asarray(W), jnp.asarray(b), K))
    x2 = x.copy()
    x2[-1] += 100.0  # perturb ONLY the last timestep
    pert = np.asarray(_causal_conv1d(jnp.asarray(x2), jnp.asarray(W), jnp.asarray(b), K))
    # earlier outputs must be untouched (no future leakage); last one must change
    np.testing.assert_allclose(base[:-1], pert[:-1], rtol=1e-5, atol=1e-5)
    assert not np.allclose(base[-1], pert[-1])


# --- B1. Input-hoisting equivalence: sequence path == per-step path --------
# The vectorized *_block_forward must match *_block_step applied stepwise
# (the AR-decode path). No conv here so the loop sees the same input.

def _make_block(block_type, seed=0, T=24):
    cfg = XLSTMConfig(
        embed_dim=32, num_heads=4, head_dim=8, num_layers=1,
        ctx_len=T, horizon_train=4, block_types=(block_type,),
        decode_mode="ar", horizon=4,
    )
    p = init_params(jax.random.PRNGKey(seed), cfg)
    bp = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), p["blocks"][0])
    st = block_init_state(cfg, 0, dtype=jnp.bfloat16)
    x = jnp.asarray(
        jax.random.normal(jax.random.PRNGKey(seed + 1), (T, cfg.embed_dim)), jnp.bfloat16
    )
    return cfg, bp, st, x


def _loop_block(step_fn, bp, x, st, cfg):
    state = st
    outs = []
    for t in range(x.shape[0]):
        h_t, state = step_fn(bp, x[t], state, cfg)
        outs.append(h_t)
    return jnp.stack(outs, 0)


@pytest.mark.parametrize("block_type", ["slstm", "mlstm"])
def test_block_forward_matches_stepwise(block_type):
    cfg, bp, st, x = _make_block(block_type)
    fwd = slstm_block_forward if block_type == "slstm" else mlstm_block_forward
    step = slstm_block_step if block_type == "slstm" else mlstm_block_step
    seq = np.asarray(fwd(bp, x, st, cfg)[0].astype(jnp.float32))
    ref = np.asarray(_loop_block(step, bp, x, st, cfg).astype(jnp.float32))
    np.testing.assert_allclose(seq, ref, rtol=2e-2, atol=2e-2)


# --- B2/B3. sLSTM cell semantics vs the paper reference (Eq. 12-15) ---------
# forget gate (exp|sigmoid) x stabilizer (per_head|per_cell), checked in
# float32 against an independent NumPy recurrence.

def _ref_slstm_recurrence(R, Wx_seq, H, Dh, forget, stab, gate_clip=8.0):
    """Independent NumPy reference for the sLSTM recurrence core (float64)."""
    R = np.asarray(R, np.float64)
    Wx_seq = np.asarray(Wx_seq, np.float64)
    T = Wx_seq.shape[0]
    h = np.zeros((H, Dh)); c = np.zeros((H, Dh)); n = np.zeros((H, Dh))
    m = np.full((H, Dh), -1e9) if stab == "per_cell" else np.full((H,), -1e9)
    outs = []
    for t in range(T):
        Rh = np.einsum("ghkj,hj->ghk", R, h)
        pre = Wx_seq[t] + Rh
        i_pre = np.clip(pre[0], -gate_clip, gate_clip)
        f_pre = np.clip(pre[1], -gate_clip, gate_clip)
        o_pre = pre[2]
        z = np.tanh(pre[3])
        logf = -np.logaddexp(0.0, -f_pre) if forget == "sigmoid" else f_pre
        m_prev = m if stab == "per_cell" else m[:, None]
        m_new = np.maximum(logf + m_prev, i_pre)              # (H, Dh)
        i_stab = np.exp(i_pre - m_new)
        f_stab = np.exp(logf + m_prev - m_new)
        c = f_stab * c + i_stab * z
        n = f_stab * n + i_stab
        h = (1.0 / (1.0 + np.exp(-o_pre))) * (c / np.maximum(np.abs(n), 1.0))
        outs.append(h.copy())
        m = m_new if stab == "per_cell" else np.max(m_new, axis=-1)
    return np.stack(outs, 0)


def _run_slstm_core(cfg, R, Wx_seq):
    st = init_slstm_block_state(cfg, dtype=jnp.float32)  # float32 -> crisp comparison

    def step(state, Wx_t):
        h, ns = slstm_recurrence_step(R, Wx_t, state, cfg)
        return ns, h

    _, h_seq = jax.lax.scan(step, st, Wx_seq)
    return np.asarray(h_seq)


@pytest.mark.parametrize(
    "forget,stab",
    [("exp", "per_head"), ("sigmoid", "per_cell"), ("sigmoid", "per_head"), ("exp", "per_cell")],
)
def test_slstm_recurrence_matches_reference(forget, stab):
    H, Dh, T = 4, 8, 20
    cfg = XLSTMConfig(
        embed_dim=H * Dh, num_heads=H, head_dim=Dh, num_layers=1,
        ctx_len=T, horizon_train=4, block_types=("slstm",), decode_mode="ar", horizon=4,
        slstm_forget_gate=forget, slstm_stabilizer=stab,
    )
    R = jax.random.normal(jax.random.PRNGKey(1), (4, H, Dh, Dh), jnp.float32) * 0.1
    Wx = jax.random.normal(jax.random.PRNGKey(2), (T, 4, H, Dh), jnp.float32) * 0.5
    got = _run_slstm_core(cfg, R, Wx)
    ref = _ref_slstm_recurrence(R, Wx, H, Dh, forget, stab)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("stab,expected", [("per_head", (4,)), ("per_cell", (4, 8))])
def test_slstm_state_m_shape(stab, expected):
    cfg = XLSTMConfig(
        embed_dim=32, num_heads=4, head_dim=8, num_layers=1,
        ctx_len=16, horizon_train=4, block_types=("slstm",), decode_mode="ar", horizon=4,
        slstm_stabilizer=stab,
    )
    st = init_slstm_block_state(cfg, dtype=jnp.float32)
    assert st.m.shape == expected


def test_slstm_paper_faithful_is_vmappable():
    """The paper-faithful path (per-cell m) must stay vmap-traceable — the conformal
    walk-forward vmaps forecast over windows."""
    cfg = XLSTMConfig(
        embed_dim=32, num_heads=4, head_dim=8, num_layers=1,
        ctx_len=24, horizon_train=4, block_types=("slstm",), decode_mode="ar", horizon=4,
        slstm_forget_gate="sigmoid", slstm_stabilizer="per_cell",
    )
    p = init_params(jax.random.PRNGKey(0), cfg)
    bp = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), p["blocks"][0])
    st = init_slstm_block_state(cfg, dtype=jnp.bfloat16)
    B, T, D = 4, 24, 32
    batch = jnp.asarray(jax.random.normal(jax.random.PRNGKey(1), (B, T, D)), jnp.bfloat16)
    f = lambda x: slstm_block_forward(bp, x, st, cfg)[0]
    vm = jax.vmap(f)(batch)
    seq = jnp.stack([f(batch[i]) for i in range(B)], axis=0)
    assert vm.shape == (B, T, D)
    assert bool(jnp.all(jnp.isfinite(vm.astype(jnp.float32))))
    np.testing.assert_allclose(
        np.asarray(vm.astype(jnp.float32)), np.asarray(seq.astype(jnp.float32)),
        rtol=2e-2, atol=2e-2,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

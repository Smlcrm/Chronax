"""Tests for the XLSTM forecaster and its xLSTM/xLSTMTime backend.

Two groups:

A. **xLSTMTime extensions** (via the full :class:`XLSTM` forecaster): RevIN
   round-trip, learnable moving-average decomposition, sLSTM block forward,
   direct-head forecast shape, accuracy on a synthetic seasonal series,
   vmap-safety of direct-mode forecast, and a backward-compat sentinel that the
   legacy ``XLSTM()`` constructor still works without the new kwargs.

B. **FlashRNN-aligned backend optimization** (arXiv 2412.07752): the depthwise
   causal conv, the input-hoisting refactor (vectorized ``*_block_forward`` must
   equal the per-step ``*_block_step`` AR-decode path), and the sLSTM cell
   semantics (forget gate exp|sigmoid x stabilizer per_head|per_cell) validated
   against a NumPy transcription of the IMPLEMENTATION's semantics — Eq. 12-15
   PLUS the chronax floor/clip deviations. This pins self-consistency, NOT
   canon fidelity. Plus vmap-safety.
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


# ===========================================================================
# NF-convention constructor aliases (neural benchmark harness protocol)
# ===========================================================================


def test_nf_alias_params_map_onto_native_names():
    m = XLSTM(h=24, input_size=72, random_seed=7)
    assert m.ctx_len == 72
    assert m.seed == 7
    assert m.horizon == 24
    # direct mode satisfied by `h` alone (no separate `horizon` needed)
    md = XLSTM(h=24, decode_mode="direct")
    assert md.horizon == 24


def test_nf_alias_h_conflicting_with_horizon_raises():
    with pytest.raises(ValueError, match="horizon"):
        XLSTM(h=24, horizon=12)
    m = XLSTM(h=24, horizon=24)  # equal values are not a conflict
    assert m.horizon == 24


def test_nf_alias_ar_predictions_unchanged():
    """In AR mode `h` only records the configured horizon: cfg.horizon feeds
    direct-mode branches exclusively, so fitted params and predictions must be
    bit-identical with or without the alias."""
    y = jnp.sin(jnp.arange(48, dtype=jnp.float32) / 4.0)
    base = XLSTM(ctx_len=16, n_epochs=2, seed=0).fit(y)
    alias = XLSTM(ctx_len=16, n_epochs=2, seed=0, h=6).fit(y)
    np.testing.assert_array_equal(
        np.asarray(base.predict(h=6)["mean"]),
        np.asarray(alias.predict(h=6)["mean"]),
    )


def test_nf_alias_max_steps_sets_total_optimizer_steps():
    """`max_steps` must be the exact total optimizer step count (NF semantics),
    overriding the epochs-derived budget — observable as len(losses)."""
    y = jnp.sin(jnp.arange(60, dtype=jnp.float32) / 4.0)
    m = XLSTM(input_size=16, n_epochs=100, max_steps=7).fit(y)
    assert m.model_["losses"].shape[0] == 7


def test_nf_alias_learning_rate_maps_to_lr():
    assert XLSTM(learning_rate=0.01).lr == 0.01


def test_ar_decode_with_revin_returns_original_scale():
    """AR decode with RevIN must denormalize: training wraps the out_proj head in
    revin_denormalize (xlstm_forward), so raw rollout emissions are in normalized
    space — decode() must map them back and roll the recurrence in normalized
    space. Regression test against an AR/RevIN scale mismatch."""
    y = 500.0 + jnp.sin(jnp.arange(64, dtype=jnp.float32) / 3.0) * 5.0
    m = XLSTM(ctx_len=16, n_epochs=2, seed=0, use_revin=True)
    m.fit(y)
    pred = m.predict(h=4)["mean"]
    assert bool(jnp.all(jnp.abs(pred - 500.0) < 250.0)), f"pred not in original scale: {pred}"


def test_harness_protocol_construct_fit_predict():
    """The neural benchmark worker constructs cls(h=..., input_size=...,
    random_seed=..., **params) then fit(y) / predict(h)["mean"]
    in the neural benchmark worker."""
    y = jnp.asarray(np.random.default_rng(0).normal(size=60), jnp.float32)
    m = XLSTM(h=4, input_size=16, random_seed=1, n_epochs=1)
    m.fit(y)
    mean = m.predict(h=4)["mean"]
    assert mean.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(mean)))


def test_xlstm_pickle_roundtrip():
    """Fitted-estimator pickle round-trip (long-standing deferred gap): the
    params-only __getstate__/__setstate__ path must survive serialization with
    bit-identical predictions from the restored estimator."""
    import pickle

    y = jnp.sin(jnp.arange(48, dtype=jnp.float32) / 4.0)
    m = XLSTM(ctx_len=16, n_epochs=2, seed=0).fit(y)
    p1 = np.asarray(m.predict(h=6)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    p2 = np.asarray(m2.predict(h=6)["mean"])
    np.testing.assert_array_equal(p1, p2)


# ===========================================================================
# Fitted fast path, conformity tails, compute dtype
# ===========================================================================

from chronax.utils import ConformalIntervals
import chronax.models.xlstm.xlstm_backend as _backend

_REVIEW_H = 24
_REVIEW_NW = 4


def _trend_seasonal_series(n=600, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float32)
    y = 50.0 + 0.15 * t + 4.0 * np.sin(2 * np.pi * t / 24) + 0.5 * rng.randn(n)
    return jnp.asarray(y, jnp.float32)


@pytest.fixture(scope="module")
def fitted_conformal_model():
    """One shared fitted paper-profile model with conformal params (read-only
    in every test below — never mutate it)."""
    m = XLSTM(
        ctx_len=64, num_layers=2, embed_dim=32, num_heads=4,
        block_types=("slstm", "slstm"), use_revin=True,
        use_decomposition=True, decode_mode="direct", horizon=_REVIEW_H,
        max_steps=50, seed=0,
        conformal_params=ConformalIntervals(n_windows=_REVIEW_NW, h=_REVIEW_H,
                                            method="conformal_distribution"),
    )
    m.fit(_trend_seasonal_series())
    return m


def test_fitted_forecast_level_emits_intervals(fitted_conformal_model):
    """The fitted fast path must emit intervals, not silently drop `level`
    (the stateless-path gate does not cover it)."""
    m = fitted_conformal_model
    out = m.forecast(y=_trend_seasonal_series(), h=_REVIEW_H, level=[90])
    assert "lo-90" in out and "hi-90" in out and "mean" in out
    assert bool(jnp.all(out["lo-90"] <= out["hi-90"]))


def test_forecast_fitted_true_raises(fitted_conformal_model):
    """X3: `fitted=True` was silently ignored on both paths; it must raise."""
    with pytest.raises(NotImplementedError, match="fitted"):
        fitted_conformal_model.forecast(
            y=_trend_seasonal_series(), h=_REVIEW_H, fitted=True
        )


def test_conformity_scores_use_true_context_tails(fitted_conformal_model):
    """Scores must equal what forecast() produces from the TRUE series prefix per
    window, not from the base class's edge-masked (plateau) array: under the base
    implementation window 0's context ends in (nw-1)*h constant values, which
    inflates its scores several-fold."""
    m = fitted_conformal_model
    y = _trend_seasonal_series()
    cs = np.asarray(m.model_["_cs"])
    n = int(y.shape[0])
    base_train_end = n - _REVIEW_NW * _REVIEW_H
    for i in range(_REVIEW_NW):
        train_end = base_train_end + i * _REVIEW_H
        expected_fc = m.forecast(y=y[:train_end], h=_REVIEW_H)["mean"]
        expected = np.asarray(y[train_end:train_end + _REVIEW_H] - expected_fc)
        np.testing.assert_allclose(
            cs[i], expected, rtol=1e-4, atol=1e-4,
            err_msg=f"window {i} scores not from the true context",
        )


def test_conformity_scores_unfitted_falls_back_to_base(fitted_conformal_model):
    """The contract suite calls .new().conformity_scores on an UNFITTED clone;
    that path must keep base semantics (stateless fit-per-window under vmap)."""
    mm = fitted_conformal_model.new()
    mm.model_ = {}
    mm.max_steps = 5  # keep the per-window vmapped re-fits cheap
    cs = mm.conformity_scores(_trend_seasonal_series(n=200))
    assert cs.shape == (_REVIEW_NW, _REVIEW_H)
    assert bool(jnp.all(jnp.isfinite(cs)))


def test_revin_denormalize_guards_zero_gamma():
    """X6: canonical RevIN divides by gamma + eps^2 — a trained-to-zero gamma
    must not produce inf."""
    p = {"gamma": jnp.zeros((1,)), "beta": jnp.zeros((1,))}
    stats = {"mu": jnp.zeros((1,)), "sigma": jnp.ones((1,))}
    out = _backend.revin_denormalize(jnp.ones((4,)), p, stats)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_conformity_scores_pad_regime_short_series():
    """X2 pad regime (adversarial-review finding 1): when train_end < ctx_len,
    tails are left-edge-padded to the static ctx length. Scores must be finite,
    correctly shaped, and window-varying — the true-context equivalence claim
    is deliberately scoped to the no-pad regime."""
    h, nw = 12, 4
    y = _trend_seasonal_series(n=80)  # base_train_end = 80-48 = 32 < ctx 64
    m = XLSTM(
        ctx_len=64, num_layers=1, embed_dim=16, num_heads=2,
        max_steps=5, seed=0,
        conformal_params=ConformalIntervals(n_windows=nw, h=h),
    )
    m.fit(y)  # fit shrinks ctx for n=80; force the pad path via a fresh call
    cs = m.conformity_scores(y)
    assert cs.shape[1] == h and cs.shape[0] >= 2
    assert bool(jnp.all(jnp.isfinite(cs)))
    assert not bool(jnp.allclose(cs[0], cs[-1]))


def test_stateless_forecast_level_emits_intervals():
    """X3 companion (adversarial-review finding 14): XLSTM's STATELESS
    forecast(level=...) is in no fleet gate — pin it here. Routes through the
    slow path (tmp fit -> predict(level))."""
    m = XLSTM(
        ctx_len=16, num_layers=1, embed_dim=16, num_heads=2,
        max_steps=5, seed=0,
        conformal_params=ConformalIntervals(n_windows=2, h=6),
    )
    out = m.forecast(y=_trend_seasonal_series(n=120), h=6, level=[80])
    assert "lo-80" in out and "hi-80" in out
    assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))


def test_compute_dtype_default_f32_and_bf16_optin():
    """Compute dtype is config-static and defaults to float32, since bf16 is a
    marked XLA-CPU pessimization; bf16 stays opt-in for GPU."""
    y = _trend_seasonal_series(n=200)
    m32 = XLSTM(ctx_len=32, max_steps=5, seed=0).fit(y)
    m16 = XLSTM(ctx_len=32, max_steps=5, seed=0, compute_dtype="bfloat16").fit(y)
    p32 = np.asarray(m32.predict(h=6)["mean"])
    p16 = np.asarray(m16.predict(h=6)["mean"])
    assert np.all(np.isfinite(p32)) and np.all(np.isfinite(p16))
    assert not np.array_equal(p32, p16)  # dtype genuinely flows into compute
    with pytest.raises(ValueError, match="compute_dtype"):
        XLSTM(compute_dtype="float16")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""Tests for the StemGNN forecaster (flax.nnx port of neuralforecast's StemGNN).

Sections: Losses -> Scaler -> Module -> Training -> Model -> Namespace. The key
anchors are the NumPy-reference forward at n_series=3 (graph path alive), the
analytic parameter-count formula, the exact-DFT GEMM checks against ``jnp.fft``,
and the N=1 degeneracy pins (input-independent constant head).
"""
import io
import logging
import pickle
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.stemgnn.stemgnn_losses import (
    LOSSES, MultiQuantileLoss, huber, mae, mse, outputsize_multiplier, resolve,
)
from chronax.models.stemgnn.stemgnn_model import StemGNN
from chronax.models.stemgnn.stemgnn_module import (
    _DFT4_IM, _DFT4_RE, _IRFFT4, GLU, StemGNNNet, StockBlockLayer, TorchGRU,
    _dft4, _irfft4,
)
from chronax.models.stemgnn.stemgnn_scaler import (
    IdentityScaler, RobustScaler, resolve_scaler,
)
from chronax.models.stemgnn.stemgnn_training import (
    _lr_schedule, build_windows, forward_loss, predict_step, train,
)
from chronax.utils import ConformalIntervals

import optax


def _make_y(T=120, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(T, dtype=np.float32)
    return jnp.asarray(20.0 + 0.1 * t + 5.0 * np.sin(t / 6.0) + 0.5 * rng.randn(T).astype(np.float32))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# === Losses ===

def test_mae_zero_at_match():
    x = jnp.arange(6.0)
    assert float(mae(x, x)) == 0.0


def test_mse_known_value():
    assert float(mse(jnp.zeros(4), 2.0 * jnp.ones(4))) == pytest.approx(4.0)


def test_huber_regions():
    # |r| <= 1 quadratic; beyond linear.
    assert float(huber(jnp.array([0.5]), jnp.array([0.0]))) == pytest.approx(0.125)
    assert float(huber(jnp.array([3.0]), jnp.array([0.0]))) == pytest.approx(2.5)


def test_resolve_string_and_multiplier():
    assert resolve("mae") is LOSSES["mae"]
    assert outputsize_multiplier(resolve("mse")) == 1
    with pytest.raises(ValueError):
        resolve("nope")


def test_mqloss_multiplier_median_and_pickle():
    q = MultiQuantileLoss((0.1, 0.5, 0.9))
    assert q.outputsize_multiplier == 3 and 0.5 in q.quantiles
    q2 = pickle.loads(pickle.dumps(q))
    assert q2.quantiles == q.quantiles
    with pytest.raises(ValueError):
        MultiQuantileLoss((0.1, 0.9))          # missing 0.5
    with pytest.raises(ValueError):
        MultiQuantileLoss((0.0, 0.5))          # out of range


def test_mqloss_pinball_value():
    q = MultiQuantileLoss((0.25, 0.5, 0.75))
    pred = jnp.zeros((1, 1, 3))
    target = jnp.ones((1, 1))                  # err = 1 for every quantile head
    # sum over quantiles of q*err = 0.25 + 0.5 + 0.75 = 1.5 — NF's effective
    # reduction (its 1/len(quantiles) factor is dead), matching the trainer's
    # masked inline branch.
    assert float(q(pred, target)) == pytest.approx(1.5)


# === Scaler ===

def test_robust_scaler_round_trip():
    s = RobustScaler()
    x = jnp.asarray(np.random.RandomState(0).randn(3, 10).astype(np.float32) * 5 + 2)
    shift, scale = s.stats(x, axis=1)
    z = s.transform(x, shift, scale)
    np.testing.assert_allclose(np.asarray(s.inverse(z, shift, scale)), np.asarray(x), rtol=1e-5, atol=1e-5)


def test_robust_scaler_torch_median_even_count():
    # torch nanmedian returns the LOWER of the two middle values: median([1,2,3,4]) = 2, not 2.5.
    s = RobustScaler()
    shift, _ = s.stats(jnp.asarray([[1.0, 2.0, 3.0, 4.0]]), axis=1)
    assert float(shift[0, 0]) == 2.0


def test_robust_scaler_mad_is_median_abs_dev():
    # Scale is the MEDIAN absolute deviation. [1,2,3,10,50]: median 3,
    # |dev| = [2,1,0,7,47], median-AD = 2 (mean-AD would be 11.4).
    s = RobustScaler()
    _, scale = s.stats(jnp.asarray([[1.0, 2.0, 3.0, 10.0, 50.0]]), axis=1)
    assert float(scale[0, 0]) == pytest.approx(2.0, abs=1e-4)


def test_robust_scaler_zero_mad_std_fallback():
    # Majority-constant window: MAD = 0 -> scale falls back to std * 0.6744897501960817.
    x = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 9.0]], dtype=np.float32)
    s = RobustScaler()
    _, scale = s.stats(jnp.asarray(x), axis=1)
    std = np.sqrt(np.mean((x - x.mean()) ** 2))
    assert float(scale[0, 0]) == pytest.approx(std * 0.6744897501960817 + 1e-6, rel=1e-5)


def test_robust_scaler_constant_window_unit_scale():
    s = RobustScaler()
    _, scale = s.stats(jnp.full((1, 8), 3.0), axis=1)
    assert float(scale[0, 0]) == pytest.approx(1.0 + 1e-6, rel=1e-6)


def test_resolve_scaler():
    assert isinstance(resolve_scaler("robust"), RobustScaler)
    assert isinstance(resolve_scaler("identity"), IdentityScaler)
    inst = RobustScaler()
    assert resolve_scaler(inst) is inst
    with pytest.raises(ValueError):
        resolve_scaler("nope")


# === Module: DFT GEMMs ===

def test_dft4_gemm_matches_jnp_fft():
    x = jnp.asarray(np.random.RandomState(1).randn(2, 4, 3, 5).astype(np.float32))
    re, im = _dft4(x)
    ref = jnp.fft.fft(x, axis=1)
    np.testing.assert_allclose(np.asarray(re), np.asarray(ref.real), atol=1e-5)
    np.testing.assert_allclose(np.asarray(im), np.asarray(ref.imag), atol=1e-5)


def test_irfft4_gemm_matches_jnp_irfft():
    rng = np.random.RandomState(2)
    re = jnp.asarray(rng.randn(2, 4, 3, 5).astype(np.float32))
    im = jnp.asarray(rng.randn(2, 4, 3, 5).astype(np.float32))
    got = _irfft4(re, im)
    # torch/numpy/jax irfft(n=4) crop the 4-bin input to the first 3 Hermitian bins.
    ref = jnp.fft.irfft(re + 1j * im, n=4, axis=1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-5)


def test_dft_constant_tables():
    # Forward twiddles for n=4 are exactly {0, ±1}; the inverse map is exact
    # quarters with ZERO columns for Im(bin 0) and Im(bin 2) (the imaginary
    # parts of the DC and Nyquist bins are ignored).
    assert set(np.unique(_DFT4_RE)) <= {-1.0, 0.0, 1.0}
    assert set(np.unique(_DFT4_IM)) <= {-1.0, 0.0, 1.0}
    quarters = _IRFFT4 * 4.0
    np.testing.assert_array_equal(quarters, np.round(quarters))
    np.testing.assert_array_equal(_IRFFT4[:, 3], np.zeros(4))   # Im0 column
    np.testing.assert_array_equal(_IRFFT4[:, 5], np.zeros(4))   # Im2 (Nyquist) column


# === Module: GLU / GRU ===

def test_glu_fused_equals_two_gemms():
    glu = GLU(6, 5, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(3).randn(7, 6).astype(np.float32))
    kernel = np.asarray(glu.linear.kernel.value)
    bias = np.asarray(glu.linear.bias.value)
    left = np.asarray(x) @ kernel[:, :5] + bias[:5]
    right = np.asarray(x) @ kernel[:, 5:] + bias[5:]
    np.testing.assert_allclose(np.asarray(glu(x)), left * _sigmoid(right), atol=1e-6)


def test_glu_shapes_and_init_bound():
    glu = GLU(10, 4, rngs=nnx.Rngs(0))
    assert glu.linear.kernel.value.shape == (10, 8)
    bound = 1.0 / np.sqrt(10)
    assert float(jnp.max(jnp.abs(glu.linear.kernel.value))) <= bound
    assert glu(jnp.ones((2, 10))).shape == (2, 4)


def test_torch_gru_matches_numpy_reference():
    gru = TorchGRU(5, 3, rngs=nnx.Rngs(0))
    x = np.random.RandomState(4).randn(4, 2, 5).astype(np.float32)   # [seq, B, I]
    w_ih = np.asarray(gru.w_ih.value)
    w_hh = np.asarray(gru.w_hh.value)
    b_ih = np.asarray(gru.b_ih.value)
    b_hh = np.asarray(gru.b_hh.value)
    h = np.zeros((2, 3), np.float32)
    outs = []
    for t in range(4):
        gi = x[t] @ w_ih.T + b_ih
        gh = h @ w_hh.T + b_hh
        i_r, i_z, i_n = np.split(gi, 3, axis=-1)
        h_r, h_z, h_n = np.split(gh, 3, axis=-1)
        r = _sigmoid(i_r + h_r)
        z = _sigmoid(i_z + h_z)
        n = np.tanh(i_n + r * h_n)              # b_hn INSIDE the reset product
        h = (1.0 - z) * n + z * h
        outs.append(h.copy())
    got = np.asarray(gru(jnp.asarray(x)))
    np.testing.assert_allclose(got, np.stack(outs), atol=1e-6)


def test_torch_gru_layout_and_init():
    gru = TorchGRU(7, 2, rngs=nnx.Rngs(0))
    assert gru.w_ih.value.shape == (6, 7)
    assert gru.w_hh.value.shape == (6, 2)
    assert gru.b_ih.value.shape == (6,) and gru.b_hh.value.shape == (6,)
    bound = 1.0 / np.sqrt(2)                    # U(±1/sqrt(hidden)) for ALL params
    for p in (gru.w_ih, gru.w_hh, gru.b_ih, gru.b_hh):
        assert float(jnp.max(jnp.abs(p.value))) <= bound


# === Module: graph ===

def _np_latent_correlation(net, x, cheb="nf_zero"):
    """NumPy reference of the latent correlation layer (deterministic path)."""
    w_ih = np.asarray(net.gru.w_ih.value, np.float64)
    w_hh = np.asarray(net.gru.w_hh.value, np.float64)
    b_ih = np.asarray(net.gru.b_ih.value, np.float64)
    b_hh = np.asarray(net.gru.b_hh.value, np.float64)
    seq = np.transpose(x, (2, 0, 1)).astype(np.float64)      # [N, B, L]
    N, B = seq.shape[0], seq.shape[1]
    H = w_hh.shape[1]
    h = np.zeros((B, H))
    outs = []
    for t in range(N):
        gi = seq[t] @ w_ih.T + b_ih
        gh = h @ w_hh.T + b_hh
        i_r, i_z, i_n = np.split(gi, 3, axis=-1)
        h_r, h_z, h_n = np.split(gh, 3, axis=-1)
        r = _sigmoid(i_r + h_r)
        z = _sigmoid(i_z + h_z)
        n = np.tanh(i_n + r * h_n)
        h = (1.0 - z) * n + z * h
        outs.append(h.copy())
    inp = np.transpose(np.stack(outs), (1, 0, 2))            # [B, N, H]
    inp = np.transpose(inp, (0, 2, 1))                       # [B, H, N]
    key = inp @ np.asarray(net.weight_key.value, np.float64)
    query = inp @ np.asarray(net.weight_query.value, np.float64)
    data = key[:, :, 0][:, :, None] + query[:, :, 0][:, None, :]
    data = np.where(data > 0, data, net.leaky_rate * data)
    e = np.exp(data - data.max(axis=2, keepdims=True))
    att = e / e.sum(axis=2, keepdims=True)
    A = att.mean(axis=0)
    degree = A.sum(axis=1)                                   # PRE-symmetrization
    A = 0.5 * (A + A.T)
    dinv = 1.0 / (np.sqrt(degree) + 1e-7)
    lap = dinv[:, None] * (np.diag(degree) - A) * dinv[None, :]
    n_ = A.shape[0]
    t0 = np.zeros((n_, n_)) if cheb == "nf_zero" else np.eye(n_)
    t1 = lap
    t2 = 2.0 * (lap @ t1) - t0
    t3 = 2.0 * (lap @ t2) - t1
    return np.stack([t0, t1, t2, t3])


def _tiny_net(n_series=1, cheb="nf_zero", mult=1, seed=0, L=8, h=4, m=2):
    return StemGNNNet(
        h=h, input_size=L, n_series=n_series, multi_layer=m,
        outputsize_multiplier=mult, chebyshev_first_term=cheb, rngs=nnx.Rngs(seed),
    )


def test_attention_matches_torch_construction():
    # data[b, i, j] = key_i + query_j (outer sum), then LeakyReLU(0.2) and row
    # softmax -> rows sum to 1.
    net = _tiny_net(n_series=3)
    x = jnp.asarray(np.random.RandomState(5).randn(4, 8, 3).astype(np.float32))
    mul_L = net._latent_correlation(x, None, True)
    ref = _np_latent_correlation(net, np.asarray(x))
    np.testing.assert_allclose(np.asarray(mul_L), ref, atol=1e-5)


def test_laplacian_matches_numpy_reference():
    net = _tiny_net(n_series=4, L=6, h=3)
    x = jnp.asarray(np.random.RandomState(6).randn(5, 6, 4).astype(np.float32))
    mul_L = net._latent_correlation(x, None, True)
    ref = _np_latent_correlation(net, np.asarray(x))
    np.testing.assert_allclose(np.asarray(mul_L), ref, atol=1e-5)


def test_mul_L_zero_at_n1_nf_zero():
    net = _tiny_net(n_series=1)
    x = jnp.asarray(np.random.RandomState(7).randn(6, 8, 1).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(net._latent_correlation(x, None, True)), 0.0)
    # Dropout-active leg: diag(degree) - A == 0 REGARDLESS of the dropout draw.
    np.testing.assert_array_equal(
        np.asarray(net._latent_correlation(x, jax.random.PRNGKey(0), False)), 0.0
    )


def test_mul_L_nonzero_at_n3():
    net = _tiny_net(n_series=3)
    x = jnp.asarray(np.random.RandomState(8).randn(4, 8, 3).astype(np.float32))
    mul_L = np.asarray(net._latent_correlation(x, None, True))
    assert np.abs(mul_L[1]).max() > 1e-6      # T1 = laplacian is alive at N>1


def test_cheb_identity_mode_n1_closed_form():
    # At N=1 lap == 0, so with T0=I the standard Chebyshev basis is
    # T0=I, T1=0, T2=2*0-I=-I, T3=0 -> mul_L = [1, 0, -1, 0].
    net = _tiny_net(n_series=1, cheb="identity")
    x = jnp.asarray(np.random.RandomState(9).randn(3, 8, 1).astype(np.float32))
    mul_L = np.asarray(net._latent_correlation(x, None, True))
    np.testing.assert_array_equal(mul_L[:, 0, 0], np.array([1.0, 0.0, -1.0, 0.0]))


# === Module: forward ===

def _np_lin(lin, x):
    return x @ np.asarray(lin.kernel.value, np.float64) + np.asarray(lin.bias.value, np.float64)


def _np_glu(glu, x):
    z = _np_lin(glu.linear, x)
    left, right = np.split(z, 2, axis=-1)
    return left * _sigmoid(right)


def _np_reference_forward(net, x, cheb="nf_zero"):
    """Full NumPy reimplementation of the StemGNN forward (np.fft as the DFT
    reference), in float64, from the net's own extracted parameters."""
    B, L, N = x.shape
    mul_L = _np_latent_correlation(net, x, cheb=cheb)
    X = np.transpose(x, (0, 2, 1)).astype(np.float64)        # [B, N, L]

    def block(idx, X):
        blk = net.stock_blocks[idx]
        gfted = np.einsum("knm,bml->bknl", mul_L, X)          # [B, 4, N, L]
        ff = np.fft.fft(gfted, axis=1)                        # reference DFT
        re = ff.real.transpose(0, 2, 1, 3).reshape(B, N, 4 * L)
        im = ff.imag.transpose(0, 2, 1, 3).reshape(B, N, 4 * L)
        for i in range(3):
            re = _np_glu(blk.glus[2 * i], re)
            im = _np_glu(blk.glus[2 * i + 1], im)
        re = re.reshape(B, N, 4, -1).transpose(0, 2, 1, 3)
        im = im.reshape(B, N, 4, -1).transpose(0, 2, 1, 3)
        iff = np.fft.irfft((re + 1j * im)[:, :3], n=4, axis=1)   # crop to 3 Hermitian bins
        igfted = np.einsum("bkns,kst->bnt", iff, np.asarray(blk.weight.value, np.float64))
        fcst = _np_lin(blk.forecast_result, _sigmoid(_np_lin(blk.forecast, igfted)))
        if idx == 0:
            back = _sigmoid(_np_lin(blk.backcast, igfted) - _np_lin(blk.backcast_short_cut, X))
        else:
            back = None
        return fcst, back, gfted

    f0, X1, gfted0 = block(0, X)
    f1, _, _ = block(1, X1)
    out = f0 + f1
    z = _np_lin(net.fc1, out)
    z = np.where(z > 0, z, 0.01 * z)                          # fc LeakyReLU slope 0.01
    out = _np_lin(net.fc2, z)
    out = np.transpose(out, (0, 2, 1)).reshape(B, net.h, net.outputsize_multiplier * N)
    return out, gfted0


def test_numpy_reference_forward_n3():
    net = _tiny_net(n_series=3, L=8, h=4, m=2)
    x = np.random.RandomState(10).randn(4, 8, 3).astype(np.float32)
    ref, gfted0 = _np_reference_forward(net, x)
    assert np.abs(gfted0).max() > 1e-4        # graph path ALIVE (non-constant input)
    got = np.asarray(net(jnp.asarray(x)))
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
    # Input-dependence at N=3 (non-constant perturbation — constant shifts are
    # near-annihilated by the Laplacian row structure).
    x2 = x + np.random.RandomState(11).randn(*x.shape).astype(np.float32)
    assert np.abs(np.asarray(net(jnp.asarray(x2))) - got).max() > 1e-5


def test_numpy_reference_forward_n3_identity_cheb():
    net = _tiny_net(n_series=3, cheb="identity", L=8, h=4, m=2)
    x = np.random.RandomState(12).randn(4, 8, 3).astype(np.float32)
    ref, _ = _np_reference_forward(net, x, cheb="identity")
    np.testing.assert_allclose(np.asarray(net(jnp.asarray(x))), ref, rtol=1e-4, atol=1e-4)


def test_n1_degeneracy_input_independent():
    # At N=1 (the default path) the output is an input-independent constant —
    # bit-identical across rows, inputs, and scales; train mode too (dropout
    # hits only the inert adjacency).
    net = _tiny_net(n_series=1)
    rng = np.random.RandomState(13)
    x1 = jnp.asarray(rng.randn(3, 8, 1).astype(np.float32) * 10)
    x2 = jnp.asarray(rng.randn(3, 8, 1).astype(np.float32) * 100 + 55)
    o1, o2 = np.asarray(net(x1)), np.asarray(net(x2))
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(o1[0], o1[1])
    key = jax.random.PRNGKey(0)
    t1 = np.asarray(net(x1, dropout_key=key, deterministic=False))
    t2 = np.asarray(net(x2, dropout_key=key, deterministic=False))
    np.testing.assert_array_equal(t1, t2)


def test_identity_cheb_restores_data_path_at_n1():
    # With T0=I the N=1 forward depends on its input.
    net = _tiny_net(n_series=1, cheb="identity")
    rng = np.random.RandomState(14)
    x1 = jnp.asarray(rng.randn(3, 8, 1).astype(np.float32))
    x2 = jnp.asarray(rng.randn(3, 8, 1).astype(np.float32))
    assert np.abs(np.asarray(net(x1)) - np.asarray(net(x2))).max() > 1e-5


def test_dropout_live_at_n3():
    net = _tiny_net(n_series=3)
    x = jnp.asarray(np.random.RandomState(15).randn(4, 8, 3).astype(np.float32))
    k1, k2 = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
    a = np.asarray(net(x, dropout_key=k1, deterministic=False))
    b = np.asarray(net(x, dropout_key=k1, deterministic=False))
    c = np.asarray(net(x, dropout_key=k2, deterministic=False))
    np.testing.assert_array_equal(a, b)       # same key -> same mask
    assert np.abs(a - c).max() > 1e-7         # different key -> different graph
    with pytest.raises(ValueError):
        net(x, deterministic=False)           # training forward requires a key


def test_block1_dead_shortcut_params():
    # backcast_short_cut is built on BOTH blocks but used only in block 0;
    # verify block 1's copy is dead by perturbing it.
    net = _tiny_net(n_series=3)
    assert hasattr(net.stock_blocks[0], "backcast")
    assert not hasattr(net.stock_blocks[1], "backcast")
    x = jnp.asarray(np.random.RandomState(16).randn(4, 8, 3).astype(np.float32))
    before = np.asarray(net(x))
    sc = net.stock_blocks[1].backcast_short_cut
    sc.kernel.value = sc.kernel.value + 100.0
    sc.bias.value = sc.bias.value - 7.0
    np.testing.assert_array_equal(np.asarray(net(x)), before)


def _expected_params(N, L, m, h, mult):
    S, O = m * L, 4 * m * L
    gru = 3 * N * L + 3 * N * N + 6 * N
    def block(i):
        p = (4 * S * S) + (S * S + S) + (S * L + L) + (L * L + L)
        p += 4 * (4 * L * O + O) + 8 * (O * O + O)            # 6 GLUs, 2 Linears each
        if i == 0:
            p += S * L + L                                     # backcast head
        return p
    return 2 * N + gru + block(0) + block(1) + (L * L + L) + (L * h * mult + h * mult)


def test_param_count_formula():
    net = _tiny_net(n_series=3, L=8, h=4, m=2, mult=2)
    actual = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(net, nnx.Param)))
    assert actual == _expected_params(3, 8, 2, 4, 2)
    # Harness dims (L=72, h=24, m=5, N=1) give the expected total.
    assert _expected_params(1, 72, 5, 24, 1) == 37_922_363


def test_net_output_shapes():
    assert _tiny_net(n_series=1)(jnp.zeros((2, 8, 1))).shape == (2, 4, 1)
    assert _tiny_net(n_series=3)(jnp.zeros((2, 8, 3))).shape == (2, 4, 3)
    assert _tiny_net(n_series=3, mult=2)(jnp.zeros((2, 8, 3))).shape == (2, 4, 6)


def test_net_rejects_bad_nstacks_and_cheb_flag():
    with pytest.raises(ValueError, match="n_stacks"):
        StemGNNNet(h=4, input_size=8, n_stacks=3, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="chebyshev_first_term"):
        StemGNNNet(h=4, input_size=8, chebyshev_first_term="eye", rngs=nnx.Rngs(0))


# === Training ===

def test_build_windows_nf_padding_semantics():
    y = jnp.arange(10.0)
    w, m = build_windows(y, input_size=3, h=2)
    assert w.shape == (7, 5) and m.shape == (7, 2)            # n = T - L
    np.testing.assert_array_equal(np.asarray(w[6]), [6.0, 7.0, 8.0, 9.0, 0.0])
    np.testing.assert_array_equal(np.asarray(m[6]), [1.0, 0.0])
    assert float(m.sum()) == 13.0                              # 6 full + 1 partial
    with pytest.raises(ValueError):
        build_windows(jnp.arange(3.0), input_size=3, h=2)


def test_masked_loss_ignores_padded_tail():
    net = _tiny_net(n_series=1, L=8, h=4)
    y = _make_y(20)
    w, m = build_windows(y, 8, 4)
    poisoned = jnp.where(m > 0, w[:, 8:], 1e6)
    w_poisoned = w.at[:, 8:].set(poisoned)
    kwargs = dict(h=4, input_size=8, scaler=resolve_scaler("robust"), loss_fn=resolve("mae"))
    clean = forward_loss(net, w, m, **kwargs)
    dirty = forward_loss(net, w_poisoned, m, **kwargs)
    np.testing.assert_allclose(float(clean), float(dirty), rtol=1e-6)


def test_lr_schedule_steplr_boundaries():
    sch = _lr_schedule(1e-3, 1000, 3)
    got = [float(sch(c)) for c in (0, 332, 333, 665, 666, 998, 999)]
    np.testing.assert_allclose(
        got, [1e-3, 1e-3, 5e-4, 5e-4, 2.5e-4, 2.5e-4, 1.25e-4], rtol=1e-6
    )


def test_lr_schedule_optax_count_semantics():
    # optax reads the count PRE-increment: update 1 sees schedule(0), update 2
    # sees schedule(1) — so a boundary at 1 halves the SECOND update, exactly
    # torch StepLR(step_size=1) stepped after each optimizer step.
    opt = optax.sgd(optax.piecewise_constant_schedule(1.0, {1: 0.5}))
    params = jnp.zeros(())
    state = opt.init(params)
    g = jnp.ones(())
    u1, state = opt.update(g, state, params)
    u2, state = opt.update(g, state, params)
    assert float(u1) == pytest.approx(-1.0)
    assert float(u2) == pytest.approx(-0.5)


def test_lr_schedule_no_decay_and_tiny_steps():
    assert _lr_schedule(1e-3, 1000, 0) == 1e-3                # decays disabled
    sch = _lr_schedule(1e-3, 3, 3)                             # d=1, boundaries {1, 2}
    np.testing.assert_allclose(
        [float(sch(c)) for c in (0, 1, 2)], [1e-3, 5e-4, 2.5e-4], rtol=1e-6
    )


def test_train_reduces_loss():
    # At N=1 (nf_zero) the head is a learned constant in scaled space; on a
    # strong trend the scaled targets sit at a consistent offset far from the
    # init constant, so training MUST move the loss down. (On a mean-reverting
    # series the init constant is already near-optimal and the per-batch losses
    # are pure noise — a vacuous test.)
    net = _tiny_net(n_series=1, L=12, h=4)
    t = np.arange(80, dtype=np.float32)
    y = jnp.asarray(5.0 * t + 0.1 * np.random.RandomState(0).randn(80).astype(np.float32))
    losses = train(net, y, h=4, input_size=12, max_steps=150, windows_batch_size=16,
                   lr=3e-2, num_lr_decays=3, seed=0, loss_fn=resolve("mae"),
                   scaler=resolve_scaler("robust"))
    assert losses.shape == (150,)
    assert float(jnp.mean(losses[-10:])) < 0.7 * float(jnp.mean(losses[:10]))


def test_train_divergence_guard():
    net = _tiny_net(n_series=1, L=12, h=4)
    with pytest.raises(RuntimeError, match="diverged"):
        train(net, _make_y(60), h=4, input_size=12, max_steps=8, windows_batch_size=8,
              lr=1e12, num_lr_decays=0, seed=0, loss_fn=resolve("mae"),
              scaler=resolve_scaler("robust"))


def test_predict_step_repeatable():
    net = _tiny_net(n_series=1, L=12, h=4)
    y = _make_y(40)
    p1 = predict_step(net, y[-12:], h=4, input_size=12, scaler=resolve_scaler("robust"))
    p2 = predict_step(net, y[-12:], h=4, input_size=12, scaler=resolve_scaler("robust"))
    np.testing.assert_array_equal(np.asarray(p1), np.asarray(p2))


def test_train_is_vmap_traceable():
    # conformity_scores vmaps forecast over CV windows; the guard must no-op, not raise.
    y = _make_y(60)

    def run(seed):
        net = StemGNNNet(h=4, input_size=12, n_series=1, multi_layer=2, rngs=nnx.Rngs(0))
        return train(net, y, h=4, input_size=12, max_steps=4, windows_batch_size=16,
                     lr=1e-3, num_lr_decays=3, seed=seed, loss_fn=resolve("mae"),
                     scaler=resolve_scaler("identity"))

    out = jax.vmap(run)(jnp.arange(3))
    assert out.shape == (3, 4) and bool(jnp.all(jnp.isfinite(out)))


# === Model ===

_TINY = dict(h=4, input_size=12, multi_layer=2, max_steps=3, windows_batch_size=16,
             random_seed=0)


def test_is_base_forecaster_and_uses_exog_false():
    from chronax.models.base_forecaster import BaseForecaster
    assert issubclass(StemGNN, BaseForecaster)
    assert StemGNN.uses_exog is False


def test_nf_defaults():
    m = StemGNN(h=24)
    assert m.input_size == 72                 # chronax ergonomic default: 3h
    assert m.n_stacks == 2
    assert m.multi_layer == 5
    assert m.dropout_rate == 0.5
    assert m.leaky_rate == 0.2
    assert m.max_steps == 1000
    assert m.learning_rate == 1e-3
    assert m.num_lr_decays == 3
    assert m.windows_batch_size == 32         # StemGNN default (not the generic 128)
    assert m.scaler_type == "robust"
    assert m.loss == "mae"
    assert m.chebyshev_first_term == "nf_zero"
    assert m.random_seed == 1
    assert m.alias == "StemGNN"


def test_nstacks_and_cheb_validation_raises():
    with pytest.raises(ValueError, match="n_stacks"):
        StemGNN(h=4, n_stacks=3)
    with pytest.raises(ValueError, match="chebyshev_first_term"):
        StemGNN(h=4, chebyshev_first_term="paper")


def test_fit_predict_shapes():
    m = StemGNN(**_TINY).fit(_make_y(60))
    out = m.predict(h=4)
    assert set(out) == {"mean"} and out["mean"].shape == (4,)
    assert bool(jnp.all(jnp.isfinite(out["mean"])))


def test_predict_smaller_h_slices_larger_raises():
    m = StemGNN(**_TINY).fit(_make_y(60))
    assert m.predict(h=2)["mean"].shape == (2,)
    with pytest.raises(ValueError, match="h <= 4"):
        m.predict(h=5)
    with pytest.raises(ValueError, match="positive"):
        m.predict(h=0)


def test_short_series_and_2d_raise():
    with pytest.raises(ValueError, match="too short"):
        StemGNN(**_TINY).fit(jnp.arange(12.0))
    with pytest.raises(ValueError, match="1-D"):
        StemGNN(**_TINY).fit(jnp.zeros((10, 2)))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        StemGNN(**_TINY).predict(h=2)


def test_fit_X_raises():
    with pytest.raises(NotImplementedError, match="Exogenous"):
        StemGNN(**_TINY).fit(_make_y(60), X=jnp.ones((60, 2)))
    with pytest.raises(NotImplementedError, match="Exogenous"):
        StemGNN(**_TINY).forecast(_make_y(60), h=4, X_future=jnp.ones((4, 2)))


def test_repeated_predict_identical():
    m = StemGNN(**_TINY).fit(_make_y(60))
    np.testing.assert_array_equal(
        np.asarray(m.predict(h=4)["mean"]), np.asarray(m.predict(h=4)["mean"])
    )


def test_forecast_equals_fit_predict():
    y = _make_y(60)
    a = StemGNN(**_TINY).forecast(y, h=4)["mean"]
    b = StemGNN(**_TINY).fit(y).predict(h=4)["mean"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)


def test_fitted_nan_head():
    y = _make_y(60)
    out = StemGNN(**_TINY).forecast(y, h=4, fitted=True)
    f = np.asarray(out["fitted"])
    assert f.shape == (60,)
    assert np.all(np.isnan(f[:12])) and np.all(np.isfinite(f[12:]))


def test_n1_prediction_scaler_stats_pin():
    # Degeneracy pin at wrapper level: the network output is input-independent,
    # so the forecast depends on the context ONLY through the scaler's
    # (median, MAD) — permuting the context leaves it bit-identical; rescaling
    # changes it.
    m = StemGNN(**_TINY).fit(_make_y(60))
    ctx = m._context
    rng = np.random.RandomState(17)
    perm = jnp.asarray(np.asarray(ctx)[rng.permutation(12)])
    p_base = predict_step(m.model_, ctx, h=4, input_size=12, scaler=m._scaler)
    p_perm = predict_step(m.model_, perm, h=4, input_size=12, scaler=m._scaler)
    np.testing.assert_array_equal(np.asarray(p_base), np.asarray(p_perm))
    p_scaled = predict_step(m.model_, ctx * 3.0, h=4, input_size=12, scaler=m._scaler)
    assert np.abs(np.asarray(p_scaled) - np.asarray(p_base)).max() > 1e-4


def test_conformal_interval_keys_twice_then_pickle():
    # predict(level) TWICE, then pickle, then predict again -- single-call
    # conformal tests are structurally blind to tracer pollution.
    m = StemGNN(**_TINY).fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    out1 = m.predict(h=4, level=[80])
    out2 = m.predict(h=4, level=[80])
    assert "lo-80" in out1 and "hi-80" in out1
    assert bool(jnp.all(out1["lo-80"] <= out1["mean"]))
    assert bool(jnp.all(out1["mean"] <= out1["hi-80"]))
    np.testing.assert_array_equal(np.asarray(out1["mean"]), np.asarray(out2["mean"]))
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(np.asarray(m2.predict(h=4)["mean"]),
                               np.asarray(out1["mean"]), rtol=1e-5, atol=1e-5)


def test_conformal_requires_params():
    m = StemGNN(**_TINY).fit(_make_y(60))
    with pytest.raises(ValueError, match="conformal_params"):
        m.predict(h=4, level=[80])


def test_native_quantile_intervals():
    m = StemGNN(loss=MultiQuantileLoss((0.1, 0.5, 0.9)), **_TINY).fit(_make_y(60))
    out = m.predict(h=4, level=[80])
    assert {"mean", "lo-80", "hi-80"} <= set(out)
    assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))


def test_untrained_quantile_level_raises():
    m = StemGNN(loss=MultiQuantileLoss((0.1, 0.5, 0.9)), **_TINY).fit(_make_y(60))
    with pytest.raises(ValueError, match="not trained"):
        m.predict(h=4, level=[95])


def test_pickle_roundtrip_point():
    m = StemGNN(**_TINY).fit(_make_y(60))
    before = np.asarray(m.predict(h=4)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(np.asarray(m2.predict(h=4)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_pickle_roundtrip_quantile():
    m = StemGNN(loss=MultiQuantileLoss((0.25, 0.5, 0.75)), **_TINY).fit(_make_y(60))
    before = np.asarray(m.predict(h=4)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    assert isinstance(m2._loss_fn, MultiQuantileLoss)
    np.testing.assert_allclose(np.asarray(m2.predict(h=4)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_vmap_forecast_matches_python_loop():
    # conformity_scores vmaps forecast over windows.
    B, T, h = 4, 60, 4
    y_batch = jnp.stack([_make_y(T, seed=i) * (i + 1) for i in range(B)])
    m = StemGNN(**_TINY)
    seq = jnp.stack([m.forecast(y=y_batch[i], h=h)["mean"] for i in range(B)])
    vm = jax.vmap(lambda y: m.forecast(y=y, h=h)["mean"])(y_batch)
    np.testing.assert_allclose(np.asarray(seq), np.asarray(vm), rtol=5e-3, atol=5e-3)


def test_beats_naive_noisy_level_series():
    # In the default (nf_zero) mode the head is a learned constant in scaled
    # space, so the forecast ~ window median + c*MAD — on a mean-reverting noisy
    # level it beats last-point naive, which anchors on one noisy draw.
    rng = np.random.RandomState(42)
    y = (50.0 + 3.0 * rng.randn(300)).astype(np.float32)
    train_y, test_y = jnp.asarray(y[:-8]), y[-8:]
    m = StemGNN(h=8, input_size=24, multi_layer=2, max_steps=150,
                windows_batch_size=32, random_seed=0)
    pred = np.asarray(m.fit(train_y).predict(h=8)["mean"])
    naive = float(train_y[-1])
    assert np.mean(np.abs(pred - test_y)) < np.mean(np.abs(naive - test_y))


def test_beats_naive_identity_cheb_on_sine():
    # With T0=I the univariate model genuinely forecasts — on a clean seasonal
    # signal it must beat last-value naive.
    t = np.arange(150, dtype=np.float32)
    y = 20.0 + 10.0 * np.sin(2.0 * np.pi * t / 12.0)
    train_y, test_y = jnp.asarray(y[:-12]), y[-12:]
    m = StemGNN(h=12, input_size=36, multi_layer=2, max_steps=200,
                learning_rate=5e-3, windows_batch_size=32,
                chebyshev_first_term="identity", random_seed=0)
    pred = np.asarray(m.fit(train_y).predict(h=12)["mean"])
    naive = float(train_y[-1])
    assert np.mean(np.abs(pred - test_y)) < 0.8 * np.mean(np.abs(naive - test_y))


# === Namespace ===

_COMPILE_LOG_RE = re.compile(r"Compiling ")


def _count_compiles(fn):
    """Run fn under jax.log_compiles and return (result, n_xla_compilations)."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    loggers = [
        logging.getLogger("jax._src.dispatch"),
        logging.getLogger("jax._src.interpreters.pxla"),
    ]
    with jax.log_compiles(True):
        for lg in loggers:
            lg.addHandler(handler)
        try:
            out = fn()
            jax.block_until_ready(out)
        finally:
            for lg in loggers:
                lg.removeHandler(handler)
    return out, len(_COMPILE_LOG_RE.findall(buf.getvalue()))


def test_count_compiles_canary():
    # Live-fire proof _count_compiles can still see a compile: a fresh jit'd
    # closure always compiles (pjit keys on callable identity), so the counter
    # must report >=1. Guards test_refit_does_not_recompile against going
    # vacuously green if a jax bump renames the log message or logger paths.
    def fresh():
        @jax.jit
        def f(x):
            return x * 7.0 + 0.5
        return f(jnp.ones((3, 4)))

    _, n = _count_compiles(fresh)
    assert n >= 1


def test_refit_does_not_recompile():
    # The training program is a module-level nnx.jit keyed on config graphdefs
    # (value-__eq__ initializers, _adam_steplr/scaler singletons) and operand
    # shapes — a refit AND a fresh same-config instance must hit the cache with
    # zero XLA compiles; fit #2 is counted directly (no uncounted settle call).
    y = jnp.asarray(np.sin(np.arange(40) / 5.0), jnp.float32)
    kw = dict(h=4, input_size=8, multi_layer=2, max_steps=3,
              windows_batch_size=4, random_seed=0)
    m = StemGNN(**kw)
    m.fit(y)                                     # first fit pays the one compile
    _, n_refit = _count_compiles(lambda: m.fit(y)._train_y)
    assert n_refit == 0
    m2 = StemGNN(**kw)
    _, n_fresh = _count_compiles(lambda: m2.fit(y)._train_y)
    assert n_fresh == 0


def test_dropout_keys_fresh_and_det_repeatable():
    # Training threads one explicit dropout key per scan step (split off the
    # seed): different keys must produce different stochastic forwards, and the
    # deterministic path must be exactly repeatable (the _forward_det contract).
    # n_series=3 keeps the graph path ALIVE — at N=1 the Laplacian branch is
    # identically zero and dropout there is inert, so the discriminator would
    # be vacuous (the documented degeneracy).
    from chronax.models.stemgnn.stemgnn_module import StemGNNNet

    net = StemGNNNet(h=4, input_size=8, n_series=3, multi_layer=2,
                     dropout_rate=0.5, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(2, 8, 3), jnp.float32)
    a = net(x, dropout_key=jax.random.PRNGKey(1), deterministic=False)
    b = net(x, dropout_key=jax.random.PRNGKey(2), deterministic=False)
    assert not np.allclose(np.asarray(a), np.asarray(b))
    c = net(x, deterministic=True)
    d = net(x, deterministic=True)
    np.testing.assert_array_equal(np.asarray(c), np.asarray(d))


def test_importable_from_models_namespace():
    import chronax.models as M
    assert M.StemGNN is StemGNN
    assert "StemGNN" in M.__all__

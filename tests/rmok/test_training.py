"""Tests for chronax.models.rmok training infra + module."""
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.rmok.rmok_losses import mae as loss_mae, resolve as resolve_loss
from chronax.models.rmok.rmok_scaler import IdentityScaler, resolve_scaler


def test_masked_mae_drops_masked_elements():
    pred = jnp.array([[1.0, 2.0]]); target = jnp.array([[0.0, 0.0]])
    mask = jnp.array([[1.0, 0.0]])
    assert float(loss_mae(pred, target, mask)) == 1.0     # only the unmasked |1-0|


def test_resolve_loss_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        resolve_loss("nope")


def test_scaler_resolve_and_roundtrip():
    x = jnp.array([[1.0, 2.0, 3.0, 100.0]])
    for name in ("identity", "robust"):
        s = resolve_scaler(name)
        shift, scale = s.stats(x, axis=1)
        z = s.transform(x, shift, scale)
        np.testing.assert_allclose(np.asarray(s.inverse(z, shift, scale)), np.asarray(x), rtol=1e-5)
    with pytest.raises(ValueError, match="Unknown scaler"):
        resolve_scaler("nope")


import jax

from chronax.models.rmok.rmok_module import RMoKNet


def _net(h=4, input_size=8, **kw):
    kw.setdefault("n_series", 1); kw.setdefault("taylor_order", 3)
    kw.setdefault("jacobi_degree", 6); kw.setdefault("wavelet_function", "mexican_hat")
    kw.setdefault("dropout", 0.1); kw.setdefault("revin_affine", True)
    return RMoKNet(h=h, input_size=input_size, rngs=nnx.Rngs(0), **kw)


def test_forward_shape_finite_and_eval_deterministic():
    net = _net()
    x = jnp.asarray(np.random.RandomState(0).randn(3, 8, 1), dtype=jnp.float32)
    a = net(x, deterministic=True); b = net(x, deterministic=True)
    assert a.shape == (3, 4, 1) and bool(jnp.all(jnp.isfinite(a)))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)   # eval idempotent (dropout off)


def test_revin_std_is_sqrt_var_plus_eps_not_std_plus_eps():
    from chronax.models.rmok.rmok_module import _revin_norm
    x = jnp.asarray([[[0.0], [0.0], [0.0], [4.0]]], dtype=jnp.float32)   # [1,4,1]
    xn, mean, std = _revin_norm(x, None, None, affine=False)
    var = float(np.var([0, 0, 0, 4]))
    np.testing.assert_allclose(float(std[0, 0, 0]), np.sqrt(var + 1e-5), rtol=1e-6)


def test_revin_roundtrip_denorm_inverts_norm():
    from chronax.models.rmok.rmok_module import _revin_norm, _revin_denorm
    x = jnp.asarray(np.random.RandomState(1).randn(2, 8, 1), dtype=jnp.float32)
    xn, mean, std = _revin_norm(x, None, None, affine=False)
    np.testing.assert_allclose(np.asarray(_revin_denorm(xn, None, None, affine=False, mean=mean, std=std)),
                               np.asarray(x), rtol=1e-5)


def test_one_hot_gate_selects_single_expert():
    net = _net(dropout=0.0)
    net.gate_weight.value = jnp.zeros_like(net.gate_weight.value)
    net.gate_bias.value = jnp.array([-1e9, -1e9, -1e9, 1e9], dtype=jnp.float32)
    x = jnp.asarray(np.random.RandomState(2).randn(2, 8, 1), dtype=jnp.float32)
    out = np.asarray(net(x, deterministic=True))
    from chronax.models.rmok.rmok_module import _revin_norm, _revin_denorm
    xn, mean, std = _revin_norm(x, net.affine_weight.value, net.affine_bias.value, affine=True)
    xd = jnp.transpose(xn, (0, 2, 1)).reshape(2, 8)
    e3 = xd @ net.linear_weight.value.T + net.linear_bias.value          # [2, 4]
    y = jnp.transpose(e3.reshape(2, 1, 4), (0, 2, 1))
    y = _revin_denorm(y, net.affine_weight.value, net.affine_bias.value, affine=True, mean=mean, std=std)
    np.testing.assert_allclose(out, np.asarray(y.reshape(2, 4, 1)), rtol=1e-4)


def test_batchnorm_running_stats_used_in_eval_after_train():
    net = _net(dropout=0.0)
    x = jnp.asarray(np.random.RandomState(3).randn(16, 8, 1), dtype=jnp.float32)
    before = np.asarray(net(x, deterministic=True))
    for _ in range(5):
        # forward-only loop: updates the BatchStat running stats, NOT the params.
        # Do not add an optimizer step here — that would let BatchNorm's per-batch
        # renormalization cancel param movement and weaken the mutant detection.
        net(x, deterministic=False)
    after = np.asarray(net(x, deterministic=True))
    assert not np.allclose(before, after)


def test_taylor_expert_hand_computed_uses_full_order():
    # Independent recomputation at order=3 with the top (i=2) coeff contributing,
    # so a mutation that slices/hardcodes a lower order is caught (not just the
    # constructor-driven shape difference).
    from chronax.models.rmok.rmok_module import _taylor
    x = jnp.array([[2.0, 3.0]], dtype=jnp.float32)          # [B=1, L=2]
    coeffs = jnp.asarray(np.array([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]]], dtype=np.float32))  # [h=1,L=2,order=3]
    bias = jnp.array([[0.1]], dtype=jnp.float32)
    xn = np.asarray(x); cn = np.asarray(coeffs)
    expected = sum((xn ** i) @ cn[0, :, i] for i in range(3)) + 0.1   # 1 + 3 + 2 + 0.1 = 6.1
    np.testing.assert_allclose(np.asarray(_taylor(x, coeffs, bias, 3))[0], expected, rtol=1e-6)


def test_jacobi_expert_hand_computed_uses_degree_terms():
    # degree=1: cols = [1, 2*tanh(x)] (a=b=1); y = coeffs·[1, 2*tanh(x)]. A mutation
    # dropping the degree>0 term would return only coeffs[...,0].
    from chronax.models.rmok.rmok_module import _jacobi
    x = jnp.array([[1.0]], dtype=jnp.float32)               # [B=1, L=1]
    coeffs = jnp.asarray(np.array([[[3.0, 5.0]]], dtype=np.float32))  # [L=1, h=1, degree+1=2]
    expected = 3.0 + 5.0 * (2.0 * np.tanh(1.0))
    np.testing.assert_allclose(float(np.asarray(_jacobi(x, coeffs, 1))[0, 0]), expected, rtol=1e-6)


def test_wavelet_invalid_raises_and_all_valid_finite():
    with pytest.raises(ValueError, match="wavelet"):
        _net(wavelet_function="nope")
    for w in ("mexican_hat", "morlet", "dog", "meyer", "shannon"):
        x = jnp.asarray(np.random.RandomState(4).randn(2, 8, 1), dtype=jnp.float32)
        assert bool(jnp.all(jnp.isfinite(_net(wavelet_function=w)(x, deterministic=True))))


def test_unused_weight1_is_a_param_but_frozen_by_zero_grad():
    net = _net()
    x = jnp.asarray(np.random.RandomState(5).randn(4, 8, 1), dtype=jnp.float32)
    g = nnx.grad(lambda m: jnp.sum(m(x, deterministic=True)))(net)
    w1_grad = [np.asarray(v.value) for p, v in nnx.to_flat_state(g) if "weight1" in "/".join(map(str, p))]
    assert w1_grad and np.all(w1_grad[0] == 0.0)                        # unused -> exactly zero grad


from chronax.models.rmok.rmok_training import (
    build_windows, predict_step, scaled_forward_loss, train)


def _y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def test_build_windows_shape_and_padding():
    y = _y(60)
    w, m = build_windows(y, input_size=36, h=12)          # NF: n_windows = 60-36
    assert w.shape == (24, 48) and m.shape == (24, 48)
    assert float(m[:, :36].min()) == 1.0                  # insample always real
    np.testing.assert_array_equal(np.asarray(m[-1, 36:]), [1.0] + [0.0] * 11)
    np.testing.assert_array_equal(np.asarray(w[-1, 37:]), np.zeros(11))
    assert float(w[-1, 36]) == float(y[-1])
    assert float(m.sum()) == 1086.0


def test_scaled_forward_loss_scalar():
    w, m = build_windows(_y(), input_size=8, h=4)
    loss = scaled_forward_loss(_net(), w[:8], m[:8], h=4, input_size=8, scaler=IdentityScaler())
    assert loss.shape == () and jnp.isfinite(loss)


def test_train_decreases_loss_and_is_deterministic():
    l1 = train(_net(), _y(), h=4, input_size=8, max_steps=30, windows_batch_size=32,
               lr=1e-2, seed=0, scaler=IdentityScaler())
    l2 = train(_net(), _y(), h=4, input_size=8, max_steps=30, windows_batch_size=32,
               lr=1e-2, seed=0, scaler=IdentityScaler())
    assert l1.shape == (30,) and float(l1[-1]) < float(l1[0])
    np.testing.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=1e-5)   # dropout+BN streams deterministic


def test_train_oversample_small_n():
    losses = train(_net(), _y(20), h=4, input_size=8, max_steps=8, windows_batch_size=64,
                   lr=1e-3, seed=0, scaler=IdentityScaler())
    assert losses.shape == (8,) and jnp.all(jnp.isfinite(losses))


def test_train_raises_on_nonfinite_loss():
    y = _y().at[50].set(jnp.nan)
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        train(_net(), y, h=4, input_size=8, max_steps=5, windows_batch_size=32,
              lr=1e-4, seed=0, scaler=IdentityScaler())


def test_sample_batch_idx_regimes():
    from chronax.models.rmok.rmok_training import _sample_batch_idx
    keys = jax.random.split(jax.random.PRNGKey(0), 7)
    idx = _sample_batch_idx(keys, n_windows=500, windows_batch_size=64)
    assert idx.shape == (7, 64) and int(idx.min()) >= 0 and int(idx.max()) < 500
    for row in np.asarray(idx):
        assert len(set(row.tolist())) == 64
    idx3 = _sample_batch_idx(keys, n_windows=10, windows_batch_size=64)
    assert idx3.shape == (7, 64) and int(idx3.min()) >= 0 and int(idx3.max()) < 10


def test_predict_step_shape_idempotent():
    net = _net(); y = _y()
    train(net, y, h=4, input_size=8, max_steps=5, windows_batch_size=32, lr=1e-3,
          seed=0, scaler=IdentityScaler())
    p1 = predict_step(net, y[-8:], h=4, input_size=8, scaler=IdentityScaler())
    p2 = predict_step(net, y[-8:], h=4, input_size=8, scaler=IdentityScaler())
    assert p1.shape == (4,)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-6)

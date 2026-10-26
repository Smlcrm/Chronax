"""Tests for chronax.models.dlinear training infra + module."""
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.dlinear.dlinear_losses import mae as loss_mae, resolve as resolve_loss
from chronax.models.dlinear.dlinear_scaler import IdentityScaler, resolve_scaler


def test_masked_mae_drops_masked_elements():
    pred = jnp.array([[1.0, 2.0]]); target = jnp.array([[0.0, 0.0]])
    mask = jnp.array([[1.0, 0.0]])
    assert float(loss_mae(pred, target, mask)) == 1.0


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


from chronax.models.dlinear.dlinear_module import DLinearNet, _series_decomp


def test_series_decomp_hand_computed_k3():
    y = jnp.array([[1.0, 2.0, 4.0, 8.0]])
    trend, seasonal = _series_decomp(y, 3)
    # edge-pad 1 both sides: [1,1,2,4,8,8]; 3-means: [4/3, 7/3, 14/3, 20/3]
    np.testing.assert_allclose(np.asarray(trend[0]), [4/3, 7/3, 14/3, 20/3], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(trend + seasonal), np.asarray(y), rtol=1e-6)


def test_series_decomp_k1_identity():
    y = jnp.array([[3.0, -1.0, 5.0]])
    trend, seasonal = _series_decomp(y, 1)
    np.testing.assert_allclose(np.asarray(trend), np.asarray(y), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(seasonal), np.zeros((1, 3)), atol=1e-6)


def test_series_decomp_kernel_larger_than_length():
    y = jnp.ones((2, 12))
    trend, seasonal = _series_decomp(y, 25)      # pad 12 each side -> length 36 -> out 12
    assert trend.shape == (2, 12)
    np.testing.assert_allclose(np.asarray(trend), np.ones((2, 12)), rtol=1e-6)


def test_series_decomp_kernel_larger_than_length_hand_values():
    # non-constant series pins edge-padding VALUES on the k>L path (the default
    # regime whenever 3*h < 25), not just shapes
    y = np.array([[1.0, 2.0, 4.0, 8.0]], dtype=np.float32)
    trend, _ = _series_decomp(jnp.asarray(y), 25)
    pad = 12
    padded = np.concatenate([np.full(pad, 1.0), y[0], np.full(pad, 8.0)])
    expected = np.array([padded[i:i + 25].mean() for i in range(4)])
    np.testing.assert_allclose(np.asarray(trend[0]), expected, rtol=1e-6)


def test_forward_matches_hand_computed_and_detects_swap():
    net = DLinearNet(h=2, input_size=4, moving_avg_window=3, rngs=nnx.Rngs(0))
    # Wt != Ws (swap-detecting: biases cancel under a swap, weights must differ)
    Wt = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    Ws = jnp.array([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
    bt = jnp.array([0.1, -0.1]); bs = jnp.array([0.2, 0.3])
    net.w_trend.value = Wt; net.b_trend.value = bt
    net.w_season.value = Ws; net.b_season.value = bs
    y = jnp.array([[1.0, 2.0, 4.0, 8.0]])       # nonzero seasonal under k=3
    trend, seasonal = _series_decomp(y, 3)
    expected = trend @ Wt.T + bt + seasonal @ Ws.T + bs
    out = net(y[..., None])
    np.testing.assert_allclose(np.asarray(out[..., 0]), np.asarray(expected), rtol=1e-6)
    swapped = trend @ Ws.T + bs + seasonal @ Wt.T + bt
    assert not np.allclose(np.asarray(expected), np.asarray(swapped))


def test_init_matches_torch_bounds_both_layers():
    net = DLinearNet(h=24, input_size=72, moving_avg_window=25, rngs=nnx.Rngs(0))
    bound = 1.0 / np.sqrt(72)
    for p, shape in ((net.w_trend, (24, 72)), (net.b_trend, (24,)),
                     (net.w_season, (24, 72)), (net.b_season, (24,))):
        a = np.asarray(p.value)
        assert a.shape == shape and np.abs(a).max() <= bound
    assert np.asarray(net.w_trend.value).std() > 0.2 * bound
    assert not np.allclose(np.asarray(net.w_trend.value), np.asarray(net.w_season.value))
    assert np.abs(np.asarray(net.b_trend.value)).max() > 0.0


from chronax.models.dlinear.dlinear_training import build_windows, predict_step, scaled_forward_loss, train


def _y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _net(h=12, input_size=36):
    return DLinearNet(h=h, input_size=input_size, moving_avg_window=25, rngs=nnx.Rngs(0))


def test_build_windows_shape_and_padding():
    y = _y(60)
    w, m = build_windows(y, input_size=36, h=12)
    assert w.shape == (24, 48) and m.shape == (24, 48)
    assert float(m[:, :36].min()) == 1.0
    np.testing.assert_array_equal(np.asarray(m[-1, 36:]), [1.0] + [0.0] * 11)
    np.testing.assert_array_equal(np.asarray(w[-1, 37:]), np.zeros(11))
    assert float(w[-1, 36]) == float(y[-1])
    assert float(m.sum()) == 1086.0


def test_scaled_forward_loss_scalar():
    w, m = build_windows(_y(), input_size=36, h=12)
    loss = scaled_forward_loss(_net(), w[:8], m[:8], h=12, input_size=36, scaler=IdentityScaler())
    assert loss.shape == () and jnp.isfinite(loss)


def test_train_decreases_loss_and_is_deterministic():
    l1 = train(_net(), _y(), h=12, input_size=36, max_steps=30, windows_batch_size=64,
               lr=1e-2, seed=0, scaler=IdentityScaler())
    l2 = train(_net(), _y(), h=12, input_size=36, max_steps=30, windows_batch_size=64,
               lr=1e-2, seed=0, scaler=IdentityScaler())
    assert l1.shape == (30,) and float(l1[-1]) < float(l1[0])
    np.testing.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=1e-5)


def test_train_oversample_small_n():
    losses = train(_net(), _y(60), h=12, input_size=36, max_steps=8, windows_batch_size=64,
                   lr=1e-3, seed=0, scaler=IdentityScaler())
    assert losses.shape == (8,) and jnp.all(jnp.isfinite(losses))


def test_train_raises_on_nonfinite_loss():
    y = _y().at[50].set(jnp.nan)
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        train(_net(), y, h=12, input_size=36, max_steps=5, windows_batch_size=64,
              lr=1e-4, seed=0, scaler=IdentityScaler())


def test_sample_batch_idx_regimes():
    import jax
    from chronax.models.dlinear.dlinear_training import _sample_batch_idx
    keys = jax.random.split(jax.random.PRNGKey(0), 7)
    idx = _sample_batch_idx(keys, n_windows=500, windows_batch_size=64)
    assert idx.shape == (7, 64)
    assert int(idx.min()) >= 0 and int(idx.max()) < 500
    for row in np.asarray(idx):
        assert len(set(row.tolist())) == 64
    idx2 = _sample_batch_idx(keys, n_windows=500, windows_batch_size=64)
    np.testing.assert_array_equal(np.asarray(idx), np.asarray(idx2))
    idx3 = _sample_batch_idx(keys, n_windows=10, windows_batch_size=64)
    assert idx3.shape == (7, 64)
    assert int(idx3.min()) >= 0 and int(idx3.max()) < 10


def test_sample_batch_idx_traced_matches_host():
    import jax
    from functools import partial
    from chronax.models.dlinear.dlinear_training import _sample_batch_idx
    keys = jax.random.split(jax.random.PRNGKey(3), 5)
    host = _sample_batch_idx(keys, 500, 64)
    traced = jax.jit(partial(_sample_batch_idx, n_windows=500, windows_batch_size=64))(keys)
    for hr, tr in zip(np.asarray(host), np.asarray(traced)):
        assert set(hr.tolist()) == set(tr.tolist())


def test_predict_step_shape_idempotent():
    net = _net(); y = _y()
    train(net, y, h=12, input_size=36, max_steps=5, windows_batch_size=64, lr=1e-3,
          seed=0, scaler=IdentityScaler())
    p1 = predict_step(net, y[-36:], h=12, input_size=36, scaler=IdentityScaler())
    p2 = predict_step(net, y[-36:], h=12, input_size=36, scaler=IdentityScaler())
    assert p1.shape == (12,)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-6)


def test_predict_step_shift_equivariant_under_robust():
    # mutation-probe gap: identity ignores shift/scale args, so the inverse-scaling
    # call-site is invisible under it. Under robust, shifting the context by c
    # shifts the median by c (MAD unchanged) => prediction must shift by exactly c.
    from chronax.models.dlinear.dlinear_scaler import RobustScaler
    net = _net(); y = _y()
    train(net, y, h=12, input_size=36, max_steps=5, windows_batch_size=64, lr=1e-3,
          seed=0, scaler=RobustScaler())
    ctx = y[-36:]
    base = np.asarray(predict_step(net, ctx, h=12, input_size=36, scaler=RobustScaler()))
    shifted = np.asarray(predict_step(net, ctx + 50.0, h=12, input_size=36, scaler=RobustScaler()))
    np.testing.assert_allclose(shifted, base + 50.0, rtol=1e-4, atol=1e-3)

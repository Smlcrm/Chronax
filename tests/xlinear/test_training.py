"""Tests for chronax.models.xlinear training infra + module."""
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.xlinear.xlinear_losses import mae as loss_mae, resolve as resolve_loss
from chronax.models.xlinear.xlinear_scaler import IdentityScaler, resolve_scaler


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


from chronax.models.xlinear.xlinear_module import XLinearNet, _gating, _revin


def test_revin_hand_computed():
    y = jnp.array([[1.0, 2.0, 3.0, 6.0]])
    yn, mean, stdev = _revin(y)
    m = 3.0
    var = np.mean((np.array([1.0, 2.0, 3.0, 6.0]) - m) ** 2)   # biased
    sd = np.sqrt(var + 1e-5)
    np.testing.assert_allclose(float(mean[0, 0]), m, rtol=1e-6)
    np.testing.assert_allclose(float(stdev[0, 0]), sd, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(yn[0]), (np.array([1.0, 2.0, 3.0, 6.0]) - m) / sd, rtol=1e-6)


def test_revin_constant_window_finite():
    yn, mean, stdev = _revin(jnp.full((2, 8), 5.0))
    np.testing.assert_allclose(float(stdev[0, 0]), np.sqrt(1e-5), rtol=1e-5)
    assert bool(jnp.all(jnp.isfinite(yn)))


def test_gating_hand_computed():
    x = jnp.array([[1.0, -2.0]])
    w1 = jnp.array([[0.5, 0.0], [0.0, 1.0], [1.0, 1.0]])   # [3, 2]
    b1 = jnp.array([0.1, 0.2, 0.0])
    w2 = jnp.array([[1.0, 0.0, 2.0], [0.0, -1.0, 0.0]])    # [2, 3]
    b2 = jnp.array([0.0, 0.3])
    hidden = np.maximum(np.array([1.0, -2.0]) @ np.asarray(w1).T + np.asarray(b1), 0.0)
    gate = 1.0 / (1.0 + np.exp(-(hidden @ np.asarray(w2).T + np.asarray(b2))))
    expected = np.array([1.0, -2.0]) * gate
    out = _gating(x, w1, b1, w2, b2)
    np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-6)


def _hand_forward(y, net, use_norm=True):
    """Numpy reimplementation of the spec steps for the hand-check test."""
    y = np.asarray(y, dtype=np.float32)                    # [B, L]
    if use_norm:
        m = y.mean(axis=1, keepdims=True)
        sd = np.sqrt(((y - m) ** 2).mean(axis=1, keepdims=True) + 1e-5)
        yn = (y - m) / sd
    else:
        yn = y
    P = lambda a: np.asarray(a)
    emb = yn @ P(net.w_proj.value).T + P(net.b_proj.value)             # [B, H]
    glob = np.broadcast_to(P(net.glob_token.value)[0, 0], emb.shape)   # [B, H]
    en_emb = np.concatenate([emb, glob], axis=-1)                      # [B, 2H]
    def gate(x, w1, b1, w2, b2):
        h = np.maximum(x @ P(w1).T + P(b1), 0.0)
        return x * (1.0 / (1.0 + np.exp(-(h @ P(w2).T + P(b2)))))
    en_atten = gate(en_emb, net.w_tg1.value, net.b_tg1.value, net.w_tg2.value, net.b_tg2.value)
    H = emb.shape[-1]
    origin_atten, glob_atten = en_atten[:, :H], en_atten[:, H:]
    # cross-channel: stack [emb, glob_atten] as 2 "channels" -> gate over channel dim
    ex = np.stack([emb, glob_atten], axis=1)                           # [B, 2, H]
    exT = np.transpose(ex, (0, 2, 1))                                  # [B, H, 2]
    ex_atten = gate(exT, net.w_cg1.value, net.b_cg1.value, net.w_cg2.value, net.b_cg2.value)
    glob2 = ex_atten[:, :, 1]                                          # [B, H] (channels[n_series:] at N=1)
    en = np.concatenate([origin_atten, glob2], axis=-1)                # [B, 2H]
    out = en @ P(net.w_head.value).T + P(net.b_head.value)             # [B, h]
    if use_norm:
        out = out * sd + m
    return out


def test_forward_matches_hand_computed():
    net = XLinearNet(h=2, input_size=4, hidden_size=3, temporal_ff=5, channel_ff=4,
                     use_norm=True, rngs=nnx.Rngs(7))
    y = jnp.array([[1.0, 2.0, 4.0, 8.0], [0.5, -1.0, 3.0, 2.0]])
    out = net(y[..., None])
    np.testing.assert_allclose(np.asarray(out[..., 0]), _hand_forward(y, net), rtol=1e-5, atol=1e-6)


def test_forward_use_norm_false_differs_and_finite():
    kw = dict(h=2, input_size=4, hidden_size=3, temporal_ff=5, channel_ff=4)
    y = jnp.array([[10.0, 12.0, 14.0, 18.0]])
    a = XLinearNet(use_norm=True, rngs=nnx.Rngs(7), **kw)(y[..., None])
    b = XLinearNet(use_norm=False, rngs=nnx.Rngs(7), **kw)(y[..., None])
    assert bool(jnp.all(jnp.isfinite(a))) and bool(jnp.all(jnp.isfinite(b)))
    assert not np.allclose(np.asarray(a), np.asarray(b))


def test_init_matches_torch_bounds_per_layer():
    # temporal_ff=200 != 2H=256 so a tg1/tg2 fan_in transposition is detectable;
    # per-layer LOWER bounds catch a too-large fan_in (tighter-than-torch uniform).
    H = 128
    net = XLinearNet(h=24, input_size=72, hidden_size=H, temporal_ff=200, channel_ff=8,
                     use_norm=True, rngs=nnx.Rngs(0))
    checks = [
        (net.w_proj, (H, 72), 72, 0.9), (net.b_proj, (H,), 72, 0.9),
        (net.w_tg1, (200, 2 * H), 2 * H, 0.9), (net.b_tg1, (200,), 2 * H, 0.9),
        (net.w_tg2, (2 * H, 200), 200, 0.9), (net.b_tg2, (2 * H,), 200, 0.9),
        (net.w_cg1, (8, 2), 2, 0.5), (net.b_cg1, (8,), 2, 0.5),
        (net.w_cg2, (2, 8), 8, 0.5), (net.b_cg2, (2,), 8, 0.5),
        (net.w_head, (24, 2 * H), 2 * H, 0.9), (net.b_head, (24,), 2 * H, 0.9),
    ]
    for p, shape, fan_in, lo in checks:
        a = np.asarray(p.value)
        bound = 1.0 / np.sqrt(fan_in)
        assert a.shape == shape, (a.shape, shape)
        assert np.abs(a).max() <= bound + 1e-7
        assert np.abs(a).max() > lo * bound        # detects a tighter (wrong-fan_in) draw
    g = np.asarray(net.glob_token.value)
    assert g.shape == (1, 1, H)
    assert 0.7 < g.std() < 1.3          # N(0,1); SE of sample std ~0.06 at H=128


from chronax.models.xlinear.xlinear_training import build_windows, predict_step, scaled_forward_loss, train


def _y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _net(h=12, input_size=36):
    return XLinearNet(h=h, input_size=input_size, hidden_size=16, temporal_ff=8,
                      channel_ff=4, use_norm=True, rngs=nnx.Rngs(0))


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
    l1 = train(_net(), _y(), h=12, input_size=36, max_steps=30, windows_batch_size=32,
               lr=1e-2, seed=0, scaler=IdentityScaler())
    l2 = train(_net(), _y(), h=12, input_size=36, max_steps=30, windows_batch_size=32,
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
        train(_net(), y, h=12, input_size=36, max_steps=5, windows_batch_size=32,
              lr=1e-4, seed=0, scaler=IdentityScaler())


def test_sample_batch_idx_regimes():
    import jax
    from chronax.models.xlinear.xlinear_training import _sample_batch_idx
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
    from chronax.models.xlinear.xlinear_training import _sample_batch_idx
    keys = jax.random.split(jax.random.PRNGKey(3), 5)
    host = _sample_batch_idx(keys, 500, 64)
    traced = jax.jit(partial(_sample_batch_idx, n_windows=500, windows_batch_size=64))(keys)
    for hr, tr in zip(np.asarray(host), np.asarray(traced)):
        assert set(hr.tolist()) == set(tr.tolist())


def test_predict_step_shape_idempotent():
    net = _net(); y = _y()
    train(net, y, h=12, input_size=36, max_steps=5, windows_batch_size=32, lr=1e-3,
          seed=0, scaler=IdentityScaler())
    p1 = predict_step(net, y[-36:], h=12, input_size=36, scaler=IdentityScaler())
    p2 = predict_step(net, y[-36:], h=12, input_size=36, scaler=IdentityScaler())
    assert p1.shape == (12,)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-6)


def test_predict_step_shift_equivariant_under_robust():
    # identity ignores shift/scale args, so the inverse-scaling call-site is
    # invisible under it; robust makes it load-bearing. RevIN (use_norm) is also
    # shift-equivariant, so +c must propagate exactly end-to-end.
    from chronax.models.xlinear.xlinear_scaler import RobustScaler
    net = _net(); y = _y()
    train(net, y, h=12, input_size=36, max_steps=5, windows_batch_size=32, lr=1e-3,
          seed=0, scaler=RobustScaler())
    ctx = y[-36:]
    base = np.asarray(predict_step(net, ctx, h=12, input_size=36, scaler=RobustScaler()))
    shifted = np.asarray(predict_step(net, ctx + 50.0, h=12, input_size=36, scaler=RobustScaler()))
    np.testing.assert_allclose(shifted, base + 50.0, rtol=1e-3, atol=1e-2)

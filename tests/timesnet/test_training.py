"""Tests for chronax.models.timesnet training infra + module."""
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.timesnet.timesnet_losses import mae as loss_mae, resolve as resolve_loss
from chronax.models.timesnet.timesnet_scaler import IdentityScaler, resolve_scaler


def test_masked_mae_drops_masked_elements():
    pred = jnp.array([[1.0, 2.0]]); target = jnp.array([[0.0, 0.0]])
    mask = jnp.array([[1.0, 0.0]])
    assert float(loss_mae(pred, target, mask)) == 1.0     # only the unmasked |1-0|


def test_resolve_loss_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        resolve_loss("nope")


def test_scaler_resolve_and_roundtrip():
    x = jnp.array([[1.0, 2.0, 3.0, 100.0]])
    for name in ("identity", "robust", "standard"):
        s = resolve_scaler(name)
        shift, scale = s.stats(x, axis=1)
        z = s.transform(x, shift, scale)
        np.testing.assert_allclose(np.asarray(s.inverse(z, shift, scale)), np.asarray(x), rtol=1e-5)
    with pytest.raises(ValueError, match="Unknown scaler"):
        resolve_scaler("nope")


def test_standard_scaler_hand_computed():
    from chronax.models.timesnet.timesnet_scaler import StandardScaler
    x = jnp.array([[1.0, 2.0, 3.0, 6.0]])
    s = StandardScaler()
    shift, scale = s.stats(x, axis=1)
    np.testing.assert_allclose(float(shift[0, 0]), 3.0, rtol=1e-6)
    np.testing.assert_allclose(float(scale[0, 0]), np.std([1.0, 2.0, 3.0, 6.0]) + 1e-6, rtol=1e-6)
    z = s.transform(x, shift, scale)
    np.testing.assert_allclose(np.asarray(s.inverse(z, shift, scale)), np.asarray(x), rtol=1e-5)


def test_standard_scaler_constant_window():
    from chronax.models.timesnet.timesnet_scaler import StandardScaler
    shift, scale = StandardScaler().stats(jnp.full((1, 8), 5.0), axis=1)
    np.testing.assert_allclose(float(scale[0, 0]), 1.0 + 1e-6, rtol=1e-6)  # std==0 -> 1.0 THEN +eps


def test_resolve_scaler_standard():
    from chronax.models.timesnet.timesnet_scaler import StandardScaler, resolve_scaler
    assert isinstance(resolve_scaler("standard"), StandardScaler)


import jax

from chronax.models.timesnet.timesnet_module import (
    TimesNetNet, _inception, _positional_encoding, _token_embed)


def test_positional_encoding_hand_formula():
    pe = np.asarray(_positional_encoding(10, 8))[0]          # [10, 8]
    pos = 3
    div = np.exp(np.arange(0, 8, 2) * -(np.log(10000.0) / 8))
    np.testing.assert_allclose(pe[pos, 0::2], np.sin(pos * div), rtol=1e-5)
    np.testing.assert_allclose(pe[pos, 1::2], np.cos(pos * div), rtol=1e-5)


def test_token_embed_circular_hand_check():
    y = jnp.array([[1.0, 2.0, 3.0, 4.0]])                    # [1, 4]
    w = jnp.zeros((2, 1, 3)).at[0, 0].set(jnp.array([1.0, 0.0, 0.0])) \
                             .at[1, 0].set(jnp.array([0.0, 0.0, 1.0]))
    out = _token_embed(y, w)                                 # [1, 4, 2]
    # circular: position t sees [y[t-1], y[t], y[t+1]] with wrap
    np.testing.assert_allclose(np.asarray(out[0, :, 0]), [4.0, 1.0, 2.0, 3.0], rtol=1e-6)  # left neighbor
    np.testing.assert_allclose(np.asarray(out[0, :, 1]), [2.0, 3.0, 4.0, 1.0], rtol=1e-6)  # right neighbor


def test_inception_mean_of_same_convs():
    x = jnp.ones((1, 1, 3, 3))
    ws = [jnp.ones((1, 1, 1, 1)), jnp.ones((1, 1, 3, 3))]    # k=1 and k=3
    bs = [jnp.zeros(1), jnp.zeros(1)]
    out = np.asarray(_inception(x, ws, bs))[0, 0]
    k3 = np.array([[4., 6., 4.], [6., 9., 6.], [4., 6., 4.]])  # SAME conv of ones
    np.testing.assert_allclose(out, (np.ones((3, 3)) + k3) / 2.0, rtol=1e-6)


def _net(h=4, input_size=8, **kw):
    kw.setdefault("hidden_size", 8); kw.setdefault("conv_hidden_size", 8)
    kw.setdefault("top_k", 2); kw.setdefault("num_kernels", 2)
    kw.setdefault("encoder_layers", 2); kw.setdefault("dropout", 0.1)
    kw.setdefault("periods", (4, 3)); kw.setdefault("freqs", (3, 4))
    return TimesNetNet(h=h, input_size=input_size, rngs=nnx.Rngs(0), **kw)


def test_forward_shape_finite_and_deterministic_without_key():
    net = _net()
    x = jnp.asarray(np.random.RandomState(0).randn(3, 8, 1), dtype=jnp.float32)
    a, b = net(x), net(x)
    assert a.shape == (3, 4, 1) and bool(jnp.all(jnp.isfinite(a)))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)


def test_forward_dropout_key_changes_output_and_none_is_eval():
    net = _net()
    x = jnp.asarray(np.random.RandomState(0).randn(2, 8, 1), dtype=jnp.float32)
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    o1, o2 = net(x, dropout_key=k1), net(x, dropout_key=k2)
    assert not np.allclose(np.asarray(o1), np.asarray(o2))
    np.testing.assert_allclose(np.asarray(net(x)), np.asarray(net(x)), rtol=1e-6)


def test_shared_layernorm_single_pair():
    net = _net(encoder_layers=2)
    # exactly ONE ln_scale/ln_bias pair; mutating it changes output
    x = jnp.asarray(np.random.RandomState(1).randn(2, 8, 1), dtype=jnp.float32)
    base = np.asarray(net(x))
    net.ln_scale.value = net.ln_scale.value * 2.0
    assert not np.allclose(base, np.asarray(net(x)))
    assert not hasattr(net, "ln_scale_2")


def test_period_grid_roundtrip_nondividing():
    # T=input_size+h=12 w/ period 5 -> rows=3, pad 3 zeros
    net = _net(h=4, input_size=8, periods=(5,), freqs=(2,), top_k=1)
    x = jnp.asarray(np.random.RandomState(2).randn(1, 8, 1), dtype=jnp.float32)
    out = net(x)   # exercises pad->reshape->conv->truncate on a non-dividing period
    assert out.shape == (1, 4, 1) and bool(jnp.all(jnp.isfinite(out)))


from chronax.models.timesnet.timesnet_training import (
    build_windows, predict_step, scaled_forward_loss, train)


def _y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def test_build_windows_shape_and_padding():
    y = _y(60)
    w, m = build_windows(y, input_size=36, h=12)          # NF: n_windows = 60-36
    assert w.shape == (24, 48) and m.shape == (24, 48)
    assert float(m[:, :36].min()) == 1.0                  # insample always real
    # Exact padding contract (guards off-by-one leaking padded zeros into the
    # loss — the failure class behind KAN's original airline accuracy gap):
    # last window covers y[23:59] + [y[59], 11 zero-pads].
    np.testing.assert_array_equal(np.asarray(m[-1, 36:]), [1.0] + [0.0] * 11)
    np.testing.assert_array_equal(np.asarray(w[-1, 37:]), np.zeros(11))
    assert float(w[-1, 36]) == float(y[-1])
    # closed-form real-point count: sum_i min(48, 60-i) for i in 0..23
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
    # same seed => identical sampling AND dropout streams
    np.testing.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=1e-5)


def test_train_oversample_small_n():
    losses = train(_net(), _y(20), h=4, input_size=8, max_steps=8, windows_batch_size=64,
                   lr=1e-3, seed=0, scaler=IdentityScaler())
    assert losses.shape == (8,) and jnp.all(jnp.isfinite(losses))


def test_train_raises_on_nonfinite_loss():
    y = _y().at[50].set(jnp.nan)                          # NaN input -> NaN loss at step 0
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        train(_net(), y, h=4, input_size=8, max_steps=5, windows_batch_size=32,
              lr=1e-4, seed=0, scaler=IdentityScaler())


def test_sample_batch_idx_regimes():
    from chronax.models.timesnet.timesnet_training import _sample_batch_idx
    keys = jax.random.split(jax.random.PRNGKey(0), 7)
    # large-n regime: uniform k-subset — distinct, in-range, deterministic
    idx = _sample_batch_idx(keys, n_windows=500, windows_batch_size=64)
    assert idx.shape == (7, 64)
    assert int(idx.min()) >= 0 and int(idx.max()) < 500
    for row in np.asarray(idx):
        assert len(set(row.tolist())) == 64          # no replacement in this regime
    idx2 = _sample_batch_idx(keys, n_windows=500, windows_batch_size=64)
    np.testing.assert_array_equal(np.asarray(idx), np.asarray(idx2))
    # small-n regime: with replacement, in-range
    idx3 = _sample_batch_idx(keys, n_windows=10, windows_batch_size=64)
    assert idx3.shape == (7, 64)
    assert int(idx3.min()) >= 0 and int(idx3.max()) < 10


def test_sample_batch_idx_traced_matches_host():
    from functools import partial
    from chronax.models.timesnet.timesnet_training import _sample_batch_idx
    keys = jax.random.split(jax.random.PRNGKey(3), 5)
    host = _sample_batch_idx(keys, 500, 64)
    traced = jax.jit(partial(_sample_batch_idx, n_windows=500, windows_batch_size=64))(keys)
    for hr, tr in zip(np.asarray(host), np.asarray(traced)):
        assert set(hr.tolist()) == set(tr.tolist())      # same uniform k-subset either path


def test_predict_step_shape_idempotent():
    net = _net(); y = _y()
    train(net, y, h=4, input_size=8, max_steps=5, windows_batch_size=32, lr=1e-3,
          seed=0, scaler=IdentityScaler())
    p1 = predict_step(net, y[-8:], h=4, input_size=8, scaler=IdentityScaler())
    p2 = predict_step(net, y[-8:], h=4, input_size=8, scaler=IdentityScaler())
    assert p1.shape == (4,)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-6)


def test_predict_step_shift_equivariant_standard_scaler():
    # Under StandardScaler the scaled insample is invariant to y -> y + c, so
    # predict_step must shift by exactly c (training-layer property; the MODEL
    # is not shift-equivariant because _compute_periods sees zero-padded tails).
    from chronax.models.timesnet.timesnet_scaler import StandardScaler
    net = _net(); y = _y()
    p = predict_step(net, y[-8:], h=4, input_size=8, scaler=StandardScaler())
    p_shift = predict_step(net, y[-8:] + 100.0, h=4, input_size=8, scaler=StandardScaler())
    np.testing.assert_allclose(np.asarray(p_shift), np.asarray(p) + 100.0, rtol=1e-4)


def test_net_rejects_mismatched_periods_freqs():
    with pytest.raises(ValueError, match="parallel tuples"):
        TimesNetNet(h=4, input_size=8, hidden_size=8, conv_hidden_size=8, top_k=2,
                    num_kernels=2, encoder_layers=1, dropout=0.1, periods=(4,), freqs=(3,),
                    rngs=nnx.Rngs(0))


def test_net_rejects_out_of_range_freq():
    # T=12 -> nonzero rfft bins are [1, 6]; freq 99 would silently clamp the gather
    with pytest.raises(ValueError, match="every freq"):
        TimesNetNet(h=4, input_size=8, hidden_size=8, conv_hidden_size=8, top_k=2,
                    num_kernels=2, encoder_layers=1, dropout=0.1, periods=(4, 3), freqs=(3, 99),
                    rngs=nnx.Rngs(0))


def test_single_shared_layernorm_leaf_pair():
    # Lock the shared-LayerNorm design: exactly one (scale, bias) pair regardless
    # of encoder_layers — a per-layer-list implementation would show 2*L leaves.
    net = _net(encoder_layers=3)
    _, state = nnx.split(net)
    ln_leaves = [tuple(p) for p, _ in nnx.to_flat_state(state) if any("ln_" in str(x) for x in p)]
    assert len(ln_leaves) == 2, ln_leaves

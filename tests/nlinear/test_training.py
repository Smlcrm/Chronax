"""Tests for chronax.models.nlinear training infra + module."""
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.nlinear.nlinear_losses import mae as loss_mae, resolve as resolve_loss
from chronax.models.nlinear.nlinear_scaler import IdentityScaler, resolve_scaler


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


from chronax.models.nlinear.nlinear_module import NLinearNet


def test_forward_matches_hand_computed():
    net = NLinearNet(h=2, input_size=3, rngs=nnx.Rngs(0))
    W = jnp.array([[1.0, 0.0, 2.0], [0.5, -1.0, 0.0]])   # [h=2, in=3]
    b = jnp.array([0.1, -0.2])
    net.weight.value = W; net.bias.value = b
    y = jnp.array([[1.0, 2.0, 4.0]])                      # last = 4
    out = net(y[..., None])                               # [1, 2, 1]
    norm = y - 4.0                                        # [-3, -2, 0]
    expected = norm @ W.T + b + 4.0
    np.testing.assert_allclose(np.asarray(out[..., 0]), np.asarray(expected), rtol=1e-6)


def test_init_matches_torch_bounds():
    net = NLinearNet(h=24, input_size=72, rngs=nnx.Rngs(0))
    bound = 1.0 / np.sqrt(72)
    w = np.asarray(net.weight.value); b = np.asarray(net.bias.value)
    assert w.shape == (24, 72) and b.shape == (24,)
    assert np.abs(w).max() <= bound and np.abs(b).max() <= bound
    assert w.std() > 0.2 * bound          # non-degenerate uniform, not zeros/normal
    assert np.abs(b).max() > 0.0          # torch uses uniform bias, NOT zeros


from chronax.models.nlinear.nlinear_training import build_windows, predict_step, scaled_forward_loss, train


def _y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _net(h=12, input_size=36):
    return NLinearNet(h=h, input_size=input_size, rngs=nnx.Rngs(0))


def test_build_windows_shape_and_padding():
    w, m = build_windows(_y(60), input_size=36, h=12)     # NF: n_windows = 60-36
    assert w.shape == (24, 48) and m.shape == (24, 48)
    assert float(m[:, :36].min()) == 1.0                  # insample always real
    assert float(m[-1, 36:].min()) == 0.0                 # last horizon tail padded


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
    y = _y().at[50].set(jnp.nan)                          # NaN input -> NaN loss at step 0
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        train(_net(), y, h=12, input_size=36, max_steps=5, windows_batch_size=64,
              lr=1e-4, seed=0, scaler=IdentityScaler())


def test_sample_batch_idx_regimes():
    import jax
    from chronax.models.nlinear.nlinear_training import _sample_batch_idx
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


def test_predict_step_shape_idempotent():
    net = _net(); y = _y()
    train(net, y, h=12, input_size=36, max_steps=5, windows_batch_size=64, lr=1e-3,
          seed=0, scaler=IdentityScaler())
    p1 = predict_step(net, y[-36:], h=12, input_size=36, scaler=IdentityScaler())
    p2 = predict_step(net, y[-36:], h=12, input_size=36, scaler=IdentityScaler())
    assert p1.shape == (12,)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-6)

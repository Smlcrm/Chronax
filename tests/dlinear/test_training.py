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

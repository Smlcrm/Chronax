"""Tests for chronax.models.kan.kan_module."""
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.kan.kan_module import KANLinear, build_grid, b_splines


def test_build_grid_shape_and_values():
    grid = build_grid(in_features=3, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0))
    assert grid.shape == (3, 5 + 2 * 3 + 1)        # [in, 12]
    expected = np.arange(-3, 9) * (2.0 / 5) - 1.0
    np.testing.assert_allclose(np.asarray(grid[0]), expected, rtol=1e-6)


def test_b_splines_shape_and_outside_grid_zero():
    grid = build_grid(2, 5, 3, (-1.0, 1.0))
    x = jnp.zeros((4, 2), dtype=jnp.float32)        # inside grid
    bs = b_splines(x, grid, spline_order=3)
    assert bs.shape == (4, 2, 5 + 3)                # [B, in, grid_size+spline_order]
    far = jnp.full((4, 2), 1000.0, dtype=jnp.float32)  # outside grid
    assert float(jnp.max(jnp.abs(b_splines(far, grid, 3)))) == 0.0


def test_b_splines_partition_of_unity_in_grid():
    grid = build_grid(1, 8, 3, (-1.0, 1.0))
    x = jnp.linspace(-0.9, 0.9, 16, dtype=jnp.float32)[:, None]   # [16, 1]
    s = jnp.sum(b_splines(x, grid, 3), axis=-1)                   # B-splines sum to 1 in-grid
    np.testing.assert_allclose(np.asarray(s[:, 0]), np.ones(16), rtol=1e-4, atol=1e-4)


def test_kanlinear_forward_shape_and_finite():
    layer = KANLinear(in_features=6, out_features=4, grid_size=5, spline_order=3,
                      scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                      enable_standalone_scale_spline=True, grid_range=(-1.0, 1.0), rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(8, 6), dtype=jnp.float32)
    out = layer(x)
    assert out.shape == (8, 4) and jnp.all(jnp.isfinite(out))
    assert layer.base_weight.value.shape == (4, 6)          # torch layout [out, in]
    assert layer.spline_weight.value.shape == (4, 6, 8)
    assert layer.spline_scaler.value.shape == (4, 6)


def test_kanlinear_grid_not_trainable():
    layer = KANLinear(3, 2, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                      scale_spline=1.0, enable_standalone_scale_spline=True, grid_range=(-1.0, 1.0), rngs=nnx.Rngs(0))
    _, state = nnx.split(layer)
    assert not isinstance(layer.grid, nnx.Param)


from chronax.models.kan.kan_module import KANNet


def test_kannet_forward_shape():
    net = KANNet(h=12, input_size=36, n_hidden_layers=1, hidden_size=16, grid_size=5,
                 spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True, grid_range=(-1.0, 1.0), rngs=nnx.Rngs(0))
    x = jnp.ones((4, 36, 1), dtype=jnp.float32)
    out = net(x)
    assert out.shape == (4, 12, 1) and out.dtype == jnp.float32
    assert len(net.layers) == 2          # [36->16, 16->12]

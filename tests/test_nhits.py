"""Tests for the NHITS forecaster (port of neuralforecast.NHITS).

Sections: Losses (drift vs the mlp canon copy) -> Module (interpolation
matrices vs torch goldens, pooling semantics, NumPy-reference forward,
param-count pin, activations, validation) -> Training (LR schedule, vmap
traceability, divergence guard) -> Model (BaseForecaster conformance) ->
Multivariate -> Namespace.

Interpolation/pooling goldens were generated with torch (neuralforecast's
engine) by probing ``F.interpolate`` / ``nn.{Max,Avg}Pool1d`` with identity
basis vectors; the full-net transplant gate lives in
``benchmarks/nhits_weight_parity.py``.
"""
from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.nhits.nhits_losses import GMM, MultiQuantileLoss, mae
from chronax.models.nhits.nhits_module import (
    ACTIVATIONS, NHITSBlock, NHITSNet, _interp_weights, _pool1d,
)
from chronax.models.nhits.nhits_scaler import RobustScaler
from chronax.models.nhits.nhits_training import _lr_schedule, train
from chronax.models.nhits.nhits_model import NHITS
from chronax.utils import ConformalIntervals


# =============================================================================
# Losses (verbatim copy of the mlp losses module — pin against drift)
# =============================================================================

class TestLossesDriftVsMLP:
    def test_gmm_matches_mlp_copy(self):
        from chronax.models.mlp.mlp_losses import GMM as MGMM
        rng = np.random.RandomState(0)
        raw = jnp.asarray(rng.randn(3, 4, 4), jnp.float32)     # 2*K with K=2, weighted=False
        y = jnp.asarray(rng.randn(3, 4), jnp.float32)
        mask = jnp.asarray((rng.rand(3, 4) > 0.3).astype(np.float32))
        g_n, g_m = GMM(n_components=2), MGMM(n_components=2)
        a_n = g_n.scale_decouple(g_n.domain_map(raw), loc=jnp.zeros((3, 1, 1)),
                                 scale=jnp.full((3, 1, 1), 2.0))
        a_m = g_m.scale_decouple(g_m.domain_map(raw), loc=jnp.zeros((3, 1, 1)),
                                 scale=jnp.full((3, 1, 1), 2.0))
        for x_n, x_m in zip(a_n, a_m):
            np.testing.assert_array_equal(np.asarray(x_n), np.asarray(x_m))
        np.testing.assert_array_equal(np.asarray(g_n(y, a_n, mask=mask)),
                                      np.asarray(g_m(y, a_m, mask=mask)))

    def test_multiplier_and_resolve(self):
        from chronax.models.nhits.nhits_losses import outputsize_multiplier, resolve
        assert outputsize_multiplier(resolve("mae")) == 1
        assert outputsize_multiplier(MultiQuantileLoss((0.1, 0.5, 0.9))) == 3
        assert outputsize_multiplier(GMM(n_components=3)) == 6
        with pytest.raises(ValueError, match="Unknown loss"):
            resolve("nope")


# =============================================================================
# Scaler (verbatim copy of the mlp scaler module — pin against drift)
# =============================================================================

class TestScalerDriftVsMLP:
    def test_robust_scaler_matches_mlp_copy(self):
        from chronax.models.mlp.mlp_scaler import RobustScaler as MRS
        rng = np.random.RandomState(0)
        x = jnp.asarray(rng.randn(5, 12), jnp.float32)
        (s1, c1), (s2, c2) = RobustScaler().stats(x), MRS().stats(x)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
        np.testing.assert_array_equal(np.asarray(c1), np.asarray(c2))

    def test_torch_median_lower_of_two_middles(self):
        from chronax.models.nhits.nhits_scaler import _torch_median
        x = jnp.asarray([[1.0, 2.0, 3.0, 10.0]])
        # torch convention: the LOWER middle order statistic (jnp.median: 2.5).
        assert float(_torch_median(x, axis=1)[0, 0]) == 2.0


# =============================================================================
# Module — interpolation matrices (torch F.interpolate goldens)
# =============================================================================

# Probed from torch 2.x: F.interpolate(eye[:, None, :], size=m, mode=...) for
# linear/nearest; cubic via the reference's bicubic-on-height-1 route
# (F.interpolate(eye[:, None, None, :], size=m, mode="bicubic")[:, 0, 0, :]).
_GOLD_LINEAR_6_4 = [[0.75, 0.0, 0.0, 0.0], [0.25, 0.25, 0.0, 0.0], [0.0, 0.75, 0.0, 0.0],
                    [0.0, 0.0, 0.75, 0.0], [0.0, 0.0, 0.25, 0.25], [0.0, 0.0, 0.0, 0.75]]
_GOLD_LINEAR_3_8 = [[1.0, 0.9375, 0.5625, 0.1875, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0625, 0.4375, 0.8125, 0.8125, 0.4375, 0.0625, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.1875, 0.5625, 0.9375, 1.0]]
_GOLD_NEAREST_6_4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
_GOLD_NEAREST_5_3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
_GOLD_CUBIC_6_4 = [[0.7734375, -0.03515625, 0.0, 0.0], [0.26171875, 0.26171875, 0.0, 0.0],
                   [-0.03515625, 0.87890625, -0.10546875, 0.0],
                   [0.0, -0.10546875, 0.87890625, -0.03515625],
                   [0.0, 0.0, 0.26171875, 0.26171875], [0.0, 0.0, -0.03515625, 0.7734375]]
_GOLD_CUBIC_3_8 = [[1.11077880859375, 0.9503173828125, 0.5701904296875, 0.1636962890625,
                    -0.09283447265625, -0.08074951171875, -0.00274658203125, 0.0],
                   [-0.11077880859375, 0.05242919921875, 0.51055908203125, 0.9291382431983948,
                    0.9291382431983948, 0.51055908203125, 0.05242919921875, -0.11077880859375],
                   [0.0, -0.00274658203125, -0.08074951171875, -0.09283447265625,
                    0.1636962890625, 0.5701904296875, 0.9503173828125, 1.11077880859375]]


class TestInterpWeights:
    @pytest.mark.parametrize("mode,n_in,n_out,gold", [
        ("linear", 6, 4, _GOLD_LINEAR_6_4),
        ("linear", 3, 8, _GOLD_LINEAR_3_8),
        ("nearest", 6, 4, _GOLD_NEAREST_6_4),
        ("nearest", 5, 3, _GOLD_NEAREST_5_3),      # tie-prone downsample: skips inputs
        ("cubic", 6, 4, _GOLD_CUBIC_6_4),
        ("cubic", 3, 8, _GOLD_CUBIC_3_8),
    ])
    def test_matches_torch_goldens(self, mode, n_in, n_out, gold):
        np.testing.assert_allclose(_interp_weights(n_in, n_out, mode), gold, atol=1e-6)

    @pytest.mark.parametrize("mode", ["linear", "nearest", "cubic"])
    def test_partition_of_unity(self, mode):
        # Every output position's weights sum to 1 (all modes interpolate a
        # constant exactly), for any size combination.
        for n_in in (1, 2, 3, 5, 8):
            for n_out in (1, 3, 4, 7, 16):
                W = _interp_weights(n_in, n_out, mode)
                np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-12)

    def test_identity_when_sizes_match(self):
        for mode in ("linear", "nearest"):
            np.testing.assert_allclose(_interp_weights(5, 5, mode), np.eye(5), atol=1e-12)

    def test_single_knot_is_constant(self):
        for mode in ("linear", "nearest", "cubic"):
            np.testing.assert_allclose(_interp_weights(1, 6, mode), np.ones((1, 6)), atol=1e-12)


# =============================================================================
# Module — pooling (torch MaxPool1d/AvgPool1d ceil_mode=True goldens)
# =============================================================================

class TestPooling:
    def test_max_ceil_partial_window(self):
        np.testing.assert_allclose(_pool1d(jnp.array([[1., 2., 3.]]), 2, avg=False), [[2., 3.]])

    def test_avg_divides_by_valid_count(self):
        np.testing.assert_allclose(_pool1d(jnp.array([[1., 2., 3.]]), 2, avg=True), [[1.5, 3.]])
        np.testing.assert_allclose(
            _pool1d(jnp.array([[1., 2., 3., 4., 5.]]), 3, avg=True), [[2., 4.5]])

    def test_k1_is_identity(self):
        x = jnp.arange(6.0).reshape(1, 6)
        np.testing.assert_array_equal(np.asarray(_pool1d(x, 1, avg=False)), np.asarray(x))

    def test_pools_last_axis_of_3d(self):
        # Exog pooling path: [B, F, T] -> [B, F, ceil(T/k)] per channel.
        x = jnp.asarray(np.arange(12.0).reshape(1, 2, 6))
        out = _pool1d(x, 4, avg=False)
        np.testing.assert_allclose(out, [[[3., 5.], [9., 11.]]])

    def test_max_handles_negatives(self):
        np.testing.assert_allclose(
            _pool1d(jnp.array([[-5., -2., -7.]]), 2, avg=False), [[-2., -7.]])


# =============================================================================
# Module — NumPy-reference forward, structure, activations, validation
# =============================================================================

def _np_ref_forward(net: NHITSNet, y: np.ndarray) -> np.ndarray:
    """Independent NumPy replication of the NHITS forward (flip, pooling,
    MLP, theta split, interpolation, doubly residual wiring, Naive1 anchor)."""
    B, L = y.shape
    resid = y[:, ::-1].copy()
    fcst = y[:, -1:][:, :, None]                              # [B, 1, 1]
    for block in net.blocks:
        k = block.k
        if k == 1:
            x = resid.copy()
        else:
            n_out = -(-L // k)
            x = np.full((B, n_out), -np.inf)
            for i in range(n_out):
                x[:, i] = resid[:, i * k: (i + 1) * k].max(axis=1)
        for j, layer in enumerate(block.layers):
            x = x @ np.asarray(layer.kernel.value) + np.asarray(layer.bias.value)
            if j > 0:
                x = np.maximum(x, 0.0)                        # ReLU after non-entry layers
            elif j == 0 and len(block.layers) > 1:
                pass                                          # entry Linear: no activation
        theta = x @ np.asarray(block.out_layer.kernel.value) + np.asarray(block.out_layer.bias.value)
        backcast = theta[:, :L]
        knots = theta[:, L:].reshape(B, block.out_features, block.n_knots)
        W = _interp_weights(block.n_knots, block.h, block.interpolation_mode)
        f = np.transpose(knots @ W, (0, 2, 1))                # [B, h, Q]
        resid = resid - backcast
        fcst = fcst + f
    return fcst


def _param_count(net) -> int:
    return sum(int(np.prod(v.shape)) for v in jax.tree.leaves(nnx.state(net, nnx.Param)))


class TestNHITSNet:
    def test_numpy_reference_forward(self):
        # Two stacks with real pooling and real interpolation; pins the flip,
        # residual, anchor, theta-split, and knot-reshape wiring end to end.
        net = NHITSNet(h=6, input_size=8, n_blocks=(1, 2), mlp_units=((8, 8),),
                       n_pool_kernel_size=(2, 1), n_freq_downsample=(3, 1),
                       rngs=nnx.Rngs(0))
        rng = np.random.RandomState(0)
        y = rng.randn(4, 8).astype(np.float32)
        ours = np.asarray(net(jnp.asarray(y)[:, :, None]))
        ref = _np_ref_forward(net, y.astype(np.float64))
        assert ours.shape == (4, 6, 1)
        np.testing.assert_allclose(ours, ref, atol=1e-5)

    def test_param_count_matches_torch_at_defaults(self):
        # Pinned against neuralforecast 3.1.7 NHITS(h=12, input_size=36) with the
        # default 3-stack structure: sum(p.numel() for p in m.blocks.parameters()).
        net = NHITSNet(h=12, input_size=36, rngs=nnx.Rngs(0))
        assert _param_count(net) == 2_468_481

    def test_block_count_follows_n_blocks(self):
        net = NHITSNet(h=4, input_size=8, n_blocks=(2, 0, 1), mlp_units=((8, 8),),
                       n_pool_kernel_size=(2, 2, 1), n_freq_downsample=(4, 2, 1),
                       rngs=nnx.Rngs(0))
        assert len(net.blocks) == 3
        assert [b.k for b in net.blocks] == [2, 2, 1]

    def test_multi_output_head_shape(self):
        net = NHITSNet(h=5, input_size=8, n_blocks=(1,), mlp_units=((8, 8),),
                       n_pool_kernel_size=(2,), n_freq_downsample=(2,),
                       outputsize_multiplier=3, rngs=nnx.Rngs(0))
        out = net(jnp.ones((2, 8, 1)))
        assert out.shape == (2, 5, 3)

    def test_naive1_anchor_reaches_every_output(self):
        # Zero every parameter: theta == 0, so each output must equal the
        # window's last (scaled) value — the Naive1 anchor broadcast onto all
        # horizon steps and output features.
        net = NHITSNet(h=4, input_size=6, n_blocks=(1,), mlp_units=((8, 8),),
                       n_pool_kernel_size=(2,), n_freq_downsample=(2,),
                       outputsize_multiplier=2, rngs=nnx.Rngs(0))
        state = nnx.state(net, nnx.Param)
        nnx.update(net, jax.tree.map(jnp.zeros_like, state))
        y = jnp.asarray(np.random.RandomState(0).randn(3, 6, 1), jnp.float32)
        out = np.asarray(net(y))
        np.testing.assert_allclose(out, np.broadcast_to(np.asarray(y)[:, -1:, :], out.shape),
                                   atol=1e-7)

    @pytest.mark.parametrize("name", ACTIVATIONS)
    def test_activations_run_finite(self, name):
        net = NHITSNet(h=3, input_size=6, n_blocks=(1,), mlp_units=((8, 8),),
                       n_pool_kernel_size=(2,), n_freq_downsample=(1,),
                       activation=name, rngs=nnx.Rngs(0))
        out = net(jnp.asarray(np.random.RandomState(0).randn(2, 6, 1), jnp.float32))
        assert bool(jnp.all(jnp.isfinite(out)))
        if name == "PReLU":
            assert float(net.blocks[0].prelu_a.value[0]) == 0.25

    def test_dropout_train_vs_eval(self):
        net = NHITSNet(h=3, input_size=6, n_blocks=(1,), mlp_units=((8, 8),),
                       n_pool_kernel_size=(1,), n_freq_downsample=(1,),
                       dropout_prob_theta=0.5, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 6, 1))
        d1, d2 = net(x, deterministic=True), net(x, deterministic=True)
        np.testing.assert_array_equal(np.asarray(d1), np.asarray(d2))
        s1, s2 = net(x, deterministic=False), net(x, deterministic=False)
        assert not np.array_equal(np.asarray(s1), np.asarray(s2))

    def test_validation_raises(self):
        mk = dict(h=4, input_size=8, rngs=nnx.Rngs(0))
        with pytest.raises(ValueError, match="equal"):
            NHITSNet(n_blocks=(1, 1), n_pool_kernel_size=(2,), n_freq_downsample=(2,), **mk)
        with pytest.raises(ValueError, match="chain"):
            NHITSNet(mlp_units=((8, 8), (16, 8)), n_blocks=(1,), n_pool_kernel_size=(2,),
                     n_freq_downsample=(2,), **mk)
        with pytest.raises(ValueError, match="at least one block"):
            NHITSNet(n_blocks=(0, 0, 0), **mk)
        with pytest.raises(ValueError, match="pooling_mode"):
            NHITSNet(pooling_mode="SumPool1d", **mk)
        with pytest.raises(ValueError, match="activation"):
            NHITSNet(activation="GELU", **mk)
        with pytest.raises(ValueError, match="interpolation_mode"):
            NHITSNet(interpolation_mode="quadratic", **mk)
        with pytest.raises(ValueError, match="Cubic"):
            NHITSNet(interpolation_mode="cubic", outputsize_multiplier=3, **mk)
        with pytest.raises(ValueError, match="dropout"):
            NHITSNet(dropout_prob_theta=1.0, **mk)
        with pytest.raises(ValueError, match=">= 1"):
            NHITSNet(n_pool_kernel_size=(0, 2, 1), **mk)


# =============================================================================
# Training
# =============================================================================

def _make_y(T=60):
    t = np.arange(T)
    return jnp.asarray(8.0 + 0.3 * t + 3.0 * np.sin(t / 3.0), jnp.float32)


def _tnet(**kw):
    base = dict(h=4, input_size=12, n_blocks=(1, 1), mlp_units=((16, 16),),
                n_pool_kernel_size=(2, 1), n_freq_downsample=(2, 1), rngs=nnx.Rngs(0))
    base.update(kw)
    return NHITSNet(**base)


class TestTraining:
    def test_lr_schedule_steplr_boundaries(self):
        sched = _lr_schedule(1e-3, 1000, 3)
        for count, lr in [(0, 1e-3), (332, 1e-3), (333, 5e-4), (665, 5e-4),
                          (666, 2.5e-4), (998, 2.5e-4), (999, 1.25e-4)]:
            np.testing.assert_allclose(float(sched(count)), lr, rtol=1e-6)

    def test_lr_schedule_disabled(self):
        assert _lr_schedule(1e-3, 1000, -1) == 1e-3
        assert _lr_schedule(1e-3, 1000, 0) == 1e-3
        # A single decay whose boundary falls at max_steps stays constant
        # (torch never reaches the post-final-step epoch).
        assert _lr_schedule(1e-3, 10, 1) == 1e-3

    def test_train_loss_decreases(self):
        net = _tnet()
        losses = train(net, _make_y(), h=4, input_size=12, max_steps=60,
                       windows_batch_size=32, lr=1e-2, num_lr_decays=3, seed=0,
                       loss_fn=mae, scaler=RobustScaler())
        assert float(losses[-5:].mean()) < float(losses[:5].mean())

    def test_train_is_vmap_traceable(self):
        # The training path must trace under vmap (the
        # BaseForecaster.conformity_scores requirement).
        def run(seed):
            net = _tnet()
            return train(net, _make_y(), h=4, input_size=12, max_steps=3,
                         windows_batch_size=16, lr=1e-3, num_lr_decays=3, seed=seed,
                         loss_fn=mae, scaler=RobustScaler())
        out = jax.vmap(run)(jnp.arange(2))
        assert out.shape == (2, 3) and bool(jnp.all(jnp.isfinite(out)))

    def test_divergence_guard_raises(self):
        net = _tnet()
        with pytest.raises(RuntimeError, match="diverged"):
            train(net, _make_y(), h=4, input_size=12, max_steps=10,
                  windows_batch_size=16, lr=1e12, num_lr_decays=-1, seed=0,
                  loss_fn=mae, scaler=RobustScaler())

    def test_minimum_length_series_trains(self):
        # T = input_size + 1: exactly one (fully h-padded) window.
        net = _tnet()
        losses = train(net, _make_y(13), h=4, input_size=12, max_steps=3,
                       windows_batch_size=4, lr=1e-3, num_lr_decays=-1, seed=0,
                       loss_fn=mae, scaler=RobustScaler())
        assert bool(jnp.all(jnp.isfinite(losses)))


# =============================================================================
# Model (BaseForecaster conformance)
# =============================================================================

def _tiny(**kw):
    base = dict(h=4, input_size=12, mlp_units=[[16, 16]], n_blocks=[1, 1],
                n_pool_kernel_size=[2, 1], n_freq_downsample=[2, 1],
                max_steps=30, windows_batch_size=16, random_seed=0)
    base.update(kw)
    return NHITS(**base)


class TestNHITSModel:
    def test_fit_predict_shapes_point(self):
        m = _tiny().fit(_make_y())
        out = m.predict(h=4)
        assert set(out) == {"mean"} and out["mean"].shape == (4,)
        out2 = m.predict(h=2)
        assert out2["mean"].shape == (2,)

    def test_gmm_native_interval_keys_and_order(self):
        m = _tiny(loss=GMM(n_components=2, num_samples=256)).fit(_make_y())
        out = m.predict(h=4, level=[80, 95])
        assert set(out) == {"mean", "lo-80", "hi-80", "lo-95", "hi-95"}
        assert bool(jnp.all(out["lo-95"] <= out["lo-80"]))
        assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))
        assert bool(jnp.all(out["hi-80"] <= out["hi-95"]))

    def test_gmm_native_ignores_conformal_params(self):
        m1 = _tiny(loss=GMM(n_components=2, num_samples=128)).fit(_make_y())
        m2 = _tiny(loss=GMM(n_components=2, num_samples=128))
        m2.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m2.fit(_make_y())
        o1, o2 = m1.predict(h=4, level=[80]), m2.predict(h=4, level=[80])
        for k in o1:
            np.testing.assert_array_equal(np.asarray(o1[k]), np.asarray(o2[k]))

    def test_mq_head_levels_and_sort(self):
        m = _tiny(loss=MultiQuantileLoss((0.1, 0.5, 0.9))).fit(_make_y())
        out = m.predict(h=4, level=[80])
        assert set(out) == {"mean", "lo-80", "hi-80"}
        assert bool(jnp.all(out["lo-80"] <= out["mean"]))
        assert bool(jnp.all(out["mean"] <= out["hi-80"]))
        with pytest.raises(ValueError, match="not trained"):
            m.predict(h=4, level=[95])

    def test_point_loss_conformal_path(self):
        m = _tiny()
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_make_y())
        out = m.predict(h=4, level=[80])
        assert {"mean", "lo-80", "hi-80"} <= set(out)
        assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))

    def test_point_loss_level_without_conformal_raises(self):
        m = _tiny().fit(_make_y())
        with pytest.raises(ValueError, match="conformal_params"):
            m.predict(h=4, level=[80])

    def test_predict_level_twice_then_pickle_then_predict_conformal(self):
        m = _tiny()
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_make_y())
        o1 = m.predict(h=4, level=[80])
        o2 = m.predict(h=4, level=[80])
        np.testing.assert_array_equal(np.asarray(o1["lo-80"]), np.asarray(o2["lo-80"]))
        m2 = pickle.loads(pickle.dumps(m))
        o3 = m2.predict(h=4, level=[80])
        np.testing.assert_array_equal(np.asarray(o1["mean"]), np.asarray(o3["mean"]))

    def test_predict_level_twice_then_pickle_then_predict_gmm(self):
        m = _tiny(loss=GMM(n_components=2, num_samples=128)).fit(_make_y())
        o1 = m.predict(h=4, level=[90])
        o2 = m.predict(h=4, level=[90])
        np.testing.assert_array_equal(np.asarray(o1["hi-90"]), np.asarray(o2["hi-90"]))
        m2 = pickle.loads(pickle.dumps(m))
        o3 = m2.predict(h=4, level=[90])
        for k in o1:
            np.testing.assert_array_equal(np.asarray(o1[k]), np.asarray(o3[k]))

    def test_forecast_stateless_matches_fit_predict(self):
        y = _make_y()
        o1 = _tiny().forecast(y, h=4)
        o2 = _tiny().fit(y).predict(h=4)
        np.testing.assert_array_equal(np.asarray(o1["mean"]), np.asarray(o2["mean"]))

    def test_forecast_fitted_values(self):
        y = _make_y()
        out = _tiny().forecast(y, h=4, fitted=True)
        assert out["fitted"].shape == y.shape
        assert bool(jnp.all(jnp.isnan(out["fitted"][:12])))
        assert bool(jnp.all(jnp.isfinite(out["fitted"][12:])))

    def test_beats_naive(self):
        t = np.arange(140)
        y = 10 + 0.5 * t + 4 * np.sin(t / 4.0)
        y_train, y_test = jnp.asarray(y[:132], jnp.float32), y[132:]
        m = NHITS(h=8, input_size=24, mlp_units=[[32, 32]], n_blocks=[1, 1],
                  n_pool_kernel_size=[2, 1], n_freq_downsample=[2, 1],
                  max_steps=300, windows_batch_size=64, random_seed=0,
                  scaler_type="robust").fit(y_train)
        pred = np.asarray(m.predict(h=8)["mean"])
        mae_nhits = np.mean(np.abs(pred - y_test))
        mae_naive = np.mean(np.abs(float(y_train[-1]) - y_test))
        assert mae_nhits < mae_naive

    def test_futr_exog_fit_predict(self):
        rng = np.random.RandomState(0)
        T, F = 60, 2
        x = jnp.asarray(rng.randn(T, F), jnp.float32)
        m = _tiny().fit(_make_y(T), futr_exog=x)
        out = m.predict(h=4, futr_exog=jnp.asarray(rng.randn(4, F), jnp.float32))
        assert out["mean"].shape == (4,)
        with pytest.raises(ValueError, match="futr_exog"):
            m.predict(h=4)

    def test_dropout_fit_and_deterministic_predict(self):
        m = _tiny(dropout_prob_theta=0.3).fit(_make_y())
        o1, o2 = m.predict(h=4), m.predict(h=4)
        np.testing.assert_array_equal(np.asarray(o1["mean"]), np.asarray(o2["mean"]))

    def test_rejects_3d_y(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            _tiny().fit(jnp.ones((30, 2, 2)))

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="too short"):
            _tiny().fit(jnp.ones((12,)))

    def test_h_gt_trained_raises(self):
        m = _tiny().fit(_make_y())
        with pytest.raises(ValueError, match="h"):
            m.predict(h=9)

    def test_default_ctor_matches_nf_defaults(self):
        m = NHITS(h=4)
        assert m.input_size == 12
        assert m.n_blocks == (1, 1, 1)
        assert m.mlp_units == ((512, 512), (512, 512), (512, 512))
        assert m.n_pool_kernel_size == (2, 2, 1)
        assert m.n_freq_downsample == (4, 2, 1)
        assert (m.pooling_mode, m.interpolation_mode) == ("MaxPool1d", "linear")
        assert (m.dropout_prob_theta, m.activation) == (0.0, "ReLU")
        assert (m.max_steps, m.learning_rate, m.num_lr_decays) == (1000, 1e-3, 3)
        assert (m.windows_batch_size, m.scaler_type, m.loss, m.random_seed) == (
            1024, "identity", "mae", 1)

    def test_vmap_forecast_matches_loop(self):
        # Eager-vs-vmap forecast EQUALITY — the execution regime
        # conformity_scores runs in. Batching changes XLA fusion, so tolerance
        # not bit-equality.
        ys = jnp.stack([_make_y(48), _make_y(48) * 1.7 + 3.0, _make_y(48)[::-1]])

        def fc(y):
            m = NHITS(h=4, input_size=12, mlp_units=[[8, 8]], n_blocks=[1],
                      n_pool_kernel_size=[2], n_freq_downsample=[2], max_steps=3,
                      windows_batch_size=16, random_seed=0)
            return m.forecast(y, h=4)["mean"]

        batched = jax.vmap(fc)(ys)
        for i in range(ys.shape[0]):
            np.testing.assert_allclose(np.asarray(fc(ys[i])), np.asarray(batched[i]),
                                       rtol=1e-5, atol=1e-5)

    def test_no_recompile_on_second_predict_and_refit(self):
        import logging
        m = _tiny(max_steps=3).fit(_make_y())
        m.predict(h=4)                                   # compile predict path
        counts = []

        class _H(logging.Handler):
            def emit(self, record):
                if "Finished XLA compilation" in record.getMessage():
                    counts.append(record.getMessage())

        handler = _H()
        loggers = [logging.getLogger("jax._src.dispatch"),
                   logging.getLogger("jax._src.interpreters.pxla")]
        with jax.log_compiles(True):
            for lg in loggers:
                lg.addHandler(handler)
            try:
                m.predict(h=4)                           # same shape: no compile
                n_repeat = len(counts)
                m2 = _tiny(max_steps=3).fit(_make_y())   # fresh same-config instance
                n_after_fit = len(counts)                # (training scan compiles per fit
                                                         #  — the known template-wide cost)
                m2.predict(h=4)                          # graphdef cache must hit
                n_refit_predict = len(counts) - n_after_fit
            finally:
                for lg in loggers:
                    lg.removeHandler(handler)
        assert n_repeat == 0, f"repeat predict compiled: {counts[:n_repeat]}"
        assert n_refit_predict == 0, f"fresh-instance predict recompiled ({n_refit_predict})"


# =============================================================================
# Multivariate (cross-learned N-series) surface
# =============================================================================

def _panel(T=72, seed=0):
    """Three independent series as columns — a plain panel, no hierarchy."""
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    cols = [10 + 0.2 * t + 2 * np.sin(t / 3.0) + 0.3 * rng.randn(T),
            30 + 0.1 * t + 3 * np.cos(t / 5.0) + 0.3 * rng.randn(T),
            5 + 0.05 * t + np.sin(t / 2.0) + 0.2 * rng.randn(T)]
    return jnp.asarray(np.stack(cols, axis=1), jnp.float32)


class TestNHITSMultivariate:
    def test_2d_fit_predict_shapes_gmm(self):
        m = _tiny(loss=GMM(n_components=2, num_samples=256)).fit(_panel())
        out = m.predict(h=4, level=[80, 95])
        assert set(out) == {"mean", "lo-80", "hi-80", "lo-95", "hi-95"}
        for v in out.values():
            assert v.shape == (4, 3)
        assert bool(jnp.all(out["lo-95"] <= out["lo-80"]))
        assert bool(jnp.all(out["hi-80"] <= out["hi-95"]))

    def test_2d_point_path(self):
        m = _tiny().fit(_panel())
        out = m.predict(h=4)
        assert set(out) == {"mean"} and out["mean"].shape == (4, 3)

    def test_rank_rule(self):
        y = _make_y()
        m1 = _tiny().fit(y)
        assert m1.predict(h=4)["mean"].shape == (4,)
        m2 = _tiny().fit(y[:, None])
        assert m2.predict(h=4)["mean"].shape == (4, 1)

    def test_2d_single_column_matches_1d(self):
        y = _make_y()
        p1 = _tiny().fit(y).predict(h=4)["mean"]
        p2 = _tiny().fit(y[:, None]).predict(h=4)["mean"]
        np.testing.assert_array_equal(np.asarray(p1), np.asarray(p2[:, 0]))

    def test_2d_columns_not_scrambled(self):
        # Disjoint value ranges per column: the robust per-window scaler anchors
        # each forecast to its own context median, so a scrambled column order
        # would land forecasts in the wrong range by construction.
        rng = np.random.RandomState(0)
        t = np.arange(60)
        panel = np.stack([1000.0 + np.sin(t / 3.0) + 0.1 * rng.randn(60),
                          5000.0 + np.cos(t / 4.0) + 0.1 * rng.randn(60),
                          9000.0 + np.sin(t / 5.0) + 0.1 * rng.randn(60)], axis=1)
        m = _tiny(scaler_type="robust", max_steps=5).fit(jnp.asarray(panel, jnp.float32))
        mean = np.asarray(m.predict(h=4)["mean"])
        for j, center in enumerate([1000.0, 5000.0, 9000.0]):
            assert np.all(np.abs(mean[:, j] - center) < 500.0)

    def test_2d_point_level_raises(self):
        m = _tiny().fit(_panel())
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        with pytest.raises(ValueError, match="native"):
            m.predict(h=4, level=[80])

    def test_2d_conformity_scores_raises(self):
        m = _tiny()
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_panel())
        with pytest.raises(ValueError, match="native"):
            m.conformity_scores(_panel())

    def test_2d_fitted_raises(self):
        with pytest.raises(NotImplementedError, match="fitted"):
            _tiny().forecast(_panel(), h=4, fitted=True)

    def test_2d_gmm_shared_futr_exog(self):
        # Distribution head + 2-D fit + shared exog: predict_params must
        # broadcast the shared exog window across the batch of contexts.
        rng = np.random.RandomState(0)
        T, F = 72, 2
        x = jnp.asarray(rng.randn(T, F), jnp.float32)
        m = _tiny(loss=GMM(n_components=2, num_samples=128)).fit(_panel(T), futr_exog=x)
        out = m.predict(h=4, level=[80],
                        futr_exog=jnp.asarray(rng.randn(4, F), jnp.float32))
        assert out["mean"].shape == (4, 3)
        assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))

    def test_2d_fit_is_vmap_traceable(self):
        def run(panel):
            m = NHITS(h=4, input_size=12, mlp_units=[[8, 8]], n_blocks=[1],
                      n_pool_kernel_size=[2], n_freq_downsample=[2], max_steps=3,
                      windows_batch_size=16, random_seed=0,
                      loss=GMM(n_components=2, num_samples=64))
            m.fit(panel)
            return m.predict(h=4)["mean"]
        panels = jnp.stack([_panel(seed=0), _panel(seed=1)])
        out = jax.vmap(run)(panels)
        assert out.shape == (2, 4, 3) and bool(jnp.all(jnp.isfinite(out)))

    def test_2d_pickle_roundtrip_and_twice(self):
        m = _tiny(loss=GMM(n_components=2, num_samples=128)).fit(_panel())
        o1 = m.predict(h=4, level=[90])
        o2 = m.predict(h=4, level=[90])
        np.testing.assert_array_equal(np.asarray(o1["hi-90"]), np.asarray(o2["hi-90"]))
        m2 = pickle.loads(pickle.dumps(m))
        o3 = m2.predict(h=4, level=[90])
        for k in o1:
            np.testing.assert_array_equal(np.asarray(o1[k]), np.asarray(o3[k]))


# =============================================================================
# Namespace
# =============================================================================

class TestNamespace:
    def test_registered(self):
        import chronax.models as M
        assert M.NHITS is NHITS
        assert "NHITS" in M.__all__


# === Refit compile-cache gates (module-level nnx.jit _train_scan) ===
import io as _io
import logging as _logging
import re as _re

_COMPILE_LOG_RE = _re.compile(r"Compiling ")


def _count_compiles(fn):
    """Run fn under jax.log_compiles and return (result, n_xla_compilations)."""
    buf = _io.StringIO()
    handler = _logging.StreamHandler(buf)
    loggers = [
        _logging.getLogger("jax._src.dispatch"),
        _logging.getLogger("jax._src.interpreters.pxla"),
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
            return x * 6.0 - 3.0
        return f(jnp.ones((2, 7)))

    _, n = _count_compiles(fresh)
    assert n >= 1


def test_refit_does_not_recompile():
    # The training program is a module-level nnx.jit keyed on config graphdefs
    # (value-__eq__ initializers and the memoized optax transform) and operand
    # shapes — a refit AND a fresh same-config instance must hit the cache with
    # zero XLA compiles; fit #2 is counted directly (no uncounted settle call).
    y = jnp.asarray(np.sin(np.arange(40) / 5.0), jnp.float32)
    kw = dict(h=4, input_size=8, max_steps=3, windows_batch_size=4, random_seed=0)
    m = NHITS(**kw)
    m.fit(y)                                     # first fit pays the one compile
    _, n_refit = _count_compiles(lambda: (m.fit(y), None)[1])
    assert n_refit == 0
    m2 = NHITS(**kw)
    _, n_fresh = _count_compiles(lambda: (m2.fit(y), None)[1])
    assert n_fresh == 0


def test_mqloss_pinball_value_nf_sum():
    # err = y - yhat. Pred zeros, target 2: sum_q max(q*2, (q-1)*2) =
    # 0.2 + 1.0 + 1.8 = 3.0 — neuralforecast's effective reduction (its
    # 1/len(quantiles) factor is dead), matching the trainer's masked branch.
    from chronax.models.nhits.nhits_losses import MultiQuantileLoss

    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    assert float(loss(jnp.zeros((1, 1, 3)), jnp.array([[2.0]]))) == pytest.approx(3.0)

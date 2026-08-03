"""Tests for the TimeMixer forecaster package (chronax.models.timemixer).

Sections: Losses -> Module -> Training -> Model -> Box-Cox, mirroring the
package's source modules. Reference behaviors follow neuralforecast 3.1.7's
``models/timemixer.py`` + ``common/_modules.py`` (MovingAvg/SeriesDecomp/RevIN/
TokenEmbedding), with convolutions expressed as shifted GEMMs (equivalence
pinned against ``lax.conv_general_dilated`` at fixed weights).
"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.timemixer.timemixer_losses import LOSSES, huber, mae, mse, resolve
from chronax.models.timemixer.timemixer_module import (
    RevIN,
    TimeMixerNet,
    TokenEmbedding,
    _dft_decomp,
    _moving_avg,
    _series_decomp,
)

# =============================================================================
# Losses
# =============================================================================


class TestLosses:
    def test_values(self):
        p = jnp.asarray([1.0, 2.0, 3.0])
        t = jnp.asarray([2.0, 2.0, 5.0])
        assert float(mae(p, t)) == pytest.approx(1.0)
        assert float(mse(p, t)) == pytest.approx((1 + 0 + 4) / 3)
        assert float(huber(p, t)) == pytest.approx((0.5 + 0 + 1.5) / 3)

    def test_resolve(self):
        assert resolve("mae") is mae
        assert resolve(mse) is mse
        with pytest.raises(ValueError, match="Unknown loss"):
            resolve("nope")
        assert set(LOSSES) == {"mae", "mse", "huber"}


# =============================================================================
# Module
# =============================================================================


def _rand(*shape, seed=0, scale=1.0):
    return jnp.asarray(np.random.RandomState(seed).randn(*shape) * scale, jnp.float32)


class TestMovingAvgAndDecomp:
    def test_moving_avg_matches_numpy_reference(self):
        # Replicate-pad (k-1)//2 on both ends, then windowed mean, stride 1.
        x = _rand(2, 10, 3)
        k = 5
        got = np.asarray(_moving_avg(x, k))
        xn = np.asarray(x)
        pad = (k - 1) // 2
        xp = np.concatenate(
            [np.repeat(xn[:, :1], pad, 1), xn, np.repeat(xn[:, -1:], pad, 1)], axis=1
        )
        ref = np.stack([xp[:, j:j + 10] for j in range(k)], 0).mean(0)
        np.testing.assert_allclose(got, ref, rtol=1e-6)

    def test_moving_avg_even_kernel_raises(self):
        with pytest.raises(ValueError, match="odd"):
            _moving_avg(_rand(1, 8, 1), 4)

    def test_series_decomp_reconstructs(self):
        x = _rand(2, 12, 2)
        season, trend = _series_decomp(x, 5)
        np.testing.assert_allclose(np.asarray(season + trend), np.asarray(x), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(trend), np.asarray(_moving_avg(x, 5)), rtol=1e-6)

    def test_dft_decomp_matches_numpy_reference(self):
        # Verbatim port of the reference expressions: rfft over the LAST axis,
        # magnitudes with the first BATCH entry zeroed, global top-k threshold,
        # irfft back; trend = x - season.
        x = _rand(2, 6, 8)
        season, trend = _dft_decomp(x, top_k=3)
        xn = np.asarray(x, np.float64)
        xf = np.fft.rfft(xn)
        freq = np.abs(xf)
        freq[0] = 0
        topk_vals = np.sort(freq, axis=-1)[..., -3:]
        thresh = topk_vals.min()                       # GLOBAL scalar min (reference quirk)
        xf[freq <= thresh] = 0
        ref_season = np.fft.irfft(xf, n=xn.shape[-1])
        np.testing.assert_allclose(np.asarray(season), ref_season, atol=1e-5)
        np.testing.assert_allclose(np.asarray(season + trend), xn, atol=1e-5)


class TestTokenEmbedding:
    def test_matches_conv_primitive(self):
        # k=3 circular conv expressed as 3 rolled GEMMs must equal the lax conv.
        emb = TokenEmbedding(c_in=3, hidden_size=8, rngs=nnx.Rngs(0))
        x = _rand(2, 10, 3)
        got = emb(x)                                            # [B, T, d]
        w = np.asarray(emb.weight.value)                        # [d, c_in, 3]
        xp = jnp.concatenate([x[:, -1:], x, x[:, :1]], axis=1)  # circular pad 1
        ref = jax.lax.conv_general_dilated(
            jnp.transpose(xp, (0, 2, 1)), jnp.asarray(w), (1,), "VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        np.testing.assert_allclose(np.asarray(got), np.asarray(jnp.transpose(ref, (0, 2, 1))),
                                   rtol=2e-5, atol=2e-6)

    def test_kaiming_scale(self):
        emb = TokenEmbedding(c_in=4, hidden_size=256, rngs=nnx.Rngs(0))
        w = np.asarray(emb.weight.value)
        assert w.std() == pytest.approx(np.sqrt(2.0 / (4 * 3)), rel=0.15)


class TestRevIN:
    def test_roundtrip_affine(self):
        r = RevIN(3, affine=True, non_norm=False, rngs=nnx.Rngs(0))
        r.gamma.value = jnp.asarray([1.5, 0.5, 2.0])
        r.beta.value = jnp.asarray([0.1, -0.2, 0.3])
        x = _rand(2, 12, 3, scale=5.0)
        z, loc, scale = r.norm(x)
        back = r.denorm(z, loc, scale)
        np.testing.assert_allclose(np.asarray(back), np.asarray(x), rtol=1e-4, atol=1e-4)

    def test_eps_squared_denorm_quirk(self):
        # The affine inverse divides by (gamma + eps**2), not (gamma + eps).
        r = RevIN(1, affine=True, non_norm=False, rngs=nnx.Rngs(0))
        r.gamma.value = jnp.asarray([2.0])
        z = jnp.ones((1, 4, 1))
        out = r.denorm(z, jnp.zeros((1, 1, 1)), jnp.ones((1, 1, 1)))
        np.testing.assert_allclose(np.asarray(out), (1.0 - 0.0) / (2.0 + 1e-5 ** 2), rtol=1e-6)

    def test_non_norm_passthrough_with_live_params(self):
        r = RevIN(2, affine=True, non_norm=True, rngs=nnx.Rngs(0))
        assert r.gamma.value.shape == (2,)                     # params exist (reference parity)
        x = _rand(1, 6, 2)
        z, loc, scale = r.norm(x)
        np.testing.assert_array_equal(np.asarray(z), np.asarray(x))
        np.testing.assert_array_equal(np.asarray(r.denorm(z, loc, scale)), np.asarray(x))

    def test_population_variance(self):
        r = RevIN(1, affine=False, non_norm=False, rngs=nnx.Rngs(0))
        x = jnp.asarray([[[1.0], [2.0], [3.0], [4.0]]])
        _, loc, scale = r.norm(x)
        assert float(loc[0, 0, 0]) == pytest.approx(2.5)
        assert float(scale[0, 0, 0]) == pytest.approx(np.sqrt(1.25 + 1e-5), rel=1e-6)


def _tiny_net(**kw):
    cfg = dict(h=4, input_size=12, n_series=1, d_model=8, d_ff=8, dropout=0.0,
               e_layers=2, top_k=3, decomp_method="moving_avg", moving_avg=5,
               channel_independence=0, down_sampling_layers=1, down_sampling_window=2,
               down_sampling_method="avg", use_norm=True, outputsize_multiplier=1)
    cfg.update(kw)
    return TimeMixerNet(rngs=nnx.Rngs(0), **cfg)


class TestTimeMixerNet:
    @pytest.mark.parametrize("ci", [0, 1], ids=["CD", "CI"])
    @pytest.mark.parametrize("n", [1, 3])
    def test_forward_shape(self, ci, n):
        net = _tiny_net(n_series=n, channel_independence=ci)
        out = net(_rand(2, 12, n), deterministic=True)
        assert out.shape == (2, 4, n)

    @pytest.mark.parametrize("method", ["avg", "max", "conv"])
    def test_downsampling_methods(self, method):
        net = _tiny_net(down_sampling_method=method, n_series=2)
        out = net(_rand(2, 12, 2), deterministic=True)
        assert out.shape == (2, 4, 2) and bool(jnp.all(jnp.isfinite(out)))

    def test_conv_downsampling_matches_primitive(self):
        # Stride-2 k=3 circular conv as shifted GEMMs vs the lax primitive.
        net = _tiny_net(down_sampling_method="conv", n_series=2)
        x = _rand(3, 12, 2)
        got = net._downsample_step(x)
        w = jnp.asarray(net.down_conv_weight.value)             # [N, N, 3]
        xc = jnp.transpose(x, (0, 2, 1))                        # [B, N, T]
        xp = jnp.concatenate([xc[:, :, -1:], xc, xc[:, :, :1]], axis=-1)
        ref = jax.lax.conv_general_dilated(xp, w, (2,), "VALID",
                                           dimension_numbers=("NCH", "OIH", "NCH"))
        np.testing.assert_allclose(np.asarray(got), np.asarray(jnp.transpose(ref, (0, 2, 1))),
                                   rtol=2e-5, atol=2e-6)

    def test_dft_decomp_mode(self):
        net = _tiny_net(decomp_method="dft_decomp", n_series=2)
        out = net(_rand(2, 12, 2), deterministic=True)
        assert out.shape == (2, 4, 2) and bool(jnp.all(jnp.isfinite(out)))

    def test_ci0_out_cross_layer_is_dead(self):
        # Reference constructs out_cross_layer for both modes but only applies it
        # at channel independence 1 — poisoning its weights at CI=0 must not move
        # the output.
        net = _tiny_net(channel_independence=0)
        x = _rand(2, 12, 1)
        before = np.asarray(net(x, deterministic=True))
        for blk in net.pdm_blocks:
            blk.out_cross_w1.kernel.value = jnp.full_like(blk.out_cross_w1.kernel.value, 1e6)
        after = np.asarray(net(x, deterministic=True))
        np.testing.assert_array_equal(before, after)

    def test_ci1_out_cross_layer_is_live(self):
        net = _tiny_net(channel_independence=1)
        x = _rand(2, 12, 1)
        before = np.asarray(net(x, deterministic=True))
        for blk in net.pdm_blocks:
            blk.out_cross_w1.kernel.value = blk.out_cross_w1.kernel.value + 0.5
        after = np.asarray(net(x, deterministic=True))
        assert not np.allclose(before, after)

    def test_use_norm_false(self):
        net = _tiny_net(use_norm=False)
        out = net(_rand(2, 12, 1, scale=100.0), deterministic=True)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_input_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            _tiny_net(input_size=13)

    def test_even_moving_avg_raises(self):
        with pytest.raises(ValueError, match="odd"):
            _tiny_net(moving_avg=4)

    def test_dropout_train_vs_eval(self):
        net = _tiny_net(dropout=0.5)
        x = _rand(2, 12, 1)
        e1 = np.asarray(net(x, deterministic=True))
        e2 = np.asarray(net(x, deterministic=True))
        t1 = np.asarray(net(x, deterministic=False))
        np.testing.assert_array_equal(e1, e2)
        assert not np.allclose(e1, t1)

    def test_input_dependence_n3(self):
        net = _tiny_net(n_series=3)
        a = np.asarray(net(_rand(1, 12, 3, seed=1), deterministic=True))
        b = np.asarray(net(_rand(1, 12, 3, seed=2), deterministic=True))
        assert not np.allclose(a, b)

    def test_outputsize_multiplier_head(self):
        net = _tiny_net(n_series=2, outputsize_multiplier=3)
        out = net(_rand(2, 12, 2), deterministic=True)
        assert out.shape == (2, 4, 6)                          # N * mult, distr_output quirk

    def test_param_count_pin(self):
        # Live trainable graph at the fixed tiny config (CD mode), counted once by
        # hand from the reference layer inventory. Guards silent structure drift.
        net = _tiny_net()
        n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(net, nnx.Param)))
        L, h, d, dff, N, layers, scales = 12, 4, 8, 8, 1, 2, 2
        widths = [12, 6]
        emb = N * 3 * d                                          # token conv, no bias
        revin = scales * 2 * N
        season = (widths[0] * widths[1] + widths[1]) + (widths[1] * widths[1] + widths[1])
        trend = (widths[1] * widths[0] + widths[0]) + (widths[0] * widths[0] + widths[0])
        cross = 2 * (d * dff + dff + dff * d + d) + 2 * d       # cross+out_cross(+LN dead)
        pdm = layers * (season + trend + cross)
        predict = sum(w * h + h for w in widths)
        proj = d * N + N
        out_res = sum(w * w + w for w in widths)
        regression = sum(w * h + h for w in widths)
        expected = emb + revin + pdm + predict + proj + out_res + regression
        assert n_params == expected


# =============================================================================
# Training
# =============================================================================
from chronax.models.timemixer.timemixer_training import (  # noqa: E402
    build_windows, forward_loss, predict_step, train,
)


def _make_y(T=60):
    t = np.arange(T)
    return jnp.asarray(10 + 0.3 * t + 2 * np.sin(t / 3.0), jnp.float32)


class TestTraining:
    def test_build_windows_padding_semantics(self):
        y = jnp.arange(1.0, 11.0)
        w, m = build_windows(y, input_size=4, h=2)
        assert w.shape == (6, 6) and m.shape == (6, 2)
        np.testing.assert_array_equal(np.asarray(w[0]), [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(np.asarray(w[5]), [6, 7, 8, 9, 10, 0])
        np.testing.assert_array_equal(np.asarray(m[5]), [1, 0])

    def test_masked_loss_ignores_padded_tail(self):
        net = _tiny_net()
        y = _make_y(20)
        w, m = build_windows(y, 12, 4)
        base = float(forward_loss(net, w, m, h=4, input_size=12, loss_fn=mae))
        w_poison = np.array(w)
        tgt = w_poison[:, 12:]
        tgt[np.array(m) == 0.0] = 1e6          # only padded target cells differ
        w_poison[:, 12:] = tgt
        w2 = jnp.asarray(w_poison)
        # Poison lands only where mask==0 within the target block.
        assert float(forward_loss(net, w2, m, h=4, input_size=12, loss_fn=mae)) == pytest.approx(base, rel=1e-6)

    def test_train_is_vmap_traceable(self):
        def run(seed):
            net = _tiny_net()
            return train(net, _make_y(), h=4, input_size=12, max_steps=3,
                         windows_batch_size=8, lr=1e-3, seed=seed, loss_fn=mae)
        out = jax.vmap(run)(jnp.arange(2))
        assert out.shape == (2, 3) and bool(jnp.all(jnp.isfinite(out)))

    def test_train_updates_params_and_losses_finite(self):
        net = _tiny_net()
        before = np.asarray(net.projection_layer.kernel.value).copy()
        losses = train(net, _make_y(), h=4, input_size=12, max_steps=10,
                       windows_batch_size=8, lr=1e-3, seed=0, loss_fn=mae)
        assert bool(jnp.all(jnp.isfinite(losses)))
        assert not np.allclose(before, np.asarray(net.projection_layer.kernel.value))

    def test_predict_step_shape_and_idempotent(self):
        net = _tiny_net()
        y = _make_y()
        p1 = predict_step(net, y, h=4, input_size=12)
        p2 = predict_step(net, y, h=4, input_size=12)
        assert p1.shape == (4, 1)
        np.testing.assert_array_equal(np.asarray(p1), np.asarray(p2))


# =============================================================================
# Model (BaseForecaster conformance)
# =============================================================================
from chronax.models.timemixer.timemixer_model import TimeMixer  # noqa: E402
from chronax.utils import ConformalIntervals  # noqa: E402


def _tiny(**kw):
    base = dict(h=4, input_size=12, d_model=8, d_ff=8, e_layers=1, moving_avg=5,
                max_steps=20, windows_batch_size=8, random_seed=0)
    base.update(kw)
    return TimeMixer(**base)


class TestModel:
    def test_fit_predict_shapes(self):
        m = _tiny().fit(_make_y())
        out = m.predict(h=4)
        assert set(out) == {"mean"} and out["mean"].shape == (4,)
        assert bool(jnp.all(jnp.isfinite(out["mean"])))

    def test_seed_determinism(self):
        y = _make_y()
        p1 = np.asarray(_tiny().fit(y).predict(h=4)["mean"])
        p2 = np.asarray(_tiny().fit(y).predict(h=4)["mean"])
        np.testing.assert_array_equal(p1, p2)

    def test_h_bounds(self):
        m = _tiny().fit(_make_y())
        assert m.predict(h=2)["mean"].shape == (2,)
        with pytest.raises(ValueError, match="h"):
            m.predict(h=9)
        with pytest.raises(ValueError, match="positive"):
            m.predict(h=0)

    def test_beats_naive(self):
        t = np.arange(140)
        y = 10 + 0.5 * t + 4 * np.sin(2 * np.pi * t / 12)
        y_train, y_test = jnp.asarray(y[:128], jnp.float32), y[128:140][:8]
        m = TimeMixer(h=8, input_size=24, d_model=16, d_ff=16, e_layers=2,
                      max_steps=300, windows_batch_size=32, random_seed=0).fit(y_train)
        pred = np.asarray(m.predict(h=8)["mean"])
        mae_model = np.mean(np.abs(pred - y_test))
        mae_naive = np.mean(np.abs(float(y_train[-1]) - y_test))
        assert mae_model < mae_naive

    def test_level_requires_conformal_params(self):
        m = _tiny().fit(_make_y())
        with pytest.raises(ValueError, match="conformal_params"):
            m.predict(h=4, level=[80])

    def test_conformal_level_path(self):
        m = _tiny(max_steps=3)
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_make_y())
        out = m.predict(h=4, level=[80])
        assert {"mean", "lo-80", "hi-80"} <= set(out)
        assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))

    def test_predict_level_twice_then_pickle(self):
        m = _tiny(max_steps=3)
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_make_y())
        o1 = m.predict(h=4, level=[80])
        m.predict(h=4, level=[80])
        m2 = pickle.loads(pickle.dumps(m))
        o3 = m2.predict(h=4, level=[80])
        np.testing.assert_array_equal(np.asarray(o1["mean"]), np.asarray(o3["mean"]))

    def test_pickle_roundtrip(self):
        m = _tiny().fit(_make_y())
        p1 = np.asarray(m.predict(h=4)["mean"])
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(p1, np.asarray(m2.predict(h=4)["mean"]))

    def test_forecast_matches_fit_predict(self):
        y = _make_y()
        o1 = _tiny().forecast(y, h=4)
        o2 = _tiny().fit(y).predict(h=4)
        np.testing.assert_array_equal(np.asarray(o1["mean"]), np.asarray(o2["mean"]))

    def test_forecast_fitted(self):
        y = _make_y()
        out = _tiny().forecast(y, h=4, fitted=True)
        assert out["fitted"].shape == y.shape
        assert bool(jnp.all(jnp.isnan(out["fitted"][:12])))
        assert bool(jnp.all(jnp.isfinite(out["fitted"][12:])))

    def test_rejects_2d_and_exog_and_short(self):
        with pytest.raises(ValueError, match="1-D"):
            _tiny().fit(jnp.ones((30, 2)))
        with pytest.raises(NotImplementedError, match="exog"):
            _tiny().fit(_make_y(), X=jnp.ones((60, 2)))
        with pytest.raises(ValueError, match="too short"):
            _tiny().fit(jnp.ones((12,)))

    def test_decoder_multiplier_guard(self):
        with pytest.raises(ValueError, match="decoder_input_size_multiplier"):
            _tiny(decoder_input_size_multiplier=1.5)

    def test_input_size_divisibility_guard_at_ctor(self):
        with pytest.raises(ValueError, match="divisible"):
            _tiny(input_size=13)

    def test_default_ctor_matches_nf_defaults(self):
        m = TimeMixer(h=4, input_size=12)
        assert (m.d_model, m.d_ff, m.e_layers, m.top_k) == (32, 32, 4, 5)
        assert (m.decomp_method, m.moving_avg, m.channel_independence) == ("moving_avg", 25, 0)
        assert (m.down_sampling_layers, m.down_sampling_window, m.down_sampling_method) == (1, 2, "avg")
        assert (m.use_norm, m.max_steps, m.learning_rate) == (True, 1000, 1e-3)
        assert (m.windows_batch_size, m.random_seed, m.loss) == (32, 1, "mae")

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
# Box-Cox
# =============================================================================
from chronax.models.timemixer.timemixer_model import _boxcox, _inv_boxcox  # noqa: E402


class TestBoxCox:
    @pytest.mark.parametrize("lam", [-0.5, 0.0, 0.5, 1.0])
    def test_roundtrip(self, lam):
        y = jnp.asarray([0.5, 1.0, 5.0, 20.0])
        np.testing.assert_allclose(np.asarray(_inv_boxcox(_boxcox(y, lam), lam)),
                                   np.asarray(y), rtol=1e-5)

    def test_off_is_plain(self):
        y = _make_y()
        p_off = np.asarray(_tiny(max_steps=5).fit(y).predict(h=4)["mean"])
        m = _tiny(max_steps=5)
        m.use_boxcox = False
        np.testing.assert_array_equal(p_off, np.asarray(m.fit(y).predict(h=4)["mean"]))

    def test_on_finite_and_sets_lambda(self):
        m = _tiny(max_steps=5, use_boxcox=True).fit(_make_y())
        assert m._bc_lambda is not None
        assert bool(jnp.all(jnp.isfinite(m.predict(h=4)["mean"])))

    def test_positivity_guard(self):
        y = np.array(_make_y())
        y[5] = -1.0
        with pytest.raises(ValueError, match="strictly positive"):
            _tiny(use_boxcox=True).fit(jnp.asarray(y))

    def test_pickle_preserves_lambda(self):
        m = _tiny(max_steps=5, use_boxcox=True).fit(_make_y())
        m2 = pickle.loads(pickle.dumps(m))
        assert m2._bc_lambda == m._bc_lambda
        np.testing.assert_array_equal(np.asarray(m.predict(h=4)["mean"]),
                                      np.asarray(m2.predict(h=4)["mean"]))


class TestBoxCoxConformal:
    def test_boxcox_conformal_level_twice_then_pickle(self):
        m = _tiny(max_steps=3, use_boxcox=True)
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_make_y())
        o1 = m.predict(h=4, level=[80])
        assert {"mean", "lo-80", "hi-80"} <= set(o1)
        assert bool(jnp.all(o1["lo-80"] <= o1["hi-80"]))
        m.predict(h=4, level=[80])
        m2 = pickle.loads(pickle.dumps(m))
        o3 = m2.predict(h=4, level=[80])
        np.testing.assert_array_equal(np.asarray(o1["mean"]), np.asarray(o3["mean"]))

    def test_boxcox_conformity_scores_unfitted_raises(self):
        m = _tiny(use_boxcox=True)
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        with pytest.raises(ValueError, match="lambda"):
            m.conformity_scores(_make_y())

    def test_boxcox_refit_reselects_lambda(self):
        t = np.arange(60)
        y_exp = jnp.asarray(np.exp(0.05 * t) + 1.0, jnp.float32)
        y_lin = jnp.asarray(10.0 + 0.5 * t, jnp.float32)
        m = _tiny(max_steps=3, use_boxcox=True)
        lam_exp = m.fit(y_exp)._bc_lambda
        lam_lin = m.fit(y_lin)._bc_lambda
        assert lam_exp != lam_lin

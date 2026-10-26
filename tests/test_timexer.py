"""Tests for the TimeXer forecaster package (chronax.models.timexer).

Sections: Losses -> Module -> Training -> Model -> Box-Cox. Reference
behaviors follow neuralforecast 3.1.7's ``models/timexer.py`` +
``common/_modules.py`` (EnEmbedding / DataEmbedding_inverted / FullAttention /
FlattenHead, non-stationary normalization).
"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.timexer.timexer_losses import LOSSES, mae, mse, resolve
from chronax.models.timexer.timexer_module import (
    AttentionLayer,
    DataEmbeddingInverted,
    EnEmbedding,
    TimeXerNet,
    _positional_embedding,
)

# =============================================================================
# Losses
# =============================================================================


class TestLosses:
    def test_values_and_resolve(self):
        p = jnp.asarray([1.0, 2.0])
        t = jnp.asarray([2.0, 4.0])
        assert float(mae(p, t)) == pytest.approx(1.5)
        assert float(mse(p, t)) == pytest.approx(2.5)
        assert resolve("mae") is mae
        with pytest.raises(ValueError, match="Unknown loss"):
            resolve("nope")
        assert set(LOSSES) == {"mae", "mse", "huber"}


# =============================================================================
# Module
# =============================================================================


def _rand(*shape, seed=0, scale=1.0):
    return jnp.asarray(np.random.RandomState(seed).randn(*shape) * scale, jnp.float32)


class TestPositionalEmbedding:
    def test_formula(self):
        pe = np.asarray(_positional_embedding(5, 8))[0]
        pos = np.arange(5)[:, None]
        div = np.exp(np.arange(0, 8, 2) * -(np.log(10000.0) / 8))
        np.testing.assert_allclose(pe[:, 0::2], np.sin(pos * div), rtol=1e-6)
        np.testing.assert_allclose(pe[:, 1::2], np.cos(pos * div), rtol=1e-6)


class TestEnEmbedding:
    def test_patching_and_glb(self):
        emb = EnEmbedding(n_vars=2, d_model=8, patch_len=4, dropout=0.0, rngs=nnx.Rngs(0))
        x = _rand(3, 2, 12)                                  # [B, N, L]
        out, n_vars = emb(x, deterministic=True)
        assert n_vars == 2 and out.shape == (6, 4, 8)        # [B*N, pn+1, d], pn=3

    def test_patch_values_match_manual_unfold(self):
        emb = EnEmbedding(n_vars=1, d_model=4, patch_len=3, dropout=0.0, rngs=nnx.Rngs(0))
        x = _rand(1, 1, 7)                                   # L=7, p=3 -> pn=2, tail dropped
        out, _ = emb(x, deterministic=True)
        assert out.shape == (1, 3, 4)                        # 2 patches + glb
        W = np.asarray(emb.value_embedding.kernel.value)     # [p, d]
        xa = np.asarray(x)[0, 0]
        pe = np.asarray(_positional_embedding(2, 4))[0]
        ref0 = xa[0:3] @ W + pe[0]
        ref1 = xa[3:6] @ W + pe[1]
        np.testing.assert_allclose(np.asarray(out[0, 0]), ref0, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(out[0, 1]), ref1, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(out[0, 2]), np.asarray(emb.glb_token.value)[0, 0, 0],
                                   rtol=1e-6)                # glb appended LAST


class TestDataEmbeddingInverted:
    def test_variate_tokens(self):
        emb = DataEmbeddingInverted(c_in=12, d_model=8, dropout=0.0, rngs=nnx.Rngs(0))
        x = _rand(2, 12, 3)                                  # [B, L, N]
        out = emb(x, deterministic=True)
        assert out.shape == (2, 3, 8)                        # one token per variate
        W = np.asarray(emb.value_embedding.kernel.value)
        b = np.asarray(emb.value_embedding.bias.value)
        ref = np.asarray(x)[0].T @ W + b
        np.testing.assert_allclose(np.asarray(out[0]), ref, rtol=1e-5)

    def test_x_mark_appends_covariate_tokens(self):
        # Covariates concat AFTER the endogenous variates (reference order):
        # N variate tokens then X covariate tokens, all through the shared Linear.
        emb = DataEmbeddingInverted(c_in=12, d_model=8, dropout=0.0, rngs=nnx.Rngs(0))
        x = _rand(2, 12, 1)                                  # [B, L, N=1]
        xm = _rand(2, 12, 2)                                 # [B, L, X=2]
        out = emb(x, x_mark=xm, deterministic=True)
        assert out.shape == (2, 3, 8)                        # 1 + 2 tokens
        # First token == endogenous-only; the two after == covariate tokens.
        endo = emb(x, deterministic=True)
        np.testing.assert_allclose(np.asarray(out[:, :1]), np.asarray(endo), rtol=1e-5)


class TestAttention:
    def test_self_attention_matches_manual(self):
        att = AttentionLayer(hidden_size=8, n_heads=2, attn_dropout=0.0, rngs=nnx.Rngs(0))
        x = _rand(2, 5, 8)
        got = np.asarray(att(x, x, deterministic=True))
        q = np.asarray(att.w_q(x)).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
        k = np.asarray(att.w_k(x)).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
        v = np.asarray(att.w_v(x)).reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
        s = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(4.0)
        w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
        ctx = np.einsum("bhqk,bhkd->bhqd", w, v).transpose(0, 2, 1, 3).reshape(2, 5, 8)
        ref = np.asarray(att.w_o(jnp.asarray(ctx)))
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)

    def test_cross_attention_uses_kv_input(self):
        att = AttentionLayer(hidden_size=8, n_heads=2, attn_dropout=0.0, rngs=nnx.Rngs(0))
        q = _rand(1, 2, 8, seed=1)
        kv1 = _rand(1, 4, 8, seed=2)
        kv2 = _rand(1, 4, 8, seed=3)
        assert not np.allclose(np.asarray(att(q, kv1, deterministic=True)),
                               np.asarray(att(q, kv2, deterministic=True)))

    def test_head_divisibility_guard(self):
        with pytest.raises(ValueError, match="divisible"):
            AttentionLayer(hidden_size=9, n_heads=2, attn_dropout=0.0, rngs=nnx.Rngs(0))


def _tiny_net(**kw):
    cfg = dict(h=4, input_size=12, n_series=1, patch_len=4, hidden_size=8,
               n_heads=2, e_layers=1, d_ff=16, dropout=0.0, use_norm=True,
               outputsize_multiplier=1)
    cfg.update(kw)
    return TimeXerNet(rngs=nnx.Rngs(0), **cfg)


class TestTimeXerNet:
    @pytest.mark.parametrize("n", [1, 3])
    @pytest.mark.parametrize("use_norm", [True, False])
    def test_forward_shape(self, n, use_norm):
        net = _tiny_net(n_series=n, use_norm=use_norm)
        out = net(_rand(2, 12, n), deterministic=True)
        assert out.shape == (2, 4, n) and bool(jnp.all(jnp.isfinite(out)))

    def test_cross_context_moves_only_glb_token_in_encoder_layer(self):
        # Inside one encoder layer, changing the cross context may only change
        # the LAST (global) token's output row — patch tokens see cross only
        # through subsequent layers.
        net = _tiny_net(n_series=2)
        layer = net.layers[0]
        x = _rand(4, 4, 8, seed=1)                            # [B*N, pn+1, d]
        c1 = _rand(2, 2, 8, seed=2)                           # [B, N, d]
        c2 = _rand(2, 2, 8, seed=3)
        o1 = np.asarray(layer(x, c1, n_vars=2, deterministic=True))
        o2 = np.asarray(layer(x, c2, n_vars=2, deterministic=True))
        np.testing.assert_allclose(o1[:, :-1], o2[:, :-1], atol=1e-6)
        assert not np.allclose(o1[:, -1], o2[:, -1])

    def test_ns_norm_scale_invariance_property(self):
        # With use_norm the network's normalized core sees identical inputs for
        # y and a*y+b; outputs differ only through the denorm affine.
        net = _tiny_net()
        y = _rand(1, 12, 1, seed=4)
        o1 = np.asarray(net(y, deterministic=True))
        o2 = np.asarray(net(y * 3.0 + 10.0, deterministic=True))
        np.testing.assert_allclose(o2, o1 * 3.0 + 10.0, rtol=1e-3, atol=1e-3)

    def test_input_dependence_n3(self):
        net = _tiny_net(n_series=3)
        a = np.asarray(net(_rand(1, 12, 3, seed=1), deterministic=True))
        b = np.asarray(net(_rand(1, 12, 3, seed=2), deterministic=True))
        assert not np.allclose(a, b)

    def test_outputsize_multiplier_head(self):
        net = _tiny_net(outputsize_multiplier=3)
        out = net(_rand(2, 12, 1), deterministic=True)
        assert out.shape == (2, 4, 3)

    def test_patch_len_guard(self):
        with pytest.raises(ValueError, match="patch_len"):
            _tiny_net(input_size=3, patch_len=4)

    def test_dropout_train_vs_eval(self):
        net = _tiny_net(dropout=0.5)
        x = _rand(2, 12, 1)
        e1 = np.asarray(net(x, deterministic=True))
        e2 = np.asarray(net(x, deterministic=True))
        t1 = np.asarray(net(x, deterministic=False))
        np.testing.assert_array_equal(e1, e2)
        assert not np.allclose(e1, t1)

    def test_param_count_pin(self):
        # Fixed tiny config, counted by hand from the reference inventory.
        net = _tiny_net()
        n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(net, nnx.Param)))
        d, dff, p, pn, L, h, N, heads, layers = 8, 16, 4, 3, 12, 4, 1, 2, 1
        en = p * d + N * d                                   # value Linear (no bias) + glb token
        ex = L * d + d                                       # inverted embedding Linear
        attn = 2 * (4 * (d * d + d))                         # self + cross QKVO
        ffn = d * dff + dff + dff * d + d
        norms = 3 * 2 * d
        enc = layers * (attn + ffn + norms) + 2 * d          # + final LayerNorm
        head = (d * (pn + 1)) * h + h
        assert n_params == en + ex + enc + head


# =============================================================================
# Training
# =============================================================================
from chronax.models.timexer.timexer_training import (  # noqa: E402
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
        np.testing.assert_array_equal(np.asarray(w[5]), [6, 7, 8, 9, 10, 0])
        np.testing.assert_array_equal(np.asarray(m[5]), [1, 0])

    def test_masked_loss_ignores_padded_tail(self):
        net = _tiny_net()
        y = _make_y(20)
        w, m = build_windows(y, 12, 4)
        base = float(forward_loss(net, w, m, h=4, input_size=12, loss_fn=mae))
        w_poison = np.array(w)
        tgt = w_poison[:, 12:]
        tgt[np.array(m) == 0.0] = 1e6
        w_poison[:, 12:] = tgt
        assert float(forward_loss(net, jnp.asarray(w_poison), m, h=4, input_size=12,
                                  loss_fn=mae)) == pytest.approx(base, rel=1e-6)

    def test_train_is_vmap_traceable(self):
        def run(seed):
            net = _tiny_net()
            return train(net, _make_y(), h=4, input_size=12, max_steps=3,
                         windows_batch_size=8, lr=1e-3, seed=seed, loss_fn=mae)
        out = jax.vmap(run)(jnp.arange(2))
        assert out.shape == (2, 3) and bool(jnp.all(jnp.isfinite(out)))

    def test_train_with_hist_exog_is_vmap_traceable(self):
        hist = jnp.asarray(np.random.RandomState(0).randn(60, 2), jnp.float32)
        def run(seed):
            net = _tiny_net(hist_exog_size=2)
            return train(net, _make_y(), h=4, input_size=12, max_steps=3,
                         windows_batch_size=8, lr=1e-3, seed=seed, loss_fn=mae,
                         hist_exog=hist)
        out = jax.vmap(run)(jnp.arange(2))
        assert out.shape == (2, 3) and bool(jnp.all(jnp.isfinite(out)))

    def test_train_updates_params(self):
        net = _tiny_net()
        before = np.asarray(net.head.kernel.value).copy()
        losses = train(net, _make_y(), h=4, input_size=12, max_steps=10,
                       windows_batch_size=8, lr=1e-3, seed=0, loss_fn=mae)
        assert bool(jnp.all(jnp.isfinite(losses)))
        assert not np.allclose(before, np.asarray(net.head.kernel.value))

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
from chronax.models.timexer.timexer_model import TimeXer  # noqa: E402
from chronax.utils import ConformalIntervals  # noqa: E402


def _tiny(**kw):
    base = dict(h=4, input_size=12, patch_len=4, hidden_size=8, n_heads=2,
                e_layers=1, d_ff=16, max_steps=20, windows_batch_size=8,
                random_seed=0)
    base.update(kw)
    return TimeXer(**base)


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

    def test_beats_naive(self):
        t = np.arange(140)
        y = 10 + 0.5 * t + 4 * np.sin(2 * np.pi * t / 12)
        y_train, y_test = jnp.asarray(y[:128], jnp.float32), y[128:140][:8]
        m = TimeXer(h=8, input_size=24, patch_len=8, hidden_size=16, n_heads=2,
                    e_layers=1, d_ff=32, max_steps=300, windows_batch_size=32,
                    random_seed=0).fit(y_train)
        pred = np.asarray(m.predict(h=8)["mean"])
        mae_model = np.mean(np.abs(pred - y_test))
        mae_naive = np.mean(np.abs(float(y_train[-1]) - y_test))
        assert mae_model < mae_naive

    def test_conformal_level_path_and_twice_then_pickle(self):
        m = _tiny(max_steps=3)
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_make_y())
        o1 = m.predict(h=4, level=[80])
        assert {"mean", "lo-80", "hi-80"} <= set(o1)
        assert bool(jnp.all(o1["lo-80"] <= o1["hi-80"]))
        m.predict(h=4, level=[80])
        m2 = pickle.loads(pickle.dumps(m))
        o3 = m2.predict(h=4, level=[80])
        np.testing.assert_array_equal(np.asarray(o1["mean"]), np.asarray(o3["mean"]))

    def test_level_requires_conformal_params(self):
        m = _tiny().fit(_make_y())
        with pytest.raises(ValueError, match="conformal_params"):
            m.predict(h=4, level=[80])

    def test_pickle_roundtrip(self):
        m = _tiny().fit(_make_y())
        p1 = np.asarray(m.predict(h=4)["mean"])
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(p1, np.asarray(m2.predict(h=4)["mean"]))

    def test_forecast_matches_fit_predict_and_fitted(self):
        y = _make_y()
        o1 = _tiny().forecast(y, h=4, fitted=True)
        o2 = _tiny().fit(y).predict(h=4)
        np.testing.assert_array_equal(np.asarray(o1["mean"]), np.asarray(o2["mean"]))
        assert o1["fitted"].shape == y.shape
        assert bool(jnp.all(jnp.isnan(o1["fitted"][:12])))

    def test_rejects_2d_y_short_and_patch_guard(self):
        with pytest.raises(ValueError, match="1-D"):
            _tiny().fit(jnp.ones((30, 2)))
        with pytest.raises(ValueError, match="too short"):
            _tiny().fit(jnp.ones((12,)))
        with pytest.raises(ValueError, match="patch_len"):
            TimeXer(h=4, input_size=3, patch_len=4)

    def test_hist_exog_fit_predict(self):
        # Historical exog (X): raw covariate variate tokens in the cross context;
        # predict needs no future X (historical-only). Same net params (the
        # ex_embedding Linear is shared), so hist widens tokens, not weights.
        X = jnp.asarray(np.random.RandomState(1).randn(60, 2), jnp.float32)
        m = _tiny().fit(_make_y(60), X=X)
        assert m._hist_size == 2
        out = m.predict(h=4)
        assert out["mean"].shape == (4,) and bool(jnp.all(jnp.isfinite(out["mean"])))

    def test_hist_exog_misaligned_raises(self):
        X = jnp.asarray(np.random.RandomState(1).randn(50, 2), jnp.float32)
        with pytest.raises(ValueError, match="historical exog.*align"):
            _tiny().fit(_make_y(60), X=X)

    def test_hist_exog_forecast_threads_X_and_rejects_futr(self):
        X = jnp.asarray(np.random.RandomState(1).randn(60, 2), jnp.float32)
        assert _tiny().forecast(_make_y(60), 4, X=X)["mean"].shape == (4,)
        with pytest.raises(NotImplementedError, match="future-known"):
            _tiny().forecast(_make_y(60), 4, X_future=X[:4])

    def test_hist_exog_conformal_refused(self):
        # TimeXer has no native interval head, so historical exog leaves no
        # interval path — conformal is refused at predict and conformity_scores.
        X = jnp.asarray(np.random.RandomState(1).randn(60, 2), jnp.float32)
        m = _tiny().fit(_make_y(60), X=X)
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        with pytest.raises(ValueError, match="historical exog"):
            m.predict(h=4, level=[80])
        with pytest.raises(ValueError, match="historical exog"):
            _tiny().conformity_scores(_make_y(60), X=X)

    def test_hist_exog_pickle_roundtrip(self):
        X = jnp.asarray(np.random.RandomState(1).randn(60, 2), jnp.float32)
        m = _tiny().fit(_make_y(60), X=X)
        o1 = m.predict(h=4)["mean"]
        o2 = pickle.loads(pickle.dumps(m)).predict(h=4)["mean"]
        np.testing.assert_array_equal(np.asarray(o1), np.asarray(o2))

    def test_default_ctor_matches_nf_defaults(self):
        m = TimeXer(h=4, input_size=16)
        assert (m.patch_len, m.hidden_size, m.n_heads, m.e_layers, m.d_ff) == (16, 512, 8, 2, 2048)
        assert (m.use_norm, m.max_steps, m.learning_rate) == (True, 1000, 1e-3)
        assert (m.windows_batch_size, m.random_seed, m.loss) == (32, 1, "mae")

    def test_no_recompile_on_second_predict_and_refit(self):
        import logging
        m = _tiny(max_steps=3).fit(_make_y())
        m.predict(h=4)
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
                m.predict(h=4)
                n_repeat = len(counts)
                m2 = _tiny(max_steps=3).fit(_make_y())
                n_after_fit = len(counts)
                m2.predict(h=4)
                n_refit_predict = len(counts) - n_after_fit
            finally:
                for lg in loggers:
                    lg.removeHandler(handler)
        assert n_repeat == 0
        assert n_refit_predict == 0


# =============================================================================
# Box-Cox
# =============================================================================
from chronax.models.timexer.timexer_model import _boxcox, _inv_boxcox  # noqa: E402


class TestBoxCox:
    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
    def test_roundtrip(self, lam):
        y = jnp.asarray([0.5, 1.0, 5.0, 20.0])
        np.testing.assert_allclose(np.asarray(_inv_boxcox(_boxcox(y, lam), lam)),
                                   np.asarray(y), rtol=1e-5)

    def test_on_finite_and_off_is_plain(self):
        y = _make_y()
        m_on = _tiny(max_steps=5, use_boxcox=True).fit(y)
        assert m_on._bc_lambda is not None
        assert bool(jnp.all(jnp.isfinite(m_on.predict(h=4)["mean"])))
        p_off1 = np.asarray(_tiny(max_steps=5).fit(y).predict(h=4)["mean"])
        p_off2 = np.asarray(_tiny(max_steps=5).fit(y).predict(h=4)["mean"])
        np.testing.assert_array_equal(p_off1, p_off2)

    def test_positivity_guard(self):
        y = np.array(_make_y())
        y[5] = -1.0
        with pytest.raises(ValueError, match="strictly positive"):
            _tiny(use_boxcox=True).fit(jnp.asarray(y))


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
            return x * 9.0 - 3.0
        return f(jnp.ones((2, 3)))

    _, n = _count_compiles(fresh)
    assert n >= 1


def test_refit_does_not_recompile():
    # The training program is a module-level nnx.jit keyed on config graphdefs
    # (value-__eq__ initializers and the memoized optax transform) and operand
    # shapes — a refit AND a fresh same-config instance must hit the cache with
    # zero XLA compiles; fit #2 is counted directly (no uncounted settle call).
    y = jnp.asarray(np.sin(np.arange(40) / 5.0), jnp.float32)
    kw = dict(h=4, input_size=16, patch_len=8, max_steps=3,
              windows_batch_size=4, random_seed=0)
    m = TimeXer(**kw)
    m.fit(y)                                     # first fit pays the one compile
    _, n_refit = _count_compiles(lambda: (m.fit(y), None)[1])
    assert n_refit == 0
    m2 = TimeXer(**kw)
    _, n_fresh = _count_compiles(lambda: (m2.fit(y), None)[1])
    assert n_fresh == 0

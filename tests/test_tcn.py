"""Tests for chronax.models.TCN.

Covers the TCN subpackage in banner sections (Losses, Scaler, Module, Training,
Model, Namespace), matching the ``test_<model>.py`` convention used elsewhere
in ``tests/``.
"""
import io
import logging
import pickle
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx


def _make_y(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return jnp.asarray(np.sin(np.arange(n) / 5.0) + 0.1 * rng.randn(n), dtype=jnp.float32)


# === Losses ===

from chronax.models.tcn.tcn_losses import (  # noqa: E402
    MultiQuantileLoss, huber, mae, mse, outputsize_multiplier, resolve,
)


def test_mae_zero_at_match():
    x = jnp.array([1.0, 2.0, 3.0])
    assert float(mae(x, x)) == 0.0


def test_mse_known_value():
    assert float(mse(jnp.array([0.0, 0.0]), jnp.array([1.0, 3.0]))) == pytest.approx(5.0)


def test_huber_regions():
    assert float(huber(jnp.array([0.0]), jnp.array([0.5]))) == pytest.approx(0.125)
    assert float(huber(jnp.array([0.0]), jnp.array([3.0]))) == pytest.approx(2.5)


def test_resolve_string_and_multiplier():
    assert resolve("mae") is mae
    assert outputsize_multiplier(mae) == 1
    with pytest.raises(ValueError, match="Unknown loss"):
        resolve("nope")


def test_mqloss_multiplier_median_and_pickle():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    assert loss.outputsize_multiplier == 3
    with pytest.raises(ValueError, match="0.5"):
        MultiQuantileLoss((0.1, 0.9))
    with pytest.raises(ValueError, match="between 0 and 1"):
        MultiQuantileLoss((0.0, 0.5, 1.0))
    assert pickle.loads(pickle.dumps(loss)).quantiles == loss.quantiles


def test_mqloss_pinball_value():
    # err = y - yhat; pred zeros, target 2 -> sum_q q*2 = sum{0.2,1.0,1.8} = 3.0 —
    # NF's effective reduction (its 1/len(quantiles) factor is dead), matching
    # the trainer's masked inline branch.
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    assert float(loss(jnp.zeros((1, 1, 3)), jnp.array([[2.0]]))) == pytest.approx(3.0)


# === Scaler ===

from chronax.models.tcn.tcn_scaler import (  # noqa: E402
    IdentityScaler, RobustScaler, resolve_scaler,
)


def test_robust_scaler_round_trip():
    x = jnp.asarray(np.random.RandomState(0).randn(3, 20), dtype=jnp.float32)
    sc = RobustScaler()
    shift, scale = sc.stats(x, axis=1)
    z = sc.transform(x, shift, scale)
    np.testing.assert_allclose(np.asarray(sc.inverse(z, shift, scale)), np.asarray(x), rtol=1e-4, atol=1e-4)


def test_robust_scaler_uses_median():
    x = jnp.asarray([[1.0, 2.0, 3.0, 4.0, 100.0]], dtype=jnp.float32)
    shift, _ = RobustScaler().stats(x, axis=1)
    assert float(shift[0, 0]) == pytest.approx(3.0)


def test_resolve_scaler():
    assert isinstance(resolve_scaler("robust"), RobustScaler)
    assert isinstance(resolve_scaler("identity"), IdentityScaler)
    with pytest.raises(ValueError):
        resolve_scaler("minmax")


# === Module ===

from chronax.models.tcn.tcn_module import (  # noqa: E402
    MLP, CausalConv1d, TCNNet, TemporalConvolutionEncoder,
)


def _numpy_causal_conv(x, w, b, dilation):
    """Reference dilated causal cross-correlation: out[t] = b + sum_i w[i] * x[t-(K-1-i)*d]."""
    K = w.shape[0]
    out = np.full_like(x, b)
    for t in range(x.shape[0]):
        for i in range(K):
            src = t - (K - 1 - i) * dilation
            if src >= 0:
                out[t] += w[i] * x[src]
    return out


def test_causal_conv_matches_numpy_reference():
    K, d = 3, 4
    conv = CausalConv1d(1, 1, K, padding=(K - 1) * d, dilation=d, activation="ReLU",
                        rngs=nnx.Rngs(0))
    w = np.asarray(conv.weight.value)[0, 0]         # [K]
    b = float(np.asarray(conv.bias.value)[0])
    x = np.random.RandomState(1).randn(30).astype(np.float32)
    expected = np.maximum(_numpy_causal_conv(x, w, b, d), 0.0)  # + ReLU
    got = np.asarray(conv(jnp.asarray(x)[None, None, :]))[0, 0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_encoder_is_causal():
    # Perturbing the input at time t must not change encoder outputs before t.
    enc = TemporalConvolutionEncoder(1, 8, 2, (1, 2, 4), "ReLU", rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(1, 32, 1), jnp.float32)
    t0 = 20
    x2 = x.at[0, t0, 0].add(100.0)
    a, b = np.asarray(enc(x)), np.asarray(enc(x2))
    np.testing.assert_array_equal(a[:, :t0], b[:, :t0])
    assert not np.allclose(a[:, t0:], b[:, t0:])


def test_encoder_channel_progression_and_shapes():
    enc = TemporalConvolutionEncoder(3, 16, 2, (1, 2), "Tanh", rngs=nnx.Rngs(0))
    out = enc(jnp.ones((2, 24, 3), jnp.float32))
    assert out.shape == (2, 24, 16)
    assert enc.layers[0].weight.value.shape == (16, 3, 2)
    assert enc.layers[1].weight.value.shape == (16, 16, 2)


def test_bad_activation_raises():
    with pytest.raises(ValueError, match="activation"):
        CausalConv1d(1, 4, 2, padding=1, dilation=1, activation="GELU", rngs=nnx.Rngs(0))


def test_mlp_single_layer_is_direct_projection():
    mlp = MLP(4, 2, hidden_size=8, num_layers=1, rngs=nnx.Rngs(0))
    assert len(mlp.layers) == 1
    assert mlp(jnp.ones((5, 4), jnp.float32)).shape == (5, 2)


def test_net_output_shape_and_param_count():
    h, L, C, K, Hd, nd = 6, 18, 8, 2, 4, 3
    dil = (1, 2, 4)
    net = TCNNet(h=h, input_size=L, kernel_size=K, dilations=dil,
                 encoder_hidden_size=C, encoder_activation="ReLU",
                 decoder_hidden_size=Hd, decoder_layers=2,
                 outputsize_multiplier=1, rngs=nnx.Rngs(0))
    out = net(jnp.ones((3, L, 1), jnp.float32))
    assert out.shape == (3, h, 1)
    # Analytic parameter count locks decoder-layer semantics and conv fan-ins:
    # encoder: (C*1*K + C) + (nd-1)*(C*C*K + C); adapter: L*h + h; decoder: C*Hd+Hd + Hd*1+1
    expected = (C * 1 * K + C) + (nd - 1) * (C * C * K + C) + (L * h + h) + (C * Hd + Hd) + (Hd * 1 + 1)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(nnx.state(net, nnx.Param)))
    assert n_params == expected


def test_net_futr_exog_channels():
    h, L, F = 4, 12, 2
    net = TCNNet(h=h, input_size=L, encoder_hidden_size=8, decoder_hidden_size=4,
                 futr_exog_size=F, outputsize_multiplier=1, rngs=nnx.Rngs(0))
    assert net.hist_encoder.layers[0].weight.value.shape[1] == 1 + F
    out = net(jnp.ones((2, L, 1), jnp.float32), futr_exog=jnp.ones((2, L + h, F), jnp.float32))
    assert out.shape == (2, h, 1)


# === Training ===

from chronax.models.tcn.tcn_training import (  # noqa: E402
    build_windows, predict_step, train,
)


def _net(h=12, L=36):
    return TCNNet(h=h, input_size=L, encoder_hidden_size=16, decoder_hidden_size=8,
                  rngs=nnx.Rngs(0))


def test_build_windows_nf_padding_semantics():
    # Series right-padded with h zeros; n = T - input_size windows; every window
    # with >=1 real target point is kept, padded tail masked out.
    y = jnp.arange(1.0, 11.0)                # T=10, values 1..10 (no zeros)
    w, m = build_windows(y, input_size=3, h=2)
    assert w.shape == (7, 5) and m.shape == (7, 2)
    np.testing.assert_array_equal(np.asarray(w[0]), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.asarray(m[0]), [1, 1])
    np.testing.assert_array_equal(np.asarray(w[6]), [7, 8, 9, 10, 0])   # padded tail
    np.testing.assert_array_equal(np.asarray(m[6]), [1, 0])             # masked out
    assert float(m.sum()) == 13.0            # 6 full windows * 2 + 1 partial target
    with pytest.raises(ValueError, match="too short"):
        build_windows(jnp.arange(3.0), input_size=3, h=2)


def test_exog_scaling_stats_use_insample_span_only():
    # Exog statistics come from the insample span only; the horizon slice is
    # transformed with those stats but never feeds them.
    from chronax.models.tcn.tcn_training import _scale_exog
    rng = np.random.RandomState(0)
    L, h, F = 8, 4, 2
    w = jnp.asarray(rng.randn(3, L + h, F), jnp.float32)
    a = _scale_exog(w, RobustScaler(), stats_len=L)
    w2 = w.at[:, L:, :].add(1e4)         # perturb horizon slice massively
    b = _scale_exog(w2, RobustScaler(), stats_len=L)
    np.testing.assert_array_equal(np.asarray(a[:, :L]), np.asarray(b[:, :L]))


def test_masked_loss_ignores_padded_tail():
    # A window whose padded target is masked must contribute only its real points.
    net = TCNNet(h=2, input_size=3, encoder_hidden_size=4, decoder_hidden_size=4,
                 dilations=(1,), rngs=nnx.Rngs(0))
    y = jnp.arange(1.0, 11.0)
    w, m = build_windows(y, input_size=3, h=2)
    from chronax.models.tcn.tcn_training import forward_loss
    full = forward_loss(net, w, m, h=2, input_size=3, scaler=IdentityScaler(), loss_fn=mae)
    # Poison the masked (padded) target cell: loss must not change.
    w_poison = w.at[6, 4].set(1e6)
    poisoned = forward_loss(net, w_poison, m, h=2, input_size=3, scaler=IdentityScaler(), loss_fn=mae)
    np.testing.assert_allclose(float(full), float(poisoned), rtol=0, atol=1e-6)


def test_train_reduces_loss():
    net = _net()
    losses = train(net, _make_y(), h=12, input_size=36, max_steps=60,
                   windows_batch_size=64, lr=1e-3, seed=0, loss_fn=mae, scaler=RobustScaler())
    assert losses.shape == (60,)
    assert float(jnp.mean(losses[-10:])) < float(jnp.mean(losses[:10]))


def test_train_divergence_guard():
    net = _net()
    with pytest.raises(RuntimeError, match="diverged"):
        train(net, _make_y(), h=12, input_size=36, max_steps=10,
              windows_batch_size=64, lr=1e9, seed=0, loss_fn=mae, scaler=RobustScaler())


def test_predict_step_repeatable():
    net = _net()
    y = _make_y()
    train(net, y, h=12, input_size=36, max_steps=5, windows_batch_size=64, lr=1e-3, seed=0,
          loss_fn=mae, scaler=RobustScaler())
    pred_a = predict_step(net, y[-36:], h=12, input_size=36, scaler=RobustScaler())
    pred_b = predict_step(net, y[-36:], h=12, input_size=36, scaler=RobustScaler())
    np.testing.assert_array_equal(np.asarray(pred_a), np.asarray(pred_b))


def test_train_is_vmap_traceable():
    # conformity_scores vmaps forecast over CV windows; the guard must no-op, not raise.
    y = _make_y()
    def run(seed):
        net = TCNNet(h=12, input_size=36, encoder_hidden_size=8, decoder_hidden_size=8,
                     rngs=nnx.Rngs(0))
        return train(net, y, h=12, input_size=36, max_steps=4, windows_batch_size=16,
                     lr=1e-3, seed=seed, loss_fn=resolve("mae"), scaler=resolve_scaler("identity"))
    out = jax.vmap(run)(jnp.arange(3))
    assert out.shape == (3, 4) and bool(jnp.all(jnp.isfinite(out)))


# === Model ===

from chronax.models.base_forecaster import BaseForecaster  # noqa: E402
from chronax.models.tcn.tcn_model import TCN  # noqa: E402
from chronax.utils import ConformalIntervals  # noqa: E402


def _tiny(**kw):
    base = dict(h=12, input_size=36, encoder_hidden_size=16, decoder_hidden_size=8,
                max_steps=20, windows_batch_size=64, random_seed=0)
    base.update(kw)
    return TCN(**base)


_COMPILE_LOG_RE = re.compile(r"Compiling ")


def _count_compiles(fn):
    """Run fn under jax.log_compiles and return (result, n_xla_compilations)."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    loggers = [
        logging.getLogger("jax._src.dispatch"),
        logging.getLogger("jax._src.interpreters.pxla"),
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
    # must report >=1. If a jax bump renames the "Compiling " message or the
    # logger paths, this fails instead of letting test_refit_does_not_recompile
    # go vacuously green on a counter that matches nothing.
    def fresh():
        @jax.jit
        def f(x):
            return x * 3.0 - 1.0
        return f(jnp.ones((2, 5)))

    _, n = _count_compiles(fresh)
    assert n >= 1


def test_refit_does_not_recompile():
    # The training program is a module-level nnx.jit whose cache keys on config
    # graphdefs (value-__eq__ initializers, _adam/scaler singletons) and operand
    # shapes — never on data or call-site closures. A refit AND a fresh
    # same-config instance must both hit the cache with zero XLA compiles;
    # fit #2 is counted directly (no uncounted settle call). No dropout-freshness
    # companion test — deliberately: the TCN forward is deterministic and the
    # only RNG (batch-index sampling) runs eagerly outside the jitted program.
    y = _make_y(60)
    m = _tiny(max_steps=3, windows_batch_size=4)
    m.fit(y)                                     # first fit pays the one compile
    _, n_refit = _count_compiles(lambda: m.fit(y)._train_y)
    assert n_refit == 0
    m2 = _tiny(max_steps=3, windows_batch_size=4)
    _, n_fresh = _count_compiles(lambda: m2.fit(y)._train_y)
    assert n_fresh == 0


def test_tcn_is_base_forecaster_and_uses_exog():
    assert issubclass(TCN, BaseForecaster)
    assert TCN(h=12).uses_exog is True


def test_nf_defaults():
    m = TCN(h=12)
    assert m.input_size == 36                       # -1 -> 3*h
    assert m.kernel_size == 2
    assert m.dilations == (1, 2, 4, 8, 16)
    assert m.encoder_hidden_size == 128
    assert m.decoder_hidden_size == 128
    assert m.decoder_layers == 2
    assert m.max_steps == 1000
    assert m.learning_rate == pytest.approx(1e-3)
    assert m.windows_batch_size == 128
    assert m.scaler_type == "robust"


def test_bad_encoder_activation_raises():
    with pytest.raises(ValueError, match="encoder_activation"):
        TCN(h=12, encoder_activation="GELU")


def test_context_size_accepted_but_inert():
    # context_size is accepted for API parity but unused, so predictions are identical.
    y = _make_y()
    a = _tiny(context_size=10).fit(y).predict(h=12)["mean"]
    b = _tiny(context_size=999).fit(y).predict(h=12)["mean"]
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_fit_predict_shapes():
    m = _tiny().fit(_make_y())
    assert m.model_ is not None
    assert m.predict(h=12)["mean"].shape == (12,)


def test_predict_smaller_h_slices_larger_raises():
    m = _tiny().fit(_make_y())
    assert m.predict(h=5)["mean"].shape == (5,)
    with pytest.raises(ValueError, match="unsupported|trained for"):
        m.predict(h=13)


def test_h_greater_than_input_size():
    # Context adapter Linear(L->h) natively handles h > input_size (no upsample path).
    m = TCN(h=12, input_size=8, encoder_hidden_size=8, decoder_hidden_size=8,
            max_steps=5, windows_batch_size=16, random_seed=0).fit(_make_y(80))
    pred = m.predict(h=12)["mean"]
    assert pred.shape == (12,)
    assert bool(jnp.all(jnp.isfinite(pred)))


def test_short_series_and_2d_raise():
    # Training needs T >= input_size+1 (h-padded partial windows), so T = input_size
    # exactly must raise and T = input_size+1 must fit.
    with pytest.raises(ValueError, match="too short|short"):
        _tiny().fit(_make_y(36))
    with pytest.raises(ValueError, match="1-D"):
        _tiny().fit(jnp.ones((200, 2), jnp.float32))


def test_fit_trains_from_input_size_plus_one():
    m = _tiny(max_steps=3).fit(_make_y(37))     # 1 partial window, target mask sum 1
    assert m.predict(h=12)["mean"].shape == (12,)


def test_torch_median_convention():
    # torch nanmedian returns the LOWER of the two middles on even lengths.
    x = jnp.asarray([[1.0, 2.0, 4.0, 8.0]])
    shift, _ = RobustScaler().stats(x, axis=1)
    assert float(shift[0, 0]) == 2.0            # jnp.median would give 3.0


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        _tiny().predict(h=12)


def test_fit_X_not_implemented():
    with pytest.raises(NotImplementedError, match="futr_exog"):
        _tiny().fit(_make_y(), X=jnp.ones((200, 1), jnp.float32))


def test_repeated_predict_identical():
    m = _tiny().fit(_make_y())
    a = np.asarray(m.predict(h=12)["mean"])
    b = np.asarray(m.predict(h=12)["mean"])
    np.testing.assert_array_equal(a, b)


def test_forecast_equals_fit_predict():
    y = _make_y()
    a = _tiny().forecast(y, h=12)["mean"]
    b = _tiny().fit(y).predict(h=12)["mean"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


def test_fitted_nan_head():
    res = _tiny().forecast(_make_y(), h=12, fitted=True)
    fitted = np.asarray(res["fitted"])
    assert fitted.shape == (200,)
    assert np.all(np.isnan(fitted[:36])) and np.all(np.isfinite(fitted[36:]))


def test_futr_exog_fit_predict_shapes():
    y = _make_y(240)
    futr = jnp.asarray(np.random.RandomState(1).randn(y.shape[0], 2), jnp.float32)
    m = _tiny(max_steps=10).fit(y, futr_exog=futr)
    out = m.predict(h=12, futr_exog=jnp.asarray(np.random.RandomState(2).randn(12, 2), jnp.float32))
    assert out["mean"].shape == (12,)
    assert bool(jnp.all(jnp.isfinite(out["mean"])))


def test_futr_required_at_predict_raises():
    y = _make_y(240)
    futr = jnp.asarray(np.random.RandomState(1).randn(y.shape[0], 1), jnp.float32)
    m = _tiny(max_steps=5).fit(y, futr_exog=futr)
    with pytest.raises(ValueError, match="futr_exog"):
        m.predict(h=12)


def test_conformal_interval_keys_twice_then_pickle():
    # predict(level) TWICE, then pickle, then predict again -- single-call conformal
    # tests are structurally blind to tracer pollution.
    m = TCN(h=4, input_size=12, encoder_hidden_size=8, decoder_hidden_size=8,
            max_steps=3, windows_batch_size=16, random_seed=0).fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    out1 = m.predict(h=4, level=[80])
    out2 = m.predict(h=4, level=[80])
    assert "lo-80" in out1 and "hi-80" in out1
    assert bool(jnp.all(out1["lo-80"] <= out1["mean"]))
    assert bool(jnp.all(out1["mean"] <= out1["hi-80"]))
    np.testing.assert_array_equal(np.asarray(out1["mean"]), np.asarray(out2["mean"]))
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(np.asarray(m2.predict(h=4)["mean"]),
                               np.asarray(out1["mean"]), rtol=1e-5, atol=1e-5)


def test_conformal_rejects_temporal_exog():
    y = _make_y(120)
    futr = jnp.asarray(np.random.RandomState(0).randn(y.shape[0], 1), jnp.float32)
    m = _tiny(max_steps=3).fit(y, futr_exog=futr)
    m.conformal_params = ConformalIntervals(h=12, n_windows=3)
    with pytest.raises(ValueError, match="temporal.*exog|MultiQuantileLoss"):
        m.predict(h=12, level=[80], futr_exog=futr[-12:])


def test_conformal_requires_params():
    m = _tiny(max_steps=3).fit(_make_y())
    with pytest.raises(ValueError, match="conformal_params"):
        m.predict(h=12, level=[80])


def test_native_quantile_intervals():
    loss = MultiQuantileLoss()
    m = _tiny(max_steps=30, loss=loss).fit(_make_y())
    out = m.predict(h=12, level=[80])
    assert set(["mean", "lo-80", "hi-80"]).issubset(out)
    assert bool(jnp.all(out["lo-80"] <= out["mean"]))
    assert bool(jnp.all(out["mean"] <= out["hi-80"]))


def test_untrained_quantile_level_raises():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    m = _tiny(max_steps=5, loss=loss).fit(_make_y())
    with pytest.raises(ValueError, match="not trained|quantile"):
        m.predict(h=12, level=[90])  # needs 0.05/0.95, only 0.1/0.5/0.9 trained


def test_pickle_roundtrip_point():
    m = _tiny().fit(_make_y())
    before = np.asarray(m.predict(h=12)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(np.asarray(m2.predict(h=12)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_pickle_roundtrip_quantile_exog():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    y = _make_y(200)
    futr = jnp.asarray(np.random.RandomState(0).randn(y.shape[0], 1), jnp.float32)
    m = _tiny(max_steps=10, loss=loss).fit(y, futr_exog=futr)
    fz = jnp.asarray(np.random.RandomState(1).randn(12, 1), jnp.float32)
    before = np.asarray(m.predict(h=12, futr_exog=fz)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    assert m2._futr_size == 1
    np.testing.assert_allclose(np.asarray(m2.predict(h=12, futr_exog=fz)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_vmap_forecast_matches_python_loop():
    # conformity_scores vmaps forecast over windows; it must match a Python loop.
    B, T, h = 4, 120, 6
    y_batch = jnp.stack([_make_y(T) * (i + 1) for i in range(B)])
    m = TCN(h=h, input_size=18, encoder_hidden_size=8, decoder_hidden_size=8,
            max_steps=4, windows_batch_size=16, random_seed=0)
    seq = jnp.stack([m.forecast(y=y_batch[i], h=h)["mean"] for i in range(B)])
    vm = jax.vmap(lambda y: m.forecast(y=y, h=h)["mean"])(y_batch)
    np.testing.assert_allclose(np.asarray(seq), np.asarray(vm), rtol=5e-3, atol=5e-3)


def test_beats_naive_on_easy_signal():
    n = 240
    y = jnp.asarray(np.sin(np.arange(n) / 5.0), dtype=jnp.float32)
    m = TCN(h=12, input_size=48, encoder_hidden_size=32, max_steps=300,
            windows_batch_size=64, random_seed=0).fit(y[:-12])
    pred = np.asarray(m.predict(h=12)["mean"])
    y_true = np.asarray(y[-12:])
    naive = np.full(12, float(y[-13]))
    assert np.mean(np.abs(pred - y_true)) < np.mean(np.abs(naive - y_true))


# === Namespace ===

def test_importable_from_models_namespace():
    import chronax.models
    from chronax.models import TCN as PublicTCN

    assert PublicTCN is TCN
    assert "TCN" in chronax.models.__all__

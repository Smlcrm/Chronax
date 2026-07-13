"""Tests for chronax.models.iTransformer.

Covers the four source modules of the itransformer subpackage in one file
(losses, module, training, model), matching the flat ``test_<model>.py``
convention used elsewhere in ``tests/``.
"""
import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.itransformer.itransformer_losses import (
    LOSSES, huber, mae, mse, resolve,
)
from chronax.models.itransformer.itransformer_module import (
    AttentionLayer,
    DataEmbeddingInverted,
    ITransformerNet,
    RevIN,
    TransEncoder,
    TransEncoderLayer,
    _resolve_activation,
)
from chronax.models.itransformer.itransformer_model import (
    iTransformer, _boxcox, _inv_boxcox, _select_boxcox_lambda,
)
from chronax.models.itransformer.itransformer_training import (
    build_windows, forward_loss, predict_step, train,
)
from chronax.utils import ConformalIntervals


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


# =============================================================================
# Losses
# =============================================================================

def test_mae_zero_at_match():
    x = jnp.array([1.0, 2.0, 3.0])
    assert float(mae(x, x)) == 0.0


def test_mse_known_value():
    pred = jnp.array([0.0, 0.0])
    target = jnp.array([1.0, 3.0])
    assert float(mse(pred, target)) == pytest.approx(5.0)  # (1 + 9) / 2


def test_huber_quadratic_region():
    pred = jnp.array([0.0])
    target = jnp.array([0.5])
    # |r| = 0.5 <= 1 -> 0.5 * r^2 = 0.125
    assert float(huber(pred, target)) == pytest.approx(0.125)


def test_huber_linear_region():
    pred = jnp.array([0.0])
    target = jnp.array([3.0])
    # |r| = 3 > 1 -> |r| - 0.5 = 2.5
    assert float(huber(pred, target)) == pytest.approx(2.5)


@pytest.mark.parametrize("name", ["mae", "mse", "huber"])
def test_loss_is_jit_and_grad_friendly(name):
    fn = LOSSES[name]
    pred = jnp.array([0.1, 0.2, 0.3])
    target = jnp.array([0.0, 0.5, 0.2])
    g = jax.grad(lambda p: fn(p, target))(pred)
    assert g.shape == pred.shape
    assert jnp.all(jnp.isfinite(g))


def test_resolve_string_returns_registry_entry():
    assert resolve("mae") is mae


def test_resolve_callable_passes_through():
    fn = lambda p, t: jnp.sum(p - t)
    assert resolve(fn) is fn


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        resolve("nope")


def test_registry_keys_are_stable():
    assert set(LOSSES.keys()) == {"mae", "mse", "huber"}


# =============================================================================
# Module
# =============================================================================

def _x(B=4, L=20, N=1):
    rng = np.random.RandomState(0)
    return jnp.asarray(rng.randn(B, L, N), dtype=jnp.float32)


def test_revin_norm_denorm_round_trip():
    revin = RevIN(num_features=1, subtract_last=False, affine=False, rngs=nnx.Rngs(0))
    x = _x()
    z, loc, scale = revin.norm(x)
    x_rec = revin.denorm(z, loc, scale)
    np.testing.assert_allclose(np.asarray(x_rec), np.asarray(x), rtol=1e-5, atol=1e-5)


def test_revin_subtract_mean_centers_on_mean():
    # iTransformer uses the per-window MEAN (subtract_last=False), not the last value.
    revin = RevIN(num_features=1, subtract_last=False, affine=False, rngs=nnx.Rngs(0))
    x = _x()
    z, loc, scale = revin.norm(x)
    np.testing.assert_allclose(
        np.asarray(loc)[:, 0, 0], np.asarray(x).mean(axis=1)[:, 0], rtol=1e-5
    )


def test_revin_uses_population_variance():
    revin = RevIN(num_features=1, subtract_last=False, affine=False, eps=1e-5, rngs=nnx.Rngs(0))
    x = _x()
    _, _, scale = revin.norm(x)
    expected = np.sqrt(np.var(np.asarray(x), axis=1, keepdims=True) + 1e-5)  # ddof=0
    np.testing.assert_allclose(np.asarray(scale), expected, rtol=1e-5)


def test_revin_outputs_float32():
    revin = RevIN(num_features=1, subtract_last=False, affine=False, rngs=nnx.Rngs(0))
    z, loc, scale = revin.norm(_x())
    assert z.dtype == jnp.float32


def test_embedding_inverts_and_projects_shape():
    B, L, N, hidden = 4, 36, 1, 32
    emb = DataEmbeddingInverted(input_size=L, hidden_size=hidden, dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((B, L, N), dtype=jnp.float32)
    out = emb(x, deterministic=True)
    assert out.shape == (B, N, hidden)   # variates become tokens
    assert out.dtype == jnp.float32


def test_embedding_multivariate_token_count_equals_n_series():
    B, L, N, hidden = 2, 36, 5, 16
    emb = DataEmbeddingInverted(input_size=L, hidden_size=hidden, dropout=0.0, rngs=nnx.Rngs(0))
    out = emb(jnp.ones((B, L, N), dtype=jnp.float32), deterministic=True)
    assert out.shape == (B, N, hidden)


def test_attention_shape_preserved():
    B, T, hidden, heads = 2, 7, 32, 4
    attn = AttentionLayer(hidden_size=hidden, n_heads=heads, attn_dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((B, T, hidden), dtype=jnp.float32)
    out = attn(x, deterministic=True)
    assert out.shape == (B, T, hidden)


def test_attention_hidden_not_divisible_raises():
    with pytest.raises(ValueError, match="divisible"):
        AttentionLayer(hidden_size=30, n_heads=4, attn_dropout=0.0, rngs=nnx.Rngs(0))


def test_attention_single_token_is_value_mix():
    # With one token, softmax over a single key is 1, so the output equals the
    # value projection passed through the output projection (no cross-token mixing).
    B, T, hidden, heads = 2, 1, 16, 2
    attn = AttentionLayer(hidden_size=hidden, n_heads=heads, attn_dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(1).randn(B, T, hidden), dtype=jnp.float32)
    out = np.asarray(attn(x, deterministic=True))
    v = np.asarray(attn.w_v(x))
    expected = np.asarray(attn.w_o(jnp.asarray(v)))
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_encoder_layer_shape():
    B, T, hidden, heads = 2, 7, 32, 4
    layer = TransEncoderLayer(hidden_size=hidden, n_heads=heads, d_ff=64,
                              dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((B, T, hidden), dtype=jnp.float32)
    out = layer(x, deterministic=True)
    assert out.shape == (B, T, hidden)


def test_encoder_stack_shape_and_layer_count():
    B, T, hidden, heads = 2, 7, 32, 4
    enc = TransEncoder(e_layers=3, hidden_size=hidden, n_heads=heads, d_ff=64,
                       dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((B, T, hidden), dtype=jnp.float32)
    out = enc(x, deterministic=True)
    assert out.shape == (B, T, hidden)
    assert len(enc.layers) == 3


def test_encoder_applies_final_layernorm():
    # The final norm_layer normalizes each token to ~zero mean across hidden.
    B, T, hidden, heads = 2, 4, 32, 4
    enc = TransEncoder(e_layers=1, hidden_size=hidden, n_heads=heads, d_ff=64,
                       dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(B, T, hidden), dtype=jnp.float32)
    out = np.asarray(enc(x, deterministic=True))
    np.testing.assert_allclose(out.mean(axis=-1), np.zeros((B, T)), atol=1e-4)


def test_resolve_activation_gelu_is_exact():
    f = _resolve_activation("gelu")
    z = jnp.array([1.0])
    exact = 1.0 * 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    np.testing.assert_allclose(float(f(z)[0]), exact, rtol=1e-5)


def test_resolve_activation_unknown_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        _resolve_activation("swish")


def _net(h=24, input_size=72, hidden=32, heads=4, layers=2):
    return ITransformerNet(
        h=h, input_size=input_size, hidden_size=hidden, n_heads=heads,
        e_layers=layers, d_ff=64, dropout=0.0, use_norm=True, rngs=nnx.Rngs(0),
    )


def test_net_forward_shape_univariate():
    net = _net()
    x = jnp.ones((4, 72, 1), dtype=jnp.float32)
    out = net(x, deterministic=True)
    assert out.shape == (4, 24, 1)
    assert out.dtype == jnp.float32


def test_net_is_n_generic_multivariate():
    # The backbone is written N-generically: [B, L, N] -> [B, h, N]. Only the
    # iTransformer wrapper fixes N=1; this guards the future multivariate path.
    net = _net()
    x = jnp.ones((2, 72, 3), dtype=jnp.float32)
    out = net(x, deterministic=True)
    assert out.shape == (2, 24, 3)


def test_net_use_norm_false_still_forwards():
    net = ITransformerNet(h=12, input_size=36, hidden_size=16, n_heads=2,
                          e_layers=1, d_ff=32, dropout=0.0, use_norm=False, rngs=nnx.Rngs(0))
    out = net(jnp.ones((2, 36, 1), dtype=jnp.float32), deterministic=True)
    assert out.shape == (2, 12, 1) and jnp.all(jnp.isfinite(out))


def test_net_hidden_not_divisible_raises():
    with pytest.raises(ValueError, match="divisible"):
        ITransformerNet(h=4, input_size=12, hidden_size=30, n_heads=4, e_layers=1,
                        d_ff=16, dropout=0.0, use_norm=True, rngs=nnx.Rngs(0))


def test_embedding_linear_init_matches_torch_bound():
    # torch nn.Linear default: weight+bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    # fan_in for the inverted embedding = input_size (the lookback length).
    L, hidden = 36, 32
    emb = DataEmbeddingInverted(input_size=L, hidden_size=hidden, dropout=0.0, rngs=nnx.Rngs(0))
    bound = 1.0 / math.sqrt(L)
    assert emb.value_embedding.bias is not None
    k = np.asarray(emb.value_embedding.kernel.value)
    b = np.asarray(emb.value_embedding.bias.value)
    assert k.max() <= bound + 1e-6 and k.min() >= -bound - 1e-6
    assert b.max() <= bound + 1e-6 and b.min() >= -bound - 1e-6
    # uses most of the range (an unbounded normal init would not look like this)
    assert k.max() > 0.5 * bound and k.min() < -0.5 * bound


# =============================================================================
# Training
# =============================================================================

def _train_net(h=12, input_size=36, hidden=16, heads=2, layers=1):
    return ITransformerNet(
        h=h, input_size=input_size, hidden_size=hidden, n_heads=heads,
        e_layers=layers, d_ff=32, dropout=0.0, use_norm=True, rngs=nnx.Rngs(0),
    )


def test_build_windows_shape():
    y = _make_y(60)
    w = build_windows(y, input_size=36, h=12)
    assert w.shape == (60 - 48 + 1, 48)


def test_build_windows_raises_when_too_short():
    with pytest.raises(ValueError, match="too short"):
        build_windows(_make_y(30), input_size=36, h=12)


def test_forward_loss_returns_scalar():
    net = _train_net()
    w = build_windows(_make_y(), input_size=36, h=12)
    loss = forward_loss(net, w[:8], h=12, input_size=36)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_gradient_step_decreases_loss():
    net = _train_net()
    y = _make_y()
    losses = train(net, y, h=12, input_size=36, max_steps=40,
                   windows_batch_size=64, lr=1e-3, seed=0)
    assert losses.shape == (40,)
    assert float(losses[-1]) < float(losses[0])


def test_train_deterministic_with_same_seed():
    y = _make_y()
    l1 = train(_train_net(), y, h=12, input_size=36, max_steps=10,
               windows_batch_size=64, lr=1e-3, seed=0)
    l2 = train(_train_net(), y, h=12, input_size=36, max_steps=10,
               windows_batch_size=64, lr=1e-3, seed=0)
    np.testing.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=1e-5)


def test_predict_step_shape_and_idempotent():
    net = _train_net()
    y = _make_y()
    train(net, y, h=12, input_size=36, max_steps=5, windows_batch_size=64, lr=1e-3, seed=0)
    p1 = predict_step(net, y[-36:], h=12, input_size=36)
    p2 = predict_step(net, y[-36:], h=12, input_size=36)
    assert p1.shape == (12,)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-6)


def test_train_oversample_with_replacement_small_n_regime():
    # n_windows < windows_batch_size -> NF with-replacement branch (the regime
    # every small benchmark series hits). y(60), input_size=36, h=12 -> n_windows=13.
    net = _train_net()
    losses = train(net, _make_y(60), h=12, input_size=36, max_steps=8,
                   windows_batch_size=64, lr=1e-3, seed=0)
    assert losses.shape == (8,)
    assert jnp.all(jnp.isfinite(losses))


def test_train_raises_on_divergence():
    net = _train_net()
    with pytest.raises(RuntimeError, match="diverged"):
        train(net, _make_y(), h=12, input_size=36, max_steps=10,
              windows_batch_size=64, lr=1e9, seed=0)


# =============================================================================
# Model (BaseForecaster conformance)
# =============================================================================

def _tiny():
    return iTransformer(h=12, input_size=36, hidden_size=16, n_heads=2, e_layers=1,
                        d_ff=32, max_steps=20, windows_batch_size=64, random_seed=0)


def test_itransformer_inherits_baseforecaster():
    assert issubclass(iTransformer, BaseForecaster)


def test_init_sets_conformal_params_to_none():
    assert _tiny().conformal_params is None


def test_fit_returns_self_and_sets_model():
    m = _tiny()
    out = m.fit(_make_y())
    assert out is m
    assert m.model_ is not None


def test_predict_deterministic_with_same_seed():
    y = _make_y()
    p1 = _tiny().fit(y).predict(h=12)["mean"]
    p2 = _tiny().fit(y).predict(h=12)["mean"]
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-5, atol=1e-5)


def test_model_beats_naive_last_value_on_easy_signal():
    # End-to-end correctness floor: catches broken loss/optimizer/RevIN-denorm/
    # projection wiring that still passes shape + determinism tests.
    n = 200
    y = jnp.asarray(np.sin(np.arange(n) / 5.0), dtype=jnp.float32)
    m = iTransformer(h=12, input_size=36, hidden_size=32, n_heads=4, e_layers=2,
                     d_ff=64, max_steps=300, windows_batch_size=64, random_seed=0)
    m.fit(y[:-12])
    pred = np.asarray(m.predict(h=12)["mean"])
    y_true = np.asarray(y[-12:])
    naive = np.full(12, float(y[-13]))
    assert np.mean(np.abs(pred - y_true)) < np.mean(np.abs(naive - y_true))


def test_predict_default_returns_h_steps():
    m = _tiny().fit(_make_y())
    assert m.predict(h=12)["mean"].shape == (12,)


def test_predict_smaller_h_slices():
    m = _tiny().fit(_make_y())
    assert m.predict(h=5)["mean"].shape == (5,)


def test_predict_larger_h_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="not supported"):
        m.predict(h=13)


def test_predict_h_below_one_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="positive"):
        m.predict(h=0)


def test_loss_custom_callable_trains():
    def my_loss(pred, target):
        return jnp.mean(jnp.abs(pred - target))
    m = iTransformer(h=12, input_size=36, hidden_size=16, n_heads=2, e_layers=1,
                     d_ff=32, max_steps=10, windows_batch_size=64, random_seed=0, loss=my_loss)
    m.fit(_make_y())
    assert jnp.all(jnp.isfinite(m.predict(h=12)["mean"]))


def test_loss_unknown_string_raises_at_fit():
    m = iTransformer(h=12, input_size=36, hidden_size=16, n_heads=2, e_layers=1,
                     d_ff=32, max_steps=2, windows_batch_size=64, random_seed=0, loss="rmse")
    with pytest.raises(ValueError, match="Unknown loss"):
        m.fit(_make_y())


def test_fit_raises_on_exog():
    with pytest.raises(NotImplementedError):
        _tiny().fit(_make_y(), X=jnp.ones((200, 1)))


def test_fit_raises_on_short_series():
    with pytest.raises(ValueError, match="too short"):
        _tiny().fit(_make_y(40))  # < input_size + h = 48


def test_fit_raises_on_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        _tiny().fit(jnp.ones((200, 2), dtype=jnp.float32))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        _tiny().predict(h=12)


def test_hidden_not_divisible_by_heads_raises_at_fit():
    m = iTransformer(h=12, input_size=36, hidden_size=30, n_heads=4, e_layers=1,
                     d_ff=32, max_steps=2, windows_batch_size=8, random_seed=0)
    with pytest.raises(ValueError, match="divisible"):
        m.fit(_make_y())


def test_pickle_round_trip_preserves_predictions():
    m = _tiny().fit(_make_y())
    before = np.asarray(m.predict(h=12)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    after = np.asarray(m2.predict(h=12)["mean"])
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-5)


def test_pickle_round_trip_preserves_params():
    m = _tiny().fit(_make_y())
    m2 = pickle.loads(pickle.dumps(m))
    assert m.model_ is not None and m2.model_ is not None
    flat_b = nnx.to_flat_state(nnx.state(m.model_))
    flat_a = nnx.to_flat_state(nnx.state(m2.model_))

    def _arrays(flat):
        # Skip Rngs PRNGKey variables — np.asarray on a key<fry> dtype raises.
        out = {}
        for p, v in flat:
            val = getattr(v, "value", None)
            if val is None or str(getattr(val, "dtype", "")).startswith("key"):
                continue
            out[tuple(p)] = val
        return out

    pb, pa = _arrays(flat_b), _arrays(flat_a)
    assert pb.keys() == pa.keys()
    for k in pb:
        np.testing.assert_array_equal(np.asarray(pb[k]), np.asarray(pa[k]))


@pytest.mark.parametrize("loss_name", ["mae", "mse", "huber"])
def test_loss_string_pickle_round_trip(loss_name):
    m = iTransformer(h=12, input_size=36, hidden_size=16, n_heads=2, e_layers=1,
                     d_ff=32, max_steps=10, windows_batch_size=64, random_seed=0,
                     loss=loss_name).fit(_make_y())
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(
        np.asarray(m2.predict(h=12)["mean"]), np.asarray(m.predict(h=12)["mean"]),
        rtol=1e-5, atol=1e-5,
    )


def test_itransformer_importable_from_models_namespace():
    from chronax.models import iTransformer as I
    assert I is iTransformer


def test_constant_series_returns_finite():
    m = _tiny().fit(jnp.ones(200, dtype=jnp.float32))
    assert jnp.all(jnp.isfinite(m.predict(h=12)["mean"]))


def test_h_equals_one():
    m = iTransformer(h=1, input_size=12, hidden_size=8, n_heads=2, e_layers=1,
                     d_ff=16, max_steps=5, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(60))
    assert m.predict(h=1)["mean"].shape == (1,)


def test_forecast_equals_fit_then_predict_same_seed():
    y = _make_y()
    a = _tiny().forecast(y, h=12)["mean"]
    b = _tiny().fit(y).predict(h=12)["mean"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


def test_forecast_raises_on_exog():
    with pytest.raises(NotImplementedError):
        _tiny().forecast(_make_y(), h=12, X=jnp.ones((200, 1)))


def test_forecast_fitted_has_nan_head_and_finite_tail():
    res = _tiny().forecast(_make_y(), h=12, fitted=True)
    fitted = np.asarray(res["fitted"])
    assert fitted.shape == (200,)
    assert np.all(np.isnan(fitted[:36]))
    assert np.all(np.isfinite(fitted[36:]))


def test_build_net_works_in_loop_construction():
    # precursor to conformity_scores' vmap: constructing the net in a loop
    nets = [_tiny()._build_net() for _ in range(3)]
    assert len(nets) == 3


def test_predict_with_level_returns_interval_keys():
    m = iTransformer(h=4, input_size=12, hidden_size=8, n_heads=2, e_layers=1,
                     d_ff=16, max_steps=2, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    out = m.predict(h=4, level=[80])
    assert "lo-80" in out and "hi-80" in out


def test_predict_with_level_raises_without_conformal_params():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="conformal_params"):
        m.predict(h=12, level=[80])


def test_conformity_scores_returns_finite_2d_array():
    m = iTransformer(h=4, input_size=12, hidden_size=8, n_heads=2, e_layers=1,
                     d_ff=16, max_steps=2, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    cs = m.conformity_scores(_make_y(80))
    assert cs.ndim == 2 and cs.shape[1] == 4
    assert jnp.all(jnp.isfinite(cs))


def test_conformal_does_not_corrupt_fitted_model():
    # conformity_scores re-fits inside a vmap; predict(level=...) must run it on a
    # throwaway copy (self.new()) so the tracer-valued refits don't overwrite
    # self.model_. Without the fix the second predict differs or raises a tracer leak.
    m = iTransformer(h=4, input_size=12, hidden_size=8, n_heads=2, e_layers=1,
                     d_ff=16, max_steps=2, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(80))
    before = np.asarray(m.predict(h=4)["mean"])
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    _ = m.predict(h=4, level=[80])
    after = np.asarray(m.predict(h=4)["mean"])
    assert np.all(np.isfinite(after))
    assert np.allclose(before, after)


def test_input_size_default_resolves_to_three_h():
    m = iTransformer(h=10)
    assert m.input_size == 30


# =============================================================================
# Box-Cox transform (use_boxcox)
# =============================================================================

@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, -0.3])
def test_boxcox_inverse_round_trip(lam):
    y = jnp.asarray([1.0, 2.0, 5.0, 10.0, 100.0], dtype=jnp.float32)
    rt = _inv_boxcox(_boxcox(y, lam), lam)
    np.testing.assert_allclose(np.asarray(rt), np.asarray(y), rtol=1e-4, atol=1e-4)


def test_boxcox_lambda_zero_is_log():
    y = jnp.asarray([1.0, 2.0, 10.0], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(_boxcox(y, 0.0)), np.log(np.asarray(y)), rtol=1e-5)


def test_select_lambda_in_grid_and_near_zero_for_exponential():
    # Exponential-growth series -> Box-Cox MLE lambda near 0 (the log regime).
    y = jnp.asarray(np.exp(np.linspace(0, 5, 100)), dtype=jnp.float32)
    lam = _select_boxcox_lambda(y)
    assert -1.0 <= lam <= 2.0
    assert abs(lam) <= 0.3


def test_use_boxcox_off_keeps_lambda_none_and_matches_plain():
    y = _make_y(200) + 2.0  # strictly positive
    m_off = iTransformer(h=12, input_size=36, hidden_size=16, n_heads=2, e_layers=1,
                         d_ff=32, max_steps=20, windows_batch_size=64, random_seed=0).fit(y)
    assert m_off._bc_lambda is None
    plain = iTransformer(h=12, input_size=36, hidden_size=16, n_heads=2, e_layers=1,
                         d_ff=32, max_steps=20, windows_batch_size=64, random_seed=0,
                         use_boxcox=False).fit(y)
    np.testing.assert_allclose(np.asarray(m_off.predict(h=12)["mean"]),
                               np.asarray(plain.predict(h=12)["mean"]), rtol=1e-5, atol=1e-5)


def test_use_boxcox_fit_predict_finite_and_sets_lambda():
    y = jnp.asarray(np.exp(np.linspace(0, 3, 200)) + 1.0, dtype=jnp.float32)
    m = iTransformer(h=12, input_size=36, hidden_size=16, n_heads=2, e_layers=1,
                     d_ff=32, max_steps=30, windows_batch_size=64, random_seed=0,
                     use_boxcox=True).fit(y)
    assert m._bc_lambda is not None
    out = m.predict(h=12)["mean"]
    assert out.shape == (12,) and jnp.all(jnp.isfinite(out))


def test_use_boxcox_requires_positive_values():
    y = _make_y(200)  # sine -> contains non-positive values
    m = iTransformer(h=12, input_size=36, hidden_size=16, n_heads=2, e_layers=1,
                     d_ff=32, max_steps=5, windows_batch_size=64, random_seed=0,
                     use_boxcox=True)
    with pytest.raises(ValueError, match="positive"):
        m.fit(y)


def test_use_boxcox_pickle_round_trip_preserves_predictions_and_lambda():
    y = jnp.asarray(np.exp(np.linspace(0, 3, 200)) + 1.0, dtype=jnp.float32)
    m = iTransformer(h=12, input_size=36, hidden_size=16, n_heads=2, e_layers=1,
                     d_ff=32, max_steps=20, windows_batch_size=64, random_seed=0,
                     use_boxcox=True).fit(y)
    before = np.asarray(m.predict(h=12)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    assert m2._bc_lambda == m._bc_lambda
    np.testing.assert_allclose(np.asarray(m2.predict(h=12)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_use_boxcox_forecast_fitted_inverts_to_original_scale():
    y = jnp.asarray(np.exp(np.linspace(0, 3, 200)) + 1.0, dtype=jnp.float32)
    res = iTransformer(h=12, input_size=36, hidden_size=16, n_heads=2, e_layers=1,
                       d_ff=32, max_steps=20, windows_batch_size=64, random_seed=0,
                       use_boxcox=True).forecast(y, h=12, fitted=True)
    fitted = np.asarray(res["fitted"])
    assert fitted.shape == (200,)
    assert np.all(np.isnan(fitted[:36]))
    tail = fitted[36:]
    assert np.all(np.isfinite(tail))
    # fitted values are back in the original (positive, ~exponential) scale
    assert tail.min() > 0

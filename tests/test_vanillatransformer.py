"""Tests for chronax.models.VanillaTransformer.

Covers the four source modules of the vanillatransformer subpackage in one file
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
from chronax.models.vanillatransformer.vanillatransformer_losses import (
    LOSSES, huber, mae, mse, resolve,
)


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


# ============================================================================
# Losses
# ============================================================================

def test_mae_zero_at_match():
    x = jnp.array([1.0, 2.0, 3.0])
    assert float(mae(x, x)) == 0.0


def test_mse_known_value():
    pred = jnp.array([0.0, 0.0])
    target = jnp.array([1.0, 3.0])
    assert float(mse(pred, target)) == pytest.approx(5.0)


def test_huber_quadratic_region():
    assert float(huber(jnp.array([0.0]), jnp.array([0.5]))) == pytest.approx(0.125)


def test_huber_linear_region():
    assert float(huber(jnp.array([0.0]), jnp.array([3.0]))) == pytest.approx(2.5)


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
    with pytest.raises(ValueError):
        resolve("not_a_loss")


from chronax.models.vanillatransformer.vanillatransformer_module import (
    AttentionLayer,
    DataEmbedding,
    TokenEmbedding,
    _positional_embedding,
    _resolve_activation,
)


# ============================================================================
# Module: inits / embeddings / attention
# ============================================================================

def test_resolve_activation_gelu_is_exact():
    z = jnp.array([0.7, -1.3, 2.0])
    got = _resolve_activation("gelu")(z)
    want = jax.nn.gelu(z, approximate=False)
    assert jnp.allclose(got, want)


def test_resolve_activation_unknown_raises():
    with pytest.raises(ValueError):
        _resolve_activation("swish")


def test_positional_embedding_shape_and_bounds():
    pe = _positional_embedding(16, 8)
    assert pe.shape == (1, 16, 8)
    assert jnp.all(jnp.abs(pe) <= 1.0 + 1e-6)


def test_token_embedding_shape():
    tok = TokenEmbedding(hidden_size=12, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 7, 1))
    out = tok(x)
    assert out.shape == (2, 7, 12)


def test_data_embedding_shape_and_dropout_identity_when_deterministic():
    emb = DataEmbedding(hidden_size=12, dropout=0.5, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 7, 1))
    out = emb(x, deterministic=True)
    assert out.shape == (2, 7, 12)


def test_attention_layer_self_shape():
    attn = AttentionLayer(hidden_size=16, n_heads=4, attn_dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((3, 5, 16))
    out = attn(x, x, deterministic=True)
    assert out.shape == (3, 5, 16)


def test_attention_layer_cross_shape():
    attn = AttentionLayer(hidden_size=16, n_heads=4, attn_dropout=0.0, rngs=nnx.Rngs(0))
    q = jnp.ones((3, 5, 16))
    kv = jnp.ones((3, 9, 16))
    out = attn(q, kv, deterministic=True)
    assert out.shape == (3, 5, 16)


def test_attention_layer_requires_divisible_heads():
    with pytest.raises(ValueError):
        AttentionLayer(hidden_size=10, n_heads=4, attn_dropout=0.0, rngs=nnx.Rngs(0))


from chronax.models.vanillatransformer.vanillatransformer_module import (
    TransDecoder,
    TransEncoder,
    VanillaTransformerNet,
)


# ============================================================================
# Module: encoder / decoder / backbone
# ============================================================================

def test_trans_encoder_shape():
    enc = TransEncoder(
        encoder_layers=2, hidden_size=16, n_heads=4, conv_hidden_size=8,
        dropout=0.0, activation="gelu", rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 6, 16))
    out = enc(x, deterministic=True)
    assert out.shape == (2, 6, 16)


def test_trans_decoder_shape_projects_to_one_channel():
    dec = TransDecoder(
        decoder_layers=1, hidden_size=16, n_heads=4, conv_hidden_size=8,
        dropout=0.0, activation="gelu", c_out=1, rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 10, 16))      # label_len + h tokens
    cross = jnp.ones((2, 6, 16))   # encoder output
    out = dec(x, cross, deterministic=True)
    assert out.shape == (2, 10, 1)


def test_net_forward_shape():
    net = VanillaTransformerNet(
        h=4, input_size=12, hidden_size=16, n_heads=4, conv_hidden_size=8,
        encoder_layers=2, decoder_layers=1, dropout=0.0, activation="gelu",
        decoder_input_size_multiplier=0.5, rngs=nnx.Rngs(0),
    )
    x = jnp.ones((3, 12, 1))
    out = net(x, deterministic=True)
    assert out.shape == (3, 4, 1)


def test_net_label_len_is_ceil():
    net = VanillaTransformerNet(
        h=4, input_size=7, hidden_size=16, n_heads=4, conv_hidden_size=8,
        encoder_layers=1, decoder_layers=1, dropout=0.0, activation="gelu",
        decoder_input_size_multiplier=0.5, rngs=nnx.Rngs(0),
    )
    assert net.label_len == math.ceil(7 * 0.5)  # == 4


def test_net_deterministic_is_repeatable():
    net = VanillaTransformerNet(
        h=4, input_size=12, hidden_size=16, n_heads=4, conv_hidden_size=8,
        encoder_layers=1, decoder_layers=1, dropout=0.5, activation="gelu",
        decoder_input_size_multiplier=0.5, rngs=nnx.Rngs(0),
    )
    x = jnp.asarray(np.random.RandomState(0).randn(2, 12, 1), dtype=jnp.float32)
    a = net(x, deterministic=True)
    b = net(x, deterministic=True)
    assert jnp.allclose(a, b)


from chronax.models.vanillatransformer.vanillatransformer_training import (
    build_windows, forward_loss, predict_step, train,
)


# ============================================================================
# Training
# ============================================================================

def test_build_windows_shape_and_content():
    y = jnp.arange(10.0)
    w = build_windows(y, input_size=4, h=2)
    assert w.shape == (5, 6)          # n = 10 - 6 + 1 = 5
    assert jnp.allclose(w[0], jnp.arange(6.0))
    assert jnp.allclose(w[-1], jnp.arange(4.0, 10.0))


def test_build_windows_too_short_raises():
    with pytest.raises(ValueError):
        build_windows(jnp.arange(3.0), input_size=4, h=2)


def _tiny_net(h=4, input_size=12):
    return VanillaTransformerNet(
        h=h, input_size=input_size, hidden_size=16, n_heads=4, conv_hidden_size=8,
        encoder_layers=1, decoder_layers=1, dropout=0.0, activation="gelu",
        decoder_input_size_multiplier=0.5, rngs=nnx.Rngs(0),
    )


def test_forward_loss_scalar_finite():
    net = _tiny_net()
    w = build_windows(_make_y(60), input_size=12, h=4)
    loss = forward_loss(net, w[:8], h=4, input_size=12)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_returns_losses_and_reduces():
    net = _tiny_net()
    losses = train(net, _make_y(120), h=4, input_size=12, max_steps=30,
                   windows_batch_size=16, lr=1e-3, seed=0)
    assert losses.shape == (30,)
    assert jnp.all(jnp.isfinite(losses))
    assert float(jnp.mean(losses[-5:])) < float(jnp.mean(losses[:5]))


def test_train_with_replacement_regime_runs():
    # n_windows (= 120 - 96 + 1 = 25) < windows_batch_size (= 64) -> WITH replacement.
    net = _tiny_net(h=24, input_size=72)
    losses = train(net, _make_y(120), h=24, input_size=72, max_steps=10,
                   windows_batch_size=64, lr=1e-3, seed=1)
    assert losses.shape == (10,)
    assert jnp.all(jnp.isfinite(losses))


def test_predict_step_shape():
    net = _tiny_net()
    out = predict_step(net, _make_y(60), h=4, input_size=12)
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))


from chronax.models.vanillatransformer.vanillatransformer_model import (
    VanillaTransformer, _boxcox, _inv_boxcox, _select_boxcox_lambda,
)
from chronax.utils import ConformalIntervals


# ============================================================================
# Model wrapper
# ============================================================================

def _fast_model(**kw):
    base = dict(h=4, input_size=12, hidden_size=16, n_heads=4, conv_hidden_size=8,
                encoder_layers=1, decoder_layers=1, dropout=0.0, max_steps=40,
                learning_rate=1e-3, windows_batch_size=16, random_seed=0)
    base.update(kw)
    return VanillaTransformer(**base)


def test_is_base_forecaster():
    assert issubclass(VanillaTransformer, BaseForecaster)


def test_input_size_default_resolves_to_3h():
    m = VanillaTransformer(h=10)
    assert m.input_size == 30


def test_fit_predict_shape_and_finite():
    m = _fast_model()
    m.fit(_make_y(120))
    out = m.predict(h=4)
    assert out["mean"].shape == (4,)
    assert jnp.all(jnp.isfinite(out["mean"]))


def test_predict_h_greater_than_trained_raises():
    m = _fast_model()
    m.fit(_make_y(120))
    with pytest.raises(ValueError):
        m.predict(h=5)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        _fast_model().predict(h=4)


def test_fit_rejects_2d():
    with pytest.raises(ValueError):
        _fast_model().fit(jnp.ones((10, 2)))


def test_fit_rejects_exog():
    with pytest.raises(NotImplementedError):
        _fast_model().fit(_make_y(120), X=jnp.ones((120, 1)))


def test_forecast_matches_fit_predict():
    y = _make_y(120)
    a = _fast_model().forecast(y, h=4)["mean"]
    b = _fast_model().fit(y).predict(h=4)["mean"]
    assert jnp.allclose(a, b)


def test_forecast_fitted_values_shape_and_nan_head():
    m = _fast_model()
    out = m.forecast(_make_y(120), h=4, fitted=True)
    fitted = out["fitted"]
    assert fitted.shape == (120,)
    assert bool(jnp.all(jnp.isnan(fitted[:12])))
    assert bool(jnp.all(jnp.isfinite(fitted[12:])))


def test_pickle_round_trip_preserves_predictions():
    m = _fast_model()
    m.fit(_make_y(120))
    before = m.predict(h=4)["mean"]
    m2 = pickle.loads(pickle.dumps(m))
    after = m2.predict(h=4)["mean"]
    assert jnp.allclose(before, after)


def test_boxcox_round_trip_identity():
    y = jnp.asarray(np.linspace(1.0, 5.0, 50), dtype=jnp.float32)
    lam = _select_boxcox_lambda(y)
    assert jnp.allclose(_inv_boxcox(_boxcox(y, lam), lam), y, atol=1e-4)


def test_boxcox_requires_positive():
    m = _fast_model(use_boxcox=True)
    with pytest.raises(ValueError):
        m.fit(_make_y(120))   # sine series has non-positive values


def test_boxcox_fit_predict_positive_series():
    y = jnp.asarray(50.0 + 40.0 * np.sin(np.arange(160) / 6.0), dtype=jnp.float32)
    m = _fast_model(use_boxcox=True)
    m.fit(y)
    out = m.predict(h=4)
    assert out["mean"].shape == (4,)
    assert jnp.all(jnp.isfinite(out["mean"]))


def test_conformal_intervals_present_when_level_set():
    m = _fast_model()
    m.fit(_make_y(160))
    m.conformal_params = ConformalIntervals(n_windows=2, h=4)
    out = m.predict(h=4, level=[80])
    assert "lo-80" in out and "hi-80" in out
    assert out["lo-80"].shape == (4,)


def test_predict_level_without_conformal_params_raises():
    m = _fast_model()
    m.fit(_make_y(120))
    with pytest.raises(ValueError):
        m.predict(h=4, level=[80])


def test_conformal_does_not_corrupt_fitted_model():
    # conformity_scores re-fits inside a vmap; it must run on a throwaway copy
    # (self.new()) so the tracer-valued refits don't overwrite self.model_.
    # Without the fix the second predict returns different values or raises a
    # tracer leak.
    m = _fast_model()
    m.fit(_make_y(160))
    before = np.asarray(m.predict(h=4)["mean"])
    m.conformal_params = ConformalIntervals(n_windows=2, h=4)
    _ = m.predict(h=4, level=[80])
    after = np.asarray(m.predict(h=4)["mean"])
    assert np.all(np.isfinite(after))
    assert np.allclose(before, after)

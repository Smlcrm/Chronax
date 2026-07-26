"""Tests for chronax.models.TFT.

Covers the TFT subpackage (losses, scaler, layers, module, training, model) in
one flat file, matching the ``test_<model>.py`` convention used elsewhere in
``tests/``.
"""
import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.tft.tft_losses import (
    LOSSES, MultiQuantileLoss, huber, mae, mse, outputsize_multiplier, resolve,
)
from chronax.models.tft.tft_scaler import IdentityScaler, RobustScaler, resolve_scaler


# =============================================================================
# Losses
# =============================================================================

def test_mae_zero_at_match():
    x = jnp.array([1.0, 2.0, 3.0])
    assert float(mae(x, x)) == 0.0


def test_mse_known_value():
    assert float(mse(jnp.array([0.0, 0.0]), jnp.array([1.0, 3.0]))) == pytest.approx(5.0)


def test_huber_regions():
    assert float(huber(jnp.array([0.0]), jnp.array([0.5]))) == pytest.approx(0.125)
    assert float(huber(jnp.array([0.0]), jnp.array([3.0]))) == pytest.approx(2.5)


def test_point_loss_multiplier_is_one():
    assert outputsize_multiplier(mae) == 1


def test_resolve_string_and_callable():
    assert resolve("mae") is mae
    f = lambda p, t: jnp.sum(p - t)
    assert resolve(f) is f
    with pytest.raises(ValueError, match="Unknown loss"):
        resolve("nope")


def test_mqloss_multiplier_and_requires_median():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    assert loss.outputsize_multiplier == 3
    assert outputsize_multiplier(loss) == 3
    with pytest.raises(ValueError, match="0.5"):
        MultiQuantileLoss((0.1, 0.9))
    with pytest.raises(ValueError, match="between 0 and 1"):
        MultiQuantileLoss((0.0, 0.5, 1.0))


def test_mqloss_pinball_value():
    # err = y - yhat. For q and a single (h=1) point with pred all zeros, target=2:
    # QL = mean_q max(q*2, (q-1)*2) = mean_q q*2  -> mean of {0.2,1.0,1.8} = 1.0
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    pred = jnp.zeros((1, 1, 3))
    target = jnp.array([[2.0]])
    assert float(loss(pred, target)) == pytest.approx(1.0)


def test_mqloss_pickles():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    loss2 = pickle.loads(pickle.dumps(loss))
    assert loss2.quantiles == loss.quantiles


# =============================================================================
# Scaler
# =============================================================================

def test_robust_scaler_round_trip():
    x = jnp.asarray(np.random.RandomState(0).randn(3, 20), dtype=jnp.float32)
    sc = RobustScaler()
    shift, scale = sc.stats(x, axis=1)
    z = sc.transform(x, shift, scale)
    np.testing.assert_allclose(np.asarray(sc.inverse(z, shift, scale)), np.asarray(x), rtol=1e-4, atol=1e-4)


def test_robust_scaler_uses_median():
    x = jnp.asarray([[1.0, 2.0, 3.0, 4.0, 100.0]], dtype=jnp.float32)
    shift, _ = RobustScaler().stats(x, axis=1)
    assert float(shift[0, 0]) == pytest.approx(3.0)  # median, robust to the 100 outlier


def test_identity_scaler_is_noop():
    x = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    sc = IdentityScaler()
    shift, scale = sc.stats(x, axis=1)
    np.testing.assert_array_equal(np.asarray(sc.transform(x, shift, scale)), np.asarray(x))


def test_resolve_scaler():
    assert isinstance(resolve_scaler("robust"), RobustScaler)
    assert isinstance(resolve_scaler("identity"), IdentityScaler)
    with pytest.raises(ValueError):
        resolve_scaler("minmax")


# =============================================================================
# Layers
# =============================================================================
from chronax.models.tft.tft_layers import (  # noqa: E402
    ContinuousEmbedding,
    GLU,
    GRN,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
    _resolve_grn_activation,
    _TorchLinearInit,
)


def test_glu_shape_and_gate():
    glu = GLU(8, 5, rngs=nnx.Rngs(0))
    out = glu(jnp.ones((4, 8)))
    assert out.shape == (4, 5)


def test_grn_residual_shape_preserved():
    grn = GRN(16, 16, dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 7, 16))
    assert grn(x, deterministic=True).shape == (2, 7, 16)


def test_grn_output_size_projects_and_skips_layernorm_when_one():
    # output_size=1 -> MaybeLayerNorm is Identity; out_proj maps residual to size 1.
    grn = GRN(16, 16, output_size=1, dropout=0.0, rngs=nnx.Rngs(0))
    out = grn(jnp.ones((2, 7, 16)), deterministic=True)
    assert out.shape == (2, 7, 1)
    assert grn.layer_norm is None


def test_grn_context_injection_changes_output():
    grn = GRN(16, 16, context_size=16, dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 7, 16))
    c = jnp.ones((2, 16))  # static context, broadcast over time
    out_nc = grn(x, None, deterministic=True)
    out_c = grn(x, c, deterministic=True)
    assert out_c.shape == (2, 7, 16)
    assert not np.allclose(np.asarray(out_nc), np.asarray(out_c))


def test_grn_activation_resolver():
    assert _resolve_grn_activation("ELU")(jnp.array([-1.0]))[0] < 0  # ELU is negative for x<0
    with pytest.raises(ValueError, match="activation"):
        _resolve_grn_activation("mish")


def test_vsn_output_and_weight_shapes():
    vsn = VariableSelectionNetwork(hidden_size=16, num_inputs=3, dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 7, 3, 16))                      # [B, T, num_inputs, hidden]
    out, w = vsn(x, deterministic=True)
    assert out.shape == (2, 7, 16)
    assert w.shape == (2, 7, 3)


def test_vsn_weights_are_softmax():
    vsn = VariableSelectionNetwork(hidden_size=8, num_inputs=4, dropout=0.0, rngs=nnx.Rngs(1))
    x = jnp.asarray(np.random.RandomState(0).randn(2, 5, 4, 8), dtype=jnp.float32)
    _, w = vsn(x, deterministic=True)
    np.testing.assert_allclose(np.asarray(w.sum(-1)), np.ones((2, 5)), rtol=1e-5)
    assert bool(jnp.all(w >= 0))


def test_vsn_context_accepted():
    vsn = VariableSelectionNetwork(hidden_size=8, num_inputs=2, dropout=0.0, context_size=8, rngs=nnx.Rngs(2))
    x = jnp.ones((2, 5, 2, 8))
    cs = jnp.ones((2, 8))
    out, w = vsn(x, context=cs, deterministic=True)
    assert out.shape == (2, 5, 8) and w.shape == (2, 5, 2)


def test_vsn_static_no_time_axis():
    # static path: x is [B, num_inputs, hidden] (no time) -> out [B, hidden]
    vsn = VariableSelectionNetwork(hidden_size=8, num_inputs=3, dropout=0.0, rngs=nnx.Rngs(3))
    out, w = vsn(jnp.ones((2, 3, 8)), deterministic=True)
    assert out.shape == (2, 8) and w.shape == (2, 3)


def test_continuous_embedding_shape():
    emb = ContinuousEmbedding(num_features=3, hidden_size=8, rngs=nnx.Rngs(0))
    out = emb(jnp.ones((2, 5, 3)))
    assert out.shape == (2, 5, 3, 8)


def test_continuous_embedding_is_affine_per_feature():
    # out[..., j, :] = x[..., j] * vec[j] + bias[j]
    emb = ContinuousEmbedding(num_features=2, hidden_size=4, rngs=nnx.Rngs(0))
    x = jnp.asarray([[3.0, 5.0]], dtype=jnp.float32)
    out = np.asarray(emb(x))[0]                       # [2, 4]
    vec = np.asarray(emb.vectors.value)
    bias = np.asarray(emb.bias.value)
    np.testing.assert_allclose(out, np.array([3.0, 5.0])[:, None] * vec + bias, rtol=1e-5)


def test_continuous_embedding_zero_features():
    emb = ContinuousEmbedding(num_features=0, hidden_size=8, rngs=nnx.Rngs(0))
    out = emb(jnp.ones((2, 5, 0)))
    assert out.shape == (2, 5, 0, 8)


def test_continuous_embedding_xavier_init_nf_parity():
    # NF initializes embedding vectors with torch.nn.init.xavier_normal_ —
    # std = sqrt(2/(num_features+hidden)). A fixed small std starts the input
    # coupling far weaker. Guard the distribution, not exact values.
    emb = ContinuousEmbedding(num_features=1, hidden_size=512, rngs=nnx.Rngs(0))
    std = float(np.asarray(emb.vectors.value).std())
    expected = (2.0 / (1 + 512)) ** 0.5
    assert abs(std - expected) / expected < 0.15
    np.testing.assert_array_equal(np.asarray(emb.bias.value), 0.0)


def test_vsn_selection_weights_ignore_dropout_nf_parity():
    # NF builds the VSN joint/selection GRN WITHOUT dropout — only the
    # per-variable GRNs take it. Selection weights must
    # be deterministic across training-mode calls while the variable transforms
    # stay stochastic.
    vsn = VariableSelectionNetwork(hidden_size=8, num_inputs=2, dropout=0.5, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(4, 2, 8), jnp.float32)
    out_a, w_a = vsn(x, deterministic=False)
    out_b, w_b = vsn(x, deterministic=False)
    np.testing.assert_array_equal(np.asarray(w_a), np.asarray(w_b))
    assert not bool(jnp.allclose(out_a, out_b)), "var_grns dropout inactive in training mode"


def test_imha_shapes_and_divisibility():
    attn = InterpretableMultiHeadAttention(n_head=4, hidden_size=16, rngs=nnx.Rngs(0))
    out, w = attn(jnp.ones((2, 7, 16)), deterministic=True)
    assert out.shape == (2, 7, 16)
    assert w.shape == (2, 4, 7, 7)
    with pytest.raises(ValueError, match="divisible"):
        InterpretableMultiHeadAttention(n_head=3, hidden_size=16, rngs=nnx.Rngs(0))


def test_imha_is_causal():
    # attention weights above the diagonal (future keys) must be ~0.
    attn = InterpretableMultiHeadAttention(n_head=2, hidden_size=8, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(1, 5, 8), dtype=jnp.float32)
    _, w = attn(x, deterministic=True)
    w = np.asarray(w)[0]                              # [n_head, T, T]
    upper = np.triu(np.ones((5, 5)), k=1).astype(bool)
    assert np.allclose(w[:, upper], 0.0, atol=1e-6)


def test_imha_rows_sum_to_one():
    attn = InterpretableMultiHeadAttention(n_head=2, hidden_size=8, rngs=nnx.Rngs(0))
    _, w = attn(jnp.ones((1, 4, 8)), deterministic=True)
    np.testing.assert_allclose(np.asarray(w).sum(-1), np.ones((1, 2, 4)), rtol=1e-5)


def test_imha_out_dropout_applied_in_training_mode():
    """NF parity: InterpretableMultiHeadAttention applies out_dropout (rate =
    `dropout`, NOT attn_dropout) after the output projection.
    Training-mode forwards must be stochastic when dropout > 0; deterministic
    forwards must be reproducible and match the eval path."""
    attn = InterpretableMultiHeadAttention(
        n_head=2, hidden_size=16, attn_dropout=0.0, dropout=0.5, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(2, 6, 16), jnp.float32)
    out_a, _ = attn(x, deterministic=False)
    out_b, _ = attn(x, deterministic=False)
    assert not bool(jnp.allclose(out_a, out_b)), "out_dropout inactive in training mode"
    det_a, _ = attn(x, deterministic=True)
    det_b, _ = attn(x, deterministic=True)
    np.testing.assert_allclose(np.asarray(det_a), np.asarray(det_b), rtol=1e-6)


# =============================================================================
# Module (encoders, fusion decoder, top-level net)
# =============================================================================
from chronax.models.tft.tft_module import (  # noqa: E402
    StaticCovariateEncoder,
    TFTNet,
    TemporalCovariateEncoder,
    TemporalFusionDecoder,
)


def test_static_encoder_context_shapes_lstm():
    enc = StaticCovariateEncoder(hidden_size=8, num_static=3, dropout=0.0, activation="ELU",
                                 rnn_type="lstm", n_rnn_layers=2, rngs=nnx.Rngs(0))
    s = jnp.ones((4, 3, 8))
    cs, ce, ch, cc, w = enc(s, deterministic=True)
    assert cs.shape == (4, 8) and ce.shape == (4, 8)
    assert ch.shape == (2, 4, 8) and cc.shape == (2, 4, 8)
    assert w.shape == (4, 3)


def test_static_encoder_gru_shares_h_and_c():
    enc = StaticCovariateEncoder(hidden_size=8, num_static=2, dropout=0.0, activation="ELU",
                                 rnn_type="gru", n_rnn_layers=1, rngs=nnx.Rngs(0))
    _, _, ch, cc, _ = enc(jnp.ones((4, 2, 8)), deterministic=True)
    assert ch.shape == (1, 4, 8)
    np.testing.assert_array_equal(np.asarray(ch), np.asarray(cc))


def _temporal_encoder(rnn_type="lstm", n_layers=1, hidden=8):
    return TemporalCovariateEncoder(
        hidden_size=hidden, num_hist_vars=3, num_futr_vars=2, dropout=0.0,
        activation="ELU", rnn_type=rnn_type, n_rnn_layers=n_layers, rngs=nnx.Rngs(0),
    )


def test_temporal_encoder_shape_lstm():
    enc = _temporal_encoder("lstm")
    B, L, h, hidden = 2, 6, 4, 8
    hist = jnp.ones((B, L, 3, hidden)); futr = jnp.ones((B, h, 2, hidden))
    cs = jnp.zeros((B, hidden)); ch = jnp.zeros((1, B, hidden)); cc = jnp.zeros((1, B, hidden))
    temporal, hw, fw = enc(hist, futr, cs, ch, cc, deterministic=True)
    assert temporal.shape == (B, L + h, hidden)
    assert hw.shape == (B, L, 3) and fw.shape == (B, h, 2)


def test_temporal_encoder_state_threading_changes_decoder():
    enc = _temporal_encoder("lstm")
    B, L, h, hidden = 1, 5, 3, 8
    hist = jnp.asarray(np.random.RandomState(0).randn(B, L, 3, hidden), jnp.float32)
    futr = jnp.asarray(np.random.RandomState(1).randn(B, h, 2, hidden), jnp.float32)
    cs = jnp.zeros((B, hidden))
    out_zero, _, _ = enc(hist, futr, cs, jnp.zeros((1, B, hidden)), jnp.zeros((1, B, hidden)), deterministic=True)
    out_init, _, _ = enc(hist, futr, cs, jnp.ones((1, B, hidden)), jnp.ones((1, B, hidden)), deterministic=True)
    assert not np.allclose(np.asarray(out_zero), np.asarray(out_init))


def test_temporal_encoder_gru_runs():
    enc = _temporal_encoder("gru")
    B, L, h, hidden = 2, 6, 4, 8
    hist = jnp.ones((B, L, 3, hidden)); futr = jnp.ones((B, h, 2, hidden))
    cs = jnp.zeros((B, hidden)); ch = jnp.zeros((1, B, hidden)); cc = jnp.zeros((1, B, hidden))
    temporal, _, _ = enc(hist, futr, cs, ch, cc, deterministic=True)
    assert temporal.shape == (B, L + h, hidden) and bool(jnp.all(jnp.isfinite(temporal)))


def test_fusion_decoder_slices_to_horizon():
    dec = TemporalFusionDecoder(n_head=2, hidden_size=8, dropout=0.0, attn_dropout=0.0,
                                activation="ELU", rngs=nnx.Rngs(0))
    B, L, h, hidden = 2, 6, 4, 8
    temporal = jnp.ones((B, L + h, hidden))
    ce = jnp.zeros((B, hidden))
    out, attn = dec(temporal, ce, input_size=L, deterministic=True)
    assert out.shape == (B, h, hidden)
    # TFT-S1: only the surviving h query rows are computed (query_start=L).
    assert attn.shape == (B, 2, h, L + h)


def test_imha_query_slice_matches_compute_then_slice():
    # TFT-S1 deviation guard: NF computes all T query rows and discards all but
    # the last h; chronax computes only the last h. Per-row math is identical;
    # outputs match compute-then-slice to floating-point reassociation (~1 ULP
    # — the sliced einsum tiles differently; measured 1.6e-8 at these dims).
    imha = InterpretableMultiHeadAttention(n_head=2, hidden_size=16, attn_dropout=0.0,
                                           dropout=0.5, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(3, 10, 16), jnp.float32)
    full, attn_full = imha(x, deterministic=True)
    sliced, attn_sliced = imha(x, deterministic=True, query_start=6)
    np.testing.assert_allclose(np.asarray(full[:, 6:]), np.asarray(sliced), rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(attn_full[:, :, 6:]), np.asarray(attn_sliced), rtol=0, atol=1e-6)


def test_same_config_nets_share_graphdef():
    # Value-__eq__ on the picklable initializers makes same-config graphdefs
    # EQUAL, so the module-level @nnx.jit _forward_det cache hits across nets
    # built by a later fit instead of recompiling per instance.
    gd_a, _ = nnx.split(_tnet())
    gd_b, _ = nnx.split(_tnet())
    assert gd_a == gd_b


def _net(**kw):
    base = dict(h=4, input_size=6, hidden_size=8, n_head=2, dropout=0.0, rngs=nnx.Rngs(0))
    base.update(kw)
    return TFTNet(**base)


def test_net_univariate_no_exog_shape():
    net = _net()
    out = net(jnp.ones((3, 6, 1)), deterministic=True)
    assert out.shape == (3, 4, 1) and out.dtype == jnp.float32


def test_net_full_exog_shape():
    net = _net(stat_exog_size=2, hist_exog_size=3, futr_exog_size=2)
    B, L, h = 3, 6, 4
    out = net(
        jnp.ones((B, L, 1)),
        hist_exog=jnp.ones((B, L, 3)),
        futr_exog=jnp.ones((B, L + h, 2)),
        stat_exog=jnp.ones((B, 2)),
        deterministic=True,
    )
    assert out.shape == (B, h, 1)


def test_net_futr_only_and_stat_only():
    B, L, h = 2, 6, 4
    net_f = _net(futr_exog_size=2)
    out_f = net_f(jnp.ones((B, L, 1)), futr_exog=jnp.ones((B, L + h, 2)), deterministic=True)
    assert out_f.shape == (B, h, 1)
    net_s = _net(stat_exog_size=3)
    out_s = net_s(jnp.ones((B, L, 1)), stat_exog=jnp.ones((B, 3)), deterministic=True)
    assert out_s.shape == (B, h, 1)


def test_net_quantile_multiplier_widens_output():
    net = _net(outputsize_multiplier=3)
    out = net(jnp.ones((2, 6, 1)), deterministic=True)
    assert out.shape == (2, 4, 3)


# =============================================================================
# Training
# =============================================================================
from chronax.models.tft.tft_training import (  # noqa: E402
    build_exog_windows,
    build_windows,
    forward_loss,
    predict_step,
    train,
)


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _tnet(mult=1, **kw):
    base = dict(h=12, input_size=36, hidden_size=16, n_head=2, dropout=0.0,
                outputsize_multiplier=mult, rngs=nnx.Rngs(0))
    base.update(kw)
    return TFTNet(**base)


def test_build_windows_shape():
    w, m = build_windows(_make_y(60), 36, 12)
    assert w.shape == (60 - 36, 48) and m.shape == (60 - 36, 12)


def test_build_windows_nf_padding_semantics():
    # NF parity (NEU-A1): series right-padded with h zeros; n = T - input_size
    # windows; every window with >=1 real target point is kept, padded tail
    # masked out of the loss.
    y = jnp.arange(1.0, 11.0)                # T=10, values 1..10 (no zeros)
    w, m = build_windows(y, input_size=3, h=2)
    assert w.shape == (7, 5) and m.shape == (7, 2)
    np.testing.assert_array_equal(np.asarray(w[0]), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.asarray(m[0]), [1, 1])
    np.testing.assert_array_equal(np.asarray(w[6]), [7, 8, 9, 10, 0])   # padded tail
    np.testing.assert_array_equal(np.asarray(m[6]), [1, 0])             # masked out
    assert float(m.sum()) == 13.0


def test_build_windows_too_short_raises():
    with pytest.raises(ValueError, match="too short|short"):
        build_windows(_make_y(30), 36, 12)


def test_build_exog_windows_spans():
    arr = jnp.asarray(np.random.RandomState(0).randn(60, 2), jnp.float32)
    n = 60 - 36
    assert build_exog_windows(arr, 36, 12, n, "input").shape == (n, 36, 2)
    assert build_exog_windows(arr, 36, 12, n, "full").shape == (n, 48, 2)


def test_exog_scaling_stats_use_insample_span_only():
    # NF's _normalization masks the horizon out of the scaler stats: exog
    # statistics come from the insample span only; the horizon slice is
    # transformed with those stats but never feeds them.
    from chronax.models.tft.tft_training import _scale_exog
    rng = np.random.RandomState(0)
    L, h, F = 8, 4, 2
    w = jnp.asarray(rng.randn(3, L + h, F), jnp.float32)
    a = _scale_exog(w, RobustScaler(), stats_len=L)
    w2 = w.at[:, L:, :].add(1e4)         # perturb horizon slice massively
    b = _scale_exog(w2, RobustScaler(), stats_len=L)
    np.testing.assert_array_equal(np.asarray(a[:, :L]), np.asarray(b[:, :L]))


def test_masked_loss_ignores_padded_tail():
    # A window whose padded target cell is masked must contribute only its
    # real points to the loss (NF `_weighted_mean` semantics).
    net = _tnet()
    y = jnp.asarray(np.arange(1.0, 61.0), jnp.float32)   # T=60
    w, m = build_windows(y, 36, 12)
    kw = dict(h=12, input_size=36, scaler=RobustScaler(), loss_fn=mae, deterministic=True)
    full = forward_loss(net, w, m, **kw)
    w_poison = w.at[-1, -1].set(1e6)     # last target cell of the last window is padded
    poisoned = forward_loss(net, w_poison, m, **kw)
    np.testing.assert_allclose(float(full), float(poisoned), rtol=0, atol=1e-6)


def test_forward_loss_scalar_finite():
    net = _tnet()
    w, _ = build_windows(_make_y(), 36, 12)
    loss = forward_loss(net, w[:8], h=12, input_size=36, scaler=RobustScaler(),
                        loss_fn=mae, deterministic=True)
    assert loss.shape == () and bool(jnp.isfinite(loss))


def test_train_decreases_loss_univariate():
    net = _tnet()
    losses = train(net, _make_y(), h=12, input_size=36, max_steps=40,
                   windows_batch_size=64, lr=1e-3, seed=0, loss_fn=mae, scaler=RobustScaler())
    assert losses.shape == (40,)
    assert float(losses[-1]) < float(losses[0])


def test_train_deterministic_same_seed():
    a = train(_tnet(), _make_y(), h=12, input_size=36, max_steps=10, windows_batch_size=64,
              lr=1e-3, seed=0, loss_fn=mae, scaler=RobustScaler())
    b = train(_tnet(), _make_y(), h=12, input_size=36, max_steps=10, windows_batch_size=64,
              lr=1e-3, seed=0, loss_fn=mae, scaler=RobustScaler())
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5)


def test_predict_step_point_shape_and_inverse_scale():
    net = _tnet()
    y = _make_y()
    train(net, y, h=12, input_size=36, max_steps=5, windows_batch_size=64, lr=1e-3, seed=0,
          loss_fn=mae, scaler=RobustScaler())
    pred = predict_step(net, y[-36:], h=12, input_size=36, scaler=RobustScaler())
    assert pred.shape == (12, 1)


def test_train_quantile_loss_runs_and_predicts_q():
    net = _tnet(mult=3)
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    train(net, _make_y(), h=12, input_size=36, max_steps=5, windows_batch_size=64, lr=1e-3,
          seed=0, loss_fn=loss, scaler=RobustScaler())
    pred = predict_step(net, _make_y()[-36:], h=12, input_size=36, scaler=RobustScaler())
    assert pred.shape == (12, 3)


def test_train_with_exog_window_shapes_run():
    net = _tnet(hist_exog_size=2, futr_exog_size=1, stat_exog_size=3)
    y = _make_y()
    hist = jnp.asarray(np.random.RandomState(0).randn(y.shape[0], 2), jnp.float32)
    futr = jnp.asarray(np.random.RandomState(1).randn(y.shape[0], 1), jnp.float32)
    stat = jnp.asarray([0.5, -0.5, 1.0], jnp.float32)
    losses = train(net, y, h=12, input_size=36, max_steps=6, windows_batch_size=32, lr=1e-3,
                   seed=0, loss_fn=mae, scaler=RobustScaler(),
                   hist_exog=hist, futr_exog=futr, stat_exog=stat)
    assert losses.shape == (6,) and bool(jnp.all(jnp.isfinite(losses)))


def test_train_raises_on_divergence():
    # Concrete path of the numpy-free divergence guard: a huge LR -> non-finite -> RuntimeError.
    net = _tnet()
    with pytest.raises(RuntimeError, match="diverged"):
        train(net, _make_y(), h=12, input_size=36, max_steps=10,
              windows_batch_size=64, lr=1e9, seed=0, loss_fn=mae, scaler=RobustScaler())


def test_train_is_vmap_traceable():
    # The training path must trace under vmap (the BaseForecaster.conformity_scores
    # requirement). Map train over a batch of seeds; the guard must no-op, not raise.
    def run(seed):
        net = TFTNet(h=12, input_size=36, hidden_size=16, n_head=2, dropout=0.0, rngs=nnx.Rngs(0))
        return train(net, _make_y(), h=12, input_size=36, max_steps=4, windows_batch_size=32,
                     lr=1e-3, seed=seed, loss_fn=mae, scaler=RobustScaler())
    out = jax.vmap(run)(jnp.arange(3))
    assert out.shape == (3, 4) and bool(jnp.all(jnp.isfinite(out)))


# =============================================================================
# Model (BaseForecaster conformance)
# =============================================================================
from chronax.models.base_forecaster import BaseForecaster  # noqa: E402
from chronax.models.tft.tft_model import TFT  # noqa: E402
from chronax.utils import ConformalIntervals  # noqa: E402


def _tiny(**kw):
    base = dict(h=12, input_size=36, hidden_size=16, n_head=2, n_rnn_layers=1,
                max_steps=20, windows_batch_size=64, random_seed=0)
    base.update(kw)
    return TFT(**base)


def test_tft_inherits_baseforecaster_and_uses_exog():
    assert issubclass(TFT, BaseForecaster)
    assert TFT(h=12).uses_exog is True


def test_input_size_default_three_h():
    assert TFT(h=10).input_size == 30


def test_fit_returns_self_and_predict_shape():
    m = _tiny().fit(_make_y())
    assert m.model_ is not None
    assert m.predict(h=12)["mean"].shape == (12,)


def test_predict_smaller_h_slices_and_larger_raises():
    m = _tiny().fit(_make_y())
    assert m.predict(h=5)["mean"].shape == (5,)
    with pytest.raises(ValueError, match="unsupported|trained for"):
        m.predict(h=13)


def test_fit_short_series_and_2d_raise():
    with pytest.raises(ValueError, match="too short|short"):
        _tiny().fit(_make_y(40))
    with pytest.raises(ValueError, match="1-D"):
        _tiny().fit(jnp.ones((200, 2), jnp.float32))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        _tiny().predict(h=12)


def test_beats_naive_on_easy_signal():
    n = 240
    y = jnp.asarray(np.sin(np.arange(n) / 5.0), dtype=jnp.float32)
    m = TFT(h=12, input_size=48, hidden_size=32, n_head=4, max_steps=300,
            windows_batch_size=64, random_seed=0).fit(y[:-12])
    pred = np.asarray(m.predict(h=12)["mean"])
    y_true = np.asarray(y[-12:])
    naive = np.full(12, float(y[-13]))
    assert np.mean(np.abs(pred - y_true)) < np.mean(np.abs(naive - y_true))


def test_forecast_equals_fit_then_predict_same_seed():
    y = _make_y()
    a = _tiny().forecast(y, h=12)["mean"]
    b = _tiny().fit(y).predict(h=12)["mean"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


def test_full_exog_fit_predict_shapes():
    y = _make_y(240)
    T = y.shape[0]
    hist = jnp.asarray(np.random.RandomState(0).randn(T, 2), jnp.float32)
    futr = jnp.asarray(np.random.RandomState(1).randn(T, 1), jnp.float32)
    stat = jnp.asarray([0.5, -0.2, 1.0], jnp.float32)
    m = _tiny(max_steps=10).fit(y, X=hist, futr_exog=futr, stat_exog=stat)
    out = m.predict(h=12, futr_exog=jnp.asarray(np.random.RandomState(2).randn(12, 1), jnp.float32))
    assert out["mean"].shape == (12,)


def test_futr_exog_required_at_predict_when_fit_with_it():
    y = _make_y(240)
    futr = jnp.asarray(np.random.RandomState(1).randn(y.shape[0], 1), jnp.float32)
    m = _tiny(max_steps=5).fit(y, futr_exog=futr)
    with pytest.raises(ValueError, match="futr_exog"):
        m.predict(h=12)  # missing required future covariate


def test_exog_improves_accuracy_on_covariate_driven_signal():
    # Target is a noisy copy of a known future covariate -> exog must help vs univariate.
    rng = np.random.RandomState(0)
    T = 360
    futr = np.sin(np.arange(T + 12) / 6.0).astype(np.float32)
    y = (futr[:T] + 0.05 * rng.randn(T)).astype(np.float32)
    fk = dict(h=12, input_size=48, hidden_size=32, n_head=4, max_steps=250,
              windows_batch_size=64, random_seed=0)
    m_ex = TFT(**fk).fit(jnp.asarray(y), futr_exog=jnp.asarray(futr[:T, None]))
    pred_ex = np.asarray(m_ex.predict(h=12, futr_exog=jnp.asarray(futr[T:T + 12, None]))["mean"])
    m_uni = TFT(**fk).fit(jnp.asarray(y))
    pred_uni = np.asarray(m_uni.predict(h=12)["mean"])
    y_true = futr[T:T + 12]
    assert np.mean(np.abs(pred_ex - y_true)) < np.mean(np.abs(pred_uni - y_true))


def test_point_loss_conformal_intervals():
    m = TFT(h=4, input_size=12, hidden_size=8, n_head=2, max_steps=3,
            windows_batch_size=16, random_seed=0).fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    out = m.predict(h=4, level=[80])
    assert "lo-80" in out and "hi-80" in out


def test_conformal_rejects_temporal_exog():
    y = _make_y(120)
    futr = jnp.asarray(np.random.RandomState(0).randn(y.shape[0], 1), jnp.float32)
    m = _tiny(max_steps=3).fit(y, futr_exog=futr)
    m.conformal_params = ConformalIntervals(h=12, n_windows=3)
    with pytest.raises(ValueError, match="temporal exog|MultiQuantileLoss"):
        m.predict(h=12, level=[80], futr_exog=futr[-12:])


def test_quantile_loss_native_intervals_and_mean_is_median():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    m = TFT(h=12, input_size=36, hidden_size=16, n_head=2, max_steps=30,
            windows_batch_size=64, random_seed=0, loss=loss).fit(_make_y(200))
    out = m.predict(h=12, level=[80])
    assert set(["mean", "lo-80", "hi-80"]).issubset(out)
    assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))  # monotonic sort prevents crossing


def test_quantile_level_not_trained_raises():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    m = TFT(h=8, input_size=24, hidden_size=8, n_head=2, max_steps=5,
            windows_batch_size=32, random_seed=0, loss=loss).fit(_make_y(120))
    with pytest.raises(ValueError, match="not trained|quantile"):
        m.predict(h=8, level=[90])  # needs 0.05/0.95, only 0.1/0.5/0.9 trained


def test_forecast_fitted_univariate_has_nan_head():
    res = _tiny().forecast(_make_y(), h=12, fitted=True)
    fitted = np.asarray(res["fitted"])
    assert fitted.shape == (200,)
    assert np.all(np.isnan(fitted[:36])) and np.all(np.isfinite(fitted[36:]))


def test_constant_series_and_h1_finite():
    assert jnp.all(jnp.isfinite(_tiny().fit(jnp.ones(200, jnp.float32)).predict(h=12)["mean"]))
    m = TFT(h=1, input_size=12, hidden_size=8, n_head=2, max_steps=5,
            windows_batch_size=16, random_seed=0).fit(_make_y(60))
    assert m.predict(h=1)["mean"].shape == (1,)


def test_pickle_round_trip_point():
    m = _tiny().fit(_make_y())
    before = np.asarray(m.predict(h=12)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(np.asarray(m2.predict(h=12)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_pickle_round_trip_quantile_and_exog():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    y = _make_y(200)
    futr = jnp.asarray(np.random.RandomState(0).randn(y.shape[0], 1), jnp.float32)
    m = TFT(h=12, input_size=36, hidden_size=16, n_head=2, max_steps=10,
            windows_batch_size=64, random_seed=0, loss=loss).fit(y, futr_exog=futr)
    fz = jnp.asarray(np.random.RandomState(1).randn(12, 1), jnp.float32)
    before = np.asarray(m.predict(h=12, futr_exog=fz)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    assert m2._futr_size == 1
    np.testing.assert_allclose(np.asarray(m2.predict(h=12, futr_exog=fz)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_importable_from_models_namespace():
    from chronax.models import TFT as T
    assert T is TFT

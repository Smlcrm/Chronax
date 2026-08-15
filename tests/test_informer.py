"""Tests for chronax.models.Informer.

Covers the Informer subpackage in banner sections (Losses, Scaler, Layers,
Module, Training, Model, Namespace), matching the ``test_<model>.py``
convention used elsewhere in ``tests/``.
"""
import io
import logging
import math
import pickle
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx


# === Losses ===

from chronax.models.informer.informer_losses import (  # noqa: E402
    LOSSES, MultiQuantileLoss, huber, mae, mse, outputsize_multiplier, resolve,
)


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
    # QL = sum_q max(q*2, (q-1)*2) = sum of {0.2,1.0,1.8} = 3.0 — NF's effective
    # reduction (its 1/len(quantiles) factor is dead), matching the trainer's
    # masked inline branch.
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    pred = jnp.zeros((1, 1, 3))
    target = jnp.array([[2.0]])
    assert float(loss(pred, target)) == pytest.approx(3.0)


def test_mqloss_pickles():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    loss2 = pickle.loads(pickle.dumps(loss))
    assert loss2.quantiles == loss.quantiles


# === Scaler ===

from chronax.models.informer.informer_scaler import (  # noqa: E402
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


# === Layers ===

from chronax.models.informer.informer_layers import (  # noqa: E402
    AttentionLayer,
    ConvLayer,
    DataEmbedding,
    TimeFeatureEmbedding,
    TokenEmbedding,
    TransDecoderLayer,
    TransEncoderLayer,
    _KaimingNormalConvInit,
    _prob_attention,
    _resolve_activation,
    _torch_linear,
    sinusoid_position_embedding,
)


def test_torch_linear_init_bound():
    # torch nn.Linear default: weight+bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    in_f, out_f = 36, 32
    lin = _torch_linear(in_f, out_f, rngs=nnx.Rngs(0))
    bound = 1.0 / math.sqrt(in_f)
    k = np.asarray(lin.kernel.value)
    b = np.asarray(lin.bias.value)
    assert k.max() <= bound + 1e-6 and k.min() >= -bound - 1e-6
    assert b.max() <= bound + 1e-6 and b.min() >= -bound - 1e-6
    # uses most of the range (an unbounded normal init would not look like this)
    assert k.max() > 0.5 * bound and k.min() < -0.5 * bound


def test_kaiming_conv_init_std():
    fan_in = 12
    init = _KaimingNormalConvInit(fan_in)
    draws = init(jax.random.PRNGKey(0), (10_000,), jnp.float32)
    std = float(jnp.std(draws))
    assert std == pytest.approx(math.sqrt(2.0 / fan_in), rel=0.1)


def test_sinusoid_shape_and_row0():
    for L, hidden in [(10, 8), (10, 7)]:  # even and odd hidden_size
        pe = sinusoid_position_embedding(L, hidden)
        assert pe.shape == (L, hidden)
        assert bool(jnp.all(jnp.isfinite(pe)))
        expected_row0 = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(hidden)])
        np.testing.assert_allclose(np.asarray(pe[0]), expected_row0, atol=1e-6)


def test_token_embedding_circular_shift_equivariance():
    emb = TokenEmbedding(1, 8, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(2, 12, 1), jnp.float32)
    out, out_rolled = emb(x), emb(jnp.roll(x, 3, axis=1))
    np.testing.assert_allclose(np.asarray(jnp.roll(out, 3, axis=1)), np.asarray(out_rolled),
                               rtol=1e-5, atol=1e-5)   # circular conv commutes with roll


def test_time_feature_embedding_no_bias():
    B, L, input_size, hidden = 2, 5, 4, 8
    emb = TimeFeatureEmbedding(input_size, hidden, rngs=nnx.Rngs(0))
    assert emb.lin.use_bias is False
    assert emb.lin.bias is None
    x = jnp.asarray(np.random.RandomState(0).randn(B, L, input_size), jnp.float32)
    assert emb(x).shape == (B, L, hidden)


def test_data_embedding_with_and_without_marks():
    B, L, c_in, hidden, exog = 2, 10, 1, 8, 3
    x = jnp.asarray(np.random.RandomState(0).randn(B, L, c_in), jnp.float32)

    no_exog = DataEmbedding(c_in=c_in, exog_input_size=0, hidden_size=hidden, dropout=0.0, rngs=nnx.Rngs(0))
    assert no_exog.temporal_embedding is None
    out_no_mark = no_exog(x, None, True)
    assert out_no_mark.shape == (B, L, hidden)

    emb = DataEmbedding(c_in=c_in, exog_input_size=exog, hidden_size=hidden, dropout=0.0, rngs=nnx.Rngs(0))
    assert emb.temporal_embedding is not None
    mark_a = jnp.asarray(np.random.RandomState(1).randn(B, L, exog), jnp.float32)
    mark_b = jnp.asarray(np.random.RandomState(2).randn(B, L, exog), jnp.float32)
    out_a = emb(x, mark_a, True)
    out_b = emb(x, mark_b, True)
    assert out_a.shape == (B, L, hidden)
    assert not np.allclose(np.asarray(out_a), np.asarray(out_b))


def test_resolve_activation_exact_gelu():
    f = _resolve_activation("gelu")
    z = jnp.array([1.0])
    exact = 1.0 * 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    np.testing.assert_allclose(float(f(z)[0]), exact, rtol=1e-5)
    with pytest.raises(ValueError, match="Unknown activation"):
        _resolve_activation("swish")


def _dense_attn(q, k, v, causal=False):           # in-test reference, [B,H,L,E] layout
    scores = jnp.einsum("bhqe,bhke->bhqk", q, k) / math.sqrt(q.shape[-1])
    if causal:
        L_Q, L_K = scores.shape[-2:]
        scores = jnp.where(jnp.triu(jnp.ones((L_Q, L_K), bool), k=1), -1e9, scores)
    return jnp.einsum("bhqk,bhke->bhqe", jax.nn.softmax(scores, -1), v)


def test_prob_attention_unmasked_equals_dense_when_clamped():
    rs = np.random.RandomState(0)                  # L=8, factor=3 -> u=L_Q, U_part=L_K
    q, k, v = (jnp.asarray(rs.randn(2, 2, 8, 4), jnp.float32) for _ in range(3))
    out = _prob_attention(q, k, v, factor=3, mask_flag=False, key=jax.random.PRNGKey(1))
    np.testing.assert_allclose(np.asarray(out), np.asarray(_dense_attn(q, k, v)),
                               rtol=1e-5, atol=1e-5)


def test_prob_attention_masked_equals_causal_dense_when_clamped():
    rs = np.random.RandomState(0)                  # L=8, factor=3 -> u=L_Q, U_part=L_K
    q, k, v = (jnp.asarray(rs.randn(2, 2, 8, 4), jnp.float32) for _ in range(3))
    out = _prob_attention(q, k, v, factor=3, mask_flag=True, key=jax.random.PRNGKey(1))
    np.testing.assert_allclose(np.asarray(out), np.asarray(_dense_attn(q, k, v, causal=True)),
                               rtol=1e-5, atol=1e-5)


def test_prob_attention_cross_shapes():
    B, H, L_Q, L_K, E = 2, 2, 20, 36, 4
    rs = np.random.RandomState(1)
    q = jnp.asarray(rs.randn(B, H, L_Q, E), jnp.float32)
    k = jnp.asarray(rs.randn(B, H, L_K, E), jnp.float32)
    v = jnp.asarray(rs.randn(B, H, L_K, E), jnp.float32)
    out = _prob_attention(q, k, v, factor=3, mask_flag=False, key=jax.random.PRNGKey(2))
    assert out.shape == (B, H, L_Q, E)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_prob_attention_sparse_regime_key_determinism():
    # L=64, factor=3 -> u=15 < L_Q and U_part=15 < L_K: a genuine sparse regime (unlike
    # the clamped L=8 tests above). Here the top-u *query selection* (m_top) is ranked by
    # `m`, the max-minus-mean sparsity measure computed from `qk_sample` -- dot products
    # of every query against a RANDOM, causality-agnostic sample of ALL L_K keys
    # (including keys "in the future" of a given query row, even when mask_flag=True). So
    # changing a future k/v row can legitimately change which past query rows land in
    # m_top and get refined by the full-attention branch, changing their output value --
    # that is expected sampling variance, not a causality bug: whichever rows ARE refined
    # still only attend to causal keys (scores are masked with -1e9 for j>i before the
    # softmax). Causality is instead pinned EXACTLY (0.0 error) by
    # test_prob_attention_masked_equals_causal_dense_when_clamped, where u==L_Q and
    # U_part==L_K make selection total and thus independent of the sampling key. So this
    # test only checks: same key -> identical output; different key -> (legitimately)
    # different output; everything finite. No future-perturbation causality assert here.
    rs = np.random.RandomState(3)
    q, k, v = (jnp.asarray(rs.randn(2, 2, 64, 4), jnp.float32) for _ in range(3))
    key_a, key_b = jax.random.PRNGKey(10), jax.random.PRNGKey(11)
    out_a1 = _prob_attention(q, k, v, factor=3, mask_flag=True, key=key_a)
    out_a2 = _prob_attention(q, k, v, factor=3, mask_flag=True, key=key_a)
    out_b = _prob_attention(q, k, v, factor=3, mask_flag=True, key=key_b)
    np.testing.assert_array_equal(np.asarray(out_a1), np.asarray(out_a2))
    assert not np.allclose(np.asarray(out_a1), np.asarray(out_b))
    assert bool(jnp.all(jnp.isfinite(out_a1))) and bool(jnp.all(jnp.isfinite(out_b)))


def test_prob_attention_masked_differs_from_unmasked():
    rs = np.random.RandomState(4)
    q, k, v = (jnp.asarray(rs.randn(2, 2, 64, 4), jnp.float32) for _ in range(3))
    key = jax.random.PRNGKey(20)
    out_masked = _prob_attention(q, k, v, factor=3, mask_flag=True, key=key)
    out_unmasked = _prob_attention(q, k, v, factor=3, mask_flag=False, key=key)
    assert not np.allclose(np.asarray(out_masked), np.asarray(out_unmasked))
    assert bool(jnp.all(jnp.isfinite(out_masked)))
    assert bool(jnp.all(jnp.isfinite(out_unmasked)))


def test_attention_layer_matches_manual_dense_in_clamped_regime():
    # n_head=2, L=8, factor=3 -> clamped regime (see above tests): _prob_attention's
    # output per head exactly equals dense attention regardless of sample_key. The
    # manual reference reassembles heads in the ALIGNED convention (transpose to
    # [B,L,H,E] before flattening), so the layer is pinned to
    # attention_mixing="official" here — the "nf" default flattens head-major
    # memory directly and is a genuinely different function at n_head > 1
    # (guarded by test_attention_mixing_modes_differ_at_multi_head).
    B, L, hidden, n_head = 2, 8, 8, 2
    layer = AttentionLayer(hidden_size=hidden, n_head=n_head, factor=3, mask_flag=False,
                           mixing="official", rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(5).randn(B, L, hidden), jnp.float32)
    out = layer(x, x, x, sample_key=jax.random.PRNGKey(7))

    def split(t):
        E = hidden // n_head
        return t.reshape(B, L, n_head, E).transpose(0, 2, 1, 3)   # [B,H,L,E]

    q, k, v = split(layer.w_q(x)), split(layer.w_k(x)), split(layer.w_v(x))
    ctx = _dense_attn(q, k, v)                                # [B,H,L,E] (clamped == prob attn exactly)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, L, hidden)      # [B,L,H,E] -> [B,L,hid]; NOT a raw [B,H,L,E] flatten
    expected = layer.w_o(ctx)

    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_attention_layer_divisibility_error():
    with pytest.raises(ValueError, match="divisible"):
        AttentionLayer(hidden_size=10, n_head=3, factor=3, mask_flag=False, rngs=nnx.Rngs(0))


def test_conv_layer_halves_length():
    # Distilling conv, NF-parity: torch Conv1d(k=3, padding=2, circular) EXPANDS the
    # sequence to L+2 (NF pins padding=2; the paper repo's padding is torch-version-
    # conditional), then BN -> ELU -> maxpool(k=3,s=2,pad=1) gives (L+1)//2 + 1.
    # Exercised for even, odd, and small L.
    c_in = 4
    layer = ConvLayer(c_in, rngs=nnx.Rngs(0))
    for L, expected in [(12, 7), (13, 8), (7, 5)]:
        x = jnp.asarray(np.random.RandomState(0).randn(2, L, c_in), jnp.float32)
        out = layer(x, use_running_average=False)
        assert out.shape == (2, expected, c_in)
        assert bool(jnp.all(jnp.isfinite(out)))


def test_conv_layer_gemm_matches_conv_primitive():
    # The distil conv is expressed as 3 shifted GEMMs, because XLA-CPU lowers
    # `convolution` at these shapes to its naive emitter inside the training scan.
    # This pins the GEMM form to the conv primitive on the same padded input
    # and parameters.
    c_in = 16
    layer = ConvLayer(c_in, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 30, c_in), dtype=jnp.float32)
    out = layer(x, use_running_average=True)
    xp = jnp.concatenate([x[:, -2:], x, x[:, :2]], axis=1)
    ref = jax.lax.conv_general_dilated(
        xp, layer.conv.kernel.value, window_strides=(1,), padding="VALID",
        dimension_numbers=("NHC", "HIO", "NHC")) + layer.conv.bias.value
    ref = layer.norm(ref, use_running_average=True)
    ref = jax.nn.elu(ref)
    ref = nnx.max_pool(ref, window_shape=(3,), strides=(2,), padding=((1, 1),))
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-4)


def test_conv_layer_batchstat_updates():
    c_in = 4
    layer = ConvLayer(c_in, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(1).randn(8, 12, c_in), jnp.float32)

    before = np.asarray(layer.norm.mean.value)
    layer(x, use_running_average=False)
    after_train = np.asarray(layer.norm.mean.value)
    assert not np.allclose(before, after_train)   # training call mutates the running mean

    frozen_before = np.asarray(layer.norm.mean.value)
    layer(x, use_running_average=True)
    frozen_after = np.asarray(layer.norm.mean.value)
    np.testing.assert_array_equal(frozen_before, frozen_after)   # eval call freezes it


def test_encoder_layer_shape():
    B, L, hidden = 2, 12, 16
    layer = TransEncoderLayer(
        hidden_size=hidden, n_head=4, conv_hidden_size=32, factor=3, dropout=0.1, rngs=nnx.Rngs(0),
    )
    x = jnp.asarray(np.random.RandomState(2).randn(B, L, hidden), jnp.float32)
    out = layer(x, sample_key=jax.random.PRNGKey(0), deterministic=True)
    assert out.shape == (B, L, hidden)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_decoder_layer_shape():
    B, hidden = 2, 16
    layer = TransDecoderLayer(
        hidden_size=hidden, n_head=4, conv_hidden_size=32, factor=3, dropout=0.1, rngs=nnx.Rngs(0),
    )
    x = jnp.asarray(np.random.RandomState(3).randn(B, 7, hidden), jnp.float32)
    cross = jnp.asarray(np.random.RandomState(4).randn(B, 5, hidden), jnp.float32)
    out = layer(x, cross, self_key=jax.random.PRNGKey(1), cross_key=jax.random.PRNGKey(2), deterministic=True)
    assert out.shape == (B, 7, hidden)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_decoder_layer_uses_cross():
    B, hidden = 2, 16
    layer = TransDecoderLayer(
        hidden_size=hidden, n_head=4, conv_hidden_size=32, factor=3, dropout=0.1, rngs=nnx.Rngs(0),
    )
    x = jnp.asarray(np.random.RandomState(5).randn(B, 7, hidden), jnp.float32)
    cross_a = jnp.asarray(np.random.RandomState(6).randn(B, 5, hidden), jnp.float32)
    cross_b = jnp.asarray(np.random.RandomState(7).randn(B, 5, hidden), jnp.float32)
    out_a = layer(x, cross_a, self_key=jax.random.PRNGKey(1), cross_key=jax.random.PRNGKey(2), deterministic=True)
    out_b = layer(x, cross_b, self_key=jax.random.PRNGKey(1), cross_key=jax.random.PRNGKey(2), deterministic=True)
    assert not np.allclose(np.asarray(out_a), np.asarray(out_b))


# === Module ===

from chronax.models.informer.informer_module import (  # noqa: E402
    InformerNet, TransEncoder, distilled_length,
)


def test_distilled_length():
    # NF-parity: each conv expands L -> L+2 (circular padding=2) before the
    # stride-2 pool, so one distil step maps L -> (L+1)//2 + 1.
    assert distilled_length(72, 1) == 37
    assert distilled_length(13, 1) == 8
    assert distilled_length(72, 2) == 20


def test_encoder_distil_output_length():
    B, L, hidden = 2, 36, 16
    for encoder_layers in (2, 3):
        enc = TransEncoder(
            encoder_layers=encoder_layers, hidden_size=hidden, n_head=2, conv_hidden_size=8,
            factor=3, dropout=0.0, distil=True, rngs=nnx.Rngs(0),
        )
        assert len(enc.conv_layers) == encoder_layers - 1   # last attn layer has no conv after it
        if encoder_layers == 3:
            # Regression guard: conv_layers must be independently constructed instances,
            # not `[ConvLayer(...)] * n` aliasing (which would silently share params).
            assert enc.conv_layers[0] is not enc.conv_layers[1]
        x = jnp.asarray(np.random.RandomState(0).randn(B, L, hidden), jnp.float32)
        keys = jax.random.split(jax.random.PRNGKey(0), encoder_layers)
        out = enc(x, sample_keys=keys, deterministic=True, use_running_average=True)
        expected_L = distilled_length(L, encoder_layers - 1)
        assert out.shape == (B, expected_L, hidden)
        assert bool(jnp.all(jnp.isfinite(out)))


def test_encoder_no_distil_preserves_length():
    B, L, hidden, encoder_layers = 2, 36, 16, 3
    enc = TransEncoder(
        encoder_layers=encoder_layers, hidden_size=hidden, n_head=2, conv_hidden_size=8,
        factor=3, dropout=0.0, distil=False, rngs=nnx.Rngs(0),
    )
    assert enc.conv_layers == []
    x = jnp.asarray(np.random.RandomState(1).randn(B, L, hidden), jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(0), encoder_layers)
    out = enc(x, sample_keys=keys, deterministic=True, use_running_average=True)
    assert out.shape == (B, L, hidden)
    assert bool(jnp.all(jnp.isfinite(out)))


def _net(h=12, input_size=36, label_len=18, futr_exog_size=0, outputsize_multiplier=1,
         encoder_layers=2, decoder_layers=1, distil=True, n_head=2, seed=0):
    return InformerNet(
        h=h, input_size=input_size, label_len=label_len, hidden_size=16, n_head=n_head,
        factor=3, conv_hidden_size=8, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        distil=distil, dropout=0.0, activation="gelu", futr_exog_size=futr_exog_size,
        outputsize_multiplier=outputsize_multiplier, rngs=nnx.Rngs(seed),
    )


def test_net_univariate_shape():
    net = _net()
    x = jnp.asarray(np.random.RandomState(0).randn(3, 36, 1), jnp.float32)
    out = net(x, sample_key=jax.random.PRNGKey(0), deterministic=True, use_running_average=True)
    assert out.shape == (3, 12, 1)
    assert out.dtype == jnp.float32


def test_net_futr_shape_and_effect():
    h, input_size, label_len = 12, 36, 18
    net = _net(h=h, input_size=input_size, label_len=label_len, futr_exog_size=2)
    x = jnp.asarray(np.random.RandomState(1).randn(2, input_size, 1), jnp.float32)
    futr_a = jnp.asarray(np.random.RandomState(2).randn(2, input_size + h, 2), jnp.float32)
    futr_b = jnp.asarray(np.random.RandomState(3).randn(2, input_size + h, 2), jnp.float32)
    out_a = net(x, futr_exog=futr_a, sample_key=jax.random.PRNGKey(0),
                deterministic=True, use_running_average=True)
    out_b = net(x, futr_exog=futr_b, sample_key=jax.random.PRNGKey(0),
                deterministic=True, use_running_average=True)
    assert out_a.shape == (2, h, 1)
    assert bool(jnp.all(jnp.isfinite(out_a)))
    assert not np.allclose(np.asarray(out_a), np.asarray(out_b))


def test_net_quantile_head_width():
    h = 12
    net = _net(h=h, outputsize_multiplier=3)
    x = jnp.asarray(np.random.RandomState(4).randn(2, 36, 1), jnp.float32)
    out = net(x, sample_key=jax.random.PRNGKey(0), deterministic=True, use_running_average=True)
    assert out.shape == (2, h, 3)


def test_net_key_determinism():
    # input_size=48, factor=3: every ProbSparse attention site in this net (both encoder
    # layers -- pre- and post-distil -- and the decoder's self/cross attention) lands in
    # the genuinely sparse regime (factor*ceil(ln L) < L), so the sample_key actually
    # changes which rows get refined. With deterministic=True and use_running_average=True
    # there is no other source of randomness, so the same key must reproduce bit-identical
    # output, while different keys should (and do, with overwhelming probability) differ.
    h, input_size, label_len = 12, 48, 24
    net = _net(h=h, input_size=input_size, label_len=label_len)
    x = jnp.asarray(np.random.RandomState(5).randn(2, input_size, 1), jnp.float32)
    key_a, key_b = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
    out_a1 = net(x, sample_key=key_a, deterministic=True, use_running_average=True)
    out_a2 = net(x, sample_key=key_a, deterministic=True, use_running_average=True)
    out_b = net(x, sample_key=key_b, deterministic=True, use_running_average=True)
    np.testing.assert_array_equal(np.asarray(out_a1), np.asarray(out_a2))
    assert not np.allclose(np.asarray(out_a1), np.asarray(out_b))


def test_net_bad_heads_raises():
    with pytest.raises(ValueError, match="divisible"):
        InformerNet(
            h=12, input_size=36, label_len=18, hidden_size=15, n_head=2,
            conv_hidden_size=8, rngs=nnx.Rngs(0),
        )


# === Training ===

from chronax.models.informer.informer_training import (  # noqa: E402
    build_exog_windows,
    build_windows,
    forward_loss,
    predict_step,
    train,
)


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


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
    with pytest.raises(ValueError, match="too short"):
        build_windows(_make_y(30), 36, 12)


def test_build_exog_windows_full_span_shape():
    n = 60 - 36
    arr = jnp.asarray(np.random.RandomState(0).randn(60, 2), jnp.float32)
    assert build_exog_windows(arr, 36, 12, n, "full").shape == (n, 48, 2)


def test_exog_scaling_stats_use_insample_span_only():
    # NF's _normalization masks the horizon out of the scaler stats: exog
    # statistics come from the insample span only; the horizon slice is
    # transformed with those stats but never feeds them.
    from chronax.models.informer.informer_training import _scale_exog
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
    net = _net()
    y = jnp.asarray(np.arange(1.0, 61.0), jnp.float32)   # T=60
    w, m = build_windows(y, 36, 12)
    kw = dict(h=12, input_size=36, scaler=RobustScaler(), loss_fn=mae,
              sample_key=jax.random.PRNGKey(0), deterministic=True)
    full = forward_loss(net, w, m, **kw)
    w_poison = w.at[-1, -1].set(1e6)     # last target cell of the last window is padded
    poisoned = forward_loss(net, w_poison, m, **kw)
    np.testing.assert_allclose(float(full), float(poisoned), rtol=0, atol=1e-6)


def test_forward_loss_scalar_finite():
    net = _net()
    w, _ = build_windows(_make_y(), 36, 12)
    loss = forward_loss(net, w[:8], h=12, input_size=36, scaler=RobustScaler(),
                        loss_fn=mae, sample_key=jax.random.PRNGKey(0), deterministic=True)
    assert loss.shape == () and bool(jnp.isfinite(loss))


def test_same_config_nets_share_graphdef():
    # Value-__eq__ on the picklable initializers makes same-config graphdefs
    # EQUAL, so the module-level @nnx.jit _forward_det cache hits across nets
    # built by a later fit instead of recompiling per instance.
    gd_a, _ = nnx.split(_net())
    gd_b, _ = nnx.split(_net())
    assert gd_a == gd_b


def test_train_decreases_loss():
    net = _net()
    losses = train(net, _make_y(), h=12, input_size=36, max_steps=40,
                   windows_batch_size=64, lr=1e-3, seed=0, loss_fn=mae, scaler=RobustScaler())
    assert losses.shape == (40,)
    assert float(losses[-1]) < float(losses[0])


def test_train_deterministic_same_seed():
    a = train(_net(), _make_y(), h=12, input_size=36, max_steps=10, windows_batch_size=64,
              lr=1e-3, seed=0, loss_fn=mae, scaler=RobustScaler())
    b = train(_net(), _make_y(), h=12, input_size=36, max_steps=10, windows_batch_size=64,
              lr=1e-3, seed=0, loss_fn=mae, scaler=RobustScaler())
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5)


def test_train_with_futr_exog_finite():
    net = _net(futr_exog_size=2)
    y = _make_y()
    futr = jnp.asarray(np.random.RandomState(1).randn(y.shape[0], 2), jnp.float32)
    losses = train(net, y, h=12, input_size=36, max_steps=6, windows_batch_size=32, lr=1e-3,
                   seed=0, loss_fn=mae, scaler=RobustScaler(), futr_exog=futr)
    assert losses.shape == (6,) and bool(jnp.all(jnp.isfinite(losses)))


def test_train_quantile_loss_and_predict_step_shape():
    net = _net(outputsize_multiplier=3)
    loss = MultiQuantileLoss()
    train(net, _make_y(), h=12, input_size=36, max_steps=5, windows_batch_size=64, lr=1e-3,
          seed=0, loss_fn=loss, scaler=RobustScaler())
    pred = predict_step(net, _make_y()[-36:], h=12, input_size=36, scaler=RobustScaler())
    assert pred.shape == (12, 3)


def test_train_divergence_raises():
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
        net = InformerNet(h=12, input_size=36, label_len=18, hidden_size=16, n_head=2,
                          conv_hidden_size=8, dropout=0.0, rngs=nnx.Rngs(0))
        return train(net, y, h=12, input_size=36, max_steps=4, windows_batch_size=16,
                     lr=1e-3, seed=seed, loss_fn=resolve("mae"), scaler=resolve_scaler("identity"))
    out = jax.vmap(run)(jnp.arange(3))
    assert out.shape == (3, 4) and bool(jnp.all(jnp.isfinite(out)))


# === Model ===

from chronax.models.base_forecaster import BaseForecaster  # noqa: E402
from chronax.models.informer.informer_model import Informer  # noqa: E402
from chronax.utils import ConformalIntervals  # noqa: E402


def _tiny(**kw):
    base = dict(h=12, input_size=36, hidden_size=16, n_head=2, conv_hidden_size=8,
                max_steps=20, windows_batch_size=64, random_seed=0)
    base.update(kw)
    return Informer(**base)


def test_informer_is_base_forecaster_and_uses_exog():
    assert issubclass(Informer, BaseForecaster)
    assert Informer(h=12).uses_exog is True


def test_input_size_default_and_label_len():
    m = Informer(h=12)
    assert m.input_size == 36
    assert m.label_len == 18


def test_bad_multiplier_raises():
    with pytest.raises(ValueError, match="decoder_input_size_multiplier"):
        Informer(h=12, decoder_input_size_multiplier=1.0)
    with pytest.raises(ValueError, match="decoder_input_size_multiplier"):
        Informer(h=12, decoder_input_size_multiplier=0.0)


def test_fit_predict_shapes():
    m = _tiny().fit(_make_y())
    assert m.model_ is not None
    assert m.predict(h=12)["mean"].shape == (12,)


def test_predict_smaller_h_slices_larger_raises():
    m = _tiny().fit(_make_y())
    assert m.predict(h=5)["mean"].shape == (5,)
    with pytest.raises(ValueError, match="unsupported|trained for"):
        m.predict(h=13)


def test_short_series_and_2d_raise():
    with pytest.raises(ValueError, match="too short|short"):
        _tiny().fit(_make_y(36))  # T == input_size: zero windows
    with pytest.raises(ValueError, match="1-D"):
        _tiny().fit(jnp.ones((200, 2), jnp.float32))


def test_fit_at_minimum_length_nf_parity():
    # NF trains on h-padded partial windows: one window (T = input_size+1) is
    # enough to fit. The old gate demanded T >= input_size+h.
    m = _tiny().fit(_make_y(37))
    pred = m.predict(h=12)["mean"]
    assert pred.shape == (12,)
    assert bool(jnp.all(jnp.isfinite(pred)))


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
    # must report >=1. Guards test_refit_does_not_recompile against going
    # vacuously green if a jax bump renames the log message or logger paths.
    def fresh():
        @jax.jit
        def f(x):
            return x * 5.0 - 2.0
        return f(jnp.ones((4, 3)))

    _, n = _count_compiles(fresh)
    assert n >= 1


def test_refit_does_not_recompile():
    # The training program is a module-level nnx.jit keyed on config graphdefs
    # (value-__eq__ initializers, _adam/scaler singletons) and operand shapes —
    # a refit AND a fresh same-config instance must hit the cache with zero XLA
    # compiles; fit #2 is counted directly (no uncounted settle call).
    y = _make_y(60)
    m = _tiny(max_steps=3, windows_batch_size=4)
    m.fit(y)                                     # first fit pays the one compile
    _, n_refit = _count_compiles(lambda: m.fit(y)._train_y)
    assert n_refit == 0
    m2 = _tiny(max_steps=3, windows_batch_size=4)
    _, n_fresh = _count_compiles(lambda: m2.fit(y)._train_y)
    assert n_fresh == 0


def test_attention_mixing_modes_identical_at_single_head():
    # At n_head == 1 the official transpose is a no-op, so both flatten
    # conventions must produce the same forward.
    from chronax.models.informer.informer_module import InformerNet

    x = jnp.asarray(np.random.RandomState(0).randn(2, 12, 1), jnp.float32)
    outs = []
    for mode in ("nf", "official"):
        net = InformerNet(h=4, input_size=12, label_len=6, hidden_size=16, n_head=1,
                          conv_hidden_size=8, dropout=0.0, attention_mixing=mode,
                          rngs=nnx.Rngs(0))
        outs.append(np.asarray(net(x, sample_key=jax.random.PRNGKey(0),
                                   deterministic=True, use_running_average=True)))
    np.testing.assert_array_equal(outs[0], outs[1])


def test_attention_mixing_modes_differ_at_multi_head():
    # At n_head > 1 the nf flatten reinterprets head-major memory — the two
    # conventions are genuinely different functions of the same weights.
    from chronax.models.informer.informer_module import InformerNet

    x = jnp.asarray(np.random.RandomState(0).randn(2, 12, 1), jnp.float32)
    outs = []
    for mode in ("nf", "official"):
        net = InformerNet(h=4, input_size=12, label_len=6, hidden_size=16, n_head=2,
                          conv_hidden_size=8, dropout=0.0, attention_mixing=mode,
                          rngs=nnx.Rngs(0))
        outs.append(np.asarray(net(x, sample_key=jax.random.PRNGKey(0),
                                   deterministic=True, use_running_average=True)))
    assert not np.allclose(outs[0], outs[1])


def test_attention_mixing_pickle_roundtrip_and_legacy_default():
    m = _tiny(attention_mixing="official", max_steps=3, windows_batch_size=4).fit(_make_y(60))
    m2 = pickle.loads(pickle.dumps(m))
    assert m2.attention_mixing == "official"
    np.testing.assert_array_equal(np.asarray(m.predict(h=12)["mean"]),
                                  np.asarray(m2.predict(h=12)["mean"]))
    # Estimators pickled before the flag existed carry no attention_mixing key;
    # they were built with the official transpose and must restore that way.
    state = m.__getstate__()
    del state["attention_mixing"]
    legacy = Informer.__new__(Informer)
    legacy.__setstate__(state)
    assert legacy.attention_mixing == "official"


def test_dropout_masks_fresh_per_scan_step_and_det_repeatable():
    # Dropout RNG must advance per nnx.scan step even at a FIXED ProbSparse
    # sample_key (the streams are independent); and the deterministic path with
    # a fixed key must be exactly repeatable (the _forward_det contract).
    from chronax.models.informer.informer_module import InformerNet

    net = InformerNet(h=4, input_size=12, label_len=6, hidden_size=16, n_head=2,
                      conv_hidden_size=8, dropout=0.5, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 12, 1))
    fixed = jax.random.PRNGKey(7)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, _):
        return carry, carry(x, sample_key=fixed, deterministic=False,
                            use_running_average=False)

    _, outs = step(net, jnp.arange(3))
    assert not np.allclose(np.asarray(outs[0]), np.asarray(outs[1]))

    a = net(x, sample_key=fixed, deterministic=True, use_running_average=True)
    b = net(x, sample_key=fixed, deterministic=True, use_running_average=True)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


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
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


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


def test_distil_false_end_to_end():
    m = _tiny(distil=False).fit(_make_y())
    pred = m.predict(h=12)["mean"]
    assert pred.shape == (12,)
    assert bool(jnp.all(jnp.isfinite(pred)))


def test_futr_exog_fit_predict_shapes():
    y = _make_y(240)
    T = y.shape[0]
    futr = jnp.asarray(np.random.RandomState(1).randn(T, 2), jnp.float32)
    m = _tiny(max_steps=10).fit(y, futr_exog=futr)
    out = m.predict(h=12, futr_exog=jnp.asarray(np.random.RandomState(2).randn(12, 2), jnp.float32))
    assert out["mean"].shape == (12,)
    assert bool(jnp.all(jnp.isfinite(out["mean"])))


def test_futr_required_at_predict_raises():
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
    fk = dict(h=12, input_size=48, hidden_size=32, n_head=4, max_steps=400,
              windows_batch_size=64, random_seed=0)
    m_ex = Informer(**fk).fit(jnp.asarray(y), futr_exog=jnp.asarray(futr[:T, None]))
    pred_ex = np.asarray(m_ex.predict(h=12, futr_exog=jnp.asarray(futr[T:T + 12, None]))["mean"])
    m_uni = Informer(**fk).fit(jnp.asarray(y))
    pred_uni = np.asarray(m_uni.predict(h=12)["mean"])
    y_true = futr[T:T + 12]
    assert np.mean(np.abs(pred_ex - y_true)) < np.mean(np.abs(pred_uni - y_true))


def test_conformal_interval_keys():
    m = Informer(h=4, input_size=12, hidden_size=8, n_head=2, conv_hidden_size=8,
                 max_steps=3, windows_batch_size=16, random_seed=0).fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    out = m.predict(h=4, level=[80])
    assert "lo-80" in out and "hi-80" in out
    assert bool(jnp.all(out["lo-80"] <= out["mean"]))
    assert bool(jnp.all(out["mean"] <= out["hi-80"]))


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
    m = Informer(h=12, input_size=36, hidden_size=16, n_head=2, conv_hidden_size=8,
                 max_steps=30, windows_batch_size=64, random_seed=0, loss=loss).fit(_make_y(200))
    out = m.predict(h=12, level=[80])
    assert set(["mean", "lo-80", "hi-80"]).issubset(out)
    assert bool(jnp.all(out["lo-80"] <= out["mean"]))
    assert bool(jnp.all(out["mean"] <= out["hi-80"]))


def test_untrained_quantile_level_raises():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    m = Informer(h=8, input_size=24, hidden_size=8, n_head=2, conv_hidden_size=8,
                 max_steps=5, windows_batch_size=32, random_seed=0, loss=loss).fit(_make_y(120))
    with pytest.raises(ValueError, match="not trained|quantile"):
        m.predict(h=8, level=[90])  # needs 0.05/0.95, only 0.1/0.5/0.9 trained


def test_pickle_roundtrip_point():
    m = _tiny().fit(_make_y())
    before = np.asarray(m.predict(h=12)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(np.asarray(m2.predict(h=12)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_pickle_roundtrip_quantile_exog():
    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    y = _make_y(200)
    futr = jnp.asarray(np.random.RandomState(0).randn(y.shape[0], 1), jnp.float32)
    m = Informer(h=12, input_size=36, hidden_size=16, n_head=2, conv_hidden_size=8,
                 max_steps=10, windows_batch_size=64, random_seed=0, loss=loss).fit(y, futr_exog=futr)
    fz = jnp.asarray(np.random.RandomState(1).randn(12, 1), jnp.float32)
    before = np.asarray(m.predict(h=12, futr_exog=fz)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    assert m2._futr_size == 1
    np.testing.assert_allclose(np.asarray(m2.predict(h=12, futr_exog=fz)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_fitted_with_exog_not_implemented():
    y = _make_y(200)
    futr = jnp.asarray(np.random.RandomState(0).randn(y.shape[0], 1), jnp.float32)
    m = _tiny(max_steps=5)
    with pytest.raises(NotImplementedError, match="exog"):
        m.forecast(y, h=12, X_future=futr[-12:], futr_exog=futr, fitted=True)


def test_vmap_forecast_matches_python_loop():
    # conformity_scores vmaps forecast over windows
    B, T, h = 4, 120, 6
    y_batch = jnp.stack([_make_y(T) * (i + 1) for i in range(B)])
    m = Informer(h=h, input_size=18, hidden_size=8, n_head=2, conv_hidden_size=8,
                 max_steps=4, windows_batch_size=16, random_seed=0)
    seq = jnp.stack([m.forecast(y=y_batch[i], h=h)["mean"] for i in range(B)])
    vm = jax.vmap(lambda y: m.forecast(y=y, h=h)["mean"])(y_batch)
    np.testing.assert_allclose(np.asarray(seq), np.asarray(vm), rtol=5e-3, atol=5e-3)


def test_beats_naive_on_easy_signal():
    n = 240
    y = jnp.asarray(np.sin(np.arange(n) / 5.0), dtype=jnp.float32)
    m = Informer(h=12, input_size=48, hidden_size=32, max_steps=300,
                windows_batch_size=64, random_seed=0).fit(y[:-12])
    pred = np.asarray(m.predict(h=12)["mean"])
    y_true = np.asarray(y[-12:])
    naive = np.full(12, float(y[-13]))
    assert np.mean(np.abs(pred - y_true)) < np.mean(np.abs(naive - y_true))


# === Namespace ===

def test_importable_from_models_namespace():
    import chronax.models
    from chronax.models import Informer as PublicInformer

    assert PublicInformer is Informer
    assert "Informer" in chronax.models.__all__

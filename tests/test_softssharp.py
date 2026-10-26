"""Tests for chronax.models.SOFTSSharp.

Covers the four source modules of the softssharp subpackage in one file (losses,
module, training, model), matching the flat ``test_<model>.py`` convention used
elsewhere in ``tests/``. SOFTSSharp is SOFTS with exactly two changes inside the
series-core fusion block — a stochastic variable-position sinusoidal encoding and
three extra dropout layers — so the module-level tests here concentrate on
STADSharp's position table, its Bernoulli gate, and the train/eval split; the rest
mirrors ``tests/test_softs.py``.
"""
import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.softssharp.softssharp_losses import (
    LOSSES, huber, mae, mse, resolve,
)
from chronax.models.softssharp.softssharp_module import (
    DataEmbeddingInverted,
    PositionalEmbedding,
    RevIN,
    SOFTSSharpNet,
    STADSharp,
    TransEncoder,
    TransEncoderLayer,
    _gelu,
    _resolve_activation,
    positional_table,
)
from chronax.models.softssharp.softssharp_model import (
    SOFTSSharp, _boxcox, _inv_boxcox, _select_boxcox_lambda,
)
from chronax.models.softssharp.softssharp_training import (
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
    # SOFTSSharp inherits SOFTS's per-window MEAN (subtract_last=False).
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
    B, L, N, hidden = 4, 24, 1, 16
    emb = DataEmbeddingInverted(input_size=L, hidden_size=hidden, dropout=0.0, rngs=nnx.Rngs(0))
    x = jnp.ones((B, L, N), dtype=jnp.float32)
    out = emb(x, deterministic=True)
    assert out.shape == (B, N, hidden)   # variates become tokens
    assert out.dtype == jnp.float32


def test_embedding_multivariate_token_count_equals_n_series():
    B, L, N, hidden = 2, 24, 5, 16
    emb = DataEmbeddingInverted(input_size=L, hidden_size=hidden, dropout=0.0, rngs=nnx.Rngs(0))
    out = emb(jnp.ones((B, L, N), dtype=jnp.float32), deterministic=True)
    assert out.shape == (B, N, hidden)


# ---- positional table (the new sinusoidal buffer) ---------------------------

def test_positional_table_shape_and_dtype():
    t = positional_table(8, 32)
    assert t.shape == (1, 32, 8)
    assert t.dtype == np.float32


def test_positional_table_is_host_numpy_not_jax():
    # Deliberate: a cached jnp value first built inside the training nnx.scan
    # trace would leak a DynamicJaxprTracer to every later call.
    t = positional_table(8, 32)
    assert isinstance(t, np.ndarray)
    assert not isinstance(t, jnp.ndarray)


def test_positional_table_row_zero_is_sin_cos_of_zero():
    t = positional_table(8, 32)
    np.testing.assert_allclose(t[0, 0], np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32))


def test_positional_table_even_columns_are_sines():
    d, max_len = 8, 16
    t = positional_table(d, max_len)[0]
    pos = np.arange(max_len, dtype=np.float32)
    for j, col in enumerate(range(0, d, 2)):
        w = math.exp(-col * math.log(10000.0) / d)
        np.testing.assert_allclose(t[:, col], np.sin(pos * w), rtol=1e-5, atol=1e-6)


def test_positional_table_odd_columns_are_cosines():
    d, max_len = 8, 16
    t = positional_table(d, max_len)[0]
    pos = np.arange(max_len, dtype=np.float32)
    for j, col in enumerate(range(1, d, 2)):
        w = math.exp(-(2 * j) * math.log(10000.0) / d)
        np.testing.assert_allclose(t[:, col], np.cos(pos * w), rtol=1e-5, atol=1e-6)


def test_positional_table_is_lru_cached():
    assert positional_table(12, 40) is positional_table(12, 40)


def test_positional_table_is_read_only():
    # Cached and shared across every module instance — mutation would corrupt all.
    t = positional_table(12, 40)
    assert t.flags.writeable is False
    with pytest.raises(ValueError):
        t[0, 0, 0] = 1.0


@pytest.mark.parametrize("d", [1, 3, 5, 7])
def test_positional_table_odd_d_series_truncates_cosine_half(d):
    # NF assumes an even d_series; here the cosine half is truncated to d // 2
    # columns so an odd hidden_size degrades instead of crashing.
    max_len = 8
    t = positional_table(d, max_len)[0]
    assert t.shape == (max_len, d)
    pos = np.arange(max_len, dtype=np.float32)
    for j, col in enumerate(range(0, d, 2)):
        w = math.exp(-col * math.log(10000.0) / d)
        np.testing.assert_allclose(t[:, col], np.sin(pos * w), rtol=1e-5, atol=1e-6)
    for j, col in enumerate(range(1, d, 2)):
        w = math.exp(-(2 * j) * math.log(10000.0) / d)
        np.testing.assert_allclose(t[:, col], np.cos(pos * w), rtol=1e-5, atol=1e-6)


def test_positional_table_rejects_non_positive_d_series():
    with pytest.raises(ValueError, match="d_series"):
        positional_table(0, 8)


def test_positional_table_rejects_non_positive_max_len():
    with pytest.raises(ValueError, match="max_len"):
        positional_table(8, 0)


def test_positional_embedding_adds_scaled_table():
    d, C = 8, 5
    pe = PositionalEmbedding(d, max_len=32)
    x = jnp.asarray(np.random.RandomState(0).randn(2, C, d), dtype=jnp.float32)
    out = np.asarray(pe(x, scale=0.25))
    expected = np.asarray(x) + 0.25 * positional_table(d, 32)[:, :C, :]
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_positional_embedding_scale_zero_is_identity():
    pe = PositionalEmbedding(8, max_len=32)
    x = jnp.asarray(np.random.RandomState(1).randn(2, 4, 8), dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(pe(x, scale=0.0)), np.asarray(x), rtol=1e-6)


def test_positional_embedding_raises_when_tokens_exceed_max_len():
    pe = PositionalEmbedding(8, max_len=3)
    x = jnp.ones((2, 4, 8), dtype=jnp.float32)
    with pytest.raises(ValueError, match="max_len"):
        pe(x)


# ---- STADSharp (series-core fusion + stochastic position encoding) ----------

def _stad(hidden=16, d_core=8, dropout=0.0, pe_keep_prob=0.5, seed=0):
    return STADSharp(hidden_size=hidden, d_core=d_core, dropout=dropout,
                     pe_keep_prob=pe_keep_prob, rngs=nnx.Rngs(seed))


def _manual_stad_forward(stad, x, scale):
    """Hand-rolled eval-mode STADSharp forward using the block's own weights."""
    B, C, _ = x.shape
    table = positional_table(stad.hidden_size, stad.positional_embedding.max_len)[:, :C, :]
    xe = x + scale * jnp.asarray(table)
    combined = stad.gen2(_gelu(stad.gen1(xe)))                 # [B, C, d_core]
    weight = jax.nn.softmax(combined, axis=1)
    core = jnp.sum(combined * weight, axis=1, keepdims=True)
    core = jnp.broadcast_to(core, (B, C, stad.d_core))
    fused = _gelu(stad.gen3(jnp.concatenate([xe, core], axis=-1)))
    return stad.gen4(fused)


def test_stad_shape_preserved():
    B, C, hidden, d_core = 2, 5, 16, 8
    stad = _stad(hidden, d_core)
    x = jnp.ones((B, C, hidden), dtype=jnp.float32)
    out = stad(x, deterministic=True)
    assert out.shape == (B, C, hidden)
    assert out.dtype == jnp.float32


def test_stad_deterministic_is_idempotent():
    stad = _stad()
    x = jnp.asarray(np.random.RandomState(1).randn(2, 6, 16), dtype=jnp.float32)
    o1 = np.asarray(stad(x, deterministic=True))
    o2 = np.asarray(stad(x, deterministic=True))
    np.testing.assert_allclose(o1, o2, rtol=1e-6)   # eval core + PE are deterministic


def test_stad_train_stochastic_pooling_differs_across_calls():
    # Multinomial pooling (and the PE Bernoulli) draw fresh keys each forward -> two
    # train-mode passes over the same input differ.
    stad = _stad()
    x = jnp.asarray(np.random.RandomState(2).randn(2, 6, 16), dtype=jnp.float32)
    o1 = np.asarray(stad(x, deterministic=False))
    o2 = np.asarray(stad(x, deterministic=False))
    assert np.any(o1 != o2)


def test_stad_eval_matches_hand_computed_forward_at_expectation_scale():
    # The whole eval branch, end to end: encoding at pe_keep_prob * pe_scale, set
    # FFN, softmax-weighted pooling, dispatch, fusion.
    stad = _stad(hidden=16, d_core=8, dropout=0.0, pe_keep_prob=0.3)
    x = jnp.asarray(np.random.RandomState(5).randn(3, 4, 16), dtype=jnp.float32)
    got = np.asarray(stad(x, deterministic=True))
    want = np.asarray(_manual_stad_forward(stad, x, 0.3 * float(stad.pe_scale.value)))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_stad_eval_scale_tracks_learned_pe_scale():
    stad = _stad(dropout=0.0, pe_keep_prob=0.5)
    stad.pe_scale.value = jnp.asarray(2.5, dtype=jnp.float32)  # pretend it trained
    x = jnp.asarray(np.random.RandomState(6).randn(2, 4, 16), dtype=jnp.float32)
    got = np.asarray(stad(x, deterministic=True))
    want = np.asarray(_manual_stad_forward(stad, x, 0.5 * 2.5))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_stad_pe_keep_prob_zero_disables_encoding_in_eval():
    stad = _stad(dropout=0.0, pe_keep_prob=0.0)
    x = jnp.asarray(np.random.RandomState(7).randn(2, 4, 16), dtype=jnp.float32)
    got = np.asarray(stad(x, deterministic=True))
    want = np.asarray(_manual_stad_forward(stad, x, 0.0))  # plain SOFTS STAD
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_stad_pe_keep_prob_zero_single_series_agrees_across_modes():
    # pe_keep_prob=0 zeroes the eval scale AND forces the train Bernoulli gate to 0,
    # so with dropout=0 the block degenerates to plain SOFTS STAD. With C=1 the
    # softmax over the series axis is 1 and the multinomial can only pick index 0,
    # so train and eval pooling coincide too (see test_softs.py's analogue).
    stad = _stad(dropout=0.0, pe_keep_prob=0.0)
    x = jnp.asarray(np.random.RandomState(3).randn(4, 1, 16), dtype=jnp.float32)
    o_det = np.asarray(stad(x, deterministic=True))
    for _ in range(5):
        o_sto = np.asarray(stad(x, deterministic=False))
        np.testing.assert_allclose(o_det, o_sto, rtol=1e-5, atol=1e-5)


def test_stad_pe_keep_prob_one_always_applies_encoding_in_train():
    # gate = (uniform([0,1)) < 1.0) is always True -> the encoding is applied at the
    # full pe_scale on every draw, so the output never equals the un-encoded one.
    stad = _stad(dropout=0.0, pe_keep_prob=1.0)
    x = jnp.asarray(np.random.RandomState(8).randn(3, 1, 16), dtype=jnp.float32)
    encoded = np.asarray(_manual_stad_forward(stad, x, 1.0))
    plain = np.asarray(_manual_stad_forward(stad, x, 0.0))
    for _ in range(25):
        out = np.asarray(stad(x, deterministic=False))
        np.testing.assert_allclose(out, encoded, rtol=1e-5, atol=1e-5)
        assert not np.allclose(out, plain, rtol=1e-4, atol=1e-4)


def test_stad_pe_keep_prob_one_single_series_agrees_across_modes():
    # At pe_keep_prob=1 the train gate (always 1) and the eval scale
    # (pe_keep_prob * pe_scale = pe_scale) coincide, so with C=1 and dropout=0 the
    # two modes must agree exactly — the remaining difference is only randomness.
    stad = _stad(dropout=0.0, pe_keep_prob=1.0)
    x = jnp.asarray(np.random.RandomState(9).randn(4, 1, 16), dtype=jnp.float32)
    o_det = np.asarray(stad(x, deterministic=True))
    for _ in range(5):
        np.testing.assert_allclose(np.asarray(stad(x, deterministic=False)),
                                   o_det, rtol=1e-5, atol=1e-5)


def test_stad_train_output_varies_but_eval_is_stable_for_intermediate_keep_prob():
    stad = _stad(hidden=16, d_core=8, dropout=0.0, pe_keep_prob=0.5)
    x = jnp.asarray(np.random.RandomState(10).randn(2, 4, 16), dtype=jnp.float32)
    train_outs = [np.asarray(stad(x, deterministic=False)) for _ in range(30)]
    assert any(not np.allclose(o, train_outs[0]) for o in train_outs[1:])
    eval_outs = [np.asarray(stad(x, deterministic=True)) for _ in range(5)]
    for o in eval_outs[1:]:
        np.testing.assert_allclose(o, eval_outs[0], rtol=1e-6)


def test_stad_gate_takes_both_values_over_many_draws():
    # A single Bernoulli per forward, shared across the batch: over many draws the
    # train output must land on BOTH the encoded and the un-encoded forward.
    stad = _stad(dropout=0.0, pe_keep_prob=0.5)
    x = jnp.asarray(np.random.RandomState(11).randn(2, 1, 16), dtype=jnp.float32)
    encoded = np.asarray(_manual_stad_forward(stad, x, 1.0))
    plain = np.asarray(_manual_stad_forward(stad, x, 0.0))
    seen_on, seen_off = False, False
    for _ in range(60):
        out = np.asarray(stad(x, deterministic=False))
        seen_on |= bool(np.allclose(out, encoded, rtol=1e-4, atol=1e-4))
        seen_off |= bool(np.allclose(out, plain, rtol=1e-4, atol=1e-4))
    assert seen_on and seen_off


def test_stad_dropout_is_a_no_op_at_inference():
    # dropout1/2/3 must be disabled under deterministic=True, so a block with
    # dropout=0.5 evaluates identically to the same-seeded dropout-free block.
    x = jnp.asarray(np.random.RandomState(12).randn(2, 4, 16), dtype=jnp.float32)
    dry = _stad(dropout=0.0, pe_keep_prob=0.5, seed=0)
    wet = _stad(dropout=0.5, pe_keep_prob=0.5, seed=0)
    np.testing.assert_allclose(np.asarray(wet(x, deterministic=True)),
                               np.asarray(dry(x, deterministic=True)), rtol=1e-6)


def test_stad_dropout_is_active_in_training():
    stad = _stad(dropout=0.5, pe_keep_prob=0.0)   # gate off -> only dropout varies
    x = jnp.asarray(np.random.RandomState(13).randn(2, 1, 16), dtype=jnp.float32)
    outs = [np.asarray(stad(x, deterministic=False)) for _ in range(10)]
    assert any(not np.allclose(o, outs[0]) for o in outs[1:])


def test_stad_core_is_shared_across_series_in_gen2_space():
    # The aggregated core is broadcast to every series before the fusion MLP; with
    # the position encoding disabled a constant-across-series input therefore gives
    # a constant-across-series output (all series see the identical core).
    stad = _stad(dropout=0.0, pe_keep_prob=0.0)
    row = np.random.RandomState(4).randn(2, 1, 16)
    x = jnp.asarray(np.repeat(row, 5, axis=1), dtype=jnp.float32)  # [2, 5, 16]
    out = np.asarray(stad(x, deterministic=True))
    for c in range(1, 5):
        np.testing.assert_allclose(out[:, c, :], out[:, 0, :], rtol=1e-5, atol=1e-5)


def test_stad_position_encoding_breaks_series_symmetry():
    # The point of the variable-position encoding: with it on, identical series are
    # no longer interchangeable (contrast with the SOFTS STAD test above).
    stad = _stad(dropout=0.0, pe_keep_prob=1.0)
    row = np.random.RandomState(14).randn(2, 1, 16)
    x = jnp.asarray(np.repeat(row, 5, axis=1), dtype=jnp.float32)
    out = np.asarray(stad(x, deterministic=True))
    assert not np.allclose(out[:, 1, :], out[:, 0, :], rtol=1e-4, atol=1e-4)


def test_stad_pe_scale_is_param_initialised_to_one():
    stad = _stad()
    assert isinstance(stad.pe_scale, nnx.Param)
    assert float(stad.pe_scale.value) == pytest.approx(1.0)
    assert stad.pe_scale.value.shape == ()
    assert stad.pe_scale.value.dtype == jnp.float32


@pytest.mark.parametrize("bad", [-0.1, 1.5, -1.0, 2.0])
def test_stad_rejects_pe_keep_prob_outside_unit_interval(bad):
    with pytest.raises(ValueError, match="pe_keep_prob"):
        STADSharp(hidden_size=8, d_core=4, dropout=0.0, pe_keep_prob=bad, rngs=nnx.Rngs(0))


def test_stad_handles_more_tokens_than_one_position_row():
    # Guards the table slice: C tokens must read rows 0..C-1, not row 0 repeated.
    stad = _stad(hidden=8, d_core=4, dropout=0.0, pe_keep_prob=1.0)
    x = jnp.zeros((1, 6, 8), dtype=jnp.float32)
    got = np.asarray(stad(x, deterministic=True))
    want = np.asarray(_manual_stad_forward(stad, x, 1.0))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


# ---- encoder ----------------------------------------------------------------

def test_encoder_layer_shape():
    B, T, hidden, d_core = 2, 7, 16, 8
    layer = TransEncoderLayer(hidden_size=hidden, d_core=d_core, d_ff=32,
                              dropout=0.0, pe_keep_prob=0.5, rngs=nnx.Rngs(0))
    x = jnp.ones((B, T, hidden), dtype=jnp.float32)
    out = layer(x, deterministic=True)
    assert out.shape == (B, T, hidden)


def test_encoder_stack_shape_and_layer_count():
    B, T, hidden, d_core = 2, 7, 16, 8
    enc = TransEncoder(e_layers=3, hidden_size=hidden, d_core=d_core, d_ff=32,
                       dropout=0.0, pe_keep_prob=0.5, rngs=nnx.Rngs(0))
    x = jnp.ones((B, T, hidden), dtype=jnp.float32)
    out = enc(x, deterministic=True)
    assert out.shape == (B, T, hidden)
    assert len(enc.layers) == 3


def test_encoder_has_no_final_layernorm():
    # NF's TransEncoder applies norm_layer only `if self.norm is not None`, and
    # SOFTSSharp constructs it positionally with no norm_layer -- so there is no
    # final normalization and the encoder output reaches `projection` directly.
    #
    # Note a zero-mean check CANNOT test this: every layer already ends in
    # `norm2`, so the encoder output is zero-mean either way. The observable
    # difference is the extra LayerNorm's LEARNABLE scale/bias, so pin the
    # parameter set instead. benchmarks/softssharp_weight_parity.py is the
    # end-to-end check; this is the unit-level guard against it creeping back.
    hidden, d_core = 16, 8
    enc = TransEncoder(e_layers=1, hidden_size=hidden, d_core=d_core, d_ff=32,
                       dropout=0.0, pe_keep_prob=0.5, rngs=nnx.Rngs(0))
    assert not hasattr(enc, "norm")
    _, params, *_ = nnx.split(enc, nnx.Param, ...)
    paths = {"/".join(str(k) for k in path) for path, _ in nnx.to_flat_state(params)}
    assert not any(p.startswith("norm/") for p in paths), sorted(paths)


def test_encoder_layer_holds_a_stadsharp():
    layer = TransEncoderLayer(hidden_size=8, d_core=4, d_ff=16, dropout=0.0,
                              pe_keep_prob=0.5, rngs=nnx.Rngs(0))
    assert isinstance(layer.stad, STADSharp)


def test_encoder_forwards_pe_keep_prob_to_every_layer():
    enc = TransEncoder(e_layers=3, hidden_size=8, d_core=4, d_ff=16, dropout=0.0,
                       pe_keep_prob=0.25, rngs=nnx.Rngs(0))
    assert [layer.stad.pe_keep_prob for layer in enc.layers] == [0.25] * 3


def test_resolve_activation_gelu_is_exact():
    f = _resolve_activation("gelu")
    z = jnp.array([1.0])
    exact = 1.0 * 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    np.testing.assert_allclose(float(f(z)[0]), exact, rtol=1e-5)


def test_resolve_activation_unknown_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        _resolve_activation("swish")


def test_gelu_helper_is_exact_erf_variant():
    z = jnp.array([-0.7, 0.0, 1.3])
    exact = np.asarray(z) * 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2.0)))
    np.testing.assert_allclose(np.asarray(_gelu(z)), exact, rtol=1e-5, atol=1e-6)


# ---- full backbone ----------------------------------------------------------

def _net(h=4, input_size=24, hidden=16, d_core=8, layers=2, pe_keep_prob=0.5):
    return SOFTSSharpNet(
        h=h, input_size=input_size, hidden_size=hidden, d_core=d_core,
        e_layers=layers, d_ff=32, dropout=0.0, use_norm=True,
        pe_keep_prob=pe_keep_prob, rngs=nnx.Rngs(0),
    )


def test_net_forward_shape_univariate():
    net = _net()
    x = jnp.ones((4, 24, 1), dtype=jnp.float32)
    out = net(x, deterministic=True)
    assert out.shape == (4, 4, 1)
    assert out.dtype == jnp.float32


def test_net_is_n_generic_multivariate():
    # The backbone is written N-generically: [B, L, N] -> [B, h, N]. Only the
    # SOFTSSharp wrapper fixes N=1; this guards the future multivariate path (where
    # the variable-position encoding starts to distinguish series).
    net = _net()
    x = jnp.ones((2, 24, 3), dtype=jnp.float32)
    out = net(x, deterministic=True)
    assert out.shape == (2, 4, 3)


def test_net_use_norm_false_still_forwards():
    net = SOFTSSharpNet(h=4, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                        d_ff=32, dropout=0.0, use_norm=False, pe_keep_prob=0.5,
                        rngs=nnx.Rngs(0))
    out = net(jnp.ones((2, 24, 1), dtype=jnp.float32), deterministic=True)
    assert out.shape == (2, 4, 1) and jnp.all(jnp.isfinite(out))


def test_net_d_core_independent_of_hidden_size():
    # Unlike attention's n_heads, d_core has no divisibility constraint with
    # hidden_size — a core wider or narrower than hidden must both forward.
    for d_core in (3, 30, 100):
        net = SOFTSSharpNet(h=4, input_size=12, hidden_size=30, d_core=d_core, e_layers=1,
                            d_ff=16, dropout=0.0, use_norm=True, pe_keep_prob=0.5,
                            rngs=nnx.Rngs(0))
        out = net(jnp.ones((2, 12, 1), dtype=jnp.float32), deterministic=True)
        assert out.shape == (2, 4, 1) and jnp.all(jnp.isfinite(out))


def test_net_odd_hidden_size_forwards():
    # Odd hidden_size exercises the truncated-cosine branch of positional_table.
    net = SOFTSSharpNet(h=4, input_size=12, hidden_size=15, d_core=8, e_layers=1,
                        d_ff=16, dropout=0.0, use_norm=True, pe_keep_prob=0.5,
                        rngs=nnx.Rngs(0))
    out = net(jnp.ones((2, 12, 1), dtype=jnp.float32), deterministic=True)
    assert out.shape == (2, 4, 1) and jnp.all(jnp.isfinite(out))


def test_net_pe_scale_is_selected_by_param_split():
    # pe_scale must land in the nnx.Param partition, otherwise nnx.Optimizer's
    # wrt=nnx.Param would never update it and the "learnable gain" is a constant.
    net = _net(layers=2)
    _, params, _ = nnx.split(net, nnx.Param, ...)
    paths = {tuple(p) for p, _ in nnx.to_flat_state(params)}
    assert ("encoder", "layers", 0, "stad", "pe_scale") in paths
    assert ("encoder", "layers", 1, "stad", "pe_scale") in paths


def test_net_pe_scale_receives_a_gradient():
    net = _net(layers=1)

    def loss_fn(m):
        return jnp.sum(m(jnp.ones((2, 24, 1), dtype=jnp.float32), deterministic=True) ** 2)

    grads = nnx.grad(loss_fn)(net)
    g = nnx.to_flat_state(grads)
    pe_grads = [v for p, v in g if tuple(p)[-1] == "pe_scale"]
    assert len(pe_grads) == 1
    assert jnp.isfinite(pe_grads[0].value)


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

def _train_net(h=12, input_size=24, hidden=16, d_core=8, layers=1):
    return SOFTSSharpNet(
        h=h, input_size=input_size, hidden_size=hidden, d_core=d_core,
        e_layers=layers, d_ff=32, dropout=0.0, use_norm=True, pe_keep_prob=0.5,
        rngs=nnx.Rngs(0),
    )


def test_build_windows_shape():
    y = _make_y(60)
    w = build_windows(y, input_size=24, h=12)
    assert w.shape == (60 - 36 + 1, 36)


def test_build_windows_raises_when_too_short():
    with pytest.raises(ValueError, match="too short"):
        build_windows(_make_y(30), input_size=36, h=12)


def test_forward_loss_returns_scalar():
    net = _train_net()
    w = build_windows(_make_y(), input_size=24, h=12)
    loss = forward_loss(net, w[:8], h=12, input_size=24)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_train_gradient_step_decreases_loss():
    net = _train_net()
    y = _make_y()
    losses = train(net, y, h=12, input_size=24, max_steps=30,
                   windows_batch_size=64, lr=1e-2, seed=0)
    assert losses.shape == (30,)
    assert float(losses[-1]) < float(losses[0])


def test_train_updates_pe_scale():
    # The learnable position gain must actually move under Adam.
    net = _train_net()
    before = float(net.encoder.layers[0].stad.pe_scale.value)
    train(net, _make_y(), h=12, input_size=24, max_steps=20,
          windows_batch_size=64, lr=1e-2, seed=0)
    after = float(net.encoder.layers[0].stad.pe_scale.value)
    assert after != before
    assert math.isfinite(after)


def test_train_deterministic_with_same_seed():
    # Same rng seed drives param init AND STADSharp's pooling + PE-gate key
    # streams, so two identical runs must match despite the stochasticity.
    y = _make_y()
    l1 = train(_train_net(), y, h=12, input_size=24, max_steps=10,
               windows_batch_size=64, lr=1e-3, seed=0)
    l2 = train(_train_net(), y, h=12, input_size=24, max_steps=10,
               windows_batch_size=64, lr=1e-3, seed=0)
    np.testing.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=1e-5)


def test_train_does_not_leak_a_tracer_from_the_cached_position_table():
    # positional_table is lru_cached; if it were built with jnp the first call
    # inside the training scan would cache a tracer and every later call would
    # raise UnexpectedTracerError. Train first, then forward outside the trace.
    net = _train_net()
    y = _make_y()
    train(net, y, h=12, input_size=24, max_steps=5, windows_batch_size=64, lr=1e-3, seed=0)
    out = net(jnp.ones((1, 24, 1), dtype=jnp.float32), deterministic=True)
    assert jnp.all(jnp.isfinite(out))


def test_predict_step_shape_and_idempotent():
    net = _train_net()
    y = _make_y()
    train(net, y, h=12, input_size=24, max_steps=5, windows_batch_size=64, lr=1e-3, seed=0)
    p1 = predict_step(net, y[-24:], h=12, input_size=24)
    p2 = predict_step(net, y[-24:], h=12, input_size=24)
    assert p1.shape == (12,)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-6)


def test_train_oversample_with_replacement_small_n_regime():
    # n_windows < windows_batch_size -> NF with-replacement branch (the regime every
    # small benchmark series hits). y(48), input_size=24, h=12 -> n_windows=13.
    net = _train_net()
    losses = train(net, _make_y(48), h=12, input_size=24, max_steps=8,
                   windows_batch_size=64, lr=1e-3, seed=0)
    assert losses.shape == (8,)
    assert jnp.all(jnp.isfinite(losses))


def test_train_raises_on_divergence():
    net = _train_net()
    with pytest.raises(RuntimeError, match="diverged"):
        train(net, _make_y(), h=12, input_size=24, max_steps=10,
              windows_batch_size=64, lr=1e9, seed=0)


# =============================================================================
# Model (BaseForecaster conformance)
# =============================================================================

def _tiny():
    return SOFTSSharp(h=12, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                      d_ff=32, max_steps=20, windows_batch_size=64, random_seed=0)


def test_softssharp_inherits_baseforecaster():
    assert issubclass(SOFTSSharp, BaseForecaster)


def test_init_sets_conformal_params_to_none():
    assert _tiny().conformal_params is None


def test_init_default_pe_keep_prob_matches_neuralforecast():
    assert SOFTSSharp(h=12).pe_keep_prob == 0.5


@pytest.mark.parametrize("bad", [-0.1, 1.5, -1.0, 2.0])
def test_init_rejects_pe_keep_prob_outside_unit_interval(bad):
    with pytest.raises(ValueError, match="pe_keep_prob"):
        SOFTSSharp(h=12, pe_keep_prob=bad)


def test_build_net_propagates_pe_keep_prob():
    m = SOFTSSharp(h=4, input_size=12, hidden_size=8, d_core=4, e_layers=1,
                   d_ff=16, max_steps=2, windows_batch_size=16, random_seed=0,
                   pe_keep_prob=0.75)
    net = m._build_net()
    assert net.encoder.layers[0].stad.pe_keep_prob == 0.75


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
    # projection/STADSharp wiring that still passes shape + determinism tests.
    n = 200
    y = jnp.asarray(np.sin(np.arange(n) / 5.0), dtype=jnp.float32)
    m = SOFTSSharp(h=12, input_size=36, hidden_size=32, d_core=32, e_layers=2,
                   d_ff=64, dropout=0.0, max_steps=300, windows_batch_size=64,
                   random_seed=0)
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
    m = SOFTSSharp(h=12, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                   d_ff=32, max_steps=10, windows_batch_size=64, random_seed=0,
                   loss=my_loss)
    m.fit(_make_y())
    assert jnp.all(jnp.isfinite(m.predict(h=12)["mean"]))


def test_loss_unknown_string_raises_at_fit():
    m = SOFTSSharp(h=12, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                   d_ff=32, max_steps=2, windows_batch_size=64, random_seed=0, loss="rmse")
    with pytest.raises(ValueError, match="Unknown loss"):
        m.fit(_make_y())


def test_fit_raises_on_exog():
    with pytest.raises(NotImplementedError):
        _tiny().fit(_make_y(), X=jnp.ones((200, 1)))


def test_fit_raises_on_short_series():
    with pytest.raises(ValueError, match="too short"):
        _tiny().fit(_make_y(30))  # < input_size + h = 36


def test_fit_raises_on_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        _tiny().fit(jnp.ones((200, 2), dtype=jnp.float32))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        _tiny().predict(h=12)


def test_pickle_round_trip_preserves_predictions():
    m = _tiny().fit(_make_y())
    before = np.asarray(m.predict(h=12)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    after = np.asarray(m2.predict(h=12)["mean"])
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-5)


def test_pickle_round_trip_preserves_learned_pe_scale():
    m = _tiny().fit(_make_y())
    before = float(m.model_.encoder.layers[0].stad.pe_scale.value)
    m2 = pickle.loads(pickle.dumps(m))
    after = float(m2.model_.encoder.layers[0].stad.pe_scale.value)
    assert after == pytest.approx(before, rel=1e-6)
    assert before != 1.0   # it trained away from its init


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
    m = SOFTSSharp(h=12, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                   d_ff=32, max_steps=10, windows_batch_size=64, random_seed=0,
                   loss=loss_name).fit(_make_y())
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(
        np.asarray(m2.predict(h=12)["mean"]), np.asarray(m.predict(h=12)["mean"]),
        rtol=1e-5, atol=1e-5,
    )


def test_softssharp_importable_from_models_namespace():
    from chronax.models import SOFTSSharp as S
    assert S is SOFTSSharp


def test_softssharp_advertised_in_models_all():
    import chronax.models as models
    assert "SOFTSSharp" in models.__all__


def test_softssharp_is_discovered_by_benchmark_registry():
    # benchmarks/ is excluded from packaging (see tests/neural/conftest.py), so put
    # the repo root on sys.path before importing the harness registry.
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from benchmarks.neural.registry import list_models
    assert "SOFTSSharp" in list_models()


def test_constant_series_returns_finite():
    m = _tiny().fit(jnp.ones(200, dtype=jnp.float32))
    assert jnp.all(jnp.isfinite(m.predict(h=12)["mean"]))


def test_h_equals_one():
    m = SOFTSSharp(h=1, input_size=12, hidden_size=8, d_core=4, e_layers=1,
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
    assert np.all(np.isnan(fitted[:24]))
    assert np.all(np.isfinite(fitted[24:]))


def test_build_net_works_in_loop_construction():
    # precursor to conformity_scores' vmap: constructing the net in a loop
    nets = [_tiny()._build_net() for _ in range(3)]
    assert len(nets) == 3


def test_predict_with_level_returns_interval_keys():
    m = SOFTSSharp(h=4, input_size=12, hidden_size=8, d_core=4, e_layers=1,
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
    m = SOFTSSharp(h=4, input_size=12, hidden_size=8, d_core=4, e_layers=1,
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
    m = SOFTSSharp(h=4, input_size=12, hidden_size=8, d_core=4, e_layers=1,
                   d_ff=16, max_steps=2, windows_batch_size=16, random_seed=0)
    m.fit(_make_y(80))
    before = np.asarray(m.predict(h=4)["mean"])
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    _ = m.predict(h=4, level=[80])
    after = np.asarray(m.predict(h=4)["mean"])
    assert np.all(np.isfinite(after))
    assert np.allclose(before, after)


def test_input_size_default_resolves_to_three_h():
    m = SOFTSSharp(h=10)
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
    m_off = SOFTSSharp(h=12, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                       d_ff=32, max_steps=20, windows_batch_size=64, random_seed=0).fit(y)
    assert m_off._bc_lambda is None
    plain = SOFTSSharp(h=12, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                       d_ff=32, max_steps=20, windows_batch_size=64, random_seed=0,
                       use_boxcox=False).fit(y)
    np.testing.assert_allclose(np.asarray(m_off.predict(h=12)["mean"]),
                               np.asarray(plain.predict(h=12)["mean"]), rtol=1e-5, atol=1e-5)


def test_use_boxcox_fit_predict_finite_and_sets_lambda():
    y = jnp.asarray(np.exp(np.linspace(0, 3, 200)) + 1.0, dtype=jnp.float32)
    m = SOFTSSharp(h=12, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                   d_ff=32, max_steps=30, windows_batch_size=64, random_seed=0,
                   use_boxcox=True).fit(y)
    assert m._bc_lambda is not None
    out = m.predict(h=12)["mean"]
    assert out.shape == (12,) and jnp.all(jnp.isfinite(out))


def test_use_boxcox_requires_positive_values():
    y = _make_y(200)  # sine -> contains non-positive values
    m = SOFTSSharp(h=12, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                   d_ff=32, max_steps=5, windows_batch_size=64, random_seed=0,
                   use_boxcox=True)
    with pytest.raises(ValueError, match="positive"):
        m.fit(y)


def test_use_boxcox_pickle_round_trip_preserves_predictions_and_lambda():
    y = jnp.asarray(np.exp(np.linspace(0, 3, 200)) + 1.0, dtype=jnp.float32)
    m = SOFTSSharp(h=12, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                   d_ff=32, max_steps=20, windows_batch_size=64, random_seed=0,
                   use_boxcox=True).fit(y)
    before = np.asarray(m.predict(h=12)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    assert m2._bc_lambda == m._bc_lambda
    np.testing.assert_allclose(np.asarray(m2.predict(h=12)["mean"]), before, rtol=1e-5, atol=1e-5)


def test_use_boxcox_forecast_fitted_inverts_to_original_scale():
    y = jnp.asarray(np.exp(np.linspace(0, 3, 200)) + 1.0, dtype=jnp.float32)
    res = SOFTSSharp(h=12, input_size=24, hidden_size=16, d_core=8, e_layers=1,
                     d_ff=32, max_steps=20, windows_batch_size=64, random_seed=0,
                     use_boxcox=True).forecast(y, h=12, fitted=True)
    fitted = np.asarray(res["fitted"])
    assert fitted.shape == (200,)
    assert np.all(np.isnan(fitted[:24]))
    tail = fitted[24:]
    assert np.all(np.isfinite(tail))
    # fitted values are back in the original (positive, ~exponential) scale
    assert tail.min() > 0

"""Tests for chronax.models.DilatedRNN.

Covers the six source modules of the dilated_rnn subpackage in one file (losses,
scaler, module, training, model, package __init__), matching the flat
``test_<model>.py`` convention used elsewhere in ``tests/``.

DilatedRNN is a JAX/Flax-NNX port of ``neuralforecast.DilatedRNN``. Its riskiest
part is the DILATION PLUMBING: NF builds the interleaved subsequences with Python
list comprehensions (``torch.cat([inputs[j::rate] ...], 1)`` and a
stack/transpose/reshape inverse), while the port replaces both with a single
row-major reshape. Those two rewrites are the thing that would silently corrupt
every forecast, so a literal NumPy transcription of NF's originals lives at the
top of the module section and the reshape versions are asserted equal to it for a
grid of ``(T, B, F, rate)`` — including lengths that are not a multiple of the
rate, i.e. the zero-padding path.
"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.dilated_rnn.dilated_rnn_losses import (
    LOSSES, huber, mae, mse, resolve,
)
from chronax.models.dilated_rnn.dilated_rnn_model import DilatedRNN
from chronax.models.dilated_rnn.dilated_rnn_module import (
    CELL_TYPES,
    DRNN,
    AttentiveLSTMLayer,
    DilatedRNNNet,
    GRUCell,
    LSTMCell,
    MLPDecoder,
    RNNCell,
    ResLSTMCell,
    drnn_layer,
    make_cell,
    pad_inputs,
    prepare_inputs,
    split_outputs,
)
from chronax.models.dilated_rnn.dilated_rnn_scaler import (
    SCALERS, IdentityScaler, RobustScaler, StandardScaler,
    resolve as resolve_scaler,
)
from chronax.models.dilated_rnn.dilated_rnn_training import (
    build_windows, make_lr_schedule, predict_step, scaled_forward_loss, train,
)
from chronax.utils import ConformalIntervals


def _make_y(n=200, period=5.0):
    return jnp.asarray(np.sin(np.arange(n) / period), dtype=jnp.float32)


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


def test_loss_registry_keys_are_stable():
    assert set(LOSSES.keys()) == {"mae", "mse", "huber"}


# =============================================================================
# Scaler
# =============================================================================

def test_robust_scaler_median_and_mad_known_values():
    # median([1,2,3,4,100]) = 3; |x - 3| = [2,1,0,1,97] -> median 1.
    x = jnp.asarray([[1.0, 2.0, 3.0, 4.0, 100.0]], dtype=jnp.float32)
    shift, scale = RobustScaler().stats(x, axis=1)
    assert float(shift[0, 0]) == pytest.approx(3.0)
    assert float(scale[0, 0]) == pytest.approx(1.0, abs=1e-5)


def test_robust_scaler_mad_zero_falls_back_to_scaled_std():
    # x = [0,0,0,0,10]: median 0, |x - median| = [0,0,0,0,10] -> MAD = 0. The NF
    # fallback estimates MAD from the std via MAD ~= 0.6745 * sigma; sigma = 4.
    x = jnp.asarray([[0.0, 0.0, 0.0, 0.0, 10.0]], dtype=jnp.float32)
    shift, scale = RobustScaler().stats(x, axis=1)
    assert float(shift[0, 0]) == pytest.approx(0.0)
    assert float(scale[0, 0]) == pytest.approx(4.0 * 0.6744897501960817, abs=1e-4)


def test_robust_scaler_constant_window_scale_is_finite_and_nonzero():
    # Degenerate window: MAD = 0 AND std = 0, so the fallback is 0 too and the
    # final ``where(scale == 0, 1.0)`` guard is what keeps the division safe.
    x = jnp.full((1, 8), 7.0, dtype=jnp.float32)
    shift, scale = RobustScaler().stats(x, axis=1)
    assert float(shift[0, 0]) == pytest.approx(7.0)
    assert np.isfinite(float(scale[0, 0]))
    assert float(scale[0, 0]) > 0.0
    assert float(scale[0, 0]) == pytest.approx(1.0, abs=1e-4)


def test_standard_scaler_uses_population_std():
    x = jnp.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=jnp.float32)
    shift, scale = StandardScaler().stats(x, axis=1)
    assert float(shift[0, 0]) == pytest.approx(3.0)
    assert float(scale[0, 0]) == pytest.approx(np.sqrt(2.0), abs=1e-5)  # ddof=0


def test_standard_scaler_constant_window_scale_is_one():
    x = jnp.full((1, 6), -2.0, dtype=jnp.float32)
    _, scale = StandardScaler().stats(x, axis=1)
    assert float(scale[0, 0]) == pytest.approx(1.0, abs=1e-4)


def test_identity_scaler_stats_are_zero_and_one():
    x = jnp.asarray(np.random.RandomState(0).randn(3, 7), dtype=jnp.float32)
    shift, scale = IdentityScaler().stats(x, axis=1)
    assert shift.shape == (3, 1) and scale.shape == (3, 1)
    np.testing.assert_array_equal(np.asarray(shift), np.zeros((3, 1)))
    np.testing.assert_array_equal(np.asarray(scale), np.ones((3, 1)))


def test_identity_scaler_transform_is_a_no_op():
    x = jnp.asarray(np.random.RandomState(1).randn(2, 5), dtype=jnp.float32)
    sc = IdentityScaler()
    shift, scale = sc.stats(x, axis=1)
    np.testing.assert_array_equal(np.asarray(sc.transform(x, shift, scale)), np.asarray(x))


@pytest.mark.parametrize("cls", [RobustScaler, StandardScaler, IdentityScaler])
def test_scaler_transform_inverse_round_trip(cls):
    sc = cls()
    x = jnp.asarray(np.random.RandomState(2).randn(4, 12) * 3.0 + 5.0, dtype=jnp.float32)
    shift, scale = sc.stats(x, axis=1)
    rt = sc.inverse(sc.transform(x, shift, scale), shift, scale)
    np.testing.assert_allclose(np.asarray(rt), np.asarray(x), rtol=1e-4, atol=1e-4)


def test_robust_transform_centers_on_the_median():
    sc = RobustScaler()
    x = jnp.asarray([[1.0, 2.0, 3.0, 4.0, 100.0]], dtype=jnp.float32)
    shift, scale = sc.stats(x, axis=1)
    z = sc.transform(x, shift, scale)
    assert float(z[0, 2]) == pytest.approx(0.0, abs=1e-6)  # the median element


def test_scaler_registry_keys_are_stable():
    assert set(SCALERS.keys()) == {"robust", "standard", "identity"}


@pytest.mark.parametrize("name,cls", [
    ("robust", RobustScaler), ("standard", StandardScaler), ("identity", IdentityScaler),
])
def test_resolve_scaler_returns_instance_of_right_class(name, cls):
    assert isinstance(resolve_scaler(name), cls)


def test_resolve_scaler_passes_instances_through():
    sc = RobustScaler()
    assert resolve_scaler(sc) is sc


def test_resolve_scaler_unknown_raises():
    with pytest.raises(ValueError, match="Unknown scaler_type"):
        resolve_scaler("minmax")


# =============================================================================
# Module — dilation plumbing
# =============================================================================
#
# Literal NumPy transcription of the neuralforecast originals
# (neuralforecast/models/dilated_rnn.py, DRNN._pad_inputs / _prepare_inputs /
# _split_outputs). The reshape-based port must agree with these EXACTLY.

def _nf_prepare_inputs(inputs, rate):        # inputs [T, B, F] -> [T/rate, rate*B, F]
    return np.concatenate([inputs[j::rate, :, :] for j in range(rate)], 1)


def _nf_split_outputs(dilated_outputs, rate):     # [S, rate*B, C] -> [S*rate, B, C]
    batchsize = dilated_outputs.shape[1] // rate
    blocks = [dilated_outputs[:, i * batchsize:(i + 1) * batchsize, :] for i in range(rate)]
    interleaved = np.stack(blocks).transpose(1, 0, 2, 3)
    return interleaved.reshape(
        dilated_outputs.shape[0] * rate, batchsize, dilated_outputs.shape[2]
    )


def _nf_pad_inputs(inputs, n_steps, rate):
    if n_steps % rate != 0:
        dilated_steps = n_steps // rate + 1
        zeros_ = np.zeros(
            (dilated_steps * rate - inputs.shape[0], inputs.shape[1], inputs.shape[2]),
            dtype=inputs.dtype,
        )
        inputs = np.concatenate((inputs, zeros_))
    return inputs


# (T, B, F, rate) — includes T divisible and T *not* divisible by rate.
_DILATION_CASES = [
    (12, 2, 3, 1),
    (12, 2, 3, 2),
    (12, 2, 3, 3),
    (12, 2, 3, 4),
    (10, 3, 2, 3),   # 10 % 3 != 0 -> padding path
    (7, 1, 1, 2),    # 7 % 2 != 0
    (9, 2, 4, 5),    # 9 % 5 != 0, rate > T/2
    (6, 1, 1, 8),    # rate > T: a single dilated step after padding
]


def _seq(T, B, F, seed=0):
    return np.arange(T * B * F, dtype=np.float32).reshape(T, B, F) + seed


@pytest.mark.parametrize("T,B,F,rate", _DILATION_CASES)
def test_pad_inputs_matches_nf(T, B, F, rate):
    x = _seq(T, B, F)
    got = np.asarray(pad_inputs(jnp.asarray(x), rate))
    np.testing.assert_array_equal(got, _nf_pad_inputs(x, T, rate))


@pytest.mark.parametrize("T,B,F,rate", _DILATION_CASES)
def test_pad_inputs_length_is_next_multiple_of_rate(T, B, F, rate):
    padded = pad_inputs(jnp.asarray(_seq(T, B, F)), rate)
    assert padded.shape[0] % rate == 0
    assert padded.shape[0] - T < rate
    assert padded.shape[1:] == (B, F)


def test_pad_inputs_is_a_noop_when_already_a_multiple():
    x = jnp.asarray(_seq(12, 2, 3))
    assert pad_inputs(x, 3) is x   # exact multiple -> returned unchanged


def test_pad_inputs_pads_with_zeros_at_the_end():
    x = jnp.asarray(_seq(7, 1, 1) + 1.0)   # strictly non-zero payload
    padded = np.asarray(pad_inputs(x, 4))
    assert padded.shape[0] == 8
    np.testing.assert_array_equal(padded[7], np.zeros((1, 1), dtype=np.float32))


@pytest.mark.parametrize("T,B,F,rate", _DILATION_CASES)
def test_prepare_inputs_matches_nf(T, B, F, rate):
    x = _nf_pad_inputs(_seq(T, B, F), T, rate)
    got = np.asarray(prepare_inputs(jnp.asarray(x), rate))
    np.testing.assert_array_equal(got, _nf_prepare_inputs(x, rate))


@pytest.mark.parametrize("T,B,F,rate", _DILATION_CASES)
def test_prepare_inputs_column_block_i_is_the_i_th_subsequence(T, B, F, rate):
    # The whole point of the reshape: columns i*B:(i+1)*B must carry inputs[i::rate].
    x = _nf_pad_inputs(_seq(T, B, F), T, rate)
    packed = np.asarray(prepare_inputs(jnp.asarray(x), rate))
    assert packed.shape == (x.shape[0] // rate, rate * B, F)
    for i in range(rate):
        np.testing.assert_array_equal(packed[:, i * B:(i + 1) * B, :], x[i::rate, :, :])


def test_prepare_inputs_rejects_non_multiple_length():
    with pytest.raises(ValueError, match="not a multiple of rate"):
        prepare_inputs(jnp.asarray(_seq(10, 2, 1)), 3)


@pytest.mark.parametrize("T,B,F,rate", _DILATION_CASES)
def test_split_outputs_matches_nf(T, B, F, rate):
    S = (T + (-T) % rate) // rate
    C = 3
    dilated = np.arange(S * rate * B * C, dtype=np.float32).reshape(S, rate * B, C)
    got = np.asarray(split_outputs(jnp.asarray(dilated), rate))
    np.testing.assert_array_equal(got, _nf_split_outputs(dilated, rate))


@pytest.mark.parametrize("T,B,F,rate", _DILATION_CASES)
def test_split_outputs_is_the_exact_inverse_of_prepare_inputs(T, B, F, rate):
    padded = _nf_pad_inputs(_seq(T, B, F), T, rate)
    packed = prepare_inputs(jnp.asarray(padded), rate)
    round_trip = np.asarray(split_outputs(packed, rate))
    np.testing.assert_array_equal(round_trip, padded)


def test_split_outputs_rejects_batch_not_multiple_of_rate():
    with pytest.raises(ValueError, match="not a multiple of rate"):
        split_outputs(jnp.zeros((4, 5, 2)), 3)


@pytest.mark.parametrize("T,B,F,rate", _DILATION_CASES)
def test_drnn_layer_restores_the_unpadded_time_length(T, B, F, rate):
    hidden = 5
    cell = make_cell("GRU", F, hidden, rngs=nnx.Rngs(0))
    x = jnp.asarray(_seq(T, B, F) / 100.0)
    out = drnn_layer(cell, x, rate)
    assert out.shape == (T, B, hidden)   # T restored, padding truncated
    assert jnp.all(jnp.isfinite(out))


def test_drnn_layer_at_rate_one_is_a_plain_scan():
    # rate=1 must degenerate to running the cell straight over the sequence.
    cell = make_cell("RNN", 2, 4, rngs=nnx.Rngs(0))
    x = jnp.asarray(_seq(6, 3, 2) / 50.0)
    got = np.asarray(drnn_layer(cell, x, 1))
    ref = np.asarray(cell.run_sequence(x, cell.init_carry(3)))
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_drnn_layer_rate_two_matches_two_independent_subsequence_scans():
    # At rate 2 the even and odd timesteps are two independent recurrences that
    # share the cell; running them separately must reproduce the interleave.
    B, F, H, T = 2, 3, 4, 8
    cell = make_cell("GRU", F, H, rngs=nnx.Rngs(0))
    x = jnp.asarray(_seq(T, B, F) / 100.0)
    got = np.asarray(drnn_layer(cell, x, 2))
    even = np.asarray(cell.run_sequence(x[0::2], cell.init_carry(B)))
    odd = np.asarray(cell.run_sequence(x[1::2], cell.init_carry(B)))
    np.testing.assert_allclose(got[0::2], even, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got[1::2], odd, rtol=1e-6, atol=1e-6)


# =============================================================================
# Module — recurrent cells
# =============================================================================

_NON_LSTM = ["RNN", "GRU"]
_LSTM_FAMILY = ["LSTM", "ResLSTM", "AttentiveLSTM"]


@pytest.mark.parametrize("name", list(CELL_TYPES))
def test_make_cell_run_sequence_shape(name):
    T, B, F, H = 5, 3, 2, 6
    cell = make_cell(name, F, H, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(T, B, F), dtype=jnp.float32)
    out = cell.run_sequence(x, cell.init_carry(B))
    assert out.shape == (T, B, H)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("name", list(CELL_TYPES))
def test_init_carry_is_all_zeros(name):
    cell = make_cell(name, 2, 6, rngs=nnx.Rngs(0))
    carry = cell.init_carry(4)
    parts = carry if isinstance(carry, tuple) else (carry,)
    for p in parts:
        assert p.shape == (4, 6)
        np.testing.assert_array_equal(np.asarray(p), np.zeros((4, 6), dtype=np.float32))


@pytest.mark.parametrize("name", _LSTM_FAMILY)
def test_init_carry_is_a_two_tuple_for_the_lstm_family(name):
    cell = make_cell(name, 2, 6, rngs=nnx.Rngs(0))
    assert cell.is_lstm is True
    carry = cell.init_carry(4)
    assert isinstance(carry, tuple) and len(carry) == 2


@pytest.mark.parametrize("name", _NON_LSTM)
def test_init_carry_is_a_bare_array_for_non_lstm_cells(name):
    cell = make_cell(name, 2, 6, rngs=nnx.Rngs(0))
    assert cell.is_lstm is False
    assert not isinstance(cell.init_carry(4), tuple)


@pytest.mark.parametrize("name", ["RNN", "GRU"])
def test_non_lstm_step_returns_carry_equal_to_output(name):
    cell = make_cell(name, 2, 6, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(3, 2), dtype=jnp.float32)
    carry, out = cell.step(cell.init_carry(3), x)
    np.testing.assert_array_equal(np.asarray(carry), np.asarray(out))


def test_rnn_cell_step_matches_torch_tanh_formula():
    F, H, B = 3, 5, 2
    cell = RNNCell(F, H, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(B, F), dtype=jnp.float32)
    h0 = jnp.asarray(np.random.RandomState(1).randn(B, H), dtype=jnp.float32)
    _, out = cell.step(h0, x)
    w_ih = np.asarray(cell.ih.kernel.value); b_ih = np.asarray(cell.ih.bias.value)
    w_hh = np.asarray(cell.hh.kernel.value); b_hh = np.asarray(cell.hh.bias.value)
    ref = np.tanh(np.asarray(x) @ w_ih + b_ih + np.asarray(h0) @ w_hh + b_hh)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)


def test_gru_cell_step_matches_torch_gru_formula():
    # torch: r = sig(i_r + h_r); z = sig(i_z + h_z); n = tanh(i_n + r * h_n);
    # h' = (1 - z) * n + z * h  — note r multiplies the hidden branch AFTER its
    # bias, which is what separates torch's GRU from the fused-bias variant.
    F, H, B = 3, 5, 4
    cell = GRUCell(F, H, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(B, F), dtype=jnp.float32)
    h0 = jnp.asarray(np.random.RandomState(1).randn(B, H), dtype=jnp.float32)
    _, out = cell.step(h0, x)

    gi = np.asarray(x) @ np.asarray(cell.ih.kernel.value) + np.asarray(cell.ih.bias.value)
    gh = np.asarray(h0) @ np.asarray(cell.hh.kernel.value) + np.asarray(cell.hh.bias.value)
    i_r, i_z, i_n = gi[:, :H], gi[:, H:2 * H], gi[:, 2 * H:]
    h_r, h_z, h_n = gh[:, :H], gh[:, H:2 * H], gh[:, 2 * H:]
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    r = sig(i_r + h_r)
    z = sig(i_z + h_z)
    n = np.tanh(i_n + r * h_n)
    ref = (1.0 - z) * n + z * np.asarray(h0)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)


def test_lstm_cell_step_matches_torch_ifgo_gate_order():
    F, H, B = 3, 5, 4
    cell = LSTMCell(F, H, rngs=nnx.Rngs(0))
    rs = np.random.RandomState(0)
    x = jnp.asarray(rs.randn(B, F), dtype=jnp.float32)
    h0 = jnp.asarray(rs.randn(B, H), dtype=jnp.float32)
    c0 = jnp.asarray(rs.randn(B, H), dtype=jnp.float32)
    (h_new, c_new), out = cell.step((h0, c0), x)

    gates = (np.asarray(x) @ np.asarray(cell.ih.kernel.value)
             + np.asarray(cell.ih.bias.value)
             + np.asarray(h0) @ np.asarray(cell.hh.kernel.value)
             + np.asarray(cell.hh.bias.value))
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    i = sig(gates[:, :H])
    f = sig(gates[:, H:2 * H])
    g = np.tanh(gates[:, 2 * H:3 * H])
    o = sig(gates[:, 3 * H:])
    c_ref = f * np.asarray(c0) + i * g
    h_ref = o * np.tanh(c_ref)
    np.testing.assert_allclose(np.asarray(c_new), c_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(h_new), h_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(h_new))


def _reslstm_reference(cell, x, h0, c0, H, use_raw_skip):
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    ifo = (np.asarray(x) @ np.asarray(cell.ii.kernel.value) + np.asarray(cell.ii.bias.value)
           + np.asarray(h0) @ np.asarray(cell.ih.kernel.value) + np.asarray(cell.ih.bias.value)
           + np.asarray(c0) @ np.asarray(cell.ic.kernel.value) + np.asarray(cell.ic.bias.value))
    i = sig(ifo[:, :H])
    f = sig(ifo[:, H:2 * H])
    o = sig(ifo[:, 2 * H:])
    g = np.tanh(np.asarray(h0) @ np.asarray(cell.hh.kernel.value)
                + np.asarray(cell.hh.bias.value))
    c_ref = f * np.asarray(c0) + i * g
    skip = np.asarray(x) if use_raw_skip else np.asarray(x) @ np.asarray(cell.ir.kernel.value)
    return o * (np.tanh(c_ref) + skip), c_ref


def test_reslstm_cell_uses_the_raw_input_skip_when_sizes_match():
    # input_size == hidden_size -> NF skips the weight_ir projection entirely.
    H, B = 5, 3
    cell = ResLSTMCell(H, H, rngs=nnx.Rngs(0))
    rs = np.random.RandomState(0)
    x = jnp.asarray(rs.randn(B, H) * 0.1, dtype=jnp.float32)
    h0 = jnp.asarray(rs.randn(B, H) * 0.1, dtype=jnp.float32)
    c0 = jnp.asarray(rs.randn(B, H) * 0.1, dtype=jnp.float32)
    (h_new, c_new), _ = cell.step((h0, c0), x)
    h_ref, c_ref = _reslstm_reference(cell, x, h0, c0, H, use_raw_skip=True)
    np.testing.assert_allclose(np.asarray(c_new), c_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(h_new), h_ref, rtol=1e-4, atol=1e-4)


def test_reslstm_cell_uses_the_bias_free_ir_projection_when_sizes_differ():
    F, H, B = 3, 5, 3
    cell = ResLSTMCell(F, H, rngs=nnx.Rngs(0))
    assert cell.ir.bias is None      # NF's weight_ir has no bias
    rs = np.random.RandomState(0)
    x = jnp.asarray(rs.randn(B, F) * 0.1, dtype=jnp.float32)
    h0 = jnp.asarray(rs.randn(B, H) * 0.1, dtype=jnp.float32)
    c0 = jnp.asarray(rs.randn(B, H) * 0.1, dtype=jnp.float32)
    (h_new, c_new), _ = cell.step((h0, c0), x)
    h_ref, c_ref = _reslstm_reference(cell, x, h0, c0, H, use_raw_skip=False)
    np.testing.assert_allclose(np.asarray(c_new), c_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(h_new), h_ref, rtol=1e-4, atol=1e-4)


def test_reslstm_allocates_ir_even_when_unused():
    # NF allocates weight_ir unconditionally; the parameter count must match in
    # both branches so a state_dict transfers either way.
    cell = ResLSTMCell(5, 5, rngs=nnx.Rngs(0))
    assert cell.ir.kernel.value.shape == (5, 5)


def test_attentive_lstm_step_raises_not_implemented():
    layer = AttentiveLSTMLayer(2, 4, rngs=nnx.Rngs(0))
    with pytest.raises(NotImplementedError, match="run_sequence"):
        layer.step(layer.init_carry(3), jnp.zeros((3, 2), dtype=jnp.float32))


def test_attentive_lstm_run_sequence_shape():
    T, B, F, H = 6, 2, 3, 4
    layer = AttentiveLSTMLayer(F, H, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(T, B, F) * 0.1, dtype=jnp.float32)
    out = layer.run_sequence(x, layer.init_carry(B))
    assert out.shape == (T, B, H)
    assert jnp.all(jnp.isfinite(out))


def test_attentive_lstm_attention_weights_sum_to_one_over_time():
    # Recompute the first step's attention distribution with the layer's own
    # weights: softmax is taken over the TIME axis, so each (batch) column of
    # beta must sum to 1 across all T scored timesteps.
    T, B, F, H = 6, 2, 3, 4
    layer = AttentiveLSTMLayer(F, H, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(T, B, F) * 0.1, dtype=jnp.float32)
    h, c = layer.init_carry(B)
    scored = jnp.concatenate(
        [x, jnp.broadcast_to(h, (T,) + h.shape), jnp.broadcast_to(c, (T,) + c.shape)],
        axis=-1,
    )
    beta = jax.nn.softmax(layer.attn_out(jnp.tanh(layer.attn_in(scored))), axis=0)
    assert beta.shape == (T, B, 1)
    np.testing.assert_allclose(np.asarray(beta.sum(axis=0)), np.ones((B, 1)), rtol=1e-5)
    assert np.all(np.asarray(beta) >= 0.0)


def test_attentive_lstm_context_is_a_convex_combination_of_the_window():
    # context = sum_t beta_t * x_t with beta summing to 1 -> it lies inside the
    # elementwise [min, max] box of the window.
    T, B, F, H = 6, 2, 3, 4
    layer = AttentiveLSTMLayer(F, H, rngs=nnx.Rngs(0))
    xnp = np.random.RandomState(0).randn(T, B, F) * 0.1
    x = jnp.asarray(xnp, dtype=jnp.float32)
    h, c = layer.init_carry(B)
    scored = jnp.concatenate(
        [x, jnp.broadcast_to(h, (T,) + h.shape), jnp.broadcast_to(c, (T,) + c.shape)],
        axis=-1,
    )
    beta = jax.nn.softmax(layer.attn_out(jnp.tanh(layer.attn_in(scored))), axis=0)
    context = np.asarray(jnp.sum(beta * x, axis=0))
    assert np.all(context <= xnp.max(axis=0) + 1e-5)
    assert np.all(context >= xnp.min(axis=0) - 1e-5)


# ---- make_cell / CELL_TYPES --------------------------------------------------

@pytest.mark.parametrize("name,cls", [
    ("GRU", GRUCell), ("RNN", RNNCell), ("LSTM", LSTMCell),
    ("ResLSTM", ResLSTMCell), ("AttentiveLSTM", AttentiveLSTMLayer),
])
def test_make_cell_returns_the_right_class(name, cls):
    assert type(make_cell(name, 2, 4, rngs=nnx.Rngs(0))) is cls


def test_make_cell_unknown_raises():
    with pytest.raises(ValueError, match="Unknown cell_type"):
        make_cell("MinimalGRU", 2, 4, rngs=nnx.Rngs(0))


def test_cell_types_content_is_stable():
    assert CELL_TYPES == ("GRU", "RNN", "LSTM", "ResLSTM", "AttentiveLSTM")


# =============================================================================
# Module — DRNN group
# =============================================================================

def test_drnn_layer_count_equals_number_of_dilations():
    drnn = DRNN(n_input=1, n_hidden=6, dilations=[1, 2, 4], rngs=nnx.Rngs(0))
    assert len(drnn.cells) == 3
    assert drnn.dilations == [1, 2, 4]


@pytest.mark.parametrize("dilations", [[1], [1, 2], [2, 4, 8]])
def test_drnn_forward_shape_is_batch_first(dilations):
    B, T, H = 3, 10, 6
    drnn = DRNN(n_input=1, n_hidden=H, dilations=dilations, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(B, T, 1) * 0.1, dtype=jnp.float32)
    out = drnn(x)
    assert out.shape == (B, T, H)
    assert jnp.all(jnp.isfinite(out))


def test_drnn_handles_length_not_divisible_by_any_rate():
    B, T, H = 2, 7, 5
    drnn = DRNN(n_input=1, n_hidden=H, dilations=[2, 4], rngs=nnx.Rngs(0))
    out = drnn(jnp.ones((B, T, 1), dtype=jnp.float32))
    assert out.shape == (B, T, H)


def test_drnn_first_cell_takes_n_input_and_the_rest_take_n_hidden():
    drnn = DRNN(n_input=1, n_hidden=6, dilations=[1, 2], cell_type="GRU", rngs=nnx.Rngs(0))
    assert drnn.cells[0].ih.kernel.value.shape == (1, 18)   # 3 * hidden
    assert drnn.cells[1].ih.kernel.value.shape == (6, 18)


def test_drnn_rejects_empty_dilations():
    with pytest.raises(ValueError, match="at least one dilation"):
        DRNN(n_input=1, n_hidden=4, dilations=[], rngs=nnx.Rngs(0))


@pytest.mark.parametrize("bad", [0, -1])
def test_drnn_rejects_rates_below_one(bad):
    with pytest.raises(ValueError, match="must be >= 1"):
        DRNN(n_input=1, n_hidden=4, dilations=[1, bad], rngs=nnx.Rngs(0))


@pytest.mark.parametrize("cell_type", list(CELL_TYPES))
def test_drnn_forwards_with_every_cell_type(cell_type):
    drnn = DRNN(n_input=1, n_hidden=4, dilations=[1, 2], cell_type=cell_type,
                rngs=nnx.Rngs(0))
    out = drnn(jnp.asarray(np.random.RandomState(0).randn(2, 8, 1) * 0.1,
                           dtype=jnp.float32))
    assert out.shape == (2, 8, 4)
    assert jnp.all(jnp.isfinite(out))


# =============================================================================
# Module — MLP decoder
# =============================================================================

def test_mlp_decoder_one_layer_is_a_bare_linear_without_relu():
    # num_layers=1 must NOT apply an activation: feed inputs whose linear output
    # is negative and check the negatives survive (a ReLU would clamp them to 0).
    dec = MLPDecoder(in_features=4, hidden_size=8, out_features=3, num_layers=1,
                     rngs=nnx.Rngs(0))
    assert len(dec.layers) == 1
    x = jnp.asarray(np.random.RandomState(0).randn(6, 4) * 5.0, dtype=jnp.float32)
    out = np.asarray(dec(x))
    lin = (np.asarray(x) @ np.asarray(dec.layers[0].kernel.value)
           + np.asarray(dec.layers[0].bias.value))
    np.testing.assert_allclose(out, lin, rtol=1e-4, atol=1e-4)
    assert out.min() < 0.0     # would be impossible behind a ReLU


def test_mlp_decoder_two_layers_shapes():
    dec = MLPDecoder(in_features=4, hidden_size=8, out_features=3, num_layers=2,
                     rngs=nnx.Rngs(0))
    assert len(dec.layers) == 2
    assert dec.layers[0].kernel.value.shape == (4, 8)
    assert dec.layers[1].kernel.value.shape == (8, 3)
    assert dec(jnp.ones((5, 4), dtype=jnp.float32)).shape == (5, 3)


def test_mlp_decoder_three_layers_shapes():
    dec = MLPDecoder(in_features=4, hidden_size=8, out_features=3, num_layers=3,
                     rngs=nnx.Rngs(0))
    assert len(dec.layers) == 3
    assert [l.kernel.value.shape for l in dec.layers] == [(4, 8), (8, 8), (8, 3)]
    assert dec(jnp.ones((5, 4), dtype=jnp.float32)).shape == (5, 3)


def test_mlp_decoder_applies_relu_between_layers_only():
    # With >= 2 layers the hidden activations are non-negative, but the FINAL
    # layer has no activation, so the output may still be negative.
    dec = MLPDecoder(in_features=4, hidden_size=8, out_features=3, num_layers=2,
                     rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(20, 4) * 5.0, dtype=jnp.float32)
    hidden = jax.nn.relu(dec.layers[0](x))
    ref = np.asarray(dec.layers[1](hidden))
    np.testing.assert_allclose(np.asarray(dec(x)), ref, rtol=1e-5, atol=1e-5)
    assert ref.min() < 0.0


@pytest.mark.parametrize("bad", [0, -1])
def test_mlp_decoder_rejects_fewer_than_one_layer(bad):
    with pytest.raises(ValueError, match="decoder_layers must be >= 1"):
        MLPDecoder(in_features=4, hidden_size=8, out_features=3, num_layers=bad,
                   rngs=nnx.Rngs(0))


# =============================================================================
# Module — DilatedRNNNet
# =============================================================================

def _net(h=4, input_size=24, hidden=8, dilations=None, cell_type="LSTM", seed=0):
    return DilatedRNNNet(
        h=h, input_size=input_size, in_features=1, cell_type=cell_type,
        dilations=[[1, 2]] if dilations is None else dilations,
        encoder_hidden_size=hidden, decoder_hidden_size=hidden, decoder_layers=2,
        rngs=nnx.Rngs(seed),
    )


def test_net_forward_shape():
    net = _net()
    out = net(jnp.ones((5, 24, 1), dtype=jnp.float32), deterministic=True)
    assert out.shape == (5, 4, 1)
    assert out.dtype == jnp.float32


def test_net_forward_is_deterministic_flag_insensitive():
    # DilatedRNN has no stochastic layers; the flag exists only for interface
    # symmetry with the other neural ports.
    net = _net()
    x = jnp.asarray(np.random.RandomState(0).randn(3, 24, 1) * 0.1, dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(net(x, deterministic=True)),
                               np.asarray(net(x, deterministic=False)), rtol=1e-6)


def test_net_context_adapter_maps_input_size_to_h():
    net = _net(h=4, input_size=24)
    assert net.context_adapter.kernel.value.shape == (24, 4)
    assert net.context_adapter.bias.value.shape == (4,)


def test_net_group_count_matches_dilations():
    net = _net(dilations=[[1, 2], [4, 8]])
    assert len(net.rnn_stack) == 2
    assert [g.dilations for g in net.rnn_stack] == [[1, 2], [4, 8]]


def test_net_second_group_consumes_encoder_hidden_size():
    net = _net(dilations=[[1], [2]], hidden=8, cell_type="GRU")
    assert net.rnn_stack[0].cells[0].ih.kernel.value.shape == (1, 24)   # in_features=1
    assert net.rnn_stack[1].cells[0].ih.kernel.value.shape == (8, 24)   # hidden


def test_net_residual_applies_between_groups_only():
    # Two nets that differ ONLY in grouping: [[1, 2]] builds one group of two
    # layers (no residual), [[1], [2]] builds two groups of one layer (residual
    # added after the second). Cells are created in the same order with the same
    # rngs stream, so the parameters are identical and any output difference is
    # the residual — the exact thing NF's `if layer_num > 0` controls.
    x = jnp.asarray(np.random.RandomState(0).randn(3, 24, 1) * 0.5, dtype=jnp.float32)
    one_group = np.asarray(_net(dilations=[[1, 2]], seed=0)(x))
    two_groups = np.asarray(_net(dilations=[[1], [2]], seed=0)(x))
    assert one_group.shape == two_groups.shape
    assert not np.allclose(one_group, two_groups)


def test_net_residual_adds_the_group_input():
    # Pin the exact residual expression: for group i > 0 the output is
    # group_i(inp) + inp, where inp is that group's INPUT (not the raw series).
    net = _net(dilations=[[1], [2]], seed=0)
    x = jnp.asarray(np.random.RandomState(1).randn(2, 24, 1) * 0.5, dtype=jnp.float32)
    out0 = net.rnn_stack[0](x)                 # first group: no residual
    out1 = net.rnn_stack[1](out0) + out0       # second group: residual
    ctx = jnp.transpose(net.context_adapter(jnp.transpose(out1, (0, 2, 1))), (0, 2, 1))
    ref = np.asarray(net.mlp_decoder(ctx))
    np.testing.assert_allclose(np.asarray(net(x)), ref, rtol=1e-5, atol=1e-5)


def test_net_single_group_has_no_residual():
    net = _net(dilations=[[1, 2]], seed=0)
    x = jnp.asarray(np.random.RandomState(2).randn(2, 24, 1) * 0.5, dtype=jnp.float32)
    enc = net.rnn_stack[0](x)                  # NOT enc + x
    ctx = jnp.transpose(net.context_adapter(jnp.transpose(enc, (0, 2, 1))), (0, 2, 1))
    ref = np.asarray(net.mlp_decoder(ctx))
    np.testing.assert_allclose(np.asarray(net(x)), ref, rtol=1e-5, atol=1e-5)


def test_net_rejects_empty_dilations():
    with pytest.raises(ValueError, match="at least one group"):
        DilatedRNNNet(h=4, input_size=24, dilations=[], encoder_hidden_size=4,
                      decoder_hidden_size=4, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("cell_type", list(CELL_TYPES))
def test_net_forwards_with_every_cell_type(cell_type):
    net = _net(hidden=4, cell_type=cell_type)
    out = net(jnp.asarray(np.random.RandomState(0).randn(2, 24, 1) * 0.1,
                          dtype=jnp.float32))
    assert out.shape == (2, 4, 1)
    assert jnp.all(jnp.isfinite(out))


def test_net_input_shorter_than_the_largest_rate_still_forwards():
    # rate 8 > input_size 6: a single dilated step after zero-padding.
    net = DilatedRNNNet(h=2, input_size=6, dilations=[[8]], encoder_hidden_size=4,
                        decoder_hidden_size=4, decoder_layers=1, rngs=nnx.Rngs(0))
    out = net(jnp.ones((2, 6, 1), dtype=jnp.float32))
    assert out.shape == (2, 2, 1) and jnp.all(jnp.isfinite(out))


# =============================================================================
# Training
# =============================================================================

def _train_net(h=4, input_size=24, hidden=8, seed=0):
    return _net(h=h, input_size=input_size, hidden=hidden, seed=seed)


def test_build_windows_shape():
    y = _make_y(60)
    w = build_windows(y, input_size=24, h=4)
    assert w.shape == (60 - 28 + 1, 28)


def test_build_windows_is_a_step_one_rolling_view():
    y = jnp.arange(10, dtype=jnp.float32)
    w = np.asarray(build_windows(y, input_size=3, h=1))
    np.testing.assert_array_equal(w[0], np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(w[1], np.array([1.0, 2.0, 3.0, 4.0]))
    assert w.shape == (7, 4)


def test_build_windows_raises_when_too_short():
    with pytest.raises(ValueError, match="too short"):
        build_windows(_make_y(20), input_size=24, h=4)


# ---- make_lr_schedule --------------------------------------------------------

def test_lr_schedule_halves_on_a_staircase():
    # num_lr_decays=3 over max_steps=300 -> step_size 100, gamma 0.5 (NF StepLR).
    sched = make_lr_schedule(1e-3, max_steps=300, num_lr_decays=3)
    assert float(sched(0)) == pytest.approx(1e-3)
    assert float(sched(99)) == pytest.approx(1e-3)      # still on the first tread
    assert float(sched(100)) == pytest.approx(5e-4)     # first halving
    assert float(sched(199)) == pytest.approx(5e-4)
    assert float(sched(200)) == pytest.approx(2.5e-4)   # second halving
    assert float(sched(300)) == pytest.approx(1.25e-4)  # third halving


@pytest.mark.parametrize("num_lr_decays", [0, -1])
def test_lr_schedule_disabled_returns_the_scalar_unchanged(num_lr_decays):
    assert make_lr_schedule(1e-3, max_steps=300, num_lr_decays=num_lr_decays) == 1e-3


def test_lr_schedule_passes_callables_through():
    fn = lambda step: 1e-3
    assert make_lr_schedule(fn, max_steps=300, num_lr_decays=3) is fn


def test_lr_schedule_step_size_is_at_least_one():
    # max_steps // num_lr_decays == 0 must not produce a zero transition period.
    sched = make_lr_schedule(1e-3, max_steps=2, num_lr_decays=5)
    assert float(sched(0)) == pytest.approx(1e-3)
    assert float(sched(1)) == pytest.approx(5e-4)


# ---- scaled_forward_loss / train / predict_step ------------------------------

def test_scaled_forward_loss_returns_a_finite_scalar():
    net = _train_net()
    w = build_windows(_make_y(), input_size=24, h=4)
    loss = scaled_forward_loss(net, w[:8], h=4, input_size=24, scaler=RobustScaler())
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_scaled_forward_loss_is_zero_only_via_the_loss_fn():
    # The loss is taken in SCALED space; a loss_fn that ignores its inputs must
    # short-circuit to exactly that constant (guards the plumbing of loss_fn).
    net = _train_net()
    w = build_windows(_make_y(), input_size=24, h=4)
    loss = scaled_forward_loss(net, w[:4], h=4, input_size=24, scaler=RobustScaler(),
                               loss_fn=lambda p, t: jnp.asarray(0.0))
    assert float(loss) == 0.0


def test_train_returns_one_loss_per_step_and_decreases():
    net = _train_net()
    losses = train(net, _make_y(), h=4, input_size=24, max_steps=30,
                   windows_batch_size=8, lr=1e-2, seed=0)
    assert losses.shape == (30,)
    assert jnp.all(jnp.isfinite(losses))
    assert float(losses[-1]) < float(losses[0])


def test_train_mutates_the_model_in_place():
    net = _train_net()
    before = np.asarray(net.context_adapter.kernel.value).copy()
    train(net, _make_y(), h=4, input_size=24, max_steps=10, windows_batch_size=8,
          lr=1e-2, seed=0)
    assert not np.allclose(before, np.asarray(net.context_adapter.kernel.value))


def test_train_is_deterministic_with_the_same_seed():
    y = _make_y()
    l1 = train(_train_net(), y, h=4, input_size=24, max_steps=10,
               windows_batch_size=8, lr=1e-3, seed=0)
    l2 = train(_train_net(), y, h=4, input_size=24, max_steps=10,
               windows_batch_size=8, lr=1e-3, seed=0)
    np.testing.assert_allclose(np.asarray(l1), np.asarray(l2), rtol=1e-6)


def test_train_accepts_an_optax_schedule_for_lr():
    net = _train_net()
    sched = make_lr_schedule(1e-2, max_steps=20, num_lr_decays=2)
    losses = train(net, _make_y(), h=4, input_size=24, max_steps=20,
                   windows_batch_size=8, lr=sched, seed=0)
    assert losses.shape == (20,) and jnp.all(jnp.isfinite(losses))


def test_train_oversample_with_replacement_small_n_regime():
    # n_windows < windows_batch_size -> NF's WITH-replacement branch, the regime
    # every small benchmark series hits. y(30), input_size=24, h=4 -> 3 windows.
    net = _train_net()
    losses = train(net, _make_y(30), h=4, input_size=24, max_steps=8,
                   windows_batch_size=8, lr=1e-3, seed=0)
    assert losses.shape == (8,)
    assert jnp.all(jnp.isfinite(losses))


def test_train_raises_on_divergence():
    # The saturating gates keep the loss finite (if huge) even at lr=1e9, so the
    # non-finite guard needs a genuinely overflowing step size to fire.
    with pytest.raises(RuntimeError, match="diverged"):
        train(_train_net(), _make_y(), h=4, input_size=24, max_steps=10,
              windows_batch_size=8, lr=1e20, seed=0)


def test_predict_step_shape_and_idempotence():
    net = _train_net()
    y = _make_y()
    train(net, y, h=4, input_size=24, max_steps=5, windows_batch_size=8, lr=1e-3, seed=0)
    sc = RobustScaler()
    p1 = predict_step(net, y, h=4, input_size=24, scaler=sc)
    p2 = predict_step(net, y, h=4, input_size=24, scaler=sc)
    assert p1.shape == (4,)
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-6)


def test_predict_step_only_reads_the_last_input_size_values():
    net = _train_net()
    y = _make_y()
    sc = RobustScaler()
    tail_only = predict_step(net, y[-24:], h=4, input_size=24, scaler=sc)
    full = predict_step(net, y, h=4, input_size=24, scaler=sc)
    np.testing.assert_allclose(np.asarray(tail_only), np.asarray(full),
                               rtol=1e-5, atol=1e-5)


def test_predict_step_inverts_back_to_the_series_scale():
    # A constant context inverts to (approximately) that constant regardless of
    # the network's scaled-space output magnitude — checks the inverse wiring.
    net = _train_net()
    sc = RobustScaler()
    y = jnp.full((40,), 100.0, dtype=jnp.float32)
    out = np.asarray(predict_step(net, y, h=4, input_size=24, scaler=sc))
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out - 100.0) < 10.0)   # scale is ~1 for a constant window


# =============================================================================
# Model (BaseForecaster conformance)
# =============================================================================

def _tiny(**kw):
    params = dict(h=4, input_size=24, encoder_hidden_size=8, decoder_hidden_size=8,
                  dilations=[[1, 2]], max_steps=20, windows_batch_size=8,
                  random_seed=0)
    params.update(kw)
    return DilatedRNN(**params)


def test_dilated_rnn_inherits_baseforecaster():
    assert issubclass(DilatedRNN, BaseForecaster)


def test_init_sets_conformal_params_to_none():
    assert _tiny().conformal_params is None


def test_init_does_not_build_the_net():
    assert _tiny().model_ is None


def test_input_size_default_resolves_to_three_h():
    assert DilatedRNN(h=10).input_size == 30


def test_input_size_explicit_is_respected():
    assert DilatedRNN(h=10, input_size=17).input_size == 17


def test_uses_exog_is_false():
    assert DilatedRNN.uses_exog is False


def test_init_rejects_unknown_cell_type():
    with pytest.raises(ValueError, match="Unknown cell_type"):
        _tiny(cell_type="MinimalGRU")


def test_init_rejects_empty_dilations():
    with pytest.raises(ValueError, match="non-empty"):
        _tiny(dilations=[])


def test_init_rejects_an_empty_dilation_group():
    with pytest.raises(ValueError, match="non-empty"):
        _tiny(dilations=[[1, 2], []])


@pytest.mark.parametrize("bad", [0, -3])
def test_init_rejects_non_positive_dilation_rates(bad):
    with pytest.raises(ValueError, match="must be >= 1"):
        _tiny(dilations=[[1, bad]])


@pytest.mark.parametrize("bad", [0, -1])
def test_init_rejects_decoder_layers_below_one(bad):
    with pytest.raises(ValueError, match="decoder_layers must be >= 1"):
        _tiny(decoder_layers=bad)


def test_init_rejects_unknown_scaler_type():
    with pytest.raises(ValueError, match="Unknown scaler_type"):
        _tiny(scaler_type="minmax")


def test_fit_returns_self_and_sets_model():
    m = _tiny()
    out = m.fit(_make_y())
    assert out is m
    assert m.model_ is not None


def test_fit_raises_on_exog():
    with pytest.raises(NotImplementedError):
        _tiny().fit(_make_y(), X=jnp.ones((200, 1)))


def test_fit_raises_on_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        _tiny().fit(jnp.ones((200, 2), dtype=jnp.float32))


def test_fit_raises_on_short_series():
    with pytest.raises(ValueError, match="too short"):
        _tiny().fit(_make_y(20))   # < input_size + h = 28


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        _tiny().predict(h=4)


def test_predict_default_returns_h_steps():
    m = _tiny().fit(_make_y())
    assert m.predict(h=4)["mean"].shape == (4,)


def test_predict_smaller_h_slices():
    m = _tiny().fit(_make_y())
    full = np.asarray(m.predict(h=4)["mean"])
    sliced = np.asarray(m.predict(h=2)["mean"])
    assert sliced.shape == (2,)
    np.testing.assert_allclose(sliced, full[:2], rtol=1e-6)


def test_predict_larger_h_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="not supported"):
        m.predict(h=5)


def test_predict_h_below_one_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="positive"):
        m.predict(h=0)


def test_predict_deterministic_with_the_same_seed():
    y = _make_y()
    p1 = _tiny().fit(y).predict(h=4)["mean"]
    p2 = _tiny().fit(y).predict(h=4)["mean"]
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-5, atol=1e-5)


def test_predict_differs_across_seeds():
    y = _make_y()
    p1 = np.asarray(_tiny(random_seed=0).fit(y).predict(h=4)["mean"])
    p2 = np.asarray(_tiny(random_seed=7).fit(y).predict(h=4)["mean"])
    assert not np.allclose(p1, p2)


@pytest.mark.parametrize("cell_type", list(CELL_TYPES))
def test_fit_predict_finite_for_every_cell_type(cell_type):
    m = _tiny(cell_type=cell_type, max_steps=5).fit(_make_y(80))
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


@pytest.mark.parametrize("scaler_type", ["robust", "standard", "identity"])
def test_fit_predict_finite_for_every_scaler(scaler_type):
    m = _tiny(scaler_type=scaler_type, max_steps=5).fit(_make_y(80))
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


def test_constant_series_returns_finite():
    m = _tiny().fit(jnp.ones(200, dtype=jnp.float32))
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


def test_loss_custom_callable_trains():
    def my_loss(pred, target):
        return jnp.mean(jnp.abs(pred - target))
    m = _tiny(loss=my_loss).fit(_make_y())
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


def test_loss_unknown_string_raises_at_fit():
    with pytest.raises(ValueError, match="Unknown loss"):
        _tiny(loss="rmse", max_steps=2).fit(_make_y())


def test_num_lr_decays_disabled_still_fits():
    m = _tiny(num_lr_decays=0).fit(_make_y())
    assert jnp.all(jnp.isfinite(m.predict(h=4)["mean"]))


def test_context_size_is_accepted_but_inert():
    # NF stores context_size and never reads it in DilatedRNN.forward; parity
    # means configs transfer unchanged AND predictions are unaffected.
    y = _make_y()
    a = np.asarray(_tiny(context_size=10).fit(y).predict(h=4)["mean"])
    b = np.asarray(_tiny(context_size=99).fit(y).predict(h=4)["mean"])
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_h_equals_one():
    m = DilatedRNN(h=1, input_size=12, encoder_hidden_size=8, decoder_hidden_size=8,
                   dilations=[[1, 2]], max_steps=5, windows_batch_size=8, random_seed=0)
    m.fit(_make_y(60))
    assert m.predict(h=1)["mean"].shape == (1,)


def test_model_beats_naive_last_value_on_easy_signal():
    # End-to-end correctness floor: catches broken loss/optimizer/scaler-inverse/
    # dilation wiring that still passes every shape and determinism test.
    n, h = 240, 4
    y = jnp.asarray(np.sin(np.arange(n) / 2.0), dtype=jnp.float32)
    m = DilatedRNN(h=h, input_size=24, encoder_hidden_size=16, decoder_hidden_size=16,
                   dilations=[[1, 2]], max_steps=400, windows_batch_size=8,
                   random_seed=0)
    m.fit(y[:-h])
    pred = np.asarray(m.predict(h=h)["mean"])
    y_true = np.asarray(y[-h:])
    naive = np.full(h, float(y[-h - 1]))
    assert np.mean(np.abs(pred - y_true)) < np.mean(np.abs(naive - y_true))


def test_build_net_works_in_loop_construction():
    # precursor to conformity_scores' vmap: constructing the net in a loop
    nets = [_tiny()._build_net() for _ in range(3)]
    assert len(nets) == 3


def test_pickle_round_trip_preserves_predictions():
    m = _tiny().fit(_make_y())
    before = np.asarray(m.predict(h=4)["mean"])
    m2 = pickle.loads(pickle.dumps(m))
    after = np.asarray(m2.predict(h=4)["mean"])
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


def test_pickle_round_trip_before_fit():
    m2 = pickle.loads(pickle.dumps(_tiny()))
    assert m2.model_ is None
    assert jnp.all(jnp.isfinite(m2.fit(_make_y()).predict(h=4)["mean"]))


@pytest.mark.parametrize("loss_name", ["mae", "mse", "huber"])
def test_loss_string_pickle_round_trip(loss_name):
    m = _tiny(loss=loss_name, max_steps=10).fit(_make_y())
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(
        np.asarray(m2.predict(h=4)["mean"]), np.asarray(m.predict(h=4)["mean"]),
        rtol=1e-5, atol=1e-5,
    )


def test_forecast_equals_fit_then_predict_same_seed():
    y = _make_y()
    a = _tiny().forecast(y, h=4)["mean"]
    b = _tiny().fit(y).predict(h=4)["mean"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


def test_forecast_raises_on_exog():
    with pytest.raises(NotImplementedError):
        _tiny().forecast(_make_y(), h=4, X=jnp.ones((200, 1)))


def test_forecast_raises_on_future_exog():
    with pytest.raises(NotImplementedError):
        _tiny().forecast(_make_y(), h=4, X_future=jnp.ones((4, 1)))


def test_forecast_fitted_has_nan_head_and_finite_tail():
    res = _tiny().forecast(_make_y(), h=4, fitted=True)
    fitted = np.asarray(res["fitted"])
    assert fitted.shape == (200,)
    assert np.all(np.isnan(fitted[:24]))     # NaN head of length input_size
    assert np.all(np.isfinite(fitted[24:]))


def test_forecast_without_fitted_has_no_fitted_key():
    assert "fitted" not in _tiny().forecast(_make_y(), h=4)


# ---- conformal intervals -----------------------------------------------------

def _conformal_model():
    return DilatedRNN(h=4, input_size=12, encoder_hidden_size=8, decoder_hidden_size=8,
                      dilations=[[1, 2]], max_steps=2, windows_batch_size=8,
                      random_seed=0)


def test_predict_with_level_returns_interval_keys():
    m = _conformal_model().fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    out = m.predict(h=4, level=[80])
    assert "lo-80" in out and "hi-80" in out


def test_predict_with_level_raises_without_conformal_params():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="conformal_params"):
        m.predict(h=4, level=[80])


def test_conformity_scores_returns_finite_2d_array():
    m = _conformal_model().fit(_make_y(80))
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    cs = m.conformity_scores(_make_y(80))
    assert cs.ndim == 2 and cs.shape[1] == 4
    assert jnp.all(jnp.isfinite(cs))


def test_conformal_does_not_corrupt_fitted_model():
    # conformity_scores re-fits inside a vmap; predict(level=...) must run it on a
    # throwaway copy (self.new()) so the tracer-valued refits don't overwrite
    # self.model_. Without the fix the second predict differs or raises a tracer leak.
    m = _conformal_model().fit(_make_y(80))
    before = np.asarray(m.predict(h=4)["mean"])
    m.conformal_params = ConformalIntervals(h=4, n_windows=3)
    _ = m.predict(h=4, level=[80])
    after = np.asarray(m.predict(h=4)["mean"])
    assert np.all(np.isfinite(after))
    assert np.allclose(before, after)


# =============================================================================
# Package exports / discovery
# =============================================================================

def test_dilated_rnn_importable_from_models_namespace():
    from chronax.models import DilatedRNN as D
    assert D is DilatedRNN


def test_dilated_rnn_in_models_all():
    import chronax.models as models
    assert "DilatedRNN" in models.__all__


def test_dilated_rnn_importable_from_subpackage_all():
    import chronax.models.dilated_rnn as pkg
    assert pkg.__all__ == ["DilatedRNN"]
    assert pkg.DilatedRNN is DilatedRNN


def test_dilated_rnn_is_auto_discovered_by_the_benchmark_registry():
    registry = pytest.importorskip("benchmarks.neural.registry")
    assert "DilatedRNN" in registry.list_models()
    assert registry.resolve_chronax("DilatedRNN") is DilatedRNN

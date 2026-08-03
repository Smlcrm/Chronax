"""Flax NNX modules for the DilatedRNN forecaster.

Univariate JAX/Flax-NNX port of ``neuralforecast.DilatedRNN`` (Chang et al. 2017,
"Dilated Recurrent Neural Networks", arXiv:1710.02224).

Architecture, following NF's ``DilatedRNN.forward`` exactly:

    [B, L, 1] --(stack of DRNN groups, residual between groups)--> [B, L, H]
             --transpose--> [B, H, L] --context_adapter Linear(L -> h)--> [B, H, h]
             --transpose--> [B, h, H] --MLP decoder--> [B, h, 1]

A **DRNN group** is a list of dilation rates (default ``[[1, 2], [4, 8]]`` = two
groups of two layers). Within a group each layer runs its recurrent cell over the
sequence *subsampled by its dilation rate*: at rate ``r`` the length-``T`` sequence
is split into ``r`` interleaved subsequences that are processed as ``r`` extra
batch rows, so one cell pass covers ``T/r`` timesteps and the cell's receptive
field grows geometrically with depth at constant cost. Between groups (not within
one) NF adds a residual connection.

The dilation is implemented as pure reshapes rather than NF's Python list
comprehensions:

* pack   ``[T, B, F] -> [T/r, r*B, F]`` (``inputs.reshape``) — column block ``i``
  holds ``inputs[i::r]``, which is exactly NF ``_prepare_inputs``' concat;
* unpack ``[T/r, r*B, H] -> [T, B, H]`` (``outputs.reshape``) — the row-major
  interleave NF's ``_split_outputs`` builds by stack/transpose/reshape.

Both identities are pinned by tests against a literal transcription of NF's
version. Sequences whose length is not a multiple of ``r`` are zero-padded at the
END and truncated after unpacking, as NF does.

Five cell types are supported, matching NF: ``GRU``, ``RNN``, ``LSTM`` (torch's
built-in cells, initialized ``U(-1/sqrt(H), 1/sqrt(H))`` like ``nn.GRU`` &c.), plus
NF's own ``ResLSTM`` and ``AttentiveLSTM`` (whose parameters NF initializes with
``torch.randn``, i.e. standard normal — reproduced here, unusual as it is, because
it materially changes the training trajectory).

``float32`` throughout.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _TorchLinearInit:
    """Picklable initializer matching torch ``nn.Linear`` default.

    torch draws both weight and bias from ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``
    (Kaiming-uniform with ``a=sqrt(5)`` reduces to this bound). Flax's
    ``nnx.Linear`` defaults to ``lecun_normal`` instead, so matching torch here is
    what makes training trajectories — and thus accuracy — comparable to
    neuralforecast.
    """

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class _TorchRNNInit:
    """Picklable initializer matching torch's built-in ``nn.RNN``/``GRU``/``LSTM``.

    ``reset_parameters`` draws EVERY weight and bias from
    ``U(-1/sqrt(hidden_size), 1/sqrt(hidden_size))`` — note the bound depends on
    the hidden size, not on the layer's fan-in.
    """

    __slots__ = ("hidden_size",)

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.hidden_size)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class _StandardNormalInit:
    """Picklable ``torch.randn`` initializer (standard normal, no fan-in scaling).

    NF's hand-written ``LSTMCell`` / ``ResLSTMCell`` allocate every parameter with
    ``torch.randn(...)``. That is a much wider init than torch's built-in cells
    use; it is reproduced verbatim so the ``ResLSTM`` / ``AttentiveLSTM`` cell types
    train along the same trajectory as the reference.
    """

    __slots__ = ()

    def __call__(self, key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype)


# =============================================================================
# Recurrent cells
# =============================================================================

class _CellBase(nnx.Module):
    """Common sequence runner for the elementwise cells.

    Subclasses implement ``step(carry, x_t) -> (carry, out)`` and set
    ``hidden_size`` / ``is_lstm``. ``run_sequence`` scans ``step`` over the leading
    (time) axis; :class:`AttentiveLSTMLayer` overrides it because its per-step
    context needs the whole sequence.
    """

    hidden_size: int
    is_lstm: bool = False

    def init_carry(self, batch: int):
        """Zero initial state, matching NF's ``torch.zeros`` hidden."""
        h = jnp.zeros((batch, self.hidden_size), dtype=jnp.float32)
        return (h, h) if self.is_lstm else h

    def step(self, carry, x_t):  # pragma: no cover - abstract
        raise NotImplementedError

    def run_sequence(self, inputs: jnp.ndarray, carry) -> jnp.ndarray:
        """inputs: [T, B, F] (time-first) -> outputs: [T, B, H]."""
        _, outs = jax.lax.scan(lambda c, x: self.step(c, x), carry, inputs)
        return outs


class RNNCell(_CellBase):
    """torch ``nn.RNN`` (tanh) cell: ``h' = tanh(x W_ih^T + b_ih + h W_hh^T + b_hh)``."""

    is_lstm = False

    def __init__(self, input_size: int, hidden_size: int, *, rngs: nnx.Rngs):
        init = _TorchRNNInit(hidden_size)
        self.hidden_size = hidden_size
        self.ih = nnx.Linear(input_size, hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        self.hh = nnx.Linear(hidden_size, hidden_size, kernel_init=init, bias_init=init, rngs=rngs)

    def step(self, carry, x_t):
        h = jnp.tanh(self.ih(x_t) + self.hh(carry))
        return h, h


class GRUCell(_CellBase):
    """torch ``nn.GRU`` cell.

    ``r = sigma(W_ir x + b_ir + W_hr h + b_hr)``,
    ``z = sigma(W_iz x + b_iz + W_hz h + b_hz)``,
    ``n = tanh(W_in x + b_in + r * (W_hn h + b_hn))``,
    ``h' = (1 - z) * n + z * h``.

    Both biases are kept separate (torch has ``b_ih`` and ``b_hh``); ``r`` gates the
    hidden contribution AFTER its bias, which is what distinguishes torch's GRU
    from the fused-bias variant ``flax.nnx.GRUCell`` implements.
    """

    is_lstm = False

    def __init__(self, input_size: int, hidden_size: int, *, rngs: nnx.Rngs):
        init = _TorchRNNInit(hidden_size)
        self.hidden_size = hidden_size
        self.ih = nnx.Linear(input_size, 3 * hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        self.hh = nnx.Linear(hidden_size, 3 * hidden_size, kernel_init=init, bias_init=init, rngs=rngs)

    def step(self, carry, x_t):
        H = self.hidden_size
        gi, gh = self.ih(x_t), self.hh(carry)
        i_r, i_z, i_n = gi[:, :H], gi[:, H:2 * H], gi[:, 2 * H:]
        h_r, h_z, h_n = gh[:, :H], gh[:, H:2 * H], gh[:, 2 * H:]
        r = jax.nn.sigmoid(i_r + h_r)
        z = jax.nn.sigmoid(i_z + h_z)
        n = jnp.tanh(i_n + r * h_n)
        h = (1.0 - z) * n + z * carry
        return h, h


class LSTMCell(_CellBase):
    """LSTM cell with torch's gate order ``(i, f, g, o)``.

    ``gates = x W_ih^T + b_ih + h W_hh^T + b_hh``; ``c' = f * c + i * g``;
    ``h' = o * tanh(c')``. Used both for ``cell_type="LSTM"`` (torch's built-in
    ``nn.LSTM``, hence the ``U(-1/sqrt(H))`` init) and, with a standard-normal init,
    inside :class:`AttentiveLSTMLayer` (NF's hand-written cell).
    """

    is_lstm = True

    def __init__(self, input_size: int, hidden_size: int, *, randn_init: bool = False,
                 rngs: nnx.Rngs):
        init = _StandardNormalInit() if randn_init else _TorchRNNInit(hidden_size)
        self.hidden_size = hidden_size
        self.ih = nnx.Linear(input_size, 4 * hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        self.hh = nnx.Linear(hidden_size, 4 * hidden_size, kernel_init=init, bias_init=init, rngs=rngs)

    def step(self, carry, x_t):
        h, c = carry
        H = self.hidden_size
        gates = self.ih(x_t) + self.hh(h)
        i = jax.nn.sigmoid(gates[:, :H])
        f = jax.nn.sigmoid(gates[:, H:2 * H])
        g = jnp.tanh(gates[:, 2 * H:3 * H])
        o = jax.nn.sigmoid(gates[:, 3 * H:])
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new


class ResLSTMCell(_CellBase):
    """NF's residual LSTM cell (``cell_type="ResLSTM"``).

    Differs from a plain LSTM in three ways, per NF ``ResLSTMCell``:
    the input, forget and output gates each read the CELL state as well
    (``+ c W_ic^T + b_ic``); the candidate is a plain affine map of ``h`` only
    (``W_hh h + b_hh``, no input term); and the output is residual —
    ``h' = o * (tanh(c') + x)`` when ``input_size == hidden_size``, otherwise
    ``h' = o * (tanh(c') + x W_ir^T)`` with a bias-free projection. All parameters
    use NF's ``torch.randn`` init.
    """

    is_lstm = True

    def __init__(self, input_size: int, hidden_size: int, *, rngs: nnx.Rngs):
        init = _StandardNormalInit()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ii = nnx.Linear(input_size, 3 * hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        self.ih = nnx.Linear(hidden_size, 3 * hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        self.ic = nnx.Linear(hidden_size, 3 * hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        self.hh = nnx.Linear(hidden_size, hidden_size, kernel_init=init, bias_init=init, rngs=rngs)
        # NF allocates weight_ir unconditionally but only applies it when the
        # shapes differ; mirrored so the parameter count matches either way.
        self.ir = nnx.Linear(input_size, hidden_size, use_bias=False, kernel_init=init, rngs=rngs)

    def step(self, carry, x_t):
        h, c = carry
        H = self.hidden_size
        ifo = self.ii(x_t) + self.ih(h) + self.ic(c)
        i = jax.nn.sigmoid(ifo[:, :H])
        f = jax.nn.sigmoid(ifo[:, H:2 * H])
        o = jax.nn.sigmoid(ifo[:, 2 * H:])
        g = jnp.tanh(self.hh(h))
        c_new = f * c + i * g
        r = jnp.tanh(c_new)
        skip = x_t if self.input_size == self.hidden_size else self.ir(x_t)
        h_new = o * (r + skip)
        return (h_new, c_new), h_new


class AttentiveLSTMLayer(_CellBase):
    """NF's attentive LSTM layer (``cell_type="AttentiveLSTM"``).

    At every step the layer scores ALL timesteps of the (dilated) input against
    the current state and feeds the attention-weighted average — not the current
    timestep — to an LSTM cell::

        score_t = W2 tanh(W1 [x_t, h, c])        for every t in the window
        beta    = softmax(score, over time)
        context = sum_t beta_t * x_t
        h, c    = LSTMCell(context, (h, c))

    Because the scored tensor is the whole window and the loop index is otherwise
    unused, the recurrence has no per-step input — ``run_sequence`` scans over a
    length-``T`` dummy axis with the window closed over, which is exactly NF's
    ``for t in range(len(inputs))``.
    """

    is_lstm = True

    def __init__(self, input_size: int, hidden_size: int, *, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size, randn_init=True, rngs=rngs)
        in_features = 2 * hidden_size + input_size
        self.attn_in = nnx.Linear(in_features, hidden_size,
                                  kernel_init=_TorchLinearInit(in_features),
                                  bias_init=_TorchLinearInit(in_features), rngs=rngs)
        self.attn_out = nnx.Linear(hidden_size, 1,
                                   kernel_init=_TorchLinearInit(hidden_size),
                                   bias_init=_TorchLinearInit(hidden_size), rngs=rngs)

    def step(self, carry, x_t):  # pragma: no cover - not used (see run_sequence)
        raise NotImplementedError(
            "AttentiveLSTMLayer attends over the whole window; use run_sequence()."
        )

    def run_sequence(self, inputs: jnp.ndarray, carry) -> jnp.ndarray:
        """inputs: [T, B, F] -> outputs: [T, B, H]."""
        n_steps = inputs.shape[0]

        def body(state, _):
            h, c = state
            h_rep = jnp.broadcast_to(h, (n_steps,) + h.shape)
            c_rep = jnp.broadcast_to(c, (n_steps,) + c.shape)
            scored = jnp.concatenate([inputs, h_rep, c_rep], axis=-1)   # [T, B, F+2H]
            logits = self.attn_out(jnp.tanh(self.attn_in(scored)))      # [T, B, 1]
            beta = jax.nn.softmax(logits, axis=0)                       # over time
            context = jnp.sum(beta * inputs, axis=0)                    # [B, F]
            return self.cell.step((h, c), context)

        _, outs = jax.lax.scan(body, carry, None, length=n_steps)
        return outs


CELL_TYPES = ("GRU", "RNN", "LSTM", "ResLSTM", "AttentiveLSTM")


def make_cell(cell_type: str, input_size: int, hidden_size: int, *, rngs: nnx.Rngs) -> _CellBase:
    """Build one recurrent cell by NF's ``cell_type`` name."""
    if cell_type == "GRU":
        return GRUCell(input_size, hidden_size, rngs=rngs)
    if cell_type == "RNN":
        return RNNCell(input_size, hidden_size, rngs=rngs)
    if cell_type == "LSTM":
        return LSTMCell(input_size, hidden_size, rngs=rngs)
    if cell_type == "ResLSTM":
        return ResLSTMCell(input_size, hidden_size, rngs=rngs)
    if cell_type == "AttentiveLSTM":
        return AttentiveLSTMLayer(input_size, hidden_size, rngs=rngs)
    raise ValueError(f"Unknown cell_type {cell_type!r}. Available: {list(CELL_TYPES)}.")


# =============================================================================
# Dilation plumbing
# =============================================================================

def pad_inputs(inputs: jnp.ndarray, rate: int) -> jnp.ndarray:
    """Zero-pad the TIME axis up to a multiple of ``rate`` (NF ``_pad_inputs``)."""
    n_steps = inputs.shape[0]
    remainder = (-n_steps) % rate
    if remainder == 0:
        return inputs
    pad = jnp.zeros((remainder,) + inputs.shape[1:], dtype=inputs.dtype)
    return jnp.concatenate([inputs, pad], axis=0)


def prepare_inputs(inputs: jnp.ndarray, rate: int) -> jnp.ndarray:
    """``[T, B, F] -> [T/r, r*B, F]``; requires ``T % r == 0`` (NF ``_prepare_inputs``).

    Column block ``i`` (columns ``i*B : (i+1)*B``) carries ``inputs[i::r]``, matching
    NF's ``torch.cat([inputs[j::rate] for j in range(rate)], 1)`` — row-major
    reshape reproduces that concat exactly, because element ``[s, i]`` of the
    ``(T/r, r)``-split time axis is timestep ``s*r + i``.
    """
    n_steps, batch, feats = inputs.shape
    if n_steps % rate != 0:
        raise ValueError(f"time length {n_steps} is not a multiple of rate {rate}")
    return inputs.reshape(n_steps // rate, rate * batch, feats)


def split_outputs(dilated_outputs: jnp.ndarray, rate: int) -> jnp.ndarray:
    """``[T/r, r*B, H] -> [T, B, H]`` (NF ``_split_outputs``), the inverse interleave."""
    n_dilated, wide_batch, hidden = dilated_outputs.shape
    if wide_batch % rate != 0:
        raise ValueError(f"batch {wide_batch} is not a multiple of rate {rate}")
    return dilated_outputs.reshape(n_dilated * rate, wide_batch // rate, hidden)


def drnn_layer(cell: _CellBase, inputs: jnp.ndarray, rate: int) -> jnp.ndarray:
    """Run one dilated recurrent layer. inputs: [T, B, F] (time-first) -> [T, B, H]."""
    n_steps = inputs.shape[0]
    padded = pad_inputs(inputs, rate)
    dilated = prepare_inputs(padded, rate)
    carry = cell.init_carry(dilated.shape[1])
    dilated_out = cell.run_sequence(dilated, carry)
    return split_outputs(dilated_out, rate)[:n_steps]


class DRNN(nnx.Module):
    """One dilated-RNN group: ``len(dilations)`` stacked layers, one per rate.

    Layer ``i`` consumes layer ``i-1``'s output at dilation ``dilations[i]``. NF's
    ``DRNN`` also returns a list of per-layer tails (``inputs[-dilation:]``); nothing
    downstream reads it, so it is not built here.
    """

    def __init__(self, *, n_input: int, n_hidden: int, dilations: list[int],
                 cell_type: str = "GRU", rngs: nnx.Rngs):
        if not dilations:
            raise ValueError("a DRNN group needs at least one dilation rate")
        if any(int(r) < 1 for r in dilations):
            raise ValueError(f"dilation rates must be >= 1; got {dilations}")
        self.dilations = [int(r) for r in dilations]
        self.cell_type = cell_type
        self.cells = [
            make_cell(cell_type, n_input if i == 0 else n_hidden, n_hidden, rngs=rngs)
            for i in range(len(self.dilations))
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [B, T, F] (batch-first, as NF's ``batch_first=True``) -> [B, T, H]."""
        h = jnp.transpose(x, (1, 0, 2))              # -> time-first
        for cell, rate in zip(self.cells, self.dilations):
            h = drnn_layer(cell, h, rate)
        return jnp.transpose(h, (1, 0, 2))           # -> batch-first


class MLPDecoder(nnx.Module):
    """NF ``common._modules.MLP`` with ReLU and dropout=0.0.

    ``num_layers=1`` is a bare ``Linear(in -> out)`` with no activation;
    ``num_layers=n >= 2`` is ``Linear(in -> hidden)``, then ``n - 2`` hidden
    ``Linear(hidden -> hidden)`` blocks, then ``Linear(hidden -> out)``, with ReLU
    after every layer except the last.
    """

    def __init__(self, *, in_features: int, hidden_size: int, out_features: int,
                 num_layers: int, rngs: nnx.Rngs):
        if num_layers < 1:
            raise ValueError(f"decoder_layers must be >= 1; got {num_layers}.")
        self.num_layers = num_layers
        if num_layers == 1:
            sizes = [(in_features, out_features)]
        else:
            sizes = [(in_features, hidden_size)]
            sizes += [(hidden_size, hidden_size)] * (num_layers - 2)
            sizes += [(hidden_size, out_features)]
        self.layers = [
            nnx.Linear(a, b, kernel_init=_TorchLinearInit(a), bias_init=_TorchLinearInit(a),
                       rngs=rngs)
            for a, b in sizes
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jax.nn.relu(x)
        return x


class DilatedRNNNet(nnx.Module):
    """Full DilatedRNN backbone: DRNN stack -> context adapter -> MLP decoder.

    I/O mirrors the other Chronax neural nets: ``__call__(x: [B, L, 1]) -> [B, h, 1]``.
    Written N-generically in the feature dimension (``in_features``), so a future
    exogenous path can widen the encoder input without touching the rest.
    """

    def __init__(self, *, h: int, input_size: int, in_features: int = 1,
                 cell_type: str = "LSTM",
                 dilations: list[list[int]] | None = None,
                 encoder_hidden_size: int = 128, decoder_hidden_size: int = 128,
                 decoder_layers: int = 2, rngs: nnx.Rngs):
        dilations = [[1, 2], [4, 8]] if dilations is None else [list(g) for g in dilations]
        if not dilations:
            raise ValueError("dilations must contain at least one group")
        self.h = h
        self.input_size = input_size
        self.dilations = dilations
        self.rnn_stack = [
            DRNN(
                n_input=in_features if g == 0 else encoder_hidden_size,
                n_hidden=encoder_hidden_size, dilations=group, cell_type=cell_type,
                rngs=rngs,
            )
            for g, group in enumerate(dilations)
        ]
        self.context_adapter = nnx.Linear(
            input_size, h, kernel_init=_TorchLinearInit(input_size),
            bias_init=_TorchLinearInit(input_size), rngs=rngs,
        )
        self.mlp_decoder = MLPDecoder(
            in_features=encoder_hidden_size, hidden_size=decoder_hidden_size,
            out_features=1, num_layers=decoder_layers, rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """x: [B, L, in_features] -> [B, h, 1].

        ``deterministic`` is accepted for interface symmetry with the other neural
        ports; DilatedRNN has no stochastic layers (NF builds its decoder with
        ``dropout=0.0`` and its cells never dropout at one layer per RNN), so the
        flag changes nothing.
        """
        encoder_input = x.astype(jnp.float32)
        output = encoder_input
        for layer_num, group in enumerate(self.rnn_stack):
            residual = encoder_input
            output = group(encoder_input)
            if layer_num > 0:
                output = output + residual        # residual BETWEEN groups only
            encoder_input = output
        output = jnp.transpose(output, (0, 2, 1))          # [B, L, H] -> [B, H, L]
        context = self.context_adapter(output)             # [B, H, L] -> [B, H, h]
        context = jnp.transpose(context, (0, 2, 1))        # -> [B, h, H]
        return self.mlp_decoder(context)                   # -> [B, h, 1]

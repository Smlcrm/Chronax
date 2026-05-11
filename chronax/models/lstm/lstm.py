from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class LSTMParams(NamedTuple):
    """Trainable parameters for one LSTM layer and a linear readout."""

    input_kernel: Array
    recurrent_kernel: Array
    bias: Array
    output_kernel: Array
    output_bias: Array


class LSTMState(NamedTuple):
    """Carry state for an LSTM layer."""

    hidden: Array
    cell: Array


@dataclass(frozen=True)
class LSTMModel:
    """A pure-JAX LSTM followed by a linear projection.

    Inputs are expected in time-major form for a single sequence,
    ``(sequence_length, input_size)``. Batched inputs use
    ``(batch_size, sequence_length, input_size)`` and are evaluated with
    :meth:`apply_batch`.
    """

    input_size: int
    hidden_size: int
    output_size: int

    def __post_init__(self) -> None:
        for name in ("input_size", "hidden_size", "output_size"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

    def init(self, key: Array) -> LSTMParams:
        """Initialize model parameters with Xavier-style uniform weights."""

        input_key, recurrent_key, output_key = jax.random.split(key, 3)
        gates_size = 4 * self.hidden_size

        return LSTMParams(
            input_kernel=xavier_uniform(input_key, (self.input_size, gates_size)),
            recurrent_kernel=xavier_uniform(
                recurrent_key, (self.hidden_size, gates_size)
            ),
            bias=jnp.zeros((gates_size,)),
            output_kernel=xavier_uniform(output_key, (self.hidden_size, self.output_size)),
            output_bias=jnp.zeros((self.output_size,)),
        )

    def initial_state(self) -> LSTMState:
        """Return a zero-valued recurrent state."""

        return LSTMState(
            hidden=jnp.zeros((self.hidden_size,)),
            cell=jnp.zeros((self.hidden_size,)),
        )

    def step(self, params: LSTMParams, state: LSTMState, x_t: Array) -> LSTMState:
        """Run one LSTM step."""

        gates = x_t @ params.input_kernel
        gates += state.hidden @ params.recurrent_kernel
        gates += params.bias

        input_gate, forget_gate, candidate, output_gate = jnp.split(gates, 4)
        input_gate = jax.nn.sigmoid(input_gate)
        forget_gate = jax.nn.sigmoid(forget_gate)
        candidate = jnp.tanh(candidate)
        output_gate = jax.nn.sigmoid(output_gate)

        cell = forget_gate * state.cell + input_gate * candidate
        hidden = output_gate * jnp.tanh(cell)
        return LSTMState(hidden=hidden, cell=cell)

    def apply(
        self, params: LSTMParams, inputs: Array, state: LSTMState | None = None
    ) -> tuple[Array, LSTMState]:
        """Evaluate one sequence.

        Returns a tuple of ``(outputs, final_state)`` where ``outputs`` has shape
        ``(sequence_length, output_size)``.
        """

        if state is None:
            state = self.initial_state()

        def scan_step(carry: LSTMState, x_t: Array) -> tuple[LSTMState, Array]:
            next_state = self.step(params, carry, x_t)
            output = next_state.hidden @ params.output_kernel + params.output_bias
            return next_state, output

        final_state, outputs = jax.lax.scan(scan_step, state, inputs)
        return outputs, final_state

    def apply_batch(
        self, params: LSTMParams, inputs: Array
    ) -> tuple[Array, LSTMState]:
        """Evaluate a batch of sequences.

        ``inputs`` must have shape ``(batch_size, sequence_length, input_size)``.
        The returned state stores hidden and cell arrays with shape
        ``(batch_size, hidden_size)``.
        """

        return jax.vmap(lambda sequence: self.apply(params, sequence))(inputs)


def xavier_uniform(key: Array, shape: tuple[int, ...]) -> Array:
    """Sample Xavier/Glorot uniform weights for a matrix-like shape."""

    if len(shape) < 2:
        raise ValueError("xavier_uniform requires at least two dimensions.")

    fan_in, fan_out = shape[-2], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
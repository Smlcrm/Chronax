"""Flax NNX modules for the GRU forecaster: encoder, decoder, full network."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def pytorch_uniform_init(hidden_size: int):
    """Match PyTorch nn.GRU / nn.Linear default: Uniform(-1/sqrt(H), 1/sqrt(H))."""
    bound = float(1.0 / jnp.sqrt(jnp.asarray(hidden_size, dtype=jnp.float32)))

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


class GRUEncoder(nnx.Module):
    """Stacked GRU encoder, scan over time, dropout BETWEEN layers (not after the last).

    All weights init via PyTorch's Uniform(-1/sqrt(H), 1/sqrt(H)) for accuracy
    parity. Note: Flax `nnx.GRUCell` fuses input and hidden biases into a single
    parameter on the input projection; PyTorch `nn.GRU` keeps them separate.
    Documented small parity gap — see `test_gru_encoder_documents_bias_parity_with_pytorch`.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        rngs: nnx.Rngs,
    ):
        init = pytorch_uniform_init(hidden_size)
        cells = []
        for i in range(n_layers):
            cell_in = in_features if i == 0 else hidden_size
            cells.append(
                nnx.GRUCell(
                    in_features=cell_in,
                    hidden_features=hidden_size,
                    kernel_init=init,
                    recurrent_kernel_init=init,
                    bias_init=init,
                    rngs=rngs,
                )
            )
        self.cells = cells
        self.hidden_size = hidden_size
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """x: [B, T, F] -> out: [B, T, H]. Float32 throughout."""
        x = x.astype(jnp.float32)
        batch, _time, _ = x.shape
        h = x
        for layer_idx, cell in enumerate(self.cells):
            carry = jnp.zeros((batch, self.hidden_size), dtype=jnp.float32)

            def body(c, x_t, _cell=cell):
                new_c, out = _cell(c, x_t)
                return new_c, out

            x_time_first = jnp.transpose(h, (1, 0, 2))
            _, outs = jax.lax.scan(body, carry, x_time_first)
            h = jnp.transpose(outs, (1, 0, 2))
            if layer_idx < len(self.cells) - 1:
                h = self.dropout(h, deterministic=deterministic)
        return h


class MLPDecoder(nnx.Module):
    """Linear -> ReLU -> Linear. Mirrors Nixtla `_modules.MLP(num_layers=2, dropout=0)`."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        rngs: nnx.Rngs,
    ):
        init = pytorch_uniform_init(max(in_features, hidden_size))
        self.in_layer = nnx.Linear(
            in_features, hidden_size, kernel_init=init, bias_init=init, rngs=rngs
        )
        self.out_layer = nnx.Linear(
            hidden_size, out_features, kernel_init=init, bias_init=init, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        return self.out_layer(nnx.relu(self.in_layer(x)))


class GRUNet(nnx.Module):
    """Encoder -> last `h` hidden states (with upsample path) -> MLPDecoder."""

    def __init__(
        self,
        *,
        in_features: int,
        encoder_hidden: int,
        encoder_layers: int,
        decoder_hidden: int,
        decoder_layers: int,
        dropout: float,
        h: int,
        input_size: int,
        rngs: nnx.Rngs,
    ):
        if decoder_layers != 2:
            raise NotImplementedError("v1 only supports decoder_layers=2.")
        self.encoder = GRUEncoder(
            in_features, encoder_hidden, encoder_layers, dropout, rngs=rngs
        )
        self.decoder = MLPDecoder(encoder_hidden, decoder_hidden, 1, rngs=rngs)
        self.h = h
        self.input_size = input_size
        if h > input_size:
            init = pytorch_uniform_init(input_size)
            self.upsample = nnx.Linear(
                input_size, h, kernel_init=init, bias_init=init, rngs=rngs
            )
        else:
            self.upsample = None

    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        hidden = self.encoder(x, deterministic=deterministic)
        if self.upsample is not None:
            hidden = jnp.transpose(hidden, (0, 2, 1))
            hidden = self.upsample(hidden)
            hidden = jnp.transpose(hidden, (0, 2, 1))
        else:
            hidden = hidden[:, -self.h :, :]
        return self.decoder(hidden, deterministic=deterministic)

"""Flax NNX modules for the GRU forecaster: encoder, decoder, full network."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _UniformInit:
    """Picklable callable that draws from Uniform(-bound, bound).

    Implemented as a class (rather than a closure) so module references to
    this initializer survive pickle.dumps — NNX `GRUCell`/`Linear` retain
    their `kernel_init` etc. as instance attributes, and a `<locals>.init`
    closure would otherwise raise `AttributeError` on serialization.
    """

    __slots__ = ("bound",)

    def __init__(self, bound: float) -> None:
        self.bound = bound

    def __call__(self, key, shape, dtype=jnp.float32):
        return jax.random.uniform(
            key, shape, dtype, minval=-self.bound, maxval=self.bound
        )


class _OrthogonalInit:
    """Picklable orthogonal initializer (Saxe et al. 2014).

    Returns a (semi-)orthogonal matrix scaled by ``scale``. For RNN/GRU
    recurrent kernels this stabilises gradients through time relative to
    a uniform init, especially for long input windows.
    """

    __slots__ = ("scale",)

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def __call__(self, key, shape, dtype=jnp.float32):
        if len(shape) < 2:
            raise ValueError(
                f"Orthogonal init requires shape with at least 2 dims; got {shape}."
            )
        n_rows = math.prod(shape[:-1])
        n_cols = shape[-1]
        matrix_shape = (n_rows, n_cols) if n_rows >= n_cols else (n_cols, n_rows)
        flat = jax.random.normal(key, matrix_shape, dtype)
        q, r = jnp.linalg.qr(flat)
        # Mezzadri (2007) sign correction: make Q deterministic across QR backends.
        q = q * jnp.sign(jnp.diag(r))[None, :]
        if n_rows < n_cols:
            q = q.T
        return (self.scale * q).reshape(shape).astype(dtype)


def uniform_one_over_sqrt_h_init(hidden_size: int) -> _UniformInit:
    """Initializer that draws from Uniform(-1/sqrt(H), 1/sqrt(H))."""
    bound = float(1.0 / jnp.sqrt(jnp.asarray(hidden_size, dtype=jnp.float32)))
    return _UniformInit(bound)


def _make_recurrent_init(name: str, hidden_size: int):
    """Resolve a recurrent-kernel initializer by name. Used by ``GRUEncoder``."""
    if name == "uniform":
        return uniform_one_over_sqrt_h_init(hidden_size)
    if name == "orthogonal":
        return _OrthogonalInit(scale=1.0)
    raise ValueError(
        f"Unknown recurrent_init {name!r}. Available: 'uniform', 'orthogonal'."
    )


RECURRENT_INITS = ("uniform", "orthogonal")


class GRUEncoder(nnx.Module):
    """Stacked GRU encoder, scan over time, dropout BETWEEN layers (not after the last).

    Input kernel and bias init from Uniform(-1/sqrt(H), 1/sqrt(H)). The
    recurrent kernel uses the initializer selected by ``recurrent_init``
    (``"uniform"`` or ``"orthogonal"``). Flax ``nnx.GRUCell`` fuses input
    and hidden biases into a single parameter on the input projection —
    pinned by ``test_gru_encoder_has_single_fused_bias_per_cell``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        rngs: nnx.Rngs,
        recurrent_init: str = "uniform",
    ):
        in_init = uniform_one_over_sqrt_h_init(hidden_size)
        rec_init = _make_recurrent_init(recurrent_init, hidden_size)
        cells = []
        for i in range(n_layers):
            cell_in = in_features if i == 0 else hidden_size
            cells.append(
                nnx.GRUCell(
                    in_features=cell_in,
                    hidden_features=hidden_size,
                    kernel_init=in_init,
                    recurrent_kernel_init=rec_init,
                    bias_init=in_init,
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
    """Linear -> ReLU -> Linear. Standard 2-layer MLP head."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        rngs: nnx.Rngs,
    ):
        init = uniform_one_over_sqrt_h_init(max(in_features, hidden_size))
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
        recurrent_init: str = "uniform",
    ):
        if decoder_layers != 2:
            raise NotImplementedError("Only decoder_layers=2 is supported.")
        self.encoder = GRUEncoder(
            in_features, encoder_hidden, encoder_layers, dropout, rngs=rngs,
            recurrent_init=recurrent_init,
        )
        self.decoder = MLPDecoder(encoder_hidden, decoder_hidden, 1, rngs=rngs)
        self.h = h
        self.input_size = input_size
        if h > input_size:
            init = uniform_one_over_sqrt_h_init(input_size)
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

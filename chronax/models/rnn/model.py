"""JAX/Flax port of Nixtla NeuralForecast's ``RNN`` model.

Mirrors the architecture and forward pass of the original PyTorch RNN
exactly, implemented in functional/Flax style:

* ``ElmanRNNCell``  — single Elman cell, parameter naming aligned with PyTorch
  (``ih`` / ``hh`` Dense layers so the four ``W_ih, W_hh, b_ih, b_hh`` tensors
  map cleanly to PyTorch's ``nn.RNN`` state_dict).
* ``RNNEncoder``    — stacked multi-layer encoder, unrolled with ``jax.lax.scan``.
* ``MLP``           — same shape and layer naming as Nixtla NeuralForecast's
  ``_modules.MLP``.
* ``RNN``           — full model: covariate concatenation, encoder, then either
  a recurrent ``proj`` head (autoregressive friendly) or an MLP decoder
  (direct multi-step forecasting, with optional sequence upsampling when
  ``h > input_size``).

Decoding strategies are *cleanly separated*:

* During training the model is called once per window with teacher-forcing
  inputs (the encoder consumes the historic targets directly).
* For inference ``autoregressive_predict`` performs single-step rollouts,
  threading the encoder's hidden state through ``rnn_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import flax.linen as fnn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RNNConfig:
    """Hyperparameters for :class:`RNN`.

    Attributes match Nixtla NeuralForecast's ``RNN`` 1:1; only architectural
    knobs are kept since training-loop knobs live in
    :mod:`chronax.models.rnn.train`.
    """

    h: int = 12
    input_size: int = 36
    encoder_hidden_size: int = 128
    encoder_n_layers: int = 2
    encoder_activation: str = "tanh"
    encoder_bias: bool = True
    encoder_dropout: float = 0.0
    decoder_hidden_size: int = 128
    decoder_layers: int = 2
    futr_exog_size: int = 0
    hist_exog_size: int = 0
    stat_exog_size: int = 0
    output_size: int = 1
    recurrent: bool = False


# ---------------------------------------------------------------------------
# MLP decoder (matches Nixtla NeuralForecast's _modules.MLP exactly)
# ---------------------------------------------------------------------------


class MLP(fnn.Module):
    """MLP head matching Nixtla NeuralForecast's ``_modules.MLP``.

    ``num_layers == 1`` collapses to a single ``Dense`` projection; otherwise
    we stack an input layer, ``num_layers - 2`` hidden layers (each
    ``Dense`` -> ``ReLU`` -> ``Dropout``) and a final unactivated ``Dense``.
    """

    out_features: int
    hidden_size: int
    num_layers: int
    dropout_rate: float = 0.0

    @fnn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if self.num_layers == 1:
            return fnn.Dense(features=self.out_features)(x)

        x = fnn.Dense(features=self.hidden_size)(x)
        x = fnn.relu(x)
        x = fnn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        for _ in range(self.num_layers - 2):
            x = fnn.Dense(features=self.hidden_size)(x)
            x = fnn.relu(x)
            x = fnn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        return fnn.Dense(features=self.out_features)(x)


# ---------------------------------------------------------------------------
# Single Elman RNN cell (matches torch.nn.RNNCell math)
# ---------------------------------------------------------------------------


class ElmanRNNCell(fnn.Module):
    """One layer of an Elman RNN.

    Computes ``h_t = act(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)`` exactly
    matching PyTorch's per-layer ``nn.RNNCell``. The ``ih`` and ``hh`` modules
    are deliberately separate so the corresponding kernels and biases can be
    transferred byte-for-byte from ``nn.RNN`` state dicts.

    The ``__call__`` signature is ``(carry, x) -> (new_carry, output)`` so the
    cell is directly compatible with :func:`flax.linen.scan` over the time
    axis. ``new_carry`` and ``output`` are both the new hidden state ``h_t``.
    """

    hidden_size: int
    activation: str = "tanh"
    use_bias: bool = True

    @fnn.compact
    def __call__(
        self, carry: jnp.ndarray, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        gate_x = fnn.Dense(self.hidden_size, use_bias=self.use_bias, name="ih")(x)
        gate_h = fnn.Dense(self.hidden_size, use_bias=self.use_bias, name="hh")(carry)
        pre = gate_x + gate_h
        if self.activation == "tanh":
            h_new = jnp.tanh(pre)
        elif self.activation == "relu":
            h_new = fnn.relu(pre)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return h_new, h_new


# ---------------------------------------------------------------------------
# Multi-layer RNN encoder (uses jax.lax.scan per layer)
# ---------------------------------------------------------------------------


class RNNEncoder(fnn.Module):
    """Stacked Elman encoder mirroring ``torch.nn.RNN(batch_first=True)``.

    Each layer is unrolled with ``flax.linen.scan`` (which wraps ``jax.lax.scan``
    while preserving correct module-parameter scoping under
    ``jit`` + ``grad`` + ``vmap``). Dropout is applied between layers but not
    after the final one, matching PyTorch.
    """

    hidden_size: int
    num_layers: int
    activation: str = "tanh"
    use_bias: bool = True
    dropout_rate: float = 0.0

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        initial_state: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Run the encoder over a [B, L, D] sequence.

        Returns ``(outputs, final_state)`` where ``outputs`` is the *last
        layer's* full hidden-state sequence ``[B, L, H]`` and ``final_state``
        stacks each layer's last hidden state into ``[num_layers, B, H]``.
        """
        B = x.shape[0]
        if initial_state is None:
            h_states = jnp.zeros((self.num_layers, B, self.hidden_size))
        else:
            h_states = initial_state

        # ``nn.scan`` lifts an ``nn.Module`` so that calling it inside scan
        # threads its parameters as broadcast (shared) variables across the
        # scan axis. ``in_axes=1`` / ``out_axes=1`` keeps the time axis at
        # position 1 so we don't need the transpose dance.
        ScanCell = fnn.scan(
            ElmanRNNCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )

        layer_input = x
        new_h: list[jnp.ndarray] = []
        for layer_idx in range(self.num_layers):
            cell = ScanCell(
                hidden_size=self.hidden_size,
                activation=self.activation,
                use_bias=self.use_bias,
                name=f"cell_{layer_idx}",
            )
            h0 = h_states[layer_idx]
            final_h, outputs = cell(h0, layer_input)
            layer_input = outputs

            if layer_idx < self.num_layers - 1 and self.dropout_rate > 0.0:
                layer_input = fnn.Dropout(
                    rate=self.dropout_rate, deterministic=deterministic
                )(layer_input)

            new_h.append(final_h)

        return layer_input, jnp.stack(new_h, axis=0)


# ---------------------------------------------------------------------------
# Full RNN model
# ---------------------------------------------------------------------------


class RNN(fnn.Module):
    """Nixtla NeuralForecast RNN ported to Flax.

    The forward pass mirrors the original PyTorch ``RNN.forward`` exactly:

    1. Concatenate ``insample_y``, ``hist_exog``, ``stat_exog`` (broadcast
       along time) and ``futr_exog[:, :L]`` along the feature axis.
    2. Run the multi-layer Elman encoder, optionally seeded by ``rnn_state``.
    3. If ``recurrent``: project every encoder timestep with a single ``Dense``
       head (suitable for autoregressive rollout).
       Else: select the last ``h`` timesteps (or upsample the time axis from
       ``input_size`` to ``h`` when ``h > input_size``), optionally append
       ``futr_exog[:, -h:]``, and pass through the MLP decoder.
    4. Slice the last ``h`` timesteps and return.
    """

    config: RNNConfig

    @fnn.compact
    def __call__(
        self,
        insample_y: jnp.ndarray,
        hist_exog: Optional[jnp.ndarray] = None,
        futr_exog: Optional[jnp.ndarray] = None,
        stat_exog: Optional[jnp.ndarray] = None,
        rnn_state: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        B, L = insample_y.shape[:2]

        encoder_input = insample_y
        if cfg.hist_exog_size > 0 and hist_exog is not None:
            encoder_input = jnp.concatenate([encoder_input, hist_exog], axis=2)
        if cfg.stat_exog_size > 0 and stat_exog is not None:
            stat_expanded = jnp.broadcast_to(
                stat_exog[:, None, :], (B, L, cfg.stat_exog_size)
            )
            encoder_input = jnp.concatenate([encoder_input, stat_expanded], axis=2)
        if cfg.futr_exog_size > 0 and futr_exog is not None:
            encoder_input = jnp.concatenate(
                [encoder_input, futr_exog[:, :L]], axis=2
            )

        encoder = RNNEncoder(
            hidden_size=cfg.encoder_hidden_size,
            num_layers=cfg.encoder_n_layers,
            activation=cfg.encoder_activation,
            use_bias=cfg.encoder_bias,
            dropout_rate=cfg.encoder_dropout,
            name="hist_encoder",
        )
        rnn_output, final_state = encoder(
            encoder_input, initial_state=rnn_state, deterministic=deterministic
        )

        if cfg.recurrent:
            output = fnn.Dense(features=cfg.output_size, name="proj")(rnn_output)
        else:
            hidden = rnn_output
            if cfg.h > cfg.input_size:
                hidden = jnp.transpose(hidden, (0, 2, 1))
                hidden = fnn.Dense(features=cfg.h, name="upsample_sequence")(hidden)
                hidden = jnp.transpose(hidden, (0, 2, 1))
            else:
                hidden = hidden[:, -cfg.h:]

            if cfg.futr_exog_size > 0 and futr_exog is not None:
                hidden = jnp.concatenate(
                    [hidden, futr_exog[:, -cfg.h:]], axis=-1
                )

            output = MLP(
                out_features=cfg.output_size,
                hidden_size=cfg.decoder_hidden_size,
                num_layers=cfg.decoder_layers,
                dropout_rate=0.0,
                name="mlp_decoder",
            )(hidden, deterministic=deterministic)

        return output[:, -cfg.h:], final_state


# ---------------------------------------------------------------------------
# Autoregressive inference helper (recurrent mode only)
# ---------------------------------------------------------------------------


def autoregressive_predict(
    model: RNN,
    params,
    insample_y: jnp.ndarray,
    h: int,
    hist_exog: Optional[jnp.ndarray] = None,
    futr_exog: Optional[jnp.ndarray] = None,
    stat_exog: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Multi-step autoregressive rollout for ``recurrent=True`` models.

    Strategy:

    * Step 0 — feed the entire history; keep the encoder's *last* hidden
      state and *last* projected prediction.
    * Steps 1..h-1 — feed the previous prediction (and the appropriate
      single ``futr_exog`` slice) one step at a time, threading the encoder
      hidden state through ``rnn_state``.

    ``hist_exog`` is intentionally only used for the history pass (it is, by
    definition, only available in the past). ``stat_exog`` is constant and
    re-used at every step.

    Args:
        model: an instance of :class:`RNN` with ``config.recurrent=True``.
        params: parameters from ``model.init`` (or transferred weights).
        insample_y: ``[B, L, 1]`` historic targets.
        h: number of forecast steps.
        hist_exog: ``[B, L, X]`` historic exogenous, or None.
        futr_exog: ``[B, L+H, F]`` exogenous covering history *and* horizon,
            or None.
        stat_exog: ``[B, S]`` static exogenous, or None.

    Returns:
        ``[B, h, output_size]`` predictions.
    """
    if not model.config.recurrent:
        raise ValueError(
            "autoregressive_predict requires model.config.recurrent=True"
        )

    L = insample_y.shape[1]
    futr_hist = futr_exog[:, :L] if futr_exog is not None else None

    output, rnn_state = model.apply(
        params,
        insample_y,
        hist_exog=hist_exog,
        futr_exog=futr_hist,
        stat_exog=stat_exog,
        rnn_state=None,
        deterministic=True,
    )
    last_pred = output[:, -1:, :]  # [B, 1, output_size]
    preds = [last_pred]

    next_input = last_pred[:, :, :1]  # feed back the first output channel
    for t in range(1, h):
        if futr_exog is not None:
            futr_t = futr_exog[:, L + t - 1: L + t, :]
        else:
            futr_t = None
        out_t, rnn_state = model.apply(
            params,
            next_input,
            hist_exog=None,
            futr_exog=futr_t,
            stat_exog=stat_exog,
            rnn_state=rnn_state,
            deterministic=True,
        )
        last_pred = out_t[:, -1:, :]
        preds.append(last_pred)
        next_input = last_pred[:, :, :1]

    return jnp.concatenate(preds, axis=1)

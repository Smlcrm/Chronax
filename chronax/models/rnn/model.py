"""JAX/Flax RNN — improved for speed and accuracy.

Key improvements over v1:
* GRU cell option (``cell_type='gru'``, default) — gating prevents vanishing
  gradients; meaningfully more accurate than vanilla Elman RNN on most
  time-series benchmarks.
* Layer normalisation after each encoder layer (``layer_norm=True``, default)
  — stabilises training, allows higher learning rates, reduces sensitivity to
  initialisation.
* ``autoregressive_predict`` rewritten with ``jax.lax.scan`` — the h-step
  rollout is compiled as one XLA op instead of h separate Python-level
  ``model.apply`` dispatches, giving a significant speedup on CPU and an even
  larger one on GPU.
* ``RNNConfig.recurrent`` defaults to ``False`` — direct MLP decoding avoids
  compounding prediction errors and is faster; better for short horizons.

Architecture unchanged for ``cell_type='elman'`` / ``layer_norm=False`` so
existing weight transfers remain valid.
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

    New fields vs v1:
        cell_type:  ``"gru"`` (default) or ``"elman"``.  GRU gating gives
                    meaningfully better accuracy; use ``"elman"`` only when
                    you need exact weight parity with a vanilla RNN checkpoint.
        layer_norm: apply LayerNorm after each encoder layer (default True).
                    Stabilises training especially with larger hidden sizes.
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
    cell_type: str = "gru"
    layer_norm: bool = True


# ---------------------------------------------------------------------------
# MLP decoder (matches Nixtla NeuralForecast's _modules.MLP exactly)
# ---------------------------------------------------------------------------


class MLP(fnn.Module):
    """MLP head matching Nixtla NeuralForecast's ``_modules.MLP``."""

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
# Elman RNN cell (unchanged; kept for weight-parity with legacy checkpoints)
# ---------------------------------------------------------------------------


class ElmanRNNCell(fnn.Module):
    """One layer of an Elman RNN.

    Computes ``h_t = act(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)``.

    ``(carry, x) -> (new_carry, output)`` so the cell is directly compatible
    with :func:`flax.linen.scan` over the time axis.
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
# GRU cell — drop-in replacement for ElmanRNNCell
# ---------------------------------------------------------------------------


class GRUCell(fnn.Module):
    """One layer of a Gated Recurrent Unit.

    Equations (PyTorch convention)::

        r_t = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
        z_t = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
        n_t = tanh(W_in x + b_in + r_t ⊙ (W_hn h + b_hn))
        h_t = (1 − z_t) ⊙ n_t + z_t ⊙ h

    Same ``(carry, x) -> (new_carry, output)`` interface as
    :class:`ElmanRNNCell` — drop-in for :func:`flax.linen.scan`.
    """

    hidden_size: int
    use_bias: bool = True

    @fnn.compact
    def __call__(
        self, carry: jnp.ndarray, x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Fused input projection for r, z, n gates
        gates_x = fnn.Dense(3 * self.hidden_size, use_bias=self.use_bias, name="gate_x")(x)
        # Hidden projection for reset + update gates (r, z share one Dense)
        ru_h = fnn.Dense(2 * self.hidden_size, use_bias=self.use_bias, name="ru_h")(carry)
        # Separate hidden projection for new gate (needed so reset gate applies)
        n_h = fnn.Dense(self.hidden_size, use_bias=self.use_bias, name="n_h")(carry)

        r_x, z_x, n_x = jnp.split(gates_x, 3, axis=-1)
        r_h, z_h = jnp.split(ru_h, 2, axis=-1)

        r = jax.nn.sigmoid(r_x + r_h)
        z = jax.nn.sigmoid(z_x + z_h)
        n = jnp.tanh(n_x + r * n_h)
        h_new = (1.0 - z) * n + z * carry
        return h_new, h_new


# ---------------------------------------------------------------------------
# Multi-layer RNN encoder
# ---------------------------------------------------------------------------


class RNNEncoder(fnn.Module):
    """Stacked Elman/GRU encoder.

    Each layer is unrolled with ``flax.linen.scan``. Optional LayerNorm and
    dropout are applied between layers (not after the final layer, matching
    PyTorch convention).

    Args:
        cell_type: ``"gru"`` (default) or ``"elman"``.
        layer_norm: apply LayerNorm after each layer's output sequence.
    """

    hidden_size: int
    num_layers: int
    activation: str = "tanh"
    use_bias: bool = True
    dropout_rate: float = 0.0
    cell_type: str = "gru"
    layer_norm: bool = True

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        initial_state: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Run the encoder over a ``[B, L, D]`` sequence.

        Returns ``(outputs, final_state)`` where ``outputs`` is the last
        layer's full hidden-state sequence ``[B, L, H]`` and ``final_state``
        stacks each layer's last hidden state into ``[num_layers, B, H]``.
        """
        B = x.shape[0]
        if initial_state is None:
            h_states = jnp.zeros((self.num_layers, B, self.hidden_size))
        else:
            h_states = initial_state

        CellClass = GRUCell if self.cell_type == "gru" else ElmanRNNCell

        layer_input = x
        new_h: list[jnp.ndarray] = []

        for layer_idx in range(self.num_layers):
            if self.cell_type == "gru":
                cell_kwargs = dict(
                    hidden_size=self.hidden_size,
                    use_bias=self.use_bias,
                )
            else:
                cell_kwargs = dict(
                    hidden_size=self.hidden_size,
                    activation=self.activation,
                    use_bias=self.use_bias,
                )

            ScanCell = fnn.scan(
                CellClass,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1,
            )
            cell = ScanCell(**cell_kwargs, name=f"cell_{layer_idx}")
            h0 = h_states[layer_idx]
            final_h, outputs = cell(h0, layer_input)
            layer_input = outputs

            # LayerNorm + Dropout between layers (not after the last layer)
            if layer_idx < self.num_layers - 1:
                if self.layer_norm:
                    layer_input = fnn.LayerNorm(name=f"ln_{layer_idx}")(layer_input)
                if self.dropout_rate > 0.0:
                    layer_input = fnn.Dropout(
                        rate=self.dropout_rate, deterministic=deterministic
                    )(layer_input)

            new_h.append(final_h)

        return layer_input, jnp.stack(new_h, axis=0)


# ---------------------------------------------------------------------------
# Full RNN model
# ---------------------------------------------------------------------------


class RNN(fnn.Module):
    """Nixtla NeuralForecast RNN ported to Flax, with GRU and LayerNorm.

    Forward pass is unchanged vs v1; new config fields (``cell_type``,
    ``layer_norm``) are threaded through to :class:`RNNEncoder`.
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
            cell_type=cfg.cell_type,
            layer_norm=cfg.layer_norm,
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

    The h-step rollout is implemented with ``jax.lax.scan`` so the entire
    loop is compiled as a single XLA op — significantly faster than the
    previous Python ``for`` loop which dispatched h separate ``model.apply``
    calls.

    Strategy:
    * History pass — feed the full history; capture encoder hidden state and
      last projected prediction.
    * Steps 1..h-1 — scan over the horizon, feeding the previous prediction
      (+ any ``futr_exog`` slice) one step at a time while threading the
      encoder hidden state.

    Args:
        model: :class:`RNN` instance with ``config.recurrent=True``.
        params: parameters from ``model.init`` or weight transfer.
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

    # ---- history pass -------------------------------------------------------
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
    first_pred = output[:, -1:, :]  # [B, 1, output_size]

    if h == 1:
        return first_pred

    # ---- horizon rollout via lax.scan ----------------------------------------
    # Python-level branch: determines XLA program shape at trace time.
    has_futr = futr_exog is not None

    if has_futr:
        # futr_exog[:, L : L+h-1] covers steps 1..h-1 → shape [B, h-1, F]
        # Transpose to [h-1, B, F] for scan to iterate over the leading axis.
        futr_scan = jnp.transpose(futr_exog[:, L : L + h - 1], (1, 0, 2))
    else:
        futr_scan = jnp.zeros(h - 1)  # dummy scalar per step; never used

    def scan_step(carry, futr_t):
        next_input, state = carry
        f_exog = futr_t[:, None, :] if has_futr else None  # [B, 1, F] or None
        out_t, new_state = model.apply(
            params,
            next_input,
            futr_exog=f_exog,
            stat_exog=stat_exog,
            rnn_state=state,
            deterministic=True,
        )
        last = out_t[:, -1:, :]  # [B, 1, output_size]
        return (last[..., :1], new_state), last

    init_carry = (first_pred[..., :1], rnn_state)
    _, rest_preds = jax.lax.scan(scan_step, init_carry, futr_scan)
    # rest_preds: [h-1, B, 1, output_size] → rearrange to [B, h-1, output_size]
    rest_preds = jnp.transpose(rest_preds, (1, 0, 2, 3))[:, :, 0, :]

    return jnp.concatenate([first_pred, rest_preds], axis=1)

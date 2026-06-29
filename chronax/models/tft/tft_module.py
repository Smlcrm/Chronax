"""TFT encoders, fusion decoder, and the top-level network (flax.nnx).

Ports ``StaticCovariateEncoder``, ``TemporalCovariateEncoder`` (LSTM/GRU
seq2seq), ``TemporalFusionDecoder``, and the top-level ``TFTNet`` from
``neuralforecast/models/tft.py``. The recurrence runs as ``nnx.scan`` over an
``nnx.LSTMCell``/``GRUCell`` with the initial carry supplied by the static
covariate encoder -- carry order is ``(c, h)`` for LSTM (verified). All paths
are vmap/scan-pure. ``float32``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from chronax.models.tft.tft_layers import (
    GLU,
    GRN,
    ContinuousEmbedding,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
    _torch_linear,
)


def _run_rnn(cell, x_seq: jnp.ndarray, init_carry):
    """Scan ``cell`` over the time axis (axis 1) of ``x_seq`` from ``init_carry``.

    Returns ``(outputs[B, T, hidden], final_carry)``. Carry-type agnostic: LSTM
    carries are ``(c, h)`` tuples, GRU carries are single arrays.
    """

    @nnx.scan(in_axes=(nnx.Carry, 1), out_axes=(nnx.Carry, 1))
    def step(carry, x_t):
        return cell(carry, x_t)

    final_carry, ys = step(init_carry, x_seq)
    return ys, final_carry


def _run_stack(cells, x_seq: jnp.ndarray, init_carries):
    """Run a stack of recurrent ``cells`` layer by layer. Returns (outputs, final_carries)."""
    h = x_seq
    finals = []
    for cell, carry in zip(cells, init_carries):
        h, final = _run_rnn(cell, h, carry)
        finals.append(final)
    return h, finals


class StaticCovariateEncoder(nnx.Module):
    """Encodes static covariates into the four TFT context vectors.

    A VSN over the static variables yields a context, which separate GRNs map to
    ``cs`` (variable-selection context), ``ce`` (enrichment context), and the
    per-layer LSTM initial hidden/cell states ``ch``/``cc`` (``cc == ch`` for GRU).
    """

    def __init__(self, hidden_size, num_static, dropout, activation,
                 rnn_type="lstm", n_rnn_layers=1, *, rngs: nnx.Rngs):
        self.rnn_type = rnn_type
        self.n_rnn_layers = n_rnn_layers
        self.vsn = VariableSelectionNetwork(
            hidden_size, num_static, dropout=dropout, activation=activation, rngs=rngs,
        )
        n_states = 2 if rnn_type == "lstm" else 1
        self.n_context = 2 + n_states * n_rnn_layers
        self.context_grns = [
            GRN(hidden_size, hidden_size, dropout=dropout, activation=activation, rngs=rngs)
            for _ in range(self.n_context)
        ]

    def __call__(self, s: jnp.ndarray, deterministic: bool = True):
        var_ctx, weights = self.vsn(s, deterministic=deterministic)          # [B, hidden]
        cs = self.context_grns[0](var_ctx, deterministic=deterministic)
        ce = self.context_grns[1](var_ctx, deterministic=deterministic)
        L = self.n_rnn_layers
        ch = jnp.stack(
            [self.context_grns[2 + i](var_ctx, deterministic=deterministic) for i in range(L)], axis=0,
        )                                                                    # [L, B, hidden]
        if self.rnn_type == "lstm":
            cc = jnp.stack(
                [self.context_grns[2 + L + i](var_ctx, deterministic=deterministic) for i in range(L)], axis=0,
            )
        else:
            cc = ch
        return cs, ce, ch, cc, weights


class TemporalCovariateEncoder(nnx.Module):
    """Locality-enhancement seq2seq: VSN-gated history/future + LSTM/GRU encoder-decoder.

    History and future variables are variable-selected (with static context
    ``cs``), the encoder recurrence runs over history from the static initial
    carry ``(cc, ch)``, the decoder recurrence runs over the future continuing
    from the encoder's final carry, and the concatenated outputs are GLU-gated
    with a residual + LayerNorm (NF ``input_gate``).
    """

    def __init__(self, hidden_size, num_hist_vars, num_futr_vars, dropout, activation,
                 rnn_type="lstm", n_rnn_layers=1, *, rngs: nnx.Rngs):
        self.rnn_type = rnn_type
        self.n_rnn_layers = n_rnn_layers
        self.history_vsn = VariableSelectionNetwork(
            hidden_size, num_hist_vars, dropout=dropout, context_size=hidden_size,
            activation=activation, rngs=rngs,
        )
        self.future_vsn = VariableSelectionNetwork(
            hidden_size, num_futr_vars, dropout=dropout, context_size=hidden_size,
            activation=activation, rngs=rngs,
        )
        Cell = nnx.LSTMCell if rnn_type == "lstm" else nnx.GRUCell
        self.encoder_cells = [
            Cell(in_features=hidden_size, hidden_features=hidden_size, rngs=rngs)
            for _ in range(n_rnn_layers)
        ]
        self.decoder_cells = [
            Cell(in_features=hidden_size, hidden_features=hidden_size, rngs=rngs)
            for _ in range(n_rnn_layers)
        ]
        self.input_gate = GLU(hidden_size, hidden_size, rngs=rngs)
        self.input_gate_ln = nnx.LayerNorm(hidden_size, epsilon=1e-3, rngs=rngs)

    def _init_carries(self, ch, cc):
        if self.rnn_type == "lstm":
            return [(cc[i], ch[i]) for i in range(self.n_rnn_layers)]   # nnx LSTM carry = (c, h)
        return [ch[i] for i in range(self.n_rnn_layers)]                 # GRU carry = h

    def __call__(self, hist, futr, cs, ch, cc, deterministic=True):
        hist_feat, hist_w = self.history_vsn(hist, context=cs, deterministic=deterministic)  # [B,L,hidden]
        futr_feat, futr_w = self.future_vsn(futr, context=cs, deterministic=deterministic)   # [B,h,hidden]
        init = self._init_carries(ch, cc)
        enc_out, enc_finals = _run_stack(self.encoder_cells, hist_feat, init)
        dec_out, _ = _run_stack(self.decoder_cells, futr_feat, enc_finals)
        feat = jnp.concatenate([hist_feat, futr_feat], axis=1)           # [B, L+h, hidden] (residual)
        temporal = jnp.concatenate([enc_out, dec_out], axis=1)           # [B, L+h, hidden]
        temporal = self.input_gate_ln(self.input_gate(temporal) + feat)
        return temporal, hist_w, futr_w


class TemporalFusionDecoder(nnx.Module):
    """Static enrichment -> masked interpretable attention -> position-wise GRN.

    Each block is gated (GLU) with a residual + LayerNorm. After attention the
    encoder steps are discarded, keeping only the ``h`` horizon steps (NF slices
    ``[input_size:]``).
    """

    def __init__(self, n_head, hidden_size, dropout, attn_dropout, activation, *, rngs: nnx.Rngs):
        self.enrichment_grn = GRN(
            hidden_size, hidden_size, context_size=hidden_size, dropout=dropout,
            activation=activation, rngs=rngs,
        )
        self.attention = InterpretableMultiHeadAttention(
            n_head, hidden_size, attn_dropout=attn_dropout, rngs=rngs,
        )
        self.attention_gate = GLU(hidden_size, hidden_size, rngs=rngs)
        self.attention_ln = nnx.LayerNorm(hidden_size, epsilon=1e-3, rngs=rngs)
        self.positionwise_grn = GRN(
            hidden_size, hidden_size, dropout=dropout, activation=activation, rngs=rngs,
        )
        self.decoder_gate = GLU(hidden_size, hidden_size, rngs=rngs)
        self.decoder_ln = nnx.LayerNorm(hidden_size, epsilon=1e-3, rngs=rngs)

    def __call__(self, temporal, ce, input_size, deterministic=True):
        enriched = self.enrichment_grn(temporal, ce, deterministic=deterministic)
        attn_out, attn_w = self.attention(enriched, deterministic=deterministic)
        x = attn_out[:, input_size:, :]
        enriched = enriched[:, input_size:, :]
        temporal = temporal[:, input_size:, :]
        x = self.attention_ln(self.attention_gate(x) + enriched)
        x = self.positionwise_grn(x, deterministic=deterministic)
        x = self.decoder_ln(self.decoder_gate(x) + temporal)
        return x, attn_w


class TFTNet(nnx.Module):
    """Full TFT backbone: embed -> static contexts -> seq2seq -> fusion decoder -> head.

    I/O mirrors NF ``TFT.forward`` but with explicit kwargs instead of a
    ``windows_batch`` dict. ``__call__`` returns ``[B, h, outputsize_multiplier]``.
    """

    def __init__(self, *, h, input_size, hidden_size=128, n_head=4, attn_dropout=0.0,
                 dropout=0.1, grn_activation="ELU", rnn_type="lstm", n_rnn_layers=1,
                 stat_exog_size=0, hist_exog_size=0, futr_exog_size=0, tgt_size=1,
                 outputsize_multiplier=1, rngs: nnx.Rngs):
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stat_exog_size = stat_exog_size
        self.hist_exog_size = hist_exog_size
        self.futr_exog_size = futr_exog_size
        self.tgt_size = tgt_size
        self.outputsize_multiplier = outputsize_multiplier
        self.rnn_type = rnn_type
        self.n_rnn_layers = n_rnn_layers

        self.tgt_embedding = ContinuousEmbedding(tgt_size, hidden_size, rngs=rngs)
        self.hist_embedding = ContinuousEmbedding(hist_exog_size, hidden_size, rngs=rngs)
        self.futr_embedding = ContinuousEmbedding(max(futr_exog_size, 1), hidden_size, rngs=rngs)
        self.stat_embedding = ContinuousEmbedding(stat_exog_size, hidden_size, rngs=rngs)

        num_hist_vars = hist_exog_size + max(futr_exog_size, 1) + tgt_size
        num_futr_vars = max(futr_exog_size, 1)
        self.has_static = stat_exog_size > 0
        if self.has_static:
            self.static_encoder = StaticCovariateEncoder(
                hidden_size, stat_exog_size, dropout, grn_activation, rnn_type, n_rnn_layers, rngs=rngs,
            )
        self.temporal_encoder = TemporalCovariateEncoder(
            hidden_size, num_hist_vars, num_futr_vars, dropout, grn_activation,
            rnn_type, n_rnn_layers, rngs=rngs,
        )
        self.temporal_fusion_decoder = TemporalFusionDecoder(
            n_head, hidden_size, dropout, attn_dropout, grn_activation, rngs=rngs,
        )
        self.output_adapter = _torch_linear(hidden_size, outputsize_multiplier, rngs=rngs)

    def __call__(self, insample_y, hist_exog=None, futr_exog=None, stat_exog=None, deterministic=True):
        # Enforce float32 throughout (repo convention). Casting at the boundary keeps
        # the LSTM carry dtype consistent even if an import has enabled x64 globally.
        insample_y = insample_y.astype(jnp.float32)
        hist_exog = None if hist_exog is None else hist_exog.astype(jnp.float32)
        futr_exog = None if futr_exog is None else futr_exog.astype(jnp.float32)
        stat_exog = None if stat_exog is None else stat_exog.astype(jnp.float32)
        B = insample_y.shape[0]
        L = self.input_size

        if self.futr_exog_size == 0:                            # NF: repeat last value over L+h
            futr = jnp.broadcast_to(insample_y[:, -1:, :], (B, L + self.h, 1))
        else:
            futr = futr_exog

        tgt_emb = self.tgt_embedding(insample_y)                # [B, L, 1, hidden]
        futr_emb = self.futr_embedding(futr)                    # [B, L+h, max(F,1), hidden]

        if self.has_static:
            s_emb = self.stat_embedding(stat_exog)              # [B, S, hidden]
            cs, ce, ch, cc, _ = self.static_encoder(s_emb, deterministic=deterministic)
        else:
            cs = jnp.zeros((B, self.hidden_size), jnp.float32)
            ce = jnp.zeros((B, self.hidden_size), jnp.float32)
            ch = jnp.zeros((self.n_rnn_layers, B, self.hidden_size), jnp.float32)
            cc = jnp.zeros((self.n_rnn_layers, B, self.hidden_size), jnp.float32)

        parts = []                                              # NF order: [hist, futr, target]
        if self.hist_exog_size > 0:
            parts.append(self.hist_embedding(hist_exog)[:, :L])
        parts.append(futr_emb[:, :L])
        parts.append(tgt_emb[:, :L])
        historical = jnp.concatenate(parts, axis=-2)            # [B, L, num_hist_vars, hidden]
        future = futr_emb[:, L:]                                # [B, h, num_futr_vars, hidden]

        temporal, _, _ = self.temporal_encoder(historical, future, cs, ch, cc, deterministic=deterministic)
        decoded, _ = self.temporal_fusion_decoder(temporal, ce, L, deterministic=deterministic)
        return self.output_adapter(decoded)                     # [B, h, outputsize_multiplier]

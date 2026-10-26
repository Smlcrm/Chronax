"""JAX/Flax TiDE model.

Faithful port of Nixtla NeuralForecast's ``TiDE`` (Time-series Dense Encoder).
Pure JAX/Flax/Optax — no PyTorch, no scipy.

Architecture:

    x        ← insample_y                               # [B, L]
    x_skip   ← Dense(h * output_size)(x)               # [B, h, output_size]

    # optional covariate projections (MLPResidual, acts on last dim)
    if hist_exog:  x ← cat([x, flatten(proj(hist_exog))], axis=1)
    if futr_exog:  x_futr ← proj(futr_exog)
                   x ← cat([x, flatten(x_futr)], axis=1)
    if stat_exog:  x ← cat([x, stat_exog], axis=1)

    x ← DenseEncoder stack(x)           # [B, hidden_size]
    x ← DenseDecoder stack(x)           # [B, decoder_output_dim * h]
    x ← reshape(x, [B, h, decoder_output_dim])

    if futr_exog:  x ← cat([x, x_futr[:, L:]], axis=2)
    x ← TemporalDecoder(x)              # [B, h, output_size]
    return x + x_skip

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as fnn
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TiDEConfig:
    """Hyperparameters for :class:`TiDE`.

    Attributes:
        h: Forecast horizon.
        input_size: History window length ``L``.
        hidden_size: MLP hidden width for encoder/decoder blocks.
        decoder_output_dim: Per-step output dimension of the dense decoder.
        temporal_decoder_dim: Hidden size of the temporal decoder MLP block.
        dropout: Dropout rate (0 = disabled).
        layernorm: Insert LayerNorm after each MLPResidual block output.
        num_encoder_layers: Number of stacked MLPResidual encoder layers.
        num_decoder_layers: Number of stacked MLPResidual decoder layers.
        temporal_width: Projected feature dimension for temporal covariates.
        futr_exog_size: Number of future exogenous features ``F``.
        hist_exog_size: Number of historic exogenous features ``X``.
        stat_exog_size: Number of static exogenous features ``S``.
        output_size: Outputs per step — 1 for point forecasts.
    """

    h: int = 24
    input_size: int = 48
    hidden_size: int = 512
    decoder_output_dim: int = 32
    temporal_decoder_dim: int = 128
    dropout: float = 0.3
    layernorm: bool = True
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    temporal_width: int = 4
    futr_exog_size: int = 0
    hist_exog_size: int = 0
    stat_exog_size: int = 0
    output_size: int = 1


# ---------------------------------------------------------------------------
# MLPResidual block
# ---------------------------------------------------------------------------


class MLPResidual(fnn.Module):
    """MLP with skip connection and optional LayerNorm.

    Mirrors NeuralForecast's ``MLPResidual`` exactly::

        h = relu(Dense(hidden_size)(x))
        h = Dense(output_dim)(h)
        h = Dropout(h)
        out = h + Dense(output_dim)(x)   # skip
        if use_layernorm: out = LayerNorm(out)

    Operates on the last axis, so both ``[B, D]`` and ``[B, T, D]`` inputs
    are handled correctly without reshaping.
    """

    hidden_size: int
    output_dim: int
    dropout_rate: float = 0.0
    use_layernorm: bool = True

    @fnn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        h = fnn.Dense(self.hidden_size, name="lin1")(x)
        h = fnn.relu(h)
        h = fnn.Dense(self.output_dim, name="lin2")(h)
        if self.dropout_rate > 0.0:
            h = fnn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(h)
        skip = fnn.Dense(self.output_dim, name="skip")(x)
        out = h + skip
        if self.use_layernorm:
            out = fnn.LayerNorm(name="norm")(out)
        return out


# ---------------------------------------------------------------------------
# Full TiDE model
# ---------------------------------------------------------------------------


class TiDE(fnn.Module):
    """Time-series Dense Encoder (TiDE).

    Input ``insample_y`` is expected as ``[B, L, 1]`` (consistent with the
    rest of Chronax). The trailing feature dim is squeezed inside the forward
    pass before concatenating covariates and running the encoder stack.

    Args:
        config: :class:`TiDEConfig` specifying the full architecture.
    """

    config: TiDEConfig

    @fnn.compact
    def __call__(
        self,
        insample_y: jnp.ndarray,                        # [B, L, 1]
        hist_exog: Optional[jnp.ndarray] = None,        # [B, L, X]
        futr_exog: Optional[jnp.ndarray] = None,        # [B, L+h, F]
        stat_exog: Optional[jnp.ndarray] = None,        # [B, S]
        deterministic: bool = True,
    ) -> jnp.ndarray:                                    # [B, h, output_size]
        cfg = self.config
        B = insample_y.shape[0]
        L = insample_y.shape[1]

        # Squeeze trailing feature dim: [B, L, 1] → [B, L]
        x = insample_y[..., 0]

        # Global skip connection: [B, L] → [B, h, output_size]
        x_skip = fnn.Dense(cfg.h * cfg.output_size, name="global_skip")(x)
        x_skip = x_skip.reshape(B, cfg.h, cfg.output_size)

        # Historic exogenous projection: [B, L, X] → [B, L * temporal_width]
        x_futr_proj = None
        if cfg.hist_exog_size > 0:
            if hist_exog is None:
                hist_exog = jnp.zeros((B, L, cfg.hist_exog_size))
            x_hist_proj = MLPResidual(
                hidden_size=cfg.hidden_size,
                output_dim=cfg.temporal_width,
                dropout_rate=cfg.dropout,
                use_layernorm=cfg.layernorm,
                name="hist_exog_projection",
            )(hist_exog, deterministic)                  # [B, L, temporal_width]
            x = jnp.concatenate([x, x_hist_proj.reshape(B, -1)], axis=1)

        # Future exogenous projection: [B, L+h, F] → [B, (L+h) * temporal_width]
        if cfg.futr_exog_size > 0:
            if futr_exog is None:
                futr_exog = jnp.zeros((B, L + cfg.h, cfg.futr_exog_size))
            x_futr_proj = MLPResidual(
                hidden_size=cfg.hidden_size,
                output_dim=cfg.temporal_width,
                dropout_rate=cfg.dropout,
                use_layernorm=cfg.layernorm,
                name="futr_exog_projection",
            )(futr_exog, deterministic)                  # [B, L+h, temporal_width]
            x = jnp.concatenate([x, x_futr_proj.reshape(B, -1)], axis=1)

        # Static exogenous: concat [B, S] to [B, *]
        if cfg.stat_exog_size > 0:
            if stat_exog is None:
                stat_exog = jnp.zeros((B, cfg.stat_exog_size))
            x = jnp.concatenate([x, stat_exog], axis=1)

        # Dense encoder: [B, encoder_input_dim] → [B, hidden_size]
        for i in range(cfg.num_encoder_layers):
            x = MLPResidual(
                hidden_size=cfg.hidden_size,
                output_dim=cfg.hidden_size,
                dropout_rate=cfg.dropout,
                use_layernorm=cfg.layernorm,
                name=f"encoder_{i}",
            )(x, deterministic)

        # Dense decoder: [B, hidden_size] → [B, decoder_output_dim * h]
        decoder_out_size = cfg.decoder_output_dim * cfg.h
        for i in range(cfg.num_decoder_layers):
            out_dim = decoder_out_size if i == cfg.num_decoder_layers - 1 else cfg.hidden_size
            x = MLPResidual(
                hidden_size=cfg.hidden_size,
                output_dim=out_dim,
                dropout_rate=cfg.dropout,
                use_layernorm=cfg.layernorm,
                name=f"decoder_{i}",
            )(x, deterministic)

        # Per-step features: [B, decoder_output_dim * h] → [B, h, decoder_output_dim]
        x = x.reshape(B, cfg.h, cfg.decoder_output_dim)

        # Append horizon slice of future projection: [B, h, decoder_output_dim + temporal_width]
        if cfg.futr_exog_size > 0 and x_futr_proj is not None:
            x_futr_h = x_futr_proj[:, L:]               # [B, h, temporal_width]
            x = jnp.concatenate([x, x_futr_h], axis=2)

        # Temporal decoder: [B, h, *] → [B, h, output_size]
        x = MLPResidual(
            hidden_size=cfg.temporal_decoder_dim,
            output_dim=cfg.output_size,
            dropout_rate=cfg.dropout,
            use_layernorm=cfg.layernorm,
            name="temporal_decoder",
        )(x, deterministic)

        return x + x_skip                               # [B, h, output_size]

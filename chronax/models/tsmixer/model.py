"""JAX/Flax TSMixer — multivariate time-series forecasting.

Faithful port of Nixtla NeuralForecast's TSMixer to JAX/Flax/Optax.

Normalisation:
  * TemporalMixing: ``fnn.BatchNorm(axis=1)`` on ``[B, N, L]`` — normalises
    over the B×L axes per N channel, exactly matching PyTorch's
    ``BatchNorm1d(n_series)`` with running_mean/running_var accumulated over
    all training steps for stable inference statistics.
  * FeatureMixing: ``fnn.BatchNorm(axis=-1)`` on ``[B, L, N]`` — normalises
    over B×L per N channel (same semantics on the feature axis).
  * RevIN: optional per-channel instance normalisation computed inline
    (no mutable state — pure functional).

All shapes follow ``[B, L, N]`` (batch, sequence, n_series) throughout.
TSMixer returns a plain ``[B, h, N]`` array (no hidden-state tuple).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as fnn
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TSMixerConfig:
    """Hyperparameters for :class:`TSMixer`.

    Args:
        h: forecast horizon.
        input_size: history window length (autoregressive lags).
        n_series: number of time series (channels).
        n_block: number of stacked MixingLayers.
        ff_dim: hidden size of the feature-axis feed-forward network.
        dropout: dropout rate applied after each mixing sub-block.
        revin: if True, apply per-channel RevIN before mixing and invert
            after the output projection.
        revin_affine: if True (and revin=True), learn per-channel affine
            parameters γ and β inside RevIN.
        temporal_norm_momentum: BatchNorm momentum for temporal mixing.
        feature_norm_momentum: BatchNorm momentum for feature mixing.
    """

    h: int = 12
    input_size: int = 36
    n_series: int = 1
    n_block: int = 2
    ff_dim: int = 64
    dropout: float = 0.1
    revin: bool = True
    revin_affine: bool = True
    temporal_norm_momentum: float = 0.05
    feature_norm_momentum: float = 0.05
    use_batchnorm: bool = True  # False → LayerNorm (better for large datasets)
    use_global_skip: bool = False  # direct input→output skip (helps small seasonal datasets)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TemporalMixing(fnn.Module):
    """Time-axis MLP with normalisation and residual connection.

    When ``use_batchnorm=True``: BatchNorm(axis=1) on [B, N, L] — normalises
    over B×L per N channel, matching PyTorch's BatchNorm1d(n_series).
    When ``use_batchnorm=False``: LayerNorm (last-axis) — normalises per
    (B, N) instance over L values; better for large datasets.
    """

    dropout: float
    use_batchnorm: bool = True

    @fnn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        residual = x
        L = x.shape[1]

        x = jnp.transpose(x, (0, 2, 1))          # [B, N, L]
        if self.use_batchnorm:
            x = fnn.BatchNorm(
                use_running_average=deterministic,
                axis=1,
                momentum=0.05,
                epsilon=1e-5,
            )(x)
        else:
            x = fnn.LayerNorm()(x)
        x = fnn.relu(fnn.Dense(L)(x))             # [B, N, L]
        x = jnp.transpose(x, (0, 2, 1))           # [B, L, N]
        x = fnn.Dropout(rate=self.dropout, deterministic=deterministic)(x)

        return x + residual


class FeatureMixing(fnn.Module):
    """Feature-axis two-layer MLP with normalisation and residual connection.

    When ``use_batchnorm=True``: BatchNorm(axis=-1) on [B, L, N] — normalises
    over B×L per N channel, matching PyTorch's BatchNorm1d(n_series).
    When ``use_batchnorm=False``: LayerNorm (last-axis) — normalises per
    (B, L) instance over N features; better for large datasets.
    """

    ff_dim: int
    dropout: float
    use_batchnorm: bool = True

    @fnn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        residual = x
        N = x.shape[-1]

        if self.use_batchnorm:
            x = fnn.BatchNorm(
                use_running_average=deterministic,
                axis=-1,
                momentum=0.05,
                epsilon=1e-5,
            )(x)
        else:
            x = fnn.LayerNorm()(x)
        x = fnn.relu(fnn.Dense(self.ff_dim)(x))   # [B, L, ff_dim]
        x = fnn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        x = fnn.Dense(N)(x)                       # [B, L, N]
        x = fnn.Dropout(rate=self.dropout, deterministic=deterministic)(x)

        return x + residual


class MixingLayer(fnn.Module):
    """One TSMixer block: temporal mixing followed by feature mixing."""

    ff_dim: int
    dropout: float
    use_batchnorm: bool = True

    @fnn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        x = TemporalMixing(
            dropout=self.dropout, use_batchnorm=self.use_batchnorm, name="temporal"
        )(x, deterministic=deterministic)
        x = FeatureMixing(
            ff_dim=self.ff_dim, dropout=self.dropout,
            use_batchnorm=self.use_batchnorm, name="feature",
        )(x, deterministic=deterministic)
        return x


# ---------------------------------------------------------------------------
# Full TSMixer model
# ---------------------------------------------------------------------------


class TSMixer(fnn.Module):
    """JAX/Flax TSMixer — multivariate time-series forecasting model.

    Accepts a batch of multivariate windows ``[B, L, N]`` and returns
    ``[B, h, N]`` forecasts. Carries ``batch_stats`` (running BatchNorm
    statistics) as a mutable variable collection.

    Training call (mutable batch_stats)::

        (predictions, updates) = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x, deterministic=False, mutable=['batch_stats']
        )

    Inference call::

        predictions = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x, deterministic=True
        )

    where ``x: [B, L, N]``.
    """

    config: TSMixerConfig

    @fnn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        cfg = self.config

        # ---- RevIN: per-channel instance normalisation ----------------------
        if cfg.revin:
            mean = jnp.mean(x, axis=1, keepdims=True)   # [B, 1, N]
            std  = jnp.std(x,  axis=1, keepdims=True) + 1e-5
            x = (x - mean) / std

            if cfg.revin_affine:
                gamma = self.param(
                    "revin_gamma", fnn.initializers.ones,  (cfg.n_series,)
                )
                beta = self.param(
                    "revin_beta",  fnn.initializers.zeros, (cfg.n_series,)
                )
                x = x * gamma[None, None, :] + beta[None, None, :]

        # Save normalised input for the global skip connection (before mixing).
        # The skip projects the raw normalised history directly to the forecast
        # horizon, letting the model combine a learned direct mapping (seasonal
        # memory) with the residual correction from the mixing stack.
        x_skip_src = x  # [B, L, N]

        # ---- Stacked mixing layers ----------------------------------------
        for i in range(cfg.n_block):
            x = MixingLayer(
                ff_dim=cfg.ff_dim,
                dropout=cfg.dropout,
                use_batchnorm=cfg.use_batchnorm,
                name=f"mixing_{i}",
            )(x, deterministic=deterministic)

        # ---- Temporal projection L → h -------------------------------------
        x = jnp.transpose(x, (0, 2, 1))          # [B, N, L]
        x = fnn.Dense(cfg.h, name="out")(x)       # [B, N, h]
        x = jnp.transpose(x, (0, 2, 1))           # [B, h, N]

        # ---- Global skip: direct L→h projection of normalised input --------
        if cfg.use_global_skip:
            sk = jnp.transpose(x_skip_src, (0, 2, 1))   # [B, N, L]
            sk = fnn.Dense(cfg.h, name="skip_out")(sk)   # [B, N, h]
            sk = jnp.transpose(sk, (0, 2, 1))            # [B, h, N]
            x = x + sk

        # ---- RevIN denormalisation -----------------------------------------
        if cfg.revin:
            if cfg.revin_affine:
                x = (x - beta[None, None, :]) / (gamma[None, None, :] + 1e-8)
            x = x * std + mean                    # broadcast [B, 1, N] → [B, h, N]

        return x

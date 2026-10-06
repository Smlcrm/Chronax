"""JAX/Flax TSMixerx — multivariate time-series forecasting with exogenous
covariates.

Faithful port of Nixtla NeuralForecast's ``TSMixerx`` to JAX/Flax/Optax,
built on the same mixing-layer family as Chronax's existing ``TSMixer`` port
(``chronax.models.tsmixer``), extended with historic / future / static
exogenous feature mixing, exactly like NF's TSMixerx extends its TSMixer.

Shapes:
    insample_y  [B, L, N]        history (batch, sequence, n_series)
    hist_exog   [B, X, L, N]     historic exogenous, X channels
    futr_exog   [B, F, L+h, N]   future exogenous, F channels
    stat_exog   [N, S]           static exogenous, S channels (not batched —
                                  one row per series, shared across all windows)

Normalisation: RevIN (per-channel instance normalisation with optional
learned affine), computed inline exactly as in Chronax's TSMixer port —
mean/std over the sequence axis, applied to ``insample_y`` before mixing and
inverted on the output.

Forward pass (mirrors NF's TSMixerx.forward):

    x        <- RevIN-normalise(insample_y)                 # [B, L, N]
    x        <- [x, hist_exog] concatenated on channel axis  # [B, 1+X, L, N]
    x        <- [x, futr_exog[:, :, :L]] concatenated        # [B, 1+X+F, L, N]
    x        <- temporal_projection(x)  (Dense: L -> h)      # [B, 1+X+F, N, h]
    x        <- feature_mixer_hist(flatten(C, N) -> ff_dim)  # [B, h, ff_dim]
    x        <- concat with feature_mixer_futr(futr_exog[:, :, L:])
    x        <- concat with feature_mixer_stat(stat_exog)
    x        <- first_mixing(x)                              # [B, h, ff_dim]
    x        <- n_block x MixingLayer[WithStaticExogenous]
    forecast <- out(x)                                        # [B, h, N]
    forecast <- RevIN-denormalise(forecast)

Only point forecasts are supported (output_size=1), matching the existing
TSMixer port's simplification.
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
class TSMixerxConfig:
    """Hyperparameters for :class:`TSMixerx`.

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
            parameters gamma and beta inside RevIN.
        use_batchnorm: True -> BatchNorm inside mixing layers (matches
            PyTorch's BatchNorm1d(n_series) via NF's TSMixer); False ->
            LayerNorm (matches NF's TSMixerx exactly, and is cheaper for the
            smaller per-series batches typical with exogenous covariates).
        futr_exog_size: Number of future exogenous channels ``F``.
        hist_exog_size: Number of historic exogenous channels ``X``.
        stat_exog_size: Number of static exogenous channels ``S``.
    """

    h: int = 12
    input_size: int = 36
    n_series: int = 1
    n_block: int = 2
    ff_dim: int = 64
    dropout: float = 0.0
    revin: bool = True
    revin_affine: bool = True
    use_batchnorm: bool = False
    futr_exog_size: int = 0
    hist_exog_size: int = 0
    stat_exog_size: int = 0


# ---------------------------------------------------------------------------
# Building blocks (self-contained duplicate of chronax.models.tsmixer.model,
# generalised to support differing in/out feature widths — required for the
# exogenous feature mixers and the first fan-in mixing layer; each model
# sub-package stays self-contained per house convention)
# ---------------------------------------------------------------------------


class TemporalMixing(fnn.Module):
    """Time-axis MLP with normalisation and residual connection.

    Operates on ``[..., T, C]`` tensors, mixing across the second-to-last
    (time) axis; the channel count ``C`` is unchanged. Matches NF's
    ``TemporalMixing`` exactly: normalisation is **post**-norm on the
    residual sum (``norm(mix(x) + x)``, not a pre-norm on the raw input),
    and — for the ``use_batchnorm=False`` (default, NF-faithful) path —
    jointly over the last two axes ``(T, C)``, matching PyTorch's
    ``nn.LayerNorm(normalized_shape=(h, num_features))``. Flax's default
    ``LayerNorm`` only reduces the last axis, so ``reduction_axes`` /
    ``feature_axes`` must be set explicitly.
    """

    dropout: float
    use_batchnorm: bool = False

    @fnn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        residual = x
        T = x.shape[-2]

        y = jnp.swapaxes(x, -1, -2)                # [..., C, T]
        y = fnn.relu(fnn.Dense(T)(y))                # [..., C, T]
        y = jnp.swapaxes(y, -1, -2)                  # [..., T, C]
        y = fnn.Dropout(rate=self.dropout, deterministic=deterministic)(y)

        y = y + residual
        if self.use_batchnorm:
            y = fnn.BatchNorm(
                use_running_average=deterministic, axis=-1, momentum=0.05, epsilon=1e-5,
            )(y)
        else:
            y = fnn.LayerNorm(
                reduction_axes=(-2, -1), feature_axes=(-2, -1), epsilon=1e-5,
            )(y)
        return y


class FeatureMixing(fnn.Module):
    """Feature-axis two-layer MLP with normalisation and residual connection.

    Supports ``out_features != in_features`` (the residual is linearly
    projected in that case) — required for the exogenous feature mixers,
    which map varying exogenous-channel counts down to ``ff_dim``. Matches
    NF's ``FeatureMixing`` exactly: normalisation is **post**-norm on the
    residual sum, jointly over the last two axes ``(T, out_features)`` for
    the ``use_batchnorm=False`` (default, NF-faithful) path — see
    :class:`TemporalMixing` docstring for why this needs explicit
    ``reduction_axes`` / ``feature_axes``.
    """

    ff_dim: int
    out_features: int
    dropout: float
    use_batchnorm: bool = False

    @fnn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        in_features = x.shape[-1]

        h = fnn.relu(fnn.Dense(self.ff_dim)(x))
        h = fnn.Dropout(rate=self.dropout, deterministic=deterministic)(h)
        h = fnn.Dense(self.out_features)(h)
        h = fnn.Dropout(rate=self.dropout, deterministic=deterministic)(h)

        residual = (
            fnn.Dense(self.out_features, name="residual_proj")(x)
            if self.out_features != in_features else x
        )
        out = h + residual
        if self.use_batchnorm:
            out = fnn.BatchNorm(
                use_running_average=deterministic, axis=-1, momentum=0.05, epsilon=1e-5,
            )(out)
        else:
            out = fnn.LayerNorm(
                reduction_axes=(-2, -1), feature_axes=(-2, -1), epsilon=1e-5,
            )(out)
        return out


class MixingLayer(fnn.Module):
    """One TSMixerx block: temporal mixing followed by feature mixing."""

    ff_dim: int
    out_features: int
    dropout: float
    use_batchnorm: bool = False

    @fnn.compact
    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        x = TemporalMixing(
            dropout=self.dropout, use_batchnorm=self.use_batchnorm, name="temporal"
        )(x, deterministic=deterministic)
        x = FeatureMixing(
            ff_dim=self.ff_dim, out_features=self.out_features, dropout=self.dropout,
            use_batchnorm=self.use_batchnorm, name="feature",
        )(x, deterministic=deterministic)
        return x


class MixingLayerWithStaticExogenous(fnn.Module):
    """Mixing layer that re-injects static exogenous features at every
    block, exactly matching NF's ``MixingLayerWithStaticExogenous``.
    """

    ff_dim: int
    dropout: float
    use_batchnorm: bool = False

    @fnn.compact
    def __call__(
        self, inputs: Tuple[jnp.ndarray, jnp.ndarray], deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x, stat_exog = inputs                                  # [B,h,ff_dim], [B,h,S]
        x_stat = FeatureMixing(
            ff_dim=self.ff_dim, out_features=self.ff_dim, dropout=self.dropout,
            use_batchnorm=self.use_batchnorm, name="feature_mixer_stat",
        )(stat_exog, deterministic=deterministic)              # [B, h, ff_dim]

        x_cat = jnp.concatenate([x, x_stat], axis=-1)          # [B, h, 2*ff_dim]
        x_out = MixingLayer(
            ff_dim=self.ff_dim, out_features=self.ff_dim, dropout=self.dropout,
            use_batchnorm=self.use_batchnorm, name="mixer",
        )(x_cat, deterministic=deterministic)                  # [B, h, ff_dim]
        return x_out, stat_exog


# ---------------------------------------------------------------------------
# Full TSMixerx model
# ---------------------------------------------------------------------------


class TSMixerx(fnn.Module):
    """JAX/Flax TSMixerx — multivariate time-series forecasting with
    exogenous covariates.

    Carries ``batch_stats`` (running BatchNorm statistics) as a mutable
    variable collection whenever ``config.use_batchnorm=True``.

    Training call (mutable batch_stats, only when use_batchnorm=True)::

        (predictions, updates) = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            insample_y, hist_exog=hist_exog, futr_exog=futr_exog, stat_exog=stat_exog,
            deterministic=False, mutable=['batch_stats']
        )

    Inference call::

        predictions = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            insample_y, hist_exog=hist_exog, futr_exog=futr_exog, stat_exog=stat_exog,
            deterministic=True,
        )
    """

    config: TSMixerxConfig

    @fnn.compact
    def __call__(
        self,
        insample_y: jnp.ndarray,                        # [B, L, N]
        hist_exog: Optional[jnp.ndarray] = None,         # [B, X, L, N]
        futr_exog: Optional[jnp.ndarray] = None,         # [B, F, L+h, N]
        stat_exog: Optional[jnp.ndarray] = None,         # [N, S]
        deterministic: bool = True,
    ) -> jnp.ndarray:                                     # [B, h, N]
        cfg = self.config
        B, L, N = insample_y.shape

        if cfg.hist_exog_size > 0 and hist_exog is None:
            hist_exog = jnp.zeros((B, cfg.hist_exog_size, L, N))
        if cfg.futr_exog_size > 0 and futr_exog is None:
            futr_exog = jnp.zeros((B, cfg.futr_exog_size, L + cfg.h, N))
        if cfg.stat_exog_size > 0 and stat_exog is None:
            stat_exog = jnp.zeros((N, cfg.stat_exog_size))

        x = insample_y

        # ---- RevIN: per-channel instance normalisation --------------------
        # NF's RevINMultivariate detaches batch_mean/batch_std from autograd
        # (they are statistics, not learned parameters); stop_gradient matches
        # that exactly — without it, gradients would flow back through the
        # normalisation constants themselves, which NF never does.
        mean = std = gamma = beta = None
        if cfg.revin:
            mean = jax.lax.stop_gradient(jnp.mean(x, axis=1, keepdims=True))     # [B, 1, N]
            std = jax.lax.stop_gradient(jnp.std(x, axis=1, keepdims=True) + 1e-5)
            x = (x - mean) / std

            if cfg.revin_affine:
                gamma = self.param("revin_gamma", fnn.initializers.ones, (cfg.n_series,))
                beta = self.param("revin_beta", fnn.initializers.zeros, (cfg.n_series,))
                x = x * gamma[None, None, :] + beta[None, None, :]

        # ---- Channel stack: [y, hist_exog, futr_exog(hist window)] --------
        x = x[:, None, :, :]                               # [B, 1, L, N]
        if cfg.hist_exog_size > 0:
            x = jnp.concatenate([x, hist_exog], axis=1)     # [B, 1+X, L, N]
        if cfg.futr_exog_size > 0:
            futr_hist = futr_exog[:, :, :L, :]               # [B, F, L, N]
            x = jnp.concatenate([x, futr_hist], axis=1)      # [B, 1+X+F, L, N]

        # ---- Temporal projection L -> h ------------------------------------
        x = jnp.transpose(x, (0, 1, 3, 2))                  # [B, C, N, L]
        x = fnn.Dense(cfg.h, name="temporal_projection")(x)  # [B, C, N, h]
        x = jnp.transpose(x, (0, 3, 1, 2))                  # [B, h, C, N]
        C = x.shape[2]
        x = x.reshape(B, cfg.h, C * N)                      # [B, h, C*N]
        x = FeatureMixing(
            ff_dim=cfg.ff_dim, out_features=cfg.ff_dim, dropout=cfg.dropout,
            use_batchnorm=cfg.use_batchnorm, name="feature_mixer_hist",
        )(x, deterministic=deterministic)                    # [B, h, ff_dim]

        # ---- Future-exogenous horizon slice --------------------------------
        if cfg.futr_exog_size > 0:
            x_futr = futr_exog[:, :, L:, :]                  # [B, F, h, N]
            x_futr = jnp.transpose(x_futr, (0, 2, 1, 3))       # [B, h, F, N]
            x_futr = x_futr.reshape(B, cfg.h, -1)             # [B, h, F*N]
            x_futr = FeatureMixing(
                ff_dim=cfg.ff_dim, out_features=cfg.ff_dim, dropout=cfg.dropout,
                use_batchnorm=cfg.use_batchnorm, name="feature_mixer_futr",
            )(x_futr, deterministic=deterministic)            # [B, h, ff_dim]
            x = jnp.concatenate([x, x_futr], axis=2)          # [B, h, 2*ff_dim]

        # ---- Static-exogenous broadcast ------------------------------------
        stat_b = None
        if cfg.stat_exog_size > 0:
            stat_flat = stat_exog.reshape(-1)                 # [N*S]
            stat_b = jnp.broadcast_to(
                stat_flat[None, None, :], (B, cfg.h, stat_flat.shape[0])
            )                                                  # [B, h, N*S]
            x_stat = FeatureMixing(
                ff_dim=cfg.ff_dim, out_features=cfg.ff_dim, dropout=cfg.dropout,
                use_batchnorm=cfg.use_batchnorm, name="feature_mixer_stat_init",
            )(stat_b, deterministic=deterministic)            # [B, h, ff_dim]
            x = jnp.concatenate([x, x_stat], axis=2)

        # ---- First mixing layer: fan concatenated features back to ff_dim -
        x = MixingLayer(
            ff_dim=cfg.ff_dim, out_features=cfg.ff_dim, dropout=cfg.dropout,
            use_batchnorm=cfg.use_batchnorm, name="first_mixing",
        )(x, deterministic=deterministic)

        # ---- N blocks of mixing layers --------------------------------------
        if cfg.stat_exog_size > 0:
            for i in range(cfg.n_block):
                x, stat_b = MixingLayerWithStaticExogenous(
                    ff_dim=cfg.ff_dim, dropout=cfg.dropout, use_batchnorm=cfg.use_batchnorm,
                    name=f"mixing_{i}",
                )((x, stat_b), deterministic=deterministic)
        else:
            for i in range(cfg.n_block):
                x = MixingLayer(
                    ff_dim=cfg.ff_dim, out_features=cfg.ff_dim, dropout=cfg.dropout,
                    use_batchnorm=cfg.use_batchnorm, name=f"mixing_{i}",
                )(x, deterministic=deterministic)

        # ---- Output projection -----------------------------------------------
        forecast = fnn.Dense(cfg.n_series, name="out")(x)     # [B, h, N]

        # ---- RevIN denormalisation ---------------------------------------------
        if cfg.revin:
            if cfg.revin_affine:
                forecast = (forecast - beta[None, None, :]) / (gamma[None, None, :] + 1e-8)
            forecast = forecast * std + mean

        return forecast

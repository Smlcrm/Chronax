"""JAX/Flax Autoformer — univariate direct-decoding forecaster.

Architecture mirrors neuralforecast's Autoformer (Wu et al., 2021) for the
univariate, no-exogenous case:
  - Circular TokenEmbedding (k=3 Conv).
  - Auto-Correlation encoder/decoder with progressive series decomposition.
  - Autoformer seasonal LayerNorm (subtracts time-mean after standard LN).
  - Final linear projection.

Pure JAX/Flax/Optax — no PyTorch, no numpy.

Reference:
    Wu, H. et al. "Autoformer: Decomposition Transformers with Auto-Correlation
    for Long-Term Series Forecasting." NeurIPS 2021.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import flax.linen as fnn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoformerConfig:
    """Hyperparameters for :class:`AutoformerModel`.

    Args:
        h: Forecast horizon.
        input_size: Context window length fed to the encoder.
        hidden_size: Embedding / attention hidden dimension.
        n_heads: Number of auto-correlation heads (must divide ``hidden_size``).
        factor: Auto-correlation top-k factor (``top_k = factor * log(L)``).
        moving_avg_window: Kernel size for the trend moving-average filter.
        encoder_layers: Number of stacked encoder layers.
        decoder_layers: Number of stacked decoder layers.
        conv_hidden_size: Hidden channels for the position-wise FFN convolutions.
        decoder_input_size_multiplier: Fraction of ``input_size`` used as the
            decoder start-token ("label") length; must be in ``(0, 1)``.
        dropout: Dropout rate applied throughout (training only).
        activation: FFN nonlinearity — ``"relu"`` or ``"gelu"``.
    """

    h: int = 24
    input_size: int = 72
    hidden_size: int = 128
    n_heads: int = 4
    factor: int = 3
    moving_avg_window: int = 25
    encoder_layers: int = 2
    decoder_layers: int = 1
    conv_hidden_size: int = 32
    decoder_input_size_multiplier: float = 0.5
    dropout: float = 0.05
    activation: str = "gelu"

    def __post_init__(self) -> None:
        for name in ("h", "input_size", "hidden_size", "n_heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        if self.activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be 'relu' or 'gelu', got {self.activation!r}.")
        ll = int(math.ceil(self.input_size * self.decoder_input_size_multiplier))
        if ll <= 0 or ll >= self.input_size:
            raise ValueError(
                f"decoder_input_size_multiplier={self.decoder_input_size_multiplier} "
                f"yields label_len={ll}; must be in (0, input_size)."
            )

    @property
    def label_len(self) -> int:
        """Decoder start-token length derived from ``decoder_input_size_multiplier``."""
        return int(math.ceil(self.input_size * self.decoder_input_size_multiplier))


# ---------------------------------------------------------------------------
# Math primitives (pure JAX, no numpy)
# ---------------------------------------------------------------------------


def _activation(name: str, x: jnp.ndarray) -> jnp.ndarray:
    if name == "relu":
        return jax.nn.relu(x)
    return jax.nn.gelu(x, approximate=False)


def moving_avg(x: jnp.ndarray, kernel_size: int) -> jnp.ndarray:
    """Edge-padded moving average on ``[B, T, C]``, output same length ``T``."""
    pad = (kernel_size - 1) // 2
    xp = jnp.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    cs = jnp.cumsum(xp, axis=1)
    cs = jnp.concatenate([jnp.zeros_like(cs[:, :1, :]), cs], axis=1)
    return (cs[:, kernel_size:, :] - cs[:, :-kernel_size, :]) / kernel_size


def series_decomp(x: jnp.ndarray, kernel_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Decompose into ``(seasonal, trend)``; trend is a moving average."""
    trend = moving_avg(x, kernel_size)
    return x - trend, trend


# ---------------------------------------------------------------------------
# Auto-correlation (FFT period discovery + time-delay aggregation)
# ---------------------------------------------------------------------------


def _time_delay_agg_training(
    values: jnp.ndarray, corr: jnp.ndarray, top_k: int
) -> jnp.ndarray:
    """Shared batch-mean top-k aggregation used during training.

    Fully vectorised gather — one XLA gather op, no loop, trivial gradient.
    """
    B, H, E, length = values.shape
    mean_value = jnp.mean(jnp.mean(corr, axis=1), axis=1)  # [B, L]
    batch_mean = jnp.mean(mean_value, axis=0)               # [L]
    _, index = jax.lax.top_k(batch_mean, top_k)             # [top_k]
    weights = mean_value[:, index]                           # [B, top_k]
    tmp_corr = jax.nn.softmax(weights, axis=-1)             # [B, top_k]
    ar = jnp.arange(length)
    roll_idx = (ar[None, :] + index[:, None]) % length      # [top_k, L]
    # values[..., roll_idx]: [B, H, E, top_k, L] via advanced indexing
    rolled = values[..., roll_idx]
    return jnp.sum(rolled * tmp_corr[:, None, None, :, None], axis=-2)  # [B, H, E, L]


def _time_delay_agg_inference(
    values: jnp.ndarray, corr: jnp.ndarray, top_k: int
) -> jnp.ndarray:
    """Per-sample top-k aggregation used during inference.

    Fully vectorised gather — one XLA gather op, no loop, trivial gradient.
    """
    B, H, E, length = values.shape
    mean_value = jnp.mean(jnp.mean(corr, axis=1), axis=1)  # [B, L]
    weights, delay = jax.lax.top_k(mean_value, top_k)       # [B, top_k]
    tmp_corr = jax.nn.softmax(weights, axis=-1)             # [B, top_k]
    ar = jnp.arange(length)
    roll_idx = (ar[None, None, :] + delay[:, :, None]) % length         # [B, top_k, L]
    # Gather per-sample: broadcast to [B, H, E, top_k, L] then take_along_axis
    v_exp = jnp.broadcast_to(values[:, :, :, None, :], (B, H, E, top_k, length))
    i_exp = jnp.broadcast_to(roll_idx[:, None, None, :, :], (B, H, E, top_k, length))
    rolled = jnp.take_along_axis(v_exp, i_exp, axis=-1)                 # [B, H, E, top_k, L]
    return jnp.sum(rolled * tmp_corr[:, None, None, :, None], axis=-2)  # [B, H, E, L]


def auto_correlation(
    queries: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    factor: int,
    training: bool = True,
) -> jnp.ndarray:
    """FFT-based auto-correlation on per-head ``[B, L, H, E]`` tensors.

    Discovers period-based dependencies via FFT cross-correlation then
    aggregates by time-delay. Dispatches training (shared batch top-k) vs
    inference (per-sample top-k) based on ``training``.
    """
    b, length, h, e = queries.shape
    s = values.shape[1]
    if length > s:
        pad = length - s
        zeros = jnp.zeros((b, pad, h, e), dtype=values.dtype)
        values = jnp.concatenate([values, zeros], axis=1)
        keys = jnp.concatenate([keys, zeros], axis=1)
    else:
        values = values[:, :length]
        keys = keys[:, :length]

    q = jnp.transpose(queries, (0, 2, 3, 1))  # [B, H, E, L]
    k = jnp.transpose(keys, (0, 2, 3, 1))
    q_fft = jnp.fft.rfft(q, axis=-1)
    k_fft = jnp.fft.rfft(k, axis=-1)
    corr = jnp.fft.irfft(q_fft * jnp.conj(k_fft), n=length, axis=-1)  # [B, H, E, L]

    top_k = max(1, int(factor * math.log(length)))
    v = jnp.transpose(values, (0, 2, 3, 1))  # [B, H, E, L]
    if training:
        agg = _time_delay_agg_training(v, corr, top_k)
    else:
        agg = _time_delay_agg_inference(v, corr, top_k)
    return jnp.transpose(agg, (0, 3, 1, 2))  # [B, L, H, E]


# ---------------------------------------------------------------------------
# He/Kaiming normal initializer (matches TokenEmbedding in neuralforecast)
# ---------------------------------------------------------------------------

_HE_NORMAL = fnn.initializers.he_normal()


# ---------------------------------------------------------------------------
# Flax linen modules
# ---------------------------------------------------------------------------


class AutoCorrelationLayer(fnn.Module):
    """Multi-head auto-correlation: Q/K/V/out projections + FFT block."""

    hidden_size: int
    n_heads: int
    factor: int

    @fnn.compact
    def __call__(
        self,
        queries: jnp.ndarray,
        keys: jnp.ndarray,
        values: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        b, length, _ = queries.shape
        s = keys.shape[1]
        q = fnn.Dense(self.hidden_size, name="q_proj")(queries).reshape(
            b, length, self.n_heads, -1
        )
        k = fnn.Dense(self.hidden_size, name="k_proj")(keys).reshape(
            b, s, self.n_heads, -1
        )
        v = fnn.Dense(self.hidden_size, name="v_proj")(values).reshape(
            b, s, self.n_heads, -1
        )
        out = auto_correlation(q, k, v, self.factor, training=not deterministic)
        return fnn.Dense(self.hidden_size, name="out_proj")(out.reshape(b, length, -1))


class SeasonalLayerNorm(fnn.Module):
    """Autoformer seasonal norm: standard LayerNorm then subtract the time-mean."""

    hidden_size: int

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_hat = fnn.LayerNorm(name="ln")(x)
        return x_hat - jnp.mean(x_hat, axis=1, keepdims=True)


class EncoderLayer(fnn.Module):
    """Encoder layer: auto-correlation + decomp, then conv FFN + decomp."""

    hidden_size: int
    conv_hidden_size: int
    n_heads: int
    factor: int
    moving_avg_window: int
    dropout_rate: float
    activation: str

    @fnn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        new_x = AutoCorrelationLayer(
            self.hidden_size, self.n_heads, self.factor, name="attn"
        )(x, x, x, deterministic)
        x = x + fnn.Dropout(rate=self.dropout_rate, name="drop1")(
            new_x, deterministic=deterministic
        )
        x, _ = series_decomp(x, self.moving_avg_window)

        y = fnn.Conv(
            features=self.conv_hidden_size,
            kernel_size=(1,),
            use_bias=False,
            kernel_init=_HE_NORMAL,
            name="conv1",
        )(x)
        y = _activation(self.activation, y)
        y = fnn.Dropout(rate=self.dropout_rate, name="drop2")(y, deterministic=deterministic)
        y = fnn.Conv(
            features=self.hidden_size,
            kernel_size=(1,),
            use_bias=False,
            kernel_init=_HE_NORMAL,
            name="conv2",
        )(y)
        y = fnn.Dropout(rate=self.dropout_rate, name="drop3")(y, deterministic=deterministic)

        res, _ = series_decomp(x + y, self.moving_avg_window)
        return res


class Encoder(fnn.Module):
    """Stack of encoder layers followed by seasonal LayerNorm."""

    n_layers: int
    hidden_size: int
    conv_hidden_size: int
    n_heads: int
    factor: int
    moving_avg_window: int
    dropout_rate: float
    activation: str

    @fnn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for i in range(self.n_layers):
            x = EncoderLayer(
                hidden_size=self.hidden_size,
                conv_hidden_size=self.conv_hidden_size,
                n_heads=self.n_heads,
                factor=self.factor,
                moving_avg_window=self.moving_avg_window,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                name=f"layer_{i}",
            )(x, deterministic)
        return SeasonalLayerNorm(self.hidden_size, name="norm")(x)


class DecoderLayer(fnn.Module):
    """Decoder layer: self + cross auto-correlation, conv FFN, trend projection."""

    hidden_size: int
    conv_hidden_size: int
    n_heads: int
    factor: int
    c_out: int
    moving_avg_window: int
    dropout_rate: float
    activation: str

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cross: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = x + fnn.Dropout(rate=self.dropout_rate, name="drop1")(
            AutoCorrelationLayer(
                self.hidden_size, self.n_heads, self.factor, name="self_attn"
            )(x, x, x, deterministic),
            deterministic=deterministic,
        )
        x, trend1 = series_decomp(x, self.moving_avg_window)

        x = x + fnn.Dropout(rate=self.dropout_rate, name="drop2")(
            AutoCorrelationLayer(
                self.hidden_size, self.n_heads, self.factor, name="cross_attn"
            )(x, cross, cross, deterministic),
            deterministic=deterministic,
        )
        x, trend2 = series_decomp(x, self.moving_avg_window)

        y = fnn.Conv(
            features=self.conv_hidden_size,
            kernel_size=(1,),
            use_bias=False,
            kernel_init=_HE_NORMAL,
            name="conv1",
        )(x)
        y = _activation(self.activation, y)
        y = fnn.Dropout(rate=self.dropout_rate, name="drop3")(y, deterministic=deterministic)
        y = fnn.Conv(
            features=self.hidden_size,
            kernel_size=(1,),
            use_bias=False,
            kernel_init=_HE_NORMAL,
            name="conv2",
        )(y)
        y = fnn.Dropout(rate=self.dropout_rate, name="drop4")(y, deterministic=deterministic)
        x, trend3 = series_decomp(x + y, self.moving_avg_window)

        residual_trend = fnn.Conv(
            features=self.c_out,
            kernel_size=(3,),
            padding="CIRCULAR",
            use_bias=False,
            kernel_init=_HE_NORMAL,
            name="trend_proj",
        )(trend1 + trend2 + trend3)
        return x, residual_trend


class Decoder(fnn.Module):
    """Stack of decoder layers + seasonal LayerNorm + final linear projection."""

    n_layers: int
    hidden_size: int
    conv_hidden_size: int
    n_heads: int
    factor: int
    c_out: int
    moving_avg_window: int
    dropout_rate: float
    activation: str

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cross: jnp.ndarray,
        trend: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        for i in range(self.n_layers):
            x, residual_trend = DecoderLayer(
                hidden_size=self.hidden_size,
                conv_hidden_size=self.conv_hidden_size,
                n_heads=self.n_heads,
                factor=self.factor,
                c_out=self.c_out,
                moving_avg_window=self.moving_avg_window,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                name=f"layer_{i}",
            )(x, cross, deterministic)
            trend = trend + residual_trend
        x = SeasonalLayerNorm(self.hidden_size, name="norm")(x)
        return fnn.Dense(self.c_out, name="projection")(x), trend


class AutoformerModel(fnn.Module):
    """Univariate Autoformer forecaster built with ``flax.linen``.

    Forward pass: ``[B, input_size, 1] -> [B, h, 1]``.

    During training pass ``deterministic=False`` and supply a ``"dropout"``
    key via ``rngs={"dropout": key}`` in ``model.apply(...)``.
    """

    config: AutoformerConfig

    @fnn.compact
    def __call__(
        self,
        insample_y: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config
        b, _, c = insample_y.shape
        h, label_len = cfg.h, cfg.label_len

        # Decompose input into seasonal + trend init for the decoder
        mean = jnp.broadcast_to(jnp.mean(insample_y, axis=1, keepdims=True), (b, h, c))
        zeros = jnp.zeros((b, h, c), dtype=insample_y.dtype)
        seasonal_init, trend_init = series_decomp(insample_y, cfg.moving_avg_window)
        trend_init = jnp.concatenate([trend_init[:, -label_len:, :], mean], axis=1)
        seasonal_init = jnp.concatenate([seasonal_init[:, -label_len:, :], zeros], axis=1)

        # Encoder
        enc_out = fnn.Conv(
            features=cfg.hidden_size,
            kernel_size=(3,),
            padding="CIRCULAR",
            use_bias=False,
            kernel_init=_HE_NORMAL,
            name="enc_embedding",
        )(insample_y)
        enc_out = fnn.Dropout(rate=cfg.dropout, name="enc_drop")(
            enc_out, deterministic=deterministic
        )
        enc_out = Encoder(
            n_layers=cfg.encoder_layers,
            hidden_size=cfg.hidden_size,
            conv_hidden_size=cfg.conv_hidden_size,
            n_heads=cfg.n_heads,
            factor=cfg.factor,
            moving_avg_window=cfg.moving_avg_window,
            dropout_rate=cfg.dropout,
            activation=cfg.activation,
            name="encoder",
        )(enc_out, deterministic)

        # Decoder
        dec_out = fnn.Conv(
            features=cfg.hidden_size,
            kernel_size=(3,),
            padding="CIRCULAR",
            use_bias=False,
            kernel_init=_HE_NORMAL,
            name="dec_embedding",
        )(seasonal_init)
        dec_out = fnn.Dropout(rate=cfg.dropout, name="dec_drop")(
            dec_out, deterministic=deterministic
        )
        seasonal_part, trend_part = Decoder(
            n_layers=cfg.decoder_layers,
            hidden_size=cfg.hidden_size,
            conv_hidden_size=cfg.conv_hidden_size,
            n_heads=cfg.n_heads,
            factor=cfg.factor,
            c_out=1,
            moving_avg_window=cfg.moving_avg_window,
            dropout_rate=cfg.dropout,
            activation=cfg.activation,
            name="decoder",
        )(dec_out, enc_out, trend_init, deterministic)

        out = trend_part + seasonal_part
        return out[:, -h:, :]

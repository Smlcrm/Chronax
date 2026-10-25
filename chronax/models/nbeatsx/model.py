"""JAX/Flax N-BEATSx model.

Faithful port of Nixtla NeuralForecast's ``NBEATSx``, built on top of Chronax's
existing NBEATS port (``chronax.models.nbeats``) and extended with exogenous
covariates, exactly like NF's NBEATSx extends its NBEATS:

- ``LayerNorm`` after every hidden Dense (same house-style deviation the
  existing NBEATS port already applies over NF's bare FC stack).
- Pure JAX/Flax/Optax — no PyTorch, no numpy at forward time.
- An additional ``"exogenous"`` block type projects future + static exogenous
  regressors directly (``ExogenousBasis``), matching NF's linear-combination
  exogenous block.
- Historic / future / static exogenous features are flattened and
  concatenated onto the block's FC input, exactly as in NF's ``NBEATSBlock``.

Architecture (mirrors NeuralForecast NBEATSx):

    residuals ← flip(insample_y)                     # most-recent-first
    forecast  ← level(insample_y)
    for each block:
        x                 = [residuals, hist_exog.flat, futr_exog.flat, stat_exog.flat]
        theta             = FC_stack(x)               # [B, n_theta]
        backcast, fcast_i = basis_project(theta, futr_exog)
        residuals         = (residuals - backcast) * mask
        forecast         += fcast_i
    return forecast                                   # [B, h]

Only point forecasts are supported (matches the existing NBEATS port's
simplification — no ``loss.outputsize_multiplier`` / multi-quantile heads).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import flax.linen as fnn
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NBEATSxConfig:
    """Hyperparameters for :class:`NBEATSx`.

    Attributes:
        h: Forecast horizon.
        input_size: History window length ``L`` fed to each block.
        stack_types: Sequence of block-stack types; each element is one of
            ``"identity"``, ``"trend"``, ``"seasonality"``, or ``"exogenous"``.
            ``"exogenous"`` requires ``futr_exog_size > 0`` or
            ``stat_exog_size > 0``.
        n_blocks: Number of blocks per stack (must have same length as
            ``stack_types``).
        mlp_units: Hidden dimension used in every Dense layer of every block.
        mlp_layers: Number of hidden Dense layers per block (output layer is
            extra).
        n_harmonics: Fourier harmonics for the seasonality basis.
        n_basis: Polynomial degree for the trend basis; basis size = n_basis+1.
        basis: Trend basis type — ``"polynomial"``, ``"legendre"``,
            ``"chebyshev"``, or ``"changepoint"``.
        activation: Activation function name (``"relu"`` matches NF default).
        shared_weights: When True all blocks within a stack share parameters.
        dropout_prob: Dropout rate (0 = disabled).
        layer_norm: When True insert LayerNorm after every hidden Dense.
        futr_exog_size: Number of future exogenous features ``F``.
        hist_exog_size: Number of historic exogenous features ``X``.
        stat_exog_size: Number of static exogenous features ``S``.
    """

    h: int = 24
    input_size: int = 48
    stack_types: Tuple[str, ...] = ("identity", "trend", "seasonality")
    n_blocks: Tuple[int, ...] = (1, 1, 1)
    mlp_units: int = 512
    mlp_layers: int = 2
    n_harmonics: int = 2
    n_basis: int = 2
    basis: str = "polynomial"
    activation: str = "relu"
    shared_weights: bool = False
    dropout_prob: float = 0.0
    layer_norm: bool = True
    futr_exog_size: int = 0
    hist_exog_size: int = 0
    stat_exog_size: int = 0


# ---------------------------------------------------------------------------
# Basis functions (self-contained duplicate of chronax.models.nbeats.model;
# each model sub-package stays self-contained per house convention)
# ---------------------------------------------------------------------------
# All functions take only Python-int arguments → XLA constant-folds at JIT.


def _polynomial_basis(length: int, n_terms: int) -> jnp.ndarray:
    """Returns ``[n_terms, length]``.  t = [0, 1/L, 2/L, …, (L-1)/L]."""
    t = jnp.arange(length, dtype=jnp.float32) / length
    return jnp.stack([t ** i for i in range(n_terms)], axis=0)


def _legendre_basis(length: int, n_terms: int) -> jnp.ndarray:
    """Legendre polynomial basis on [-1, 1]. Returns ``[n_terms, length]``."""
    x = jnp.linspace(-1.0, 1.0, length, dtype=jnp.float32)
    rows = [jnp.ones(length, dtype=jnp.float32)]  # P_0 = 1
    if n_terms > 1:
        rows.append(x)                              # P_1 = x
    for k in range(1, n_terms - 1):
        p_next = ((2 * k + 1) * x * rows[-1] - k * rows[-2]) / (k + 1)
        rows.append(p_next)
    return jnp.stack(rows[:n_terms], axis=0)


def _chebyshev_basis(length: int, n_terms: int) -> jnp.ndarray:
    """Chebyshev-T polynomial basis on [-1, 1]. Returns ``[n_terms, length]``."""
    x = jnp.linspace(-1.0, 1.0, length, dtype=jnp.float32)
    rows = [jnp.ones(length, dtype=jnp.float32)]   # T_0 = 1
    if n_terms > 1:
        rows.append(x)                               # T_1 = x
    for _ in range(1, n_terms - 1):
        rows.append(2.0 * x * rows[-1] - rows[-2])  # recurrence
    return jnp.stack(rows[:n_terms], axis=0)


def _changepoint_basis(length: int, n_terms: int) -> jnp.ndarray:
    """Changepoint basis. Returns ``[n_terms, length]``."""
    t = jnp.linspace(0.0, 1.0, length, dtype=jnp.float32)            # [L]
    cp = jnp.linspace(0.0, 1.0, n_terms + 1, dtype=jnp.float32)[1:]  # [n_terms]
    return jnp.maximum(0.0, t[None, :] - cp[:, None])                 # [n_terms, L]


_TREND_BASIS = {
    "polynomial": _polynomial_basis,
    "legendre":   _legendre_basis,
    "chebyshev":  _chebyshev_basis,
    "changepoint": _changepoint_basis,
}

# He-uniform matches PyTorch's default Kaiming-uniform for ReLU layers.
_HE_UNIFORM = fnn.initializers.he_uniform()


def _get_trend_basis(length: int, n_terms: int, basis: str) -> jnp.ndarray:
    fn = _TREND_BASIS.get(basis)
    if fn is None:
        raise ValueError(
            f"Unknown trend basis {basis!r}. Choose from {list(_TREND_BASIS)}."
        )
    return fn(length, n_terms)


def _seasonality_basis(
    backcast_size: int, forecast_size: int, harmonics: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fourier basis matching NeuralForecast's ``SeasonalityBasis``.

    Returns:
        backcast_basis: ``[2*n_freqs, backcast_size]``
        forecast_basis: ``[2*n_freqs, forecast_size]``
    """
    n_freqs = int(math.ceil(harmonics / 2 * forecast_size)) - (harmonics - 1)
    n_nz = n_freqs - 1
    freq_nz = jnp.arange(harmonics, harmonics + n_nz, dtype=jnp.float32) / harmonics
    freq = jnp.concatenate([jnp.zeros(1, dtype=jnp.float32), freq_nz])  # [n_freqs]

    t_back = jnp.arange(backcast_size, dtype=jnp.float32) / forecast_size
    t_fore = jnp.arange(forecast_size, dtype=jnp.float32) / forecast_size

    grid_back = -2.0 * math.pi * t_back[:, None] * freq[None, :]  # [L, n_freqs]
    grid_fore =  2.0 * math.pi * t_fore[:, None] * freq[None, :]  # [h, n_freqs]

    bb = jnp.concatenate([jnp.cos(grid_back), jnp.sin(grid_back)], axis=1).T  # [2K, L]
    fb = jnp.concatenate([jnp.cos(grid_fore), jnp.sin(grid_fore)], axis=1).T  # [2K, h]
    return bb, fb


def _harmonic_size(harmonics: int, h: int) -> int:
    return 2 * (int(math.ceil(harmonics / 2 * h)) - (harmonics - 1))


def _n_theta(
    block_type: str,
    input_size: int,
    h: int,
    n_basis: int,
    n_harmonics: int,
    futr_input_size: int = 0,
    stat_input_size: int = 0,
) -> int:
    """Total MLP output size (theta) for a given block type."""
    if block_type == "identity":
        return input_size + h
    if block_type == "trend":
        return 2 * (n_basis + 1)
    if block_type == "seasonality":
        return 2 * _harmonic_size(n_harmonics, h)
    if block_type == "exogenous":
        return 2 * (futr_input_size + stat_input_size)
    raise ValueError(f"Unknown block type: {block_type!r}")


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "relu":       fnn.relu,
    "tanh":       jnp.tanh,
    "selu":       fnn.selu,
    "gelu":       fnn.gelu,
    "sigmoid":    fnn.sigmoid,
    "leaky_relu": fnn.leaky_relu,
    "softplus":   fnn.softplus,
}


def _activate(x: jnp.ndarray, name: str) -> jnp.ndarray:
    fn = _ACTIVATIONS.get(name.lower())
    if fn is None:
        raise ValueError(f"Unsupported activation {name!r}. Choose from {list(_ACTIVATIONS)}.")
    return fn(x)


# ---------------------------------------------------------------------------
# Exogenous basis
# ---------------------------------------------------------------------------


class ExogenousBasis(fnn.Module):
    """Linear-combination-of-regressors basis (NF's ``ExogenousBasis``).

    Unlike the trend/seasonality/identity bases, this basis is *data
    dependent*: the backcast/forecast projection vectors come directly from
    the (augmented) future-exogenous tensor rather than a fixed learned or
    analytic matrix.

    Args:
        forecast_size: Horizon ``h``; used to split ``futr_exog`` into its
            backcast (``[:-h]``) and forecast (``[-h:]``) slices.
    """

    forecast_size: int

    @fnn.compact
    def __call__(
        self, theta: jnp.ndarray, futr_exog: jnp.ndarray  # [B, n_theta], [B, L+h, C]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        backcast_basis = jnp.transpose(futr_exog[:, : -self.forecast_size, :], (0, 2, 1))  # [B, C, L]
        forecast_basis = jnp.transpose(futr_exog[:, -self.forecast_size :, :], (0, 2, 1))  # [B, C, h]
        cut = forecast_basis.shape[1]  # = C (channel count)

        backcast_theta = theta[:, cut:]  # [B, C]
        forecast_theta = theta[:, :cut]  # [B, C]  (point forecast: out_features=1)

        backcast = jnp.einsum("bp,bpt->bt", backcast_theta, backcast_basis)  # [B, L]
        forecast = jnp.einsum("bp,bpt->bt", forecast_theta, forecast_basis)  # [B, h]
        return backcast, forecast


# ---------------------------------------------------------------------------
# N-BEATSx block
# ---------------------------------------------------------------------------


class NBEATSxBlock(fnn.Module):
    """Single N-BEATSx block: FC stack → theta → basis projection.

    Args:
        config: Shared model config.
        block_type: One of ``"identity"``, ``"trend"``, ``"seasonality"``,
            ``"exogenous"``.
    """

    config: NBEATSxConfig
    block_type: str

    @fnn.compact
    def __call__(
        self,
        residuals: jnp.ndarray,                        # [B, L]
        hist_exog: Optional[jnp.ndarray] = None,        # [B, L, X]
        futr_exog: Optional[jnp.ndarray] = None,        # [B, L+h, F]
        stat_exog: Optional[jnp.ndarray] = None,        # [B, S]
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        B = residuals.shape[0]
        if self.block_type == "exogenous" and cfg.futr_exog_size == 0 and cfg.stat_exog_size == 0:
            raise ValueError(
                "The 'exogenous' block requires futr_exog_size > 0 or "
                "stat_exog_size > 0."
            )
        n_out = _n_theta(
            self.block_type, cfg.input_size, cfg.h, cfg.n_basis, cfg.n_harmonics,
            futr_input_size=cfg.futr_exog_size, stat_input_size=cfg.stat_exog_size,
        )

        # FC input: [ Y_[t-L:t], X_[t-L:t], F_[t-L:t+H], S ] — flattened & concatenated,
        # exactly matching NF's NBEATSBlock.forward.
        x = residuals
        if cfg.hist_exog_size > 0:
            x = jnp.concatenate([x, hist_exog.reshape(B, -1)], axis=1)
        if cfg.futr_exog_size > 0:
            x = jnp.concatenate([x, futr_exog.reshape(B, -1)], axis=1)
        if cfg.stat_exog_size > 0:
            x = jnp.concatenate([x, stat_exog.reshape(B, -1)], axis=1)

        for layer_idx in range(cfg.mlp_layers):
            x = fnn.Dense(cfg.mlp_units, name=f"fc_{layer_idx}", kernel_init=_HE_UNIFORM)(x)
            if cfg.layer_norm:
                x = fnn.LayerNorm(name=f"ln_{layer_idx}")(x)
            x = _activate(x, cfg.activation)
            if cfg.dropout_prob > 0:
                x = fnn.Dropout(rate=cfg.dropout_prob, deterministic=deterministic)(x)

        # Near-zero init for the identity block keeps its initial contribution
        # close to zero so the forecast starts at the level baseline; other
        # block types retain Lecun-uniform so their basis coefficients start
        # with enough signal to capture structure quickly.
        _theta_init = (
            fnn.initializers.truncated_normal(stddev=0.01)
            if self.block_type == "identity"
            else fnn.initializers.lecun_uniform()
        )
        theta = fnn.Dense(n_out, name="theta", kernel_init=_theta_init)(x)  # [B, n_theta]

        if self.block_type == "identity":
            backcast = theta[:, : cfg.input_size]
            forecast = theta[:, cfg.input_size :]

        elif self.block_type == "trend":
            poly_size = cfg.n_basis + 1
            bb = _get_trend_basis(cfg.input_size, poly_size, cfg.basis)
            fb = _get_trend_basis(cfg.h, poly_size, cfg.basis)
            backcast_theta = theta[:, :poly_size]
            forecast_theta = theta[:, poly_size:]
            backcast = jnp.einsum("bp,pl->bl", backcast_theta, bb)
            forecast = jnp.einsum("bp,ph->bh", forecast_theta, fb)

        elif self.block_type == "seasonality":
            bb, fb = _seasonality_basis(cfg.input_size, cfg.h, cfg.n_harmonics)
            K = bb.shape[0]
            backcast_theta = theta[:, :K]
            forecast_theta = theta[:, K:]
            backcast = jnp.einsum("bk,kl->bl", backcast_theta, bb)
            forecast = jnp.einsum("bk,kh->bh", forecast_theta, fb)

        elif self.block_type == "exogenous":
            if cfg.futr_exog_size > 0 and cfg.stat_exog_size > 0:
                stat_b = jnp.broadcast_to(
                    stat_exog[:, None, :], (B, cfg.input_size + cfg.h, cfg.stat_exog_size)
                )
                futr_aug = jnp.concatenate([futr_exog, stat_b], axis=2)
            elif cfg.futr_exog_size > 0:
                futr_aug = futr_exog
            else:  # cfg.stat_exog_size > 0 (guaranteed by the guard above)
                futr_aug = jnp.broadcast_to(
                    stat_exog[:, None, :], (B, cfg.input_size + cfg.h, cfg.stat_exog_size)
                )
            backcast, forecast = ExogenousBasis(forecast_size=cfg.h)(theta, futr_aug)

        else:
            raise ValueError(f"Unknown block type: {self.block_type!r}")

        return backcast, forecast  # [B, L], [B, h]


# ---------------------------------------------------------------------------
# Full N-BEATSx model
# ---------------------------------------------------------------------------


class NBEATSx(fnn.Module):
    """N-BEATSx with doubly-residual stacking and exogenous covariates.

    Input ``insample_y`` should already be scaled (the forecaster applies
    ``RobustScaler`` per window before calling ``model.apply``).

    Forward pass:

    1. Level init: ``forecast = mean(insample_y[:, -h//2:])``.
    2. ``residuals = flip(insample_y)`` (most-recent step first, NF
       convention). ``hist_exog`` / ``futr_exog`` / ``stat_exog`` are passed
       through **unflipped**, exactly as in NF.
    3. For each block: ``forecast += block_forecast``; ``residuals -= backcast``.
    4. Return ``forecast``.

    Args:
        config: :class:`NBEATSxConfig` specifying architecture.
    """

    config: NBEATSxConfig

    @fnn.compact
    def __call__(
        self,
        insample_y: jnp.ndarray,                       # [B, L]
        hist_exog: Optional[jnp.ndarray] = None,        # [B, L, X]
        futr_exog: Optional[jnp.ndarray] = None,        # [B, L+h, F]
        stat_exog: Optional[jnp.ndarray] = None,        # [B, S]
        insample_mask: Optional[jnp.ndarray] = None,    # [B, L] 1=valid 0=padded
        deterministic: bool = True,
    ) -> jnp.ndarray:                                    # [B, h]
        cfg = self.config
        B = insample_y.shape[0]

        if insample_mask is None:
            insample_mask = jnp.ones_like(insample_y)
        if cfg.hist_exog_size > 0 and hist_exog is None:
            hist_exog = jnp.zeros((B, cfg.input_size, cfg.hist_exog_size))
        if cfg.futr_exog_size > 0 and futr_exog is None:
            futr_exog = jnp.zeros((B, cfg.input_size + cfg.h, cfg.futr_exog_size))
        if cfg.stat_exog_size > 0 and stat_exog is None:
            stat_exog = jnp.zeros((B, cfg.stat_exog_size))

        # NF convention: only insample_y / insample_mask are time-reversed.
        residuals = jnp.flip(insample_y, axis=-1)
        mask_flip = jnp.flip(insample_mask, axis=-1)

        # Level init: mean of the last (h//2) observations (reduces noise
        # sensitivity vs. a single-point Naive1 level; n_level is a static
        # Python int so XLA sees it as a constant slice).
        n_level = max(1, cfg.h // 2)
        level = jnp.mean(insample_y[:, -n_level:], axis=-1, keepdims=True)  # [B, 1]
        forecast = jnp.zeros((B, cfg.h)) + level

        block_cache: Dict[str, NBEATSxBlock] = {}

        for i, (stack_type, n_blk) in enumerate(zip(cfg.stack_types, cfg.n_blocks)):
            for j in range(n_blk):
                if cfg.shared_weights and j > 0:
                    block = block_cache[f"{stack_type}_0"]
                else:
                    bname = f"block_{stack_type}_{j}"
                    block = NBEATSxBlock(config=cfg, block_type=stack_type, name=bname)
                    block_cache[f"{stack_type}_{j}"] = block

                backcast, forecast_i = block(
                    residuals, hist_exog, futr_exog, stat_exog, deterministic
                )
                residuals = (residuals - backcast) * mask_flip
                forecast = forecast + forecast_i

        return forecast  # [B, h]

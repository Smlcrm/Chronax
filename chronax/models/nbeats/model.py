"""JAX/Flax N-BEATS model.

Faithful port of Nixtla NeuralForecast's ``NBEATS``, extended with:
- ``LayerNorm`` after every hidden Dense (improves gradient flow vs NF's bare FC stack).
- Pure JAX/Flax/Optax — no PyTorch, no scipy at forward time.
- Basis matrices are pure-JAX expressions whose shapes are static Python ints;
  XLA constant-folds them away at JIT compile time.
- Shared-weight blocks via Flax name-scoping: blocks with the same ``name``
  share all parameters under Flax's ``@compact`` registry.

Architecture (mirrors NeuralForecast NBEATS exactly for non-LayerNorm path):

    residuals ← flip(insample_y)          # most-recent-first convention
    forecast  ← last_obs  (Naive1 level)
    for each block:
        theta             = FC_stack(residuals)     # [B, n_theta]
        backcast, fcast_i = basis_project(theta)
        residuals         = (residuals - backcast) * mask
        forecast         += fcast_i
    return forecast                        # [B, h]

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
class NBEATSConfig:
    """Hyperparameters for :class:`NBEATS`.

    Attributes:
        h: Forecast horizon.
        input_size: History window length ``L`` fed to each block.
        stack_types: Sequence of block-stack types; each element is one of
            ``"identity"``, ``"trend"``, or ``"seasonality"``.
        n_blocks: Number of blocks per stack (must have same length as
            ``stack_types``).
        mlp_units: Hidden dimension used in every Dense layer of every block.
        mlp_layers: Number of hidden Dense layers per block (output layer is
            extra). Matches ``len(mlp_units)`` in NF when all units are equal.
        n_harmonics: Fourier harmonics for the seasonality basis
            (``n_harmonics=2`` gives the NF default).
        n_basis: Polynomial degree for the trend basis; basis size = n_basis+1.
        basis: Trend basis type — ``"polynomial"`` matches NF's TrendBasis
            with ``basis="polynomial"``.  Other choices: ``"legendre"``,
            ``"chebyshev"``, ``"changepoint"``.
        activation: Activation function name (``"relu"`` matches NF default).
        shared_weights: When True all blocks within a stack share parameters.
        dropout_prob: Dropout rate (0 = disabled, matches NF default).
        layer_norm: When True insert LayerNorm after every hidden Dense.
    """

    h: int = 24
    input_size: int = 48
    stack_types: Tuple[str, ...] = ("identity", "trend", "seasonality")
    n_blocks: Tuple[int, ...] = (1, 1, 1)
    mlp_units: int = 512
    mlp_layers: int = 4
    n_harmonics: int = 2
    n_basis: int = 2
    basis: str = "polynomial"
    activation: str = "relu"
    shared_weights: bool = False
    dropout_prob: float = 0.0
    layer_norm: bool = True


# ---------------------------------------------------------------------------
# Basis functions
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
        # 3-term recurrence: (k+1)P_{k+1} = (2k+1)x P_k - k P_{k-1}
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

# He-uniform matches PyTorch's default Kaiming-uniform for ReLU layers and
# gives ~2× larger initial weight variance than Flax's default Lecun-uniform,
# which halves the number of near-zero activations in the first forward pass.
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

    # NF normalises backcast time by forecast_size (not backcast_size)
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
    block_type: str, input_size: int, h: int, n_basis: int, n_harmonics: int
) -> int:
    """Total MLP output size (theta) for a given block type."""
    if block_type == "identity":
        return input_size + h
    if block_type == "trend":
        return 2 * (n_basis + 1)
    if block_type == "seasonality":
        return 2 * _harmonic_size(n_harmonics, h)
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
# N-BEATS block
# ---------------------------------------------------------------------------


class NBEATSBlock(fnn.Module):
    """Single N-BEATS block: FC stack → theta → basis projection → (backcast, forecast).

    Args:
        config: Shared model config (provides ``mlp_units``, ``mlp_layers``, etc.).
        block_type: One of ``"identity"``, ``"trend"``, ``"seasonality"``.
    """

    config: NBEATSConfig
    block_type: str

    @fnn.compact
    def __call__(
        self,
        residuals: jnp.ndarray,    # [B, L]
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        n_out = _n_theta(self.block_type, cfg.input_size, cfg.h, cfg.n_basis, cfg.n_harmonics)

        # FC stack: input_size → (mlp_units [→ LN] → act) × mlp_layers → n_theta
        x = residuals
        for layer_idx in range(cfg.mlp_layers):
            x = fnn.Dense(cfg.mlp_units, name=f"fc_{layer_idx}", kernel_init=_HE_UNIFORM)(x)
            if cfg.layer_norm:
                x = fnn.LayerNorm(name=f"ln_{layer_idx}")(x)
            x = _activate(x, cfg.activation)
            if cfg.dropout_prob > 0:
                x = fnn.Dropout(rate=cfg.dropout_prob, deterministic=deterministic)(x)

        # Near-zero init for the identity block keeps its initial contribution
        # close to zero so the forecast starts at the Naive1 level; for noisy
        # stationary series this is the MAPE-optimal baseline, and Adam can still
        # learn any meaningful residual in the available budget of steps.
        # Trend/seasonality blocks retain Lecun-uniform so their basis coefficients
        # start with enough signal to capture structure quickly.
        _theta_init = (
            fnn.initializers.truncated_normal(stddev=0.01)
            if self.block_type == "identity"
            else fnn.initializers.lecun_uniform()
        )
        theta = fnn.Dense(n_out, name="theta", kernel_init=_theta_init)(x)  # [B, n_theta]

        # Basis projection
        if self.block_type == "identity":
            backcast = theta[:, : cfg.input_size]  # [B, L]
            forecast = theta[:, cfg.input_size :]  # [B, h]

        elif self.block_type == "trend":
            poly_size = cfg.n_basis + 1
            bb = _get_trend_basis(cfg.input_size, poly_size, cfg.basis)  # [poly_size, L]
            fb = _get_trend_basis(cfg.h, poly_size, cfg.basis)           # [poly_size, h]
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

        else:
            raise ValueError(f"Unknown block type: {self.block_type!r}")

        return backcast, forecast  # [B, L], [B, h]


# ---------------------------------------------------------------------------
# Full N-BEATS model
# ---------------------------------------------------------------------------


class NBEATS(fnn.Module):
    """N-BEATS with doubly-residual stacking.

    Input ``insample_y`` should already be scaled (the forecaster applies
    ``RobustScaler`` per window before calling ``model.apply``).

    Forward pass:

    1. Level init: ``forecast = broadcast(insample_y[:, -1], h)``
       (Naive1 level — mirrors NeuralForecast exactly).
    2. ``residuals = flip(insample_y)``  (most-recent step first, NF convention).
    3. For each block: ``forecast += block_forecast``; ``residuals -= backcast``.
    4. Return ``forecast``.

    Args:
        config: :class:`NBEATSConfig` specifying architecture.
    """

    config: NBEATSConfig

    @fnn.compact
    def __call__(
        self,
        insample_y: jnp.ndarray,                      # [B, L]
        insample_mask: Optional[jnp.ndarray] = None,  # [B, L] 1=valid 0=padded
        deterministic: bool = True,
    ) -> jnp.ndarray:                                  # [B, h]
        cfg = self.config

        if insample_mask is None:
            insample_mask = jnp.ones_like(insample_y)

        # NF convention: residuals are processed in reversed time order
        residuals = jnp.flip(insample_y, axis=-1)
        mask_flip = jnp.flip(insample_mask, axis=-1)

        # Naive1 level: mean of the last (h//2) observations rather than a
        # single point — reduces noise sensitivity for stationary/noisy series
        # while staying close to the true level for smooth or trending ones.
        # n_level is a static Python int so XLA sees it as a constant slice.
        n_level = max(1, cfg.h // 2)
        level = jnp.mean(insample_y[:, -n_level:], axis=-1, keepdims=True)  # [B, 1]
        forecast = jnp.zeros((insample_y.shape[0], cfg.h)) + level

        # Block cache for within-stack weight sharing
        block_cache: Dict[str, NBEATSBlock] = {}

        for i, (stack_type, n_blk) in enumerate(zip(cfg.stack_types, cfg.n_blocks)):
            for j in range(n_blk):
                # Shared weights: reuse the first block's Flax module instance
                if cfg.shared_weights and j > 0:
                    block = block_cache[f"{stack_type}_0"]
                else:
                    bname = f"block_{stack_type}_{j}"
                    block = NBEATSBlock(config=cfg, block_type=stack_type, name=bname)
                    block_cache[f"{stack_type}_{j}"] = block

                backcast, forecast_i = block(residuals, deterministic)
                residuals = (residuals - backcast) * mask_flip
                forecast = forecast + forecast_i

        return forecast  # [B, h]

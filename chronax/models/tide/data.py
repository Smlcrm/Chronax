"""Data pipeline for Chronax TiDE.

Two training paths are provided:

1. **Window-based** (primary path, matches NeuralForecast):
   ``build_windows`` + ``split_train_val_windows`` extract every possible
   sliding window from a 1-D series. ``RobustScaler`` applies per-window
   median/MAD normalisation inside the JIT step. This is the path used by
   :class:`~chronax.models.tide.forecaster.TiDEForecaster`.

2. **Batch-based** (exogenous path):
   ``create_batch`` assembles batches from pre-split series with optional
   hist/futr/stat exogenous arrays — useful when covariates are present.

Batch format (both paths produce this dict shape):

    insample_y      [B, L, 1]
    outsample_y     [B, h, 1]
    available_mask  [B, L, 1]   1 = real, 0 = left-padded
    sample_mask     [B, h, 1]   1 = include in loss, 0 = ignore
    hist_exog       [B, L, X]   or None
    futr_exog       [B, L+h, F] or None
    stat_exog       [B, S]      or None

No NumPy — every array is a ``jnp.ndarray``, including the sliding-window
extraction (done via a single vectorised gather).
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import jax.numpy as jnp
from jax import random


# ---------------------------------------------------------------------------
# Robust per-window scaler (pure JAX — used inside JIT-compiled train steps)
# ---------------------------------------------------------------------------

_MAD_TO_STD = 0.6744897501960817   # scipy.stats.norm.ppf(0.75), pre-computed
_EPS = 1e-6


class RobustScaler:
    """Median + MAD scaler with std-based fallback when MAD == 0.

    Stateless: ``stats`` returns ``(shift, scale)`` so the same values can be
    reused to inverse-transform predictions.

    Example::

        scaler = RobustScaler()
        shift, scale = scaler.stats(insample, axis=1)   # [B,L] -> [B,1]
        z    = scaler.transform(insample, shift, scale)
        pred = scaler.inverse(z_hat, shift, scale)
    """

    def stats(
        self, x: jnp.ndarray, axis: int = 1
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        median = jnp.median(x, axis=axis, keepdims=True)
        mad = jnp.median(jnp.abs(x - median), axis=axis, keepdims=True)
        mean = jnp.mean(x, axis=axis, keepdims=True)
        std = jnp.sqrt(jnp.mean((x - mean) ** 2, axis=axis, keepdims=True))
        fallback = std * _MAD_TO_STD
        scale = jnp.where(mad == 0.0, fallback, mad)
        scale = jnp.where(scale == 0.0, 1.0, scale) + _EPS
        return median, scale

    def transform(
        self, x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray
    ) -> jnp.ndarray:
        return (x - shift) / scale

    def inverse(
        self, z: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray
    ) -> jnp.ndarray:
        return z * scale + shift


# ---------------------------------------------------------------------------
# Sliding window extraction (single vectorised gather)
# ---------------------------------------------------------------------------


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> jnp.ndarray:
    """Return ``[n_windows, input_size + h]`` rolling windows (stride = 1).

    Args:
        y: 1-D float32 series of length ``T``.
        input_size: History window length ``L``.
        h: Forecast horizon.

    Raises:
        ValueError: When ``T < input_size + h``.
    """
    y = jnp.asarray(y, dtype=jnp.float32).ravel()
    win = input_size + h
    n = y.shape[0] - win + 1
    if n <= 0:
        raise ValueError(
            f"Series length {y.shape[0]} is too short for "
            f"input_size={input_size}, h={h} (need at least {win})."
        )
    idx = jnp.arange(win)[None, :] + jnp.arange(n)[:, None]
    return y[idx]  # [n, win]


def split_train_val_windows(
    windows: jnp.ndarray, *, val_fraction: float = 0.1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Chronological split: earliest windows train, latest windows validate.

    When ``val_fraction <= 0`` all windows go to the training set.
    """
    n = windows.shape[0]
    if n < 1:
        raise ValueError("windows must be non-empty.")
    if val_fraction <= 0.0 or n < 2:
        return windows, windows[:0]
    n_val = max(1, int(n * val_fraction))
    n_train = max(1, n - n_val)
    return windows[:n_train], windows[n_train:]


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------


def pad_sequence(
    y: jnp.ndarray,
    target_len: int,
    pad_value: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Left-pad / left-truncate a 1-D series to ``target_len``.

    Returns ``(padded, mask)`` where mask is 1 on real observations.
    """
    T = y.shape[0]
    if T >= target_len:
        return y[-target_len:].astype(jnp.float32), jnp.ones(target_len, dtype=jnp.float32)
    pad_width = target_len - T
    padded = jnp.concatenate(
        [jnp.full(pad_width, pad_value, dtype=jnp.float32), y.astype(jnp.float32)]
    )
    mask = jnp.concatenate(
        [jnp.zeros(pad_width, dtype=jnp.float32), jnp.ones(T, dtype=jnp.float32)]
    )
    return padded, mask


def _pad_2d_left(arr: jnp.ndarray, target_len: int) -> jnp.ndarray:
    """Left-pad a ``[T, F]`` array on the time axis to ``target_len``."""
    T = arr.shape[0]
    if T >= target_len:
        return arr[-target_len:].astype(jnp.float32)
    pad_width = target_len - T
    return jnp.pad(
        arr.astype(jnp.float32),
        ((pad_width, 0), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )


# ---------------------------------------------------------------------------
# Covariate alignment
# ---------------------------------------------------------------------------


def align_covariates(
    hist_exog: Optional[jnp.ndarray],
    futr_exog: Optional[jnp.ndarray],
    stat_exog: Optional[jnp.ndarray],
    input_size: int,
    h: int,
) -> Dict[str, Optional[jnp.ndarray]]:
    """Align per-series exogenous arrays to model input expectations.

    ``hist_exog`` → left-padded to ``[input_size, X]``.
    ``futr_exog`` → left-padded to ``[input_size + h, F]``.
    ``stat_exog`` → cast to float32, returned as-is.
    """
    out: Dict[str, Optional[jnp.ndarray]] = {
        "hist_exog": None,
        "futr_exog": None,
        "stat_exog": None,
    }
    if hist_exog is not None:
        out["hist_exog"] = _pad_2d_left(hist_exog, input_size)
    if futr_exog is not None:
        out["futr_exog"] = _pad_2d_left(futr_exog, input_size + h)
    if stat_exog is not None:
        out["stat_exog"] = stat_exog.astype(jnp.float32)
    return out


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


def _split_y(y: jnp.ndarray, h: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Split a 1-D series into ``(history, horizon)`` of length ``[*, h]``."""
    T = y.shape[0]
    if T > h:
        return y[:-h], y[-h:].astype(jnp.float32)
    if T >= h:
        return y[:1], y[-h:].astype(jnp.float32)
    pad_width = h - T
    y_out = jnp.pad(y.astype(jnp.float32), (pad_width, 0), constant_values=0.0)
    return y[:1], y_out


def create_batch(
    y_series: List[jnp.ndarray],
    input_size: int,
    h: int,
    hist_exog_list: Optional[List[Optional[jnp.ndarray]]] = None,
    futr_exog_list: Optional[List[Optional[jnp.ndarray]]] = None,
    stat_exog_list: Optional[List[Optional[jnp.ndarray]]] = None,
    sample_mask_list: Optional[List[Optional[jnp.ndarray]]] = None,
) -> Dict[str, Optional[jnp.ndarray]]:
    """Build a single JAX batch from a list of complete time series.

    Each series is split at the ``-h`` cut point. History is left-padded to
    ``input_size`` (real observations tracked by ``available_mask``); horizon
    is padded to ``h`` (loss tracked by ``sample_mask``).

    Returns:
        Dict with keys::

            insample_y      [B, L, 1]
            outsample_y     [B, h, 1]
            available_mask  [B, L, 1]
            sample_mask     [B, h, 1]
            hist_exog       [B, L, X]  or None
            futr_exog       [B, L+h, F] or None
            stat_exog       [B, S]     or None
    """
    insample_ys, outsample_ys = [], []
    available_masks, sample_masks = [], []

    for i, y in enumerate(y_series):
        y_hist, y_out = _split_y(y, h)
        padded_hist, hist_mask = pad_sequence(y_hist, input_size)
        insample_ys.append(padded_hist)
        available_masks.append(hist_mask)
        outsample_ys.append(y_out)

        if sample_mask_list is not None and sample_mask_list[i] is not None:
            user_mask = sample_mask_list[i].astype(jnp.float32)
            if user_mask.shape[0] != h:
                _, user_mask = pad_sequence(user_mask, h, pad_value=0.0)
            sample_masks.append(user_mask)
        else:
            sample_masks.append(jnp.ones(h, dtype=jnp.float32))

    batch: Dict[str, Optional[jnp.ndarray]] = {
        "insample_y":     jnp.stack(insample_ys)[:, :, None],      # [B, L, 1]
        "outsample_y":    jnp.stack(outsample_ys)[:, :, None],     # [B, h, 1]
        "available_mask": jnp.stack(available_masks)[:, :, None],  # [B, L, 1]
        "sample_mask":    jnp.stack(sample_masks)[:, :, None],     # [B, h, 1]
    }

    def _has(lst: Optional[List]) -> bool:
        return lst is not None and any(x is not None for x in lst)

    def _feat_dim(lst: List, default: int = 1) -> int:
        return next((x.shape[-1] for x in lst if x is not None), default)

    if _has(hist_exog_list):
        F = _feat_dim(hist_exog_list)
        batch["hist_exog"] = jnp.stack([
            _pad_2d_left(x, input_size) if x is not None
            else jnp.zeros((input_size, F), dtype=jnp.float32)
            for x in hist_exog_list
        ])
    else:
        batch["hist_exog"] = None

    if _has(futr_exog_list):
        F = _feat_dim(futr_exog_list)
        batch["futr_exog"] = jnp.stack([
            _pad_2d_left(x, input_size + h) if x is not None
            else jnp.zeros((input_size + h, F), dtype=jnp.float32)
            for x in futr_exog_list
        ])
    else:
        batch["futr_exog"] = None

    if _has(stat_exog_list):
        F = _feat_dim(stat_exog_list)
        batch["stat_exog"] = jnp.stack([
            x.astype(jnp.float32) if x is not None
            else jnp.zeros(F, dtype=jnp.float32)
            for x in stat_exog_list
        ])
    else:
        batch["stat_exog"] = None

    return batch


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------


def batch_generator(
    y_series: List[jnp.ndarray],
    input_size: int,
    h: int,
    batch_size: int,
    hist_exog_list: Optional[List[Optional[jnp.ndarray]]] = None,
    futr_exog_list: Optional[List[Optional[jnp.ndarray]]] = None,
    stat_exog_list: Optional[List[Optional[jnp.ndarray]]] = None,
    sample_mask_list: Optional[List[Optional[jnp.ndarray]]] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Dict[str, Optional[jnp.ndarray]]]:
    """Iterate over ``y_series`` yielding fully-prepared JAX batches."""
    n = len(y_series)
    indices = jnp.arange(n)
    if shuffle:
        indices = random.permutation(random.PRNGKey(seed), indices)

    for start in range(0, n, batch_size):
        idx = indices[start: start + batch_size]
        sub = lambda lst: ([lst[int(i)] for i in idx] if lst is not None else None)
        yield create_batch(
            y_series=[y_series[int(i)] for i in idx],
            input_size=input_size,
            h=h,
            hist_exog_list=sub(hist_exog_list),
            futr_exog_list=sub(futr_exog_list),
            stat_exog_list=sub(stat_exog_list),
            sample_mask_list=sub(sample_mask_list),
        )

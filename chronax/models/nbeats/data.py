"""Data pipeline for Chronax N-BEATS.

Three concerns are separated:

1. ``RobustScaler``          — per-window median/MAD normalisation (pure JAX).
2. Window building           — ``build_windows`` / ``split_train_val_windows``
                               extract sliding windows from a 1-D series.
3. Batch construction        — ``create_batch`` assembles JAX batches for the
                               masked-loss training path (mirrors RNN data.py).

The window-based path (``build_windows`` + ``RobustScaler``) is the primary
path used by ``NBEATSForecaster``; it applies per-window scaling inside the
``train_step`` JIT kernel, so no global statistics need to be tracked.

No NumPy — every array is a ``jnp.ndarray``, including the sliding-window
extraction (done via a single vectorised gather).
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import jax.numpy as jnp
from jax import random


# ---------------------------------------------------------------------------
# Robust per-window scaler (pure JAX — operates on device arrays)
# ---------------------------------------------------------------------------

_MAD_TO_STD = 0.6744897501960817  # scipy.stats.norm.ppf(0.75), pre-computed
_EPS = 1e-6


class RobustScaler:
    """Median + MAD scaler with std-based fallback when MAD == 0.

    Stateless: ``stats`` computes ``(shift, scale)`` so the same statistics
    can be reused to inverse-transform predictions.

    Example::

        scaler = RobustScaler()
        shift, scale = scaler.stats(insample, axis=1)   # [B,L] -> [B,1]
        z    = scaler.transform(insample, shift, scale)  # [B,L]
        pred = scaler.inverse(z_hat, shift, scale)       # [B,h]
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


def _split_y(y: jnp.ndarray, h: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    T = y.shape[0]
    if T > h:
        return y[:-h], y[-h:].astype(jnp.float32)
    if T >= h:
        return y[:1], y[-h:].astype(jnp.float32)
    pad_width = h - T
    y_out = jnp.pad(y.astype(jnp.float32), (pad_width, 0), constant_values=0.0)
    return y[:1], y_out


# ---------------------------------------------------------------------------
# Batch construction (masked-loss path, mirrors RNN / Autoformer data.py)
# ---------------------------------------------------------------------------


def create_batch(
    y_series: List[jnp.ndarray],
    input_size: int,
    h: int,
    sample_mask_list: Optional[List[Optional[jnp.ndarray]]] = None,
) -> Dict[str, Optional[jnp.ndarray]]:
    """Build a JAX batch from a list of complete time series.

    Each series is split into ``(history, horizon)``. History is
    padded/truncated to ``input_size`` (yielding ``available_mask``) and
    horizon is padded/truncated to ``h`` (yielding ``sample_mask``).

    Returns a dict with JAX arrays::

        insample_y      [B, input_size]   (2-D, no trailing feature dim)
        outsample_y     [B, h]
        available_mask  [B, input_size]   1 = real, 0 = padded
        sample_mask     [B, h]            1 = include in loss, 0 = ignore
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

    return {
        "insample_y": jnp.stack(insample_ys),          # [B, L]
        "outsample_y": jnp.stack(outsample_ys),         # [B, h]
        "available_mask": jnp.stack(available_masks),   # [B, L]
        "sample_mask": jnp.stack(sample_masks),         # [B, h]
    }


# ---------------------------------------------------------------------------
# Batch generator (masked-loss path)
# ---------------------------------------------------------------------------


def batch_generator(
    y_series: List[jnp.ndarray],
    input_size: int,
    h: int,
    batch_size: int,
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
        sub_mask = (
            [sample_mask_list[int(i)] for i in idx]
            if sample_mask_list is not None
            else None
        )
        yield create_batch(
            y_series=[y_series[int(i)] for i in idx],
            input_size=input_size,
            h=h,
            sample_mask_list=sub_mask,
        )

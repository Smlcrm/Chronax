"""Data pipeline and scaling utilities for the Chronax FEDformer model.

Self-contained -- no imports from other chronax models. Three concerns:

1. ``RobustScaler`` -- per-window median/MAD normalisation (pure JAX; runs on
   device arrays inside the training and inference loops).
2. Window building -- ``build_windows`` / ``split_train_val_windows`` slice a
   1-D numpy series into the sliding windows the window-based loop consumes.
3. Batch construction -- ``pad_sequence`` / ``create_batch`` / ``batch_generator``
   produce padded, masked JAX batches for the masked-loss training paradigm.

WHY scale per window?
    Long series can drift in level and variance. Normalising each window by its
    own median and spread makes the model's job *stationary* across windows; we
    invert the transform on the predictions to return to the original units.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Robust per-window scaler (pure JAX -- used on device arrays)
# ---------------------------------------------------------------------------

# Median Absolute Deviation -> standard-deviation conversion constant
# (scipy.stats.norm.ppf(0.75)); lets the MAD-based scale match a Gaussian std.
_MAD_TO_STD = 0.6744897501960817
_EPS = 1e-6


def _torch_median(x: jnp.ndarray, axis: int = 1, keepdims: bool = True) -> jnp.ndarray:
    """Median matching ``torch.median`` / ``nanmedian`` (lower middle on even n).

    ``jnp.median`` averages the two central elements for even lengths; PyTorch
    takes index ``(n-1)//2`` after sorting. RobustScaler windows use even
    ``input_size`` (72), so this matters for NF parity.
    """
    x_sorted = jnp.sort(x, axis=axis)
    idx = (x.shape[axis] - 1) // 2
    med = jnp.take(x_sorted, idx, axis=axis)
    if keepdims:
        med = jnp.expand_dims(med, axis=axis)
    return med


class RobustScaler:
    """Median + MAD scaler with a std-based fallback when MAD == 0.

    Stateless: ``stats`` computes ``(shift, scale)`` so the *same* statistics can
    later invert predictions. Robust statistics (median/MAD) are preferred over
    mean/std because they are insensitive to the spikes common in real series.

    Example::

        scaler = RobustScaler()
        shift, scale = scaler.stats(x, axis=1)   # [B, L] -> [B, 1]
        z = scaler.transform(x, shift, scale)
        x_again = scaler.inverse(z, shift, scale)
    """

    def stats(
        self, x: jnp.ndarray, axis: int = 1
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(shift, scale)`` computed along ``axis`` (keepdims).

        ``shift`` is the median; ``scale`` is the MAD, falling back to a
        std-derived value when the MAD is zero (e.g. a near-constant window),
        and finally to 1 to avoid division by zero.
        """
        median = _torch_median(x, axis=axis, keepdims=True)
        mad = _torch_median(jnp.abs(x - median), axis=axis, keepdims=True)
        mean = jnp.mean(x, axis=axis, keepdims=True)
        std = jnp.sqrt(jnp.mean((x - mean) ** 2, axis=axis, keepdims=True))
        fallback = std * _MAD_TO_STD
        scale = jnp.where(mad == 0.0, fallback, mad)
        scale = jnp.where(scale == 0.0, 1.0, scale) + _EPS
        return median, scale

    def transform(
        self, x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray
    ) -> jnp.ndarray:
        """Normalise: ``(x - shift) / scale``."""
        return (x - shift) / scale

    def inverse(
        self, z: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray
    ) -> jnp.ndarray:
        """Invert the normalisation: ``z * scale + shift``."""
        return z * scale + shift


# ---------------------------------------------------------------------------
# Window building (numpy -- host-side preprocessing)
# ---------------------------------------------------------------------------


def build_exog_windows(
    arr: np.ndarray | jnp.ndarray,
    input_size: int,
    h: int,
    n_windows: int,
    span: str = "full",
) -> jnp.ndarray:
    """Rolling windows of an exog array ``[T, F]``.

    Matches :func:`build_windows` right-padding of ``h`` zeros when
    ``span="full"``, yielding ``[n_windows, input_size+h, F]``.
    ``span="input"`` yields ``[n_windows, input_size, F]`` without pad.
    """
    arr = jnp.asarray(arr, dtype=jnp.float32)
    if arr.ndim != 2:
        raise ValueError(f"exog array must be 2-D [T, F]; got shape {arr.shape}.")
    length = input_size if span == "input" else input_size + h
    if span == "full":
        pad = jnp.zeros((h, arr.shape[1]), dtype=arr.dtype)
        arr = jnp.concatenate([arr, pad], axis=0)
    idx = jnp.arange(length)[None, :] + jnp.arange(n_windows)[:, None]
    return arr[idx]


def build_windows(
    y: np.ndarray, input_size: int, h: int
) -> Tuple[np.ndarray, np.ndarray]:
    """NF-style rolling windows with right-padding (``ConstantPad1d((0, h))``).

    Right-pads ``y`` with ``h`` zeros before unfolding, yielding
    ``len(y) - input_size`` windows of length ``input_size + h`` — including
    partial-horizon windows whose context reaches the end of the series.
    Returns ``(windows, mask)`` where ``mask`` is 1 on real points and 0 on the
    padded tail so the loss can drop padded horizon steps.

    Raises:
        ValueError: When ``T < input_size + 1`` (no window with a real outsample).
    """
    y = np.asarray(y, dtype=np.float32).ravel()
    n_real = len(y)
    n = n_real - input_size  # NF keeps windows with >=1 valid outsample
    if n < 1:
        raise ValueError(
            f"Series length {n_real} is too short for "
            f"input_size={input_size}, h={h} (need at least {input_size + 1})."
        )
    window_size = input_size + h
    y_pad = np.concatenate([y, np.zeros(h, dtype=np.float32)])
    avail = np.concatenate([np.ones(n_real, dtype=np.float32), np.zeros(h, dtype=np.float32)])
    idx = np.arange(window_size)[None, :] + np.arange(n)[:, None]
    return y_pad[idx], avail[idx]


def split_train_val_windows(
    windows: np.ndarray,
    masks: np.ndarray,
    *,
    val_fraction: float = 0.1,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Chronological split: earliest windows train, latest validate.

    A time-ordered split (no shuffling) avoids leaking future information into
    the training set. When ``val_fraction <= 0`` all windows train (empty val).
    Returns ``((train_windows, train_masks), (val_windows, val_masks))``.
    """
    n = len(windows)
    if n < 1:
        raise ValueError("windows must be non-empty.")
    if val_fraction <= 0.0 or n < 2:
        empty_w, empty_m = windows[:0], masks[:0]
        return (windows, masks), (empty_w, empty_m)
    n_val = max(1, int(n * val_fraction))
    n_train = max(1, n - n_val)
    return (
        (windows[:n_train], masks[:n_train]),
        (windows[n_train:], masks[n_train:]),
    )


# ---------------------------------------------------------------------------
# Padding helpers (for batch-based, masked-loss training)
# ---------------------------------------------------------------------------


def pad_sequence(
    y: np.ndarray,
    target_len: int,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Left-pad / left-truncate a 1-D series to ``target_len``.

    Returns ``(padded, mask)`` where ``mask`` is 1 on real observations and 0 on
    padding -- the mask is what the masked losses use to ignore padded steps.
    """
    T = len(y)
    if T >= target_len:
        return y[-target_len:].astype(np.float32), np.ones(target_len, dtype=np.float32)
    pad_width = target_len - T
    padded = np.concatenate(
        [np.full(pad_width, pad_value, dtype=np.float32), y.astype(np.float32)]
    )
    mask = np.concatenate(
        [np.zeros(pad_width, dtype=np.float32), np.ones(T, dtype=np.float32)]
    )
    return padded, mask


def _split_y(y: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split a full series into ``(history, last-h horizon)`` with guards."""
    if len(y) > h:
        return y[:-h], y[-h:].astype(np.float32)
    if len(y) >= h:
        return y[:1], y[-h:].astype(np.float32)
    pad_width = h - len(y)
    y_out = np.pad(y.astype(np.float32), (pad_width, 0), constant_values=0.0)
    return y[:1], y_out


def create_batch(
    y_series: List[np.ndarray],
    input_size: int,
    h: int,
    sample_mask_list: Optional[List[Optional[np.ndarray]]] = None,
) -> Dict[str, Optional[jnp.ndarray]]:
    """Build a single JAX batch from a list of complete time series.

    Each series is split into ``(history, horizon)``; the history is
    padded/truncated to ``input_size`` (yielding ``available_mask``) and the
    horizon to ``h`` (yielding ``sample_mask``).

    Returns a dict of JAX arrays::

        insample_y     [B, input_size, 1]
        outsample_y    [B, h, 1]
        available_mask [B, input_size, 1]   1 = real, 0 = padded
        sample_mask    [B, h, 1]            1 = compute loss, 0 = ignore
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
            user_mask = sample_mask_list[i].astype(np.float32)
            if user_mask.shape[0] != h:
                _, user_mask = pad_sequence(user_mask, h, pad_value=0.0)
            sample_masks.append(user_mask)
        else:
            sample_masks.append(np.ones(h, dtype=np.float32))

    return {
        "insample_y": jnp.array(np.stack(insample_ys)[:, :, None]),
        "outsample_y": jnp.array(np.stack(outsample_ys)[:, :, None]),
        "available_mask": jnp.array(np.stack(available_masks)[:, :, None]),
        "sample_mask": jnp.array(np.stack(sample_masks)[:, :, None]),
    }


def batch_generator(
    y_series: List[np.ndarray],
    input_size: int,
    h: int,
    batch_size: int,
    sample_mask_list: Optional[List[Optional[np.ndarray]]] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Dict[str, Optional[jnp.ndarray]]]:
    """Iterate over ``y_series`` yielding fully-prepared JAX batches."""
    n = len(y_series)
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        sub_mask = (
            [sample_mask_list[i] for i in idx]
            if sample_mask_list is not None
            else None
        )
        yield create_batch(
            y_series=[y_series[i] for i in idx],
            input_size=input_size,
            h=h,
            sample_mask_list=sub_mask,
        )

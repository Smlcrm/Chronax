"""Functional data pipeline for the Chronax TSMixer model.

TSMixer is a multivariate model — all N series are processed jointly in a
single [B, L, N] window. The pipeline is therefore organised around a single
2D array [T, N] rather than a list of univariate series as in the RNN module.

Main entry points:
    :func:`create_windows`   Extract all valid sliding windows from [T, N].
    :func:`make_batch`       Build a JAX batch from pre-extracted windows.
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np


def create_windows(
    y: np.ndarray,
    input_size: int,
    h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract all valid sliding windows from a ``[T, N]`` multivariate series.

    If the series is shorter than ``input_size + h``, the history is zero-
    padded on the left and a single window is returned.

    Args:
        y:          ``[T, N]`` float array (or 1-D for univariate).
        input_size: history window length ``L``.
        h:          forecast horizon.

    Returns:
        insample  ``[W, L, N]`` float32 — history windows.
        outsample ``[W, h, N]`` float32 — corresponding horizon targets.
    """
    if y.ndim == 1:
        y = y[:, None]
    y = y.astype(np.float32)
    T, N = y.shape
    total = input_size + h

    if T < total:
        pad = np.zeros((total - T, N), dtype=np.float32)
        y_padded = np.vstack([pad, y])
        insample  = y_padded[:input_size][None]   # [1, L, N]
        outsample = y_padded[input_size:][None]   # [1, h, N]
    else:
        n_windows = T - total + 1
        # Stride tricks: O(1) view construction, O(W*L*N) copy — avoids Python loop.
        st = y.strides  # (bytes_per_row, bytes_per_col)
        win_strides = (st[0], st[0], st[1])
        insample = np.array(
            np.lib.stride_tricks.as_strided(y, shape=(n_windows, input_size, N), strides=win_strides),
            dtype=np.float32,
        )
        outsample = np.array(
            np.lib.stride_tricks.as_strided(y[input_size:], shape=(n_windows, h, N), strides=win_strides),
            dtype=np.float32,
        )

    return insample.astype(np.float32), outsample.astype(np.float32)


def make_batch(
    insample: np.ndarray,
    outsample: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Build a JAX batch dict from pre-extracted window arrays.

    This is a thin indexing wrapper intended for use inside the training loop
    where the full window cache has been transferred to device once and each
    step only selects a random subset.

    Args:
        insample:  ``[W, L, N]`` all history windows (host or device numpy).
        outsample: ``[W, h, N]`` all horizon windows.
        indices:   ``[B]`` integer indices selecting windows for this batch.

    Returns:
        ``{"insample_y": [B, L, N], "outsample_y": [B, h, N],
           "sample_mask": [B, h, N]}`` with all-ones mask.
    """
    ins = insample[indices]
    out = outsample[indices]
    return {
        "insample_y":   jnp.array(ins),
        "outsample_y":  jnp.array(out),
        "sample_mask":  jnp.ones_like(jnp.array(out)),
    }

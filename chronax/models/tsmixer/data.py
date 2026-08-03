"""Functional data pipeline for the Chronax TSMixer model.

TSMixer is a multivariate model — all N series are processed jointly in a
single [B, L, N] window. The pipeline is therefore organised around a single
2D array [T, N] rather than a list of univariate series as in the RNN module.

Main entry points:
    :func:`create_windows`   Extract all valid sliding windows from [T, N].
    :func:`make_batch`       Build a JAX batch from pre-extracted windows.

No NumPy — every array is a ``jnp.ndarray``. The sliding-window extraction
uses a single vectorised gather rather than NumPy's stride-trick view, since
JAX arrays don't support raw strided views.
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp


def create_windows(
    y: jnp.ndarray,
    input_size: int,
    h: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    y = y.astype(jnp.float32)
    T, N = y.shape
    total = input_size + h

    if T < total:
        pad = jnp.zeros((total - T, N), dtype=jnp.float32)
        y_padded = jnp.concatenate([pad, y], axis=0)
        insample  = y_padded[:input_size][None]   # [1, L, N]
        outsample = y_padded[input_size:][None]   # [1, h, N]
    else:
        n_windows = T - total + 1
        # Single vectorised gather: window i covers rows [i, i+input_size) /
        # [input_size+i, input_size+i+h). O(W*L*N) / O(W*h*N) gathers, no
        # Python loop and no NumPy stride tricks.
        in_idx = jnp.arange(input_size)[None, :] + jnp.arange(n_windows)[:, None]
        out_idx = input_size + jnp.arange(h)[None, :] + jnp.arange(n_windows)[:, None]
        insample = y[in_idx]    # [n_windows, L, N]
        outsample = y[out_idx]  # [n_windows, h, N]

    return insample.astype(jnp.float32), outsample.astype(jnp.float32)


def make_batch(
    insample: jnp.ndarray,
    outsample: jnp.ndarray,
    indices: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Build a JAX batch dict from pre-extracted window arrays.

    This is a thin indexing wrapper intended for use inside the training loop
    where the full window cache has been transferred to device once and each
    step only selects a random subset.

    Args:
        insample:  ``[W, L, N]`` all history windows (device array).
        outsample: ``[W, h, N]`` all horizon windows.
        indices:   ``[B]`` integer indices selecting windows for this batch.

    Returns:
        ``{"insample_y": [B, L, N], "outsample_y": [B, h, N],
           "sample_mask": [B, h, N]}`` with all-ones mask.
    """
    ins = insample[indices]
    out = outsample[indices]
    return {
        "insample_y":   ins,
        "outsample_y":  out,
        "sample_mask":  jnp.ones_like(out),
    }

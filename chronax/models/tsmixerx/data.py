"""Functional data pipeline for the Chronax TSMixerx model.

TSMixerx is a multivariate model — all N series are processed jointly. Two
paths are provided:

1. **Window-based, no exogenous covariates** (primary path, matches
   Chronax TSMixer): :func:`create_windows` extracts every valid sliding
   window from a single ``[T, N]`` array. Used by
   :class:`~chronax.models.tsmixerx.forecaster.TSMixerxForecaster`.

2. **Batch-based, with exogenous covariates**: :func:`create_windows_exog`
   additionally windows ``hist_exog`` / ``futr_exog`` arrays alongside ``y``,
   producing tensors shaped for :class:`~chronax.models.tsmixerx.model.TSMixerx`
   directly (``[W, X, L, N]`` / ``[W, F, L+h, N]``). ``stat_exog`` is static
   per-series ([N, S]) and is not windowed.

No NumPy — every array is a ``jnp.ndarray``. The sliding-window extraction
uses a single vectorised gather rather than NumPy's stride-trick view, since
JAX arrays don't support raw strided views.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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
        insample = y_padded[:input_size][None]   # [1, L, N]
        outsample = y_padded[input_size:][None]  # [1, h, N]
    else:
        n_windows = T - total + 1
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
    """Build a JAX batch dict from pre-extracted window arrays (no exog).

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
        "insample_y":  ins,
        "outsample_y": out,
        "sample_mask": jnp.ones_like(out),
    }


# ---------------------------------------------------------------------------
# Exogenous windowing
# ---------------------------------------------------------------------------


def _pad_time_left(arr: jnp.ndarray, target_len: int) -> jnp.ndarray:
    """Left-pad a ``[T, C, N]`` array on the time axis to ``target_len``."""
    T = arr.shape[0]
    if T >= target_len:
        return arr[-target_len:].astype(jnp.float32)
    pad_width = target_len - T
    return jnp.pad(
        arr.astype(jnp.float32),
        ((pad_width, 0), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )


def create_windows_exog(
    y: jnp.ndarray,
    input_size: int,
    h: int,
    hist_exog: Optional[jnp.ndarray] = None,   # [T, X, N]
    futr_exog: Optional[jnp.ndarray] = None,   # [T, F, N]
) -> Dict[str, Optional[jnp.ndarray]]:
    """Extract aligned sliding windows for ``y`` plus optional exogenous arrays.

    Args:
        y: ``[T, N]`` multivariate target series.
        input_size: history window length ``L``.
        h: forecast horizon.
        hist_exog: ``[T, X, N]`` historic exogenous features, or None.
        futr_exog: ``[T, F, N]`` future exogenous features (must span the
            full series — the ``[L+h]``-length windows are extracted from
            it directly), or None.

    Returns:
        Dict with keys:
            insample_y  [W, L, N]
            outsample_y [W, h, N]
            hist_exog   [W, X, L, N]    or None
            futr_exog   [W, F, L+h, N]  or None

    Raises:
        ValueError: If ``y`` is shorter than ``input_size + h`` (unlike
            :func:`create_windows`, exogenous alignment requires an exact
            sliding-window grid, so short series are not auto-padded here).
    """
    if y.ndim == 1:
        y = y[:, None]
    y = y.astype(jnp.float32)
    T, N = y.shape
    total = input_size + h
    n_windows = T - total + 1
    if n_windows <= 0:
        raise ValueError(
            f"Series length {T} is too short for input_size={input_size}, "
            f"h={h} (need at least {total})."
        )

    in_idx = jnp.arange(input_size)[None, :] + jnp.arange(n_windows)[:, None]     # [W, L]
    full_idx = jnp.arange(total)[None, :] + jnp.arange(n_windows)[:, None]         # [W, L+h]
    out_idx = input_size + jnp.arange(h)[None, :] + jnp.arange(n_windows)[:, None]  # [W, h]

    out: Dict[str, Optional[jnp.ndarray]] = {
        "insample_y":  y[in_idx],                       # [W, L, N]
        "outsample_y": y[out_idx],                       # [W, h, N]
    }

    if hist_exog is not None:
        hist_exog = hist_exog.astype(jnp.float32)
        windowed = hist_exog[in_idx]                     # [W, L, X, N]
        out["hist_exog"] = jnp.transpose(windowed, (0, 2, 1, 3))  # [W, X, L, N]
    else:
        out["hist_exog"] = None

    if futr_exog is not None:
        futr_exog = futr_exog.astype(jnp.float32)
        windowed = futr_exog[full_idx]                   # [W, L+h, F, N]
        out["futr_exog"] = jnp.transpose(windowed, (0, 2, 1, 3))  # [W, F, L+h, N]
    else:
        out["futr_exog"] = None

    return out


def make_batch_exog(
    windows: Dict[str, Optional[jnp.ndarray]],
    indices: jnp.ndarray,
    stat_exog: Optional[jnp.ndarray] = None,   # [N, S], static — never windowed
) -> Dict[str, Optional[jnp.ndarray]]:
    """Select a batch of pre-extracted (possibly exogenous) windows.

    Args:
        windows: Output of :func:`create_windows_exog`.
        indices: ``[B]`` integer indices selecting windows for this batch.
        stat_exog: ``[N, S]`` static exogenous, passed through unchanged
            (identical for every window / batch).

    Returns:
        Dict with ``insample_y``, ``outsample_y``, ``sample_mask``, and
        (when present) ``hist_exog`` / ``futr_exog`` / ``stat_exog``.
    """
    ins = windows["insample_y"][indices]
    out = windows["outsample_y"][indices]
    batch: Dict[str, Optional[jnp.ndarray]] = {
        "insample_y":  ins,
        "outsample_y": out,
        "sample_mask": jnp.ones_like(out),
    }
    batch["hist_exog"] = (
        windows["hist_exog"][indices] if windows.get("hist_exog") is not None else None
    )
    batch["futr_exog"] = (
        windows["futr_exog"][indices] if windows.get("futr_exog") is not None else None
    )
    batch["stat_exog"] = stat_exog.astype(jnp.float32) if stat_exog is not None else None
    return batch

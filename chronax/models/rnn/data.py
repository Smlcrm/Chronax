"""Functional data pipeline for the Chronax RNN model.

The pipeline is intentionally minimal — no PyTorch / Lightning / DataLoader
dependencies. It produces ``jnp.ndarray`` batches that are JIT-ready and
shape-compatible with :class:`chronax.models.rnn.model.RNN`.

The two masks come from Nixtla's vocabulary:

* ``available_mask`` — 1 where the *historic* target is real data, 0 where it
  is padding. Used by the model to ignore padded steps when needed and by the
  evaluator to know what was actually provided.
* ``sample_mask``    — 1 on horizon steps that should contribute to the loss
  (e.g. drop holidays / outliers). Produced from the user's per-series
  outsample mask, defaults to all-ones.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------


def pad_sequence(
    y: np.ndarray,
    target_len: int,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Left-pad / left-truncate a 1D series to ``target_len``.

    The mask is 1 on real observations and 0 on the padded prefix, matching
    ``available_mask`` semantics.
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


def _pad_2d_left(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Left-pad a 2D ``[T, F]`` array on the time axis."""
    T = arr.shape[0]
    if T >= target_len:
        return arr[-target_len:].astype(np.float32)
    pad_width = target_len - T
    return np.pad(
        arr.astype(np.float32),
        ((pad_width, 0), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )


# ---------------------------------------------------------------------------
# Covariate alignment
# ---------------------------------------------------------------------------


def align_covariates(
    hist_exog: Optional[np.ndarray],
    futr_exog: Optional[np.ndarray],
    stat_exog: Optional[np.ndarray],
    input_size: int,
    h: int,
) -> Dict[str, Optional[np.ndarray]]:
    """Align per-series exogenous arrays to model expectations.

    ``hist_exog`` is left-padded to length ``input_size``. ``futr_exog`` must
    cover both history and the forecast horizon, so it is left-padded to
    ``input_size + h``. ``stat_exog`` is returned as-is (cast to float32).
    """
    out: Dict[str, Optional[np.ndarray]] = {
        "hist_exog": None,
        "futr_exog": None,
        "stat_exog": None,
    }
    if hist_exog is not None:
        out["hist_exog"] = _pad_2d_left(hist_exog, input_size)
    if futr_exog is not None:
        out["futr_exog"] = _pad_2d_left(futr_exog, input_size + h)
    if stat_exog is not None:
        out["stat_exog"] = stat_exog.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


def _split_y(y: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split a single series into ``(history, horizon)``.

    Pads as needed so we always return ``(y_hist, y_out)`` where
    ``len(y_out) == h``.
    """
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
    hist_exog_list: Optional[List[Optional[np.ndarray]]] = None,
    futr_exog_list: Optional[List[Optional[np.ndarray]]] = None,
    stat_exog_list: Optional[List[Optional[np.ndarray]]] = None,
    sample_mask_list: Optional[List[Optional[np.ndarray]]] = None,
) -> Dict[str, Optional[jnp.ndarray]]:
    """Build a single JAX batch from a list of complete time series.

    Every series is split into ``(history, horizon)``; the history is
    padded/truncated to ``input_size`` (yielding ``available_mask``) and the
    horizon is padded/truncated to ``h`` (yielding ``sample_mask``, optionally
    AND-ed with the user-supplied per-series mask).

    Args:
        y_series: list of 1D series, each of length ``T_i`` (the model uses
            the last ``h`` of each as the outsample target).
        input_size: history window length ``L``.
        h: forecast horizon.
        hist_exog_list: per-series ``[T_i, X]`` historic exog, or None.
        futr_exog_list: per-series ``[T_i + h, F]`` future exog, or None.
        stat_exog_list: per-series ``[S]`` static exog, or None.
        sample_mask_list: per-series ``[h]`` (or compatible) horizon masks,
            or None for "all ones".

    Returns:
        Dict with JAX arrays:
            ``insample_y``    [B, L, 1]
            ``outsample_y``   [B, h, 1]
            ``available_mask``[B, L, 1]   1=real history, 0=padded
            ``sample_mask``   [B, h, 1]   1=loss applies, 0=ignore
            ``hist_exog``     [B, L, X] or None
            ``futr_exog``     [B, L+h, F] or None
            ``stat_exog``     [B, S] or None
    """
    B = len(y_series)
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

    batch: Dict[str, Optional[jnp.ndarray]] = {
        "insample_y": jnp.array(np.stack(insample_ys)[:, :, None]),
        "outsample_y": jnp.array(np.stack(outsample_ys)[:, :, None]),
        "available_mask": jnp.array(np.stack(available_masks)[:, :, None]),
        "sample_mask": jnp.array(np.stack(sample_masks)[:, :, None]),
    }

    def _has(lst: Optional[List]) -> bool:
        return lst is not None and any(x is not None for x in lst)

    def _feat_dim(lst: List, default: int = 1) -> int:
        return next((x.shape[-1] for x in lst if x is not None), default)

    if _has(hist_exog_list):
        F = _feat_dim(hist_exog_list)
        hist_arr = np.stack([
            _pad_2d_left(x, input_size) if x is not None
            else np.zeros((input_size, F), dtype=np.float32)
            for x in hist_exog_list
        ])
        batch["hist_exog"] = jnp.array(hist_arr)
    else:
        batch["hist_exog"] = None

    if _has(futr_exog_list):
        F = _feat_dim(futr_exog_list)
        futr_arr = np.stack([
            _pad_2d_left(x, input_size + h) if x is not None
            else np.zeros((input_size + h, F), dtype=np.float32)
            for x in futr_exog_list
        ])
        batch["futr_exog"] = jnp.array(futr_arr)
    else:
        batch["futr_exog"] = None

    if _has(stat_exog_list):
        F = _feat_dim(stat_exog_list)
        stat_arr = np.stack([
            x.astype(np.float32) if x is not None
            else np.zeros(F, dtype=np.float32)
            for x in stat_exog_list
        ])
        batch["stat_exog"] = jnp.array(stat_arr)
    else:
        batch["stat_exog"] = None

    return batch


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def batch_generator(
    y_series: List[np.ndarray],
    input_size: int,
    h: int,
    batch_size: int,
    hist_exog_list: Optional[List[Optional[np.ndarray]]] = None,
    futr_exog_list: Optional[List[Optional[np.ndarray]]] = None,
    stat_exog_list: Optional[List[Optional[np.ndarray]]] = None,
    sample_mask_list: Optional[List[Optional[np.ndarray]]] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Dict[str, Optional[jnp.ndarray]]]:
    """Iterate over ``y_series`` yielding fully-prepared JAX batches.

    Side-effect-free: uses a local ``np.random.RandomState`` so calls with the
    same ``seed`` produce identical batch order.
    """
    n = len(y_series)
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        idx = indices[start: start + batch_size]
        sub = lambda lst: ([lst[i] for i in idx] if lst is not None else None)
        yield create_batch(
            y_series=[y_series[i] for i in idx],
            input_size=input_size,
            h=h,
            hist_exog_list=sub(hist_exog_list),
            futr_exog_list=sub(futr_exog_list),
            stat_exog_list=sub(stat_exog_list),
            sample_mask_list=sub(sample_mask_list),
        )

"""Functional data pipeline for the Chronax RNN model.

The pipeline is intentionally minimal — no PyTorch / Lightning / DataLoader
dependencies, and no NumPy: every array is a ``jnp.ndarray`` throughout, so
the pipeline stays JIT- and grad-friendly end to end.

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
from jax import random


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------


def pad_sequence(
    y: jnp.ndarray,
    target_len: int,
    pad_value: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Left-pad / left-truncate a 1D series to ``target_len``.

    The mask is 1 on real observations and 0 on the padded prefix, matching
    ``available_mask`` semantics.
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
    """Left-pad a 2D ``[T, F]`` array on the time axis."""
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
    """Align per-series exogenous arrays to model expectations.

    ``hist_exog`` is left-padded to length ``input_size``. ``futr_exog`` must
    cover both history and the forecast horizon, so it is left-padded to
    ``input_size + h``. ``stat_exog`` is returned as-is (cast to float32).
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
    """Split a single series into ``(history, horizon)``.

    Pads as needed so we always return ``(y_hist, y_out)`` where
    ``len(y_out) == h``.
    """
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
        "insample_y": jnp.stack(insample_ys)[:, :, None],
        "outsample_y": jnp.stack(outsample_ys)[:, :, None],
        "available_mask": jnp.stack(available_masks)[:, :, None],
        "sample_mask": jnp.stack(sample_masks)[:, :, None],
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
# Generator
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
    """Iterate over ``y_series`` yielding fully-prepared JAX batches.

    Side-effect-free: uses a local ``jax.random`` key so calls with the same
    ``seed`` produce identical batch order.
    """
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

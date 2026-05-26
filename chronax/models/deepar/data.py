"""Data pipeline for DeepAR model (JAX/FLAX, NumPy-free implementation).

The pipeline produces ``jnp.ndarray`` batches compatible with the DeepAR model.
It maintains two masks following Nixtla's vocabulary:
- available_mask: 1 where historic target is real, 0 where padded
- sample_mask: 1 on horizon steps included in loss, 0 otherwise (for holidays/outliers)
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Padding helpers (JAX-native, no NumPy arrays)
# ---------------------------------------------------------------------------


def pad_sequence(
    y: jnp.ndarray,
    target_len: int,
    pad_value: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Left-pad / left-truncate a 1D series to ``target_len`` (JAX version).

    Args:
        y: 1D array of values
        target_len: Desired output length
        pad_value: Value to use for padding

    Returns:
        (padded_array, mask) where mask is 1 on real data, 0 on padding
    """
    T = y.shape[0]
    
    # Truncate if needed
    y_trunc = y[-target_len:] if T >= target_len else y
    
    if T >= target_len:
        return y[-target_len:].astype(jnp.float32), jnp.ones(target_len, dtype=jnp.float32)

    # Pad with left-padding
    pad_width = target_len - T
    padded = jnp.concatenate([
        jnp.full(pad_width, pad_value, dtype=jnp.float32),
        y_trunc.astype(jnp.float32)
    ])
    mask = jnp.concatenate([
        jnp.zeros(pad_width, dtype=jnp.float32),
        jnp.ones(T, dtype=jnp.float32)
    ])
    return padded, mask


def _pad_2d_left(arr: jnp.ndarray, target_len: int) -> jnp.ndarray:
    """Left-pad a 2D ``[T, F]`` array on the time axis (JAX version)."""
    T = arr.shape[0]
    F = arr.shape[1] if arr.ndim > 1 else 1
    
    if T >= target_len:
        return arr[-target_len:].astype(jnp.float32)
    
    pad_width = target_len - T
    return jnp.pad(
        arr.astype(jnp.float32),
        ((pad_width, 0), (0, 0)) if arr.ndim > 1 else (pad_width, 0),
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
    """Align exogenous arrays to model expectations (JAX version).

    Args:
        hist_exog: Historic exogenous features, shape [T, X] or None
        futr_exog: Future exogenous features, shape [T+h, F] or None
        stat_exog: Static exogenous features, shape [S] or None
        input_size: History window length
        h: Forecast horizon

    Returns:
        Dict with aligned JAX arrays
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
    """Split a single series into ``(history, horizon)`` (JAX version).
    
    Args:
        y: 1D series
        h: Forecast horizon
    
    Returns:
        (y_hist, y_out) where y_hist is history and y_out is padded to length h
    """
    L = y.shape[0]
    
    if L > h:
        y_hist = y[:-h]
        y_out = y[-h:].astype(jnp.float32)
    elif L == h:
        y_hist = y[:1]
        y_out = y[-h:].astype(jnp.float32)
    else:
        # L < h: pad y_out to length h
        pad_width = h - L
        y_hist = y[:1]
        y_out = jnp.pad(y.astype(jnp.float32), (pad_width, 0), constant_values=0.0)
    
    return y_hist, y_out


def create_batch(
    y_series: List[jnp.ndarray],
    input_size: int,
    h: int,
    hist_exog_list: Optional[List[Optional[jnp.ndarray]]] = None,
    futr_exog_list: Optional[List[Optional[jnp.ndarray]]] = None,
    stat_exog_list: Optional[List[Optional[jnp.ndarray]]] = None,
    sample_mask_list: Optional[List[Optional[jnp.ndarray]]] = None,
) -> Dict[str, Optional[jnp.ndarray]]:
    """Build a single JAX batch from a list of time series.

    Args:
        y_series: List of 1D JAX arrays, each of length T_i
        input_size: History window length L
        h: Forecast horizon
        hist_exog_list: Per-series [T_i, X] historic exog, or None
        futr_exog_list: Per-series [T_i + h, F] future exog, or None
        stat_exog_list: Per-series [S] static exog, or None
        sample_mask_list: Per-series [h] horizon masks, or None

    Returns:
        Dict with JAX arrays ready for model
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

        # Handle sample mask
        if sample_mask_list is not None and sample_mask_list[i] is not None:
            user_mask = sample_mask_list[i].astype(jnp.float32)
            if user_mask.shape[0] != h:
                _, user_mask = pad_sequence(user_mask, h, pad_value=0.0)
            sample_masks.append(user_mask)
        else:
            sample_masks.append(jnp.ones(h, dtype=jnp.float32))

    # Stack into batch arrays
    batch: Dict[str, Optional[jnp.ndarray]] = {
        "insample_y": jnp.stack(insample_ys)[:, :, None],  # [B, L, 1]
        "outsample_y": jnp.stack(outsample_ys)[:, :, None],  # [B, h, 1]
        "available_mask": jnp.stack(available_masks)[:, :, None],  # [B, L, 1]
        "sample_mask": jnp.stack(sample_masks)[:, :, None],  # [B, h, 1]
    }

    def _has(lst: Optional[List]) -> bool:
        return lst is not None and any(x is not None for x in lst)

    def _feat_dim(lst: List, default: int = 1) -> int:
        for x in lst:
            if x is not None:
                return x.shape[-1] if x.ndim > 1 else 1
        return default

    # Handle historic exogenous features
    if _has(hist_exog_list):
        F = _feat_dim(hist_exog_list)
        hist_arr = jnp.stack([
            _pad_2d_left(x, input_size) if x is not None
            else jnp.zeros((input_size, F), dtype=jnp.float32)
            for x in hist_exog_list
        ])
        batch["hist_exog"] = hist_arr
    else:
        batch["hist_exog"] = None

    # Handle future exogenous features
    if _has(futr_exog_list):
        F = _feat_dim(futr_exog_list)
        futr_arr = jnp.stack([
            _pad_2d_left(x, input_size + h) if x is not None
            else jnp.zeros((input_size + h, F), dtype=jnp.float32)
            for x in futr_exog_list
        ])
        batch["futr_exog"] = futr_arr
    else:
        batch["futr_exog"] = None

    # Handle static exogenous features
    if _has(stat_exog_list):
        F = _feat_dim(stat_exog_list)
        stat_arr = jnp.stack([
            x.astype(jnp.float32) if x is not None
            else jnp.zeros(F, dtype=jnp.float32)
            for x in stat_exog_list
        ])
        batch["stat_exog"] = stat_arr
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

    Args:
        y_series: List of 1D JAX arrays
        input_size: History window length
        h: Forecast horizon
        batch_size: Batch size
        hist_exog_list: Historic exogenous features per series
        futr_exog_list: Future exogenous features per series
        stat_exog_list: Static exogenous features per series
        sample_mask_list: Sample masks per series
        shuffle: Whether to shuffle series
        seed: Random seed for shuffling

    Yields:
        Dict with batched JAX arrays
    """
    from jax import random
    
    n = len(y_series)
    key = random.PRNGKey(seed)
    indices = jnp.arange(n)
    
    if shuffle:
        key, subkey = random.split(key)
        indices = random.permutation(subkey, indices)

    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        
        def _subset(lst: Optional[List]) -> Optional[List]:
            if lst is None:
                return None
            return [lst[i] for i in idx]
        
        yield create_batch(
            y_series=[y_series[int(i)] for i in idx],
            input_size=input_size,
            h=h,
            hist_exog_list=_subset(hist_exog_list),
            futr_exog_list=_subset(futr_exog_list),
            stat_exog_list=_subset(stat_exog_list),
            sample_mask_list=_subset(sample_mask_list),
        )

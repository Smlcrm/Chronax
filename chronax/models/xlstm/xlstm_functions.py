"""High-level glue for the xLSTM / xLSTMTime forecaster.

Wires together the pure-functional pieces in :mod:`xlstm_backend` into a
training entry point (:func:`xlstm_f`) and a stateless decode wrapper
(:func:`forecast_xlstm`). Compiled XLA kernels are cached via
``functools.lru_cache`` keyed on the static ``XLSTMConfig`` plus
hyperparameters that affect compilation, mirroring the cached-jit
pattern used in ``ets/ets_functions.py`` and ``holt._get_holt_optimizer``.

Direct vs AR mode:
- AR (``cfg.decode_mode == "ar"``): training uses shifted-by-1 windows
  of length ``ctx_len + horizon_train``; loss on per-position next-step
  prediction. Inference autoregresses ``h`` steps from the final state.
- Direct (``cfg.decode_mode == "direct"``): training uses windows of
  ``ctx_len`` input + ``horizon`` target; the model emits all H steps
  in one forward via the direct linear head. Loss on (B, H) vs (B, H).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax, random
import optax

from chronax.utils.loss_functions import mean_absolute_error

from .xlstm_backend import (
    XLSTMConfig,
    init_params,
    xlstm_forward,
    decode,
    decode_direct,
)


# =============================================================================
# Window builders
# =============================================================================

def make_windows_ar(z: jnp.ndarray, ctx_len: int, horizon_train: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """AR-mode windows: input (T,), target shifted by 1. Both shape (n_win, ctx_len+horizon_train)."""
    seq_len = ctx_len + horizon_train
    n = z.shape[0]
    n_win = n - seq_len
    idx = jnp.arange(n_win)
    X = jax.vmap(lambda i: lax.dynamic_slice(z, (i,), (seq_len,)))(idx)
    Y = jax.vmap(lambda i: lax.dynamic_slice(z, (i + 1,), (seq_len,)))(idx)
    return X, Y


def make_windows_direct(z: jnp.ndarray, ctx_len: int, horizon: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Direct-mode windows: input (ctx_len,), target (horizon,) starting at ctx_len."""
    seq_len = ctx_len + horizon
    n = z.shape[0]
    n_win = n - seq_len + 1  # last allowed start is n - seq_len
    n_win = max(1, n_win)
    idx = jnp.arange(n_win)
    X = jax.vmap(lambda i: lax.dynamic_slice(z, (i,), (ctx_len,)))(idx)
    Y = jax.vmap(lambda i: lax.dynamic_slice(z, (i + ctx_len,), (horizon,)))(idx)
    return X, Y


# Backward-compat alias used by older call sites
def make_windows(z, ctx_len, horizon_train):
    return make_windows_ar(z, ctx_len, horizon_train)


# =============================================================================
# Optimizer
# =============================================================================

def _build_optimizer(total_steps: int, lr: float, weight_decay: float):
    warmup = max(1, min(50, total_steps // 10))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=max(1, total_steps),
        end_value=lr * 0.1,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )


# =============================================================================
# Training loop (cached, jitted)
# =============================================================================

@lru_cache(maxsize=128)
def _get_train_loop(cfg: XLSTMConfig, n_win: int, batch_size: int, total_steps: int,
                    lr: float, weight_decay: float):
    """Return a jitted training loop closed over the static hyperparameters."""
    optimizer = _build_optimizer(total_steps, lr, weight_decay)

    def _loss_fn(params, batch_x, batch_y):
        def fwd(x):
            preds, _ = xlstm_forward(params, x, cfg)
            return preds  # AR: (T,); Direct: (H,)
        preds = jax.vmap(fwd)(batch_x)  # AR: (B,T); Direct: (B,H)
        return mean_absolute_error(preds.astype(jnp.float32), batch_y.astype(jnp.float32))

    grad_fn = jax.value_and_grad(_loss_fn)

    @jax.jit
    def _train(params, X, Y, key):
        opt_state = optimizer.init(params)

        def step_fn(carry, _):
            params, opt_state, key = carry
            key, subkey = random.split(key)
            idx = random.randint(subkey, (batch_size,), 0, n_win)
            batch_x = jnp.take(X, idx, axis=0)
            batch_y = jnp.take(Y, idx, axis=0)
            loss, grads = grad_fn(params, batch_x, batch_y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, opt_state, key), loss

        (params, _, _), losses = lax.scan(
            step_fn, (params, opt_state, key), xs=None, length=total_steps
        )
        return params, losses

    return _train


# =============================================================================
# Decoder dispatch
# =============================================================================

@lru_cache(maxsize=128)
def _get_decoder(cfg: XLSTMConfig, h: int):
    """Return a jitted decoder for forecast horizon ``h``.

    In direct mode, ``h`` must equal ``cfg.horizon`` (the static head dim).
    In AR mode, ``h`` is the static unroll length.
    """
    if cfg.decode_mode == "direct":
        if h != cfg.horizon:
            raise ValueError(
                f"Direct-mode decoder requires h == cfg.horizon ({cfg.horizon}); got {h}."
            )
        @jax.jit
        def _dec(params, z_tail):
            return decode_direct(params, z_tail, cfg)
        return _dec

    @jax.jit
    def _dec(params, z_tail):
        return decode(params, z_tail, h, cfg)
    return _dec


# =============================================================================
# Top-level training entry
# =============================================================================

def xlstm_f(z: jnp.ndarray, cfg: XLSTMConfig, key: jax.Array, *,
            n_epochs: int, batch_size: int, lr: float, weight_decay: float):
    """Train an xLSTM / xLSTMTime model on series ``z``.

    Series-level normalization (z-score) is the caller's responsibility when
    ``cfg.use_revin == False``. When RevIN is on, pass the RAW series — the
    model handles normalization internally.

    Returns a dict: ``{"params", "losses", "cfg"}``.
    """
    n = int(z.shape[0])

    if cfg.decode_mode == "direct":
        seq_len = cfg.ctx_len + cfg.horizon
        if n < seq_len:
            raise ValueError(
                f"Series length {n} too short for ctx_len={cfg.ctx_len}+horizon={cfg.horizon} (direct mode)."
            )
        X, Y = make_windows_direct(z, cfg.ctx_len, cfg.horizon)
    else:
        seq_len = cfg.ctx_len + cfg.horizon_train
        if n <= seq_len:
            raise ValueError(
                f"Series length {n} too short for ctx_len={cfg.ctx_len}+horizon_train={cfg.horizon_train} (AR mode)."
            )
        X, Y = make_windows_ar(z, cfg.ctx_len, cfg.horizon_train)

    n_win = int(X.shape[0])
    bs = int(min(batch_size, n_win))
    total_steps = int(n_epochs * max(1, n_win // bs))

    k_init, k_train = random.split(key)
    params = init_params(k_init, cfg)

    train_fn = _get_train_loop(cfg, n_win, bs, total_steps, float(lr), float(weight_decay))
    params, losses = train_fn(params, X, Y, k_train)

    return {"params": params, "losses": losses, "cfg": cfg}


def forecast_xlstm(model_dict: dict, h: int) -> jnp.ndarray:
    """Forecast h steps using stored trained params.

    Returns a (h,) float32 array in the *normalised* scale when
    ``cfg.use_revin == False``. The caller denormalises with the stored
    ``mu`` and ``std``. When RevIN is on, returns the original-scale forecast
    (denormalisation happens inside the model).
    """
    cfg: XLSTMConfig = model_dict["cfg"]
    params = model_dict["params"]
    z_tail = model_dict["z_tail"]
    dec = _get_decoder(cfg, int(h))
    return dec(params, z_tail)

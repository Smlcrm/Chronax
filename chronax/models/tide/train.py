"""Training utilities for Chronax TiDE.

Pure JAX/Flax/Optax — no PyTorch, no numpy at training time.

Two training paths are provided:

1. **Window-based** (``train_step_windows`` / ``eval_step_windows``):
   Primary path. Accepts raw ``[B, input_size + h]`` windows, applies
   per-window ``RobustScaler`` (median/MAD) internally, then optimises the
   supplied loss. Used by :class:`~chronax.models.tide.forecaster.TiDEForecaster`
   and mirrors the NBEATS training path exactly.

2. **Batch-based** (``train_step`` / ``eval_step``):
   Compatibility path for pre-assembled ``{insample_y, outsample_y, ...}``
   dicts — required when exogenous variables are present.

LR schedule options:
    - ``cosine_decay_steps > 0``: warmup-cosine schedule.
    - ``num_lr_decays > 0``:     NF-style StepLR (gamma=0.5).
    - otherwise: constant learning rate.

Optimizer: ``chain(clip_by_global_norm, adam[w])``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from chronax.models.tide.loss import masked_mae
from chronax.models.tide.data import RobustScaler
from chronax.models.tide.model import TiDE, TiDEConfig

# Module-level scaler singleton — re-used across all JIT calls so JAX never
# sees it as a varying closure value that would trigger recompilation.
_SCALER = RobustScaler()


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------


class TrainState(train_state.TrainState):
    """Standard Flax TrainState; aliased for forward compatibility."""
    pass


# ---------------------------------------------------------------------------
# Optimizer builder (module-level cache avoids JIT recompilation)
# ---------------------------------------------------------------------------

_OPT_CACHE: Dict[tuple, optax.GradientTransformation] = {}


def _build_optimizer(
    learning_rate: float,
    grad_clip: float,
    cosine_decay_steps: int,
    warmup_steps: int,
    num_lr_decays: int,
    max_steps: int,
    weight_decay: float,
) -> optax.GradientTransformation:
    key = (learning_rate, grad_clip, cosine_decay_steps, warmup_steps,
           num_lr_decays, max_steps, weight_decay)
    if key in _OPT_CACHE:
        return _OPT_CACHE[key]

    if cosine_decay_steps > 0:
        if warmup_steps > 0:
            schedule: optax.ScalarOrSchedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=cosine_decay_steps,
                end_value=learning_rate * 1e-2,
            )
        else:
            schedule = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=cosine_decay_steps,
                alpha=1e-2,
            )
    elif num_lr_decays > 0:
        decay_every = max(max_steps // num_lr_decays, 1)
        schedules = [
            optax.constant_schedule(learning_rate * (0.5 ** i))
            for i in range(num_lr_decays + 1)
        ]
        boundaries = [decay_every * (i + 1) for i in range(num_lr_decays)]
        schedule = optax.join_schedules(schedules, boundaries)
    else:
        schedule = learning_rate

    base_opt = (
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
        if weight_decay > 0.0
        else optax.adam(learning_rate=schedule)
    )
    tx = (
        optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
        if grad_clip > 0.0
        else base_opt
    )
    _OPT_CACHE[key] = tx
    return tx


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def create_train_state(
    rng: jax.Array,
    config: TiDEConfig,
    *,
    learning_rate: float = 1e-3,
    grad_clip: float = 1.0,
    cosine_decay_steps: int = 0,
    warmup_steps: int = 0,
    num_lr_decays: int = -1,
    max_steps: int = 1000,
    weight_decay: float = 0.0,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> TrainState:
    """Initialise model parameters and optimiser state.

    Args:
        rng: PRNG key for parameter initialisation.
        config: Model architecture config.
        learning_rate: Adam(W) base learning rate.
        grad_clip: Global gradient-norm clip (0 = disabled).
        cosine_decay_steps: Warmup-cosine schedule length (0 = disabled).
        warmup_steps: Linear warmup steps for the cosine schedule.
        num_lr_decays: NF-style StepLR decays (-1 = constant LR).
        max_steps: Total training steps (used for StepLR spacing).
        weight_decay: Use AdamW when > 0; plain Adam otherwise.
        optimizer: Pre-built transform that overrides all defaults.
    """
    model = TiDE(config=config)
    init_rng, _ = jax.random.split(rng)

    # Build dummy inputs that cover all optional code paths
    dummy_y = jnp.zeros((1, config.input_size, 1))
    dummy_hist = (
        jnp.zeros((1, config.input_size, config.hist_exog_size))
        if config.hist_exog_size > 0 else None
    )
    dummy_futr = (
        jnp.zeros((1, config.input_size + config.h, config.futr_exog_size))
        if config.futr_exog_size > 0 else None
    )
    dummy_stat = (
        jnp.zeros((1, config.stat_exog_size))
        if config.stat_exog_size > 0 else None
    )

    params = model.init(
        init_rng,
        insample_y=dummy_y,
        hist_exog=dummy_hist,
        futr_exog=dummy_futr,
        stat_exog=dummy_stat,
        deterministic=True,
    )

    if optimizer is None:
        optimizer = _build_optimizer(
            learning_rate=learning_rate,
            grad_clip=grad_clip,
            cosine_decay_steps=cosine_decay_steps,
            warmup_steps=warmup_steps,
            num_lr_decays=num_lr_decays,
            max_steps=max_steps,
            weight_decay=weight_decay,
        )

    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


# ---------------------------------------------------------------------------
# Forward + loss
# ---------------------------------------------------------------------------


def _forward_loss(
    params: Any,
    apply_fn: Callable,
    batch: Dict[str, Optional[jnp.ndarray]],
    loss_fn: Callable,
    rng: Optional[jax.Array],
    deterministic: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    rngs = {"dropout": rng} if (rng is not None and not deterministic) else {}
    output = apply_fn(
        params,
        insample_y=batch["insample_y"],
        hist_exog=batch.get("hist_exog"),
        futr_exog=batch.get("futr_exog"),
        stat_exog=batch.get("stat_exog"),
        deterministic=deterministic,
        rngs=rngs,
    )  # [B, h, output_size]
    loss = loss_fn(
        y=batch["outsample_y"],
        y_hat=output,
        mask=batch.get("sample_mask"),
    )
    return loss, output


# ---------------------------------------------------------------------------
# JIT-compiled steps
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("loss_fn",))
def train_step(
    state: TrainState,
    batch: Dict[str, Optional[jnp.ndarray]],
    rng: jax.Array,
    loss_fn: Callable = masked_mae,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled training step.

    Returns ``(new_state, loss, predictions)``.
    """
    def compute(params):
        return _forward_loss(
            params=params,
            apply_fn=state.apply_fn,
            batch=batch,
            loss_fn=loss_fn,
            rng=rng,
            deterministic=False,
        )

    (loss, preds), grads = jax.value_and_grad(compute, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, preds


@partial(jax.jit, static_argnames=("loss_fn",))
def eval_step(
    state: TrainState,
    batch: Dict[str, Optional[jnp.ndarray]],
    loss_fn: Callable = masked_mae,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation step. Deterministic, no gradients."""
    return _forward_loss(
        params=state.params,
        apply_fn=state.apply_fn,
        batch=batch,
        loss_fn=loss_fn,
        rng=None,
        deterministic=True,
    )


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------


def train_loop(
    state: TrainState,
    train_batches: Iterable[Dict[str, Optional[jnp.ndarray]]],
    *,
    num_epochs: int = 10,
    rng: Optional[jax.Array] = None,
    eval_batches: Optional[Iterable[Dict[str, Optional[jnp.ndarray]]]] = None,
    loss_fn: Callable = masked_mae,
) -> Tuple[TrainState, List[Dict[str, float]]]:
    """Drive ``train_step`` / ``eval_step`` over multiple epochs."""
    if rng is None:
        rng = jax.random.PRNGKey(0)

    history: List[Dict[str, float]] = []

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in train_batches:
            rng, step_rng = jax.random.split(rng)
            state, loss, _ = train_step(state, batch, step_rng, loss_fn=loss_fn)
            epoch_losses.append(float(loss))

        info: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": (
                sum(epoch_losses) / max(len(epoch_losses), 1)
                if epoch_losses else 0.0
            ),
        }
        if eval_batches is not None:
            evals = []
            for batch in eval_batches:
                loss, _ = eval_step(state, batch, loss_fn=loss_fn)
                evals.append(float(loss))
            info["eval_loss"] = sum(evals) / max(len(evals), 1) if evals else 0.0

        history.append(info)

    return state, history


# ---------------------------------------------------------------------------
# Window-based training steps (primary path — mirrors NBEATS train.py)
# ---------------------------------------------------------------------------
# These accept raw [B, input_size + h] windows, apply per-window RobustScaler
# inside the JIT, and call model.apply with [B, L, 1] shaped inputs.
# ``input_size`` is a static arg so XLA re-traces only when the horizon changes.


def _window_forward_loss(
    params: Any,
    apply_fn: Callable,
    windows: jnp.ndarray,       # [B, L + h]
    input_size: int,
    loss_fn: Callable,
    rng: Optional[jax.Array],
    deterministic: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    insample = windows[:, :input_size]                            # [B, L]
    target   = windows[:, input_size:]                            # [B, h]
    shift, scale = _SCALER.stats(insample, axis=1)                # [B, 1]
    insample_z = _SCALER.transform(insample, shift, scale)        # [B, L]
    target_z   = _SCALER.transform(target,   shift, scale)        # [B, h]
    insample_3d = insample_z[:, :, None]                          # [B, L, 1]
    target_3d   = target_z[:, :, None]                            # [B, h, 1]
    rngs = {"dropout": rng} if (rng is not None and not deterministic) else {}
    forecast = apply_fn(                                          # [B, h, 1]
        params, insample_3d, deterministic=deterministic, rngs=rngs
    )
    loss = loss_fn(target_3d, forecast)
    return loss, forecast


@partial(jax.jit, static_argnames=("loss_fn", "input_size"))
def train_step_windows(
    state: TrainState,
    windows: jnp.ndarray,       # [B, L + h]
    rng: jax.Array,
    input_size: int,
    loss_fn: Callable = masked_mae,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled window-based training step with per-window RobustScaler.

    Args:
        state: Current ``TrainState``.
        windows: Raw ``[B, input_size + h]`` windows (unscaled).
        rng: PRNG key for dropout.
        input_size: History length ``L`` (static; triggers re-trace when changed).
        loss_fn: ``(y_true, y_pred) -> scalar`` (default: ``masked_mae``).

    Returns:
        ``(new_state, loss, predictions)``
    """
    def compute(params):
        return _window_forward_loss(
            params=params, apply_fn=state.apply_fn, windows=windows,
            input_size=input_size, loss_fn=loss_fn, rng=rng, deterministic=False,
        )

    (loss, preds), grads = jax.value_and_grad(compute, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, preds


@partial(jax.jit, static_argnames=("loss_fn", "input_size"))
def eval_step_windows(
    state: TrainState,
    windows: jnp.ndarray,       # [B, L + h]
    input_size: int,
    loss_fn: Callable = masked_mae,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled window-based evaluation step. Deterministic, no gradients."""
    return _window_forward_loss(
        params=state.params, apply_fn=state.apply_fn, windows=windows,
        input_size=input_size, loss_fn=loss_fn, rng=None, deterministic=True,
    )


# ---------------------------------------------------------------------------
# Standard (mean/std) per-window training steps
# ---------------------------------------------------------------------------
# Alternative to the RobustScaler path above.  Uses per-window mean/std
# (instance normalisation) rather than median/MAD.  Better for small,
# smooth datasets (e.g. < 1000 training points) where the median estimator
# can be noisy on short windows.


_STD_EPS: float = 1e-6


def _window_forward_loss_std(
    params: Any,
    apply_fn: Callable,
    windows: jnp.ndarray,
    input_size: int,
    loss_fn: Callable,
    rng: Optional[jax.Array],
    deterministic: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    insample = windows[:, :input_size]
    target   = windows[:, input_size:]
    shift = jnp.mean(insample, axis=1, keepdims=True)               # [B, 1]
    scale = jnp.std(insample,  axis=1, keepdims=True) + _STD_EPS   # [B, 1]
    insample_3d = ((insample - shift) / scale)[:, :, None]
    target_3d   = ((target   - shift) / scale)[:, :, None]
    rngs = {"dropout": rng} if (rng is not None and not deterministic) else {}
    forecast = apply_fn(params, insample_3d, deterministic=deterministic, rngs=rngs)
    return loss_fn(target_3d, forecast), forecast


@partial(jax.jit, static_argnames=("loss_fn", "input_size"))
def train_step_windows_std(
    state: TrainState,
    windows: jnp.ndarray,
    rng: jax.Array,
    input_size: int,
    loss_fn: Callable = masked_mae,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled window-based training step with per-window mean/std scaler."""
    def compute(params):
        return _window_forward_loss_std(
            params=params, apply_fn=state.apply_fn, windows=windows,
            input_size=input_size, loss_fn=loss_fn, rng=rng, deterministic=False,
        )
    (loss, preds), grads = jax.value_and_grad(compute, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, preds


@partial(jax.jit, static_argnames=("loss_fn", "input_size"))
def eval_step_windows_std(
    state: TrainState,
    windows: jnp.ndarray,
    input_size: int,
    loss_fn: Callable = masked_mae,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled window-based eval step with per-window mean/std scaler."""
    return _window_forward_loss_std(
        params=state.params, apply_fn=state.apply_fn, windows=windows,
        input_size=input_size, loss_fn=loss_fn, rng=None, deterministic=True,
    )


# ---------------------------------------------------------------------------
# Raw (globally-pre-normalised) window training steps
# ---------------------------------------------------------------------------
# These skip the per-window RobustScaler entirely.  Use them when the
# forecaster applies a global (per-series) StandardScaler before windowing,
# matching NeuralForecast's ``scaler_type="standard"`` behaviour.  The
# per-window scaler removes level/trend information that a globally normalised
# series still encodes; omitting it lets the model exploit that signal.


@partial(jax.jit, static_argnames=("loss_fn", "input_size"))
def train_step_windows_raw(
    state: TrainState,
    windows: jnp.ndarray,       # [B, L + h]  already globally normalised
    rng: jax.Array,
    input_size: int,
    loss_fn: Callable = masked_mae,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled training step for pre-normalised windows (no per-window scaler)."""
    def compute(params):
        insample_3d = windows[:, :input_size, None]   # [B, L, 1]
        target_3d   = windows[:, input_size:,  None]  # [B, h, 1]
        rngs = {"dropout": rng}
        forecast = state.apply_fn(params, insample_3d, deterministic=False, rngs=rngs)
        return loss_fn(target_3d, forecast), forecast

    (loss, preds), grads = jax.value_and_grad(compute, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, preds


@partial(jax.jit, static_argnames=("loss_fn", "input_size"))
def eval_step_windows_raw(
    state: TrainState,
    windows: jnp.ndarray,       # [B, L + h]  already globally normalised
    input_size: int,
    loss_fn: Callable = masked_mae,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled eval step for pre-normalised windows (no per-window scaler)."""
    insample_3d = windows[:, :input_size, None]   # [B, L, 1]
    target_3d   = windows[:, input_size:,  None]  # [B, h, 1]
    forecast = state.apply_fn(state.params, insample_3d, deterministic=True)
    return loss_fn(target_3d, forecast), forecast

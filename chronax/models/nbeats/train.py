"""Training utilities for Chronax N-BEATS.

Pure JAX/Flax/Optax — no PyTorch, no numpy.

All long-lived state is held in ``flax.training.train_state.TrainState`` and
all randomness is threaded explicitly through ``jax.random.PRNGKey``.

Two training paths are provided:

1. **Window-based** (``train_step`` / ``eval_step``):
   Primary path. Accepts raw ``[B, input_size + h]`` windows, applies
   per-window ``RobustScaler`` internally, then optimises ``masked_mae``
   (or any supplied loss). Used by :class:`NBEATSForecaster`.

2. **Batch-based** (``train_batch_step`` / ``eval_batch_step``):
   Compatibility path. Accepts pre-scaled
   ``{insample_y, outsample_y, sample_mask}`` dicts and masked loss.
   Mirrors the RNN / Autoformer training API for cross-model comparisons.

LR schedule:
    ``create_train_state`` accepts ``cosine_decay_steps`` + ``warmup_steps``
    for a warmup-cosine schedule, OR ``num_lr_decays`` for the NF-style
    StepLR (gamma=0.5 per decay). Both RNN and TSMixer benchmarks use the
    cosine path.

Optimizer structure (same as RNN / TSMixer benchmarks):
    ``chain(clip_by_global_norm(grad_clip), adam[w](schedule))``
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from chronax.models.nbeats.data import RobustScaler
from chronax.models.nbeats.loss import masked_mae
from chronax.models.nbeats.model import NBEATS, NBEATSConfig


# ---------------------------------------------------------------------------
# Module-level caches — ensure the same apply_fn / tx objects are reused so
# JAX JIT never recompiles for the same config + optimizer combo.
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[NBEATSConfig, NBEATS] = {}
_OPT_CACHE: Dict[tuple, optax.GradientTransformation] = {}


def _get_model(config: NBEATSConfig) -> NBEATS:
    if config not in _MODEL_CACHE:
        _MODEL_CACHE[config] = NBEATS(config=config)
    return _MODEL_CACHE[config]


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

    # Build LR schedule
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
        # NF-style StepLR: halve LR every (max_steps // num_lr_decays) steps
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
# TrainState
# ---------------------------------------------------------------------------


class TrainState(train_state.TrainState):
    """Standard Flax TrainState; aliased for forward compatibility."""

    pass


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def create_train_state(
    rng: jax.Array,
    config: NBEATSConfig,
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
        cosine_decay_steps: Use warmup-cosine schedule decaying over this many
            steps (0 = disabled; takes precedence over ``num_lr_decays``).
        warmup_steps: Linear warmup steps for the cosine schedule.
        num_lr_decays: NF-style StepLR decays with gamma=0.5 (used only when
            ``cosine_decay_steps <= 0``; -1 = constant LR).
        max_steps: Total training steps (used for StepLR spacing).
        weight_decay: Use AdamW when > 0; plain Adam otherwise.
        optimizer: Pre-built transform that overrides all defaults.
    """
    model = _get_model(config)
    init_rng, _ = jax.random.split(rng)

    dummy_y = jnp.zeros((1, config.input_size), dtype=jnp.float32)
    params = model.init(init_rng, dummy_y, deterministic=True)

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
# Per-window scaler singleton (reused across all JIT calls)
# ---------------------------------------------------------------------------

_SCALER = RobustScaler()


# ---------------------------------------------------------------------------
# Window-based training steps (primary path)
# ---------------------------------------------------------------------------


def _window_forward_loss(
    params: Any,
    apply_fn: Callable,
    windows: jnp.ndarray,    # [B, L + h]
    input_size: int,
    loss_fn: Callable,
    rng: Optional[jax.Array],
    deterministic: bool,
    scale: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    insample = windows[:, :input_size]   # [B, L]
    target = windows[:, input_size:]     # [B, h]
    if scale:
        shift, scale_v = _SCALER.stats(insample, axis=1)   # [B, 1]
        insample_z = _SCALER.transform(insample, shift, scale_v)
        target_z = _SCALER.transform(target, shift, scale_v)
    else:
        insample_z = insample
        target_z = target
    rngs = {"dropout": rng} if (rng is not None and not deterministic) else {}
    forecast = apply_fn(params, insample_z, deterministic=deterministic, rngs=rngs)
    loss = loss_fn(target_z, forecast)
    return loss, forecast


@partial(jax.jit, static_argnames=("loss_fn", "input_size", "scale"))
def train_step(
    state: TrainState,
    windows: jnp.ndarray,    # [B, L + h]
    rng: jax.Array,
    input_size: int,
    loss_fn: Callable = masked_mae,
    scale: bool = True,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled window-based training step.

    Args:
        state: Current ``TrainState``.
        windows: Raw ``[B, input_size + h]`` windows.
        rng: PRNG key for dropout.
        input_size: History length (static; re-compiles when changed).
        loss_fn: ``(y_true, y_pred) -> scalar`` (default: ``masked_mae``).
        scale: Apply per-window RobustScaler when True (static flag).

    Returns:
        ``(new_state, loss, predictions)``
    """
    def compute(params):
        return _window_forward_loss(
            params=params, apply_fn=state.apply_fn, windows=windows,
            input_size=input_size, loss_fn=loss_fn, rng=rng, deterministic=False,
            scale=scale,
        )

    (loss, preds), grads = jax.value_and_grad(compute, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, preds


@partial(jax.jit, static_argnames=("loss_fn", "input_size", "scale"))
def eval_step(
    state: TrainState,
    windows: jnp.ndarray,
    input_size: int,
    loss_fn: Callable = masked_mae,
    scale: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled window-based evaluation step. Deterministic, no grads."""
    return _window_forward_loss(
        params=state.params, apply_fn=state.apply_fn, windows=windows,
        input_size=input_size, loss_fn=loss_fn, rng=None, deterministic=True,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Batch-based training steps (compatibility / masked-loss path)
# ---------------------------------------------------------------------------


def _batch_forward_loss(
    params: Any,
    apply_fn: Callable,
    batch: Dict[str, Optional[jnp.ndarray]],
    loss_fn: Callable,
    rng: Optional[jax.Array],
    deterministic: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    rngs = {"dropout": rng} if (rng is not None and not deterministic) else {}
    pred = apply_fn(
        params,
        batch["insample_y"],
        batch.get("available_mask"),
        deterministic=deterministic,
        rngs=rngs,
    )  # [B, h]
    loss = loss_fn(
        y=batch["outsample_y"],
        y_hat=pred,
        mask=batch.get("sample_mask"),
    )
    return loss, pred


@partial(jax.jit, static_argnames=("loss_fn",))
def train_batch_step(
    state: TrainState,
    batch: Dict[str, Optional[jnp.ndarray]],
    rng: jax.Array,
    loss_fn: Callable = masked_mae,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled batch-based training step on pre-scaled inputs."""

    def compute(params):
        return _batch_forward_loss(
            params=params, apply_fn=state.apply_fn, batch=batch,
            loss_fn=loss_fn, rng=rng, deterministic=False,
        )

    (loss, preds), grads = jax.value_and_grad(compute, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, preds


@partial(jax.jit, static_argnames=("loss_fn",))
def eval_batch_step(
    state: TrainState,
    batch: Dict[str, Optional[jnp.ndarray]],
    loss_fn: Callable = masked_mae,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled batch evaluation step. Deterministic, no grads."""
    return _batch_forward_loss(
        params=state.params, apply_fn=state.apply_fn, batch=batch,
        loss_fn=loss_fn, rng=None, deterministic=True,
    )


# ---------------------------------------------------------------------------
# Epoch-based loop (batch-based path)
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
    """Drive ``train_batch_step`` / ``eval_batch_step`` over multiple epochs."""
    if rng is None:
        rng = jax.random.PRNGKey(0)

    history: List[Dict[str, float]] = []

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in train_batches:
            rng, step_rng = jax.random.split(rng)
            state, loss, _ = train_batch_step(state, batch, step_rng, loss_fn=loss_fn)
            epoch_losses.append(float(loss))

        info: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": (
                sum(epoch_losses) / max(len(epoch_losses), 1) if epoch_losses else 0.0
            ),
        }

        if eval_batches is not None:
            evals = []
            for batch in eval_batches:
                loss, _ = eval_batch_step(state, batch, loss_fn=loss_fn)
                evals.append(float(loss))
            info["eval_loss"] = sum(evals) / max(len(evals), 1) if evals else 0.0

        history.append(info)

    return state, history

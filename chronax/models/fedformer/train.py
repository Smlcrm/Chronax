"""Training utilities for the Chronax FEDformer model (pure JAX/Flax/Optax).

Self-contained -- no imports from other chronax models.

All long-lived state lives in a ``flax.training.train_state.TrainState`` and all
randomness is threaded explicitly through ``jax.random.PRNGKey`` (functional --
no hidden global RNG), which is what lets the steps be ``jax.jit``-compiled.

Two interchangeable training paradigms are provided:

1. **Window-based** (``train_window_step`` / ``eval_window_step``):
   Accepts raw ``[B, input_size + h]`` windows, applies a per-window
   ``RobustScaler`` *inside* the step, then optimises ``mae`` / ``mse``. This is
   what :class:`FEDformerForecaster` uses (mirrors neuralforecast's recipe).

2. **Batch-based** (``train_step`` / ``eval_step`` / ``train_loop``):
   Accepts pre-scaled ``{insample_y, outsample_y, sample_mask}`` dicts and
   optimises a masked loss -- handy when some horizon steps must be ignored.

Schedule / control helpers mirror common neuralforecast knobs:
    ``make_lr_schedule``   -- StepLR-style gamma=0.5 per decay.
    ``sample_batch_indices`` -- random with-replacement window sampling.
    ``should_stop_early``  -- patience-based early-stop check.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from chronax.models.fedformer.data import RobustScaler
from chronax.models.fedformer.loss import mae as _mae
from chronax.models.fedformer.loss import masked_mae
from chronax.models.fedformer.model import FEDformerConfig, FEDformerModel


# ---------------------------------------------------------------------------
# Module-level caches -- reuse the same apply_fn / optimizer objects across
# TrainState instances so JAX's JIT cache is hit (no needless recompilation
# for an identical config).
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}
_OPTIMIZER_CACHE: dict = {}


def _get_model(config: FEDformerConfig) -> FEDformerModel:
    """Return a cached ``FEDformerModel`` for ``config`` (one apply_fn per config)."""
    if config not in _MODEL_CACHE:
        _MODEL_CACHE[config] = FEDformerModel(config=config)
    return _MODEL_CACHE[config]


def _get_optimizer(
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    num_lr_decays: int,
    max_steps: int,
) -> optax.GradientTransformation:
    """Return a cached optimiser keyed by its hyperparameters."""
    key = (learning_rate, weight_decay, grad_clip, num_lr_decays, max_steps)
    if key not in _OPTIMIZER_CACHE:
        lr_schedule = make_lr_schedule(learning_rate, max_steps, num_lr_decays)
        if weight_decay > 0.0:
            base_opt = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
        else:
            base_opt = optax.adam(learning_rate=lr_schedule)
        if grad_clip > 0.0:
            # Global-norm clipping stabilises the FFT-heavy gradients.
            optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
        else:
            optimizer = base_opt
        _OPTIMIZER_CACHE[key] = optimizer
    return _OPTIMIZER_CACHE[key]


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------


class TrainState(train_state.TrainState):
    """Standard Flax TrainState; aliased here for forward compatibility."""

    pass


# ---------------------------------------------------------------------------
# Schedule and sampling helpers
# ---------------------------------------------------------------------------


def make_lr_schedule(
    learning_rate: float,
    max_steps: int,
    num_lr_decays: int,
) -> optax.ScalarOrSchedule:
    """StepLR-style schedule that halves the LR ``num_lr_decays`` times.

    Returns a constant LR when ``num_lr_decays <= 0``. Otherwise the LR is
    piecewise-constant, multiplied by 0.5 at evenly spaced step boundaries --
    the same gamma=0.5 StepLR recipe neuralforecast uses.
    """
    if num_lr_decays <= 0:
        return learning_rate
    decay_every = max(max_steps // num_lr_decays, 1)
    schedules = [
        optax.constant_schedule(learning_rate * (0.5 ** i))
        for i in range(num_lr_decays + 1)
    ]
    boundaries = [decay_every * (i + 1) for i in range(num_lr_decays)]
    return optax.join_schedules(schedules, boundaries)


def sample_batch_indices(
    key: jax.Array,
    n_train: int,
    batch_size: int,
    n_steps: int,
) -> jax.Array:
    """Sample ``[n_steps, batch_size]`` window indices with replacement."""
    return jax.random.choice(key, n_train, shape=(n_steps, batch_size), replace=True)


def should_stop_early(
    *,
    early_stop_patience_steps: int,
    checks_without_improvement: int,
) -> bool:
    """True when validation has not improved for ``patience`` consecutive checks."""
    return (
        early_stop_patience_steps > 0
        and checks_without_improvement >= early_stop_patience_steps
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def create_train_state(
    rng: jax.Array,
    config: FEDformerConfig,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    num_lr_decays: int = -1,
    max_steps: int = 1000,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> TrainState:
    """Initialise model parameters and optimiser state.

    Args:
        rng: PRNG key for parameter initialisation.
        config: Model architecture config.
        learning_rate: Peak learning rate.
        weight_decay: If > 0 use ``optax.adamw``, else ``optax.adam``.
        grad_clip: Global gradient-norm clip threshold (0 = disabled).
        num_lr_decays: Number of StepLR decays (gamma=0.5); -1 = constant LR.
        max_steps: Total training steps (sets decay-schedule spacing).
        optimizer: Optional pre-built transform; overrides all defaults.
    """
    model = _get_model(config)
    init_rng, _ = jax.random.split(rng)

    # A single dummy window is enough to materialise all parameter shapes.
    dummy_y = jnp.zeros((1, config.input_size, 1))
    params = model.init({"params": init_rng, "dropout": init_rng}, dummy_y, deterministic=True)

    if optimizer is None:
        optimizer = _get_optimizer(
            learning_rate, weight_decay, grad_clip, num_lr_decays, max_steps
        )

    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


# ---------------------------------------------------------------------------
# Batch-based training (pre-scaled inputs, masked loss)
# ---------------------------------------------------------------------------


def _batch_forward_loss(
    params: Any,
    apply_fn: Callable,
    batch: Dict[str, Optional[jnp.ndarray]],
    loss_fn: Callable,
    rng: Optional[jax.Array],
    deterministic: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass + masked loss on a pre-scaled batch dict."""
    rngs = {"dropout": rng} if (rng is not None and not deterministic) else {}
    pred = apply_fn(
        params,
        batch["insample_y"],
        deterministic=deterministic,
        rngs=rngs,
    )  # [B, h, 1]
    loss = loss_fn(
        y=batch["outsample_y"],
        y_hat=pred,
        mask=batch.get("sample_mask"),
    )
    return loss, pred


@partial(jax.jit, static_argnames=("loss_fn",))
def train_step(
    state: TrainState,
    batch: Dict[str, Optional[jnp.ndarray]],
    rng: jax.Array,
    loss_fn: Callable = masked_mae,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled training step on a pre-scaled batch.

    Returns ``(new_state, loss, predictions)``.
    """

    def compute(params):
        return _batch_forward_loss(
            params=params,
            apply_fn=state.apply_fn,
            batch=batch,
            loss_fn=loss_fn,
            rng=rng,
            deterministic=False,
        )

    (loss, predictions), grads = jax.value_and_grad(compute, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, predictions


@partial(jax.jit, static_argnames=("loss_fn",))
def eval_step(
    state: TrainState,
    batch: Dict[str, Optional[jnp.ndarray]],
    loss_fn: Callable = masked_mae,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation step. Deterministic, no gradients."""
    return _batch_forward_loss(
        params=state.params,
        apply_fn=state.apply_fn,
        batch=batch,
        loss_fn=loss_fn,
        rng=None,
        deterministic=True,
    )


# ---------------------------------------------------------------------------
# Window-based training (raw windows, per-window RobustScaler)
# ---------------------------------------------------------------------------

_SCALER = RobustScaler()


def _window_forward_loss(
    params: Any,
    apply_fn: Callable,
    windows: jnp.ndarray,
    input_size: int,
    loss_fn: Callable,
    rng: Optional[jax.Array],
    deterministic: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward + loss on raw ``[B, input_size + h]`` windows with per-window scaling.

    The history and target are scaled by the *history's* statistics so the model
    never sees the target's scale (no leakage); the loss is computed in scaled
    space, which keeps the gradient magnitudes comparable across windows.
    """
    insample = windows[:, :input_size]   # [B, L]
    target = windows[:, input_size:]     # [B, h]
    shift, scale = _SCALER.stats(insample, axis=1)                      # [B, 1] each
    insample_z = _SCALER.transform(insample, shift, scale)[..., None]   # [B, L, 1]
    target_z = _SCALER.transform(target, shift, scale)                  # [B, h]
    rngs = {"dropout": rng} if (rng is not None and not deterministic) else {}
    pred = apply_fn(params, insample_z, deterministic=deterministic, rngs=rngs)  # [B, h, 1]
    loss = loss_fn(pred[..., 0], target_z)
    return loss, pred


@partial(jax.jit, static_argnames=("loss_fn", "input_size"))
def train_window_step(
    state: TrainState,
    windows: jnp.ndarray,
    rng: jax.Array,
    input_size: int,
    loss_fn: Callable = _mae,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled training step on raw ``[B, input_size + h]`` windows.

    Per-window ``RobustScaler`` is applied internally. ``input_size`` is static
    (changing it triggers a recompile).
    """

    def compute(params):
        return _window_forward_loss(
            params=params,
            apply_fn=state.apply_fn,
            windows=windows,
            input_size=input_size,
            loss_fn=loss_fn,
            rng=rng,
            deterministic=False,
        )

    (loss, predictions), grads = jax.value_and_grad(compute, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, predictions


@partial(jax.jit, static_argnames=("loss_fn", "input_size"))
def eval_window_step(
    state: TrainState,
    windows: jnp.ndarray,
    input_size: int,
    loss_fn: Callable = _mae,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation on raw windows. Deterministic, no gradients."""
    return _window_forward_loss(
        params=state.params,
        apply_fn=state.apply_fn,
        windows=windows,
        input_size=input_size,
        loss_fn=loss_fn,
        rng=None,
        deterministic=True,
    )


# ---------------------------------------------------------------------------
# Epoch-based loop (batch-based)
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
    """Drive ``train_step`` / ``eval_step`` over multiple epochs.

    ``train_batches`` may be a list (re-iterated each epoch) or a generator
    (materialise it first if you need multiple epochs).
    """
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
                sum(epoch_losses) / max(len(epoch_losses), 1) if epoch_losses else 0.0
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

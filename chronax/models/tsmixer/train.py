"""Training utilities for the Chronax TSMixer model.

Pure JAX/Flax/Optax. All long-lived state is held in a custom Flax
``TrainState`` that carries both model parameters and BatchNorm running
statistics (``batch_stats``) as separate pytree fields.

TSMixer's forward pass returns a plain ``[B, h, N]`` tensor; the training step
also updates ``batch_stats`` via the mutable collection returned by
``model.apply(..., mutable=['batch_stats'])``.

Public API:
    :func:`create_train_state` — initialise model parameters + optimiser.
    :func:`train_step`          — JIT-compiled forward + backward + update.
    :func:`eval_step`           — JIT-compiled deterministic evaluation.
    :func:`scan_train_loop`     — all-steps-at-once XLA scan (no batch_stats).
    :func:`train_loop`          — convenience epoch loop.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from chronax.models.tsmixer.loss import masked_mae
from chronax.models.tsmixer.model import TSMixer, TSMixerConfig


class TrainState(train_state.TrainState):
    """Extends Flax TrainState with a ``batch_stats`` field for BatchNorm.

    Keeping ``batch_stats`` separate from ``params`` ensures the Adam
    optimizer only tracks and updates the trainable parameters; running
    statistics are updated via exponential moving average inside
    ``train_step``, never via gradient descent.
    """

    batch_stats: Any


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def create_train_state(
    rng: jax.Array,
    config: TSMixerConfig,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    cosine_decay_steps: int = 0,
    warmup_steps: int = 0,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> TrainState:
    """Initialise model parameters, BatchNorm statistics, and optimiser state.

    Args:
        rng: PRNG key for parameter initialisation.
        config: model architecture config.
        learning_rate: peak learning rate.
        weight_decay: if > 0, use AdamW instead of Adam.
        grad_clip: global gradient-norm clip threshold (0 = disabled).
        cosine_decay_steps: if > 0, use warmup + cosine decay schedule.
        warmup_steps: linear warmup steps prepended to the cosine decay.
        optimizer: optional pre-built optax transform; overrides all defaults.
    """
    model = TSMixer(config=config)
    dummy_x = jnp.zeros((1, config.input_size, config.n_series))
    variables = model.init(rng, dummy_x, deterministic=False)

    params     = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    if optimizer is None:
        if cosine_decay_steps > 0:
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=max(1, warmup_steps),
                decay_steps=cosine_decay_steps,
                end_value=learning_rate * 0.01,
            )
        else:
            lr_schedule = learning_rate

        base_opt = (
            optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
            if weight_decay > 0.0
            else optax.adam(learning_rate=lr_schedule)
        )
        optimizer = (
            optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
            if grad_clip > 0.0
            else base_opt
        )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("loss_fn",))
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: jax.Array,
    loss_fn: Callable = masked_mae,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled training step.

    Differentiates only w.r.t. ``state.params`` (trainable weights);
    ``state.batch_stats`` is updated via the running-average returned by
    ``model.apply(..., mutable=['batch_stats'])``.

    Returns ``(new_state, loss, predictions)``.
    """

    def compute(params):
        output, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch["insample_y"],
            deterministic=False,
            rngs={"dropout": rng},
            mutable=["batch_stats"],
        )
        loss = loss_fn(
            y=batch["outsample_y"],
            y_hat=output,
            mask=batch.get("sample_mask"),
        )
        return loss, (output, updates)

    (loss, (predictions, updates)), grads = jax.value_and_grad(
        compute, has_aux=True
    )(state.params)

    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(batch_stats=updates["batch_stats"])
    return new_state, loss, predictions


@partial(jax.jit, static_argnames=("loss_fn",))
def eval_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    loss_fn: Callable = masked_mae,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation step. Uses running BatchNorm statistics."""
    output = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch["insample_y"],
        deterministic=True,
    )
    loss = loss_fn(
        y=batch["outsample_y"],
        y_hat=output,
        mask=batch.get("sample_mask"),
    )
    return loss, output


# ---------------------------------------------------------------------------
# Scan loop (no batch_stats update — use train_step loop for BatchNorm)
# ---------------------------------------------------------------------------


@jax.jit
def scan_train_loop(
    state: TrainState,
    all_ins: jnp.ndarray,
    all_out: jnp.ndarray,
    all_rngs: jnp.ndarray,
) -> Tuple[TrainState, jnp.ndarray]:
    """Train for S steps via jax.lax.scan.

    Note: does not update batch_stats (BatchNorm running statistics). Use the
    Python ``train_step`` loop when the model uses BatchNorm.
    """

    def step_fn(carry: TrainState, xs: tuple) -> tuple:
        ins_b, out_b, step_rng = xs

        def compute(params):
            output, updates = carry.apply_fn(
                {"params": params, "batch_stats": carry.batch_stats},
                ins_b, deterministic=False, rngs={"dropout": step_rng},
                mutable=["batch_stats"],
            )
            return masked_mae(y=out_b, y_hat=output), (output, updates)

        (loss, (_, updates)), grads = jax.value_and_grad(compute, has_aux=True)(carry.params)
        new_carry = carry.apply_gradients(grads=grads)
        new_carry = new_carry.replace(batch_stats=updates["batch_stats"])
        return new_carry, loss

    return jax.lax.scan(step_fn, state, (all_ins, all_out, all_rngs))


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def train_loop(
    state: TrainState,
    train_batches: Iterable[Dict[str, jnp.ndarray]],
    *,
    num_epochs: int = 10,
    rng: Optional[jax.Array] = None,
    eval_batches: Optional[Iterable[Dict[str, jnp.ndarray]]] = None,
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

"""Training utilities for the Chronax RNN model.

Pure JAX/Flax/Optax. All long-lived state is held in
``flax.training.train_state.TrainState`` and all randomness is threaded
explicitly through ``jax.random.PRNGKey`` — there is no hidden global state.

* :func:`create_train_state`  - build a JIT-friendly train state from a config.
* :func:`train_step`           - JIT'd forward + backward + optax update.
* :func:`eval_step`            - JIT'd forward, no grads, deterministic.
* :func:`train_loop`           - convenience epoch loop on top of the above.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from chronax.models.rnn.loss import masked_mae
from chronax.models.rnn.model import RNN, RNNConfig


class TrainState(train_state.TrainState):
    """Standard Flax TrainState; aliased for forward compatibility."""

    pass


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def create_train_state(
    rng: jax.Array,
    config: RNNConfig,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> TrainState:
    """Initialise model parameters and optimiser state.

    Args:
        rng: PRNG key used for parameter initialisation only.
        config: model architecture config.
        learning_rate: passed to the default Adam(W) optimiser.
        weight_decay: if > 0 we use ``optax.adamw``, else ``optax.adam``.
        optimizer: optional pre-built optax optimiser; overrides the
            ``learning_rate`` / ``weight_decay`` defaults.
    """
    model = RNN(config=config)
    init_rng, _ = jax.random.split(rng)

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
        if weight_decay > 0.0:
            optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optax.adam(learning_rate=learning_rate)

    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------


def _forward_loss(
    params,
    apply_fn: Callable,
    batch: Dict[str, Optional[jnp.ndarray]],
    loss_fn: Callable,
    rng: Optional[jax.Array],
    deterministic: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run the model and compute the masked loss for ``batch``."""
    rngs = {"dropout": rng} if (rng is not None and not deterministic) else {}
    output, _ = apply_fn(
        params,
        insample_y=batch["insample_y"],
        hist_exog=batch.get("hist_exog"),
        futr_exog=batch.get("futr_exog"),
        stat_exog=batch.get("stat_exog"),
        deterministic=deterministic,
        rngs=rngs,
    )
    loss = loss_fn(
        y=batch["outsample_y"],
        y_hat=output,
        mask=batch.get("sample_mask"),
    )
    return loss, output


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("loss_fn",))
def train_step(
    state: TrainState,
    batch: Dict[str, Optional[jnp.ndarray]],
    rng: jax.Array,
    loss_fn: Callable = masked_mae,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled training step using ``jax.value_and_grad``.

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
    return _forward_loss(
        params=state.params,
        apply_fn=state.apply_fn,
        batch=batch,
        loss_fn=loss_fn,
        rng=None,
        deterministic=True,
    )


# ---------------------------------------------------------------------------
# Loop
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

    ``train_batches`` may be a list, generator factory, or any iterable. If it
    is a generator object that gets exhausted after one pass, you should
    materialise it (e.g. into a list) before calling this function.
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

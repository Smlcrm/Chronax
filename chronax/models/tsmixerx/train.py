"""Training utilities for the Chronax TSMixerx model.

Pure JAX/Flax/Optax. All long-lived state is held in a custom Flax
``TrainState`` that carries both model parameters and BatchNorm running
statistics (``batch_stats``) as separate pytree fields — a no-op empty dict
whenever ``config.use_batchnorm=False`` (the default, matching NF's
LayerNorm-based TSMixerx).

TSMixerx's forward pass returns a plain ``[B, h, N]`` tensor and optionally
accepts ``hist_exog`` / ``futr_exog`` / ``stat_exog``; the training step also
updates ``batch_stats`` via the mutable collection returned by
``model.apply(..., mutable=['batch_stats'])``.

Public API:
    :func:`create_train_state` — initialise model parameters + optimiser.
    :func:`train_step`          — JIT-compiled forward + backward + update.
    :func:`eval_step`           — JIT-compiled deterministic evaluation.
    :func:`train_loop`          — convenience epoch loop.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from chronax.models.tsmixerx.loss import masked_mae
from chronax.models.tsmixerx.model import TSMixerx, TSMixerxConfig


class TrainState(train_state.TrainState):
    """Extends Flax TrainState with a ``batch_stats`` field for BatchNorm.

    Keeping ``batch_stats`` separate from ``params`` ensures the Adam
    optimizer only tracks and updates the trainable parameters; running
    statistics are updated via exponential moving average inside
    ``train_step``, never via gradient descent.
    """

    batch_stats: Any


# ---------------------------------------------------------------------------
# Module-level model cache — ``TrainState.apply_fn`` (a bound method of the
# ``TSMixerx`` instance) is static w.r.t. ``jax.jit``, so a fresh instance per
# call forces XLA to retrace/recompile ``train_step``/``eval_step`` even for
# an identical config. Reusing the same instance for a given config (mirrors
# ``chronax.models.nbeatsx.train._get_model`` / ``tide.train._get_model``)
# lets JIT's cache hit across repeated ``fit()`` calls with the same config —
# e.g. every seed in a multi-seed benchmark loop.
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[TSMixerxConfig, TSMixerx] = {}


def _get_model(config: TSMixerxConfig) -> TSMixerx:
    if config not in _MODEL_CACHE:
        _MODEL_CACHE[config] = TSMixerx(config=config)
    return _MODEL_CACHE[config]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def create_train_state(
    rng: jax.Array,
    config: TSMixerxConfig,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    cosine_decay_steps: int = 0,
    warmup_steps: int = 0,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> TrainState:
    """Initialise model parameters, BatchNorm statistics, and optimiser state.

    Builds dummy exogenous inputs covering every optional code path so the
    parameter pytree always matches ``config`` regardless of which of
    ``hist_exog`` / ``futr_exog`` / ``stat_exog`` are used at train time.
    """
    model = _get_model(config)
    N = config.n_series
    dummy_x = jnp.zeros((1, config.input_size, N))
    dummy_hist = (
        jnp.zeros((1, config.hist_exog_size, config.input_size, N))
        if config.hist_exog_size > 0 else None
    )
    dummy_futr = (
        jnp.zeros((1, config.futr_exog_size, config.input_size + config.h, N))
        if config.futr_exog_size > 0 else None
    )
    dummy_stat = (
        jnp.zeros((N, config.stat_exog_size))
        if config.stat_exog_size > 0 else None
    )

    variables = model.init(
        rng, dummy_x,
        hist_exog=dummy_hist, futr_exog=dummy_futr, stat_exog=dummy_stat,
        deterministic=False,
    )

    params = variables["params"]
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
    batch: Dict[str, Optional[jnp.ndarray]],
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
            hist_exog=batch.get("hist_exog"),
            futr_exog=batch.get("futr_exog"),
            stat_exog=batch.get("stat_exog"),
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
    batch: Dict[str, Optional[jnp.ndarray]],
    loss_fn: Callable = masked_mae,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation step. Uses running BatchNorm statistics."""
    output = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch["insample_y"],
        hist_exog=batch.get("hist_exog"),
        futr_exog=batch.get("futr_exog"),
        stat_exog=batch.get("stat_exog"),
        deterministic=True,
    )
    loss = loss_fn(
        y=batch["outsample_y"],
        y_hat=output,
        mask=batch.get("sample_mask"),
    )
    return loss, output


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

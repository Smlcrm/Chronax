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
    ``sample_batch_indices`` -- NF-style window sampling (with/without replacement).
    ``should_stop_early``  -- patience-based early-stop check.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from chronax.models.fedformer.data import (
    RobustScaler,
    build_exog_windows,
    build_windows,
    split_train_val_windows,
)
from chronax.models.fedformer.loss import mae as _mae
from chronax.models.fedformer.loss import masked_mae
from chronax.models.fedformer.model import FEDformerConfig, FEDformerModel

import numpy as np


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
            # Match torch.optim.Adam defaults (b1=0.9, b2=0.999, eps=1e-8).
            base_opt = optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8)
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
    """Sample ``[n_steps, batch_size]`` window indices (NF-style).

    With replacement when ``n_train < batch_size`` (NF ``torch.randint``);
    without replacement via permutation slice otherwise (NF ``randperm[:B]``).
    """
    keys = jax.random.split(key, n_steps)
    if n_train < batch_size:
        def sample_one(k):
            return jax.random.choice(k, n_train, shape=(batch_size,), replace=True)
    else:
        def sample_one(k):
            return jax.random.permutation(k, n_train)[:batch_size]
    return jax.vmap(sample_one)(keys)


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
    if config.futr_exog_size > 0:
        dummy_f = jnp.zeros((1, config.input_size + config.h, config.futr_exog_size))
        params = model.init(
            {"params": init_rng, "dropout": init_rng},
            dummy_y,
            dummy_f,
            deterministic=True,
        )
    else:
        params = model.init(
            {"params": init_rng, "dropout": init_rng}, dummy_y, deterministic=True
        )

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


def _scale_exog(windows: jnp.ndarray, scaler: RobustScaler) -> jnp.ndarray:
    """Per-channel per-window robust scaling of ``[B, T, F]`` exog."""
    shift, scale = scaler.stats(windows, axis=1)  # [B, 1, F]
    return scaler.transform(windows, shift, scale)


def _window_forward_loss(
    params: Any,
    apply_fn: Callable,
    windows: jnp.ndarray,
    masks: jnp.ndarray,
    input_size: int,
    loss_fn: Callable,
    rng: Optional[jax.Array],
    deterministic: bool,
    futr_windows: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward + loss on raw ``[B, input_size + h]`` windows with per-window scaling.

    The history and target are scaled by the *history's* statistics so the model
    never sees the target's scale (no leakage); the loss is computed in scaled
    space, which keeps the gradient magnitudes comparable across windows.
    Padded horizon steps (NF right-pad) are excluded via ``masks``.
    """
    insample = windows[:, :input_size]   # [B, L]
    target = windows[:, input_size:]     # [B, h]
    out_mask = masks[:, input_size:]     # [B, h]
    shift, scale = _SCALER.stats(insample, axis=1)                      # [B, 1] each
    insample_z = _SCALER.transform(insample, shift, scale)[..., None]   # [B, L, 1]
    target_z = _SCALER.transform(target, shift, scale)                  # [B, h]
    futr_z = _scale_exog(futr_windows, _SCALER) if futr_windows is not None else None
    rngs = {"dropout": rng} if (rng is not None and not deterministic) else {}
    pred = apply_fn(
        params, insample_z, futr_exog=futr_z, deterministic=deterministic, rngs=rngs
    )  # [B, h, 1]
    loss = loss_fn(pred[..., 0], target_z, out_mask)
    return loss, pred


@partial(jax.jit, static_argnames=("loss_fn", "input_size"))
def train_window_step(
    state: TrainState,
    windows: jnp.ndarray,
    masks: jnp.ndarray,
    rng: jax.Array,
    input_size: int,
    loss_fn: Callable = _mae,
    futr_windows: Optional[jnp.ndarray] = None,
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
            masks=masks,
            input_size=input_size,
            loss_fn=loss_fn,
            rng=rng,
            deterministic=False,
            futr_windows=futr_windows,
        )

    (loss, predictions), grads = jax.value_and_grad(compute, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, predictions


@partial(jax.jit, static_argnames=("loss_fn", "input_size"), donate_argnums=(0,))
def scan_train_steps(
    state: TrainState,
    all_windows: jnp.ndarray,
    all_masks: jnp.ndarray,
    batch_idx: jnp.ndarray,
    rng_seq: jnp.ndarray,
    input_size: int,
    loss_fn: Callable = _mae,
    all_futr: Optional[jnp.ndarray] = None,
) -> Tuple[TrainState, jnp.ndarray]:
    """Run ``lax.scan`` over precomputed window indices (no per-step host sync).

    Args:
        all_windows: Full train windows ``[n_train, L+h]``.
        all_masks: Matching availability masks.
        batch_idx: ``[n_steps, batch_size]`` indices into ``all_windows``.
        rng_seq: ``[n_steps, 2]`` dropout keys.
        all_futr: Optional exog windows ``[n_train, L+h, F]``.
    """

    def body(carry: TrainState, xs):
        idx, rng = xs
        windows = all_windows[idx]
        masks = all_masks[idx]
        futr = all_futr[idx] if all_futr is not None else None

        def compute(params):
            return _window_forward_loss(
                params=params,
                apply_fn=carry.apply_fn,
                windows=windows,
                masks=masks,
                input_size=input_size,
                loss_fn=loss_fn,
                rng=rng,
                deterministic=False,
                futr_windows=futr,
            )

        (loss, _), grads = jax.value_and_grad(compute, has_aux=True)(carry.params)
        return carry.apply_gradients(grads=grads), loss

    return jax.lax.scan(body, state, (batch_idx, rng_seq))


def _raise_if_nonfinite(losses: jnp.ndarray) -> None:
    """Host-side finite check after a scan chunk (KAN-style)."""
    try:
        losses_host = np.asarray(losses)
    except jax.errors.TracerArrayConversionError:
        return
    bad = ~np.isfinite(losses_host)
    if bad.any():
        first_bad = int(np.argmax(bad))
        raise RuntimeError(
            f"Non-finite loss ({losses_host[first_bad]}) at step {first_bad}. "
            "Training diverged. Lower learning_rate or batch_size, or check the series."
        )


@partial(jax.jit, static_argnames=("loss_fn", "input_size"))
def eval_window_step(
    state: TrainState,
    windows: jnp.ndarray,
    masks: jnp.ndarray,
    input_size: int,
    loss_fn: Callable = _mae,
    futr_windows: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled evaluation on raw windows. Deterministic, no gradients."""
    return _window_forward_loss(
        params=state.params,
        apply_fn=state.apply_fn,
        windows=windows,
        masks=masks,
        input_size=input_size,
        loss_fn=loss_fn,
        rng=None,
        deterministic=True,
        futr_windows=futr_windows,
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


# ---------------------------------------------------------------------------
# High-level univariate train / predict (BaseForecaster path)
# ---------------------------------------------------------------------------


def _snapshot(state: TrainState) -> dict:
    """Deep-copy params for best-checkpoint restore."""
    return jax.tree.map(jnp.asarray, state.params)


def _restore(state: TrainState, saved_params: dict) -> TrainState:
    return state.replace(params=saved_params)


def predict_step(
    state: TrainState,
    context: jnp.ndarray,
    *,
    h: int,
    input_size: int,
    scaler: Optional[RobustScaler] = None,
    futr_full: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Deterministic forecast from a 1-D context window of length ``input_size``.

    Returns a 1-D array of shape ``(h,)`` on the original scale.
    ``futr_full`` is optional future-known exog of shape ``[input_size + h, F]``.
    """
    if scaler is None:
        scaler = _SCALER
    insample = jnp.asarray(context, dtype=jnp.float32).ravel()
    if insample.shape[0] != input_size:
        raise ValueError(
            f"context length {insample.shape[0]} != input_size={input_size}."
        )
    x = insample[None, :]  # [1, L]
    shift, scale = scaler.stats(x, axis=1)
    x_z = scaler.transform(x, shift, scale)[..., None]  # [1, L, 1]
    futr_z = None
    if futr_full is not None:
        futr = jnp.asarray(futr_full, dtype=jnp.float32)[None, ...]  # [1, L+h, F]
        futr_z = _scale_exog(futr, scaler)
    pred_z = state.apply_fn(
        state.params, x_z, futr_exog=futr_z, deterministic=True
    )  # [1, h, 1]
    return scaler.inverse(pred_z[0, :, 0], shift[0, 0], scale[0, 0])


def train(
    y: jnp.ndarray,
    *,
    config: FEDformerConfig,
    max_steps: int = 1000,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    num_lr_decays: int = 3,
    val_fraction: float = 0.1,
    val_check_steps: int = 100,
    early_stop_patience_steps: int = -1,
    grad_clip: float = 1.0,
    weight_decay: float = 0.0,
    loss_fn: Callable = _mae,
    random_seed: int = 1,
    verbose: bool = False,
    futr_exog: Optional[jnp.ndarray] = None,
) -> TrainState:
    """Train on a univariate 1-D series; return best-validation ``TrainState``.

    Raises ``RuntimeError`` if a non-finite training loss is observed.
    ``futr_exog`` optional ``[T, F]`` future-known covariates aligned with ``y``.
    """
    y_np = np.asarray(y, dtype=np.float32).ravel()
    all_windows, all_masks = build_windows(y_np, config.input_size, config.h)
    (train_windows_np, train_masks_np), (val_windows_np, val_masks_np) = (
        split_train_val_windows(all_windows, all_masks, val_fraction=val_fraction)
    )
    n_train = len(train_windows_np)
    has_val = len(val_windows_np) > 0
    n_all = len(all_windows)

    train_futr_dev = None
    val_futr_dev = None
    if futr_exog is not None:
        futr_all = build_exog_windows(
            futr_exog, config.input_size, config.h, n_all, span="full"
        )
        n_val = len(val_windows_np)
        if n_val > 0:
            train_futr_np = np.asarray(futr_all[:-n_val])
            val_futr_np = np.asarray(futr_all[-n_val:])
            val_futr_dev = jax.device_put(jnp.asarray(val_futr_np))
        else:
            train_futr_np = np.asarray(futr_all)
        train_futr_dev = jax.device_put(jnp.asarray(train_futr_np))

    train_windows_dev = jax.device_put(jnp.asarray(train_windows_np))
    train_masks_dev = jax.device_put(jnp.asarray(train_masks_np))
    val_windows_dev = (
        jax.device_put(jnp.asarray(val_windows_np)) if has_val else None
    )
    val_masks_dev = (
        jax.device_put(jnp.asarray(val_masks_np)) if has_val else None
    )

    rng = jax.random.PRNGKey(random_seed)
    init_rng, rng = jax.random.split(rng)
    state = create_train_state(
        init_rng,
        config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        num_lr_decays=num_lr_decays,
        max_steps=max_steps,
    )

    sample_rng, drop_rng = jax.random.split(rng)
    batch_idx = sample_batch_indices(sample_rng, n_train, batch_size, max_steps)
    step_keys = jax.random.split(drop_rng, max_steps)
    # Prefetch all batches on device once. Per-step JIT + deferred finite
    # check (no host sync each step). scan_train_steps remains available for
    # callers that want a pure lax.scan over a chunk.
    batch_windows = train_windows_dev[batch_idx]  # [max_steps, B, L+h]
    batch_masks = train_masks_dev[batch_idx]
    batch_futr = train_futr_dev[batch_idx] if train_futr_dev is not None else None

    best_params = _snapshot(state)
    best_val_loss = float("inf")
    checks_without_improvement = 0
    input_size = config.input_size
    chunk = val_check_steps if val_check_steps > 0 else max_steps
    all_losses: List[jnp.ndarray] = []

    for step in range(max_steps):
        futr_b = batch_futr[step] if batch_futr is not None else None
        state, loss, _ = train_window_step(
            state,
            batch_windows[step],
            batch_masks[step],
            step_keys[step],
            input_size,
            loss_fn,
            futr_windows=futr_b,
        )
        all_losses.append(loss)

        at_chunk_end = (step + 1) % chunk == 0 or step == max_steps - 1
        if not at_chunk_end:
            continue

        _raise_if_nonfinite(jnp.stack(all_losses))

        if verbose:
            msg = f"  step {step:>5}: train_loss={float(loss):.5f}"
            if has_val:
                val_loss_val, _ = eval_window_step(
                    state, val_windows_dev, val_masks_dev,
                    input_size=input_size, loss_fn=loss_fn,
                    futr_windows=val_futr_dev,
                )
                msg += f"  val_loss={float(val_loss_val):.5f}"
            print(msg)

        if not has_val:
            continue

        val_loss_val, _ = eval_window_step(
            state, val_windows_dev, val_masks_dev,
            input_size=input_size, loss_fn=loss_fn,
            futr_windows=val_futr_dev,
        )
        val_scalar = float(val_loss_val)
        if val_scalar < best_val_loss:
            best_val_loss = val_scalar
            best_params = _snapshot(state)
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        if should_stop_early(
            early_stop_patience_steps=early_stop_patience_steps,
            checks_without_improvement=checks_without_improvement,
        ):
            if verbose:
                print(f"  Early stopping at step {step + 1}.")
            break

    if has_val:
        state = _restore(state, best_params)
    return state

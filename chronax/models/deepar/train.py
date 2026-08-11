"""Training loop for DeepAR (JAX/FLAX/OPTAX, NumPy-free)."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax

from .loss import nll_gaussian_masked


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


class TrainState:
    """Lightweight training-state container (params + optimizer state)."""

    def __init__(
        self,
        params: dict,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
    ):
        self.params = params
        self.opt_state = opt_state
        self.optimizer = optimizer
        self.step = 0

    def apply_gradients(self, grads: dict) -> "TrainState":
        updates, opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)
        new_state = TrainState(params, opt_state, self.optimizer)
        new_state.step = self.step + 1
        return new_state


# ---------------------------------------------------------------------------
# Loss function builder
# ---------------------------------------------------------------------------


def make_loss_fn(model: nn.Module, h: int) -> Callable:
    """Return a loss function ``(params, batch, rng) -> scalar``.

    The batch dict contains:
        insample_y:   [B, L, 1]
        outsample_y:  [B, h, 1]
        sample_mask:  [B, h, 1]
        futr_exog:    [B, L+h, F] or None
        stat_exog:    [B, S] or None

    Lag features (lag-1/7/h/2h) are always concatenated onto ``futr_exog`` so
    the ``/proj`` input width matches :func:`forecast_mc` (1 + F + 4).
    """
    from .model import _build_lag_features

    def loss_fn(params: dict, batch: dict, rng: jnp.ndarray) -> jnp.ndarray:
        y_hist = batch["insample_y"]      # [B, L, 1]
        y_true = batch["outsample_y"]     # [B, h, 1]
        avail_mask = batch["available_mask"] # [B, L, 1]
        sample_mask = batch["sample_mask"]  # [B, h, 1]

        futr_exog = batch.get("futr_exog")   # [B, L+h, F] or None
        stat_exog = batch.get("stat_exog")   # [B, S] or None

        # Concatenate history and future to form full sequence
        y_seq = jnp.concatenate([y_hist, y_true], axis=1) # [B, L+h, 1]
        mask_seq = jnp.concatenate([avail_mask, sample_mask], axis=1) # [B, L+h, 1]

        def single_series_loss(y_s, mask_s, futr_s, stat_s):
            # y_s: [L+h, 1], mask_s: [L+h, 1], futr_s: [L+h, F] or None, stat_s: [S]
            y_full = y_s[:, 0]  # [L+h]
            lags = _build_lag_features(y_full, h)  # [L+h, 4]
            xf = lags if futr_s is None else jnp.concatenate([futr_s, lags], -1)

            y_in = y_full[:-1]  # [L+h-1]
            y_target = y_s[1:]  # [L+h-1, 1]
            m_target = mask_s[1:]  # [L+h-1, 1]
            f_in = xf[1:]  # [L+h-1, F+4]

            mu, sigma = model.apply(
                params,
                y_in,
                f_in,    # futr_exog (+ lags)
                stat_s,
                True,    # training
                rngs={"dropout": rng},
            )  # mu, sigma: [L+h-1]

            nll = nll_gaussian_masked(
                y_target[None, ...], # [1, L+h-1, 1]
                mu[None, ...],       # [1, L+h-1]
                sigma[None, ...],    # [1, L+h-1]
                m_target[None, ...], # [1, L+h-1, 1]
            )
            return nll

        # Vmap over batch dimension. If futr_exog or stat_exog is None, we set in_axes to None for them.
        in_axes = (0, 0, 0 if futr_exog is not None else None, 0 if stat_exog is not None else None)
        
        batch_losses = jax.vmap(single_series_loss, in_axes=in_axes)(
            y_seq,
            mask_seq,
            futr_exog,
            stat_exog,
        )
        return jnp.mean(batch_losses)

    return loss_fn


# ---------------------------------------------------------------------------
# Single training step (JIT-compiled)
# ---------------------------------------------------------------------------


def make_train_step(model: nn.Module, loss_fn: Callable) -> Callable:
    """Return a JIT-compiled ``(state, batch, rng) -> (state, metrics)``."""

    @jax.jit
    def train_step(
        state: TrainState,
        batch: dict,
        rng: jnp.ndarray,
    ) -> Tuple[TrainState, dict]:
        loss, grads = jax.value_and_grad(
            lambda p: loss_fn(p, batch, rng)
        )(state.params)
        state = state.apply_gradients(grads)
        return state, {"loss": loss}

    return train_step


# ---------------------------------------------------------------------------
# High-level training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_data: List,
    num_steps: int,
    batch_size: int,
    input_size: int = 100,
    h: int = 24,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    warmup_steps: int = 100,
    grad_clip: float = 1.0,
    valid_data: Optional[List] = None,
    early_stop_patience: int = -1,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[dict, List, List]:
    """Train a DeepAR model.

    Args:
        model: DeepAR_EncDec instance.
        train_data: List of (y_series, exog_dict) tuples.
        num_steps: Total training steps.
        batch_size: Batch size.
        input_size: History window length used to build batches.
        h: Forecast horizon used to build batches.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight-decay.
        warmup_steps: Linear warmup steps.
        grad_clip: Gradient clipping norm.
        valid_data: Optional validation data.
        early_stop_patience: -1 disables early stopping.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        (params, train_losses, valid_losses)
    """
    from .data import create_batch

    key = random.PRNGKey(seed)
    key, init_key = random.split(key)

    # Dummy init: use a single constant series so shapes are inferred
    dummy_y = jnp.ones(input_size, dtype=jnp.float32)

    params = model.init(
        {"params": init_key, "dropout": init_key},
        dummy_y,
        None,   # hist_exog
        None,   # futr_exog
        None,   # x_static
        True,
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(num_steps - warmup_steps, 1),
        end_value=1e-6,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    state = TrainState(params, tx.init(params), tx)

    loss_fn = make_loss_fn(model, h=h)
    train_step_fn = make_train_step(model, loss_fn)

    train_losses: List[float] = []
    valid_losses: List[float] = []

    for step in range(num_steps):
        key, bkey, skey = random.split(key, 3)
        # Sample batch_size indices
        indices = random.randint(bkey, (batch_size,), 0, len(train_data))
        batch_y = [jnp.array(train_data[int(i)][0]) for i in indices]
        
        batch = create_batch(y_series=batch_y, input_size=input_size, h=h)

        state, metrics = train_step_fn(state, batch, skey)
        train_losses.append(float(metrics["loss"]))

        if verbose and (step + 1) % 100 == 0:
            avg = jnp.mean(jnp.array(train_losses[-100:]))
            print(f"Step {step+1:5d} | Loss: {float(avg):.4f}")

    return state.params, train_losses, valid_losses

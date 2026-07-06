"""High-level fit/forecast adapter around the Chronax FEDformer.

Mirrors the ``neuralforecast.NeuralForecast(models=[FEDformer(...)]).fit(df).predict()``
flow but with a lightweight ``fit(y) -> self`` / ``forecast(y) -> ndarray`` API.

Training recipe (matches neuralforecast's windowed convention):
  - Build every rolling window of length ``input_size + h`` from the series.
  - Apply a per-window ``RobustScaler`` (median / MAD) on each forward pass.
  - Track the best-validation parameters and restore them before returning.

Both univariate (a single 1-D array) and panel (a list of 1-D arrays) inputs are
supported. Panel mode trains a *single shared model* over all series, exactly
like the upstream implementation.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from chronax.models.fedformer.data import (
    RobustScaler,
    build_windows,
    split_train_val_windows,
)
from chronax.models.fedformer.loss import LossFn, resolve
from chronax.models.fedformer.model import FEDformerConfig
from chronax.models.fedformer.train import (
    TrainState,
    _get_model,
    create_train_state,
    eval_window_step,
    should_stop_early,
    train_window_step,
)


SeriesLike = Union[np.ndarray, Sequence[np.ndarray]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_series_list(y: SeriesLike) -> List[np.ndarray]:
    """Normalise ``y`` to a list of 1-D float32 arrays."""
    if isinstance(y, np.ndarray):
        return [y.astype(np.float32, copy=False).ravel()]
    return [np.asarray(s, dtype=np.float32).ravel() for s in y]


def _concat_windows(y_list: List[np.ndarray], input_size: int, h: int) -> np.ndarray:
    """Concatenate rolling windows from all series into ``[N, input_size+h]``."""
    parts: List[np.ndarray] = []
    for y in y_list:
        if len(y) < input_size + h:
            continue  # series too short to yield even one window
        parts.append(build_windows(y, input_size, h))
    if not parts:
        raise ValueError(
            f"No series is long enough to extract windows "
            f"(input_size={input_size}, h={h})."
        )
    return np.concatenate(parts, axis=0)


def _snapshot(state: TrainState) -> dict:
    """Deep-copy params (for best-checkpoint restore)."""
    return jax.tree.map(jnp.asarray, state.params)


def _restore(state: TrainState, saved_params: dict) -> TrainState:
    return state.replace(params=saved_params)


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class FEDformerForecaster:
    """High-level fit/forecast wrapper around :class:`FEDformerModel`.

    Args:
        config: Model architecture config (must include ``h`` and ``input_size``).
        max_steps: Total optimiser steps.
        learning_rate: Adam peak learning rate.
        batch_size: Windows sampled per training step.
        num_lr_decays: StepLR decays (gamma=0.5 each); -1 = constant LR.
        val_fraction: Fraction of windows held out for validation (0 = none).
        val_check_steps: Evaluate validation loss every this many steps.
        early_stop_patience_steps: Stop if val loss does not improve for this
            many consecutive checks; -1 = no early stopping.
        grad_clip: Global gradient-norm clip (0 = disabled).
        weight_decay: AdamW weight decay (0 = plain Adam).
        loss: Window-level loss; string key or callable.
        seed: PRNG seed for parameter init and window sampling.
    """

    def __init__(
        self,
        config: FEDformerConfig,
        *,
        max_steps: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_lr_decays: int = 3,
        val_fraction: float = 0.1,
        val_check_steps: int = 100,
        early_stop_patience_steps: int = -1,
        grad_clip: float = 1.0,
        weight_decay: float = 0.0,
        loss: "str | LossFn" = "mae",
        seed: int = 1,
    ):
        self.config = config
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_lr_decays = num_lr_decays
        self.val_fraction = val_fraction
        self.val_check_steps = val_check_steps
        self.early_stop_patience_steps = early_stop_patience_steps
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.loss = resolve(loss)
        self.seed = seed

        self._state: Optional[TrainState] = None
        self._scaler: RobustScaler = RobustScaler()

    # ------------------------------------------------------------------ properties

    @property
    def state(self) -> TrainState:
        if self._state is None:
            raise RuntimeError("FEDformerForecaster has not been fit yet.")
        return self._state

    @property
    def fitted(self) -> bool:
        return self._state is not None

    # ------------------------------------------------------------------ fit

    def fit(self, y: SeriesLike, *, verbose: bool = False) -> "FEDformerForecaster":
        """Train the model in-place on ``y``.

        Args:
            y: A single 1-D series or a list of 1-D series (panel mode).
            verbose: Print train/validation loss periodically.

        Returns:
            ``self`` (for chaining).
        """
        y_list = _as_series_list(y)
        cfg = self.config

        # Build every rolling window from every series.
        all_windows = _concat_windows(y_list, cfg.input_size, cfg.h)  # [N, L+h]

        # Chronological train/val split.
        train_windows_np, val_windows_np = split_train_val_windows(
            all_windows, val_fraction=self.val_fraction
        )
        n_train = len(train_windows_np)
        has_val = len(val_windows_np) > 0

        # One-time host -> device transfer.
        train_windows_dev = jax.device_put(jnp.asarray(train_windows_np))
        val_windows_dev = (
            jax.device_put(jnp.asarray(val_windows_np)) if has_val else None
        )

        # Initialise train state.
        rng = jax.random.PRNGKey(self.seed)
        init_rng, rng = jax.random.split(rng)
        self._state = create_train_state(
            init_rng,
            cfg,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            num_lr_decays=self.num_lr_decays,
            max_steps=self.max_steps,
        )

        np_rng = np.random.RandomState(self.seed)
        best_params = _snapshot(self._state)
        best_val_loss = float("inf")
        checks_without_improvement = 0
        log_every = max(1, self.max_steps // 10)

        for step in range(self.max_steps):
            rng, step_rng = jax.random.split(rng)
            # Sample a batch of windows with replacement.
            win_idx = np_rng.choice(n_train, size=self.batch_size, replace=True)
            windows_batch = train_windows_dev[win_idx]

            self._state, loss, _ = train_window_step(
                self._state, windows_batch, step_rng,
                input_size=cfg.input_size, loss_fn=self.loss,
            )

            if verbose and (step % log_every == 0 or step == self.max_steps - 1):
                msg = f"  step {step:>5}: train_loss={float(loss):.5f}"
                if has_val:
                    val_loss_val, _ = eval_window_step(
                        self._state, val_windows_dev,
                        input_size=cfg.input_size, loss_fn=self.loss,
                    )
                    msg += f"  val_loss={float(val_loss_val):.5f}"
                print(msg)

            # Validation check + early stopping.
            if has_val and (step + 1) % self.val_check_steps == 0:
                val_loss_val, _ = eval_window_step(
                    self._state, val_windows_dev,
                    input_size=cfg.input_size, loss_fn=self.loss,
                )
                val_scalar = float(val_loss_val)
                if val_scalar < best_val_loss:
                    best_val_loss = val_scalar
                    best_params = _snapshot(self._state)
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1

                if should_stop_early(
                    early_stop_patience_steps=self.early_stop_patience_steps,
                    checks_without_improvement=checks_without_improvement,
                ):
                    if verbose:
                        print(f"  Early stopping at step {step + 1}.")
                    break

        # Restore the best-validation checkpoint.
        if has_val:
            self._state = _restore(self._state, best_params)

        return self

    # ------------------------------------------------------------------ forecast

    def forecast(
        self,
        y: Optional[SeriesLike] = None,
        h: Optional[int] = None,
    ) -> np.ndarray:
        """Predict the horizon from the last ``input_size`` points of each series.

        Args:
            y: Series to forecast for (required; training data is not retained).
            h: Optional horizon check -- must equal ``config.h`` if given.

        Returns:
            ``[h]`` for univariate input, or ``[n_series, h]`` for panel input.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .forecast().")
        if y is None:
            raise ValueError(
                "y must be provided to forecast (training data is not retained)."
            )
        if h is not None and h != self.config.h:
            raise ValueError(
                f"This forecaster was configured with h={self.config.h}; got h={h}. "
                f"Construct a new forecaster for a different horizon."
            )

        y_list = _as_series_list(y)
        squeeze = isinstance(y, np.ndarray)
        cfg = self.config
        model = _get_model(cfg)
        preds_list: List[np.ndarray] = []

        for series in y_list:
            # Use the most recent input_size points as the encoder context.
            insample = jnp.asarray(series[-cfg.input_size:], dtype=jnp.float32)[None, :]
            # Per-window robust scaling (then invert on the output).
            shift, scale = self._scaler.stats(insample, axis=1)              # [1, 1]
            insample_z = self._scaler.transform(insample, shift, scale)[..., None]
            pred_z = model.apply(self._state.params, insample_z, deterministic=True)
            pred = self._scaler.inverse(pred_z[0, :, 0], shift[0, 0], scale[0, 0])
            preds_list.append(np.asarray(pred))

        preds = np.stack(preds_list, axis=0)  # [N, h]
        return preds[0] if squeeze else preds

    # ------------------------------------------------------------------ convenience

    def fit_predict(self, y: SeriesLike, **kwargs) -> np.ndarray:
        """Fit on ``y`` then immediately forecast for the same series."""
        fit_kwargs = {k: v for k, v in kwargs.items() if k == "verbose"}
        self.fit(y, **fit_kwargs)
        return self.forecast(y=y)

    def with_config(self, **overrides) -> "FEDformerForecaster":
        """Return a fresh (unfitted) forecaster with overridden config fields."""
        new_cfg = replace(self.config, **overrides)
        return FEDformerForecaster(
            new_cfg,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            num_lr_decays=self.num_lr_decays,
            val_fraction=self.val_fraction,
            val_check_steps=self.val_check_steps,
            early_stop_patience_steps=self.early_stop_patience_steps,
            grad_clip=self.grad_clip,
            weight_decay=self.weight_decay,
            loss=self.loss,
            seed=self.seed,
        )

    def __repr__(self) -> str:
        return (
            f"FEDformerForecaster(config={asdict(self.config)}, "
            f"max_steps={self.max_steps}, learning_rate={self.learning_rate}, "
            f"batch_size={self.batch_size}, fitted={self.fitted})"
        )

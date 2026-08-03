"""High-level fit/forecast adapter around Chronax N-BEATS.

Mirrors ``neuralforecast.NeuralForecast(models=[NBEATS(...)]).fit(df).predict()``
with a familiar ``fit(y) -> self`` / ``forecast(y) -> ndarray`` interface.

Training strategy:
  - Extract all rolling windows from the input series.
  - Apply per-window ``RobustScaler`` (median / MAD) inside each JIT step.
  - Optionally hold out the latest windows for validation + early stopping.
  - Restore the best-validation checkpoint before returning.

Both univariate (single 1-D array) and panel (list of 1-D arrays) modes are
supported.  Panel mode trains a single shared model over the union of all
series, exactly like the upstream NeuralForecast implementation.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Callable, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.nbeats.data import (
    RobustScaler,
    build_windows,
    split_train_val_windows,
)
from chronax.models.nbeats.loss import LossFn
from chronax.models.nbeats.loss import resolve as _resolve_loss
from chronax.models.nbeats.model import NBEATS, NBEATSConfig
from chronax.models.nbeats.train import (
    TrainState,
    _get_model,
    create_train_state,
    eval_step,
    train_step,
)


SeriesLike = Union[jnp.ndarray, Sequence[jnp.ndarray]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_series_list(y: SeriesLike) -> List[jnp.ndarray]:
    """Normalise ``y`` to a list of 1-D float32 series.

    Anything that isn't a ``list``/``tuple`` of series (e.g. a plain
    ``jax.numpy`` array, as passed by the benchmark harness) is treated as a
    single univariate series.
    """
    if not isinstance(y, (list, tuple)):
        return [jnp.asarray(y, dtype=jnp.float32).ravel()]
    return [jnp.asarray(s, dtype=jnp.float32).ravel() for s in y]


def _concat_windows(y_list: List[jnp.ndarray], input_size: int, h: int) -> jnp.ndarray:
    """Collect all rolling windows from every series into ``[N, L+h]``."""
    parts: List[jnp.ndarray] = []
    for y in y_list:
        if y.shape[0] < input_size + h:
            continue
        parts.append(build_windows(y, input_size, h))
    if not parts:
        raise ValueError(
            f"No series is long enough to extract windows "
            f"(input_size={input_size}, h={h})."
        )
    return jnp.concatenate(parts, axis=0)


def _snapshot(state: TrainState) -> dict:
    return jax.tree.map(jnp.asarray, state.params)


def _restore(state: TrainState, saved_params: dict) -> TrainState:
    return state.replace(params=saved_params)


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class NBEATSForecaster(BaseForecaster):
    """High-level fit/forecast wrapper around :class:`~chronax.models.nbeats.model.NBEATS`.

    Args:
        h: Forecast horizon.
        input_size: History window length; -1 (default) uses ``3 * h``.
        stack_types, n_blocks, mlp_units, mlp_layers, n_harmonics, n_basis,
        basis, activation, shared_weights, dropout_prob, layer_norm: forwarded
        to :class:`NBEATSConfig`.
        max_steps: Total optimiser steps.
        learning_rate: Adam(W) peak learning rate.
        batch_size: Windows sampled per training step.
        random_seed: PRNG seed for parameter init and window sampling.
        alias: Display name for external reporting.
        loss: Registered name (``"mae"``, ``"mse"``) from
            :mod:`chronax.models.nbeats.loss` or a callable ``(y, y_hat) -> scalar``.
        scale: Apply per-window ``RobustScaler`` during training and inference.
            Strongly recommended; matches NeuralForecast's ``scaler_type``.
        grad_clip: Global gradient-norm clip (0 = disabled).
        weight_decay: AdamW weight decay (0 = plain Adam).
        use_lr_schedule: Use warmup-cosine LR schedule; when False uses
            constant ``learning_rate``.
        window_sampling: Use all rolling windows (True, recommended) vs only
            the last window per series (False).
        val_fraction: Fraction of windows held out for validation (0 = none).
        val_check_steps: Evaluate validation loss every this many steps.
        early_stop_patience_steps: Stop when val loss does not improve for
            this many consecutive checks; -1 = disabled.
    """

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        stack_types: Tuple[str, ...] = ("identity", "trend", "seasonality"),
        n_blocks: Tuple[int, ...] = (1, 1, 1),
        mlp_units: int = 512,
        mlp_layers: int = 4,
        n_harmonics: int = 2,
        n_basis: int = 2,
        basis: str = "polynomial",
        activation: str = "relu",
        shared_weights: bool = False,
        dropout_prob: float = 0.0,
        layer_norm: bool = True,
        *,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 1024,
        random_seed: int = 0,
        alias: str = "NBEATS",
        loss: Union[str, LossFn] = "mae",
        scale: bool = False,
        grad_clip: float = 0.0,
        weight_decay: float = 0.0,
        use_lr_schedule: bool = False,
        num_lr_decays: int = 3,
        window_sampling: bool = True,
        val_fraction: float = 0.1,
        val_check_steps: int = 100,
        early_stop_patience_steps: int = -1,
    ):
        if input_size < 1:
            input_size = 3 * h
        self.config = NBEATSConfig(
            h=h,
            input_size=input_size,
            stack_types=stack_types,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            mlp_layers=mlp_layers,
            n_harmonics=n_harmonics,
            n_basis=n_basis,
            basis=basis,
            activation=activation,
            shared_weights=shared_weights,
            dropout_prob=dropout_prob,
            layer_norm=layer_norm,
        )
        self.h = h
        self.input_size = input_size
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.seed = random_seed
        self.alias = alias
        self.loss = loss
        self.scale = scale
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.use_lr_schedule = use_lr_schedule
        self.num_lr_decays = num_lr_decays
        self.window_sampling = window_sampling
        self.val_fraction = val_fraction
        self.val_check_steps = val_check_steps
        self.early_stop_patience_steps = early_stop_patience_steps

        self._state: Optional[TrainState] = None
        self._scaler: RobustScaler = RobustScaler()
        self._fit_y: Optional[SeriesLike] = None

    @property
    def _loss_fn(self) -> LossFn:
        """Resolve ``self.loss`` to a callable. Lazy so string forms pickle."""
        return _resolve_loss(self.loss)

    # ------------------------------------------------------------------ properties

    @property
    def state(self) -> TrainState:
        if self._state is None:
            raise RuntimeError("NBEATSForecaster has not been fit yet.")
        return self._state

    @property
    def fitted(self) -> bool:
        return self._state is not None

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        y: SeriesLike,
        *,
        verbose: bool = False,
    ) -> "NBEATSForecaster":
        """Train the model in-place on ``y``.

        Args:
            y: A single 1-D series or a list of 1-D series (panel mode).
            verbose: Print training / validation loss every
                ``max(1, max_steps // 10)`` steps.

        Returns:
            ``self`` (for chaining).
        """
        y_list = _as_series_list(y)
        self._fit_y = y
        cfg = self.config

        # Build all rolling windows from every series
        all_windows = _concat_windows(y_list, cfg.input_size, cfg.h)  # [N, L+h]

        if not self.window_sampling:
            # Use only the last window from each series (no augmentation)
            all_windows = all_windows[[-1]]

        train_wins, val_wins = split_train_val_windows(
            all_windows, val_fraction=self.val_fraction
        )
        n_train = train_wins.shape[0]
        has_val = val_wins.shape[0] > 0

        # Upload to device once
        train_wins = jax.device_put(train_wins)
        val_wins = jax.device_put(val_wins) if has_val else None

        # Initialise train state
        rng = jax.random.PRNGKey(self.seed)
        init_rng, rng = jax.random.split(rng)
        warmup = max(1, self.max_steps // 10) if self.use_lr_schedule else 0
        self._state = create_train_state(
            init_rng, cfg,
            learning_rate=self.learning_rate,
            grad_clip=self.grad_clip,
            cosine_decay_steps=self.max_steps if self.use_lr_schedule else 0,
            warmup_steps=warmup,
            num_lr_decays=self.num_lr_decays if not self.use_lr_schedule else -1,
            max_steps=self.max_steps,
            weight_decay=self.weight_decay,
        )

        best_params = _snapshot(self._state)
        best_val_loss = float("inf")
        checks_no_improve = 0
        log_every = max(1, self.max_steps // 10)

        for step in range(self.max_steps):
            rng, idx_rng, step_rng = jax.random.split(rng, 3)
            idx = jax.random.randint(idx_rng, (self.batch_size,), 0, n_train)
            windows_batch = train_wins[idx]

            self._state, loss, _ = train_step(
                self._state, windows_batch, step_rng,
                input_size=cfg.input_size, loss_fn=self._loss_fn,
                scale=self.scale,
            )

            if verbose and (step % log_every == 0 or step == self.max_steps - 1):
                msg = f"  step {step:>5}: train_loss={float(loss):.5f}"
                if has_val:
                    val_loss, _ = eval_step(
                        self._state, val_wins,
                        input_size=cfg.input_size, loss_fn=self._loss_fn,
                        scale=self.scale,
                    )
                    msg += f"  val_loss={float(val_loss):.5f}"
                print(msg)

            # Validation checkpoint + early stopping
            if has_val and (step + 1) % self.val_check_steps == 0:
                val_loss, _ = eval_step(
                    self._state, val_wins,
                    input_size=cfg.input_size, loss_fn=self._loss_fn,
                    scale=self.scale,
                )
                val_scalar = float(val_loss)
                if val_scalar < best_val_loss:
                    best_val_loss = val_scalar
                    best_params = _snapshot(self._state)
                    checks_no_improve = 0
                else:
                    checks_no_improve += 1

                if (
                    self.early_stop_patience_steps > 0
                    and checks_no_improve >= self.early_stop_patience_steps
                ):
                    if verbose:
                        print(f"  Early stopping at step {step + 1}.")
                    break

        if has_val:
            self._state = _restore(self._state, best_params)

        return self

    # ------------------------------------------------------------------ forecast

    def forecast(
        self,
        y: Optional[SeriesLike] = None,
        h: Optional[int] = None,
    ) -> jnp.ndarray:
        """Produce point forecasts from the last ``input_size`` of each series.

        Args:
            y: Series to forecast for (required; training data is not retained).
            h: Horizon override — must match ``config.h`` if provided.

        Returns:
            ``jnp.ndarray`` of shape ``[h]`` for univariate input or
            ``[n_series, h]`` for panel input.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .forecast().")
        if y is None:
            raise ValueError("y must be provided to forecast (training data is not retained).")
        if h is not None and h != self.config.h:
            raise ValueError(
                f"This forecaster was configured with h={self.config.h}; "
                f"got h={h}. Construct a new forecaster for a different horizon."
            )

        y_list = _as_series_list(y)
        squeeze = not isinstance(y, (list, tuple))
        cfg = self.config
        model = _get_model(cfg)
        preds_list: List[jnp.ndarray] = []

        for series in y_list:
            insample = jnp.asarray(series[-cfg.input_size:], dtype=jnp.float32)[None, :]  # [1, L]
            if self.scale:
                shift, scale = self._scaler.stats(insample, axis=1)    # [1, 1]
                insample_z = self._scaler.transform(insample, shift, scale)
            else:
                insample_z = insample
                shift, scale = jnp.zeros((1, 1)), jnp.ones((1, 1))

            forecast_z = model.apply(
                self._state.params, insample_z, deterministic=True
            )  # [1, h]

            if self.scale:
                forecast_out = self._scaler.inverse(forecast_z, shift, scale)
            else:
                forecast_out = forecast_z

            preds_list.append(forecast_out[0])

        preds = jnp.stack(preds_list, axis=0)  # [N, h]
        return preds[0] if squeeze else preds

    # ------------------------------------------------------------------ convenience

    def predict(
        self,
        h: Optional[int] = None,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[Union[int, float]]] = None,
    ) -> dict:
        """Forecast from the series passed to :meth:`fit`.

        Satisfies the ``BaseForecaster`` contract: unlike :meth:`forecast`,
        this does not take ``y`` explicitly — it reuses the series cached at
        fit time.

        Args:
            h: Forecast horizon. Defaults to ``config.h`` when None.
            X: Reserved for future exogenous regressors; unused.
            level: Not yet supported for this model.

        Returns:
            dict: ``{"mean": jnp.ndarray}``.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .predict().")
        if level is not None:
            raise NotImplementedError(
                "NBEATSForecaster.predict(level=...) is not yet supported."
            )
        preds = self.forecast(y=self._fit_y, h=h)
        return {"mean": jnp.asarray(preds)}

    def fit_predict(self, y: SeriesLike, **kwargs) -> jnp.ndarray:
        """Fit on ``y`` and immediately forecast for the same series."""
        fit_kwargs = {k: v for k, v in kwargs.items() if k == "verbose"}
        self.fit(y, **fit_kwargs)
        return self.forecast(y=y)

    def with_config(self, **overrides) -> "NBEATSForecaster":
        """Return a fresh (unfitted) forecaster with overridden config fields."""
        new_cfg = replace(self.config, **overrides)
        return NBEATSForecaster(
            **asdict(new_cfg),
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            alias=self.alias,
            loss=self.loss,
            scale=self.scale,
            grad_clip=self.grad_clip,
            weight_decay=self.weight_decay,
            use_lr_schedule=self.use_lr_schedule,
            num_lr_decays=self.num_lr_decays,
            window_sampling=self.window_sampling,
            val_fraction=self.val_fraction,
            val_check_steps=self.val_check_steps,
            early_stop_patience_steps=self.early_stop_patience_steps,
        )

    def __repr__(self) -> str:
        return (
            f"NBEATSForecaster(config={asdict(self.config)}, "
            f"max_steps={self.max_steps}, learning_rate={self.learning_rate}, "
            f"batch_size={self.batch_size}, fitted={self.fitted})"
        )

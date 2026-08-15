"""High-level fit/forecast adapter for Chronax TiDE.

Mirrors ``neuralforecast.NeuralForecast(models=[TiDE(...)]).fit(df).predict()``
with a familiar ``fit(y) -> self`` / ``forecast(y) -> ndarray`` interface.

Training strategy (identical to the NBEATS forecaster):
  - Extract all rolling windows from the input series.
  - Apply per-window ``RobustScaler`` (median/MAD) inside each JIT step.
  - Optionally hold out the latest windows for validation + early stopping.
  - Restore the best-validation checkpoint before returning.

Both univariate (single 1-D array) and panel (list of 1-D arrays) modes are
supported.  Panel mode trains a single shared model over the union of all
series, exactly like the upstream NeuralForecast implementation.

Exogenous variables (hist_exog, futr_exog, stat_exog) require the lower-level
``train_step`` / ``eval_step`` / ``create_batch`` API from this package.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Callable, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.tide.data import (
    RobustScaler,
    build_windows,
    split_train_val_windows,
)
from chronax.models.tide.loss import LossFn
from chronax.models.tide.loss import resolve as _resolve_loss
from chronax.models.tide.model import TiDE, TiDEConfig
from chronax.models.tide.train import (
    TrainState,
    create_train_state,
    eval_step_windows,
    eval_step_windows_raw,
    eval_step_windows_std,
    train_step_windows,
    train_step_windows_raw,
    train_step_windows_std,
)


SeriesLike = Union[jnp.ndarray, Sequence[jnp.ndarray]]

_scaler = RobustScaler()


# ---------------------------------------------------------------------------
# JIT-compiled inference builder
# ---------------------------------------------------------------------------


_STD_EPS: float = 1e-6


def _make_jit_infer(model: TiDE, scale: bool, use_std_scaler: bool = False) -> Callable:
    """Return a JIT-compiled function ``(params, insample [1,L]) → [1,h]``.

    Fuses scaler + forward pass + inverse-scaler into a single XLA kernel so
    warm ``forecast()`` calls are as fast as a plain matrix-multiply chain.
    ``model`` is captured in the closure (Flax modules are frozen dataclasses,
    so JAX can hash them and cache the trace correctly).
    """
    if scale and use_std_scaler:
        @jax.jit
        def _infer(params: dict, insample: jnp.ndarray) -> jnp.ndarray:
            shift = jnp.mean(insample, axis=1, keepdims=True)
            sc    = jnp.std(insample,  axis=1, keepdims=True) + _STD_EPS
            pred_z = model.apply(params, ((insample - shift) / sc)[:, :, None], deterministic=True)
            return pred_z[:, :, 0] * sc + shift                         # [1, h]
    elif scale:
        @jax.jit
        def _infer(params: dict, insample: jnp.ndarray) -> jnp.ndarray:
            shift, scale_val = _scaler.stats(insample, axis=1)
            insample_z = _scaler.transform(insample, shift, scale_val)
            pred_z = model.apply(params, insample_z[:, :, None], deterministic=True)
            return _scaler.inverse(pred_z[:, :, 0], shift, scale_val)   # [1, h]
    else:
        @jax.jit
        def _infer(params: dict, insample: jnp.ndarray) -> jnp.ndarray:
            pred_z = model.apply(params, insample[:, :, None], deterministic=True)
            return pred_z[:, :, 0]                                       # [1, h]
    return _infer


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
    """Collect all rolling windows from every long-enough series."""
    parts: List[jnp.ndarray] = []
    for y in y_list:
        if y.shape[0] >= input_size + h:
            parts.append(build_windows(y, input_size, h))
    if not parts:
        raise ValueError(
            f"No series is long enough to extract windows "
            f"(input_size={input_size}, h={h})."
        )
    return jnp.concatenate(parts, axis=0)   # [N_total, L + h]


def _snapshot(state: TrainState) -> dict:
    return jax.tree.map(jnp.asarray, state.params)


def _restore(state: TrainState, saved_params: dict) -> TrainState:
    return state.replace(params=saved_params)


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class TiDEForecaster(BaseForecaster):
    """High-level fit/forecast wrapper for :class:`~chronax.models.tide.model.TiDE`.

    Args:
        h: Forecast horizon.
        input_size: History window length; -1 (default) uses ``3 * h``.
        hidden_size, decoder_output_dim, temporal_decoder_dim, dropout,
        layernorm, num_encoder_layers, num_decoder_layers, temporal_width,
        futr_exog_size, hist_exog_size, stat_exog_size, output_size: forwarded
        to :class:`TiDEConfig`.
        max_steps: Total optimiser steps.
        learning_rate: Adam(W) peak learning rate.
        batch_size: Windows sampled per training step.
        random_seed: PRNG seed for parameter init and window sampling.
        alias: Display name for external reporting.
        loss: Registered name (``"mae"``, ``"mse"``) from
            :mod:`chronax.models.tide.loss` or a callable ``(y, y_hat) -> scalar``.
        scale: Apply per-window ``RobustScaler`` during training and inference.
            Strongly recommended; matches NeuralForecast's ``scaler_type``.
        grad_clip: Global gradient-norm clip (0 = disabled).
        weight_decay: AdamW weight decay (0 = plain Adam).
        use_lr_schedule: Warmup-cosine LR schedule; False = constant LR.
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
        hidden_size: int = 512,
        decoder_output_dim: int = 32,
        temporal_decoder_dim: int = 128,
        dropout: float = 0.3,
        layernorm: bool = True,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        temporal_width: int = 4,
        futr_exog_size: int = 0,
        hist_exog_size: int = 0,
        stat_exog_size: int = 0,
        output_size: int = 1,
        *,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 1024,
        random_seed: int = 0,
        alias: str = "TiDE",
        loss: Union[str, LossFn] = "mae",
        scale: bool = True,
        global_scale: bool = False,
        use_std_scaler: Optional[bool] = True,
        grad_clip: float = 0.0,
        weight_decay: float = 0.0,
        use_lr_schedule: bool = False,
        window_sampling: bool = True,
        val_fraction: float = 0.1,
        val_check_steps: int = 100,
        early_stop_patience_steps: int = -1,
    ):
        if input_size < 1:
            input_size = 3 * h
        self.config = TiDEConfig(
            h=h,
            input_size=input_size,
            hidden_size=hidden_size,
            decoder_output_dim=decoder_output_dim,
            temporal_decoder_dim=temporal_decoder_dim,
            dropout=dropout,
            layernorm=layernorm,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            temporal_width=temporal_width,
            futr_exog_size=futr_exog_size,
            hist_exog_size=hist_exog_size,
            stat_exog_size=stat_exog_size,
            output_size=output_size,
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
        self.global_scale = global_scale
        self.use_std_scaler = use_std_scaler
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.use_lr_schedule = use_lr_schedule
        self.window_sampling = window_sampling
        self.val_fraction = val_fraction
        self.val_check_steps = val_check_steps
        self.early_stop_patience_steps = early_stop_patience_steps

        self._state: Optional[TrainState] = None
        self._jit_infer: Optional[Callable] = None
        self._global_scalers: Optional[List[tuple]] = None  # [(mean, std), ...] per series
        self._fit_y: Optional[SeriesLike] = None

    @property
    def _loss_fn(self) -> LossFn:
        """Resolve ``self.loss`` to a callable. Lazy so string forms pickle."""
        return _resolve_loss(self.loss)

    # ------------------------------------------------------------------ properties

    @property
    def state(self) -> TrainState:
        if self._state is None:
            raise RuntimeError("TiDEForecaster has not been fit yet.")
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
    ) -> "TiDEForecaster":
        """Train the model in-place on ``y``.

        All rolling windows are extracted from every series long enough to
        yield at least one window. Per-window ``RobustScaler`` is applied
        inside the JIT-compiled ``train_step_windows``.

        Args:
            y: A single 1-D series or a list of 1-D series (panel mode).
            verbose: Print loss every ``max(1, max_steps // 10)`` steps.

        Returns:
            ``self`` (for chaining).
        """
        y_list = _as_series_list(y)
        self._fit_y = y
        cfg = self.config

        # Auto-select scaler: mean/std for short series (<1000 pts), median/MAD otherwise.
        if self.use_std_scaler is None:
            train_len = max((len(s) for s in y_list), default=0)
            _use_std_scaler = train_len < 1000
        else:
            _use_std_scaler = self.use_std_scaler

        if self.global_scale:
            # Global per-series StandardScaler — matches NF's scaler_type="standard".
            # Fit on the full training series; all windows from that series are
            # normalised with the SAME statistics, preserving level/trend information
            # across windows (which per-window centering would erase).
            self._global_scalers = []
            y_for_windows: List[jnp.ndarray] = []
            for s in y_list:
                g_mean = float(jnp.mean(s))
                g_std  = max(float(jnp.std(s)), 1e-6)
                self._global_scalers.append((g_mean, g_std))
                y_for_windows.append(((s - g_mean) / g_std).astype(jnp.float32))
        else:
            y_for_windows = y_list

        all_windows = _concat_windows(y_for_windows, cfg.input_size, cfg.h)  # [N, L+h]

        if not self.window_sampling:
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

            if self.global_scale:
                self._state, loss, _ = train_step_windows_raw(
                    self._state, windows_batch, step_rng,
                    input_size=cfg.input_size, loss_fn=self._loss_fn,
                )
            elif _use_std_scaler:
                self._state, loss, _ = train_step_windows_std(
                    self._state, windows_batch, step_rng,
                    input_size=cfg.input_size, loss_fn=self._loss_fn,
                )
            else:
                self._state, loss, _ = train_step_windows(
                    self._state, windows_batch, step_rng,
                    input_size=cfg.input_size, loss_fn=self._loss_fn,
                )

            if verbose and (step % log_every == 0 or step == self.max_steps - 1):
                msg = f"  step {step:>5}: train_loss={float(loss):.5f}"
                if has_val:
                    _eval_fn = (eval_step_windows_raw if self.global_scale
                                else eval_step_windows_std if _use_std_scaler
                                else eval_step_windows)
                    val_loss, _ = _eval_fn(
                        self._state, val_wins,
                        input_size=cfg.input_size, loss_fn=self._loss_fn,
                    )
                    msg += f"  val_loss={float(val_loss):.5f}"
                print(msg)

            # Validation checkpoint + early stopping
            if has_val and (step + 1) % self.val_check_steps == 0:
                _eval_fn = (eval_step_windows_raw if self.global_scale
                            else eval_step_windows_std if _use_std_scaler
                            else eval_step_windows)
                val_loss, _ = _eval_fn(
                    self._state, val_wins,
                    input_size=cfg.input_size, loss_fn=self._loss_fn,
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

        # Build and pre-warm the fused JIT inference function.
        # When global_scale=True the input to _jit_infer is already globally
        # normalised; no per-window scaler should be applied (scale=False).
        model = TiDE(cfg)
        infer_scale = False if self.global_scale else self.scale
        self._jit_infer = _make_jit_infer(model, infer_scale,
                                          use_std_scaler=_use_std_scaler)
        _ = jax.block_until_ready(
            self._jit_infer(self._state.params, jnp.zeros((1, cfg.input_size)))
        )

        return self

    # ------------------------------------------------------------------ forecast

    def forecast(
        self,
        y: Optional[SeriesLike] = None,
        h: Optional[int] = None,
    ) -> jnp.ndarray:
        """Produce point forecasts from the last ``input_size`` of each series.

        Per-window ``RobustScaler`` statistics are computed on the last
        ``input_size`` observations of each series (matching the scaler applied
        during training) and the forecast is inverse-transformed before return.

        Args:
            y: Series to forecast for (required).
            h: Horizon override — must match ``config.h`` if provided.

        Returns:
            ``jnp.ndarray`` of shape ``[h]`` for univariate input or
            ``[n_series, h]`` for panel input.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .forecast().")
        if y is None:
            raise ValueError("y must be provided (training data is not retained).")
        if h is not None and h != self.config.h:
            raise ValueError(
                f"This forecaster was configured with h={self.config.h}; "
                f"got h={h}. Construct a new forecaster for a different horizon."
            )

        y_list = _as_series_list(y)
        squeeze = not isinstance(y, (list, tuple))
        cfg = self.config
        preds_list: List[jnp.ndarray] = []

        for i, series in enumerate(y_list):
            if self.global_scale and self._global_scalers is not None:
                # Apply the same global scaler that was fit on the training series.
                # Falls back to stats from the provided series if the index is new.
                if i < len(self._global_scalers):
                    g_mean, g_std = self._global_scalers[i]
                else:
                    g_mean = float(jnp.mean(series))
                    g_std  = max(float(jnp.std(series)), 1e-6)
                series_norm = ((series - g_mean) / g_std).astype(jnp.float32)
                insample = jnp.asarray(
                    series_norm[-cfg.input_size:], dtype=jnp.float32
                )[None, :]                                          # [1, L] globally normalised
                # _jit_infer has scale=False; it returns globally-normalised predictions
                pred_norm = self._jit_infer(self._state.params, insample)  # [1, h]
                pred_out = pred_norm[0] * g_std + g_mean                   # [h] original scale
            else:
                insample = jnp.asarray(
                    series[-cfg.input_size:], dtype=jnp.float32
                )[None, :]                                          # [1, L]
                # _jit_infer has scale=True; it applies per-window scaler internally
                pred_out = self._jit_infer(self._state.params, insample)[0]  # [h]
            preds_list.append(pred_out)

        preds = jnp.stack(preds_list, axis=0)                        # [N, h]
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
                "TiDEForecaster.predict(level=...) is not yet supported."
            )
        preds = self.forecast(y=self._fit_y, h=h)
        return {"mean": jnp.asarray(preds)}

    def fit_predict(self, y: SeriesLike, **kwargs) -> jnp.ndarray:
        """Fit on ``y`` and immediately forecast for the same series."""
        fit_kwargs = {k: v for k, v in kwargs.items() if k == "verbose"}
        self.fit(y, **fit_kwargs)
        return self.forecast(y=y)

    def with_config(self, **overrides) -> "TiDEForecaster":
        """Return a fresh (unfitted) forecaster with overridden config fields."""
        new_cfg = replace(self.config, **overrides)
        return TiDEForecaster(
            **asdict(new_cfg),
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            alias=self.alias,
            loss=self.loss,
            scale=self.scale,
            global_scale=self.global_scale,
            use_std_scaler=self.use_std_scaler,
            grad_clip=self.grad_clip,
            weight_decay=self.weight_decay,
            use_lr_schedule=self.use_lr_schedule,
            window_sampling=self.window_sampling,
            val_fraction=self.val_fraction,
            val_check_steps=self.val_check_steps,
            early_stop_patience_steps=self.early_stop_patience_steps,
        )

    def __repr__(self) -> str:
        return (
            f"TiDEForecaster(config={asdict(self.config)}, "
            f"max_steps={self.max_steps}, learning_rate={self.learning_rate}, "
            f"batch_size={self.batch_size}, fitted={self.fitted})"
        )

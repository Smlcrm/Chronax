"""High-level fit/forecast adapter for Chronax TiDE.

Mirrors ``neuralforecast.NeuralForecast(models=[TiDE(...)]).fit(df).predict()``
with a familiar ``fit(y) -> self`` / ``forecast(y) -> ndarray`` interface.

Training strategy (the same scheme as the NBEATS forecaster):

  - Extract all rolling windows from the input series.
  - Normalise inside each JIT step: per-window mean/std by default,
    per-window ``RobustScaler`` (median / median absolute deviation, MAD)
    with ``use_std_scaler=False``, or per-series global mean/std with
    ``global_scale=True``.
  - Optionally hold out the latest windows for validation + early stopping.
  - Restore the best-validation checkpoint before returning.

Both univariate (single 1-D array) and panel (list of 1-D arrays) modes are
supported.  Panel mode trains a single shared model over the union of all
series, exactly like the upstream NeuralForecast implementation.

Exogenous variables (hist_exog, futr_exog, stat_exog) require the lower-level
``train_step`` / ``eval_step`` / ``create_batch`` functions from this package.
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

    Fuses scaler + forward pass + inverse-scaler into a single XLA (accelerated
    linear algebra) kernel so
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
    ``jax.numpy`` array) is treated as a
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

    Port of neuralforecast's ``TiDE`` training and inference flow. Trains
    one shared model over a single series or a panel of series. Point
    forecasts only: :meth:`predict` refuses ``level`` and :meth:`forecast`
    returns a bare array rather than the :class:`BaseForecaster` dict, so
    the base conformal machinery is not available for this model.
    Exogenous inputs are not accepted by this wrapper; use the lower-level
    ``train_step`` / ``create_batch`` functions of the package for them.

    Args:
        h (int): Forecast horizon.
        input_size (int): History window length; ``-1`` (default) uses
            ``3 * h``.
        hidden_size (int): Hidden width of the encoder and decoder residual
            blocks. Default ``512``.
        decoder_output_dim (int): Per-step output dimension of the dense
            decoder. Default ``32``.
        temporal_decoder_dim (int): Hidden size of the temporal decoder
            block. Default ``128``.
        dropout (float): Dropout rate (``0`` = disabled). Default ``0.3``.
        layernorm (bool): Apply LayerNorm after each residual block.
            Default ``True``.
        num_encoder_layers (int): Stacked encoder residual blocks. Default
            ``1``.
        num_decoder_layers (int): Stacked decoder residual blocks. Default
            ``1``.
        temporal_width (int): Projected feature dimension of temporal
            covariates. Default ``4``.
        futr_exog_size (int): Number of future exogenous features ``F``.
            Default ``0``.
        hist_exog_size (int): Number of historic exogenous features ``X``.
            Default ``0``.
        stat_exog_size (int): Number of static exogenous features ``S``.
            Default ``0``.
        output_size (int): Outputs per step; ``1`` for point forecasts.
            Default ``1``.
        max_steps (int): Total optimiser steps. Default ``1000``.
        learning_rate (float): Adam or AdamW peak learning rate. Default
            ``1e-3``.
        batch_size (int): Windows sampled per training step. Default
            ``1024``.
        random_seed (int): PRNG seed for parameter init and window sampling.
            Default ``0``.
        alias (str): Display name for external reporting. Default
            ``"TiDE"``.
        loss (str | LossFn): Registered name (``"mae"``, ``"mse"``) from
            :mod:`chronax.models.tide.loss` or a callable
            ``(y, y_hat) -> scalar``. Default ``"mae"``.
        scale (bool): Apply the per-window scaler inside the fused inference
            function. The training step normalises every window regardless
            of this flag, so ``False`` leaves inference unnormalised while
            training was normalised. Ignored when ``global_scale=True``.
            Default ``True``.
        global_scale (bool): Normalise each series with its own training
            mean and std, shared by all windows of that series, and apply no
            per-window scaler; the series statistics are stored and reused
            by :meth:`forecast`. Default ``False``.
        use_std_scaler (bool | None): Per-window scaler choice when
            ``global_scale=False``: ``True`` (default) uses the window
            mean and std, ``False`` uses the ``RobustScaler`` (median and
            median absolute deviation, MAD), ``None`` picks mean/std when
            the longest series has fewer than 1000 points and the
            ``RobustScaler`` otherwise.
        grad_clip (float): Global gradient-norm clip (``0`` = disabled).
            Default ``0.0``.
        weight_decay (float): AdamW weight decay (``0`` = plain Adam).
            Default ``0.0``.
        use_lr_schedule (bool): ``True`` uses a linear-warmup, cosine-decay
            schedule over ``max_steps`` with warmup ``max_steps // 10``;
            ``False`` (default) keeps the learning rate constant.
        window_sampling (bool): Train on all rolling windows (``True``,
            default) or only the last window of the panel (``False``).
        val_fraction (float): Fraction of the latest windows held out for
            validation (``0`` = none). The best validation checkpoint is
            restored at the end of :meth:`fit`. Default ``0.1``.
        val_check_steps (int): Evaluate the validation loss every this many
            steps. Default ``100``.
        early_stop_patience_steps (int): Stop when the validation loss has
            not improved for this many consecutive checks; ``-1``
            (default) disables early stopping.
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
        yield at least one window. Normalisation runs inside the
        JIT-compiled training step as selected by ``global_scale`` and
        ``use_std_scaler``. The fused inference function is built and
        warmed up at the end of the call.

        Args:
            y (jnp.ndarray | Sequence[jnp.ndarray]): A single 1-D series of
                shape ``(T,)`` or a list of 1-D series (panel mode).
            verbose (bool): Print training and validation loss every
                ``max(1, max_steps // 10)`` steps. Default ``False``.

        Returns:
            TiDEForecaster: ``self``.

        Raises:
            ValueError: If no series is at least ``input_size + h`` long.
        """
        y_list = _as_series_list(y)
        self._fit_y = y
        cfg = self.config

        # use_std_scaler=None: mean/std for series under 1000 points, median/MAD otherwise.
        if self.use_std_scaler is None:
            train_len = max((len(s) for s in y_list), default=0)
            _use_std_scaler = train_len < 1000
        else:
            _use_std_scaler = self.use_std_scaler

        if self.global_scale:
            # Global per-series StandardScaler (neuralforecast's scaler_type="standard").
            # Fit on the full training series; all windows from that series are
            # normalised with the same statistics, so level and trend information
            # survives across windows (per-window centring would erase it).
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
        """Produce point forecasts from the last ``input_size`` points of each series.

        Uses the fitted parameters; does not re-fit. When ``scale=True`` the
        context window is normalised with per-window statistics of the
        context (or with the stored per-series statistics when
        ``global_scale=True``) and the forecast is inverse-transformed before
        return. Unlike the :class:`BaseForecaster` contract this returns a
        bare array, not a dict.

        Args:
            y (jnp.ndarray | Sequence[jnp.ndarray] | None): Series to
                forecast, shape ``(T,)`` or a list of 1-D series. Required;
                ``None`` raises because training data is not retained here
                (:meth:`predict` reuses the series passed to :meth:`fit`).
            h (int | None): Horizon; must equal ``config.h`` when given.

        Returns:
            jnp.ndarray: Shape ``(h,)`` for a single series or
            ``(n_series, h)`` for panel input.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``y`` is ``None``, or ``h`` differs from
                ``config.h``.
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

        Unlike :meth:`forecast`, this does not take ``y``; it reuses the
        series cached at fit time.

        Args:
            h (int | None): Forecast horizon; must equal ``config.h`` when
                given. Defaults to ``config.h``.
            X (jnp.ndarray | None): Accepted for interface parity; ignored.
            level (list[int] | None): Not supported; any value raises.

        Returns:
            dict: ``{"mean": jnp.ndarray}`` with ``"mean"`` of shape ``(h,)``
            for a single series or ``(n_series, h)`` for panel input. No
            interval keys are ever emitted.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            NotImplementedError: If ``level`` is not ``None``.
            ValueError: If ``h`` differs from ``config.h``.
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

"""Autoformer forecaster: BaseForecaster wrapper around the Flax Linen backbone.

Univariate direct-decoding Autoformer matching neuralforecast.Autoformer for the
no-exogenous case. Training uses rolling windows with per-window RobustScaler;
``model_`` holds a Flax ``TrainState`` after fit.
"""
from __future__ import annotations

import warnings
from typing import Callable, Union

import jax.numpy as jnp

from chronax.models.autoformer.data import RobustScaler
from chronax.models.autoformer.loss import LossFn, resolve as _resolve_loss
from chronax.models.autoformer.model import AutoformerConfig
from chronax.models.autoformer.train import TrainState, predict_step, train
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals


class Autoformer(BaseForecaster):
    """Univariate Autoformer forecaster (JAX/Flax Linen port of neuralforecast.Autoformer).

    Maintenance Status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via
        ``predict(level=...)``. Defaults match neuralforecast.Autoformer.
    """

    uses_exog = True

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        hidden_size: int = 128,
        n_heads: int = 4,
        factor: int = 3,
        moving_avg_window: int = 25,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        conv_hidden_size: int = 32,
        decoder_input_size_multiplier: float = 0.5,
        dropout: float = 0.05,
        activation: str = "gelu",
        max_steps: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_lr_decays: int = 3,
        val_fraction: float = 0.1,
        val_check_steps: int = 100,
        early_stop_patience_steps: int = -1,
        grad_clip: float = 1.0,
        weight_decay: float = 0.0,
        random_seed: int = 1,
        alias: str = "Autoformer",
        loss: Union[str, LossFn] = "mae",
    ):
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.factor = factor
        self.moving_avg_window = moving_avg_window
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.conv_hidden_size = conv_hidden_size
        self.decoder_input_size_multiplier = decoder_input_size_multiplier
        self.dropout = dropout
        self.activation = activation
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_lr_decays = num_lr_decays
        self.val_fraction = val_fraction
        self.val_check_steps = val_check_steps
        self.early_stop_patience_steps = early_stop_patience_steps
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.alias = alias
        self.loss = loss
        self.conformal_params: ConformalIntervals | None = None
        self.model_: TrainState | None = None
        self._context: jnp.ndarray | None = None
        self._train_y: jnp.ndarray | None = None
        self._futr_size: int = 0
        self._futr_ctx: jnp.ndarray | None = None
        self._scaler = RobustScaler()

    @property
    def _loss_fn(self) -> LossFn:
        return _resolve_loss(self.loss)

    @property
    def _has_temporal_exog(self) -> bool:
        return self._futr_size > 0

    def _build_config(self) -> AutoformerConfig:
        return AutoformerConfig(
            h=self.h,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            factor=self.factor,
            moving_avg_window=self.moving_avg_window,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            conv_hidden_size=self.conv_hidden_size,
            decoder_input_size_multiplier=self.decoder_input_size_multiplier,
            dropout=self.dropout,
            activation=self.activation,
            futr_exog_size=self._futr_size,
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None, *, futr_exog=None) -> "Autoformer":
        """Fit on a univariate 1-D series.

        Future-known exogenous: pass ``futr_exog`` of shape ``(T, F)`` aligned
        with ``y``. Historical ``X=`` is not supported.
        """
        if X is not None:
            raise NotImplementedError(
                "Autoformer supports future-known exog only; pass futr_exog="
            )
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] < self.input_size + self.h:
            raise ValueError(
                f"Series length {y.shape[0]} too short for "
                f"input_size={self.input_size} + h={self.h}."
            )
        futr_exog = None if futr_exog is None else jnp.asarray(futr_exog, jnp.float32)
        if futr_exog is not None and futr_exog.shape[0] != y.shape[0]:
            raise ValueError(
                f"futr_exog must align with y at fit (len {y.shape[0]}); "
                f"got {futr_exog.shape[0]}."
            )
        self._futr_size = 0 if futr_exog is None else int(futr_exog.shape[1])

        self.model_ = train(
            y,
            config=self._build_config(),
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            num_lr_decays=self.num_lr_decays,
            val_fraction=self.val_fraction,
            val_check_steps=self.val_check_steps,
            early_stop_patience_steps=self.early_stop_patience_steps,
            grad_clip=self.grad_clip,
            weight_decay=self.weight_decay,
            loss_fn=self._loss_fn,
            random_seed=self.random_seed,
            futr_exog=futr_exog,
        )
        self._context = y[-self.input_size :]
        self._train_y = y
        self._futr_ctx = None if futr_exog is None else futr_exog[-self.input_size :]
        return self

    def _raw_predict(self, futr_exog=None) -> jnp.ndarray:
        futr_full = None
        if self._futr_size > 0:
            if futr_exog is None:
                raise ValueError(
                    "This Autoformer was fit with future-known exog; predict requires "
                    f"futr_exog of shape (h={self.h}, F={self._futr_size})."
                )
            futr_exog = jnp.asarray(futr_exog, jnp.float32)
            if futr_exog.shape != (self.h, self._futr_size):
                raise ValueError(
                    f"futr_exog must be (h={self.h}, F={self._futr_size}); "
                    f"got {futr_exog.shape}."
                )
            futr_full = jnp.concatenate([self._futr_ctx, futr_exog], axis=0)
        return predict_step(
            self.model_,
            self._context,
            h=self.h,
            input_size=self.input_size,
            scaler=self._scaler,
            futr_full=futr_full,
        )

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        *,
        futr_exog=None,
    ) -> dict:
        """Forecast ``h`` steps from the fitted context."""
        if self.model_ is None or self._context is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        if h < 1:
            raise ValueError(f"h must be a positive integer; got h={h}.")
        if h > self.h:
            raise ValueError(
                f"Autoformer was trained for h={self.h}; predict(h={h}) is not supported. "
                f"Pass h <= {self.h} or re-fit with a larger h."
            )
        full = self._raw_predict(futr_exog=futr_exog)
        fcst = {"mean": full[:h]}
        if level is not None:
            if self._has_temporal_exog:
                raise ValueError(
                    "Conformal intervals are not supported with temporal "
                    "(future-known) exog. Omit level=..."
                )
            if self.conformal_params is None:
                raise ValueError(
                    "predict(h, level=...) requires `model.conformal_params` to be set. "
                    "Note: conformity_scores re-fits the model per CV window."
                )
            if self._train_y is None:
                raise RuntimeError("Call fit(y) before predict(h, level=...).")
            cs = self.new().conformity_scores(self._train_y)
            fcst = BaseForecaster.add_confidence_intervals(
                fcst, cs, level, self.conformal_params.method
            )
        return fcst

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
        *,
        futr_exog=None,
    ) -> dict:
        """Stateless fit-then-predict on ``y``.

        ``X`` is unsupported (future-known only). ``futr_exog`` = history
        ``(T, F)``; ``X_future`` = horizon ``(h, F)``.
        """
        if X is not None:
            raise NotImplementedError(
                "Autoformer supports future-known exog only; pass futr_exog= / X_future="
            )
        self.fit(y, futr_exog=futr_exog)
        result = self.predict(h=h, futr_exog=X_future, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
        """One-step-ahead fitted values; first ``input_size`` entries are NaN."""
        if self._train_y is None or self.model_ is None:
            raise RuntimeError("Call fit(y) before computing fitted values.")
        if self._has_temporal_exog:
            raise NotImplementedError(
                "Fitted values are not supported when the model was fit with futr_exog."
            )
        y = self._train_y
        n_windows = y.shape[0] - self.input_size
        if n_windows <= 0:
            return jnp.full((y.shape[0],), jnp.nan, dtype=jnp.float32)
        idx = jnp.arange(self.input_size)[None, :] + jnp.arange(n_windows)[:, None]
        in_windows = y[idx]
        shift, scale = self._scaler.stats(in_windows, axis=1)
        x_z = self._scaler.transform(in_windows, shift, scale)[..., None]
        pred_z = self.model_.apply_fn(self.model_.params, x_z, deterministic=True)
        first_step_z = pred_z[:, 0, 0:1]
        finite_part = self._scaler.inverse(first_step_z, shift, scale)[:, 0]
        nan_head = jnp.full((self.input_size,), jnp.nan, dtype=jnp.float32)
        return jnp.concatenate([nan_head, finite_part])


class AutoformerForecaster(Autoformer):
    """Deprecated config-based constructor; prefer :class:`Autoformer`."""

    def __init__(
        self,
        config: AutoformerConfig,
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
        loss: Union[str, LossFn, Callable] = "mae",
        seed: int = 1,
    ):
        warnings.warn(
            "AutoformerForecaster(config=...) is deprecated; "
            "use Autoformer(h=..., input_size=..., random_seed=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            h=config.h,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            factor=config.factor,
            moving_avg_window=config.moving_avg_window,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            conv_hidden_size=config.conv_hidden_size,
            decoder_input_size_multiplier=config.decoder_input_size_multiplier,
            dropout=config.dropout,
            activation=config.activation,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_lr_decays=num_lr_decays,
            val_fraction=val_fraction,
            val_check_steps=val_check_steps,
            early_stop_patience_steps=early_stop_patience_steps,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            random_seed=seed,
            alias="Autoformer",
            loss=loss if not callable(loss) or isinstance(loss, str) else loss,
        )
        # Allow callables that are already resolved LossFn
        if callable(loss) and not isinstance(loss, str):
            self.loss = loss
        if getattr(config, "futr_exog_size", 0):
            self._futr_size = int(config.futr_exog_size)

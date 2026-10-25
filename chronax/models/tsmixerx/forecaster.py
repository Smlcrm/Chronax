"""High-level fit/forecast adapter around the Chronax TSMixerx.

TSMixerx is a multivariate model — all N series must be provided together as
a single ``[T, N]`` array (or a list of N equal-length 1-D arrays). A single
shared model processes all channels simultaneously.

Normalisation strategy — per-window (on by default):
    The insample mean/std of each extracted window is computed and both the
    insample and outsample portions of that window are normalised with those
    statistics (matches NeuralForecast's ``scaler_type="standard"``), on top
    of the model's own internal RevIN. Empirically this materially reduces
    seed-to-seed accuracy variance versus relying on RevIN alone (NF's
    default is ``scaler_type="identity"`` — set ``scale=False`` to match it).

Future-known exogenous covariates (``futr_exog``, ``n_series=1`` only) are
supported directly through :meth:`fit` / :meth:`forecast` / :meth:`predict`
(``uses_exog = True``): the exogenous feature width is not known at
construction time (the benchmark harness, like NF, builds the forecaster
before it has seen any data), so ``self.config`` is rebuilt with the observed
``futr_exog_size`` the first time :meth:`fit` sees exog data — mirroring the
lazy-config pattern already used by :class:`~chronax.models.mlp.mlp_model.MLP`.
Historic (``hist_exog``) and static (``stat_exog``) covariates, and
``n_series > 1`` with exog, still require the lower-level
``create_windows_exog`` / ``make_batch_exog`` / ``train_step`` API — construct
:class:`~chronax.models.tsmixerx.model.TSMixerx` directly with a config
carrying ``hist_exog_size`` / ``stat_exog_size`` for those.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import List, Optional, Union

import jax
import jax.numpy as jnp

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.tsmixerx.data import create_windows, create_windows_exog, make_batch_exog
from chronax.models.tsmixerx.loss import LossFn
from chronax.models.tsmixerx.loss import resolve as _resolve_loss
from chronax.models.tsmixerx.model import TSMixerx, TSMixerxConfig
from chronax.models.tsmixerx.train import TrainState, _get_model, create_train_state, eval_step, train_step


SeriesLike = Union[jnp.ndarray, List[jnp.ndarray]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_multivariate(y: SeriesLike) -> jnp.ndarray:
    """Normalise ``y`` to ``[T, N]`` float32.

    Anything that isn't a ``list``/``tuple`` of series (e.g. a plain
    ``jax.numpy`` array, as passed by the benchmark harness) is treated as a
    single series / already-multivariate array.
    """
    if not isinstance(y, (list, tuple)):
        arr = jnp.asarray(y, dtype=jnp.float32)
        return arr[:, None] if arr.ndim == 1 else arr
    arrays = [jnp.asarray(s, dtype=jnp.float32).ravel() for s in y]
    T = arrays[0].shape[0]
    if not all(a.shape[0] == T for a in arrays):
        raise ValueError(
            "All series passed to TSMixerxForecaster must have the same length "
            "(TSMixerx is a multivariate model that jointly models all channels)."
        )
    return jnp.stack(arrays, axis=1)  # [T, N]


def _normalise_windows(
    ins: jnp.ndarray,
    out: jnp.ndarray,
) -> tuple:
    """Per-window normalisation of (insample, outsample) pairs."""
    win_mean = ins.mean(axis=1, keepdims=True)                        # [W, 1, N]
    win_std  = jnp.maximum(ins.std(axis=1, keepdims=True), 1e-5)      # [W, 1, N]
    return (ins - win_mean) / win_std, (out - win_mean) / win_std


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class TSMixerxForecaster(BaseForecaster):
    """High-level fit/forecast wrapper around :class:`chronax.models.tsmixerx.TSMixerx`.

    Args:
        h: forecast horizon.
        input_size: history window length; -1 (default) uses ``3 * h``.
        n_series: number of time series (channels). Must match the number of
            channels in the data passed to :meth:`fit`.
        n_block, ff_dim, dropout, revin, revin_affine, use_batchnorm,
        futr_exog_size, hist_exog_size, stat_exog_size: forwarded to
        :class:`TSMixerxConfig`. The window-based :meth:`fit` / :meth:`forecast`
        path does not pass exogenous covariates through — see module
        docstring for the lower-level batch API when covariates are needed.
        max_steps: number of optimiser steps performed by :meth:`fit`.
        learning_rate: Adam / AdamW peak learning rate.
        batch_size: number of windows per training step.
        random_seed: PRNG seed for parameter init and window sampling.
        alias: display name for external reporting.
        loss: registered name (``"mae"``, ``"mse"``) from
            :mod:`chronax.models.tsmixerx.loss` or a callable ``(y, y_hat) -> scalar``.
        scale: if True, each training window is normalised by its own mean/std
            (per-window, not per-series global); inference context is normalised
            with the same convention.
        grad_clip: global gradient-norm clipping threshold (0 = disabled).
        use_lr_schedule: if True, wrap Adam with warmup + cosine decay
            decaying to 1% of ``learning_rate`` over ``max_steps``.
        weight_decay: if > 0, use AdamW instead of Adam.
    """

    uses_exog = True

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        n_series: int = 1,
        n_block: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.0,
        revin: bool = True,
        revin_affine: bool = True,
        use_batchnorm: bool = False,
        futr_exog_size: int = 0,
        hist_exog_size: int = 0,
        stat_exog_size: int = 0,
        *,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        random_seed: int = 0,
        alias: str = "TSMixerx",
        loss: Union[str, LossFn] = "mae",
        scale: bool = True,
        grad_clip: float = 0.0,
        use_lr_schedule: bool = True,
        weight_decay: float = 0.0,
        val_fraction: float = 0.0,
        val_check_steps: int = 100,
    ):
        if input_size < 1:
            input_size = 3 * h
        self._futr_size = futr_exog_size
        self._futr_ctx: Optional[jnp.ndarray] = None
        self.config = TSMixerxConfig(
            h=h,
            input_size=input_size,
            n_series=n_series,
            n_block=n_block,
            ff_dim=ff_dim,
            dropout=dropout,
            revin=revin,
            revin_affine=revin_affine,
            use_batchnorm=use_batchnorm,
            futr_exog_size=futr_exog_size,
            hist_exog_size=hist_exog_size,
            stat_exog_size=stat_exog_size,
        )
        self.h                = h
        self.input_size        = input_size
        self.max_steps       = max_steps
        self.learning_rate   = learning_rate
        self.batch_size      = batch_size
        self.random_seed     = random_seed
        self.seed            = random_seed
        self.alias           = alias
        self.loss            = loss
        self.scale           = scale
        self.grad_clip       = grad_clip
        self.use_lr_schedule = use_lr_schedule
        self.weight_decay    = weight_decay
        self.val_fraction    = val_fraction
        self.val_check_steps = val_check_steps

        self._state: Optional[TrainState] = None
        self._fit_y: Optional[SeriesLike] = None

    @property
    def _loss_fn(self) -> LossFn:
        """Resolve ``self.loss`` to a callable. Lazy so string forms pickle."""
        return _resolve_loss(self.loss)

    # ---------------------------------------------------------------- props

    @property
    def fitted(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> TrainState:
        if self._state is None:
            raise RuntimeError("TSMixerxForecaster has not been fit yet.")
        return self._state

    # --------------------------------------------------------- fit / forecast

    def fit(
        self,
        y: SeriesLike,
        X: Optional[jnp.ndarray] = None,
        *,
        futr_exog: Optional[jnp.ndarray] = None,
        verbose: bool = False,
    ) -> "TSMixerxForecaster":
        """Train the model on ``y``, optionally with future-known exog.

        Args:
            y: ``[T, N]`` multivariate array or list of N equal-length 1-D
                arrays. All channels are modelled jointly.
            X: alias for ``futr_exog`` (``BaseForecaster`` convention).
            futr_exog: ``[T, F]`` future-known exogenous features aligned with
                ``y`` (``n_series=1`` only). The exogenous width ``F`` is not
                known at construction time, so ``self.config`` is rebuilt here
                with the observed ``futr_exog_size`` — mirrors the lazy-config
                pattern used by :class:`~chronax.models.mlp.mlp_model.MLP`.
            verbose: print loss every ``max(1, max_steps // 10)`` steps.
        """
        if futr_exog is None:
            futr_exog = X

        y_mv = _as_multivariate(y)  # [T, N]
        self._fit_y = y
        T, N = y_mv.shape

        if N != self.config.n_series:
            raise ValueError(
                f"config.n_series={self.config.n_series} but data has {N} channels. "
                "Reconstruct the forecaster with the correct n_series."
            )

        if futr_exog is not None:
            return self._fit_with_exog(y_mv, futr_exog, verbose=verbose)

        self._futr_size = 0
        self._futr_ctx = None
        if self.config.futr_exog_size != 0:
            self.config = replace(self.config, futr_exog_size=0)

        ins_all, out_all = create_windows(y_mv, self.config.input_size, self.config.h)

        if self.scale:
            ins_all, out_all = _normalise_windows(ins_all, out_all)

        n_windows = ins_all.shape[0]
        B         = self.batch_size

        n_val = (
            max(1, int(n_windows * self.val_fraction))
            if self.val_fraction > 0 and n_windows > 1
            else 0
        )
        n_train  = n_windows - n_val
        has_val  = n_val > 0 and n_train > 0

        if has_val:
            train_ins = ins_all[:n_train]
            train_out = out_all[:n_train]
            val_batch = {
                "insample_y":  ins_all[n_train:],
                "outsample_y": out_all[n_train:],
                "sample_mask": jnp.ones(
                    (n_val, self.config.h, self.config.n_series), dtype=jnp.float32
                ),
            }
        else:
            train_ins = ins_all
            train_out = out_all

        rng = jax.random.PRNGKey(self.seed)
        init_rng, perm_rng, rng = jax.random.split(rng, 3)
        self._state = create_train_state(
            init_rng,
            self.config,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            cosine_decay_steps=self.max_steps if self.use_lr_schedule else 0,
            warmup_steps=max(1, self.max_steps // 10) if self.use_lr_schedule else 0,
        )

        log_every = max(1, self.max_steps // 10)

        total_needed = self.max_steps * B
        if n_train <= total_needed:
            n_reps = (total_needed + n_train - 1) // n_train
            perm_keys = jax.random.split(perm_rng, n_reps)
            flat = jnp.concatenate(
                [jax.random.permutation(k, n_train) for k in perm_keys]
            )[:total_needed]
        else:
            flat = jax.random.choice(perm_rng, n_train, shape=(total_needed,), replace=False)

        ins_seq   = jax.device_put(train_ins[flat])   # [S*B, L, N]
        out_seq   = jax.device_put(train_out[flat])   # [S*B, h, N]
        ones_mask = jnp.ones((B, self.config.h, self.config.n_series), dtype=jnp.float32)

        best_params      = jax.tree.map(jnp.asarray, self._state.params)      if has_val else None
        best_batch_stats = jax.tree.map(jnp.asarray, self._state.batch_stats) if has_val else None
        best_val_loss    = float("inf")

        for step in range(self.max_steps):
            s = step * B
            batch = {
                "insample_y":  ins_seq[s : s + B],
                "outsample_y": out_seq[s : s + B],
                "sample_mask": ones_mask,
            }
            rng, step_rng = jax.random.split(rng)
            self._state, loss, _ = train_step(self._state, batch, step_rng, loss_fn=self._loss_fn)

            if has_val and (step + 1) % self.val_check_steps == 0:
                val_loss, _ = eval_step(self._state, val_batch, loss_fn=self._loss_fn)
                if float(val_loss) < best_val_loss:
                    best_val_loss    = float(val_loss)
                    best_params      = jax.tree.map(jnp.asarray, self._state.params)
                    best_batch_stats = jax.tree.map(jnp.asarray, self._state.batch_stats)

            if verbose and (step % log_every == 0 or step == self.max_steps - 1):
                print(f"  step {step:>5}: train_loss={float(loss):.5f}")

        if has_val and best_params is not None:
            self._state = self._state.replace(
                params=best_params, batch_stats=best_batch_stats
            )

        return self

    def _fit_with_exog(
        self,
        y_mv: jnp.ndarray,
        futr_exog: jnp.ndarray,
        *,
        verbose: bool = False,
    ) -> "TSMixerxForecaster":
        """Batch-based training path used when :meth:`fit` receives ``futr_exog``.

        Rebuilds ``self.config`` with the observed exogenous width, windows
        ``y`` and ``futr_exog`` together via :func:`create_windows_exog`, and
        trains with the already exog-capable :func:`train_step` — no new
        training-step logic is needed, only exog-aware batch construction.
        """
        cfg = self.config
        N = y_mv.shape[1]
        if N != 1:
            raise NotImplementedError(
                "TSMixerxForecaster future-exog support currently requires "
                f"n_series=1 (got n_series={N}); use the lower-level "
                "create_windows_exog / make_batch_exog / train_step API for "
                "multivariate exog."
            )

        futr_exog = jnp.asarray(futr_exog, dtype=jnp.float32)
        if futr_exog.ndim == 1:
            futr_exog = futr_exog[:, None]
        if futr_exog.shape[0] != y_mv.shape[0]:
            raise ValueError(
                f"futr_exog must align with y (len {y_mv.shape[0]}); "
                f"got {futr_exog.shape[0]}."
            )
        F = futr_exog.shape[1]
        self._futr_size = F
        cfg = replace(cfg, futr_exog_size=F)
        self.config = cfg

        futr_exog_3d = futr_exog[:, :, None]  # [T, F, N=1]
        windows = create_windows_exog(
            y_mv, cfg.input_size, cfg.h, futr_exog=futr_exog_3d
        )
        if self.scale:
            ins_n, out_n = _normalise_windows(
                windows["insample_y"], windows["outsample_y"]
            )
            windows = {**windows, "insample_y": ins_n, "outsample_y": out_n}

        self._futr_ctx = futr_exog[-cfg.input_size:]  # [L, F], cached for forecast()

        n_windows = windows["insample_y"].shape[0]
        B = self.batch_size

        rng = jax.random.PRNGKey(self.seed)
        init_rng, rng = jax.random.split(rng)
        self._state = create_train_state(
            init_rng,
            cfg,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            cosine_decay_steps=self.max_steps if self.use_lr_schedule else 0,
            warmup_steps=max(1, self.max_steps // 10) if self.use_lr_schedule else 0,
        )

        log_every = max(1, self.max_steps // 10)

        for step in range(self.max_steps):
            rng, idx_rng, step_rng = jax.random.split(rng, 3)
            idx = jax.random.randint(idx_rng, (B,), 0, n_windows)
            batch = make_batch_exog(windows, idx)
            self._state, loss, _ = train_step(
                self._state, batch, step_rng, loss_fn=self._loss_fn
            )

            if verbose and (step % log_every == 0 or step == self.max_steps - 1):
                print(f"  step {step:>5}: train_loss={float(loss):.5f}")

        return self

    def forecast(
        self,
        y: SeriesLike,
        h: Optional[int] = None,
        *,
        futr_exog: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Produce ``h``-step ahead forecasts.

        Args:
            y: ``[T, N]`` series. The last ``config.input_size`` timesteps are
                used as the conditioning window.
            h: accepted for API symmetry; must equal ``config.h`` if provided.
            futr_exog: ``[h, F]`` future-known exogenous values for the
                forecast horizon — required iff this forecaster was
                :meth:`fit` with ``futr_exog``. The preceding ``input_size``
                exog values come from the context cached at fit time.

        Returns:
            ``[h, N]`` forecast array, or ``[h]`` if the input was 1-D.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .forecast().")
        if h is not None and h != self.config.h:
            raise ValueError(
                f"This forecaster was configured with h={self.config.h}; "
                f"got h={h}. Reconstruct with a matching config."
            )

        y_mv    = _as_multivariate(y)          # [T, N]
        squeeze = not isinstance(y, (list, tuple)) and jnp.ndim(y) == 1

        L = self.config.input_size
        T = y_mv.shape[0]
        if T >= L:
            context = y_mv[-L:].astype(jnp.float32)                   # [L, N]
        else:
            pad     = jnp.zeros((L - T, y_mv.shape[1]), dtype=jnp.float32)
            context = jnp.concatenate([pad, y_mv.astype(jnp.float32)], axis=0)  # [L, N]

        futr_exog_3d = None
        if self._futr_size > 0:
            if y_mv.shape[1] != 1:
                raise NotImplementedError(
                    "TSMixerxForecaster future-exog support currently requires n_series=1."
                )
            if futr_exog is None:
                raise ValueError(
                    "This TSMixerxForecaster was fit with futr_exog; forecast requires "
                    f"futr_exog of shape (h={self.config.h}, F={self._futr_size})."
                )
            futr_exog = jnp.asarray(futr_exog, dtype=jnp.float32)
            if futr_exog.ndim == 1:
                futr_exog = futr_exog[:, None]
            if futr_exog.shape != (self.config.h, self._futr_size):
                raise ValueError(
                    f"futr_exog must be (h={self.config.h}, F={self._futr_size}); "
                    f"got {futr_exog.shape}."
                )
            futr_full = jnp.concatenate([self._futr_ctx, futr_exog], axis=0)  # [L+h, F]
            futr_exog_3d = jnp.transpose(futr_full, (1, 0))[None, :, :, None]  # [1, F, L+h, 1]

        if self.scale:
            ctx_mean = context.mean(axis=0)                           # [N]
            ctx_std  = jnp.maximum(context.std(axis=0), 1e-5)        # [N]
            context_in = (context - ctx_mean[None]) / ctx_std[None]  # [L, N]
        else:
            context_in = context
            ctx_mean   = jnp.zeros(context.shape[1], dtype=jnp.float32)
            ctx_std    = jnp.ones(context.shape[1],  dtype=jnp.float32)

        x     = context_in[None]   # [1, L, N]
        model = _get_model(self.config)
        variables = {"params": self._state.params, "batch_stats": self._state.batch_stats}
        preds = model.apply(
            variables, x, futr_exog=futr_exog_3d, deterministic=True
        )   # [1, h, N]
        preds = preds[0]        # [h, N]

        if self.scale:
            preds = preds * ctx_std[None] + ctx_mean[None]

        return preds[:, 0] if squeeze else preds

    def predict(
        self,
        h: Optional[int] = None,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[Union[int, float]]] = None,
        *,
        futr_exog: Optional[jnp.ndarray] = None,
    ) -> dict:
        """Forecast from the series passed to :meth:`fit`.

        Satisfies the ``BaseForecaster`` contract: unlike :meth:`forecast`,
        this does not take ``y`` explicitly — it reuses the series cached at
        fit time.

        Args:
            h: Forecast horizon. Defaults to ``config.h`` when None.
            X: alias for ``futr_exog`` (``BaseForecaster`` convention).
            level: Not yet supported for this model.
            futr_exog: ``[h, F]`` future-known exogenous values — required
                iff this forecaster was :meth:`fit` with ``futr_exog``.

        Returns:
            dict: ``{"mean": jnp.ndarray}``.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .predict().")
        if level is not None:
            raise NotImplementedError(
                "TSMixerxForecaster.predict(level=...) is not yet supported."
            )
        if futr_exog is None:
            futr_exog = X
        preds = self.forecast(y=self._fit_y, h=h, futr_exog=futr_exog)
        return {"mean": jnp.asarray(preds)}

    def fit_predict(
        self,
        y: SeriesLike,
        *,
        verbose: bool = False,
    ) -> jnp.ndarray:
        """Convenience: fit on ``y`` and immediately return forecasts."""
        self.fit(y, verbose=verbose)
        return self.forecast(y)

    # ---------------------------------------------------------------- utils

    def with_config(self, **overrides) -> "TSMixerxForecaster":
        """Return a fresh (unfitted) forecaster with overridden config fields."""
        new_cfg = replace(self.config, **overrides)
        return TSMixerxForecaster(
            **asdict(new_cfg),
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            alias=self.alias,
            loss=self.loss,
            scale=self.scale,
            grad_clip=self.grad_clip,
            use_lr_schedule=self.use_lr_schedule,
            weight_decay=self.weight_decay,
            val_fraction=self.val_fraction,
            val_check_steps=self.val_check_steps,
        )

    def __repr__(self) -> str:
        return (
            f"TSMixerxForecaster(config={asdict(self.config)}, "
            f"max_steps={self.max_steps}, learning_rate={self.learning_rate}, "
            f"batch_size={self.batch_size}, fitted={self.fitted})"
        )

"""High-level fit/forecast adapter around the Chronax TSMixer.

TSMixer is a multivariate model — all N series must be provided together as a
single ``[T, N]`` array (or a list of N equal-length 1-D arrays). A single
shared model processes all channels simultaneously.

Normalisation strategy — per-window:
    For each extracted window the insample mean/std are computed and both the
    insample and outsample portions of that window are normalised with those
    statistics.  This matches NeuralForecast's ``scaler_type="standard"``
    pipeline and ensures the model always sees consistently scaled input
    regardless of where in the series the window falls.  During inference the
    context window is normalised with its own mean/std; predictions are
    denormalised with the same statistics before returning.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from chronax.models.tsmixer.data import create_windows
from chronax.models.tsmixer.loss import masked_mae
from chronax.models.tsmixer.model import TSMixer, TSMixerConfig
from chronax.models.tsmixer.train import TrainState, create_train_state, train_step


SeriesLike = Union[np.ndarray, List[np.ndarray]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_multivariate(y: SeriesLike) -> np.ndarray:
    """Normalise ``y`` to ``[T, N]`` float32."""
    if isinstance(y, np.ndarray):
        arr = y.astype(np.float32)
        return arr[:, None] if arr.ndim == 1 else arr
    arrays = [np.asarray(s, dtype=np.float32).ravel() for s in y]
    T = len(arrays[0])
    if not all(len(a) == T for a in arrays):
        raise ValueError(
            "All series passed to TSMixerForecaster must have the same length "
            "(TSMixer is a multivariate model that jointly models all channels)."
        )
    return np.stack(arrays, axis=1)  # [T, N]


def _normalise_windows(
    ins: np.ndarray,
    out: np.ndarray,
) -> tuple:
    """Per-window normalisation of (insample, outsample) pairs.

    Computes per-channel mean and std from the insample portion of each window
    and applies the same statistics to both insample and outsample.  Matches
    NeuralForecast's ``scaler_type="standard"`` behaviour.
    """
    win_mean = ins.mean(axis=1, keepdims=True)                        # [W, 1, N]
    win_std  = np.maximum(ins.std(axis=1, keepdims=True), 1e-5)       # [W, 1, N]
    return (ins - win_mean) / win_std, (out - win_mean) / win_std


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


class TSMixerForecaster:
    """High-level fit/forecast wrapper around :class:`chronax.models.tsmixer.TSMixer`.

    Args:
        config: model architecture config. ``config.n_series`` must match the
            number of channels in the data passed to :meth:`fit`.
        max_steps: number of optimiser steps performed by :meth:`fit`.
        learning_rate: Adam / AdamW peak learning rate.
        batch_size: number of windows per training step.
        seed: PRNG seed for parameter init and window sampling.
        scale: if True, each training window is normalised by its own mean/std
            (per-window, not per-series global); inference context is normalised
            with the same convention.
        grad_clip: global gradient-norm clipping threshold (0 = disabled).
        use_lr_schedule: if True, wrap Adam with warmup + cosine decay
            decaying to 1 % of ``learning_rate`` over ``max_steps``.
        weight_decay: if > 0, use AdamW instead of Adam.
        loss_fn: loss function passed to :func:`train_step`.  Defaults to
            :func:`masked_mae`.  Use :func:`masked_mse` for datasets where
            minimising squared error (RMSE) is the priority.
    """

    def __init__(
        self,
        config: TSMixerConfig,
        *,
        max_steps: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        seed: int = 0,
        scale: bool = True,
        grad_clip: float = 1.0,
        use_lr_schedule: bool = True,
        weight_decay: float = 0.0,
        loss_fn=masked_mae,
    ):
        self.config          = config
        self.max_steps       = max_steps
        self.learning_rate   = learning_rate
        self.batch_size      = batch_size
        self.seed            = seed
        self.scale           = scale
        self.grad_clip       = grad_clip
        self.use_lr_schedule = use_lr_schedule
        self.weight_decay    = weight_decay
        self.loss_fn         = loss_fn

        self._state: Optional[TrainState] = None

    # ---------------------------------------------------------------- props

    @property
    def fitted(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> TrainState:
        if self._state is None:
            raise RuntimeError("TSMixerForecaster has not been fit yet.")
        return self._state

    # --------------------------------------------------------- fit / forecast

    def fit(
        self,
        y: SeriesLike,
        *,
        verbose: bool = False,
    ) -> "TSMixerForecaster":
        """Train the model on ``y``.

        Args:
            y: ``[T, N]`` multivariate array or list of N equal-length 1-D
                arrays. All channels are modelled jointly.
            verbose: print loss every ``max(1, max_steps // 10)`` steps.
        """
        y_mv = _as_multivariate(y)  # [T, N]
        T, N = y_mv.shape

        if N != self.config.n_series:
            raise ValueError(
                f"config.n_series={self.config.n_series} but data has {N} channels. "
                "Reconstruct the forecaster with the correct n_series."
            )

        ins_np, out_np = create_windows(y_mv, self.config.input_size, self.config.h)

        if self.scale:
            ins_np, out_np = _normalise_windows(ins_np, out_np)

        n_windows = ins_np.shape[0]
        B         = self.batch_size

        rng = jax.random.PRNGKey(self.seed)
        init_rng, rng = jax.random.split(rng)
        warmup = max(1, self.max_steps // 10)
        self._state = create_train_state(
            init_rng,
            self.config,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            cosine_decay_steps=self.max_steps if self.use_lr_schedule else 0,
            warmup_steps=warmup if self.use_lr_schedule else 0,
        )

        np_rng    = np.random.RandomState(self.seed)
        log_every = max(1, self.max_steps // 10)

        # Build the full epoch-cycle permutation, then pre-arrange the window
        # arrays in that order so every training step reads a *contiguous* slice
        # rather than scatter-gathering random indices.  On CPU, contiguous reads
        # are dramatically faster (cache-friendly, prefetcher works) compared to
        # GatherNd over 6k+ non-contiguous rows.
        total_needed = self.max_steps * B
        if n_windows <= total_needed:
            n_reps = (total_needed + n_windows - 1) // n_windows
            flat = np.concatenate(
                [np_rng.permutation(n_windows) for _ in range(n_reps)]
            )[:total_needed]
        else:
            flat = np_rng.choice(n_windows, size=total_needed, replace=False)

        # Pre-arrange: index once in numpy (CPU), then ship to device once.
        ins_seq = jax.device_put(jnp.array(ins_np[flat]))   # [S*B, L, N]
        out_seq = jax.device_put(jnp.array(out_np[flat]))   # [S*B, h, N]
        ones_mask = jnp.ones((B, self.config.h, self.config.n_series), dtype=jnp.float32)

        for step in range(self.max_steps):
            s = step * B
            batch = {
                "insample_y":  ins_seq[s : s + B],   # contiguous slice [B, L, N]
                "outsample_y": out_seq[s : s + B],   # contiguous slice [B, h, N]
                "sample_mask": ones_mask,
            }
            rng, step_rng = jax.random.split(rng)
            self._state, loss, _ = train_step(self._state, batch, step_rng, loss_fn=self.loss_fn)

            if verbose and (step % log_every == 0 or step == self.max_steps - 1):
                print(f"  step {step:>5}: train_loss={float(loss):.5f}")

        return self

    def forecast(
        self,
        y: SeriesLike,
        h: Optional[int] = None,
    ) -> np.ndarray:
        """Produce ``h``-step ahead forecasts.

        Args:
            y: ``[T, N]`` series. The last ``config.input_size`` timesteps are
                used as the conditioning window.
            h: accepted for API symmetry; must equal ``config.h`` if provided.

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
        squeeze = isinstance(y, np.ndarray) and y.ndim == 1

        L = self.config.input_size
        T = y_mv.shape[0]
        if T >= L:
            context = y_mv[-L:].astype(np.float32)                   # [L, N]
        else:
            pad     = np.zeros((L - T, y_mv.shape[1]), dtype=np.float32)
            context = np.vstack([pad, y_mv.astype(np.float32)])       # [L, N]

        if self.scale:
            ctx_mean = context.mean(axis=0)                           # [N]
            ctx_std  = np.maximum(context.std(axis=0), 1e-5)         # [N]
            context_in = (context - ctx_mean[None]) / ctx_std[None]  # [L, N]
        else:
            context_in = context
            ctx_mean   = np.zeros(context.shape[1], dtype=np.float32)
            ctx_std    = np.ones(context.shape[1],  dtype=np.float32)

        x     = jnp.array(context_in[None])   # [1, L, N]
        model = TSMixer(self.config)
        variables = {"params": self._state.params, "batch_stats": self._state.batch_stats}
        preds = model.apply(variables, x, deterministic=True)   # [1, h, N]
        preds_np = np.asarray(preds)[0]        # [h, N]

        if self.scale:
            preds_np = preds_np * ctx_std[None] + ctx_mean[None]

        return preds_np[:, 0] if squeeze else preds_np

    def fit_predict(
        self,
        y: SeriesLike,
        *,
        verbose: bool = False,
    ) -> np.ndarray:
        """Convenience: fit on ``y`` and immediately return forecasts."""
        self.fit(y, verbose=verbose)
        return self.forecast(y)

    # ---------------------------------------------------------------- utils

    def with_config(self, **overrides) -> "TSMixerForecaster":
        """Return a fresh (unfitted) forecaster with overridden config fields."""
        new_cfg = replace(self.config, **overrides)
        return TSMixerForecaster(
            new_cfg,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            seed=self.seed,
            scale=self.scale,
            grad_clip=self.grad_clip,
            use_lr_schedule=self.use_lr_schedule,
            weight_decay=self.weight_decay,
        )

    def __repr__(self) -> str:
        return (
            f"TSMixerForecaster(config={asdict(self.config)}, "
            f"max_steps={self.max_steps}, learning_rate={self.learning_rate}, "
            f"batch_size={self.batch_size}, fitted={self.fitted})"
        )

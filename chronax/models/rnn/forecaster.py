"""High-level fit/forecast adapter around the Chronax RNN.

The low-level building blocks in :mod:`chronax.models.rnn.{model,train,data}`
expose JAX-native primitives (``RNN.apply``, ``train_step``,
``autoregressive_predict``). For benchmarking and ad-hoc usage it is more
convenient to mirror the
``neuralforecast.NeuralForecast(models=[RNN(...)]).fit(df).predict()`` flow
with a familiar ``fit(y) -> self`` / ``forecast(h) -> ndarray`` interface.

This module provides exactly that. It is intentionally thin: all training
happens through :func:`chronax.models.rnn.train.train_step`, all inference
through :func:`chronax.models.rnn.model.autoregressive_predict` (recurrent)
or a single ``model.apply`` (direct), and all batching through
:mod:`chronax.models.rnn.data`.

Both univariate (a single 1D ``np.ndarray``) and panel (a list of 1D arrays)
modes are supported. Panel mode trains a *single shared model* over the
union of all series, exactly like the upstream NeuralForecast RNN.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from chronax.models.rnn.data import create_batch
from chronax.models.rnn.model import (
    RNN,
    RNNConfig,
    autoregressive_predict,
)
from chronax.models.rnn.train import (
    TrainState,
    create_train_state,
    train_step,
)


SeriesLike = Union[np.ndarray, Sequence[np.ndarray]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_series_list(y: SeriesLike) -> List[np.ndarray]:
    """Normalise ``y`` to a ``list[np.ndarray]`` of 1D float32 arrays."""
    if isinstance(y, np.ndarray):
        return [y.astype(np.float32, copy=False).ravel()]
    out: List[np.ndarray] = []
    for s in y:
        arr = np.asarray(s, dtype=np.float32).ravel()
        out.append(arr)
    return out


def _build_window_cache(
    y_list: List[np.ndarray],
    input_size: int,
    h: int,
) -> tuple:
    """Pre-extract every valid sliding window from all series.

    Avoids 128 Python ``_sample_window`` calls + ``create_batch`` overhead
    per training step.  After calling once at the start of ``fit``, each
    step is a cheap numpy fancy-index + single ``jnp.array`` transfer.

    Returns:
        insample   [N, input_size] float32  — history windows
        outsample  [N, h]          float32  — horizon targets
        avail_mask [N, input_size] float32  — 1 = real data, 0 = padded
    """
    total = input_size + h
    all_ins: List[np.ndarray] = []
    all_out: List[np.ndarray] = []
    all_msk: List[np.ndarray] = []
    for y in y_list:
        n = len(y)
        if n < total:
            ins = np.zeros(input_size, dtype=np.float32)
            ins[input_size - min(n, input_size):] = y[:input_size]
            out = np.zeros(h, dtype=np.float32)
            avail = min(n - input_size, h)
            if avail > 0:
                out[h - avail:] = y[input_size : input_size + avail]
            msk = np.zeros(input_size, dtype=np.float32)
            msk[input_size - min(n, input_size):] = 1.0
            all_ins.append(ins)
            all_out.append(out)
            all_msk.append(msk)
        else:
            for start in range(n - total + 1):
                all_ins.append(y[start : start + input_size].astype(np.float32))
                all_out.append(y[start + input_size : start + total].astype(np.float32))
                all_msk.append(np.ones(input_size, dtype=np.float32))
    return (
        np.stack(all_ins),
        np.stack(all_out),
        np.stack(all_msk),
    )


def _scaler_fit(y_list: List[np.ndarray]) -> List[tuple]:
    """Per-series ``(mean, std)`` standardisation parameters.

    Standardisation is done outside the model so the loss is well-conditioned
    regardless of the dataset scale. Mirrors NeuralForecast's default
    ``scaler_type='robust'`` behaviour at a basic level (mean/std rather than
    median/MAD) which is plenty for the comparisons we run here.
    """
    stats = []
    for s in y_list:
        mu = float(np.mean(s)) if s.size > 0 else 0.0
        sd = float(np.std(s)) if s.size > 0 else 1.0
        sd = sd if sd > 1e-8 else 1.0
        stats.append((mu, sd))
    return stats


def _apply_scale(y_list: List[np.ndarray], stats: List[tuple]) -> List[np.ndarray]:
    return [(s - mu) / sd for s, (mu, sd) in zip(y_list, stats)]


def _invert_scale(yhat: np.ndarray, stats: List[tuple]) -> np.ndarray:
    """Invert per-series scaling. ``yhat`` shape: ``[N, h]``."""
    out = np.empty_like(yhat)
    for i, (mu, sd) in enumerate(stats):
        out[i] = yhat[i] * sd + mu
    return out


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


def _sample_window(
    y: np.ndarray,
    input_size: int,
    h: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return a random window of length ``input_size + h`` from ``y``.

    If the series is shorter than ``input_size + h``, the full series is
    returned and ``create_batch`` handles left-padding.
    """
    total = input_size + h
    if len(y) <= total:
        return y
    start = rng.randint(0, len(y) - total + 1)
    return y[start : start + total]


class RNNForecaster:
    """High-level fit/forecast wrapper around :class:`chronax.models.rnn.RNN`.

    Args:
        config: model architecture config. Must include ``h`` and
            ``input_size``.
        max_steps: number of optimiser steps performed by :meth:`fit`.
        learning_rate: Adam learning rate.
        batch_size: number of windows per training step.
        seed: PRNG seed for parameter init and training shuffling.
        scale: if True, standardise each series before training and undo the
            scaling on the forecast. Strongly recommended for raw real-world
            data (e.g. airline passengers, hourly temperature).
        grad_clip: global gradient-norm clip threshold (0 = disabled).
        window_sampling: if True, each training step samples a random
            ``input_size + h`` window. When no exogenous variables are
            present, all valid windows are pre-extracted into a numpy cache
            at the start of ``fit``; each step then becomes a cheap fancy-
            index instead of 128 Python ``_sample_window`` calls.
        use_lr_schedule: if True, wrap Adam with a warmup + cosine-decay
            schedule decaying to 1 % of ``learning_rate``. Set False to use
            a constant learning rate (matches NeuralForecast's default
            optimizer behaviour).

    Notes:
        ``forecast(h)`` ignores the ``h`` argument when set; the model's
        configured ``config.h`` is always used. We accept ``h=None`` for
        symmetry with NeuralForecast's ``predict(h=...)`` API.
    """

    def __init__(
        self,
        config: RNNConfig,
        *,
        max_steps: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        seed: int = 0,
        scale: bool = True,
        grad_clip: float = 1.0,
        window_sampling: bool = True,
        use_lr_schedule: bool = True,
        weight_decay: float = 0.0,
    ):
        self.config = config
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.scale = scale
        self.grad_clip = grad_clip
        self.window_sampling = window_sampling
        self.use_lr_schedule = use_lr_schedule
        self.weight_decay = weight_decay

        self._state: Optional[TrainState] = None
        self._scaler: Optional[List[tuple]] = None
        self._n_series: Optional[int] = None

    # ----------------------------------------------------------- properties

    @property
    def state(self) -> TrainState:
        if self._state is None:
            raise RuntimeError("RNNForecaster has not been fit yet.")
        return self._state

    @property
    def fitted(self) -> bool:
        return self._state is not None

    # ----------------------------------------------------------- fit / forecast

    def fit(
        self,
        y: SeriesLike,
        *,
        hist_exog: Optional[Sequence[np.ndarray]] = None,
        futr_exog: Optional[Sequence[np.ndarray]] = None,
        stat_exog: Optional[Sequence[np.ndarray]] = None,
        verbose: bool = False,
    ) -> "RNNForecaster":
        """Train the model in-place on ``y``.

        Args:
            y: a single 1D series or a list of 1D series (panel).
            hist_exog: per-series ``[T_i, X]`` historic covariates, or None.
            futr_exog: per-series ``[T_i + h, F]`` future covariates that
                cover both history and the forecast horizon, or None.
            stat_exog: per-series ``[S]`` static covariates, or None.
            verbose: print loss every ``max(1, max_steps // 10)`` steps.
        """
        y_list = _as_series_list(y)
        self._n_series = len(y_list)

        if self.scale:
            self._scaler = _scaler_fit(y_list)
            y_scaled = _apply_scale(y_list, self._scaler)
        else:
            self._scaler = [(0.0, 1.0)] * len(y_list)
            y_scaled = y_list

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

        np_rng = np.random.RandomState(self.seed)
        n = len(y_scaled)
        log_every = max(1, self.max_steps // 10)

        # ---- window cache (fast path) ----------------------------------------
        # Pre-extract all valid windows once and move the whole cache to the
        # JAX device.  Each training step then does only a numpy random-index
        # selection followed by a device-side JAX gather — eliminating the
        # per-step host-to-device transfer that dominated step time when using
        # jnp.array(...) inside the loop.
        _has_exog = hist_exog is not None or futr_exog is not None or stat_exog is not None
        if self.window_sampling and not _has_exog:
            _ins_np, _out_np, _msk_np = _build_window_cache(
                y_scaled, self.config.input_size, self.config.h
            )
            # One-time host→device transfer for the entire cache.
            _ins_dev = jax.device_put(jnp.array(_ins_np))   # [N, L]
            _out_dev = jax.device_put(jnp.array(_out_np))   # [N, h]
            _n_wins = _ins_np.shape[0]
            _use_cache = True
        else:
            _use_cache = False

        _ones_mask = jnp.ones((self.batch_size, self.config.h, 1), dtype=jnp.float32)

        for step in range(self.max_steps):
            if _use_cache:
                # numpy RNG only; index into device arrays (no host→device per step)
                win_idx = np_rng.choice(_n_wins, size=self.batch_size, replace=True)
                batch = {
                    "insample_y": _ins_dev[win_idx, :, None],
                    "outsample_y": _out_dev[win_idx, :, None],
                    "sample_mask": _ones_mask,
                }
            else:
                idx = np_rng.choice(n, size=self.batch_size, replace=True)
                sub = lambda lst: ([lst[i] for i in idx] if lst is not None else None)
                if self.window_sampling:
                    y_batch = [
                        _sample_window(
                            y_scaled[i], self.config.input_size, self.config.h, np_rng
                        )
                        for i in idx
                    ]
                else:
                    y_batch = [y_scaled[i] for i in idx]
                batch = create_batch(
                    y_series=y_batch,
                    input_size=self.config.input_size,
                    h=self.config.h,
                    hist_exog_list=sub(hist_exog),
                    futr_exog_list=sub(futr_exog),
                    stat_exog_list=sub(stat_exog),
                )

            rng, step_rng = jax.random.split(rng)
            self._state, loss, _ = train_step(self._state, batch, step_rng)

            if verbose and (step % log_every == 0 or step == self.max_steps - 1):
                print(f"  step {step:>5}: train_loss={float(loss):.5f}")

        return self

    def forecast(
        self,
        y: Optional[SeriesLike] = None,
        h: Optional[int] = None,
        *,
        hist_exog: Optional[Sequence[np.ndarray]] = None,
        futr_exog: Optional[Sequence[np.ndarray]] = None,
        stat_exog: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        """Produce horizon predictions.

        Args:
            y: optional series to forecast for. If None, uses the same series
                fit() was last called with — currently unsupported (the
                forecaster does not retain training data); pass it explicitly.
            h: optional explicit horizon. If None, uses ``self.config.h``.
                (We do not yet support changing the horizon at predict time;
                this argument is accepted for API symmetry only.)

        Returns:
            ``np.ndarray`` of shape ``[h]`` for univariate input or
            ``[n_series, h]`` for panel input.
        """
        if not self.fitted:
            raise RuntimeError("Call .fit() before .forecast().")
        if y is None:
            raise ValueError("y must be provided to forecast (panel state is not retained).")
        if h is not None and h != self.config.h:
            raise ValueError(
                f"This forecaster was configured with h={self.config.h}; "
                f"got h={h}. Construct a new forecaster for a different horizon."
            )

        y_list = _as_series_list(y)
        squeeze = isinstance(y, np.ndarray)

        # Scale using *the same* per-series stats if shapes match; otherwise
        # fit fresh stats on the provided series.
        if self.scale and self._scaler is not None and len(self._scaler) == len(y_list):
            stats = self._scaler
        elif self.scale:
            stats = _scaler_fit(y_list)
        else:
            stats = [(0.0, 1.0)] * len(y_list)
        y_scaled = _apply_scale(y_list, stats) if self.scale else y_list

        # ``create_batch`` reserves the last ``h`` points of each input series
        # as the outsample target, so to forecast *into the future* we append
        # ``h`` zero placeholders. They are discarded — only the model output
        # is used.
        y_padded = [
            np.concatenate([s, np.zeros(self.config.h, dtype=np.float32)]) for s in y_scaled
        ]

        batch = create_batch(
            y_series=y_padded,
            input_size=self.config.input_size,
            h=self.config.h,
            hist_exog_list=list(hist_exog) if hist_exog is not None else None,
            futr_exog_list=list(futr_exog) if futr_exog is not None else None,
            stat_exog_list=list(stat_exog) if stat_exog is not None else None,
        )

        model = RNN(self.config)
        if self.config.recurrent:
            preds = autoregressive_predict(
                model,
                self._state.params,
                batch["insample_y"],
                h=self.config.h,
                hist_exog=batch.get("hist_exog"),
                futr_exog=batch.get("futr_exog"),
                stat_exog=batch.get("stat_exog"),
            )
        else:
            preds, _ = model.apply(
                self._state.params,
                insample_y=batch["insample_y"],
                hist_exog=batch.get("hist_exog"),
                futr_exog=batch.get("futr_exog"),
                stat_exog=batch.get("stat_exog"),
                deterministic=True,
            )

        # preds: [N, h, output_size]
        preds_np = np.asarray(preds)[..., 0]  # [N, h]
        preds_np = _invert_scale(preds_np, stats) if self.scale else preds_np
        return preds_np[0] if squeeze else preds_np

    def fit_predict(
        self,
        y: SeriesLike,
        **kwargs,
    ) -> np.ndarray:
        """Fit on ``y`` and immediately forecast for the same series."""
        # Split fit-only kwargs (verbose) from forecast-only kwargs.
        fit_kwargs = {}
        fcst_kwargs = {}
        for k, v in kwargs.items():
            if k == "verbose":
                fit_kwargs[k] = v
            elif k in ("hist_exog", "futr_exog", "stat_exog"):
                fit_kwargs[k] = v
                fcst_kwargs[k] = v
            else:
                raise TypeError(f"Unknown kwarg for fit_predict: {k!r}")
        self.fit(y, **fit_kwargs)
        return self.forecast(y=y, **fcst_kwargs)

    # ----------------------------------------------------------- utility

    def with_config(self, **overrides) -> "RNNForecaster":
        """Return a fresh (unfitted) forecaster with overridden config fields."""
        new_cfg = replace(self.config, **overrides)
        return RNNForecaster(
            new_cfg,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            seed=self.seed,
            scale=self.scale,
            grad_clip=self.grad_clip,
            window_sampling=self.window_sampling,
            use_lr_schedule=self.use_lr_schedule,
            weight_decay=self.weight_decay,
        )

    def __repr__(self) -> str:
        return (
            f"RNNForecaster(config={asdict(self.config)}, max_steps={self.max_steps}, "
            f"learning_rate={self.learning_rate}, batch_size={self.batch_size}, "
            f"grad_clip={self.grad_clip}, window_sampling={self.window_sampling}, "
            f"fitted={self.fitted})"
        )

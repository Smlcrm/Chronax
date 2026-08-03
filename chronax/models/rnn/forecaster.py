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
:mod:`chronax.models.rnn.data`. No NumPy — every array is a ``jnp.ndarray``.

Both univariate (a single 1D ``jnp.ndarray``) and panel (a list of 1D arrays)
modes are supported. Panel mode trains a *single shared model* over the
union of all series, exactly like the upstream NeuralForecast RNN.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax import random

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.rnn.data import create_batch
from chronax.models.rnn.loss import LossFn
from chronax.models.rnn.loss import resolve as _resolve_loss
from chronax.models.rnn.model import (
    RNN,
    RNNConfig,
    autoregressive_predict,
)
from chronax.models.rnn.train import (
    TrainState,
    create_train_state,
    eval_step,
    eval_step_recurrent,
    train_step,
    train_step_recurrent,
)


SeriesLike = Union[jnp.ndarray, Sequence[jnp.ndarray]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_series_list(y: SeriesLike) -> List[jnp.ndarray]:
    """Normalise ``y`` to a ``list[jnp.ndarray]`` of 1D float32 arrays.

    Anything that isn't a ``list``/``tuple`` of series (e.g. a plain
    ``jax.numpy`` array, as passed by the benchmark harness) is treated as a
    single univariate series.
    """
    if not isinstance(y, (list, tuple)):
        return [jnp.asarray(y, dtype=jnp.float32).ravel()]
    return [jnp.asarray(s, dtype=jnp.float32).ravel() for s in y]


def _build_window_cache(
    y_list: List[jnp.ndarray],
    input_size: int,
    h: int,
) -> tuple:
    """Pre-extract every valid sliding window from all series.

    Avoids 128 Python ``_sample_window`` calls + ``create_batch`` overhead
    per training step. After calling once at the start of ``fit``, each step
    is a cheap fancy-index into an already-stacked device array.

    Returns:
        insample   [N, input_size] float32  — history windows
        outsample  [N, h]          float32  — horizon targets
        avail_mask [N, input_size] float32  — 1 = real data, 0 = padded
    """
    total = input_size + h
    all_ins: List[jnp.ndarray] = []
    all_out: List[jnp.ndarray] = []
    all_msk: List[jnp.ndarray] = []
    for y in y_list:
        n = y.shape[0]
        if n < total:
            avail_hist = min(n, input_size)
            pad_hist = input_size - avail_hist
            hist_part = y[:avail_hist].astype(jnp.float32)
            ins = jnp.concatenate([jnp.zeros(pad_hist, dtype=jnp.float32), hist_part])
            msk = jnp.concatenate(
                [jnp.zeros(pad_hist, dtype=jnp.float32), jnp.ones(avail_hist, dtype=jnp.float32)]
            )
            avail_out = max(min(n - input_size, h), 0)
            pad_out = h - avail_out
            out_part = (
                y[input_size: input_size + avail_out].astype(jnp.float32)
                if avail_out > 0 else jnp.zeros(0, dtype=jnp.float32)
            )
            out = jnp.concatenate([jnp.zeros(pad_out, dtype=jnp.float32), out_part])
            all_ins.append(ins)
            all_out.append(out)
            all_msk.append(msk)
        else:
            for start in range(n - total + 1):
                all_ins.append(y[start: start + input_size].astype(jnp.float32))
                all_out.append(y[start + input_size: start + total].astype(jnp.float32))
                all_msk.append(jnp.ones(input_size, dtype=jnp.float32))
    return (
        jnp.stack(all_ins),
        jnp.stack(all_out),
        jnp.stack(all_msk),
    )


def _scaler_fit(y_list: List[jnp.ndarray]) -> List[tuple]:
    """Per-series robust (median/IQR) scaling matching NeuralForecast's scaler_type='robust'."""
    stats = []
    for s in y_list:
        if s.size == 0:
            stats.append((0.0, 1.0))
            continue
        q25 = float(jnp.percentile(s, 25.0))
        q50 = float(jnp.percentile(s, 50.0))
        q75 = float(jnp.percentile(s, 75.0))
        iqr = q75 - q25
        iqr = iqr if iqr > 1e-8 else 1.0
        stats.append((q50, iqr))
    return stats


def _apply_scale(y_list: List[jnp.ndarray], stats: List[tuple]) -> List[jnp.ndarray]:
    return [(s - mu) / sd for s, (mu, sd) in zip(y_list, stats)]


def _invert_scale(yhat: jnp.ndarray, stats: List[tuple]) -> jnp.ndarray:
    """Invert per-series scaling. ``yhat`` shape: ``[N, h]``."""
    return jnp.stack([yhat[i] * sd + mu for i, (mu, sd) in enumerate(stats)])


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


def _sample_window(
    y: jnp.ndarray,
    input_size: int,
    h: int,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Return a random window of length ``input_size + h`` from ``y``.

    If the series is shorter than ``input_size + h``, the full series is
    returned and ``create_batch`` handles left-padding.
    """
    total = input_size + h
    n = y.shape[0]
    if n <= total:
        return y
    start = int(random.randint(key, (), 0, n - total + 1))
    return y[start: start + total]


class RNNForecaster(BaseForecaster):
    """High-level fit/forecast wrapper around :class:`chronax.models.rnn.RNN`.

    Args:
        h: forecast horizon.
        input_size: history window length; -1 (default) uses ``3 * h``.
        encoder_hidden_size, encoder_n_layers, encoder_activation,
        encoder_bias, encoder_dropout, decoder_hidden_size, decoder_layers,
        futr_exog_size, hist_exog_size, stat_exog_size, output_size,
        recurrent, cell_type, layer_norm: forwarded to :class:`RNNConfig`.
        max_steps: number of optimiser steps performed by :meth:`fit`.
        learning_rate: Adam learning rate.
        batch_size: number of windows per training step.
        random_seed: PRNG seed for parameter init and training shuffling.
        alias: display name for external reporting.
        loss: registered name (``"mae"``, ``"mse"``) from
            :mod:`chronax.models.rnn.loss` or a callable ``(y, y_hat) -> scalar``.
        scale: if True, standardise each series before training and undo the
            scaling on the forecast. Strongly recommended for raw real-world
            data (e.g. airline passengers, hourly temperature).
        grad_clip: global gradient-norm clip threshold (0 = disabled).
        window_sampling: if True, each training step samples a random
            ``input_size + h`` window. When no exogenous variables are
            present, all valid windows are pre-extracted into a device cache
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
        h: int,
        input_size: int = -1,
        encoder_hidden_size: int = 128,
        encoder_n_layers: int = 2,
        encoder_activation: str = "tanh",
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        futr_exog_size: int = 0,
        hist_exog_size: int = 0,
        stat_exog_size: int = 0,
        output_size: int = 1,
        recurrent: bool = False,
        cell_type: str = "elman",
        layer_norm: bool = False,
        *,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        random_seed: int = 0,
        alias: str = "RNN",
        loss: Union[str, LossFn] = "mae",
        scale: bool = True,
        grad_clip: float = 1.0,
        window_sampling: bool = True,
        use_lr_schedule: bool = True,
        weight_decay: float = 0.0,
        val_fraction: float = 0.1,
        val_check_steps: int = 100,
    ):
        if input_size < 1:
            input_size = 3 * h
        self.config = RNNConfig(
            h=h,
            input_size=input_size,
            encoder_hidden_size=encoder_hidden_size,
            encoder_n_layers=encoder_n_layers,
            encoder_activation=encoder_activation,
            encoder_bias=encoder_bias,
            encoder_dropout=encoder_dropout,
            decoder_hidden_size=decoder_hidden_size,
            decoder_layers=decoder_layers,
            futr_exog_size=futr_exog_size,
            hist_exog_size=hist_exog_size,
            stat_exog_size=stat_exog_size,
            output_size=output_size,
            recurrent=recurrent,
            cell_type=cell_type,
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
        self.window_sampling = window_sampling
        self.use_lr_schedule = use_lr_schedule
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.val_check_steps = val_check_steps

        self._state: Optional[TrainState] = None
        self._scaler: Optional[List[tuple]] = None
        self._n_series: Optional[int] = None
        self._fit_y: Optional[SeriesLike] = None
        self._use_window_scale: bool = False

    @property
    def _loss_fn(self):
        """Resolve ``self.loss`` to a callable. Lazy so string forms pickle."""
        return _resolve_loss(self.loss)

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
        hist_exog: Optional[Sequence[jnp.ndarray]] = None,
        futr_exog: Optional[Sequence[jnp.ndarray]] = None,
        stat_exog: Optional[Sequence[jnp.ndarray]] = None,
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
        self._fit_y = y

        if self.scale:
            self._scaler = _scaler_fit(y_list)
            y_scaled = _apply_scale(y_list, self._scaler)
        else:
            self._scaler = [(0.0, 1.0)] * len(y_list)
            y_scaled = y_list

        rng = random.PRNGKey(self.seed)
        init_rng, rng = random.split(rng)
        warmup = max(1, self.max_steps // 10)

        # Adaptive LR + warmup: estimate dataset size before the cache build so
        # we can configure the optimizer appropriately.
        # Small datasets (n_train < batch_size, e.g. AirlinePassengers with 23
        # windows): per-window normalized targets can be 1–7 MAD units above
        # zero. With standard LR=1e-3, Adam moves the output bias ~1 unit in
        # 1000 steps — not enough to converge. 3× LR reaches ~3 units (the
        # mean training target). Also shorten warmup so fewer steps are spent
        # at near-zero LR.
        _total = self.config.input_size + self.config.h
        _prelim_wins = sum(
            max(0, y.shape[0] - _total + 1) for y in y_list
        )
        _prelim_nval = (
            max(1, int(_prelim_wins * self.val_fraction))
            if self.val_fraction > 0 and _prelim_wins > 1 else 0
        )
        _prelim_ntrain = max(1, _prelim_wins - _prelim_nval)
        _is_small = _prelim_ntrain < self.batch_size
        effective_lr = self.learning_rate * (3.0 if _is_small else 1.0)
        # Shorter warmup for small datasets: 100→10 steps
        effective_warmup = (
            max(1, self.max_steps // 100) if _is_small
            else warmup
        )

        self._state = create_train_state(
            init_rng,
            self.config,
            learning_rate=effective_lr,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            cosine_decay_steps=self.max_steps if self.use_lr_schedule else 0,
            warmup_steps=effective_warmup if self.use_lr_schedule else 0,
        )

        n = len(y_scaled)
        log_every = max(1, self.max_steps // 10)

        # ---- window cache (fast path) ----------------------------------------
        # Pre-extract all valid windows once and move the whole cache to the
        # JAX device. Each training step then does only a random-index
        # selection followed by a device-side JAX gather — eliminating the
        # per-step host-to-device transfer that dominated step time when using
        # jnp.array(...) inside the loop.
        _has_exog = hist_exog is not None or futr_exog is not None or stat_exog is not None
        if self.window_sampling and not _has_exog:
            # Build cache from raw (unscaled) series, then apply per-window
            # local normalization. Each window is centered and scaled by its
            # own input-window mean/std — matching NF's window-level Scaler.
            # This removes local level differences (trend, regime shifts) so
            # the model learns shape, not absolute value.
            _ins_dev, _out_dev, _ = _build_window_cache(
                y_list, self.config.input_size, self.config.h
            )
            if self.scale:
                # Robust per-window normalization: median + MAD.
                # Matches NF's TemporalNorm(scaler_type='robust').
                _win_med = jnp.median(_ins_dev, axis=1)          # [N]
                _win_mad = jnp.median(
                    jnp.abs(_ins_dev - _win_med[:, None]), axis=1
                )                                                   # [N]
                _win_mad = jnp.where(_win_mad < 1e-8, jnp.ones_like(_win_mad), _win_mad)
                _ins_dev = (_ins_dev - _win_med[:, None]) / _win_mad[:, None]
                _out_dev = (_out_dev - _win_med[:, None]) / _win_mad[:, None]
                self._use_window_scale = True
            else:
                self._use_window_scale = False
            _n_wins = _ins_dev.shape[0]
            _use_cache = True

            # Hold out the last val_fraction of windows (time-ordered) for
            # validation + best-checkpoint tracking. This mirrors the implicit
            # regularisation NF gets from val_check_steps during Lightning training.
            _n_val = (
                max(1, int(_n_wins * self.val_fraction))
                if self.val_fraction > 0 and _n_wins > 1
                else 0
            )
            _n_train = _n_wins - _n_val
            _has_val = _n_val > 0 and _n_train > 0

            if _has_val:
                _train_ins = _ins_dev[:_n_train]
                _train_out = _out_dev[:_n_train]
                _val_batch = {
                    "insample_y":  _ins_dev[_n_train:, :, None],
                    "outsample_y": _out_dev[_n_train:, :, None],
                    "sample_mask": jnp.ones((_n_val, self.config.h, 1), dtype=jnp.float32),
                }
                if self.config.recurrent:
                    # Pre-build 1-step-ahead targets for validation.
                    # next_step_y[k] = y[L-h+k+1], the correct 1-step target
                    # for output position L-h+k (which sees y[0:L-h+k+1]).
                    _h = self.config.h
                    _val_ins_r = _ins_dev[_n_train:, :, None]           # [N_val, L, 1]
                    _val_next_step = jnp.concatenate([
                        _ins_dev[_n_train:, -(_h - 1):, None],          # [N_val, h-1, 1]
                        _out_dev[_n_train:, :1, None],                  # [N_val, 1, 1]
                    ], axis=1)                                            # [N_val, h, 1]
            else:
                _train_ins = _ins_dev
                _train_out = _out_dev
                _n_train = _n_wins
        else:
            _use_cache = False
            _has_val = False
            _n_train = n  # unused in non-cache path but satisfies type checker

        # Adaptive effective batch: for small datasets (n_train < batch_size),
        # use all training windows per step (full-batch gradient). This matches
        # NF's effective gradient quality — NF uses windows_batch_size=128 which
        # covers nearly all windows for a 23-window dataset — at lower compute cost.
        _effective_batch = min(self.batch_size, _n_train) if _use_cache else self.batch_size
        _ones_mask = jnp.ones((_effective_batch, self.config.h, 1), dtype=jnp.float32)

        # Best-checkpoint state for the validation path
        _best_params = jax.tree.map(jnp.asarray, self._state.params) if _has_val else None
        _best_val_loss = float("inf")

        # Epoch-based shuffled sampling: always active when cache is available.
        # Iterates through all training windows in random order before repeating,
        # guaranteeing every window receives equal gradient signal.
        _use_epoch_shuffle = _use_cache
        if _use_epoch_shuffle:
            rng, _perm_rng = random.split(rng)
            _epoch_perm = random.permutation(_perm_rng, _n_train)  # [N]
            _epoch_ptr = 0

        for step in range(self.max_steps):
            rng, idx_rng, window_rng, step_rng = random.split(rng, 4)
            if _use_cache:
                if _use_epoch_shuffle:
                    if _epoch_ptr + _effective_batch > _n_train:
                        rng, _perm_rng = random.split(rng)
                        _epoch_perm = random.permutation(_perm_rng, _n_train)
                        _epoch_ptr = 0
                    win_idx = _epoch_perm[_epoch_ptr: _epoch_ptr + _effective_batch]
                    _epoch_ptr += _effective_batch
                else:
                    win_idx = random.randint(idx_rng, (_effective_batch,), 0, _n_train)
                if self.config.recurrent:
                    _h = self.config.h
                    _ins_b = _train_ins[win_idx, :, None]   # [B, L, 1]
                    _out_b = _train_out[win_idx, :, None]   # [B, h, 1]
                    # next_step_y[k] = y[L-h+k+1]: last h-1 insample steps
                    # + first outsample step = [B, h, 1]
                    _next_step = jnp.concatenate(
                        [_ins_b[:, -(_h - 1):, :], _out_b[:, :1, :]], axis=1
                    )
                    self._state, loss, _ = train_step_recurrent(
                        self._state, _ins_b, _next_step, step_rng, loss_fn=self._loss_fn
                    )
                else:
                    batch = {
                        "insample_y": _train_ins[win_idx, :, None],
                        "outsample_y": _train_out[win_idx, :, None],
                        "sample_mask": _ones_mask,
                    }
                    self._state, loss, _ = train_step(
                        self._state, batch, step_rng, loss_fn=self._loss_fn
                    )
            else:
                idx = random.randint(idx_rng, (self.batch_size,), 0, n)
                sub = lambda lst: ([lst[int(i)] for i in idx] if lst is not None else None)
                if self.window_sampling:
                    window_keys = random.split(window_rng, len(idx))
                    y_batch = [
                        _sample_window(
                            y_scaled[int(i)], self.config.input_size, self.config.h, wk
                        )
                        for i, wk in zip(idx, window_keys)
                    ]
                else:
                    y_batch = [y_scaled[int(i)] for i in idx]
                batch = create_batch(
                    y_series=y_batch,
                    input_size=self.config.input_size,
                    h=self.config.h,
                    hist_exog_list=sub(hist_exog),
                    futr_exog_list=sub(futr_exog),
                    stat_exog_list=sub(stat_exog),
                )
                if self.config.recurrent:
                    _h = self.config.h
                    _next_step = jnp.concatenate(
                        [batch["insample_y"][:, -(_h - 1):, :], batch["outsample_y"][:, :1, :]], axis=1
                    )  # [B, h, 1]
                    self._state, loss, _ = train_step_recurrent(
                        self._state, batch["insample_y"], _next_step, step_rng, loss_fn=self._loss_fn
                    )
                else:
                    self._state, loss, _ = train_step(
                        self._state, batch, step_rng, loss_fn=self._loss_fn
                    )

            # Validation checkpoint: save best model seen so far
            if _has_val and (step + 1) % self.val_check_steps == 0:
                if self.config.recurrent:
                    val_loss, _ = eval_step_recurrent(
                        self._state, _val_ins_r, _val_next_step, loss_fn=self._loss_fn
                    )
                else:
                    val_loss, _ = eval_step(self._state, _val_batch, loss_fn=self._loss_fn)
                if float(val_loss) < _best_val_loss:
                    _best_val_loss = float(val_loss)
                    _best_params = jax.tree.map(jnp.asarray, self._state.params)

            if verbose and (step % log_every == 0 or step == self.max_steps - 1):
                print(f"  step {step:>5}: train_loss={float(loss):.5f}")

        # Restore the best-validation checkpoint to avoid returning an overfit model
        if _has_val and _best_params is not None:
            self._state = self._state.replace(params=_best_params)

        return self

    def forecast(
        self,
        y: Optional[SeriesLike] = None,
        h: Optional[int] = None,
        *,
        hist_exog: Optional[Sequence[jnp.ndarray]] = None,
        futr_exog: Optional[Sequence[jnp.ndarray]] = None,
        stat_exog: Optional[Sequence[jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Produce horizon predictions.

        Args:
            y: optional series to forecast for. If None, uses the same series
                fit() was last called with — currently unsupported (the
                forecaster does not retain training data); pass it explicitly.
            h: optional explicit horizon. If None, uses ``self.config.h``.
                (We do not yet support changing the horizon at predict time;
                this argument is accepted for API symmetry only.)

        Returns:
            ``jnp.ndarray`` of shape ``[h]`` for univariate input or
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
        squeeze = not isinstance(y, (list, tuple))

        if self._use_window_scale:
            # Per-window robust normalization: median + MAD of the context
            # window, matching what fit() did to each training window so the
            # model sees the same input distribution at inference time.
            win_means: List[float] = []
            win_stds:  List[float] = []
            y_scaled: List[jnp.ndarray] = []
            L = self.config.input_size
            for s in y_list:
                ctx = s[-L:] if s.shape[0] >= L else s
                wmed = float(jnp.median(ctx))
                wmad = float(jnp.median(jnp.abs(ctx - wmed)))
                if wmad < 1e-8:
                    wmad = 1.0
                win_means.append(wmed)
                win_stds.append(wmad)
                y_scaled.append((s - wmed) / wmad)
        else:
            # Fall back to series-level scaling (used when exogenous vars were
            # present during fit, so the cache path was not taken).
            if self.scale and self._scaler is not None and len(self._scaler) == len(y_list):
                stats = self._scaler
            elif self.scale:
                stats = _scaler_fit(y_list)
            else:
                stats = [(0.0, 1.0)] * len(y_list)
            y_scaled = _apply_scale(y_list, stats) if self.scale else list(y_list)

        # ``create_batch`` reserves the last ``h`` points of each input series
        # as the outsample target, so to forecast *into the future* we append
        # ``h`` zero placeholders. They are discarded — only the model output
        # is used.
        y_padded = [
            jnp.concatenate([s, jnp.zeros(self.config.h, dtype=jnp.float32)]) for s in y_scaled
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
        preds = preds[..., 0]  # [N, h]
        if self._use_window_scale:
            preds = jnp.stack([
                preds[i] * win_stds[i] + win_means[i] for i in range(len(y_list))
            ])
        else:
            preds = _invert_scale(preds, stats) if self.scale else preds
        return preds[0] if squeeze else preds

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
                "RNNForecaster.predict(level=...) is not yet supported."
            )
        preds = self.forecast(y=self._fit_y, h=h)
        return {"mean": jnp.asarray(preds)}

    def fit_predict(
        self,
        y: SeriesLike,
        **kwargs,
    ) -> jnp.ndarray:
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
            **asdict(new_cfg),
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            alias=self.alias,
            loss=self.loss,
            scale=self.scale,
            grad_clip=self.grad_clip,
            window_sampling=self.window_sampling,
            use_lr_schedule=self.use_lr_schedule,
            weight_decay=self.weight_decay,
            val_fraction=self.val_fraction,
            val_check_steps=self.val_check_steps,
        )

    def __repr__(self) -> str:
        return (
            f"RNNForecaster(config={asdict(self.config)}, max_steps={self.max_steps}, "
            f"learning_rate={self.learning_rate}, batch_size={self.batch_size}, "
            f"grad_clip={self.grad_clip}, window_sampling={self.window_sampling}, "
            f"fitted={self.fitted})"
        )

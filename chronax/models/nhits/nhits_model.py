"""NHITS forecaster: BaseForecaster wrapper around the flax.nnx backbone.

Univariate or N-series forecasting through hierarchically interpolated MLP
stacks (port of neuralforecast.NHITS). A 2-D ``fit(y (T, n_series))``
cross-learns ONE global network over all columns — every column's h-padded
windows are pooled into a single training set, per-window scaling keeps each
series' scale local, and predictions come from each column's own tail
context (channel-independent: the network never mixes series). Output rank
follows input rank (``(h,)`` vs ``(h, n_series)``).

Point, multi-quantile, and GMM distribution losses; intervals come from the
conformal path (point loss, 1-D fits, no temporal exog), natively from the
quantile heads, or natively from Monte-Carlo samples of the fitted mixture
(the loss type decides the interval mechanism, and a distribution head
ignores ``conformal_params``). The point forecast of a distribution head is
the analytic mixture mean. ``float32`` throughout.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.nhits.nhits_losses import (
    GMM, MultiQuantileLoss, outputsize_multiplier, resolve as _resolve_loss,
)
from chronax.models.nhits.nhits_module import NHITSNet
from chronax.models.nhits.nhits_scaler import resolve_scaler
from chronax.models.nhits.nhits_training import (
    build_exog_windows, build_windows, predict_params, predict_step, train_on_windows,
)
from chronax.utils import ConformalIntervals


def _nearest_q_index(quantiles, target: float) -> int:
    """Host-side index of the quantile closest to ``target``. Operates on the static
    quantile tuple (compile-time constants), so it is independent of any traced array."""
    return min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - target))


class NHITS(BaseForecaster):
    """NHITS: Neural Hierarchical Interpolation for Time Series (flax.nnx port
    of neuralforecast.NHITS; Challu et al. 2023, arXiv:2201.12886).

    Fully connected blocks arranged in doubly residual stacks: each block
    max/avg-pools the input window at its stack's rate, emits a backcast that
    is subtracted from the running residual, and forecasts a small set of
    knots that are interpolated up to the horizon — stacks with aggressive
    pooling and few knots capture low frequencies, the finest stack captures
    high frequencies. ``fit`` accepts a single series ``(T,)`` or an N-series
    panel ``(T, n_series)``; a 2-D fit cross-learns one global network over
    all columns (channel-independent — each column is forecast from its own
    tail context) and predictions follow the input rank. Future-known
    exogenous inputs are supported (``uses_exog = True``; shared across
    columns on a 2-D fit); historical and static exog are not modeled. Point
    (``"mae"``/``"mse"``/``"huber"``), multi-quantile (``MultiQuantileLoss``), or
    Gaussian-mixture (``GMM``) losses; with a GMM head the model is a
    probabilistic forecaster whose intervals come from seeded Monte-Carlo
    samples of the predictive mixture in original units. ``float32``
    throughout.
    """

    uses_exog = True

    def __init__(self, h, input_size=-1, n_blocks=(1, 1, 1),
                 mlp_units=((512, 512), (512, 512), (512, 512)),
                 n_pool_kernel_size=(2, 2, 1), n_freq_downsample=(4, 2, 1),
                 pooling_mode="MaxPool1d", interpolation_mode="linear",
                 dropout_prob_theta=0.0, activation="ReLU",
                 max_steps=1000, learning_rate=1e-3, num_lr_decays=3,
                 windows_batch_size=1024, scaler_type="identity", loss="mae",
                 quantile_sort=True, random_seed=1, alias="NHITS"):
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        # Tuples: immutable across new() clones and value-stable static config.
        self.n_blocks = tuple(int(b) for b in n_blocks)
        self.mlp_units = tuple((int(a), int(b)) for a, b in mlp_units)
        self.n_pool_kernel_size = tuple(int(k) for k in n_pool_kernel_size)
        self.n_freq_downsample = tuple(int(f) for f in n_freq_downsample)
        self.pooling_mode = pooling_mode
        self.interpolation_mode = interpolation_mode
        self.dropout_prob_theta = float(dropout_prob_theta)
        self.activation = activation
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.windows_batch_size = windows_batch_size
        self.scaler_type = scaler_type
        self.loss = loss
        self.quantile_sort = quantile_sort
        self.random_seed = random_seed
        self.alias = alias
        self.conformal_params: ConformalIntervals | None = None
        self.model_: NHITSNet | None = None
        self._futr_size = 0
        self._contexts = None
        self._train_y = None
        self._train_rank = None
        self._futr_ctx = None

    # ---- helpers -------------------------------------------------------------
    @property
    def _loss_fn(self):
        return _resolve_loss(self.loss)

    @property
    def _scaler(self):
        return resolve_scaler(self.scaler_type)

    @property
    def _is_distribution(self) -> bool:
        return bool(getattr(self._loss_fn, "is_distribution_output", False))

    @property
    def _has_temporal_exog(self) -> bool:
        return self._futr_size > 0

    def _build_net(self) -> NHITSNet:
        return NHITSNet(
            h=self.h, input_size=self.input_size, futr_exog_size=self._futr_size,
            n_blocks=self.n_blocks, mlp_units=self.mlp_units,
            n_pool_kernel_size=self.n_pool_kernel_size,
            n_freq_downsample=self.n_freq_downsample,
            pooling_mode=self.pooling_mode,
            interpolation_mode=self.interpolation_mode,
            dropout_prob_theta=self.dropout_prob_theta,
            activation=self.activation,
            outputsize_multiplier=outputsize_multiplier(self._loss_fn),
            rngs=nnx.Rngs(self.random_seed),
        )

    # ---- fit -----------------------------------------------------------------
    def fit(self, y, X=None, *, futr_exog=None) -> "NHITS":
        if X is not None:
            raise NotImplementedError(
                "NHITS supports future-known exog only; pass futr_exog="
            )
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim not in (1, 2):
            raise ValueError(f"y must be 1-D or 2-D (T, n_series); got shape {y.shape}.")
        y2 = y[:, None] if y.ndim == 1 else y               # [T, N]
        if y2.shape[0] <= self.input_size:
            # Need at least one window: T >= input_size + 1 (partial windows are h-padded).
            raise ValueError(
                f"Series length {y2.shape[0]} too short for input_size={self.input_size} "
                f"(need at least input_size+1)."
            )
        futr_exog = None if futr_exog is None else jnp.asarray(futr_exog, jnp.float32)
        if futr_exog is not None and futr_exog.shape[0] != y2.shape[0]:
            raise ValueError(f"futr_exog must align with y at fit (len {y2.shape[0]}); got {futr_exog.shape[0]}.")

        self._futr_size = 0 if futr_exog is None else int(futr_exog.shape[1])

        L, h = self.input_size, self.h
        n_series = y2.shape[1]
        # Pool every column's h-padded windows and cross-learn ONE network.
        # Exog windows (shared, calendar-style) repeat per column to stay
        # aligned with the pooled target windows. The column count is a static
        # config-derived shape, so the loop unrolls cleanly under jit/vmap.
        windows, masks = [], []
        for j in range(n_series):
            w, m = build_windows(y2[:, j], L, h)
            windows.append(w)
            masks.append(m)
        futr_pooled = None
        if futr_exog is not None:
            fw = build_exog_windows(futr_exog, L, h, windows[0].shape[0], "full")
            futr_pooled = jnp.concatenate([fw] * n_series)
        net = self._build_net()
        train_on_windows(
            net, jnp.concatenate(windows), jnp.concatenate(masks),
            h=h, input_size=L, max_steps=self.max_steps,
            windows_batch_size=self.windows_batch_size, lr=self.learning_rate,
            num_lr_decays=self.num_lr_decays, seed=self.random_seed,
            loss_fn=self._loss_fn, scaler=self._scaler,
            futr_windows=futr_pooled,
        )
        self.model_ = net
        self._contexts = y2[-L:, :].T                       # [N, L]
        self._train_y = y
        self._train_rank = y.ndim
        self._futr_ctx = None if futr_exog is None else futr_exog[-L:]
        return self

    # ---- predict -------------------------------------------------------------
    def _assemble_futr_full(self, futr_exog):
        if self._futr_size == 0:
            return None
        if futr_exog is None:
            raise ValueError(
                "This NHITS was fit with future-known exog; predict requires futr_exog of shape "
                f"(h={self.h}, F={self._futr_size})."
            )
        futr_exog = jnp.asarray(futr_exog, jnp.float32)
        if futr_exog.shape != (self.h, self._futr_size):
            raise ValueError(f"futr_exog must be (h={self.h}, F={self._futr_size}); got {futr_exog.shape}.")
        return jnp.concatenate([self._futr_ctx, futr_exog], axis=0)

    def _orient(self, arr: jnp.ndarray) -> jnp.ndarray:
        """[N, h] -> (h,) for a 1-D fit, (h, N) otherwise (rank follows input)."""
        return arr[0] if self._train_rank == 1 else arr.T

    def predict(self, h, X=None, *, futr_exog=None, level=None) -> dict:
        if self.model_ is None or self._contexts is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        if h < 1:
            raise ValueError(f"h must be a positive integer; got {h}.")
        if h > self.h:
            raise ValueError(
                f"NHITS was trained for h={self.h}; predict(h={h}) is unsupported. Pass h <= {self.h}."
            )
        if self._is_distribution:
            return self._format_distribution(h, level, futr_exog)
        futr_full = self._assemble_futr_full(futr_exog)
        full = predict_step(
            self.model_, self._contexts, h=self.h, input_size=self.input_size,
            scaler=self._scaler, futr_full=futr_full,
        )                                                     # [N, h_train, mult]
        if full.shape[-1] == 1:
            fcst = {"mean": self._orient(full[:, :h, 0])}
            if level is not None:
                if self._train_rank == 2:
                    raise ValueError(
                        "Conformal intervals are not supported on a 2-D (multi-series) fit; "
                        "use loss=MultiQuantileLoss([...]) or loss=GMM(...) for native intervals."
                    )
                fcst = self._add_conformal(fcst, level)
            return fcst
        return self._format_quantiles(full, h, level)

    def _format_distribution(self, h, level, futr_exog) -> dict:
        """Analytic mixture mean plus (optionally) native sample-quantile intervals."""
        loss = self._loss_fn
        distr_args = predict_params(
            self.model_, self._contexts, input_size=self.input_size,
            scaler=self._scaler, loss_fn=loss,
            futr_full=self._assemble_futr_full(futr_exog),
        )                                                     # arrays [N, h_train, K]
        fcst = {"mean": self._orient(loss.analytic_mean(distr_args)[:, :h])}
        if level is not None:
            samples = loss.sample(distr_args, key=jax.random.PRNGKey(self.random_seed))
            samples = samples[:, :h]                          # [N, h, S]
            for lv in sorted(level):
                lo_q = (100 - lv) / 200.0
                fcst[f"lo-{lv}"] = self._orient(jnp.quantile(samples, lo_q, axis=-1))
                fcst[f"hi-{lv}"] = self._orient(jnp.quantile(samples, 1.0 - lo_q, axis=-1))
        return fcst

    def _add_conformal(self, fcst, level):
        if self._has_temporal_exog:
            raise ValueError(
                "Conformal intervals are not supported with temporal (historical/future) exog. "
                "Use loss=MultiQuantileLoss([...]) or loss=GMM(...) for native intervals, or omit level."
            )
        if self.conformal_params is None:
            raise ValueError(
                "predict(level=...) requires `model.conformal_params` (a ConformalIntervals). "
                "conformity_scores re-fits per CV window under vmap -- expect minutes."
            )
        # Run the walk-forward on a clone: conformity_scores re-fits under vmap, and
        # those fit() writes would leave leaked tracers on this fitted estimator.
        cs = self.new().conformity_scores(self._train_y)
        return BaseForecaster.add_confidence_intervals(fcst, cs, level, self.conformal_params.method)

    def _format_quantiles(self, full, h, level) -> dict:
        qs = list(self._loss_fn.quantiles)
        full = full[:, :h]                                    # [N, h, Q]
        if self.quantile_sort:
            full = jnp.sort(full, axis=-1)                    # guarantee non-crossing
        median_idx = _nearest_q_index(qs, 0.5)
        fcst = {"mean": self._orient(full[..., median_idx])}
        if level is not None:
            for lv in sorted(level):
                lo_t, hi_t = (100 - lv) / 200.0, 1.0 - (100 - lv) / 200.0
                lo_i, hi_i = _nearest_q_index(qs, lo_t), _nearest_q_index(qs, hi_t)
                if abs(qs[lo_i] - lo_t) > 1e-6 or abs(qs[hi_i] - hi_t) > 1e-6:
                    raise ValueError(
                        f"level {lv} needs quantiles ({lo_t:.3f}, {hi_t:.3f}) which were not trained "
                        f"(have {qs}). Train MultiQuantileLoss including those quantiles."
                    )
                fcst[f"lo-{lv}"] = self._orient(full[..., lo_i])
                fcst[f"hi-{lv}"] = self._orient(full[..., hi_i])
        return fcst

    # ---- forecast ------------------------------------------------------------
    def forecast(self, y, h, X=None, X_future=None, *, futr_exog=None,
                 level=None, fitted=False) -> dict:
        """Stateless fit-then-predict. ``X`` is unsupported (NHITS models
        future-known exog only); ``X_future`` = future-known exog for the horizon
        ``(h, F)``; ``futr_exog`` = its history ``(T, F)``."""
        self.fit(y, X=X, futr_exog=futr_exog)
        result = self.predict(h=h, futr_exog=X_future, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def conformity_scores(self, y, X=None) -> jnp.ndarray:
        y = jnp.asarray(y)
        if y.ndim != 1:
            raise ValueError(
                "Conformal intervals are supported on 1-D fits only; a 2-D "
                "(multi-series) fit's intervals are the native quantile surfaces "
                "(loss=MultiQuantileLoss([...]) or loss=GMM(...) with predict(level=...))."
            )
        return super().conformity_scores(y, X)

    def _compute_fitted_values(self) -> jnp.ndarray:
        if self._train_rank == 2:
            raise NotImplementedError("fitted=True is not supported for 2-D (multi-series) fits.")
        if self._has_temporal_exog:
            raise NotImplementedError("fitted=True is not supported with exogenous inputs in v1.")
        y = self._train_y
        L = self.input_size
        n = y.shape[0] - L
        if n <= 0:
            return jnp.full((y.shape[0],), jnp.nan, dtype=jnp.float32)
        idx = jnp.arange(L)[None, :] + jnp.arange(n)[:, None]
        in_win = y[idx]                                       # [n, L]
        if self._is_distribution:
            distr_args = predict_params(
                self.model_, in_win, input_size=L, scaler=self._scaler,
                loss_fn=self._loss_fn,
            )
            one_step = self._loss_fn.analytic_mean(distr_args)[:, 0]
            return jnp.concatenate([jnp.full((L,), jnp.nan, dtype=jnp.float32), one_step])
        scaler = self._scaler
        shift, scale = scaler.stats(in_win, axis=1)
        z = scaler.transform(in_win, shift, scale)[..., None]
        pred = self.model_(z)                                 # [n, h, mult]
        col = 0 if pred.shape[-1] == 1 else _nearest_q_index(self._loss_fn.quantiles, 0.5)
        one_step = scaler.inverse(pred[:, 0, col], shift[:, 0], scale[:, 0])
        return jnp.concatenate([jnp.full((L,), jnp.nan, dtype=jnp.float32), one_step])

    # ---- pickle --------------------------------------------------------------
    def __getstate__(self) -> dict:
        """Serialize only nnx param state; the GraphDef is rebuilt on unpickle."""
        state = self.__dict__.copy()
        if state.get("model_") is not None:
            _, full_state = nnx.split(state["model_"])
            state["model_"] = ("__params_only__", full_state)
        return state

    def __setstate__(self, state: dict) -> None:
        m = state.get("model_")
        if isinstance(m, tuple) and m and m[0] == "__params_only__":
            saved = m[1]
            state["model_"] = None
            self.__dict__.update(state)
            net = self._build_net()           # uses restored _futr_size/loss
            nnx.update(net, saved)
            self.model_ = net
            return
        self.__dict__.update(state)

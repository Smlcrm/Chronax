"""TFT forecaster: BaseForecaster wrapper around the flax.nnx backbone.

Univariate + exogenous (static / historical / future-known) point and
multi-quantile forecasting. Exog sizes are inferred at fit; intervals come from
the conformal path (point loss, no temporal exog) or natively from the quantile
heads. ``float32`` throughout.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.tft.tft_losses import (
    MultiQuantileLoss, outputsize_multiplier, resolve as _resolve_loss,
)
from chronax.models.tft.tft_module import TFTNet
from chronax.models.tft.tft_scaler import resolve_scaler
from chronax.models.tft.tft_training import predict_step, train
from chronax.utils import ConformalIntervals


def _nearest_q_index(quantiles, target: float) -> int:
    """Host-side index of the quantile closest to ``target``. Operates on the static
    quantile tuple (compile-time constants), so it is independent of any traced array."""
    return min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - target))


class TFT(BaseForecaster):
    """Temporal Fusion Transformer (flax.nnx port of neuralforecast.TFT)."""

    uses_exog = True

    def __init__(self, h, input_size=-1, hidden_size=128, n_head=4, attn_dropout=0.0,
                 dropout=0.1, grn_activation="ELU", rnn_type="lstm", n_rnn_layers=1,
                 max_steps=1000, learning_rate=1e-3, windows_batch_size=1024,
                 scaler_type="robust", loss="mae", quantile_sort=True,
                 random_seed=1, alias="TFT"):
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.grn_activation = grn_activation
        self.rnn_type = rnn_type
        self.n_rnn_layers = n_rnn_layers
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.windows_batch_size = windows_batch_size
        self.scaler_type = scaler_type
        self.loss = loss
        self.quantile_sort = quantile_sort
        self.random_seed = random_seed
        self.alias = alias
        self.conformal_params: ConformalIntervals | None = None
        self.model_: TFTNet | None = None
        self._hist_size = 0
        self._futr_size = 0
        self._stat_size = 0
        self._context = None
        self._train_y = None
        self._hist_ctx = None
        self._futr_ctx = None
        self._stat = None

    # ---- helpers -------------------------------------------------------------
    @property
    def _loss_fn(self):
        return _resolve_loss(self.loss)

    @property
    def _scaler(self):
        return resolve_scaler(self.scaler_type)

    @property
    def _has_temporal_exog(self) -> bool:
        return self._hist_size > 0 or self._futr_size > 0

    def _build_net(self) -> TFTNet:
        return TFTNet(
            h=self.h, input_size=self.input_size, hidden_size=self.hidden_size,
            n_head=self.n_head, attn_dropout=self.attn_dropout, dropout=self.dropout,
            grn_activation=self.grn_activation, rnn_type=self.rnn_type,
            n_rnn_layers=self.n_rnn_layers, stat_exog_size=self._stat_size,
            hist_exog_size=self._hist_size, futr_exog_size=self._futr_size,
            outputsize_multiplier=outputsize_multiplier(self._loss_fn),
            rngs=nnx.Rngs(self.random_seed),
        )

    # ---- fit -----------------------------------------------------------------
    def fit(self, y, X=None, *, futr_exog=None, stat_exog=None) -> "TFT":
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] < self.input_size + self.h:
            raise ValueError(
                f"Series length {y.shape[0]} too short for input_size={self.input_size} + h={self.h}."
            )
        X = None if X is None else jnp.asarray(X, jnp.float32)
        futr_exog = None if futr_exog is None else jnp.asarray(futr_exog, jnp.float32)
        stat_exog = None if stat_exog is None else jnp.asarray(stat_exog, jnp.float32)
        if X is not None and X.shape[0] != y.shape[0]:
            raise ValueError(f"X (historical exog) must align with y (len {y.shape[0]}); got {X.shape[0]}.")
        if futr_exog is not None and futr_exog.shape[0] != y.shape[0]:
            raise ValueError(f"futr_exog must align with y at fit (len {y.shape[0]}); got {futr_exog.shape[0]}.")

        self._hist_size = 0 if X is None else int(X.shape[1])
        self._futr_size = 0 if futr_exog is None else int(futr_exog.shape[1])
        self._stat_size = 0 if stat_exog is None else int(stat_exog.shape[0])

        net = self._build_net()
        train(
            net, y, h=self.h, input_size=self.input_size, max_steps=self.max_steps,
            windows_batch_size=self.windows_batch_size, lr=self.learning_rate,
            seed=self.random_seed, loss_fn=self._loss_fn, scaler=self._scaler,
            hist_exog=X, futr_exog=futr_exog, stat_exog=stat_exog,
        )
        L = self.input_size
        self.model_ = net
        self._context = y[-L:]
        self._train_y = y
        self._hist_ctx = None if X is None else X[-L:]
        self._futr_ctx = None if futr_exog is None else futr_exog[-L:]
        self._stat = stat_exog
        return self

    # ---- predict -------------------------------------------------------------
    def _raw_predict(self, futr_exog=None) -> jnp.ndarray:
        futr_full = None
        if self._futr_size > 0:
            if futr_exog is None:
                raise ValueError(
                    "This TFT was fit with future-known exog; predict requires futr_exog of shape "
                    f"(h={self.h}, F={self._futr_size})."
                )
            futr_exog = jnp.asarray(futr_exog, jnp.float32)
            if futr_exog.shape != (self.h, self._futr_size):
                raise ValueError(f"futr_exog must be (h={self.h}, F={self._futr_size}); got {futr_exog.shape}.")
            futr_full = jnp.concatenate([self._futr_ctx, futr_exog], axis=0)
        return predict_step(
            self.model_, self._context, h=self.h, input_size=self.input_size,
            scaler=self._scaler, hist_context=self._hist_ctx, futr_full=futr_full, stat=self._stat,
        )

    def predict(self, h, X=None, *, futr_exog=None, level=None) -> dict:
        if self.model_ is None or self._context is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        if h < 1:
            raise ValueError(f"h must be a positive integer; got {h}.")
        if h > self.h:
            raise ValueError(
                f"TFT was trained for h={self.h}; predict(h={h}) is unsupported. Pass h <= {self.h}."
            )
        full = self._raw_predict(futr_exog=futr_exog)        # [h_train, mult]
        if full.shape[-1] == 1:
            fcst = {"mean": full[:h, 0]}
            if level is not None:
                fcst = self._add_conformal(fcst, level)
            return fcst
        return self._format_quantiles(full, h, level)

    def _add_conformal(self, fcst, level):
        if self._has_temporal_exog:
            raise ValueError(
                "Conformal intervals are not supported with temporal (historical/future) exog. "
                "Use loss=MultiQuantileLoss([...]) for native intervals, or omit level."
            )
        if self.conformal_params is None:
            raise ValueError(
                "predict(level=...) requires `model.conformal_params` (a ConformalIntervals). "
                "conformity_scores re-fits per CV window under vmap -- expect minutes."
            )
        cs = self.conformity_scores(self._train_y)
        return BaseForecaster.add_confidence_intervals(fcst, cs, level, self.conformal_params.method)

    def _format_quantiles(self, full, h, level) -> dict:
        qs = list(self._loss_fn.quantiles)
        full = full[:h]                                       # [h, Q]
        if self.quantile_sort:
            full = jnp.sort(full, axis=-1)                    # guarantee non-crossing
        median_idx = _nearest_q_index(qs, 0.5)
        fcst = {"mean": full[:, median_idx]}
        if level is not None:
            for lv in sorted(level):
                lo_t, hi_t = (100 - lv) / 200.0, 1.0 - (100 - lv) / 200.0
                lo_i, hi_i = _nearest_q_index(qs, lo_t), _nearest_q_index(qs, hi_t)
                if abs(qs[lo_i] - lo_t) > 1e-6 or abs(qs[hi_i] - hi_t) > 1e-6:
                    raise ValueError(
                        f"level {lv} needs quantiles ({lo_t:.3f}, {hi_t:.3f}) which were not trained "
                        f"(have {qs}). Train MultiQuantileLoss including those quantiles."
                    )
                fcst[f"lo-{lv}"] = full[:, lo_i]
                fcst[f"hi-{lv}"] = full[:, hi_i]
        return fcst

    # ---- forecast ------------------------------------------------------------
    def forecast(self, y, h, X=None, X_future=None, *, futr_exog=None, stat_exog=None,
                 level=None, fitted=False) -> dict:
        """Stateless fit-then-predict. ``X`` = historical exog over ``y``; ``X_future`` =
        future-known exog for the horizon ``(h, F)``; ``futr_exog`` = its history ``(T, F)``."""
        self.fit(y, X=X, futr_exog=futr_exog, stat_exog=stat_exog)
        result = self.predict(h=h, futr_exog=X_future, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
        if self._has_temporal_exog or self._stat_size:
            raise NotImplementedError("fitted=True is not supported with exogenous inputs in v1.")
        y = self._train_y
        L = self.input_size
        n = y.shape[0] - L
        if n <= 0:
            return jnp.full((y.shape[0],), jnp.nan, dtype=jnp.float32)
        idx = jnp.arange(L)[None, :] + jnp.arange(n)[:, None]
        in_win = y[idx]                                       # [n, L]
        scaler = self._scaler
        shift, scale = scaler.stats(in_win, axis=1)
        z = scaler.transform(in_win, shift, scale)[..., None]
        pred = self.model_(z, deterministic=True)            # [n, h, mult]
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
            net = self._build_net()           # uses restored _hist_size/_futr_size/_stat_size/loss
            nnx.update(net, saved)
            self.model_ = net
            return
        self.__dict__.update(state)

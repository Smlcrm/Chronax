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
    """Temporal Fusion Transformer (TFT; flax.nnx port of neuralforecast.TFT).

    Lim et al., 2021: https://arxiv.org/abs/1912.09363. Static, historical
    and future-known covariates are embedded per variable and passed through
    variable selection networks; a recurrent encoder (long short-term memory
    (LSTM) or gated recurrent unit (GRU)) over the ``input_size + h`` span
    is followed by static enrichment, interpretable multi-head attention and
    a position-wise gated residual network; only the last ``h`` positions
    reach the output head. All three exog kinds are supported
    (``uses_exog = True``). Point or multi-quantile losses; conformal or
    native quantile intervals. ``float32`` throughout.

    Args:
        h (int): Forecast horizon the network is trained for.
        input_size (int): Encoder window length ``L``. ``-1`` (default)
            resolves to ``3 * h``.
        hidden_size (int): Width of every embedding, gated residual network
            and recurrent state. Default ``128``.
        n_head (int): Attention heads; must divide ``hidden_size``.
            Default ``4``.
        attn_dropout (float): Dropout on attention weights. Default ``0.0``.
        dropout (float): Dropout in gated residual networks and after
            attention. Default ``0.1``.
        grn_activation (str): Activation of the variable-selection gated
            residual networks, matched case-insensitively against ``"ELU"``,
            ``"ReLU"``, ``"Softplus"``, ``"Tanh"``, ``"SELU"``,
            ``"LeakyReLU"``, ``"Sigmoid"``. The static-context networks always
            use ELU. Default ``"ELU"``. Unknown names raise ``ValueError`` at
            ``fit``.
        rnn_type (str): ``"lstm"`` selects LSTM cells; any other value
            selects GRU cells (no validation). Default ``"lstm"``.
        n_rnn_layers (int): Stacked recurrent layers. Default ``1``.
        max_steps (int): Optimizer steps. Default ``1000``.
        learning_rate (float): Adam learning rate (constant). Default ``1e-3``.
        windows_batch_size (int): Training windows per step. Default ``1024``.
        scaler_type (str | Scaler): Per-window scaler, ``"identity"`` or
            ``"robust"`` (median / mean absolute deviation), or a ``Scaler``
            instance. Default ``"robust"``. Resolved at ``fit``.
        loss (str | callable | MultiQuantileLoss): ``"mae"``, ``"mse"``,
            ``"huber"``, a callable ``(pred, target) -> scalar``, or a
            :class:`~chronax.models.tft.tft_losses.MultiQuantileLoss` for
            native quantile intervals. Default ``"mae"``. Unknown names raise
            ``ValueError`` at ``fit``.
        quantile_sort (bool): Sort quantile outputs so they do not cross.
            Default ``True``.
        random_seed (int): Seed for parameter init, batch sampling and
            dropout. Default ``1``.
        alias (str): Display name. Default ``"TFT"``.

    Note:
        Set ``conformal_params`` to a :class:`~chronax.utils.ConformalIntervals`
        to enable conformal intervals on a point-loss model. ``model_`` holds
        the trained network after ``fit``.
    """

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
        """Train a fresh network on ``y`` and store the tail contexts.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)`` with
                ``T >= input_size + 1``.
            X (jnp.ndarray | None): Historical exog aligned with ``y``,
                shape ``(T, F_hist)``; known over the input span only.
            futr_exog (jnp.ndarray | None): Future-known exog aligned with
                ``y``, shape ``(T, F_futr)``. Its horizon values must then be
                passed to ``predict``.
            stat_exog (jnp.ndarray | None): Static covariates of the series,
                shape ``(S,)``.

        Returns:
            TFT: ``self``, with ``model_`` set.

        Raises:
            ValueError: ``y`` is not 1-D, ``T < input_size + 1``, ``X`` /
                ``futr_exog`` has a different length than ``y``, or ``loss`` /
                ``scaler_type`` / ``grn_activation`` is an unknown name.
        """
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] < self.input_size + 1:
            # neuralforecast trains on h-padded partial windows, so one window (T >= L+1) suffices.
            raise ValueError(
                f"Series length {y.shape[0]} too short for input_size={self.input_size} "
                f"(need at least input_size+1)."
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
        """Forecast the first ``h`` steps of the trained horizon.

        Args:
            h (int): Steps to return, ``1 <= h <= self.h``.
            X (None): Ignored; historical exog is read from the ``fit`` call.
            futr_exog (jnp.ndarray | None): Horizon values of the
                future-known exog, shape ``(self.h, F_futr)``. Required when
                ``fit`` received ``futr_exog``; ignored otherwise.
            level (list[int] | None): Confidence levels in (0, 100), e.g.
                ``[80, 95]``. With a point loss the intervals are conformal
                (``conformal_params`` required); with a
                ``MultiQuantileLoss`` they are read from the trained
                quantiles.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``; the 0.5-quantile head for a
            quantile loss) plus ``"lo-{lv}"`` / ``"hi-{lv}"`` (shape ``(h,)``)
            per level when ``level`` is given.

        Raises:
            RuntimeError: Called before ``fit``.
            ValueError: ``h < 1`` or ``h > self.h``; ``futr_exog`` missing
                or not ``(self.h, F_futr)`` on a model fit with future exog;
                ``level`` with a point loss while the model was fit with
                historical or future exog, or while ``conformal_params`` is
                ``None``; ``level`` with a quantile loss whose quantiles do
                not contain ``(100 - lv) / 200`` and ``1 - (100 - lv) / 200``.
                The conformal path also raises the base-class ``ValueError``
                when the series is too short for two cross-validation windows.
        """
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
        # Run the walk-forward on a clone: conformity_scores re-fits under vmap, and
        # those fit() writes would leave leaked tracers on this fitted estimator.
        cs = self.new().conformity_scores(self._train_y)
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
        """Stateless fit-then-predict.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Steps to return, ``1 <= h <= self.h``.
            X (jnp.ndarray | None): Historical exog aligned with ``y``,
                shape ``(T, F_hist)``.
            X_future (jnp.ndarray | None): Future-known exog for the horizon,
                shape ``(self.h, F_futr)``; required when ``futr_exog`` is
                given.
            futr_exog (jnp.ndarray | None): Future-known exog history aligned
                with ``y``, shape ``(T, F_futr)``.
            stat_exog (jnp.ndarray | None): Static covariates, shape ``(S,)``.
            level (list[int] | None): Confidence levels in (0, 100).
            fitted (bool): Also return one-step-ahead in-sample predictions.

        Returns:
            dict: As :meth:`predict`, plus ``"fitted"`` (shape ``(T,)``, the
            first ``input_size`` entries ``NaN``) when ``fitted=True``.

        Raises:
            NotImplementedError: ``fitted=True`` on a model fit with any exog
                (historical, future or static).
            ValueError: Any ``fit`` or ``predict`` error.
        """
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

"""Temporal convolutional network (TCN) forecaster: BaseForecaster wrapper around the flax.nnx backbone.

Univariate point and multi-quantile forecasting with future-known and
historical exogenous inputs, via a dilated causal convolution encoder and a
multilayer perceptron (MLP) decoder. Exog size is inferred at fit; intervals come from the conformal
path (point loss, no temporal exog) or natively from the quantile heads.
``float32`` throughout.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.tcn.tcn_losses import (
    MultiQuantileLoss, outputsize_multiplier, resolve as _resolve_loss,
)
from chronax.models.tcn.tcn_module import ACTIVATIONS, TCNNet
from chronax.models.tcn.tcn_scaler import resolve_scaler
from chronax.models.tcn.tcn_training import predict_step, train
from chronax.utils import ConformalIntervals


def _nearest_q_index(quantiles, target: float) -> int:
    """Host-side index of the quantile closest to ``target``. Operates on the static
    quantile tuple (compile-time constants), so it is independent of any traced array."""
    return min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - target))


class TCN(BaseForecaster):
    """Temporal convolutional network (TCN) with a multilayer perceptron
    (MLP) decoder (flax.nnx port of neuralforecast.TCN).

    Lea et al., 2016: https://arxiv.org/abs/1608.08242. The historical
    encoder is a stack of dilated causal 1-D convolutions (exponentially
    increasing dilations give an exponentially large receptive field), a
    context adapter ``Linear(input_size -> h)`` maps the encoded history onto
    the forecasting window in a single pass (no autoregressive loop), and a
    per-timestep MLP decodes each horizon step. Future-known (``futr_exog=``)
    and historical (``X=``) exogenous inputs are supported
    (``uses_exog = True``): future-known exog joins the encoder channels over
    the input span and residual-concatenates its horizon slice before the
    decoder, while historical exog joins the encoder channels only (it has no
    horizon values). Static exog is not modeled. ``context_size`` is accepted
    for neuralforecast interface parity but is unused. Point or
    multi-quantile losses; conformal or native quantile intervals.
    ``float32`` throughout.

    Args:
        h (int): Forecast horizon the network is trained for.
        input_size (int): Encoder window length ``L``. ``-1`` (default)
            resolves to ``3 * h``.
        kernel_size (int): Convolution kernel width. Default ``2``.
        dilations (Sequence[int] | None): Dilation of each encoder layer, one
            layer per entry. ``None`` (default) resolves to
            ``(1, 2, 4, 8, 16)``.
        encoder_hidden_size (int): Channels of every encoder layer.
            Default ``128``.
        encoder_activation (str): Activation after each convolution,
            ``"ReLU"`` or ``"Tanh"``. Default ``"ReLU"``.
        context_size (int): Accepted for neuralforecast interface parity;
            unused. Default ``10``.
        decoder_hidden_size (int): Width of the decoder MLP's hidden layers.
            Default ``128``.
        decoder_layers (int): Linear layers in the decoder MLP (ReLU between
            them). Default ``2``.
        max_steps (int): Optimizer steps. Default ``1000``.
        learning_rate (float): Adam learning rate (constant). Default ``1e-3``.
        windows_batch_size (int): Training windows per step. Default ``128``.
        scaler_type (str | Scaler): Per-window scaler, ``"identity"`` or
            ``"robust"`` (median / mean absolute deviation), or a ``Scaler``
            instance. Default ``"robust"``. Resolved at ``fit``.
        loss (str | callable | MultiQuantileLoss): ``"mae"``, ``"mse"``,
            ``"huber"``, a callable ``(pred, target) -> scalar``, or a
            :class:`~chronax.models.tcn.tcn_losses.MultiQuantileLoss` for
            native quantile intervals. Default ``"mae"``. Unknown names raise
            ``ValueError`` at ``fit``.
        quantile_sort (bool): Sort quantile outputs so they do not cross.
            Default ``True``.
        random_seed (int): Seed for parameter init and batch sampling.
            Default ``1``.
        alias (str): Display name. Default ``"TCN"``.

    Note:
        Set ``conformal_params`` to a :class:`~chronax.utils.ConformalIntervals`
        to enable conformal intervals on a point-loss model. ``model_`` holds
        the trained network after ``fit``.

    Raises:
        ValueError: ``encoder_activation`` not ``"ReLU"``/``"Tanh"``.
    """

    uses_exog = True

    def __init__(self, h, input_size=-1, kernel_size=2, dilations=None,
                 encoder_hidden_size=128, encoder_activation="ReLU", context_size=10,
                 decoder_hidden_size=128, decoder_layers=2, max_steps=1000,
                 learning_rate=1e-3, windows_batch_size=128, scaler_type="robust",
                 loss="mae", quantile_sort=True, random_seed=1, alias="TCN"):
        if input_size < 1:
            input_size = 3 * h
        if encoder_activation not in ACTIVATIONS:
            raise ValueError(
                f"encoder_activation must be one of {sorted(ACTIVATIONS)}; got {encoder_activation!r}."
            )
        self.h = h
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.dilations = (1, 2, 4, 8, 16) if dilations is None else tuple(dilations)
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_activation = encoder_activation
        self.context_size = context_size          # accepted for interface parity; unused
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.windows_batch_size = windows_batch_size
        self.scaler_type = scaler_type
        self.loss = loss
        self.quantile_sort = quantile_sort
        self.random_seed = random_seed
        self.alias = alias
        self.conformal_params: ConformalIntervals | None = None
        self.model_: TCNNet | None = None
        self._hist_size = 0
        self._futr_size = 0
        self._context = None
        self._train_y = None
        self._hist_ctx = None
        self._futr_ctx = None

    # ---- helpers -------------------------------------------------------------
    @property
    def _loss_fn(self):
        return _resolve_loss(self.loss)

    @property
    def _scaler(self):
        return resolve_scaler(self.scaler_type)

    @property
    def _has_temporal_exog(self) -> bool:
        return self._futr_size > 0 or self._hist_size > 0

    def _build_net(self) -> TCNNet:
        return TCNNet(
            h=self.h, input_size=self.input_size, kernel_size=self.kernel_size,
            dilations=self.dilations, encoder_hidden_size=self.encoder_hidden_size,
            encoder_activation=self.encoder_activation,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_layers=self.decoder_layers, hist_exog_size=self._hist_size,
            futr_exog_size=self._futr_size,
            outputsize_multiplier=outputsize_multiplier(self._loss_fn),
            rngs=nnx.Rngs(self.random_seed),
        )

    # ---- fit -----------------------------------------------------------------
    def fit(self, y, X=None, *, futr_exog=None) -> "TCN":
        """Train a fresh network on ``y`` and store the tail contexts.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)`` with
                ``T >= input_size + 1``.
            X (jnp.ndarray | None): Historical exog aligned with ``y``,
                shape ``(T, F_hist)``; known over the input span only.
            futr_exog (jnp.ndarray | None): Future-known exog aligned with
                ``y``, shape ``(T, F_futr)``. Its horizon values must then be
                passed to ``predict``.

        Returns:
            TCN: ``self``, with ``model_`` set.

        Raises:
            ValueError: ``y`` is not 1-D, ``T < input_size + 1``, ``X`` /
                ``futr_exog`` has a different length than ``y``, or ``loss`` /
                ``scaler_type`` is an unknown name.
        """
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] <= self.input_size:
            # Need at least one window: T >= input_size + 1 (partial windows are h-padded).
            raise ValueError(
                f"Series length {y.shape[0]} too short for input_size={self.input_size} "
                f"(need at least input_size+1)."
            )
        # X = historical exog (T, F): known only over the input span. futr_exog =
        # future-known exog (T, F): also supplied for the horizon at predict.
        hist_exog = None if X is None else jnp.asarray(X, jnp.float32)
        if hist_exog is not None and hist_exog.shape[0] != y.shape[0]:
            raise ValueError(f"X (historical exog) must align with y at fit (len {y.shape[0]}); got {hist_exog.shape[0]}.")
        futr_exog = None if futr_exog is None else jnp.asarray(futr_exog, jnp.float32)
        if futr_exog is not None and futr_exog.shape[0] != y.shape[0]:
            raise ValueError(f"futr_exog must align with y at fit (len {y.shape[0]}); got {futr_exog.shape[0]}.")

        self._hist_size = 0 if hist_exog is None else int(hist_exog.shape[1])
        self._futr_size = 0 if futr_exog is None else int(futr_exog.shape[1])

        net = self._build_net()
        train(
            net, y, h=self.h, input_size=self.input_size, max_steps=self.max_steps,
            windows_batch_size=self.windows_batch_size, lr=self.learning_rate,
            seed=self.random_seed, loss_fn=self._loss_fn, scaler=self._scaler,
            hist_exog=hist_exog, futr_exog=futr_exog,
        )
        L = self.input_size
        self.model_ = net
        self._context = y[-L:]
        self._train_y = y
        self._hist_ctx = None if hist_exog is None else hist_exog[-L:]   # [L, F]
        self._futr_ctx = None if futr_exog is None else futr_exog[-L:]
        return self

    # ---- predict -------------------------------------------------------------
    def _raw_predict(self, futr_exog=None) -> jnp.ndarray:
        futr_full = None
        if self._futr_size > 0:
            if futr_exog is None:
                raise ValueError(
                    "This TCN was fit with future-known exog; predict requires futr_exog of shape "
                    f"(h={self.h}, F={self._futr_size})."
                )
            futr_exog = jnp.asarray(futr_exog, jnp.float32)
            if futr_exog.shape != (self.h, self._futr_size):
                raise ValueError(f"futr_exog must be (h={self.h}, F={self._futr_size}); got {futr_exog.shape}.")
            futr_full = jnp.concatenate([self._futr_ctx, futr_exog], axis=0)
        return predict_step(
            self.model_, self._context, h=self.h, input_size=self.input_size,
            scaler=self._scaler, hist_full=self._hist_ctx, futr_full=futr_full,
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
                f"TCN was trained for h={self.h}; predict(h={h}) is unsupported. Pass h <= {self.h}."
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
    def forecast(self, y, h, X=None, X_future=None, *, futr_exog=None,
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
            level (list[int] | None): Confidence levels in (0, 100).
            fitted (bool): Also return one-step-ahead in-sample predictions.

        Returns:
            dict: As :meth:`predict`, plus ``"fitted"`` (shape ``(T,)``, the
            first ``input_size`` entries ``NaN``) when ``fitted=True``.

        Raises:
            NotImplementedError: ``fitted=True`` on a model fit with exog.
            ValueError: Any ``fit`` or ``predict`` error.
        """
        self.fit(y, X=X, futr_exog=futr_exog)
        result = self.predict(h=h, futr_exog=X_future, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def conformity_scores(self, y, X=None) -> jnp.ndarray:
        """Walk-forward conformity scores (see the base class).

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (None): Historical exog is refused here; conformal intervals
                need an exog-free fit.

        Returns:
            jnp.ndarray: Signed residuals, shape ``(n_windows, h)``.

        Raises:
            ValueError: ``X`` is not ``None``; ``conformal_params`` is
                ``None``; or the series is too short for two
                cross-validation windows.
        """
        if X is not None:
            # Temporal exog (hist here) -> native (quantile) intervals, not
            # conformal, same refusal as _add_conformal. Guarding here keeps a
            # passed X out of the base cross-validation vmap, where it could
            # not raise under trace.
            raise ValueError(
                "Conformal intervals are not supported with temporal (historical/future) exog. "
                "Use loss=MultiQuantileLoss([...]) for native intervals, or omit level."
            )
        return super().conformity_scores(y, X)

    def _compute_fitted_values(self) -> jnp.ndarray:
        if self._has_temporal_exog:
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
            net = self._build_net()           # uses restored _hist_size/_futr_size/loss
            nnx.update(net, saved)
            self.model_ = net
            return
        self.__dict__.update(state)

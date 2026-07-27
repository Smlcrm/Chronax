"""XLinear forecaster: BaseForecaster wrapper around the JAX/Flax-NNX backbone."""
from __future__ import annotations

from typing import Callable, Union

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.xlinear.xlinear_losses import LossFn, resolve as _resolve_loss
from chronax.models.xlinear.xlinear_module import XLinearNet
from chronax.models.xlinear.xlinear_scaler import Scaler, resolve_scaler as _resolve_scaler
from chronax.models.xlinear.xlinear_training import predict_step, train


class XLinear(BaseForecaster):
    """Univariate XLinear forecaster (JAX/Flax-NNX port of neuralforecast.XLinear).

    Univariate adaptation of NF's multivariate gated-linear model (the benchmark
    harness runs the NF reference at ``n_series=1``): internal RevIN
    normalization (``use_norm``, on by default), a Linear projection to
    ``hidden_size``, a learned global token, temporal gating over the
    concatenated embedding, cross-channel gating over the [embedding, global]
    pair, and a linear head. Defaults match neuralforecast 3.1.7
    (hidden_size=128, temporal_ff=256, channel_ff=8, max_steps=1000,
    learning_rate=1e-3, windows_batch_size=32, identity scaler, MAE loss) so the
    harness compares both libraries at native settings.

    Maintenance Status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via
        ``predict(level=...)``, pickle round-trip, and ``forecast(fitted=True)``.

    NF's dropout knobs (all default 0.0) are intentionally not exposed; exogenous
    inputs raise NotImplementedError. ``input_size=-1`` (default; any value < 1)
    expands to ``3*h`` — a Chronax convenience; neuralforecast requires
    ``input_size`` explicitly. RevIN is INTERNAL and independent of the outer
    ``scaler`` knob ("identity" default, matching NF; "robust" also available).
    NF trains this model through its multivariate dataset path — at n_series=1 it
    degenerates to the univariate protocol used here, but batching semantics are
    not bit-verified (same accepted caveat as iTransformer). ``float32``
    throughout, matching torch/neuralforecast defaults.
    """

    uses_exog = False

    def __init__(self, h: int, input_size: int = -1, hidden_size: int = 128,
                 temporal_ff: int = 256, channel_ff: int = 8, use_norm: bool = True,
                 max_steps: int = 1000, learning_rate: Union[float, Callable[[int], float]] = 1e-3,
                 windows_batch_size: int = 32, loss: Union[str, LossFn] = "mae",
                 scaler: Union[str, Scaler] = "identity", random_seed: int = 1, alias: str = "XLinear"):
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.temporal_ff = temporal_ff
        self.channel_ff = channel_ff
        self.use_norm = use_norm
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.windows_batch_size = windows_batch_size
        self.loss = loss
        self.scaler = scaler
        self.random_seed = random_seed
        self.alias = alias
        self.conformal_params = None
        self.model_: XLinearNet | None = None
        self._context: jnp.ndarray | None = None
        self._train_y: jnp.ndarray | None = None

    @property
    def _loss_fn(self) -> LossFn:
        return _resolve_loss(self.loss)

    @property
    def _scaler(self) -> Scaler:
        return _resolve_scaler(self.scaler)

    def _build_net(self) -> XLinearNet:
        return XLinearNet(h=self.h, input_size=self.input_size, hidden_size=self.hidden_size,
                          temporal_ff=self.temporal_ff, channel_ff=self.channel_ff,
                          use_norm=self.use_norm, rngs=nnx.Rngs(self.random_seed))

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "XLinear":
        """Fit on a 1-D series. Raises NotImplementedError on exog; ValueError if
        y is not 1-D or shorter than input_size+h; RuntimeError on divergence."""
        if X is not None:
            raise NotImplementedError("Exogenous variables are not supported.")
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] < self.input_size + self.h:
            raise ValueError(f"Series length {y.shape[0]} too short for input_size={self.input_size} + h={self.h}.")
        scaler = self._scaler  # resolve early so an unknown scaler raises before building the net
        net = self._build_net()
        train(net, y, h=self.h, input_size=self.input_size, max_steps=self.max_steps,
              windows_batch_size=self.windows_batch_size, lr=self.learning_rate, seed=self.random_seed,
              scaler=scaler, loss_fn=self._loss_fn)
        self.model_ = net
        self._context = y[-self.input_size:]
        self._train_y = y
        return self

    def predict(self, h: int, X: jnp.ndarray | None = None,
                level: list[int | float] | None = None) -> dict:
        """Forecast h steps (1 <= h <= self.h). ``level`` -> inherited conformal path."""
        if self.model_ is None or self._context is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        if h < 1:
            raise ValueError(f"h must be a positive integer; got h={h}.")
        if h > self.h:
            raise ValueError(f"XLinear was trained for h={self.h}; predict(h={h}) is not supported. "
                             f"Pass h <= {self.h} or re-fit with a larger h.")
        full = predict_step(self.model_, self._context, h=self.h, input_size=self.input_size, scaler=self._scaler)
        fcst = {"mean": full[:h]}
        if level is not None:
            if self.conformal_params is None:
                raise ValueError(
                    "predict(h, level=...) requires `model.conformal_params` to be set. "
                    "Note: conformity_scores re-fits per CV window.")
            if self._train_y is None:
                raise RuntimeError("Call fit(y) before predict(h, level=...).")
            cs = self.conformity_scores(self._train_y)
            fcst = BaseForecaster.add_confidence_intervals(fcst, cs, level, self.conformal_params.method)
        return fcst

    def forecast(self, y: jnp.ndarray, h: int, X: jnp.ndarray | None = None,
                 X_future: jnp.ndarray | None = None, level: list[int | float] | None = None,
                 fitted: bool = False) -> dict:
        """Stateless fit-then-predict; ``fitted=True`` adds one-step-ahead fitted values."""
        if X is not None or X_future is not None:
            raise NotImplementedError("Exogenous variables are not supported.")
        self.fit(y)
        result = self.predict(h=h, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
        """One-step-ahead fitted values; first input_size entries are NaN."""
        if self._train_y is None or self.model_ is None:
            raise RuntimeError("Call fit(y) before computing fitted values.")
        y = self._train_y
        n_windows = y.shape[0] - self.input_size
        if n_windows <= 0:
            return jnp.full((y.shape[0],), jnp.nan, dtype=jnp.float32)
        idx = jnp.arange(self.input_size)[None, :] + jnp.arange(n_windows)[:, None]
        in_windows = y[idx]
        scaler = self._scaler
        shift, scale = scaler.stats(in_windows, axis=1)
        x_z = scaler.transform(in_windows, shift, scale)[..., None]
        pred_z = self.model_(x_z)                         # [n_windows, h, 1]
        first = scaler.inverse(pred_z[:, 0, 0:1], shift, scale)[:, 0]
        nan_head = jnp.full((self.input_size,), jnp.nan, dtype=jnp.float32)
        return jnp.concatenate([nan_head, first])

    def __getstate__(self) -> dict:
        """Serialize NNX Param state only and rebuild the GraphDef via
        _build_net() on load — smaller payload and robust to flax-internal
        GraphDef changes across versions (same convention as KAN)."""
        state = self.__dict__.copy()
        if state.get("model_") is not None:
            _, full_state = nnx.split(state["model_"])
            state["model_"] = ("__params_only__", full_state)
        return state

    def __setstate__(self, state: dict) -> None:
        m = state.get("model_")
        if isinstance(m, tuple) and m and m[0] == "__params_only__":
            saved_state = m[1]
            state["model_"] = None
            self.__dict__.update(state)
            net = self._build_net()
            nnx.update(net, saved_state)
            self.model_ = net
            return
        self.__dict__.update(state)

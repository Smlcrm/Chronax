"""RMoK forecaster: BaseForecaster wrapper around the JAX/Flax-NNX backbone."""
from __future__ import annotations

from typing import Callable, Union

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.rmok.rmok_losses import LossFn, resolve as _resolve_loss
from chronax.models.rmok.rmok_module import _WAVELETS, RMoKNet
from chronax.models.rmok.rmok_scaler import Scaler, resolve_scaler as _resolve_scaler
from chronax.models.rmok.rmok_training import predict_step, train


class RMoK(BaseForecaster):
    """Univariate RMoK forecaster (JAX/Flax-NNX port of neuralforecast.RMoK).

    Reversible Mixture of KAN experts (Han et al., 2024). Reversible instance
    normalization wraps a softmax-gated mixture of four experts that each map the
    input window to the horizon: a Taylor-KAN (polynomial expansion), a Jacobi-KAN
    (Jacobi-polynomial recurrence on ``tanh``-squashed input), a Wave-KAN (a
    wavelet transform followed by BatchNorm), and a plain linear layer. Defaults
    match neuralforecast 3.1.7 (``taylor_order=3, jacobi_degree=6,
    wavelet_function="mexican_hat", dropout=0.1, revin_affine=True,
    max_steps=1000, learning_rate=1e-3, windows_batch_size=32``, identity scaler,
    MAE loss) so the benchmark harness compares both libraries at native settings.

    This is a UNIVARIATE adaptation of a model neuralforecast marks multivariate:
    ``n_series`` is fixed to 1 internally (the benchmark auto-injects ``n_series=1``
    on the neuralforecast side). RevIN normalizes over the single channel; the
    WaveKAN expert's BatchNorm running statistics are trained through the fit and
    used at prediction (a ``deterministic`` flag drives both BatchNorm and dropout,
    like DeepNPTS).

    Maintenance Status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via
        ``predict(level=...)``, pickle round-trip (BatchNorm running stats and all
        params are preserved), and ``forecast(fitted=True)``.

    ``input_size=-1`` (default) expands to ``3*h`` — a Chronax convenience;
    neuralforecast requires ``input_size`` explicitly.

    ``scaler``: ``"identity"`` (default, matches neuralforecast — RevIN does the
    normalization) or ``"robust"``. Constructor guards (``taylor_order >= 0``,
    ``jacobi_degree >= 0``, ``wavelet_function`` in the five KAN4TSF wavelets,
    ``0 <= dropout < 1``) are mostly NF-faithful; ``dropout < 1`` is deliberately
    stricter than neuralforecast (``dropout=1`` would divide by zero here). Only
    ``wavelet_function="mexican_hat"`` (the default) is verified at parity with
    neuralforecast; the other four run but are not parity-checked. ``float32``
    throughout, matching torch/neuralforecast defaults.
    """

    uses_exog = False

    def __init__(self, h: int, input_size: int = -1, taylor_order: int = 3,
                 jacobi_degree: int = 6, wavelet_function: str = "mexican_hat",
                 dropout: float = 0.1, revin_affine: bool = True, max_steps: int = 1000,
                 learning_rate: Union[float, Callable[[int], float]] = 1e-3,
                 windows_batch_size: int = 32, loss: Union[str, LossFn] = "mae",
                 scaler: Union[str, Scaler] = "identity", random_seed: int = 1,
                 alias: str = "RMoK"):
        if input_size < 1:
            input_size = 3 * h
        if taylor_order < 0:
            raise ValueError(f"taylor_order must be >= 0; got {taylor_order}.")
        if jacobi_degree < 0:
            raise ValueError(f"jacobi_degree must be >= 0; got {jacobi_degree}.")
        if wavelet_function not in _WAVELETS:
            raise ValueError(f"wavelet_function must be one of {_WAVELETS}; got {wavelet_function!r}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0); got {dropout}.")
        self.h = h
        self.input_size = input_size
        self.taylor_order = taylor_order
        self.jacobi_degree = jacobi_degree
        self.wavelet_function = wavelet_function
        self.dropout = dropout
        self.revin_affine = revin_affine
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.windows_batch_size = windows_batch_size
        self.loss = loss
        self.scaler = scaler
        self.random_seed = random_seed
        self.alias = alias
        self.conformal_params = None
        self.model_: RMoKNet | None = None
        self._context: jnp.ndarray | None = None
        self._train_y: jnp.ndarray | None = None

    @property
    def _loss_fn(self) -> LossFn:
        return _resolve_loss(self.loss)

    @property
    def _scaler(self) -> Scaler:
        return _resolve_scaler(self.scaler)

    def _build_net(self) -> RMoKNet:
        return RMoKNet(h=self.h, input_size=self.input_size, n_series=1,
                       taylor_order=self.taylor_order, jacobi_degree=self.jacobi_degree,
                       wavelet_function=self.wavelet_function, dropout=self.dropout,
                       revin_affine=self.revin_affine, rngs=nnx.Rngs(self.random_seed))

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "RMoK":
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
            raise ValueError(f"RMoK was trained for h={self.h}; predict(h={h}) is not supported. "
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
        pred_z = self.model_(x_z, deterministic=True)     # [n_windows, h, 1]
        first = scaler.inverse(pred_z[:, 0, 0:1], shift, scale)[:, 0]
        nan_head = jnp.full((self.input_size,), jnp.nan, dtype=jnp.float32)
        return jnp.concatenate([nan_head, first])

    def __getstate__(self) -> dict:
        """Serialize NNX state (params + BatchNorm running stats) and rebuild the
        GraphDef via _build_net() on load — smaller payload and robust to
        flax-internal GraphDef changes across versions (same convention as KAN)."""
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

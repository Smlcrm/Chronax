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

    Reversible Mixture of KAN (Kolmogorov-Arnold Network) experts (Han et al.,
    2024). Reversible instance normalization (RevIN) wraps a softmax-gated
    mixture of four experts that each map the input window to the horizon: a
    Taylor-KAN (polynomial expansion), a Jacobi-KAN (Jacobi-polynomial
    recurrence on ``tanh``-squashed input), a Wave-KAN (a wavelet transform
    followed by BatchNorm), and a plain linear layer. Defaults match
    neuralforecast 3.1.7. ``float32`` throughout, matching torch and
    neuralforecast defaults.

    This is a univariate adaptation of a model neuralforecast marks
    multivariate: ``n_series`` is fixed to 1 internally. RevIN normalizes each
    window over time; the WaveKAN expert's BatchNorm running statistics are
    trained through the fit and used at prediction (a ``deterministic`` flag
    drives both BatchNorm and dropout, like DeepNPTS). Only
    ``wavelet_function="mexican_hat"`` (the default) has a neuralforecast parity
    test; the other four wavelets run but are not parity-checked.

    Maintenance status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via
        ``predict(level=...)``, pickle round-trip (BatchNorm running stats and
        all params are preserved), and ``forecast(fitted=True)``.

    Args:
        h (int): Forecast horizon.
        input_size (int): Input window length. Any value ``< 1`` (default
            ``-1``) resolves to ``3 * h``; neuralforecast requires an explicit
            value.
        taylor_order (int): Number of polynomial terms in the Taylor-KAN
            expert; must be ``>= 0``. Default ``3``.
        jacobi_degree (int): Degree of the Jacobi-KAN recurrence; must be
            ``>= 0``. Default ``6``.
        wavelet_function (str): One of ``"mexican_hat"``, ``"morlet"``,
            ``"dog"``, ``"meyer"``, ``"shannon"``. Default ``"mexican_hat"``.
        dropout (float): Dropout rate on the normalized input, in
            ``[0.0, 1.0)``. Default ``0.1``.
        revin_affine (bool): Learn an affine weight and bias inside RevIN.
            Default ``True``.
        max_steps (int): Number of Adam steps. Default ``1000``.
        learning_rate (float | Callable[[int], float]): Adam learning rate, a
            scalar or an ``optax`` schedule. Default ``1e-3``.
        windows_batch_size (int): Rolling windows sampled per step. Default
            ``32``.
        loss (str | LossFn): Registry name (``"mae"``, ``"mse"``, ``"huber"``)
            or a callable ``(pred, target, mask=None) -> scalar``. Default
            ``"mae"``.
        scaler (str | Scaler): Per-window scaler: ``"identity"`` (default,
            matches neuralforecast), ``"robust"`` (median and median absolute
            deviation, MAD), or a ``Scaler`` instance.
        random_seed (int): Seed for parameter init and window sampling.
            Default ``1``.
        alias (str): Display name. Default ``"RMoK"``.

    Raises:
        ValueError: If ``taylor_order < 0``, ``jacobi_degree < 0``,
            ``wavelet_function`` is not one of the five names above, or
            ``dropout`` is outside ``[0.0, 1.0)``. The ``dropout < 1`` bound
            is stricter than neuralforecast: ``dropout=1`` would zero the whole
            normalized input and feed the experts all-zeros.

    Note:
        A callable ``loss`` or ``learning_rate`` must itself be picklable for
        the fitted estimator to pickle. Unknown ``loss`` or ``scaler`` names
        raise ``ValueError`` at ``fit`` time.
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
        """Fit the network on a 1-D series.

        Builds the network and runs ``max_steps`` Adam steps over rolling
        windows of length ``input_size + h``, sampled per step the way
        neuralforecast does.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)`` with
                ``T >= input_size + h``.
            X (jnp.ndarray | None): Exogenous regressors. Not supported; must
                be ``None``.

        Returns:
            RMoK: ``self`` with ``model_`` populated.

        Raises:
            NotImplementedError: If ``X`` is not ``None``.
            ValueError: If ``y`` is not 1-D, if it is shorter than
                ``input_size + h``, or if ``loss`` or ``scaler`` is an unknown
                registry name.
            RuntimeError: If a non-finite training loss is observed (the
                training run diverged).
        """
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
        """Forecast ``h`` steps from the fitted context.

        The network is direct-decoded for ``self.h`` steps and the first ``h``
        are returned.

        Args:
            h (int): Forecast horizon, ``1 <= h <= self.h``.
            X (jnp.ndarray | None): Exogenous regressors. Ignored.
            level (list[int | float] | None): Confidence levels in (0, 100),
                e.g. ``[80, 95]``. When given, conformal intervals come from
                ``BaseForecaster.conformity_scores``, which re-fits the model on
                every cross-validation (CV) window under ``vmap``; requires
                ``self.conformal_params``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``) plus ``"lo-{lv}"`` and
            ``"hi-{lv}"`` (shape ``(h,)``) for each level when ``level`` is
            given.

        Raises:
            RuntimeError: If called before ``fit``.
            ValueError: If ``h < 1``, if ``h > self.h``, or if ``level`` is
                given while ``self.conformal_params`` is ``None``.
        """
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
        """Fit on ``y`` and forecast ``h`` steps in one call.

        Equivalent to ``self.fit(y).predict(h=h, level=level)``; the fitted
        state stays on the estimator.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon, ``1 <= h <= self.h``.
            X (jnp.ndarray | None): Exogenous regressors. Not supported; must
                be ``None``.
            X_future (jnp.ndarray | None): Future exogenous regressors. Not
                supported; must be ``None``.
            level (list[int | float] | None): Confidence levels; see
                ``predict``.
            fitted (bool): If ``True``, add a ``"fitted"`` key with
                one-step-ahead in-sample predictions, shape ``(T,)``; the first
                ``input_size`` entries are NaN.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``), the interval keys when ``level``
            is given, and ``"fitted"`` when ``fitted=True``.

        Raises:
            NotImplementedError: If ``X`` or ``X_future`` is not ``None``.
            ValueError: Forwarded from ``fit`` and ``predict``.
            RuntimeError: Forwarded from ``fit`` and ``predict``.
        """
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
        """Serialize NNX state (params + WaveKAN BatchNorm running stats) and
        rebuild the GraphDef via ``_build_net()`` on load: a smaller payload that
        does not depend on flax-internal GraphDef layout. A full ``nnx.split``
        captures both ``nnx.Param`` and the ``nnx.BatchStat`` running stats, so a
        reloaded model predicts identically (same convention as DeepNPTS, the
        sibling that also carries BatchNorm state)."""
        state = self.__dict__.copy()
        if state.get("model_") is not None:
            _, full_state = nnx.split(state["model_"])
            state["model_"] = ("__nnx_state__", full_state)
        return state

    def __setstate__(self, state: dict) -> None:
        m = state.get("model_")
        if isinstance(m, tuple) and m and m[0] == "__nnx_state__":
            saved_state = m[1]
            state["model_"] = None
            self.__dict__.update(state)
            net = self._build_net()
            nnx.update(net, saved_state)
            self.model_ = net
            return
        self.__dict__.update(state)

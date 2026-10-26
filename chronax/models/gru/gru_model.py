"""Gated recurrent unit (GRU) forecaster: BaseForecaster wrapper around the
JAX/Flax-NNX backbone.

Delegates the network to ``gru_module.GRUNet``, the training loop to
``gru_training.train``, per-window scaling to ``gru_scaler.RobustScaler`` and
single-window inference to ``gru_training.predict_step``.

Example:
    >>> import jax.numpy as jnp
    >>> from chronax.models import GRU
    >>> y = jnp.asarray([1.0, 2.0, 1.5, 1.7, 2.2, 1.9, 2.1, 2.4])
    >>> model = GRU(h=2, input_size=4, hidden_size=16, max_steps=20)
    >>> model.fit(y).predict(h=2)["mean"].shape
    (2,)
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from typing import Callable, Union

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.gru.gru_losses import LossFn, resolve as _resolve_loss
from chronax.models.gru.gru_module import GRUNet
from chronax.models.gru.gru_scaler import RobustScaler
from chronax.models.gru.gru_training import predict_step, train


class GRU(BaseForecaster):
    """Univariate gated recurrent unit (GRU) forecaster.

    The encoder is a stack of Flax NNX ``GRUCell`` layers rolled over time
    with ``jax.lax.scan``; the decoder is a two-layer multilayer perceptron
    (MLP). Each rolling window is z-scored with a robust median/MAD (median
    absolute deviation) scaler, and the model trains in scaled space with
    Optax ``adam`` and a pluggable point loss. The network is direct-decoded
    for exactly ``h`` steps; ``predict(h=k)`` slices for any ``k <= h``.

    Maintenance status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via
        ``predict(level=...)``, pickle round-trip, and ``forecast(fitted=True)``.

    Args:
        h (int): Forecast horizon.
        input_size (int): Input window length. Any value ``< 1`` (default
            ``-1``) resolves to ``3 * h``.
        hidden_size (int): Encoder hidden width. Default ``200``.
        n_layers (int): Number of stacked GRU layers. Default ``2``.
        decoder_hidden_size (int): Hidden width of the MLP decoder. Default
            ``128``.
        dropout (float): Dropout rate between encoder layers. Default ``0.0``.
        max_steps (int): Number of Adam steps. Default ``1000``.
        learning_rate (float | Callable[[int], float]): Adam learning rate, a
            scalar or an ``optax`` schedule. Default ``1e-3``.
        batch_size (int): Rolling windows sampled per step. Default ``128``.
        random_seed (int): Seed for parameter init and window sampling.
            Default ``1``.
        alias (str): Display name. Default ``"GRU"``.
        loss (str | LossFn): Registry name (``"mae"``, ``"mse"``, ``"huber"``)
            or a callable ``(pred, target) -> scalar``. Default ``"mae"``.
        recurrent_init (str): Initializer for the recurrent kernels:
            ``"uniform"`` (``U(-1/sqrt(H), 1/sqrt(H))``) or ``"orthogonal"``
            (Saxe-style). Input kernels and biases always use the uniform
            init. Default ``"uniform"``.

    Attributes:
        model_ (GRUNet | None): Fitted network after ``fit``.
        conformal_params (ConformalIntervals | None): Conformal settings read
            by ``predict(level=...)``; ``None`` until set by the caller.

    Note:
        A callable ``loss`` or ``learning_rate`` must itself be picklable for
        the fitted estimator to pickle. An unknown ``loss`` name raises
        ``ValueError`` at ``fit`` time. The decoder depth is fixed at two
        layers. The instance is stateful and not safe for concurrent
        mutation.
    """

    uses_exog = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        hidden_size: int = 200,
        n_layers: int = 2,
        decoder_hidden_size: int = 128,
        dropout: float = 0.0,
        max_steps: int = 1000,
        learning_rate: Union[float, Callable[[int], float]] = 1e-3,
        batch_size: int = 128,
        random_seed: int = 1,
        alias: str = "GRU",
        loss: Union[str, LossFn] = "mae",
        recurrent_init: str = "uniform",
    ):
        """Initialize a GRU forecaster.

        Stores hyperparameters; the network is built at ``fit`` time, so
        construction is cheap and side-effect free. Constructor arguments are
        documented on the class.
        """
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.dropout = dropout
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.alias = alias
        self.loss = loss
        self.recurrent_init = recurrent_init
        self.conformal_params = None
        self.model_: GRUNet | None = None
        self._context: jnp.ndarray | None = None  # last `input_size` of fit-time y
        self._train_y: jnp.ndarray | None = None  # full fit-time series (for fitted-values)
        self._scaler = RobustScaler()

    @property
    def _loss_fn(self) -> LossFn:
        """Resolve ``self.loss`` to a callable. Lazy so string forms pickle."""
        return _resolve_loss(self.loss)

    def _build_net(self) -> GRUNet:
        return GRUNet(
            in_features=1,
            encoder_hidden=self.hidden_size,
            encoder_layers=self.n_layers,
            decoder_hidden=self.decoder_hidden_size,
            decoder_layers=2,  # GRUNet enforces this invariant.
            dropout=self.dropout,
            h=self.h,
            input_size=self.input_size,
            rngs=nnx.Rngs(self.random_seed),
            recurrent_init=self.recurrent_init,
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "GRU":
        """Fit the network on a 1-D series.

        Builds the network and runs ``max_steps`` Adam steps over rolling
        windows of length ``input_size + h``, sampled uniformly with
        replacement at every step. Each window is scaled by its own robust
        median/MAD statistics before the forward pass, and the loss is taken
        in scaled space.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)`` with
                ``T >= input_size + h``.
            X (jnp.ndarray | None): Exogenous regressors. Not supported; must
                be ``None``.

        Returns:
            GRU: ``self`` with ``model_`` populated.

        Raises:
            NotImplementedError: If ``X`` is not ``None``.
            ValueError: If ``y`` is not 1-D, if it is shorter than
                ``input_size + h``, or if ``loss`` is an unknown
                registry name.
            RuntimeError: If a non-finite training loss is observed (the
                training run diverged).
        """
        if X is not None:
            raise NotImplementedError("Exogenous variables are not supported in v1.")
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] < self.input_size + self.h:
            raise ValueError(
                f"Series length {y.shape[0]} too short for "
                f"input_size={self.input_size} + h={self.h}."
            )
        net = self._build_net()
        train(
            net, y,
            h=self.h, input_size=self.input_size,
            max_steps=self.max_steps, batch_size=self.batch_size,
            lr=self.learning_rate, seed=self.random_seed,
            scaler=self._scaler,
            loss_fn=self._loss_fn,
        )
        self.model_ = net
        # Only the final `input_size` window is needed for direct-decoding
        # forecast; storing the full series would be wasteful for long inputs.
        self._context = y[-self.input_size :]
        self._train_y = y
        return self

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
    ) -> dict:
        """Forecast ``h`` steps from the fitted context.

        The network is direct-decoded for ``self.h`` steps and the first ``h``
        are returned.

        Args:
            h (int): Forecast horizon, ``h <= self.h``.
            X (jnp.ndarray | None): Exogenous regressors. Ignored.
            level (list[int | float] | None): Confidence levels in (0, 100),
                e.g. ``[80, 95]``. When given, conformal intervals come from
                ``BaseForecaster.conformity_scores``, which re-fits the model on
                every cross-validation (CV) window under ``vmap``; requires
                ``self.conformal_params``. Expect minutes per call on this
                model, scaling with ``n_windows`` and ``max_steps``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``) plus ``"lo-{lv}"`` and
            ``"hi-{lv}"`` (shape ``(h,)``) for each level when ``level`` is
            given.

        Raises:
            RuntimeError: If called before ``fit``.
            ValueError: If ``h > self.h``, or if ``level`` is given while
                ``self.conformal_params`` is ``None``.
        """
        if h > self.h:
            raise ValueError(
                f"GRU was trained for h={self.h}; predict(h={h}) is not supported. "
                f"Pass h <= {self.h} or re-fit with a larger h."
            )
        if self.model_ is None or self._context is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        full = predict_step(
            self.model_, self._context,
            h=self.h, input_size=self.input_size, scaler=self._scaler,
        )
        fcst = {"mean": full[:h]}
        if level is not None:
            if self.conformal_params is None:
                raise ValueError(
                    "predict(h, level=...) requires `model.conformal_params` to be "
                    "set before calling. Example:\n"
                    "    model.conformal_params = ConformalIntervals(n_windows=N, h=H)\n"
                    "Note: conformity_scores re-fits the model per CV window. On GRU "
                    "this is expensive — expect MINUTES per call, scaling linearly "
                    "with n_windows and with max_steps. Reduce n_windows or max_steps "
                    "if interactive feedback matters."
                )
            if self._train_y is None:
                raise RuntimeError("Call fit(y) before predict(h, level=...).")
            # Run the walk-forward on a clone: conformity_scores re-fits under vmap,
            # and those fit() writes would leave leaked tracers on this fitted estimator.
            cs = self.new().conformity_scores(self._train_y)
            method = self.conformal_params.method
            fcst = BaseForecaster.add_confidence_intervals(fcst, cs, level, method)
        return fcst

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
    ) -> dict:
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
            raise NotImplementedError("Exogenous variables are not supported in v1.")
        self.fit(y)
        result = self.predict(h=h, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
        """One-step-ahead fitted values over rolling windows of the training
        series. Returns shape ``(len(y),)`` where the first ``input_size``
        entries are NaN (no input window available) and the rest are the
        model's one-step-ahead predictions.
        """
        if self._train_y is None or self.model_ is None:
            raise RuntimeError("Call fit(y) before computing fitted values.")
        y = self._train_y
        n_windows = y.shape[0] - self.input_size
        if n_windows <= 0:
            return jnp.full((y.shape[0],), jnp.nan, dtype=jnp.float32)

        idx = jnp.arange(self.input_size)[None, :] + jnp.arange(n_windows)[:, None]
        in_windows = y[idx]  # [n_windows, input_size]
        shift, scale = self._scaler.stats(in_windows, axis=1)
        x_z = self._scaler.transform(in_windows, shift, scale)[..., None]
        pred_z = self.model_(x_z, deterministic=True)  # [n_windows, h, 1]
        first_step_z = pred_z[:, 0, 0:1]  # [n_windows, 1]
        finite_part = self._scaler.inverse(first_step_z, shift, scale)[:, 0]

        nan_head = jnp.full((self.input_size,), jnp.nan, dtype=jnp.float32)
        return jnp.concatenate([nan_head, finite_part])

    def __getstate__(self) -> dict:
        """Make the fitted model picklable.

        The full GraphDef returned by `nnx.split` retains references to JAX
        ufuncs (e.g. `jnp.tanh`, `jax.nn.sigmoid`) used as `activation_fn` /
        `gate_fn` on `nnx.GRUCell`. Those ufuncs aren't pickle-stable across
        JAX's lazy loading, so we serialize *only* the state (parameters +
        RNG streams) and rebuild the GraphDef on unpickle by re-running
        `_build_net()` from the same constructor args. The fitted parameter
        values are then loaded back via `nnx.update`.
        """
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
            # Rebuild a fresh network from the same constructor args, then
            # overwrite its state with the saved parameters.
            net = self._build_net()
            nnx.update(net, saved_state)
            self.model_ = net
            return
        self.__dict__.update(state)


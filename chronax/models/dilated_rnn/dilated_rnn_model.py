"""DilatedRNN forecaster: BaseForecaster wrapper around the JAX/Flax-NNX backbone."""
from __future__ import annotations

from typing import Callable, Sequence, Union

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.dilated_rnn.dilated_rnn_losses import LossFn, resolve as _resolve_loss
from chronax.models.dilated_rnn.dilated_rnn_module import CELL_TYPES, DilatedRNNNet
from chronax.models.dilated_rnn.dilated_rnn_scaler import Scaler, resolve as _resolve_scaler
from chronax.models.dilated_rnn.dilated_rnn_training import (
    make_lr_schedule, predict_step, train,
)
from chronax.utils import ConformalIntervals

_DEFAULT_DILATIONS = ((1, 2), (4, 8))


class DilatedRNN(BaseForecaster):
    """Univariate DilatedRNN forecaster (JAX/Flax-NNX port of neuralforecast.DilatedRNN).

    DilatedRNN (Chang et al. 2017, "Dilated Recurrent Neural Networks") stacks
    recurrent neural network (RNN) layers that each read the sequence
    subsampled at their own dilation rate. A layer at rate ``r`` splits the
    window into ``r`` interleaved subsequences and runs them as extra batch
    rows, so its receptive field spans ``r`` timesteps per recurrent step at
    unchanged cost; stacking rates ``1, 2, 4, 8`` gives an exponentially
    growing receptive field with a linear number of steps, which is the point
    of the architecture (and its answer to vanishing gradients over long
    windows).

    Layers are organised into groups (``dilations``), with a residual
    connection added between groups. The final hidden sequence is mapped from
    the lookback length to the horizon by a single linear "context adapter"
    (``Linear(input_size -> h)``), then decoded pointwise by a multilayer
    perceptron (MLP): the model is direct multi-step, not recursive.

    Each rolling window is scaled by a robust median/MAD (median absolute
    deviation) scaler (neuralforecast's ``scaler_type="robust"`` default), the
    model trains in that scaled space with Optax ``adam`` under
    neuralforecast's halving learning-rate staircase (``num_lr_decays``), and
    predictions are inverted with the prediction context's statistics.
    ``float32`` throughout. Defaults match neuralforecast.DilatedRNN.

    Maintenance status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via
        ``predict(level=...)``, pickle round-trip, and ``forecast(fitted=True)``.

    Args:
        h (int): Forecast horizon.
        input_size (int): Input window length. Any value ``< 1`` (default
            ``-1``) resolves to ``3 * h``.
        cell_type (str): One of ``"LSTM"`` (long short-term memory),
            ``"GRU"`` (gated recurrent unit), ``"RNN"``, ``"ResLSTM"``,
            ``"AttentiveLSTM"``. Default ``"LSTM"``.
        dilations (Sequence[Sequence[int]]): Groups of dilation rates. Each
            inner sequence is one group of stacked layers; a residual
            connection is added between groups. Default ``((1, 2), (4, 8))``.
        encoder_hidden_size (int): Recurrent hidden width (all layers).
            Default ``128``.
        context_size (int): Accepted for signature parity with neuralforecast,
            which stores it but never reads it in ``DilatedRNN.forward``. No
            effect here either. Default ``10``.
        decoder_hidden_size (int): Hidden width of the MLP decoder. Default
            ``128``.
        decoder_layers (int): Total layers in the MLP decoder (``1`` = bare
            linear). Default ``2``.
        max_steps (int): Number of Adam steps. Default ``1000``.
        learning_rate (float | Callable[[int], float]): Scalar, or an
            ``optax`` schedule callable. A callable is used as-is and
            ``num_lr_decays`` is ignored. Default ``1e-3``.
        num_lr_decays (int): Number of times the learning rate halves, evenly
            spread over ``max_steps`` (neuralforecast's ``StepLR``). ``<= 0``
            disables decay. Default ``3``.
        windows_batch_size (int): Rolling windows sampled per step. Default
            ``128``.
        scaler_type (str): Per-window scaler: ``"robust"`` (default),
            ``"standard"`` or ``"identity"``.
        random_seed (int): Seed for parameter init and window sampling.
            Default ``1``.
        alias (str): Display name. Default ``"DilatedRNN"``.
        loss (str | LossFn): Registry name (``"mae"``, ``"mse"``, ``"huber"``)
            or a callable. Default ``"mae"``.

    Raises:
        ValueError: On an unknown ``cell_type`` or ``scaler_type``, an empty
            or non-positive ``dilations`` spec, or ``decoder_layers < 1``.

    Note:
        A callable ``loss`` or ``learning_rate`` must itself be picklable for
        the fitted estimator to pickle. An unknown ``loss`` name raises
        ``ValueError`` at ``fit`` time.
    """

    uses_exog = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        cell_type: str = "LSTM",
        dilations: Sequence[Sequence[int]] = _DEFAULT_DILATIONS,
        encoder_hidden_size: int = 128,
        context_size: int = 10,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        max_steps: int = 1000,
        learning_rate: Union[float, Callable[[int], float]] = 1e-3,
        num_lr_decays: int = 3,
        windows_batch_size: int = 128,
        scaler_type: str = "robust",
        random_seed: int = 1,
        alias: str = "DilatedRNN",
        loss: Union[str, LossFn] = "mae",
    ):
        """Initialize a DilatedRNN forecaster.

        Stores hyperparameters; the network is built at ``fit`` time, so
        construction is cheap and side-effect free. Constructor arguments are
        documented on the class.
        """
        if input_size < 1:
            input_size = 3 * h
        if cell_type not in CELL_TYPES:
            raise ValueError(
                f"Unknown cell_type {cell_type!r}. Available: {list(CELL_TYPES)}."
            )
        groups = [list(g) for g in dilations]
        if not groups or any(len(g) == 0 for g in groups):
            raise ValueError(
                f"dilations must be a non-empty sequence of non-empty groups; got {dilations!r}."
            )
        if any(int(r) < 1 for g in groups for r in g):
            raise ValueError(f"dilation rates must be >= 1; got {dilations!r}.")
        if decoder_layers < 1:
            raise ValueError(f"decoder_layers must be >= 1; got {decoder_layers}.")
        _resolve_scaler(scaler_type)   # fail fast on an unknown name
        self.h = h
        self.input_size = input_size
        self.cell_type = cell_type
        self.dilations = groups
        self.encoder_hidden_size = encoder_hidden_size
        self.context_size = context_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.windows_batch_size = windows_batch_size
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        self.alias = alias
        self.loss = loss
        self.conformal_params: ConformalIntervals | None = None
        self.model_: DilatedRNNNet | None = None
        self._context: jnp.ndarray | None = None
        self._train_y: jnp.ndarray | None = None

    @property
    def _loss_fn(self) -> LossFn:
        """Resolve ``self.loss`` to a callable. Lazy so string forms pickle."""
        return _resolve_loss(self.loss)

    @property
    def _scaler(self) -> Scaler:
        """Resolve ``self.scaler_type`` to a Scaler. Lazy so the name pickles."""
        return _resolve_scaler(self.scaler_type)

    def _build_net(self) -> DilatedRNNNet:
        return DilatedRNNNet(
            h=self.h, input_size=self.input_size, in_features=1,
            cell_type=self.cell_type, dilations=self.dilations,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_layers=self.decoder_layers,
            rngs=nnx.Rngs(self.random_seed),
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "DilatedRNN":
        """Fit the network on a 1-D series.

        Builds the network and runs ``max_steps`` Adam steps over rolling
        windows of length ``input_size + h``, sampled per step the way
        neuralforecast does, under the halving learning-rate staircase.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)`` with
                ``T >= input_size + h``.
            X (jnp.ndarray | None): Exogenous regressors. Not supported; must
                be ``None``.

        Returns:
            DilatedRNN: ``self`` with ``model_`` populated.

        Raises:
            NotImplementedError: If ``X`` is not ``None``.
            ValueError: If ``y`` is not 1-D, if it is shorter than
                ``input_size + h``, or if ``loss`` is an unknown
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
            raise ValueError(
                f"Series length {y.shape[0]} too short for "
                f"input_size={self.input_size} + h={self.h}."
            )
        net = self._build_net()
        train(
            net, y, h=self.h, input_size=self.input_size, max_steps=self.max_steps,
            windows_batch_size=self.windows_batch_size,
            lr=make_lr_schedule(self.learning_rate, self.max_steps, self.num_lr_decays),
            seed=self.random_seed, scaler=self._scaler, loss_fn=self._loss_fn,
        )
        self.model_ = net
        self._context = y[-self.input_size:]
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
            h (int): Forecast horizon, ``1 <= h <= self.h``.
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
            ValueError: If ``h < 1``, if ``h > self.h``, or if ``level`` is
                given while ``self.conformal_params`` is ``None``.
        """
        if self.model_ is None or self._context is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        if h < 1:
            raise ValueError(f"h must be a positive integer; got h={h}.")
        if h > self.h:
            raise ValueError(
                f"DilatedRNN was trained for h={self.h}; predict(h={h}) is not supported. "
                f"Pass h <= {self.h} or re-fit with a larger h."
            )
        full = predict_step(self.model_, self._context, h=self.h,
                            input_size=self.input_size, scaler=self._scaler)
        fcst = {"mean": full[:h]}
        if level is not None:
            if self.conformal_params is None:
                raise ValueError(
                    "predict(h, level=...) requires `model.conformal_params` to be set. "
                    "Note: conformity_scores re-fits the model per CV window; on "
                    "DilatedRNN this is expensive — expect MINUTES per call, scaling "
                    "with n_windows and max_steps."
                )
            if self._train_y is None:
                raise RuntimeError("Call fit(y) before predict(h, level=...).")
            # Run on a throwaway copy: conformity_scores re-fits inside a vmap, and
            # calling it on self would overwrite this instance's fitted nnx weights
            # (self.model_) with tracers, breaking a later predict().
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
            raise NotImplementedError("Exogenous variables are not supported.")
        self.fit(y)
        result = self.predict(h=h, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
        """One-step-ahead fitted values; first ``input_size`` entries are NaN
        (all-NaN if the series is too short to form any rolling window).

        Each window is scaled with its OWN robust statistics and the prediction is
        inverted with the same pair, matching how ``predict`` treats the final window.
        """
        if self._train_y is None or self.model_ is None:
            raise RuntimeError("Call fit(y) before computing fitted values.")
        y = self._train_y
        n_windows = y.shape[0] - self.input_size
        if n_windows <= 0:
            return jnp.full((y.shape[0],), jnp.nan, dtype=jnp.float32)
        idx = jnp.arange(self.input_size)[None, :] + jnp.arange(n_windows)[:, None]
        in_windows = y[idx]                                  # [n_windows, L]
        scaler = self._scaler
        shift, scale = scaler.stats(in_windows, axis=1)
        z = scaler.transform(in_windows, shift, scale)[..., None]
        pred_z = self.model_(z, deterministic=True)          # [n_windows, h, 1]
        finite_part = scaler.inverse(pred_z[:, 0, 0], shift[:, 0], scale[:, 0])
        nan_head = jnp.full((self.input_size,), jnp.nan, dtype=jnp.float32)
        return jnp.concatenate([nan_head, finite_part])

    def __getstate__(self) -> dict:
        """Serialize only NNX state (params). The GraphDef holds references to JAX
        ufuncs that are not pickle-stable, so we drop it and rebuild via
        ``_build_net()`` on unpickle, then reload the saved state with ``nnx.update``.
        (DilatedRNN has no normalization layers with running stats, only
        ``nnx.Param`` variables are captured.)
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
            net = self._build_net()
            nnx.update(net, saved_state)
            self.model_ = net
            return
        self.__dict__.update(state)

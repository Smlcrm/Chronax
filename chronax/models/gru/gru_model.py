"""
File: gru_model.py

High-level Purpose:
    Provides a JAX/Flax/Optax GRU forecaster wrapper with a stable class
    interface for fitting and forecasting univariate time series.

Problem Solved:
    Encapsulates GRU training (rolling-window sampling, per-window robust
    scaling, JIT-compiled SGD with Adam) and direct-decoding inference behind
    a `BaseForecaster`-compatible API so downstream pipelines can treat a
    neural model identically to ARIMA / ETS / etc.

Architectural Role:
    Sits at the model-interface layer. Delegates network definition to
    `gru_module.GRUNet`, training loop to `gru_training.train`, scaling to
    `gru_scaler.RobustScaler`, and stateless single-window inference to
    `gru_training.predict_step`.

Major Classes/Functions:
    - `GRU`: Direct-decoding univariate GRU forecaster.

External Dependencies:
    - `jax`, `jax.numpy`, `flax.nnx`
    - Internal modules: `gru_module`, `gru_training`, `gru_scaler`,
      `base_forecaster`

Expected Inputs and Outputs:
    - Input: a 1-D `jnp.ndarray`-compatible series and a forecast horizon.
    - Output: dictionaries containing mean forecasts.

Example:
    >>> import jax.numpy as jnp
    >>> from chronax.models import GRU
    >>> y = jnp.asarray([1.0, 2.0, 1.5, 1.7, 2.2, 1.9, 2.1, 2.4])
    >>> model = GRU(h=2, input_size=4, hidden_size=16, max_steps=20)
    >>> model.fit(y)
    >>> model.predict(h=2)["mean"].shape
    (2,)

Assumptions:
    - The series is non-empty, 1-D, numeric, and finite.
    - `len(y) >= input_size + h` at fit time.

Side Effects:
    - Mutates instance state (`model_`, cached context window).
    - JIT-compiles a forward pass on first `predict` call after `fit`.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.gru.gru_module import GRUNet
from chronax.models.gru.gru_scaler import RobustScaler
from chronax.models.gru.gru_training import predict_step, train


class GRU(BaseForecaster):
    """
    GRU

    Description:
        Univariate gated-recurrent-unit forecaster. The encoder is a stack of
        Flax NNX `GRUCell`s rolled over time with `jax.lax.scan`; the decoder
        is a 2-layer MLP. Each rolling window is z-scored with a robust
        median/MAD scaler, and the model is trained with MAE loss in scaled
        space using Optax `adam`.

    Maintenance Status:
        Frozen v1 — this model is shipped as-is and will not receive feature
        additions or bug-fix releases unless a downstream consumer surfaces a
        correctness issue. Knobs intentionally kept minimal. For wider
        configuration (custom optimizer, custom loss, exogenous variables,
        recursive decoding, early stopping), prefer one of the stats-based
        forecasters in ``chronax.models`` or a peer neural library.

    Attributes:
        uses_exog (bool): Indicates support for exogenous regressors. False
            in v1.
        h (int): Forecast horizon (model is direct-decoded for exactly `h`
            steps; `predict(h=k)` slices for any ``k <= h``).
        input_size (int): Length of the input window. Defaults to ``3 * h``
            when ``-1`` is passed.
        hidden_size (int): GRU encoder hidden dimension.
        n_layers (int): Number of stacked GRU layers.
        decoder_hidden_size (int): MLP decoder hidden dimension.
        dropout (float): Inter-layer dropout in the encoder.
        max_steps (int): Number of optimizer steps during fit.
        learning_rate (float): Adam learning rate.
        batch_size (int): Number of windows per gradient step.
        random_seed (int): Seed used for parameter init and batch sampling.
        alias (str): Display name for external reporting.
        ``model_`` (GRUNet | None): Fitted network after ``fit``.
        conformal_params: Always None in v1 (intervals not implemented).

    Args:
        h (int): Forecast horizon.
        input_size (int): Input-window length; -1 (default) uses ``3 * h``.
        hidden_size (int): Encoder hidden dimension.
        n_layers (int): Number of stacked GRU cells.
        decoder_hidden_size (int): MLP-decoder hidden dimension.
        dropout (float): Inter-layer dropout rate.
        max_steps (int): Number of training steps.
        learning_rate (float): Adam learning rate.
        batch_size (int): Number of rolling windows sampled per step.
        random_seed (int): Random seed for reproducibility.
        alias (str): Friendly model name.

    Methods:
        fit(): Train the network on a 1-D series.
        predict(): Forecast ``h`` steps from the last fitted context.
        forecast(): Stateless fit-then-predict.

    Returns:
        Produces forecasts via dictionaries keyed by ``mean``.

    Example:
        >>> import jax.numpy as jnp
        >>> from chronax.models import GRU
        >>> y = jnp.asarray([1.0, 2.0, 1.5, 1.7, 2.2, 1.9, 2.1, 2.4])
        >>> model = GRU(h=2, input_size=4, hidden_size=16, max_steps=20)
        >>> model.fit(y).predict(h=2)["mean"].shape
        (2,)

    Notes:
        This class is stateful and not thread-safe for concurrent mutation.
        Defaults: `hidden_size=200`, `n_layers=2`, `decoder_hidden_size=128`,
        `max_steps=1000`, `learning_rate=1e-3`.
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
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        random_seed: int = 1,
        alias: str = "GRU",
    ):
        """
        Initialize a GRU forecaster.

        Detailed Description:
            Stores hyperparameters; the network is not built until ``fit`` is
            called (so ``__init__`` is cheap and side-effect free).

        Args:
            h (int): Forecast horizon.
            input_size (int): Input-window length. Use -1 for ``3 * h``.
            hidden_size (int): Encoder hidden dimension.
            n_layers (int): Number of stacked GRU cells.
            decoder_hidden_size (int): MLP-decoder hidden dimension.
            dropout (float): Inter-layer dropout rate.
            max_steps (int): Optimizer steps for ``fit``.
            learning_rate (float): Adam learning rate.
            batch_size (int): Windows per training step.
            random_seed (int): Random seed.
            alias (str): User-facing model name.

        Returns:
            None: Constructor initializes estimator state.

        Notes:
            ``decoder_layers`` is hard-wired to 2 in v1; expose it again
            once the decoder supports configurable depth.
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
        self.conformal_params = None
        self.model_: GRUNet | None = None
        self._context: jnp.ndarray | None = None  # last `input_size` of fit-time y
        self._train_y: jnp.ndarray | None = None  # full fit-time series (for fitted-values)
        self._scaler = RobustScaler()

    def _build_net(self) -> GRUNet:
        return GRUNet(
            in_features=1,
            encoder_hidden=self.hidden_size,
            encoder_layers=self.n_layers,
            decoder_hidden=self.decoder_hidden_size,
            decoder_layers=2,  # v1 only; GRUNet enforces this invariant.
            dropout=self.dropout,
            h=self.h,
            input_size=self.input_size,
            rngs=nnx.Rngs(self.random_seed),
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "GRU":
        """
        Fit the GRU on a univariate series.

        Detailed Description:
            Builds the network with the configured hyperparameters, then runs
            ``self.max_steps`` Adam steps over rolling windows of length
            ``input_size + h`` sampled uniformly at random. Each window is
            normalized with the per-window robust scaler before the forward
            pass; loss is MAE in scaled space.

        Args:
            y (jnp.ndarray): 1-D series of length ``>= input_size + h``.
            X: Reserved for future exogenous regressors; must be None in v1.

        Returns:
            GRU: ``self``, with ``model_`` populated.

        Raises:
            NotImplementedError: If ``X`` is provided.
            ValueError: If ``y`` is not 1-D, or shorter than
                ``input_size + h``.
            RuntimeError: If a non-finite training loss is observed
                (training diverged); message includes the offending step.

        Side Effects:
            Mutates ``self.model_`` and ``self._context``.
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
        """
        Forecast from the fitted context.

        Detailed Description:
            Runs a single deterministic forward pass on the cached context
            window (last ``input_size`` of the fit-time series), then slices
            to the requested ``h``. Always materializes ``self.h`` outputs
            internally so the JIT cache is shared across calls regardless of
            the caller's ``h``.

        Args:
            h (int): Forecast horizon. Must satisfy ``h <= self.h``.
            X: Reserved for future exogenous regressors; ignored in v1.
            level (list[int | float] | None): If provided, returns conformal
                prediction intervals as additional ``lo-XX``/``hi-XX`` keys
                (e.g. ``lo-80``, ``hi-80`` for level=[80]). Requires
                ``self.conformal_params`` to be set. The inherited
                conformity-score computation re-fits the model per CV
                window; on GRU expect MINUTES per call. Reduce
                ``n_windows`` or ``max_steps`` if interactive feedback
                matters.

        Returns:
            dict: ``{"mean": jnp.ndarray of shape (h,)}``.

        Raises:
            ValueError: If ``h > self.h``.
            NotImplementedError: If ``level`` is provided.
            RuntimeError: If called before ``fit``.
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
            cs = self.conformity_scores(self._train_y)
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
        """
        Stateless fit-then-predict.

        Detailed Description:
            Equivalent to ``self.fit(y).predict(h=h, level=level)`` (plus an
            extra ``"fitted"`` key when ``fitted=True``) and numerically
            identical to that pattern when seeded the same way.

        Args:
            y (jnp.ndarray): 1-D training series.
            h (int): Forecast horizon (``<= self.h``).
            X / X_future: Reserved; must be None in v1.
            level: Optional confidence levels (e.g. [80, 95]). When set,
                returns conformal intervals via the inherited
                BaseForecaster.add_confidence_intervals path. See note on
                cost (minutes per call) in ``predict``.
            fitted (bool): If True, the returned dict additionally contains
                a ``"fitted"`` key with one-step-ahead predictions over the
                training series. The first ``input_size`` entries are NaN
                (no input window available); the remaining entries are
                finite. Matches the convention used by other Chronax models.

        Returns:
            dict: ``{"mean": ..., optional "fitted": ...}``.

        Raises:
            NotImplementedError: If ``X`` or ``X_future`` is non-None, or if
                ``level`` is non-None.
            ValueError: Forwarded from ``fit`` / ``predict``.
            RuntimeError: Forwarded from ``fit`` (training divergence) or
                ``predict``.
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


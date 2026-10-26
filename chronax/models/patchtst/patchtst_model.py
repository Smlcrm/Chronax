"""PatchTST forecaster: BaseForecaster wrapper around the JAX/Flax-NNX backbone."""
from __future__ import annotations

from typing import Callable, Union

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.patchtst.patchtst_losses import LossFn, resolve as _resolve_loss
from chronax.models.patchtst.patchtst_module import PatchTSTNet
from chronax.models.patchtst.patchtst_training import predict_step, train


class PatchTST(BaseForecaster):
    """Univariate PatchTST forecaster (JAX/Flax-NNX port of neuralforecast.PatchTST).

    The encoder patches the input window, embeds each patch, and runs a stack
    of transformer layers (residual attention + BatchNorm + GELU feed-forward)
    with reversible instance normalization (RevIN) applied inside the network.
    Trained in original scale with Optax ``adam`` and a pluggable point loss.
    ``float32`` throughout. Defaults match neuralforecast.PatchTST.

    Maintenance status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via
        ``predict(level=...)``, pickle round-trip, and ``forecast(fitted=True)``.

    Args:
        h (int): Forecast horizon.
        input_size (int): Input window length. Any value ``< 1`` (default
            ``-1``) resolves to ``3 * h``.
        patch_len (int): Patch length; clamped to ``input_size + stride`` as in
            neuralforecast. Default ``16``.
        stride (int): Patch stride. Default ``8``.
        hidden_size (int): Patch embedding width. Default ``128``.
        n_heads (int): Attention heads. Default ``16``.
        encoder_layers (int): Encoder layers. Default ``3``.
        linear_hidden_size (int): Feed-forward hidden width. Default ``256``.
        dropout (float): Dropout rate in the patch embedding and encoder
            layers. Default ``0.2``.
        fc_dropout (float): Accepted for neuralforecast-signature parity;
            inert, because neuralforecast uses it only in its disabled
            pretrain head. Default ``0.2``.
        head_dropout (float): Dropout rate on the flatten head. Default
            ``0.0``.
        attn_dropout (float): Dropout rate on the attention weights. Default
            ``0.0``.
        activation (str): ``"gelu"`` or ``"relu"``. Default ``"gelu"``.
        revin (bool): Apply RevIN. Default ``True``.
        revin_affine (bool): Learn an affine weight and bias inside RevIN.
            Default ``False``.
        revin_subtract_last (bool): Subtract the last window value instead of
            the mean inside RevIN. Default ``True``.
        max_steps (int): Number of Adam steps. Default ``5000``.
        learning_rate (float | Callable[[int], float]): Adam learning rate, a
            scalar or an ``optax`` schedule. Default ``1e-4``.
        windows_batch_size (int): Rolling windows sampled per step. Default
            ``1024``.
        random_seed (int): Seed for parameter init, dropout and window
            sampling. Default ``1``.
        alias (str): Display name. Default ``"PatchTST"``.
        loss (str | LossFn): Registry name (``"mae"``, ``"mse"``, ``"huber"``)
            or a callable. Default ``"mae"``.

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
        patch_len: int = 16,
        stride: int = 8,
        hidden_size: int = 128,
        n_heads: int = 16,
        encoder_layers: int = 3,
        linear_hidden_size: int = 256,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        revin: bool = True,
        revin_affine: bool = False,
        revin_subtract_last: bool = True,
        max_steps: int = 5000,
        learning_rate: Union[float, Callable[[int], float]] = 1e-4,
        windows_batch_size: int = 1024,
        random_seed: int = 1,
        alias: str = "PatchTST",
        loss: Union[str, LossFn] = "mae",
    ):
        """Initialize a PatchTST forecaster.

        Stores hyperparameters; the network is built lazily at ``fit`` time so
        construction is cheap and side-effect free. Defaults match
        neuralforecast.PatchTST. ``input_size=-1`` resolves to ``3 * h``. ``loss``
        is a registry name (``"mae"``/``"mse"``/``"huber"``) or a callable;
        ``learning_rate`` is a scalar or an ``optax.ScalarOrSchedule``;
        ``activation`` is ``"gelu"`` or ``"relu"``. If the fitted estimator will
        be pickled, any callable passed for ``loss``/``learning_rate`` must itself
        be picklable (a class-based callable or module-level function).
        """
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.patch_len = patch_len
        self.stride = stride
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.encoder_layers = encoder_layers
        self.linear_hidden_size = linear_hidden_size
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.attn_dropout = attn_dropout
        self.activation = activation
        self.revin = revin
        self.revin_affine = revin_affine
        self.revin_subtract_last = revin_subtract_last
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.windows_batch_size = windows_batch_size
        self.random_seed = random_seed
        self.alias = alias
        self.loss = loss
        self.conformal_params = None
        self.model_: PatchTSTNet | None = None
        self._context: jnp.ndarray | None = None
        self._train_y: jnp.ndarray | None = None

    @property
    def _loss_fn(self) -> LossFn:
        """Resolve ``self.loss`` to a callable. Lazy so string forms pickle."""
        return _resolve_loss(self.loss)

    def _build_net(self) -> PatchTSTNet:
        return PatchTSTNet(
            h=self.h, input_size=self.input_size, patch_len=self.patch_len,
            stride=self.stride, hidden_size=self.hidden_size, n_heads=self.n_heads,
            encoder_layers=self.encoder_layers, linear_hidden_size=self.linear_hidden_size,
            dropout=self.dropout, fc_dropout=self.fc_dropout,
            head_dropout=self.head_dropout, attn_dropout=self.attn_dropout,
            activation=self.activation,
            revin=self.revin, revin_affine=self.revin_affine,
            revin_subtract_last=self.revin_subtract_last, rngs=nnx.Rngs(self.random_seed),
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "PatchTST":
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
            PatchTST: ``self`` with ``model_`` populated.

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
            windows_batch_size=self.windows_batch_size, lr=self.learning_rate,
            seed=self.random_seed, loss_fn=self._loss_fn,
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
                f"PatchTST was trained for h={self.h}; predict(h={h}) is not supported. "
                f"Pass h <= {self.h} or re-fit with a larger h."
            )
        full = predict_step(self.model_, self._context, h=self.h, input_size=self.input_size)
        fcst = {"mean": full[:h]}
        if level is not None:
            if self.conformal_params is None:
                raise ValueError(
                    "predict(h, level=...) requires `model.conformal_params` to be set. "
                    "Note: conformity_scores re-fits the model per CV window; on PatchTST "
                    "this is expensive — expect MINUTES per call, scaling with n_windows "
                    "and max_steps."
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
            raise NotImplementedError("Exogenous variables are not supported.")
        self.fit(y)
        result = self.predict(h=h, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
        """One-step-ahead fitted values; first ``input_size`` entries are NaN
        (all-NaN if the series is too short to form any rolling window)."""
        if self._train_y is None or self.model_ is None:
            raise RuntimeError("Call fit(y) before computing fitted values.")
        y = self._train_y
        n_windows = y.shape[0] - self.input_size
        if n_windows <= 0:
            return jnp.full((y.shape[0],), jnp.nan, dtype=jnp.float32)
        idx = jnp.arange(self.input_size)[None, :] + jnp.arange(n_windows)[:, None]
        in_windows = y[idx][..., None]                      # [n_windows, L, 1]
        pred = self.model_(in_windows, deterministic=True, use_running_average=True)
        finite_part = pred[:, 0, 0]                         # one-step-ahead
        nan_head = jnp.full((self.input_size,), jnp.nan, dtype=jnp.float32)
        return jnp.concatenate([nan_head, finite_part])

    def __getstate__(self) -> dict:
        """Serialize only NNX state (params + BatchNorm running stats + RNG).

        ``nnx.split`` captures every Variable type, including ``BatchStat``
        running mean/var. The GraphDef holds references to JAX ufuncs that are
        not pickle-stable, so we drop it and rebuild via ``_build_net()`` on
        unpickle, then reload the saved state with ``nnx.update``.
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

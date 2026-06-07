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

    Maintenance Status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via
        ``predict(level=...)``, pickle round-trip, and ``forecast(fitted=True)``.

    The encoder patches the input window, embeds each patch, and runs a stack of
    transformer layers (residual attention + BatchNorm + GELU feed-forward) with
    RevIN normalization applied inside the network. Trained in original scale
    with Optax ``adam`` and a pluggable point loss. ``float32`` throughout.
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

    def predict(self, h, X=None, level=None) -> dict:
        if self.model_ is None or self._context is None:
            raise RuntimeError("Call fit(y) before predict(h).")
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
            cs = self.conformity_scores(self._train_y)
            method = self.conformal_params.method
            fcst = BaseForecaster.add_confidence_intervals(fcst, cs, level, method)
        return fcst

    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False) -> dict:
        if X is not None or X_future is not None:
            raise NotImplementedError("Exogenous variables are not supported.")
        self.fit(y)
        result = self.predict(h=h, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
        """One-step-ahead fitted values; first ``input_size`` entries are NaN."""
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

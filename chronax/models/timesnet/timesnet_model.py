"""TimesNet forecaster: BaseForecaster wrapper around the JAX/Flax-NNX backbone."""
from __future__ import annotations

from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.timesnet.timesnet_losses import LossFn, resolve as _resolve_loss
from chronax.models.timesnet.timesnet_module import TimesNetNet
from chronax.models.timesnet.timesnet_scaler import Scaler, resolve_scaler as _resolve_scaler
from chronax.models.timesnet.timesnet_training import predict_step, train


def _compute_periods(y: np.ndarray, input_size: int, h: int, top_k: int,
                     scaler: Scaler) -> tuple[tuple, tuple]:
    """Static per-fit periods from the CONFIGURED-scaler-scaled training windows.

    DISCLOSED DIVERGENCE from neuralforecast: NF re-selects top-k per batch from
    the hidden (embedded + predict_linear-extended) representation, whose
    init-time spectrum is dominated by positional-encoding/init artifacts; we
    select once per fit from the data's own spectrum (see class docstring).
    Windows are the training windows (right-padded with h zeros, like
    ``build_windows``). Freq 0 zeroed; period = T // freq; frequencies returned
    in descending amplitude order (torch.topk order)."""
    T = input_size + h
    n = len(y) - input_size
    ypad = np.concatenate([y, np.zeros(h, dtype=y.dtype)])
    windows = np.stack([ypad[i:i + T] for i in range(n)])
    ins = jnp.asarray(windows[:, :input_size])
    shift, scale = scaler.stats(ins, axis=1)
    wz = np.asarray(scaler.transform(jnp.asarray(windows), shift, scale))
    amp = np.abs(np.fft.rfft(wz, axis=1)).mean(axis=0)
    amp[0] = 0.0
    freqs = tuple(int(f) for f in np.argsort(amp)[-top_k:][::-1])
    periods = tuple(T // f for f in freqs)
    return periods, freqs


class TimesNet(BaseForecaster):
    """Univariate TimesNet forecaster (JAX/Flax-NNX port of neuralforecast.TimesNet).

    TimesNet (Wu et al., 2023) folds the input window into 2-D [cycle x period]
    grids at its top-k dominant periods, applies inception-style 2-D convolutions
    (kernels 1..2*num_kernels-1, averaged) to model intra- and inter-period
    variation, and aggregates the per-period branches with amplitude-softmax
    weights. Architecture per neuralforecast 3.1.7: circular-conv token embedding
    + fixed sinusoidal positional encoding with active dropout, a Linear
    extension of the time axis from input_size to input_size+h, encoder_layers
    TimesBlocks each followed by ONE shared LayerNorm, and a Linear projection;
    the forecast is the last h steps. Defaults match neuralforecast (hidden_size
    =64, dropout=0.1, conv_hidden_size=64, top_k=5, num_kernels=6,
    encoder_layers=2, max_steps=1000, learning_rate=1e-4, windows_batch_size=64,
    standard scaler, MAE loss) so the benchmark harness compares both libraries
    at native settings.

    Period selection — deliberate, disclosed divergence from neuralforecast:
    JAX requires static shapes, so the top-k periods cannot be recomputed per
    batch in-graph as neuralforecast does. Chronax computes them ONCE per
    ``fit()`` on the host, from the amplitude spectrum of the configured-scaler-
    scaled training windows (mean over windows, freq 0 zeroed), and shares the
    static set across encoder layers; the per-sample softmax weights at those
    frequencies remain input-dependent and in-graph (formula faithful).
    Neuralforecast instead recomputes top-k per layer per batch from the hidden
    sequence — at init that operand's spectrum is dominated by positional-
    encoding and random-init artifacts rather than the data (observed dev-time:
    zero overlap with the raw-data top-k on the airline series), so the data
    spectrum is also the more principled selector. Consequences are adjudicated
    by the committed accuracy benchmark, not assumed. ``conformity_scores``'
    vmapped re-fits see traced data and REUSE the parent fit's periods
    (``periods_``/``freqs_``, also pickled).

    Maintenance Status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via
        ``predict(level=...)``, pickle round-trip, and ``forecast(fitted=True)``.

    ``input_size=-1`` (default) expands to ``3*h`` — a Chronax convenience;
    neuralforecast requires ``input_size`` explicitly.

    ``scaler``: ``"standard"`` (default, matches neuralforecast's
    ``scaler_type='standard'``), ``"identity"``, or ``"robust"``. Constructor
    guards (``top_k`` within the window's nonzero rfft bins, ``num_kernels``/
    ``encoder_layers`` >= 1) are Chronax additions — neuralforecast fails later
    and more obscurely on the same inputs. ``float32`` throughout, matching
    torch/neuralforecast defaults.
    """

    uses_exog = False

    def __init__(self, h: int, input_size: int = -1, hidden_size: int = 64,
                 dropout: float = 0.1, conv_hidden_size: int = 64, top_k: int = 5,
                 num_kernels: int = 6, encoder_layers: int = 2, max_steps: int = 1000,
                 learning_rate: Union[float, Callable[[int], float]] = 1e-4,
                 windows_batch_size: int = 64, loss: Union[str, LossFn] = "mae",
                 scaler: Union[str, Scaler] = "standard", random_seed: int = 1,
                 alias: str = "TimesNet"):
        if input_size < 1:
            input_size = 3 * h
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {top_k}.")
        if top_k > (input_size + h) // 2:
            raise ValueError(
                f"top_k={top_k} exceeds the {(input_size + h) // 2} nonzero rfft bins of a "
                f"length-{input_size + h} window (input_size={input_size} + h={h}).")
        if num_kernels < 1:
            raise ValueError(f"num_kernels must be >= 1; got {num_kernels}.")
        if encoder_layers < 1:
            raise ValueError(f"encoder_layers must be >= 1; got {encoder_layers}.")
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.conv_hidden_size = conv_hidden_size
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.encoder_layers = encoder_layers
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.windows_batch_size = windows_batch_size
        self.loss = loss
        self.scaler = scaler
        self.random_seed = random_seed
        self.alias = alias
        self.conformal_params = None
        self.model_: TimesNetNet | None = None
        self.periods_: tuple | None = None
        self.freqs_: tuple | None = None
        self._context: jnp.ndarray | None = None
        self._train_y: jnp.ndarray | None = None

    @property
    def _loss_fn(self) -> LossFn:
        return _resolve_loss(self.loss)

    @property
    def _scaler(self) -> Scaler:
        return _resolve_scaler(self.scaler)

    def _build_net(self) -> TimesNetNet:
        if self.periods_ is None or self.freqs_ is None:
            raise RuntimeError("Periods must be computed (via fit) before building the net.")
        return TimesNetNet(h=self.h, input_size=self.input_size, hidden_size=self.hidden_size,
                           conv_hidden_size=self.conv_hidden_size, top_k=self.top_k,
                           num_kernels=self.num_kernels, encoder_layers=self.encoder_layers,
                           dropout=self.dropout, periods=self.periods_, freqs=self.freqs_,
                           rngs=nnx.Rngs(self.random_seed))

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "TimesNet":
        """Fit on a 1-D series. Raises NotImplementedError on exog; ValueError if
        y is not 1-D or shorter than input_size+h; RuntimeError on divergence.

        Computes the static periods on the host first (see class docstring).
        Under a JAX trace (``conformity_scores``' vmapped re-fits) the data is
        not concrete, so the parent fit's ``periods_`` are reused; a traced fit
        without a prior concrete fit raises RuntimeError."""
        if X is not None:
            raise NotImplementedError("Exogenous variables are not supported.")
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] < self.input_size + self.h:
            raise ValueError(f"Series length {y.shape[0]} too short for input_size={self.input_size} + h={self.h}.")
        scaler = self._scaler  # resolve early so an unknown scaler raises before building the net
        try:
            y_host = np.asarray(y)
        except jax.errors.TracerArrayConversionError:
            if self.periods_ is None or self.freqs_ is None:
                raise RuntimeError(
                    "TimesNet.fit under a JAX trace (e.g. conformity_scores) requires a prior "
                    "concrete fit: static periods are computed from concrete data and reused "
                    "for traced re-fits.") from None
        else:
            self.periods_, self.freqs_ = _compute_periods(
                y_host, self.input_size, self.h, self.top_k, scaler)
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
            raise ValueError(f"TimesNet was trained for h={self.h}; predict(h={h}) is not supported. "
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

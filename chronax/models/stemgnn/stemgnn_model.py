"""StemGNN forecaster: BaseForecaster wrapper around the flax.nnx backbone.

Univariate point and multi-quantile forecasting via the Spectral Temporal Graph
Neural Network (Cao et al., 2020, https://arxiv.org/abs/2103.07719), a flax.nnx
port of ``neuralforecast.models.StemGNN``. Intervals come from the conformal
path (point loss) or natively from the quantile heads. No exogenous support.
``float32`` throughout.

StemGNN is natively multivariate; this wrapper runs it at ``n_series=1``, where
it is degenerate: the single-node graph Laplacian is exactly 0 and the Chebyshev
expansion uses ``T0 = zeros`` (not identity), so the forecast head sees only
layer biases. The network output is then a learned CONSTANT per horizon step in
per-window-scaled space, and the only data-dependent part of the forecast is the
robust scaler's inverse (~ window median + c * window MAD).
``chebyshev_first_term="identity"`` opts into the paper's ``T0 = I``, restoring a
real data path so the univariate model genuinely forecasts.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.stemgnn.stemgnn_losses import (
    MultiQuantileLoss, outputsize_multiplier, resolve as _resolve_loss,
)
from chronax.models.stemgnn.stemgnn_module import StemGNNNet
from chronax.models.stemgnn.stemgnn_scaler import resolve_scaler
from chronax.models.stemgnn.stemgnn_training import predict_step, train
from chronax.utils import ConformalIntervals


def _nearest_q_index(quantiles, target: float) -> int:
    """Host-side index of the quantile closest to ``target``. Operates on the static
    quantile tuple (compile-time constants), so it is independent of any traced array."""
    return min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - target))


class StemGNN(BaseForecaster):
    """StemGNN: Spectral Temporal Graph Neural Network
    (flax.nnx port of neuralforecast's StemGNN).

    Cao et al., 2020 -- https://arxiv.org/abs/2103.07719. A latent correlation
    layer (GRU over the series axis + additive attention) learns a graph over
    series; two residual ``StockBlockLayer`` stacks apply a 4-term Chebyshev
    "GFT" of its normalized Laplacian, a 4-point DFT over the Chebyshev-order
    axis with GLU filtering (the Spe-Seq cell), a per-order graph-conv kernel,
    and sigmoid-gated forecast/backcast heads; a final MLP maps to the horizon.
    Trained with a StepLR schedule (``num_lr_decays``). This wrapper is
    univariate (``n_series = 1`` internally — see the module docstring for the
    N=1 degeneracy and the ``chebyshev_first_term`` escape hatch). No exogenous
    support. Point or multi-quantile losses; conformal or native quantile
    intervals. ``float32`` throughout.
    """

    uses_exog = False

    def __init__(self, h, input_size=-1, n_stacks=2, multi_layer=5,
                 dropout_rate=0.5, leaky_rate=0.2, max_steps=1000,
                 learning_rate=1e-3, num_lr_decays=3, windows_batch_size=32,
                 scaler_type="robust", loss="mae", quantile_sort=True,
                 chebyshev_first_term="nf_zero", random_seed=1, alias="StemGNN"):
        if input_size < 1:
            input_size = 3 * h
        if n_stacks != 2:
            raise ValueError("StemGNN currently only supports n_stacks=2.")
        if chebyshev_first_term not in ("nf_zero", "identity"):
            raise ValueError(
                "chebyshev_first_term must be 'nf_zero' (NF parity) or 'identity' "
                f"(paper T0=I); got {chebyshev_first_term!r}."
            )
        self.h = h
        self.input_size = input_size
        self.n_stacks = n_stacks
        self.multi_layer = multi_layer
        self.dropout_rate = dropout_rate
        self.leaky_rate = leaky_rate
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.windows_batch_size = windows_batch_size
        self.scaler_type = scaler_type
        self.loss = loss
        self.quantile_sort = quantile_sort
        self.chebyshev_first_term = chebyshev_first_term
        self.random_seed = random_seed
        self.alias = alias
        self.conformal_params: ConformalIntervals | None = None
        self.model_: StemGNNNet | None = None
        self._context = None
        self._train_y = None

    # ---- helpers -------------------------------------------------------------
    @property
    def _loss_fn(self):
        return _resolve_loss(self.loss)

    @property
    def _scaler(self):
        return resolve_scaler(self.scaler_type)

    def _build_net(self) -> StemGNNNet:
        return StemGNNNet(
            h=self.h, input_size=self.input_size, n_series=1,
            n_stacks=self.n_stacks, multi_layer=self.multi_layer,
            dropout_rate=self.dropout_rate, leaky_rate=self.leaky_rate,
            outputsize_multiplier=outputsize_multiplier(self._loss_fn),
            chebyshev_first_term=self.chebyshev_first_term,
            rngs=nnx.Rngs(self.random_seed),
        )

    # ---- fit -----------------------------------------------------------------
    def fit(self, y, X=None) -> "StemGNN":
        if X is not None:
            raise NotImplementedError("Exogenous variables are not supported.")
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] <= self.input_size:
            # Partial (h-padded) windows need only T >= input_size + 1.
            raise ValueError(
                f"Series length {y.shape[0]} too short for input_size={self.input_size} "
                f"(need at least input_size+1)."
            )
        net = self._build_net()
        train(
            net, y, h=self.h, input_size=self.input_size, max_steps=self.max_steps,
            windows_batch_size=self.windows_batch_size, lr=self.learning_rate,
            num_lr_decays=self.num_lr_decays, seed=self.random_seed,
            loss_fn=self._loss_fn, scaler=self._scaler,
        )
        self.model_ = net
        self._context = y[-self.input_size:]
        self._train_y = y
        return self

    # ---- predict -------------------------------------------------------------
    def predict(self, h, X=None, level=None) -> dict:
        if self.model_ is None or self._context is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        if h < 1:
            raise ValueError(f"h must be a positive integer; got {h}.")
        if h > self.h:
            raise ValueError(
                f"StemGNN was trained for h={self.h}; predict(h={h}) is unsupported. "
                f"Pass h <= {self.h}."
            )
        full = predict_step(
            self.model_, self._context, h=self.h, input_size=self.input_size,
            scaler=self._scaler,
        )                                                     # [h_train, mult]
        if full.shape[-1] == 1:
            fcst = {"mean": full[:h, 0]}
            if level is not None:
                fcst = self._add_conformal(fcst, level)
            return fcst
        return self._format_quantiles(full, h, level)

    def _add_conformal(self, fcst, level):
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
    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False) -> dict:
        """Stateless fit-then-predict. Exogenous inputs are unsupported."""
        if X_future is not None:
            raise NotImplementedError("Exogenous variables are not supported.")
        self.fit(y, X=X)
        result = self.predict(h=h, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
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
            net = self._build_net()           # uses restored loss/flag config
            nnx.update(net, saved)
            self.model_ = net
            return
        self.__dict__.update(state)

"""HINT: hierarchical forecast reconciliation wrapper (port of neuralforecast.HINT).

A hierarchy of series is described by a summing matrix ``S`` of shape
``(n_total, n_bottom)`` whose rows list every node (aggregates first, then the
bottom rows, which must form an identity block): ``y_total = S @ y_bottom``.
HINT trains ONE probabilistic base network on the windows of every series in
the hierarchy (cross-learning with per-window scale decoupling), draws
Monte-Carlo sample paths from each series' predictive mixture, and restores the
aggregation constraints by projecting the sample tensor with ``SP = S @ P`` —
bootstrap sample reconciliation. Coherence is a per-sample (joint) property:
the reconciled mean is exactly coherent, while marginal quantiles of coherent
samples do not sum across the hierarchy in general.

References:
    Olivares et al., "Hierarchically Coherent Multivariate Mixture Networks" —
    https://arxiv.org/abs/2305.07089
    Olivares et al., "Probabilistic Hierarchical Forecasting with Deep Poisson
    Mixtures" — https://arxiv.org/abs/2110.13179
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.mlp.mlp_model import MLP
from chronax.models.mlp.mlp_training import build_windows, predict_params, train_on_windows
from chronax.utils import ConformalIntervals


def get_bottomup_P(S: np.ndarray) -> np.ndarray:
    """BottomUp reconciliation matrix ``P = [0 | I]`` of shape (bottom, total):
    aggregate forecasts are discarded and rebuilt from the bottom rows.

    Reference: Orcutt, Watts & Edwards (1968), "Data aggregation and
    information loss", The American Economic Review 58.
    """
    S = np.asarray(S, dtype=np.float64)
    n_series = len(S)
    n_agg = n_series - S.shape[1]
    P = np.zeros_like(S)
    P[n_agg:, :] = S[n_agg:, :]
    return P.T


def _mintrace_P(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    # MinT equation 10: P = J - (J W U) pinv(U' W U) U' with U' = [I | -A],
    # J = [0 | I]; the zero-constraint form of the trace-minimizing projection.
    n_hiers, n_bottom = S.shape
    n_agg = n_hiers - n_bottom
    A = S[:n_agg, :]
    U = np.hstack((np.eye(n_agg), -A)).T
    J = np.hstack((np.zeros((n_bottom, n_agg)), np.eye(n_bottom)))
    return J - (J @ W @ U) @ np.linalg.pinv(U.T @ W @ U) @ U.T


def get_mintrace_ols_P(S: np.ndarray) -> np.ndarray:
    """MinTrace reconciliation matrix with identity error covariance (OLS).

    Reference: Wickramasuriya, Athanasopoulos & Hyndman (2019), "Optimal
    forecast reconciliation for hierarchical and grouped time series through
    trace minimization", JASA 114.
    """
    S = np.asarray(S, dtype=np.float64)
    return _mintrace_P(S, np.eye(S.shape[0]))


def get_mintrace_wls_P(S: np.ndarray) -> np.ndarray:
    """MinTrace reconciliation matrix with structural weights
    ``W = diag(S @ 1)`` (each node weighted by its bottom-series count).

    Reference: Wickramasuriya, Turlach & Hyndman (2020), "Optimal non-negative
    forecast reconciliation", Statistics and Computing 30.
    """
    S = np.asarray(S, dtype=np.float64)
    return _mintrace_P(S, np.diag(S @ np.ones((S.shape[1],))))


_RECONCILIATIONS = {
    "BottomUp": get_bottomup_P,
    "MinTraceOLS": get_mintrace_ols_P,
    "MinTraceWLS": get_mintrace_wls_P,
    "Identity": None,
}


class HINT(BaseForecaster):
    """HINT: Hierarchical Mixture Network (flax.nnx port of neuralforecast.HINT).

    Wraps a probabilistic base forecaster (an :class:`~chronax.models.mlp.MLP`
    with a distribution loss such as :class:`~chronax.models.mlp.GMM`) into a
    coherent hierarchical forecaster. ``fit`` expects ``y`` of shape
    ``(T, n_total)`` with columns ordered exactly as the rows of ``S``
    (aggregates first, bottom identity block last); a 1-D series is accepted
    when ``S`` is 1x1. One network is trained on the pooled h-padded windows of
    all series (batch sampling is uniform over the pooled set), so the
    hierarchy is cross-learned with per-window scale decoupling. ``predict``
    draws seeded Monte-Carlo sample paths per series, reconciles them with
    ``SP = S @ P``, and emits the reconciled analytic mean plus native
    sample-quantile intervals; output arrays are ``(h,)`` for a 1-D fit and
    ``(h, n_total)`` otherwise. ``reconciliation="Identity"`` skips
    reconciliation entirely.

    Conformal intervals follow the base-class contract on 1-D fits only; a
    hierarchical fit's intervals are the native reconciled-sample quantiles.
    Exogenous inputs are not supported. The passed ``model`` is used purely as
    a configuration carrier and is never fitted or mutated.
    """

    uses_exog = False

    def __init__(self, h: int, S, model, reconciliation: str = "BottomUp",
                 alias: str = "HINT"):
        if getattr(model, "h", None) != h:
            raise ValueError(f"Model h {getattr(model, 'h', None)} does not match HINT h {h}")
        if not isinstance(model, MLP):
            raise ValueError(
                f"HINT requires an MLP base model (got {type(model).__name__}): the "
                "hierarchical trainer drives the mlp package's window/training internals."
            )
        if not getattr(model._loss_fn, "is_distribution_output", False):
            raise ValueError(
                f"The base model's loss {model.loss!r} is not a probabilistic objective"
            )
        if reconciliation not in _RECONCILIATIONS:
            raise ValueError(f"Reconciliation {reconciliation} not available")

        S = np.asarray(S, dtype=np.float64)
        if S.ndim != 2 or S.shape[0] < S.shape[1] or S.shape[1] < 1:
            raise ValueError(f"S must be 2-D (n_total, n_bottom) with n_total >= n_bottom; got {S.shape}.")
        n_total, n_bottom = S.shape
        if not np.array_equal(S[n_total - n_bottom:], np.eye(n_bottom)):
            raise ValueError(
                "The last n_bottom rows of S must be the identity block (bottom series "
                "last, aggregates first) — every reconciliation here assumes that layout."
            )

        self.h = h
        self.S = S
        self.model = model
        self.reconciliation = reconciliation
        self.alias = alias
        p_fn = _RECONCILIATIONS[reconciliation]
        self.SP = None if p_fn is None else S @ p_fn(S=S)
        self.conformal_params: ConformalIntervals | None = None
        self.model_ = None
        self._contexts = None
        self._train_y = None
        self._train_rank = None

    # ---- fit -----------------------------------------------------------------
    def fit(self, y, X=None) -> "HINT":
        if X is not None:
            raise ValueError("HINT does not support exogenous inputs; fit with y only.")
        y = jnp.asarray(y, dtype=jnp.float32)
        n_total = self.S.shape[0]
        if y.ndim == 1:
            if n_total != 1:
                raise ValueError(
                    f"1-D y requires a 1x1 S; this hierarchy has {n_total} rows. "
                    f"Pass y of shape (T, {n_total}) with columns ordered as the rows of S."
                )
            y2 = y[:, None]
            rank = 1
        elif y.ndim == 2:
            if y.shape[1] != n_total:
                raise ValueError(
                    f"y has {y.shape[1]} columns but S has {n_total} rows; the hierarchy "
                    "requires one column per node, ordered as the rows of S."
                )
            y2 = y
            rank = 2
        else:
            raise ValueError(f"y must be 1-D or 2-D; got shape {y.shape}.")

        cfg = self.model
        L = cfg.input_size
        if y2.shape[0] <= L:
            raise ValueError(
                f"Series length {y2.shape[0]} too short for input_size={L} "
                f"(need at least input_size+1)."
            )

        # Pool every series' h-padded windows and cross-learn ONE network.
        # The column count is static config (S's row count), so the loop unrolls
        # cleanly under jit/vmap.
        windows, masks = [], []
        for j in range(n_total):
            w, m = build_windows(y2[:, j], L, cfg.h)
            windows.append(w)
            masks.append(m)
        net = cfg._build_net()
        train_on_windows(
            net, jnp.concatenate(windows), jnp.concatenate(masks),
            h=cfg.h, input_size=L, max_steps=cfg.max_steps,
            windows_batch_size=cfg.windows_batch_size, lr=cfg.learning_rate,
            seed=cfg.random_seed, loss_fn=cfg._loss_fn, scaler=cfg._scaler,
        )
        self.model_ = net
        self._contexts = y2[-L:, :].T          # [n_total, L]
        self._train_y = y
        self._train_rank = rank
        return self

    # ---- predict -------------------------------------------------------------
    def _orient(self, arr: jnp.ndarray) -> jnp.ndarray:
        """[n_total, h] -> (h,) for a 1-D fit, (h, n_total) otherwise."""
        return arr[0] if self._train_rank == 1 else arr.T

    def predict(self, h, X=None, level=None) -> dict:
        if self.model_ is None or self._contexts is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        if X is not None:
            raise ValueError("HINT does not support exogenous inputs.")
        if h < 1:
            raise ValueError(f"h must be a positive integer; got {h}.")
        if h > self.h:
            raise ValueError(
                f"HINT was trained for h={self.h}; predict(h={h}) is unsupported. Pass h <= {self.h}."
            )
        cfg = self.model
        loss = cfg._loss_fn
        distr_args = predict_params(
            self.model_, self._contexts, input_size=cfg.input_size,
            scaler=cfg._scaler, loss_fn=loss,
        )                                                     # arrays [n_total, h_train, K]
        mean = loss.analytic_mean(distr_args)                 # [n_total, h_train]
        sp = None if self.SP is None else jnp.asarray(self.SP, mean.dtype)
        if sp is not None:
            mean = sp @ mean                                  # exact: SP is linear
        fcst = {"mean": self._orient(mean[:, :h])}
        if level is not None:
            samples = loss.sample(distr_args, key=jax.random.PRNGKey(cfg.random_seed))
            if sp is not None:
                samples = jnp.einsum("ij,jhs->ihs", sp, samples)
            samples = samples[:, :h]                          # [n_total, h, S]
            for lv in sorted(level):
                lo_q = (100 - lv) / 200.0
                fcst[f"lo-{lv}"] = self._orient(jnp.quantile(samples, lo_q, axis=-1))
                fcst[f"hi-{lv}"] = self._orient(jnp.quantile(samples, 1.0 - lo_q, axis=-1))
        return fcst

    # ---- forecast ------------------------------------------------------------
    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False) -> dict:
        """Stateless fit-then-predict on the hierarchy matrix ``y``."""
        if X_future is not None:
            raise ValueError("HINT does not support exogenous inputs.")
        if fitted:
            raise NotImplementedError("fitted=True is not supported by HINT.")
        self.fit(y, X=X)
        return self.predict(h=h, level=level)

    # ---- conformal -----------------------------------------------------------
    def conformity_scores(self, y, X=None) -> jnp.ndarray:
        y = jnp.asarray(y)
        if y.ndim != 1:
            raise ValueError(
                "Conformal intervals are supported on 1-D fits only; a hierarchical "
                "fit's intervals are the native reconciled-sample quantiles "
                "(predict(level=...) without conformal_params)."
            )
        # Run the walk-forward on a discarded clone: the CV vmap re-fits per
        # window, and those fit() writes would leave escaped tracers on this
        # estimator. This method is HINT's direct conformal surface (predict
        # emits native intervals), so the clone lives here, not in a caller.
        return BaseForecaster.conformity_scores(self.new(), y, X)

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
            net = self.model._build_net()
            nnx.update(net, saved)
            self.model_ = net
            return
        self.__dict__.update(state)

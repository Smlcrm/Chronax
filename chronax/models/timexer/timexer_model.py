"""TimeXer forecaster: BaseForecaster wrapper around the flax.nnx backbone.

Univariate wrapper (``n_series=1``) over the N-generic patch-transformer
network. Point losses; intervals via the conformal path. Normalization lives
inside the network (non-stationary norm), so windows train in original scale;
optional Box-Cox variance stabilization mirrors the fleet's ``use_boxcox``
convention (off = faithful port). Exogenous inputs are not modeled in this
port — without them the reference's cross-attention context is the endogenous
variate embedding itself. ``float32`` throughout.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.timexer.timexer_losses import resolve as _resolve_loss
from chronax.models.timexer.timexer_module import TimeXerNet
from chronax.models.timexer.timexer_training import predict_step, train
from chronax.utils import ConformalIntervals

# --- Box-Cox variance-stabilizing transform (self-contained) ---------------
# Mirrors the ``use_boxcox`` convention of chronax.models.TBATS: stabilize the
# variance of multiplicative / strongly-trending series before modelling, then
# invert on the output. ``lambda = 0`` is the log transform. The lambda is
# selected once at fit time by maximizing the Box-Cox profile log-likelihood
# (the standard MLE), kept self-contained so this module does not depend on
# another model's internals.

_BOXCOX_LAMBDA_GRID = jnp.linspace(-1.0, 2.0, 31)  # step 0.1, includes 0.0 (log)


def _boxcox(y: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Box-Cox transform of strictly-positive ``y``. ``lam=0`` -> ``log(y)``."""
    y = jnp.asarray(y, dtype=jnp.float32)
    if abs(lam) < 1e-8:
        return jnp.log(y)
    return (jnp.power(y, lam) - 1.0) / lam


def _inv_boxcox(z: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Inverse Box-Cox. Clamps the domain so ``lam != 0`` never yields NaN."""
    z = jnp.asarray(z, dtype=jnp.float32)
    if abs(lam) < 1e-8:
        return jnp.exp(z)
    base = jnp.maximum(lam * z + 1.0, 1e-8)
    return jnp.power(base, 1.0 / lam)


def _select_boxcox_lambda(y: jnp.ndarray) -> float:
    """Pick lambda by maximizing the Box-Cox profile log-likelihood (MLE).

    Runs eagerly (concrete ``y``), returning a plain ``float`` so it pickles
    trivially with the fitted estimator.
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    n = y.size
    log_sum = jnp.sum(jnp.log(y))

    def ll(lam: float) -> jnp.ndarray:
        z = _boxcox(y, lam)
        var = jnp.maximum(jnp.var(z), 1e-12)
        return -0.5 * n * jnp.log(var) + (lam - 1.0) * log_sum

    lls = jnp.array([ll(float(lam)) for lam in _BOXCOX_LAMBDA_GRID])
    return float(_BOXCOX_LAMBDA_GRID[int(jnp.argmax(lls))])


class TimeXer(BaseForecaster):
    """TimeXer: patch transformer with a global-token exogenous pathway
    (flax.nnx port of neuralforecast.TimeXer).

    Wang et al., 2024 — https://arxiv.org/abs/2402.19072. The endogenous
    window is patch-embedded per variate with one learnable GLOBAL token
    appended; encoder layers self-attend over patch tokens while only the
    global token cross-attends to variate-level embeddings of the full window,
    and a per-variate flatten head maps the token stack onto the horizon,
    wrapped in non-stationary normalization. Historical exogenous inputs
    (``fit(y, X=(T, F))``, ``uses_exog = True``) enter that cross-attention
    context as one raw variate token per covariate — the reference's pathway;
    it is historical-only (no future-known covariates) and static exog is not
    modeled. Point losses; conformal intervals (unavailable with historical
    exog — no native interval head). ``float32`` throughout.
    """

    uses_exog = True

    def __init__(self, h, input_size=-1, patch_len=16, hidden_size=512,
                 n_heads=8, e_layers=2, d_ff=2048, dropout=0.1, use_norm=True,
                 use_boxcox=False, loss="mae", max_steps=1000,
                 learning_rate=1e-3, windows_batch_size=32, random_seed=1,
                 alias="TimeXer"):
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.patch_len = patch_len
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_norm = use_norm
        self.use_boxcox = use_boxcox
        self.loss = loss
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.windows_batch_size = windows_batch_size
        self.random_seed = random_seed
        self.alias = alias
        self.conformal_params: ConformalIntervals | None = None
        self.model_: TimeXerNet | None = None
        self._train_y = None
        self._hist_size = 0
        self._hist_ctx = None
        self._bc_lambda: float | None = None
        self._bc_frozen = False
        # Validate structural config eagerly (same checks the net applies).
        self._build_net()

    @property
    def _has_temporal_exog(self) -> bool:
        return self._hist_size > 0

    # ---- helpers -------------------------------------------------------------
    @property
    def _loss_fn(self):
        return _resolve_loss(self.loss)

    def _build_net(self) -> TimeXerNet:
        return TimeXerNet(
            h=self.h, input_size=self.input_size, n_series=1,
            patch_len=self.patch_len, hidden_size=self.hidden_size,
            n_heads=self.n_heads, e_layers=self.e_layers, d_ff=self.d_ff,
            dropout=self.dropout, use_norm=self.use_norm,
            hist_exog_size=self._hist_size,
            outputsize_multiplier=1, rngs=nnx.Rngs(self.random_seed),
        )

    def _to_model_scale(self, y: jnp.ndarray) -> jnp.ndarray:
        return _boxcox(y, self._bc_lambda) if self._bc_lambda is not None else y

    def _from_model_scale(self, z: jnp.ndarray) -> jnp.ndarray:
        return _inv_boxcox(z, self._bc_lambda) if self._bc_lambda is not None else z

    # ---- fit -----------------------------------------------------------------
    def fit(self, y, X=None) -> "TimeXer":
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] <= self.input_size:
            raise ValueError(
                f"Series length {y.shape[0]} too short for input_size={self.input_size} "
                f"(need at least input_size+1)."
            )
        # X = historical exog (T, F): known only over the input span (the
        # reference's TimeXer is historical-only — no future-known pathway). Fed
        # to the network raw (only the target is non-stationary-normalized).
        hist_exog = None if X is None else jnp.asarray(X, jnp.float32)
        if hist_exog is not None and hist_exog.shape[0] != y.shape[0]:
            raise ValueError(f"X (historical exog) must align with y at fit (len {y.shape[0]}); got {hist_exog.shape[0]}.")
        self._hist_size = 0 if hist_exog is None else int(hist_exog.shape[1])
        if self.use_boxcox:
            # Validation and lambda selection are eager-surface operations: a
            # user-facing fit re-selects lambda for its concrete series, while
            # conformal-CV re-fits (traced windows of the already-validated
            # series) reuse the frozen value — the fitted-params-reuse class,
            # which keeps this whole method vmap-traceable.
            if not (self._bc_frozen and self._bc_lambda is not None):
                if bool(jnp.any(y <= 0)):
                    raise ValueError(
                        "use_boxcox=True requires strictly positive values; "
                        "found non-positive entries in y."
                    )
                self._bc_lambda = _select_boxcox_lambda(y)
        else:
            self._bc_lambda = None

        net = self._build_net()
        train(
            net, self._to_model_scale(y), h=self.h, input_size=self.input_size,
            max_steps=self.max_steps, windows_batch_size=self.windows_batch_size,
            lr=self.learning_rate, seed=self.random_seed, loss_fn=self._loss_fn,
            hist_exog=hist_exog,
        )
        self.model_ = net
        # Raw series kept so conformity_scores re-applies Box-Cox through the
        # public forecast API. Covariates kept raw; predict reads the last L rows.
        self._train_y = y
        self._hist_ctx = None if hist_exog is None else hist_exog[-self.input_size:]  # [L, X]
        return self

    # ---- predict -------------------------------------------------------------
    def predict(self, h, X=None, level=None) -> dict:
        if self.model_ is None or self._train_y is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        if h < 1:
            raise ValueError(f"h must be a positive integer; got {h}.")
        if h > self.h:
            raise ValueError(
                f"TimeXer was trained for h={self.h}; predict(h={h}) is unsupported. "
                f"Pass h <= {self.h}."
            )
        full = predict_step(self.model_, self._to_model_scale(self._train_y),
                            h=self.h, input_size=self.input_size,
                            hist_full=self._hist_ctx)             # [h_train, 1]
        mean = self._from_model_scale(full[:h, 0])
        fcst = {"mean": mean}
        if level is not None:
            if self._has_temporal_exog:
                raise ValueError(
                    "Conformal intervals are not supported with historical exog (TimeXer has no "
                    "native interval head); refit without X for conformal intervals, or omit level."
                )
            if self.conformal_params is None:
                raise ValueError(
                    "predict(level=...) requires `model.conformal_params` (a ConformalIntervals). "
                    "conformity_scores re-fits per CV window under vmap -- expect minutes."
                )
            # Run the walk-forward on a clone: conformity_scores re-fits under
            # vmap, and those fit() writes would leave leaked tracers here.
            cs = self.new().conformity_scores(self._train_y)
            fcst = BaseForecaster.add_confidence_intervals(
                fcst, cs, level, self.conformal_params.method)
        return fcst

    def conformity_scores(self, y, X=None):
        """Walk-forward conformity scores; with ``use_boxcox=True`` the CV
        re-fits reuse the lambda frozen by the eager ``fit`` (selection cannot
        run on traced windows)."""
        if X is not None:
            # Historical exog uses no conformal path (TimeXer has no native
            # interval head); guarding here keeps a passed X out of the base CV
            # vmap, where the refusal could not raise under trace.
            raise ValueError(
                "Conformal intervals are not supported with historical exog. "
                "Refit without X for conformal intervals, or omit level."
            )
        if self.use_boxcox:
            if self._bc_lambda is None:
                raise ValueError(
                    "use_boxcox=True requires a fitted estimator before conformal CV: "
                    "fit() freezes the Box-Cox lambda that the CV re-fits reuse."
                )
            self._bc_frozen = True
        try:
            return super().conformity_scores(y, X)
        finally:
            self._bc_frozen = False

    # ---- forecast ------------------------------------------------------------
    def forecast(self, y, h, X=None, X_future=None, level=None, fitted=False) -> dict:
        """Stateless fit-then-predict. ``X`` = historical exog ``(T, F)``; TimeXer
        is historical-only, so ``X_future`` is unsupported."""
        if X_future is not None:
            raise NotImplementedError("TimeXer models historical exog only (no future-known pathway); pass X=, not X_future=.")
        self.fit(y, X=X)
        result = self.predict(h=h, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
        if self._has_temporal_exog:
            raise NotImplementedError("fitted=True is not supported with historical exog.")
        y = self._to_model_scale(self._train_y)
        L = self.input_size
        n = y.shape[0] - L
        idx = jnp.arange(L)[None, :] + jnp.arange(n)[:, None]
        from chronax.models.timexer.timexer_training import _forward_det
        pred = _forward_det(self.model_, y[idx][..., None], None)  # [n, h, 1]
        one_step = self._from_model_scale(pred[:, 0, 0])
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
            net = self._build_net()
            nnx.update(net, saved)
            self.model_ = net
            return
        self.__dict__.update(state)

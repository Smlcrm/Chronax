"""User-facing :class:`XLSTM` forecaster.

mLSTM/sLSTM-based deep recurrent forecaster with the standard Chronax
model contract: extend :class:`BaseForecaster`, expose ``fit``/``predict``
/``forecast``, integrate with the conformal-intervals framework.

The constructor exposes the vanilla mLSTM recipe (the defaults) and an
xLSTMTime-inspired profile (Alharthi & Mahmood 2024) behind opt-in kwargs.
The recurrence runs over the TIME axis of the univariate series. Canonical
xLSTMTime runs its recurrence over the channel axis of a multivariate panel
(a univariate series would degenerate to a single recurrent step); that
channel-axis formulation and its training pipeline are not reproduced here.
Opt-in kwargs:

  - ``block_types``: per-layer ``"mlstm"`` or ``"slstm"``.
  - ``use_revin``: reversible instance normalization (per-series stats).
  - ``use_decomposition``: learnable moving-average trend / seasonal split.
  - ``decode_mode``: ``"ar"`` (autoregressive, AR) or ``"direct"`` (linear D -> H head).

The defaults (``("mlstm",) * num_layers``, RevIN off, decomposition off, AR
decode) give the vanilla mLSTM forecaster.

Conformal note
--------------
Conformal intervals are computed with the trained parameters held fixed
across walk-forward windows; calibration reflects autoregressive (or
direct-head) variance, not retraining variance. In direct mode the
conformal horizon must equal ``cfg.horizon``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

from chronax import utils
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals

from .xlstm_backend import XLSTMConfig
from .xlstm_functions import xlstm_f, forecast_xlstm, _get_decoder


class XLSTM(BaseForecaster):
    """xLSTM / xLSTMTime forecaster.

    A stack of mLSTM/sLSTM blocks trained on sliding windows of a z-scored
    univariate series. Exogenous regressors are not supported
    (``uses_exog = False``); ``X`` is accepted and ignored. Prediction
    intervals are conformal and need ``conformal_params`` at construction:
    the scores are computed at ``fit`` time with the trained parameters held
    fixed across walk-forward windows and cached in ``model_["_cs"]``.

    Parameters
    ----------
    ctx_len : int, default 64
        Sliding-window context length. Must be ``>= 4``. Shrunk automatically
        at ``fit`` when ``T < ctx_len + horizon_train + 1`` (AR mode) or
        ``T < ctx_len + horizon + 1`` (direct mode).
    horizon_train : int, default 8
        Training horizon in AR mode. Must be ``>= 1``.
    num_layers : int, default 2
        Number of blocks. Must be ``>= 1``.
    embed_dim : int, default 64
        Block width ``D``. Must be a positive multiple of ``num_heads``.
    num_heads : int, default 4
        Number of heads per block.
    n_epochs : int, default 100
        Training epochs; the step budget is ``n_epochs * (n_windows //
        batch_size)`` unless ``max_steps`` is given. Must be ``>= 1``.
    batch_size : int, default 32
        Windows per optimizer step. Must be ``>= 1``.
    lr : float, default 1e-3
        Peak learning rate (warmup + cosine schedule).
    weight_decay : float, default 1e-5
        AdamW weight decay.
    gate_clip : float, default 8.0
        Clipping bound on the input/forget gate pre-activations.
    seed : int, default 0
        Pseudo-random number generator (PRNG) seed for parameter initialisation and batch sampling.
    alias : str, default "XLSTM"
        Display name.
    conformal_params : ConformalIntervals, optional
        Conformal configuration. When given, conformity scores are computed
        and cached at ``fit``; ``predict(level=...)`` needs them. In direct
        mode ``conformal_params.h`` must equal ``horizon``.
    block_types : tuple[str, ...], optional
        Per-layer block type, each ``"mlstm"`` or ``"slstm"``; length must
        equal ``num_layers``. ``None`` means ``("mlstm",) * num_layers``.
    use_revin : bool, default False
        Reversible instance normalization inside the model. When True the
        outer z-score in ``fit`` is skipped.
    revin_affine : bool, default True
        Whether RevIN has learnable affine ``gamma``/``beta``.
    use_decomposition : bool, default False
        Learnable moving-average trend + seasonal split, processed through
        shared-weight branches and summed.
    decomp_kernel : int, default 25
        Moving-average kernel size of the decomposition.
    decode_mode : {"ar", "direct"}, default "ar"
        ``"ar"``: autoregressive decode from a per-position next-step head,
        any ``h``. ``"direct"``: linear head ``D -> horizon`` emitting all
        steps at once; every ``h`` passed to ``predict``/``forecast`` and
        ``conformal_params.h`` must equal ``horizon``.
    horizon : int, optional
        Direct-head output length. Required when ``decode_mode == "direct"``.
    use_conv1d_in_slstm : bool, default False
        Apply a causal depthwise Conv1D before each sLSTM recurrence.
    conv1d_kernel : int, default 4
        Causal-Conv1D kernel size.
    slstm_forget_gate : {"exp", "sigmoid"}, default "exp"
        sLSTM forget-gate form. ``"exp"`` uses the raw pre-activation in the
        log-space recurrence; ``"sigmoid"`` uses the log-sigmoid forget gate
        of the xLSTM equations (Eq. 13-15, arXiv 2412.07752). The
        ``max(|n|, 1)`` normalizer floor and ``gate_clip`` apply in both
        forms (see the backend module docstring). sLSTM blocks only.
    slstm_stabilizer : {"per_head", "per_cell"}, default "per_head"
        sLSTM log-space stabilizer granularity: one scalar per head, or one
        per cell as in Eq. 15 of the paper.
    compute_dtype : {"float32", "bfloat16"}, default "float32"
        Forward/backward compute dtype; master parameters stay float32.
        ``"bfloat16"`` is intended for graphics processing units (GPU); XLA (Accelerated Linear
        Algebra) on the central processing unit (CPU) has no fast bf16 path.
    h : int, optional
        Alias for ``horizon`` (neuralforecast convention). Raises if an
        explicit ``horizon`` differs.
    input_size : int, optional
        Alias for ``ctx_len``; takes precedence when given.
    random_seed : int, optional
        Alias for ``seed``; takes precedence when given.
    max_steps : int, optional
        Exact total optimizer step count, overriding the epochs-derived
        budget so training compute is independent of series length.
        Must be ``>= 1``.
    learning_rate : float, optional
        Alias for ``lr``; takes precedence when given.

    Raises
    ------
    ValueError
        If ``h`` conflicts with ``horizon``; if ``max_steps``, ``ctx_len``,
        ``horizon_train``, ``num_layers``, ``n_epochs`` or ``batch_size`` is
        below its minimum; if ``embed_dim`` is not a positive multiple of
        ``num_heads``; if ``block_types`` has the wrong length or an unknown
        entry; if ``decode_mode``, ``slstm_forget_gate``, ``slstm_stabilizer``
        or ``compute_dtype`` is not one of its allowed values; or if
        ``decode_mode == "direct"`` without ``horizon``.

    Attributes
    ----------
    model_ : dict
        Empty before ``fit``. After ``fit``: ``"params"``, ``"cfg"``
        (:class:`XLSTMConfig`), ``"mu"``/``"std"`` (z-score stats, ``0``/``1``
        under RevIN), ``"z_tail"`` (last ``ctx_len`` normalised values),
        ``"losses"`` and, when ``conformal_params`` is set, ``"_cs"``.
    """

    uses_exog = False

    def __init__(
        self,
        ctx_len: int = 64,
        horizon_train: int = 8,
        num_layers: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        n_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        gate_clip: float = 8.0,
        seed: int = 0,
        alias: str = "XLSTM",
        conformal_params: Optional[ConformalIntervals] = None,
        # ── xLSTMTime extensions ──────────────────────────────────────────────
        block_types: Optional[Tuple[str, ...]] = None,
        use_revin: bool = False,
        revin_affine: bool = True,
        use_decomposition: bool = False,
        decomp_kernel: int = 25,
        decode_mode: str = "ar",
        horizon: Optional[int] = None,
        use_conv1d_in_slstm: bool = False,
        conv1d_kernel: int = 4,
        slstm_forget_gate: str = "exp",
        slstm_stabilizer: str = "per_head",
        compute_dtype: str = "float32",
        # ── neuralforecast-convention aliases (see docstring) ────────────────
        h: Optional[int] = None,
        input_size: Optional[int] = None,
        random_seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        if h is not None:
            if horizon is not None and int(horizon) != int(h):
                raise ValueError(
                    f"h ({h}) conflicts with horizon ({horizon}); pass only one."
                )
            horizon = int(h)
        if input_size is not None:
            ctx_len = int(input_size)
        if random_seed is not None:
            seed = int(random_seed)
        if learning_rate is not None:
            lr = float(learning_rate)
        if max_steps is not None and max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        if ctx_len < 4:
            raise ValueError(f"ctx_len must be >= 4, got {ctx_len}")
        if horizon_train < 1:
            raise ValueError(f"horizon_train must be >= 1, got {horizon_train}")
        if embed_dim < num_heads or embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be a positive multiple of num_heads ({num_heads})"
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # Normalise block_types
        if block_types is None:
            block_types = tuple(["mlstm"] * num_layers)
        else:
            block_types = tuple(block_types)
            if len(block_types) != num_layers:
                raise ValueError(
                    f"block_types length ({len(block_types)}) must equal num_layers ({num_layers})"
                )
            for bt in block_types:
                if bt not in ("mlstm", "slstm"):
                    raise ValueError(f"block_types entries must be 'mlstm' or 'slstm', got {bt!r}")

        if decode_mode not in ("ar", "direct"):
            raise ValueError(f"decode_mode must be 'ar' or 'direct', got {decode_mode!r}")
        if decode_mode == "direct" and horizon is None:
            raise ValueError("decode_mode='direct' requires `horizon` to be set.")
        if slstm_forget_gate not in ("exp", "sigmoid"):
            raise ValueError(
                f"slstm_forget_gate must be 'exp' or 'sigmoid', got {slstm_forget_gate!r}"
            )
        if slstm_stabilizer not in ("per_head", "per_cell"):
            raise ValueError(
                f"slstm_stabilizer must be 'per_head' or 'per_cell', got {slstm_stabilizer!r}"
            )
        if compute_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                f"compute_dtype must be 'float32' or 'bfloat16', got {compute_dtype!r}"
            )

        self.ctx_len = int(ctx_len)
        self.horizon_train = int(horizon_train)
        self.num_layers = int(num_layers)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.gate_clip = float(gate_clip)
        self.seed = int(seed)
        self.alias = alias
        self.conformal_params = conformal_params

        self.block_types = block_types
        self.use_revin = bool(use_revin)
        self.revin_affine = bool(revin_affine)
        self.use_decomposition = bool(use_decomposition)
        self.decomp_kernel = int(decomp_kernel)
        self.decode_mode = decode_mode
        self.horizon = None if horizon is None else int(horizon)
        self.use_conv1d_in_slstm = bool(use_conv1d_in_slstm)
        self.conv1d_kernel = int(conv1d_kernel)
        self.slstm_forget_gate = slstm_forget_gate
        self.slstm_stabilizer = slstm_stabilizer
        self.compute_dtype = compute_dtype

        self.model_ = {}

    def _build_cfg(self, n_obs: int) -> XLSTMConfig:
        """Construct an XLSTMConfig, shrinking ``ctx_len`` for short series.

        In direct mode the shrink keeps ``horizon`` fixed (it is the user
        contract). In AR mode it keeps ``horizon_train``.
        """
        if self.decode_mode == "direct":
            seq_needed = self.ctx_len + self.horizon + 1
            if n_obs < seq_needed:
                ctx = max(4, n_obs - self.horizon - 1)
            else:
                ctx = self.ctx_len
        else:
            seq_needed = self.ctx_len + self.horizon_train + 1
            if n_obs < seq_needed:
                ctx = max(4, n_obs - self.horizon_train - 1)
            else:
                ctx = self.ctx_len

        return XLSTMConfig(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            head_dim=self.embed_dim // self.num_heads,
            num_layers=self.num_layers,
            ctx_len=int(ctx),
            horizon_train=self.horizon_train,
            gate_clip=self.gate_clip,
            block_types=self.block_types,
            use_revin=self.use_revin,
            revin_affine=self.revin_affine,
            use_decomposition=self.use_decomposition,
            decomp_kernel=self.decomp_kernel,
            decode_mode=self.decode_mode,
            horizon=int(self.horizon) if self.horizon is not None else self.horizon_train,
            use_conv1d_in_slstm=self.use_conv1d_in_slstm,
            conv1d_kernel=self.conv1d_kernel,
            slstm_forget_gate=self.slstm_forget_gate,
            slstm_stabilizer=self.slstm_stabilizer,
            compute_dtype=self.compute_dtype,
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "XLSTM":
        """Train on a univariate series and cache the decode state.

        Without RevIN the series is z-scored with its full-sample mean and
        standard deviation; with RevIN the raw series is passed through.
        ``ctx_len`` is shrunk (to at least 4) when the series is too short.
        When ``conformal_params`` is set, conformity scores are computed and
        stored in ``model_["_cs"]``.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``. Non-float input is cast to
            float32.
        X : jnp.ndarray or None
            Ignored.

        Returns
        -------
        XLSTM
            Self, with ``model_`` populated.

        Raises
        ------
        ValueError
            In direct mode, if ``conformal_params.h != horizon``; if the
            series is still too short after the context shrink
            (``T <= ctx_len + horizon_train`` in AR mode, ``T < ctx_len +
            horizon`` in direct mode); or, with ``conformal_params`` set, if
            fewer than two conformal windows fit.
        """
        y_f = utils.ensure_float(y)
        cfg = self._build_cfg(int(y_f.shape[0]))

        if cfg.decode_mode == "direct" and self.conformal_params is not None:
            if self.conformal_params.h != cfg.horizon:
                raise ValueError(
                    f"Direct-mode conformal requires conformal_params.h ({self.conformal_params.h}) "
                    f"== cfg.horizon ({cfg.horizon})."
                )

        if cfg.use_revin:
            # RevIN handles per-instance stats inside the model; pass raw y.
            mu = jnp.asarray(0.0, dtype=jnp.float32)
            std = jnp.asarray(1.0, dtype=jnp.float32)
            z = y_f.astype(jnp.float32)
        else:
            mu = jnp.mean(y_f)
            std = jnp.std(y_f) + 1e-8
            z = (y_f - mu) / std

        key = random.PRNGKey(self.seed)
        result = xlstm_f(
            z, cfg, key,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_steps=self.max_steps,
        )
        self.model_ = {
            "params": result["params"],
            "cfg": cfg,
            "mu": mu,
            "std": std,
            "z_tail": z[-cfg.ctx_len:],
            "losses": result["losses"],
        }
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y_f, X=X)
            self.model_["_cs"] = cs
        return self

    def conformity_scores(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> jnp.ndarray:
        """Walk-forward conformity scores.

        Fitted instances use the trained parameters for every window: each
        window's true context tail ``y[train_end - ctx_len : train_end]``
        (left-padded with ``y[0]`` when ``train_end < ctx_len``) is decoded
        ``h`` steps with the cached decoder, vmapped over the window axis.
        No value at index ``>= train_end`` enters a window's forecast, but
        the parameters and the (non-RevIN) ``mu``/``std`` are full-series
        quantities, so calibration is mildly optimistic relative to a
        per-window re-fit. Unfitted instances fall back to the base
        implementation, which re-fits per window under ``vmap``.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``.
        X : jnp.ndarray or None
            Ignored.

        Returns
        -------
        jnp.ndarray
            Residuals ``y_test - forecast`` of shape ``(n_windows, h)`` with
            ``n_windows = min(conformal_params.n_windows, (T - 1) // h)``.

        Raises
        ------
        ValueError
            If ``conformal_params`` is ``None``; in direct mode if
            ``conformal_params.h != horizon``; or if fewer than two windows
            fit (``T < 2 * h + 1``).
        """
        if not self.model_:
            return super().conformity_scores(y, X=X)
        if self.conformal_params is None:
            raise ValueError(
                "The instance attribute conformal_params must be initialized as a conformal_intervals object."
            )
        cfg: XLSTMConfig = self.model_["cfg"]
        h = self.conformal_params.h
        if cfg.decode_mode == "direct" and h != cfg.horizon:
            raise ValueError(
                f"Direct-mode conformal requires conformal_params.h ({h}) == cfg.horizon ({cfg.horizon})."
            )
        y_f = utils.ensure_float(y)
        n = int(y_f.shape[0])
        n_windows = min(self.conformal_params.n_windows, (n - 1) // h)
        if n_windows < 2:
            raise ValueError(
                f"Conformal prediction requires at least {2 * h + 1:,} samples per window; series has {n:,}."
            )
        ctx = cfg.ctx_len
        base_train_end = n - n_windows * h

        tails, tests = [], []
        for i in range(n_windows):  # static Python ints; vmap/jit-safe
            train_end = base_train_end + i * h
            start = max(0, train_end - ctx)
            tail = y_f[start:train_end]
            pad = ctx - (train_end - start)
            if pad > 0:  # static per-window amount (short-series edge)
                tail = jnp.concatenate([jnp.repeat(y_f[:1], pad), tail])
            tails.append(tail)
            tests.append(y_f[train_end:train_end + h])
        tails_arr = jnp.stack(tails)   # (n_windows, ctx)
        tests_arr = jnp.stack(tests)   # (n_windows, h)

        mu, std = self.model_["mu"], self.model_["std"]
        dec = _get_decoder(cfg, int(h))
        params = self.model_["params"]

        def one_window(tail):
            if cfg.use_revin:
                return dec(params, tail.astype(jnp.float32))  # decoder normalizes internally
            z_tail = (tail - mu) / std
            return dec(params, z_tail) * std + mu

        fcst = jax.vmap(one_window)(tails_arr)  # (n_windows, h)
        return tests_arr - fcst.astype("float32")

    def _decode_from_y(self, y: jnp.ndarray, h: int) -> jnp.ndarray:
        """Take the last ctx_len of y, decode h steps, denormalise if needed."""
        cfg: XLSTMConfig = self.model_["cfg"]
        mu = self.model_["mu"]
        std = self.model_["std"]
        if cfg.use_revin:
            z = y.astype(jnp.float32)
        else:
            z = (y - mu) / std
        z_tail = z[-cfg.ctx_len:]
        dec = _get_decoder(cfg, int(h))
        preds = dec(self.model_["params"], z_tail)
        if cfg.use_revin:
            return preds  # already in original scale
        return preds * std + mu

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
    ) -> dict:
        """Forecast ``h`` steps from the cached context tail.

        Parameters
        ----------
        h : int
            Forecast horizon. In direct mode it must equal ``horizon``.
        X : jnp.ndarray or None
            Ignored.
        level : list[int | float] or None
            Confidence levels in (0, 100), e.g. ``[80, 95]``. Uses the
            conformity scores cached at ``fit``.

        Returns
        -------
        dict
            ``"mean"`` of shape ``(h,)``, plus ``"lo-{lv}"`` and ``"hi-{lv}"``
            of shape ``(h,)`` per level when ``level`` is given.

        Raises
        ------
        ValueError
            If called before ``fit``; in direct mode if ``h != horizon``; if
            ``level`` is given while ``conformal_params`` is ``None`` or no
            scores were cached at ``fit``.
        """
        if not self.model_:
            raise ValueError("Model must be fitted before predict().")
        cfg: XLSTMConfig = self.model_["cfg"]
        if cfg.decode_mode == "direct" and int(h) != cfg.horizon:
            raise ValueError(
                f"Direct-mode predict requires h == cfg.horizon ({cfg.horizon}); got {h}."
            )
        normed = forecast_xlstm(self.model_, int(h))
        if cfg.use_revin:
            mean = normed  # already denormalised inside the model
        else:
            mean = normed * self.model_["std"] + self.model_["mu"]
        result = {"mean": mean}
        if level is not None:
            if self.conformal_params is None:
                raise ValueError(
                    "level requested but conformal_params is None; pass conformal_params at construction."
                )
            cs = self.model_.get("_cs")
            if cs is None:
                raise ValueError("Conformity scores not cached. Refit with conformal_params set.")
            result = self.add_confidence_intervals(result, cs, level, self.conformal_params.method)
        return result

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
    ) -> dict:
        """h-step forecast from ``y``.

        When ``model_`` is populated, the trained parameters are reused and
        the forecast is decoded from the last ``ctx_len`` values of ``y``;
        intervals then use the conformity scores cached at ``fit``. When
        unfitted, a clone is fitted on ``y`` and its ``predict`` is returned.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``.
        h : int
            Forecast horizon. In direct mode it must equal ``horizon``.
        X : jnp.ndarray or None
            Ignored.
        X_future : jnp.ndarray or None
            Ignored.
        level : list[int | float] or None
            Confidence levels in (0, 100). Requires ``conformal_params``.
        fitted : bool
            Must be ``False``; in-sample fitted values are not available.

        Returns
        -------
        dict
            ``"mean"`` of shape ``(h,)``, plus ``"lo-{lv}"`` and ``"hi-{lv}"``
            of shape ``(h,)`` per level when ``level`` is given.

        Raises
        ------
        NotImplementedError
            If ``fitted=True``.
        ValueError
            In direct mode if ``h != horizon``; if ``level`` is given while
            ``conformal_params`` is ``None`` or no scores are cached; on the
            unfitted path, anything ``fit`` raises.
        """
        y_f = utils.ensure_float(y)
        if fitted:
            raise NotImplementedError(
                "fitted=True (in-sample fitted values) is not supported by XLSTM. "
                "Call predict()/forecast() without `fitted`."
            )
        if self.model_:
            cfg: XLSTMConfig = self.model_["cfg"]
            if cfg.decode_mode == "direct" and int(h) != cfg.horizon:
                raise ValueError(
                    f"Direct-mode forecast requires h == cfg.horizon ({cfg.horizon}); got {h}."
                )
            mean = self._decode_from_y(y_f, int(h))
            result = {"mean": mean}
            if level is not None:
                if self.conformal_params is None:
                    raise ValueError(
                        "level requested but conformal_params is None; pass conformal_params at construction."
                    )
                cs = self.model_.get("_cs")
                if cs is None:
                    raise ValueError(
                        "Conformity scores not cached. Refit with conformal_params set."
                    )
                result = self.add_confidence_intervals(
                    result, cs, level, self.conformal_params.method
                )
            return result
        tmp = self.new()
        tmp.model_ = {}
        tmp.fit(y_f)
        return tmp.predict(h=int(h), level=level)

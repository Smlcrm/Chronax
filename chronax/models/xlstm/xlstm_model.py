"""User-facing :class:`XLSTM` forecaster.

mLSTM/sLSTM-based deep recurrent forecaster with the standard Chronax
model contract: extend :class:`BaseForecaster`, expose ``fit``/``predict``
/``forecast``, integrate with the conformal-intervals framework.

The constructor exposes both the vanilla mLSTM recipe (current defaults)
and the full xLSTMTime framework (Alharthi & Mahmood 2024) behind opt-in
kwargs:

  - ``block_types`` — per-layer ``"mlstm"`` or ``"slstm"``.
  - ``use_revin`` — reversible instance normalization (per-series stats).
  - ``use_decomposition`` — learnable moving-average trend / seasonal split.
  - ``decode_mode`` — ``"ar"`` (autoregressive) or ``"direct"`` (linear D→H head).

Defaults reproduce the original vanilla XLSTM behavior (``("mlstm",)*L``,
RevIN off, decomposition off, AR decode), so existing tests pass unchanged.

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

    Parameters
    ----------
    ctx_len, horizon_train : int
        Sliding-window context length and training horizon (AR mode).
    num_layers, embed_dim, num_heads : int
        Stack depth and per-block width. ``embed_dim`` must be divisible
        by ``num_heads``.
    n_epochs, batch_size, lr, weight_decay : training hyperparameters.
    gate_clip : float
        Pre-gate clipping bound on input/forget gate logits.
    seed : int
        PRNG seed for parameter initialisation and batch sampling.
    alias : str
        Identifier used by :meth:`__repr__`.
    conformal_params : ConformalIntervals, optional
        If provided, conformity scores are computed and cached at fit time.

    xLSTMTime extensions
    --------------------
    block_types : tuple[str, ...], optional
        Per-layer block type; ``None`` (default) -> ``("mlstm",) * num_layers``.
        Each entry must be ``"mlstm"`` or ``"slstm"``.
    use_revin : bool
        Reversible instance normalization. When True, the outer z-score in
        ``fit`` is skipped (RevIN handles per-instance stats internally).
    revin_affine : bool
        Whether RevIN has learnable affine ``gamma``/``beta``.
    use_decomposition : bool
        Learnable moving-average trend + seasonal split, processed through
        shared-weight branches and summed.
    decomp_kernel : int
        Moving-average kernel size. Paper default 25.
    decode_mode : {"ar", "direct"}
        ``"ar"`` -> autoregressive decode (per-position next-step head).
        ``"direct"`` -> direct linear head ``D -> horizon`` (xLSTMTime canonical).
    horizon : int, optional
        Required when ``decode_mode == "direct"``. Caller's forecast ``h`` must
        equal this value.
    use_conv1d_in_slstm : bool
        Apply causal Conv1D before sLSTM recurrence.
    conv1d_kernel : int
        Causal-Conv1D kernel size.
    slstm_forget_gate : {"exp", "sigmoid"}
        sLSTM forget-gate form. ``"exp"`` (default) preserves the original
        behavior; ``"sigmoid"`` uses the log-sigmoid forget gate of the xLSTM
        paper (Eq. 13-15, arXiv 2412.07752). sLSTM blocks only.
    slstm_stabilizer : {"per_head", "per_cell"}
        sLSTM log-space stabilizer granularity. ``"per_head"`` (default) keeps the
        original collapsed stabilizer; ``"per_cell"`` matches the paper (Eq. 15).
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
    ) -> None:
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

        self.ctx_len = int(ctx_len)
        self.horizon_train = int(horizon_train)
        self.num_layers = int(num_layers)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
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
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "XLSTM":
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
        """h-step forecast.

        Fast path: when ``self.model_`` is populated (e.g. inside conformal
        walk-forward), reuses trained params + decodes from the last
        ``ctx_len`` of ``y``. Slow path (no prior fit): fits a temporary
        model on ``y`` then predicts.
        """
        y_f = utils.ensure_float(y)
        if self.model_:
            cfg: XLSTMConfig = self.model_["cfg"]
            if cfg.decode_mode == "direct" and int(h) != cfg.horizon:
                raise ValueError(
                    f"Direct-mode forecast requires h == cfg.horizon ({cfg.horizon}); got {h}."
                )
            mean = self._decode_from_y(y_f, int(h))
            return {"mean": mean}
        tmp = self.new()
        tmp.model_ = {}
        tmp.fit(y_f)
        return tmp.predict(h=int(h), level=level)

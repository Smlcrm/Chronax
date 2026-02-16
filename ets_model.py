# ets_model.py
from __future__ import annotations
from typing import Dict, List, Optional
import os

import jax.numpy as jnp

from conformal_intervals import ConformalIntervals
from utils import ensure_float, _add_fitted_pi, calculate_sigma
from base_forecaster import BaseForecaster
from ets_functions import ets_f, forecast_ets, forward_ets

_PHI_LOWER = 0.8
_PHI_UPPER = 0.98


def _init_jax_compilation_cache() -> None:
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.join(os.path.dirname(__file__), ".jax_cache")
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir


_init_jax_compilation_cache()


class ETS(BaseForecaster):
    r"""
    Fixed-spec ETS (Error, Trend, Seasonality).
    """

    def __init__(
        self,
        season_length: int = 1,
        model: str = "ANN",
        damped: Optional[bool] = None,
        phi: Optional[float] = None,
        max_iter: Optional[int] = None,
        optax_lr: float = 1e-2,
        optax_clip: float = 1.0,
        alias: str = "ETS",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        self.season_length = season_length
        self.model = model
        if damped is None:
            damped = False
        self.damped = damped
        if phi is not None:
            if not isinstance(phi, float):
                raise ValueError("phi must be `None` or float.")
            if not (_PHI_LOWER <= phi <= _PHI_UPPER):
                raise ValueError(f"Valid range for phi is [{_PHI_LOWER}, {_PHI_UPPER}]")
        self.phi = phi
        self.max_iter = max_iter
        self.optax_lr = optax_lr
        self.optax_clip = optax_clip
        self.alias = alias
        self.conformal_params = prediction_intervals
        self.optax_steps = max_iter

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "ETS":
        y = ensure_float(y)
        self.model_ = ets_f(
            y,
            m=self.season_length,
            model=self.model,
            damped=self.damped,
            phi=self.phi,
            optax_steps=self.max_iter,
            optax_lr=self.optax_lr,
            optax_clip=self.optax_clip,
        )
        self.model_["actual_residuals"] = y - self.model_["fitted"]

        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None
        return self

    def predict(
        self, h: int, X: Optional[jnp.ndarray] = None, level: Optional[List[int]] = None
    ) -> Dict[str, jnp.ndarray]:
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        fcst = forecast_ets(self.model_, h=h, level=level)
        out = {"mean": fcst["mean"]}

        if level is None:
            return out

        level_sorted = sorted(level)
        if self.conformal_params is not None:
            if self._cs is None:
                raise ValueError(
                    "Conformity scores not cached. Fit with conformal_params set, "
                    "or use forecast(y, ...) which recomputes them."
                )
            return self.add_confidence_intervals(
                fcst=out, cs=self._cs, level=level_sorted, method=self.conformal_params.method
            )

        out.update({f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level_sorted)})
        out.update({f"hi-{l}": fcst[f"hi-{l}"] for l in level_sorted})
        return out

    def predict_in_sample(self, level: Optional[List[int]] = None) -> Dict[str, jnp.ndarray]:
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            residuals = self.model_["actual_residuals"]
            # se = _calculate_sigma(residuals, len(residuals) - self.model_["n_params"])
            se = calculate_sigma(residuals, len(residuals) - self.model_["n_params"])

            res = _add_fitted_pi(res=res, se=se, level=level)
        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        y = ensure_float(y)
        mod = ets_f(
            y,
            m=self.season_length,
            model=self.model,
            damped=self.damped,
            phi=self.phi,
            optax_steps=self.optax_steps,
            optax_lr=self.optax_lr,
            optax_clip=self.optax_clip,
        )
        fcst = forecast_ets(mod, h=h, level=level)

        keys = ["mean"]
        if fitted:
            keys.append("fitted")
        out = {k: fcst[k] for k in keys}

        if level is None:
            return out

        level_sorted = sorted(level)
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=X)
            out = self.add_confidence_intervals(
                fcst=out, cs=cs, level=level_sorted, method=self.conformal_params.method
            )
        else:
            out = {
                **out,
                **{f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level_sorted)},
                **{f"hi-{l}": fcst[f"hi-{l}"] for l in level_sorted},
            }

        if fitted:
            # se = _calculate_sigma(y - mod["fitted"], len(y) - mod["n_params"])
            se = calculate_sigma(y - mod["fitted"], len(y) - mod["n_params"])
            out = _add_fitted_pi(res=out, se=se, level=level_sorted)
        return out

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        y = ensure_float(y)
        mod = forward_ets(self.model_, y=y)
        fcst = forecast_ets(mod, h=h, level=level)

        keys = ["mean"]
        if fitted:
            keys.append("fitted")
        out = {k: fcst[k] for k in keys}

        if level is None:
            return out

        level_sorted = sorted(level)
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=X)
            out = self.add_confidence_intervals(
                fcst=out, cs=cs, level=level_sorted, method=self.conformal_params.method
            )
        else:
            out.update({f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level_sorted)})
            out.update({f"hi-{l}": fcst[f"hi-{l}"] for l in level_sorted})

            if fitted:
                # se = _calculate_sigma(y - mod["fitted"], len(y) - int(mod["n_params"]))
                se = calculate_sigma(y - mod["fitted"], len(y) - int(mod["n_params"]))
                out = _add_fitted_pi(res=out, se=se, level=level_sorted)

        return out
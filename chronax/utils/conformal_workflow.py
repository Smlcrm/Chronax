"""Model-bound conformal prediction workflow (config resolution, scores, intervals).

Mirrors the former ``chronax.utils.utils`` Section 4 helpers with a functional API
and the same JAX tracer fallback path for ``conformity_scores``.
"""

from typing import List, Optional, Union

import jax
import jax.numpy as jnp

from chronax.utils.conformal_methods import get_conformal_method


def resolve_conformal_params(model: object):
    """Resolve and synchronize conformal config aliases on a model instance.

    Args:
        model: A forecaster instance that may expose ``conformal_params`` and/or
            ``prediction_intervals``.

    Returns:
        The effective conformal configuration object, or ``None`` if neither is set.
    """
    conformal_params = getattr(model, "conformal_params", None)
    prediction_intervals = getattr(model, "prediction_intervals", None)

    if conformal_params is None and prediction_intervals is None:
        return None

    effective = conformal_params if conformal_params is not None else prediction_intervals
    if conformal_params is not None and prediction_intervals is not None:
        # Keep BaseForecaster parity by preferring conformal_params when both exist.
        effective = conformal_params

    setattr(model, "conformal_params", effective)
    setattr(model, "prediction_intervals", effective)
    return effective


def compute_conformity_scores(
    model: object, y: jnp.ndarray, X: Optional[jnp.ndarray]
) -> jnp.ndarray:
    """Compute conformity scores with a safe fallback for non-vmap-compatible models.

    Args:
        model: Forecaster providing ``conformity_scores`` and, in the fallback path,
            ``forecast`` and conformal configuration.
        y: Training time series.
        X: Optional exogenous variables aligned with ``y``.

    Returns:
        Conformity score array of shape ``(n_windows, h)`` (walk-forward windows
        by horizon), matching the contract expected by conformal interval methods.
    """
    try:
        return model.conformity_scores(y, X)
    except (jax.errors.TracerBoolConversionError, jax.errors.ConcretizationTypeError):
        conformal_cfg = resolve_conformal_params(model)
        if conformal_cfg is None:
            raise

        from chronax.utils.utils import ensure_float as _ensure_float

        n_windows = conformal_cfg.n_windows
        h = conformal_cfg.h
        y = _ensure_float(y)
        n_samples = y.size
        n_windows = min(n_windows, (n_samples - 1) // h)
        if n_windows < 2:
            raise ValueError(
                f"Conformal prediction requires at least {2 * h + 1:,} samples per window; series has {n_samples:,}."
            )

        test_size = n_windows * h
        base_train_end = n_samples - test_size
        window_scores = []

        for i_window in range(int(n_windows)):
            train_end = int(base_train_end + i_window * h)
            y_train = y[:train_end]
            y_test = y[train_end : train_end + h]
            if X is not None:
                X_train = X[:train_end]
                X_test = X[train_end : train_end + h]
            else:
                X_train = None
                X_test = None

            fcst_window = model.forecast(h=h, y=y_train, X=X_train, X_future=X_test)
            window_scores.append(y_test - fcst_window["mean"].astype("float32"))

        return jnp.stack(window_scores, axis=0)


def add_confidence_intervals(
    fcst: dict,
    cs: jnp.ndarray,
    level: Union[List[float], List[int]],
    method: str,
) -> dict:
    """Canonical conformal interval dispatcher used across the codebase.

    Args:
        fcst: Forecast dict (must include ``mean``).
        cs: Conformity scores of shape ``(W, h)``.
        level: Confidence levels in ``[0, 100]``.
        method: Conformal method name accepted by :func:`get_conformal_method`.

    Returns:
        ``fcst`` augmented with ``lo-{lv}`` / ``hi-{lv}`` keys per level.
    """
    conformal_fn = get_conformal_method(method)
    return conformal_fn(fcst=fcst, cs=cs, level=level)


def add_conformal_intervals(
    model: object,
    fcst: dict,
    y: Optional[jnp.ndarray],
    X: Optional[jnp.ndarray],
    level: Optional[List[int]],
) -> dict:
    """Add conformal prediction intervals to a forecast dict.

    If ``y`` is provided, computes fresh conformal scores; otherwise uses stored scores.

    Args:
        model: A forecaster instance.
        fcst: Forecast dict to augment.
        y: Training series (``None`` to use stored scores on ``model._cs``).
        X: Optional exogenous variables.
        level: Confidence levels (0-100).

    Returns:
        Updated forecast dict with interval keys, or ``fcst`` unchanged when no
        conformal configuration or ``level`` is set.
    """
    conformal_cfg = resolve_conformal_params(model)
    if conformal_cfg is not None and level is not None:
        if y is not None:
            cs = compute_conformity_scores(model, y, X)
        else:
            cs = getattr(model, "_cs", None)
            if cs is None:
                raise ValueError("Conformity scores are missing. Run fit first or provide y to recompute them.")
        return add_confidence_intervals(fcst=fcst, cs=cs, level=level, method=conformal_cfg.method)
    return fcst


def add_predict_conformal_intervals(
    model: object,
    fcst: dict,
    level: Optional[List[int]],
) -> dict:
    """Add conformal intervals for the predict() path (uses stored scores).

    Args:
        model: A fitted forecaster instance.
        fcst: Forecast dict to augment.
        level: Confidence levels (0-100).

    Returns:
        Updated forecast dict with interval keys.
    """
    return add_conformal_intervals(model, fcst=fcst, y=None, X=None, level=level)


def store_conformity_scores(
    model: object, y: jnp.ndarray, X: Optional[jnp.ndarray]
) -> None:
    """Compute and store conformal scores on the model instance.

    Args:
        model: A forecaster instance with conformal configuration and ``conformity_scores``.
        y: Training time series.
        X: Optional exogenous variables.
    """
    if resolve_conformal_params(model) is not None:
        model._cs = compute_conformity_scores(model, y, X)

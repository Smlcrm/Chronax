"""Auto-discovering model registry for the neural harness.

A model is benchmarkable iff it is a ``chronax.models`` ``BaseForecaster`` whose
``__init__`` accepts ``h``, ``input_size``, and ``random_seed`` — the neural
windowed-model signature — and is not in ``_EXCLUDE``. That filter naturally
selects the neural ports and excludes statistical models (no ``input_size``) and
non-``BaseForecaster`` classes.

Every such model is a JAX port of the same-named ``neuralforecast`` class, so the
NF name is the identity. Nothing else is hardcoded: dropping a faithful port into
``chronax/models/__init__.py`` makes it appear in the benchmark automatically.
"""
from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chronax.models.base_forecaster import BaseForecaster

_CORE_ARGS = {"h", "input_size", "random_seed"}

# Explicit paired-run exclusions (owner decision, 2026-07-06): Chronax ``XLSTM``
# is an xLSTMTime-inspired time-axis adaptation, NOT architecture-1:1 with NF's
# ``xLSTM`` — the paired accept-gate comparison is not meaningful. Its ctor gained
# ``h``/``input_size``/``random_seed`` NF-alias params (XLSTM re-enablement), so
# the signature filter alone no longer excludes it; the exclusion is now explicit.
_EXCLUDE = {"XLSTM"}


def _discover() -> dict[str, type]:
    """Map benchmarkable model name -> Chronax class, from chronax.models."""
    from chronax.models.base_forecaster import BaseForecaster

    models = importlib.import_module("chronax.models")
    names = getattr(models, "__all__", None) or dir(models)
    out: dict[str, type] = {}
    for name in names:
        if name.startswith("_") or name in _EXCLUDE:
            continue
        try:
            obj = getattr(models, name)  # may trigger a lazy import
        except Exception:
            continue  # lazy import failed (e.g. flax version guard) -> not available
        if not (inspect.isclass(obj) and issubclass(obj, BaseForecaster)):
            continue
        try:
            params = set(inspect.signature(obj.__init__).parameters)
        except (ValueError, TypeError):
            continue
        if _CORE_ARGS <= params:
            out[name] = obj
    return out


def list_models() -> list[str]:
    """Benchmarkable Chronax neural models, auto-discovered from chronax.models."""
    return sorted(_discover())


def resolve_chronax(name: str) -> "type[BaseForecaster]":
    """Return the Chronax forecaster class for `name`."""
    models = _discover()
    if name not in models:
        raise KeyError(f"{name!r} is not a benchmarkable Chronax model; known: {sorted(models)}")
    return models[name]


def nf_model_name(name: str) -> str:
    """The neuralforecast.models class name for `name` (identity — Chronax ports NF)."""
    models = _discover()
    if name not in models:
        raise KeyError(f"{name!r} is not a benchmarkable Chronax model; known: {sorted(models)}")
    return name

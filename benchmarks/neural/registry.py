"""Static model registry for the neural harness.

name -> {NF model-name string, lazy Chronax class}. Chronax classes are imported
lazily (mirrors the statistical ModelRegistry) so importing this module never
pulls in JAX. The NF side needs only the model-name string; the neuralforecast
import happens inside the .venv-nf subprocess, not in the main env.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from chronax.models.base_forecaster import BaseForecaster

# NF neuralforecast.models class names happen to equal the Chronax class names.
_NF_NAMES = {"GRU": "GRU", "PatchTST": "PatchTST", "KAN": "KAN"}


def _make_factory(model_name: str) -> Callable[[], type[BaseForecaster]]:
    def _factory() -> type[BaseForecaster]:
        return getattr(importlib.import_module("chronax.models"), model_name)
    return _factory


# Dispatch dict derived from _NF_NAMES: _NF_NAMES is the single source of truth.
# Adding a fourth model requires only one entry above.
_CHRONAX_FACTORIES: dict[str, Callable[[], type[BaseForecaster]]] = {
    name: _make_factory(name) for name in _NF_NAMES
}


def list_models() -> list[str]:
    """Registered model names."""
    return list(_NF_NAMES)


def nf_model_name(name: str) -> str:
    """The neuralforecast.models class name for `name`."""
    if name not in _NF_NAMES:
        raise KeyError(f"unknown model {name!r}; known: {sorted(_NF_NAMES)}")
    return _NF_NAMES[name]


def resolve_chronax(name: str) -> type[BaseForecaster]:
    """Lazily import and return the Chronax forecaster class for `name`."""
    if name not in _CHRONAX_FACTORIES:
        raise KeyError(f"unknown model {name!r}; known: {sorted(_NF_NAMES)}")
    return _CHRONAX_FACTORIES[name]()

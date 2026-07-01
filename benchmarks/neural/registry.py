"""Static model registry for the neural harness.

name -> {NF model-name string, lazy Chronax class}. Chronax classes are imported
lazily (mirrors the statistical ModelRegistry) so importing this module never
pulls in JAX. The NF side needs only the model-name string; the neuralforecast
import happens inside the .venv-nf subprocess, not in the main env.
"""
from __future__ import annotations

# NF neuralforecast.models class names happen to equal the Chronax class names.
_NF_NAMES = {"GRU": "GRU", "PatchTST": "PatchTST", "KAN": "KAN"}


def list_models() -> list[str]:
    """Registered model names."""
    return list(_NF_NAMES)


def nf_model_name(name: str) -> str:
    """The neuralforecast.models class name for `name`."""
    if name not in _NF_NAMES:
        raise KeyError(f"unknown model {name!r}; known: {sorted(_NF_NAMES)}")
    return _NF_NAMES[name]


def resolve_chronax(name: str):
    """Lazily import and return the Chronax forecaster class for `name`."""
    if name == "GRU":
        from chronax.models import GRU
        return GRU
    if name == "PatchTST":
        from chronax.models import PatchTST
        return PatchTST
    if name == "KAN":
        from chronax.models import KAN
        return KAN
    raise KeyError(f"unknown model {name!r}; known: {sorted(_NF_NAMES)}")

"""Offline unit tests for benchmarks/neural/registry.py auto-discovery."""
import pytest

from benchmarks.neural import registry
from chronax.models import Autoformer, FEDformer, GRU, KAN, PatchTST, TFT, iTransformer

_WIRED = {"GRU", "PatchTST", "KAN", "TFT", "iTransformer", "Autoformer", "FEDformer"}


def test_discovers_wired_neural_models():
    assert _WIRED <= set(registry.list_models())


def test_excludes_non_benchmarkable():
    models = set(registry.list_models())
    # statistical models (no input_size), XLSTM (ctx_len/horizon_train convention)
    # must not be auto-discovered.
    for name in ["ARIMA", "AutoETS", "Naive", "XLSTM"]:
        assert name not in models


@pytest.mark.parametrize(
    "name,cls",
    [
        ("GRU", GRU),
        ("PatchTST", PatchTST),
        ("KAN", KAN),
        ("TFT", TFT),
        ("iTransformer", iTransformer),
        ("Autoformer", Autoformer),
        ("FEDformer", FEDformer),
    ],
)
def test_resolve_chronax(name, cls):
    assert registry.resolve_chronax(name) is cls


@pytest.mark.parametrize("name", sorted(_WIRED))
def test_nf_model_name_identity(name):
    assert registry.nf_model_name(name) == name


def test_unknown_model_raises_keyerror():
    with pytest.raises(KeyError):
        registry.resolve_chronax("NoSuchModel")
    with pytest.raises(KeyError):
        registry.nf_model_name("NoSuchModel")

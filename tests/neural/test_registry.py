"""Offline unit tests for benchmarks/neural/registry.py."""
import pytest

from benchmarks.neural import registry
from chronax.models import GRU, KAN, PatchTST, TFT, iTransformer


def test_list_models():
    assert sorted(registry.list_models()) == ["GRU", "KAN", "PatchTST", "TFT", "iTransformer"]


@pytest.mark.parametrize("name,cls", [("GRU", GRU), ("PatchTST", PatchTST), ("KAN", KAN),
                                      ("TFT", TFT), ("iTransformer", iTransformer)])
def test_resolve_chronax(name, cls):
    assert registry.resolve_chronax(name) is cls


@pytest.mark.parametrize("name", ["GRU", "PatchTST", "KAN", "TFT", "iTransformer"])
def test_nf_model_name_identity(name):
    assert registry.nf_model_name(name) == name


def test_unknown_model_raises_keyerror():
    with pytest.raises(KeyError, match="unknown model"):
        registry.resolve_chronax("DeepAR")
    with pytest.raises(KeyError, match="unknown model"):
        registry.nf_model_name("DeepAR")

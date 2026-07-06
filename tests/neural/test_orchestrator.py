"""Offline unit tests for orchestrator glue in benchmarks/neural/run.py."""
import pytest

from benchmarks.neural.run import NeuralBenchError, check_nf_venv


def test_check_nf_venv_ok_when_chronax_only(tmp_path):
    # nixtla not requested -> no venv needed, returns None
    assert check_nf_venv(["chronax"], tmp_path / "missing" / "python") is None


def test_check_nf_venv_fails_fast_when_missing(tmp_path):
    with pytest.raises(NeuralBenchError, match="setup_nf_venv.sh"):
        check_nf_venv(["chronax", "nixtla"], tmp_path / "missing" / "python")


def test_check_nf_venv_ok_when_present(tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n")
    assert check_nf_venv(["nixtla"], py) is None

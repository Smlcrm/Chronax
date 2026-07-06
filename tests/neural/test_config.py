"""Offline unit tests for config load/validation in benchmarks/neural/run.py."""
import textwrap

import pytest

from benchmarks.neural.run import ConfigError, load_config, model_params, validate_config


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


_VALID = """
experiment: { h: 24, input_size: 72, seeds: [42, 43], warmup_seeds: 1, threads: 1 }
datasets:
  - { name: A, path: p.csv, ds_col: d, y_col: y, freq: D }
overrides:
  iTransformer:
    nf_params: { n_series: 1 }
"""


def test_load_valid_config(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID))
    assert cfg["experiment"]["h"] == 24
    assert cfg["overrides"]["iTransformer"]["nf_params"]["n_series"] == 1


def test_overrides_are_optional(tmp_path):
    no_overrides = _VALID[: _VALID.index("overrides:")]
    cfg = load_config(_write(tmp_path, no_overrides))
    assert not cfg.get("overrides")


def test_missing_experiment_key_raises(tmp_path):
    bad = _VALID.replace("threads: 1 }", "}")
    with pytest.raises(ConfigError, match="experiment missing keys"):
        load_config(_write(tmp_path, bad))


def test_warmup_out_of_range_raises(tmp_path):
    bad = _VALID.replace("warmup_seeds: 1", "warmup_seeds: 2")  # 2 not < len(seeds)=2
    with pytest.raises(ConfigError, match="warmup_seeds"):
        load_config(_write(tmp_path, bad))


def test_bad_override_key_raises(tmp_path):
    bad = _VALID.replace("nf_params: { n_series: 1 }", "bogus_key: { x: 1 }")
    with pytest.raises(ConfigError, match="override for"):
        load_config(_write(tmp_path, bad))


def test_validate_config_is_callable_directly():
    with pytest.raises(ConfigError, match="missing 'experiment'"):
        validate_config({"datasets": []})


def test_model_params_forces_mae_and_applies_override(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID))
    # a model with no override -> just the forced MAE loss on both sides
    chx, nf = model_params(cfg, "GRU")
    assert chx == {"loss": "mae"} and nf == {"loss": "MAE"}
    # iTransformer -> override merged on top of the forced loss
    _, nf_it = model_params(cfg, "iTransformer")
    assert nf_it == {"loss": "MAE", "n_series": 1}


def test_malformed_yaml_raises_config_error(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("key: [unclosed bracket\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(str(p))


def test_empty_yaml_raises_config_error(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("")
    with pytest.raises(ConfigError, match="config is empty or not a YAML mapping"):
        load_config(str(p))


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(str(tmp_path / "does_not_exist.yaml"))

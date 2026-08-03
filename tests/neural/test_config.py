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
  - { name: A, kind: univariate, path: p.csv, ds_col: d, y_col: y, freq: D }
overrides:
  GRU:
    nf_params: { max_steps: 3 }
"""


def test_load_valid_config(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID))
    assert cfg["experiment"]["h"] == 24
    assert cfg["overrides"]["GRU"]["nf_params"]["max_steps"] == 3


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
    bad = _VALID.replace("nf_params: { max_steps: 3 }", "bogus_key: { x: 1 }")
    with pytest.raises(ConfigError, match="override for"):
        load_config(_write(tmp_path, bad))


def test_validate_config_is_callable_directly():
    with pytest.raises(ConfigError, match="missing 'experiment'"):
        validate_config({"datasets": []})


def test_model_params_forces_mae_and_applies_override(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID))
    # a model with no override -> just the forced MAE loss on both sides
    chx, nf = model_params(cfg, "KAN")
    assert chx == {"loss": "mae"} and nf == {"loss": "MAE"}
    # GRU -> override merged on top of the forced loss
    _, nf_gru = model_params(cfg, "GRU")
    assert nf_gru == {"loss": "MAE", "max_steps": 3}


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


def test_kind_defaults_to_univariate_when_omitted(tmp_path):
    text = """
    experiment: { h: 24, input_size: 72, seeds: [42, 43], warmup_seeds: 1, threads: 1 }
    datasets:
      - { name: A, path: p.csv, ds_col: d, y_col: y, freq: D }
    """
    cfg = load_config(_write(tmp_path, text))
    assert cfg["datasets"][0].get("kind", "univariate") == "univariate"


def test_multivariate_dataset_accepted(tmp_path):
    text = """
    experiment: { h: 24, input_size: 72, seeds: [42, 43], warmup_seeds: 1, threads: 1 }
    datasets:
      - name: M
        kind: multivariate
        path: m.csv
        ds_col: Datetime
        y_cols: [a, b, c]
        freq: D
        h: 7
        input_size: 35
    """
    cfg = load_config(_write(tmp_path, text))
    assert cfg["datasets"][0]["y_cols"] == ["a", "b", "c"]
    assert cfg["datasets"][0]["h"] == 7


def test_covariate_dataset_accepted(tmp_path):
    text = """
    experiment: { h: 24, input_size: 72, seeds: [42, 43], warmup_seeds: 1, threads: 1 }
    datasets:
      - name: C
        kind: covariate
        path: c.csv
        ds_col: Datetime
        y_col: y
        hist_exog_cols: [x1, x2]
        freq: D
    """
    cfg = load_config(_write(tmp_path, text))
    assert cfg["datasets"][0]["hist_exog_cols"] == ["x1", "x2"]


def test_multivariate_requires_y_cols(tmp_path):
    text = """
    experiment: { h: 24, input_size: 72, seeds: [42, 43], warmup_seeds: 1, threads: 1 }
    datasets:
      - { name: M, kind: multivariate, path: m.csv, ds_col: d, freq: D }
    """
    with pytest.raises(ConfigError, match="y_cols"):
        load_config(_write(tmp_path, text))


def test_covariate_requires_hist_exog_cols(tmp_path):
    text = """
    experiment: { h: 24, input_size: 72, seeds: [42, 43], warmup_seeds: 1, threads: 1 }
    datasets:
      - { name: C, kind: covariate, path: c.csv, ds_col: d, y_col: y, freq: D }
    """
    with pytest.raises(ConfigError, match="hist_exog_cols"):
        load_config(_write(tmp_path, text))


def test_too_many_features_rejected(tmp_path):
    text = """
    experiment: { h: 24, input_size: 72, seeds: [42, 43], warmup_seeds: 1, threads: 1 }
    datasets:
      - name: M
        kind: multivariate
        path: m.csv
        ds_col: Datetime
        y_cols: [a, b, c, d, e]
        freq: D
    """
    with pytest.raises(ConfigError, match="y_cols"):
        load_config(_write(tmp_path, text))


def test_shipped_config_yaml_validates():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(str(root / "benchmarks" / "neural" / "config.yaml"))
    kinds = {d.get("kind", "univariate") for d in cfg["datasets"]}
    assert kinds == {"univariate", "multivariate", "covariate"}
    assert len(cfg["datasets"]) == 9

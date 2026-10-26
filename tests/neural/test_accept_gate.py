"""Offline unit tests for the accept-gate + summaries in benchmarks/neural/run.py."""
import pandas as pd
import pytest

from benchmarks.neural.run import (NeuralBenchError, accept_gate_report,
                                   check_committed_report, is_canonical,
                                   paired_accuracy, paired_speed, summarize)

SEEDS = [42, 43, 44, 45, 46]
MODELS, DATASETS, LIBS = ["GRU"], ["A"], ["chronax", "nixtla"]


def _row(lib, model, dataset, seed, i, mae, wc):
    return {"library": lib, "dataset": dataset, "model": model, "seed": seed,
            "iter_idx": i, "is_warmup": i < 1, "mae": mae, "smape": mae,
            "wallclock_s": wc, "error": ""}


def _canonical_df(chronax_mae, nixtla_mae, chronax_wc, nixtla_wc):
    rows = []
    for i, s in enumerate(SEEDS):
        rows.append(_row("chronax", "GRU", "A", s, i, chronax_mae[i], chronax_wc[i]))
        rows.append(_row("nixtla", "GRU", "A", s, i, nixtla_mae[i], nixtla_wc[i]))
    return pd.DataFrame(rows)


def test_paired_accuracy_noise_flip():
    # chronax always 0.1 BELOW nixtla per seed, but per-lib std is huge:
    chr_mae = [10.0, 20.0, 30.0, 40.0, 50.0]
    nf_mae = [10.1, 20.1, 30.1, 40.1, 50.1]
    df = _canonical_df(chr_mae, nf_mae, [1]*5, [2]*5)
    acc = paired_accuracy(df, "GRU", "A")
    assert acc["pass"] is True                       # paired delta < 0
    assert abs(acc["paired_mean_delta"] + 0.1) < 1e-9
    assert acc["chronax_std"] > 10                   # per-lib dispersion is large


def test_paired_speed_after_warmup_only():
    # warmup seed (i=0) chronax is slow; after-warmup chronax is faster.
    df = _canonical_df([1]*5, [1]*5, [100, 1, 1, 1, 1], [5, 5, 5, 5, 5])
    spd = paired_speed(df, "GRU", "A", warmup_seeds=1)
    assert spd["n"] == 4 and spd["pass"] is True
    assert abs(spd["chronax_mean"] - 1.0) < 1e-9     # warmup 100 excluded


def test_is_canonical_true_for_full_set():
    df = _canonical_df([1]*5, [1]*5, [1]*5, [1]*5)
    assert is_canonical(df, MODELS, DATASETS, LIBS, SEEDS) is True


def test_is_canonical_false_when_seed_missing():
    df = _canonical_df([1]*5, [1]*5, [1]*5, [1]*5).iloc[:-1]  # drop last nixtla seed
    assert is_canonical(df, MODELS, DATASETS, LIBS, SEEDS) is False


def test_is_canonical_false_with_error_row():
    df = _canonical_df([1]*5, [1]*5, [1]*5, [1]*5)
    df.loc[0, "error"] = "RuntimeError: diverged"
    assert is_canonical(df, MODELS, DATASETS, LIBS, SEEDS) is False


def test_is_canonical_false_for_resumed_iter_idx():
    # Resume pattern: all seeds present exactly once (key Counter matches) but the
    # chronax worker restarted iter_idx at 0 for the resumed leftovers, so the
    # per-group iter_idx multiset is [0,1,0,1,2] != range(5). Must be non-canonical
    # (contaminated warmup), even though seed-completeness passes.
    df = _canonical_df([1]*5, [1]*5, [1]*5, [1]*5)
    resumed_iter = [0, 1, 0, 1, 2]  # done [42,43] then resumed [44,45,46] from 0
    chronax_rows = df.index[df.library == "chronax"].tolist()
    for idx, it in zip(chronax_rows, resumed_iter):
        df.loc[idx, "iter_idx"] = it
    assert is_canonical(df, MODELS, DATASETS, LIBS, SEEDS) is False


def test_accept_gate_report_non_canonical_notice():
    df = _canonical_df([1]*5, [1]*5, [1]*5, [1]*5).iloc[:-1]
    txt = accept_gate_report(df, MODELS, DATASETS, LIBS, SEEDS, warmup_seeds=1)
    assert "non-canonical" in txt.lower()
    assert "OVERALL" not in txt


def test_accept_gate_report_chronax_only_is_non_canonical():
    # A chronax-only frame must NOT yield a verdict: the gate requires BOTH libs.
    df = _canonical_df([1]*5, [1]*5, [1]*5, [1]*5)
    df = df[df.library == "chronax"]
    txt = accept_gate_report(df, MODELS, DATASETS, LIBS, SEEDS, warmup_seeds=1)
    assert "non-canonical" in txt.lower() and "OVERALL" not in txt


def test_accept_gate_report_overall_pass():
    df = _canonical_df([1]*5, [2]*5, [1]*5, [2]*5)  # chronax more accurate + faster
    txt = accept_gate_report(df, MODELS, DATASETS, LIBS, SEEDS, warmup_seeds=1)
    assert "OVERALL VERDICT: PASS" in txt


def test_accept_gate_report_overall_fail_lists_cell():
    df = _canonical_df([3]*5, [2]*5, [1]*5, [2]*5)  # chronax LESS accurate
    txt = accept_gate_report(df, MODELS, DATASETS, LIBS, SEEDS, warmup_seeds=1)
    assert "OVERALL VERDICT: FAIL" in txt and "GRU/A" in txt


def test_summarize_has_after_warmup_columns():
    df = _canonical_df([1]*5, [1]*5, [10, 1, 1, 1, 1], [5]*5)
    s = summarize(df, warmup_seeds=1, group_cols=["model", "library", "dataset"])
    assert "wallclock_mean_after" in s.columns and "wallclock_std_after" in s.columns
    chr_after = s.loc[("GRU", "chronax", "A"), "wallclock_mean_after"]
    assert abs(chr_after - 1.0) < 1e-9  # warmup 10 excluded


def test_check_committed_report_pass():
    chronax = _canonical_df([1]*5, [1]*5, [2, 1, 1, 1, 1], [9]*5)
    chronax = chronax[chronax.library == "chronax"]
    committed = pd.DataFrame([{"dataset": "A", "mae_mean": 2.0,
                               "wallclock_mean_after": 5.0}])
    txt = check_committed_report(chronax, committed, warmup_seeds=1)
    assert "PASS" in txt  # chronax mae 1.0 < 2.0 and after-warmup 1.0 < 5.0


def test_check_committed_report_raises_on_legacy_summary():
    # Migrated GRU summary predates the pinned-thread recapture (no *_after col).
    chx = _canonical_df([1]*5, [1]*5, [1]*5, [1]*5)
    chx = chx[chx.library == "chronax"]
    legacy = pd.DataFrame([{"dataset": "A", "mae_mean": 2.0, "wallclock_mean": 5.0}])
    with pytest.raises(NeuralBenchError, match="wallclock_mean_after"):
        check_committed_report(chx, legacy, warmup_seeds=1)

"""Neural benchmark orchestrator (Chronax vs Nixtla neuralforecast).

Built up across tasks: config load/validate + error taxonomy (Task 4), resume
load (Task 8), accept-gate (Task 9), orchestration/modes + baseline metadata
(Task 10).
"""
from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yaml


class NeuralBenchError(Exception):
    """Base error for the neural benchmark harness."""


class ConfigError(NeuralBenchError):
    """Raised when config.yaml is missing required keys or has bad values."""


_REQUIRED_EXP = {"h", "input_size", "seeds", "warmup_seeds", "threads"}


def validate_config(cfg: dict) -> None:
    """Raise ConfigError if `cfg` is missing required structure."""
    if not isinstance(cfg, dict):
        raise ConfigError("config is empty or not a YAML mapping")
    if "experiment" not in cfg:
        raise ConfigError("missing 'experiment' section")
    exp = cfg["experiment"]
    missing = _REQUIRED_EXP - set(exp)
    if missing:
        raise ConfigError(f"experiment missing keys: {sorted(missing)}")
    if not cfg.get("datasets"):
        raise ConfigError("no datasets defined")
    if not cfg.get("models"):
        raise ConfigError("no models defined")
    for m in cfg["models"]:
        if not {"name", "chronax_params", "nf_params"} <= set(m):
            raise ConfigError(f"model entry incomplete: {m}")
    warm, n = exp["warmup_seeds"], len(exp["seeds"])
    if not (0 <= warm < n):
        raise ConfigError(f"warmup_seeds {warm} out of range for {n} seeds")


def load_config(path: str) -> dict:
    """Load and validate config.yaml.

    A missing or unreadable file surfaces as ConfigError (not a bare OSError) so
    every config failure stays inside the NeuralBenchError taxonomy.
    """
    try:
        f = open(path)
    except OSError as e:  # FileNotFoundError / PermissionError / ...
        raise ConfigError(f"cannot read config {path!r}: {e}") from e
    with f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML: {e}") from e
    validate_config(cfg)
    return cfg


FIELDS = ["library", "dataset", "model", "seed", "iter_idx", "is_warmup",
          "mae", "smape", "wallclock_s", "error"]

_NUM_COLS = ["mae", "smape", "wallclock_s"]


def load_results_tolerant(path: str | Path) -> pd.DataFrame:
    """Read the results CSV, dropping any torn line left by a crash.

    The worker streams one full row per seed via csv.DictWriter, so a clean line
    has exactly len(FIELDS) fields (error messages with commas/newlines are
    quoted). We parse with the stdlib csv reader — quote-aware, so a multi-line
    quoted error field is reassembled into one record — and keep only rows whose
    field count is exactly len(FIELDS). A crash-truncated trailing line has too
    few fields and is dropped. This does NOT rely on NaN-key/ParserError: pandas
    NaN-pads a short final line instead of raising, so a torn line truncated
    after the key columns must be caught by field count, not by dropna-on-keys.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=FIELDS)
    with p.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != FIELDS:
        return pd.DataFrame(columns=FIELDS)
    good = [r for r in rows[1:] if len(r) == len(FIELDS)]
    df = pd.DataFrame(good, columns=FIELDS)
    if len(df):
        df["seed"] = df["seed"].astype(int)
        df["iter_idx"] = df["iter_idx"].astype(int)
        for c in _NUM_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["error"] = df["error"].fillna("").astype(str)
    return df.reset_index(drop=True)


def done_keys(df: pd.DataFrame) -> set:
    """Set of (model, dataset, library, seed) rows already present."""
    return {(r.model, r.dataset, r.library, int(r.seed)) for r in df.itertuples()}


def remaining_seeds(model: str, dataset: str, library: str, seeds: Sequence[int], done: set[tuple]) -> list[int]:
    """Seeds not yet present for this {model,dataset,library}."""
    return [s for s in seeds if (model, dataset, library, int(s)) not in done]


from collections import Counter

# The accept-gate certifies a verdict only for a full paired run over BOTH
# libraries — independent of which --libs the current invocation actually ran.
REQUIRED_LIBS = ["chronax", "nixtla"]


def summarize(df: pd.DataFrame, warmup_seeds: int, group_cols: list) -> pd.DataFrame:
    """Mean/std over all seeds for MAE/sMAPE/wall-clock, plus an after-warmup
    wall-clock mean/std over the non-warmup seeds (spec §8)."""
    total = df.groupby(group_cols).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        smape_mean=("smape", "mean"), smape_std=("smape", "std"),
        wallclock_mean=("wallclock_s", "mean"), wallclock_std=("wallclock_s", "std"),
    ).round(4)
    after = df[df.iter_idx >= warmup_seeds].groupby(group_cols).agg(
        wallclock_mean_after=("wallclock_s", "mean"),
        wallclock_std_after=("wallclock_s", "std"),
    ).round(4)
    return total.join(after)


def _paired(df, model, dataset, value_col, sub_filter=None):
    d = df if sub_filter is None else df[sub_filter(df)]
    def series(lib):
        s = d[(d.model == model) & (d.dataset == dataset) & (d.library == lib)]
        return s.sort_values("seed").set_index("seed")[value_col]
    c, n = series("chronax"), series("nixtla")
    seeds = c.index.intersection(n.index)
    delta = c.loc[seeds] - n.loc[seeds]
    return c, n, seeds, delta


def paired_accuracy(df, model, dataset) -> dict:
    c, n, seeds, delta = _paired(df, model, dataset, "mae")
    return {
        "chronax_mean": float(c.mean()), "chronax_std": float(c.std(ddof=1)),
        "nixtla_mean": float(n.mean()), "nixtla_std": float(n.std(ddof=1)),
        "paired_mean_delta": float(delta.mean()), "n": int(len(seeds)),
        "pass": bool(delta.mean() <= 0),  # chronax accuracy >= NF (spec §8)
    }


def paired_speed(df, model, dataset, warmup_seeds) -> dict:
    c, n, seeds, delta = _paired(
        df, model, dataset, "wallclock_s",
        sub_filter=lambda d: d.iter_idx >= warmup_seeds)
    return {
        "chronax_mean": float(c.mean()), "chronax_std": float(c.std(ddof=1)),
        "nixtla_mean": float(n.mean()), "nixtla_std": float(n.std(ddof=1)),
        "paired_mean_delta": float(delta.mean()), "n": int(len(seeds)),
        "pass": bool(c.mean() < n.mean()),  # chronax after-warmup speed > NF
    }


def is_canonical(df, models, datasets, libs, seeds) -> bool:
    """True iff df is exactly one clean, full, single-run set — the only file the
    gate will certify (spec §9). Requires:
      1. every (model,dataset,library,seed) present exactly once (no missing/dup),
      2. no error rows, and
      3. per (model,dataset,library) the iter_idx values are exactly
         range(len(seeds)).

    Check 3 is what makes a RESUMED file non-canonical: a resume runs only the
    remaining seeds, so the worker restarts iter_idx at 0 and the group's
    iter_idx multiset (e.g. [0,0,1,1,2]) no longer equals range(len(seeds)) even
    though every seed is present once. Resumed seeds re-pay JIT warmup, so their
    wall-clock is contaminated and the gate must refuse.
    """
    want = list(seeds)
    expected = Counter((m, d, l, int(s))
                       for m in models for d in datasets for l in libs for s in want)
    actual = Counter((r.model, r.dataset, r.library, int(r.seed))
                     for r in df.itertuples())
    if actual != expected:
        return False
    if df["error"].fillna("").astype(str).str.len().gt(0).any():
        return False
    want_iter = list(range(len(want)))
    for m in models:
        for d in datasets:
            for l in libs:
                grp = df[(df.model == m) & (df.dataset == d) & (df.library == l)]
                if sorted(int(x) for x in grp["iter_idx"]) != want_iter:
                    return False
    return True


def accept_gate_report(df, models, datasets, libs, seeds, warmup_seeds) -> str:
    """Per-cell paired accept-gate table + overall verdict, or a non-canonical
    notice (spec §8/§9). `libs` is the REQUIRED library set (both chronax and
    nixtla) — a run missing a library, resumed, incomplete, or carrying error
    rows is non-canonical. Returns text; never raises, never exits non-zero."""
    if not is_canonical(df, models, datasets, libs, seeds):
        return ("=== Accept-gate ===\n"
                "non-canonical timing (resumed / incomplete / missing a library / "
                "error rows present) — rerun a clean full run for a verdict.")
    lines = ["=== Accept-gate (paired per-seed; chronax vs nixtla) ==="]
    overall_pass, fails = True, []
    for model in models:
        for dataset in datasets:
            acc = paired_accuracy(df, model, dataset)
            spd = paired_speed(df, model, dataset, warmup_seeds)
            cell_pass = acc["pass"] and spd["pass"]
            overall_pass = overall_pass and cell_pass
            cell = f"{model}/{dataset}"
            if not cell_pass:
                fails.append(cell)
            lines.append(
                f"[{'PASS' if cell_pass else 'FAIL'}] {cell}  "
                f"MAE chx={acc['chronax_mean']:.4f}±{acc['chronax_std']:.4f} "
                f"nf={acc['nixtla_mean']:.4f}±{acc['nixtla_std']:.4f} "
                f"Δ={acc['paired_mean_delta']:+.4f}(n={acc['n']})  |  "
                f"time(after) chx={spd['chronax_mean']:.2f}±{spd['chronax_std']:.2f} "
                f"nf={spd['nixtla_mean']:.2f}±{spd['nixtla_std']:.2f} "
                f"Δ={spd['paired_mean_delta']:+.2f}(n={spd['n']})")
    verdict = "PASS" if overall_pass else f"FAIL ({', '.join(fails)})"
    lines.append(f"OVERALL VERDICT: {verdict}")
    return "\n".join(lines)


def check_committed_report(chronax_df, committed_summary, warmup_seeds) -> str:
    """Coarse (unpaired) chronax-vs-committed-NF-summary check (spec §8
    --check-committed). Not the paired canonical verdict; cheap iteration.

    Requires a post-recapture summary (with the after-warmup wall-clock column);
    the migrated legacy GRU summary predates it, so guard explicitly rather than
    KeyError mid-report."""
    if "wallclock_mean_after" not in committed_summary.columns:
        raise NeuralBenchError(
            "committed baseline summary lacks the 'wallclock_mean_after' column "
            "(pre-pinned-thread schema) — recapture it with --refresh-baseline.")
    chx = summarize(chronax_df, warmup_seeds, group_cols=["dataset"])
    lines = ["=== Check against committed NF baseline (unpaired) ==="]
    committed = committed_summary.set_index("dataset")
    for dataset, row in chx.iterrows():
        nf = committed.loc[dataset]
        acc_pass = row["mae_mean"] <= nf["mae_mean"]
        spd_pass = row["wallclock_mean_after"] < nf["wallclock_mean_after"]
        verdict = "PASS" if (acc_pass and spd_pass) else "FAIL"
        lines.append(
            f"[{verdict}] {dataset}  MAE chx={row['mae_mean']:.4f} "
            f"nf={nf['mae_mean']:.4f}  time(after) chx={row['wallclock_mean_after']:.2f} "
            f"nf={nf['wallclock_mean_after']:.2f}")
    return "\n".join(lines)

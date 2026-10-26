"""Offline unit tests for resume logic in benchmarks/neural/run.py."""
from benchmarks.neural.run import (FIELDS, done_keys, load_results_tolerant,
                                   remaining_seeds)

_HEADER = ",".join(FIELDS)
_ROW_A = "chronax,AirlinePassengers,GRU,42,0,True,10.0,5.0,1.2,"
_ROW_B = "chronax,AirlinePassengers,GRU,43,1,False,11.0,6.0,1.1,"


def test_load_missing_returns_empty(tmp_path):
    df = load_results_tolerant(tmp_path / "nope.csv")
    assert list(df.columns) == FIELDS and len(df) == 0


def test_load_clean_rows(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("\n".join([_HEADER, _ROW_A, _ROW_B]) + "\n")
    df = load_results_tolerant(p)
    assert len(df) == 2


def test_load_tolerates_torn_trailing_line(tmp_path):
    p = tmp_path / "r.csv"
    # simulate a crash mid-write: last line truncated (no newline, too few fields).
    # It truncates AFTER the key columns, so field-count detection (not NaN-key)
    # is what must drop it.
    p.write_text("\n".join([_HEADER, _ROW_A, _ROW_B]) + "\nchronax,AirlinePassengers,GRU,44,2,Fal")
    df = load_results_tolerant(p)
    assert len(df) == 2  # torn line (6 fields != len(FIELDS)) dropped
    assert done_keys(df) == {("GRU", "AirlinePassengers", "chronax", 42),
                             ("GRU", "AirlinePassengers", "chronax", 43)}


def test_remaining_seeds_skips_done():
    done = {("GRU", "AirlinePassengers", "chronax", 42)}
    rem = remaining_seeds("GRU", "AirlinePassengers", "chronax", [42, 43, 44], done)
    assert rem == [43, 44]

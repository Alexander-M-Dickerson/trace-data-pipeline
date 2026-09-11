"""Prove `_diff` on known-equal and known-diff fixtures."""
import numpy as np
import pandas as pd

from lib import validate_core as vc


def _base(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "cusip": [f"C{i:05d}" for i in range(n)],
        "date": pd.Timestamp("2010-01-31"),
        "ret_vw": rng.normal(0, 0.02, n),
        "nret": rng.integers(1, 20, n).astype("int16"),
        "flag": rng.integers(0, 2, n).astype("int8"),
    })


def test_known_equal_passes():
    a = _base()
    b = a.copy()
    b["cusip"] = b["cusip"].astype("category")          # dtype noise must not fail the diff
    b["ret_vw"] = a["ret_vw"] + 1e-9                    # sub-tolerance perturbation
    passed, report = vc.diff_frames(a, b, key=("cusip", "date"))
    assert passed, vc.format_report(report)
    assert report["key_gaps_ours"] == 0 and report["key_gaps_golden"] == 0


def test_float_out_of_tol_fails():
    a = _base()
    b = a.copy()
    b.loc[3, "ret_vw"] += 1e-4                          # >> 1e-6 tolerance
    passed, report = vc.diff_frames(a, b, key=("cusip", "date"))
    assert not passed
    rec = next(r for r in report["columns"] if r["col"] == "ret_vw")
    assert not rec["pass"] and rec["max_abs_d"] > 1e-6


def test_rate_tol_rescues_rate_cols():
    a = _base()
    b = a.copy()
    b["ret_vw"] = a["ret_vw"] + 5e-5                    # inside 1e-4, outside 1e-6
    passed_strict, _ = vc.diff_frames(a, b, key=("cusip", "date"))
    passed_rate, _ = vc.diff_frames(a, b, key=("cusip", "date"), rate_cols=("ret_vw",))
    assert not passed_strict and passed_rate


def test_exact_col_mismatch_fails():
    a = _base()
    b = a.copy()
    b.loc[5, "nret"] = 99
    passed, report = vc.diff_frames(a, b, key=("cusip", "date"))
    rec = next(r for r in report["columns"] if r["col"] == "nret")
    assert not passed and rec["kind"] == "exact" and rec["n_mismatch"] == 1


def test_nan_mismatch_fails():
    a = _base()
    b = a.copy()
    b.loc[7, "ret_vw"] = np.nan
    passed, report = vc.diff_frames(a, b, key=("cusip", "date"))
    rec = next(r for r in report["columns"] if r["col"] == "ret_vw")
    assert not passed and rec["nan_mismatch"] == 1


def test_both_nan_is_equal():
    a = _base()
    a.loc[2, "ret_vw"] = np.nan
    b = a.copy()
    passed, report = vc.diff_frames(a, b, key=("cusip", "date"))
    assert passed, vc.format_report(report)


def test_key_gap_fails():
    a = _base()
    b = _base().iloc[:-1]                               # golden missing one key
    passed, report = vc.diff_frames(a, b, key=("cusip", "date"))
    assert not passed and report["key_gaps_ours"] == 1 and report["key_gaps_golden"] == 0


def test_column_set_reported_not_fatal():
    a = _base().drop(columns=["flag"])
    b = _base().assign(extra=1.0)
    passed, report = vc.diff_frames(a, b, key=("cusip", "date"))
    assert report["cols_only_golden"] == ["flag", "extra"] or \
           set(report["cols_only_golden"]) == {"flag", "extra"}
    assert passed  # shared columns all match; the caller owns column-set policy

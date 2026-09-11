"""validate_core.py -- the golden-diff engine (`_diff` is the contract).

Compares OUR step output to the matching golden intermediate on a key, with per-column tolerances:
exact for counts/codes/flags/strings/dates; FLOAT_TOL for prices/returns/signals; RATE_TOL for
rate-like winsorized float32 columns (ytm, cs, betas). PASS iff no key gaps and every shared column
within tolerance. The report is JSON-serializable and lands verbatim in the run manifest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import _stage2_settings as cfg

# dtypes compared exactly (mismatch count must be 0); floats get a tolerance
_EXACT_KINDS = "iubOSUM"  # int, uint, bool, object, bytes/str, unicode, datetime


def _load(obj: pd.DataFrame | Path | str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.read_parquet(obj, columns=list(columns) if columns else None)


def _norm_key(df: pd.DataFrame, key: Sequence[str]) -> pd.DataFrame:
    """Normalize join-key dtypes so category-vs-string / ns-vs-us never causes a false gap."""
    for k in key:
        col = df[k]
        if isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object:
            df[k] = col.astype(str)
        elif np.issubdtype(col.dtype, np.datetime64):
            df[k] = pd.to_datetime(col).dt.tz_localize(None)
    return df


def diff_frames(
    ours: pd.DataFrame | Path | str,
    golden: pd.DataFrame | Path | str,
    key: Sequence[str] = ("cusip", "date"),
    float_tol: float = cfg.FLOAT_TOL,
    rate_tol: float = cfg.RATE_TOL,
    rate_cols: Iterable[str] = (),
    exact_cols: Iterable[str] = (),
    ignore_cols: Iterable[str] = (),
    join: str = "outer",
    col_tols: dict[str, float] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Join `ours` vs `golden` on `key`; return (passed, report).

    join='outer' (default): both key sets must coincide (key_gaps == 0 both sides).
    join='golden': golden's keys must all exist in ours (key_gaps_golden == 0); extra keys on our
    side are reported but allowed -- for blocks whose grain is wider than the golden target
    (e.g. per-bond signals vs the panel-key mmn file).

    passed  iff the key condition holds AND every shared non-key column passes:
      - float columns: max|d| <= tol on both-non-null rows AND nan_mismatch == 0
        (tol = rate_tol for rate_cols, else float_tol)
      - exact columns (ints/flags/strings/dates or listed in exact_cols): 0 mismatches
        (both-null counts as equal)
    Columns present on only one side are reported (never silently dropped) but don't fail the
    diff -- the caller decides whether the column set itself is part of the gate.
    """
    key = list(key)
    rate_cols, exact_cols, ignore_cols = set(rate_cols), set(exact_cols), set(ignore_cols)

    o = _norm_key(_load(ours).copy(), key)
    g = _norm_key(_load(golden).copy(), key)

    shared = [c for c in g.columns if c in o.columns and c not in key and c not in ignore_cols]
    only_ours = [c for c in o.columns if c not in g.columns and c not in key]
    only_golden = [c for c in g.columns if c not in o.columns and c not in key]

    m = o[key + shared].merge(g[key + shared], on=key, how="outer",
                              suffixes=("_o", "_g"), indicator=True)
    gaps_ours = int((m["_merge"] == "left_only").sum())
    gaps_golden = int((m["_merge"] == "right_only").sum())
    both = m[m["_merge"] == "both"]

    columns: list[dict[str, Any]] = []
    all_pass = gaps_golden == 0 and (join == "golden" or gaps_ours == 0)
    for c in shared:
        a, b = both[f"{c}_o"], both[f"{c}_g"]
        is_float = (a.dtype.kind == "f" or b.dtype.kind == "f") and c not in exact_cols
        rec: dict[str, Any] = {"col": c, "n_both": int(len(both))}
        if is_float:
            av = a.astype("float64").to_numpy()
            bv = b.astype("float64").to_numpy()
            an, bn = np.isnan(av), np.isnan(bv)
            nan_mismatch = int((an != bn).sum())
            ok = ~an & ~bn
            max_d = float(np.max(np.abs(av[ok] - bv[ok]))) if ok.any() else 0.0
            tol = (col_tols or {}).get(c, rate_tol if c in rate_cols else float_tol)
            passed = nan_mismatch == 0 and max_d <= tol
            rec.update(kind="rate" if c in rate_cols else "float", tol=tol,
                       max_abs_d=max_d, nan_mismatch=nan_mismatch, n_compared=int(ok.sum()))
        else:
            if isinstance(a.dtype, pd.CategoricalDtype) or a.dtype == object or \
               isinstance(b.dtype, pd.CategoricalDtype) or b.dtype == object:
                a, b = a.astype(str), b.astype(str)  # 'nan' == 'nan' -> both-null equal
                neq = (a != b)
            else:
                neq = ~((a == b) | (a.isna() & b.isna()))
            n_mismatch = int(neq.sum())
            passed = n_mismatch == 0
            rec.update(kind="exact", n_mismatch=n_mismatch)
        rec["pass"] = bool(passed)
        columns.append(rec)
        all_pass &= passed

    report = {
        "key": key, "rows_ours": int(len(o)), "rows_golden": int(len(g)),
        "key_gaps_ours": gaps_ours, "key_gaps_golden": gaps_golden,
        "cols_only_ours": only_ours, "cols_only_golden": only_golden,
        "columns": columns, "pass": bool(all_pass),
    }
    return bool(all_pass), report


def format_report(report: dict[str, Any], max_rows: int = 200) -> str:
    """Human-readable verdict table for logs; the JSON report stays the machine record."""
    lines = [
        f"rows ours={report['rows_ours']:,} golden={report['rows_golden']:,} "
        f"| key gaps ours={report['key_gaps_ours']:,} golden={report['key_gaps_golden']:,}",
    ]
    if report["cols_only_ours"]:
        lines.append(f"cols only-ours: {report['cols_only_ours']}")
    if report["cols_only_golden"]:
        lines.append(f"cols only-golden: {report['cols_only_golden']}")
    for rec in report["columns"][:max_rows]:
        if rec["kind"] == "exact":
            detail = f"mismatches={rec['n_mismatch']:,}"
        else:
            detail = (f"max|d|={rec['max_abs_d']:.3g} tol={rec['tol']:.0e} "
                      f"nan_mm={rec['nan_mismatch']:,}")
        lines.append(f"  {'PASS' if rec['pass'] else 'FAIL'}  {rec['col']:<16} [{rec['kind']}] {detail}")
    lines.append(f"VERDICT: {'PASS' if report['pass'] else 'FAIL'}")
    return "\n".join(lines)

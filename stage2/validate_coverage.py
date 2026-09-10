"""validate_coverage.py -- guard against silent column staleness in the EXTEND-mode (ours) panel.

The golden-vintage pins (the treasury cutoff, the pinned factor panel, the frozen AUX merges) can
silently cap a column several months before the panel end. The `TRET_MAX_DATE` pin once did exactly
this: `tret` -> `ret_vwx` -> the BBW x-factors -> the whole bbw_factors block (inner join) -> every
rolling beta / ivol / sysmom -- 43 columns died 6 months before the panel's own last month, while
the panel still reported rows there (debug.md M13 / assumptions.md A19).

The golden validators CANNOT catch this: they diff against the golden, which is capped the same way,
so both sides agree on NULL. This guard is orthogonal -- it asserts, on the ours panel ALONE, that
every column that is healthy mid-panel still reaches within `max_lag` months of the panel's own max
month. A column empty >= max_lag+1 months early, despite being populated earlier, is a straggler.

    validate_coverage.py                                   # main_panel_ours, max_lag=1
    validate_coverage.py --panel <path> --max-lag 1 --json-out report.json
    validate_coverage.py --panel output/panel/main_panel_ours_plus.parquet

Why max_lag=1 by default: forward-difference columns (`lib`, `libd`, and the *_bgn terminal returns)
legitimately cannot compute the panel's FINAL month (they need month t+1), so they lose exactly one
month -- that is correct, not a straggler. Anything that dies 2+ months early is the bug shape.
A column is only judged if it was "healthy" (coverage >= HEALTHY_FRAC) at its own last populated
month -- naturally-sparse or always-empty columns are reported separately, never failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

import _stage2_settings as cfg

HEALTHY_FRAC = 0.30      # a column counts as "was healthy" if >=30% of bonds had it at its last month
DATE_COL = "date"


def _month_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Whole calendar months from b to a (a >= b => >= 0)."""
    return (a.year - b.year) * 12 + (a.month - b.month)


def check_coverage(panel: Path, max_lag: int = 1) -> tuple[bool, dict]:
    """Return (ok, report). ok=False iff any healthy column ends > max_lag months before panel max."""
    con = duckdb.connect()
    cols = con.execute(f"SELECT * FROM read_parquet('{panel}') LIMIT 0").df().columns.tolist()
    value_cols = [c for c in cols if c != DATE_COL]

    panel_max = con.execute(f"SELECT MAX({DATE_COL}) FROM read_parquet('{panel}')").fetchone()[0]
    panel_max = pd.Timestamp(panel_max)

    # one pass: last month each column is non-null
    sel = ", ".join(f'MAX(CASE WHEN "{c}" IS NOT NULL THEN {DATE_COL} END) AS "{c}"' for c in value_cols)
    last = con.execute(f"SELECT {sel} FROM read_parquet('{panel}')").df().iloc[0]

    # coverage of each column IN its own last month (to tell stragglers from naturally-sparse cols)
    stragglers, sparse, ok_cols = [], [], []
    for c in value_cols:
        lm = last[c]
        if pd.isna(lm):
            sparse.append({"col": c, "reason": "all-null"})
            continue
        lm = pd.Timestamp(lm)
        lag = _month_diff(panel_max, lm)
        if lag <= max_lag:
            ok_cols.append(c)
            continue
        # ends early -- is it a real straggler (healthy at its last month) or just sparse?
        n, nn = con.execute(
            f'SELECT COUNT(*), COUNT("{c}") FROM read_parquet(\'{panel}\') WHERE {DATE_COL} = ?',
            [lm.to_pydatetime()]).fetchone()
        frac = (nn / n) if n else 0.0
        rec = {"col": c, "last_month": str(lm.date()), "lag_months": lag,
               "coverage_at_last": round(frac, 3)}
        (stragglers if frac >= HEALTHY_FRAC else sparse).append(rec)

    stragglers.sort(key=lambda r: (-r["lag_months"], r["col"]))
    ok = len(stragglers) == 0
    report = {
        "panel": str(panel), "panel_max_month": str(panel_max.date()), "max_lag": max_lag,
        "n_cols": len(value_cols), "n_ok": len(ok_cols),
        "n_stragglers": len(stragglers), "n_sparse_ignored": len(sparse),
        "stragglers": stragglers, "sparse_or_empty": sparse,
    }
    return ok, report


def format_report(report: dict) -> str:
    lines = [
        f"panel={Path(report['panel']).name}  max_month={report['panel_max_month']}  "
        f"max_lag={report['max_lag']}",
        f"columns: {report['n_ok']}/{report['n_cols']} reach the frontier; "
        f"{report['n_stragglers']} stragglers; {report['n_sparse_ignored']} sparse/empty (ignored)",
    ]
    if report["stragglers"]:
        lines.append("STRAGGLERS (healthy mid-panel, then die early -- the pin-cap shape):")
        for r in report["stragglers"]:
            lines.append(f"  {r['col']:24s} last={r['last_month']}  "
                         f"lag={r['lag_months']}mo  cov@last={r['coverage_at_last']:.0%}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="ours-panel column-coverage guard")
    ap.add_argument("--panel", type=Path, default=cfg.PANEL_DIR / "main_panel_ours.parquet")
    ap.add_argument("--max-lag", type=int, default=1,
                    help="max months a healthy column may end before the panel max (default 1: "
                         "allows forward-difference terminal-edge columns like lib/libd)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    ok, report = check_coverage(args.panel, args.max_lag)
    print(format_report(report))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=1))
    print("[coverage] overall:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

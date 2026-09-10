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

    validate_coverage.py                                   # the configured build, max_lag=1
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


# Columns whose sample legitimately ends early because the SOURCE data does, not because
# anything here is wrong. Each entry names the upstream limit so it can be re-checked; a
# column is only excused while the reason still holds.
#
# Keeping these out of the failure set matters: a guard that always reports FAIL is a
# guard people stop reading, and then a REAL straggler goes unnoticed. Anything not listed
# here still fails.
UPSTREAM_LIMITED = {
    "b_cptlt": "He-Kelly-Manela intermediary capital: published data ends 2025-05",
    "b_dcpi": "FRED CPIAUCSL has no 2025-10 observation, so the CPI change is unavailable",
    "b_cpi_vol6": "same CPI gap; the 6-month rolling window cannot clear it",
}


def check_coverage(panel: Path, max_lag: int = 1,
                   allow_upstream: bool = True) -> tuple[bool, dict]:
    """Return (ok, report). ok=False iff any healthy column ends > max_lag months before panel max.

    Columns in UPSTREAM_LIMITED are reported separately and do not fail the check unless
    `allow_upstream=False`.
    """
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
    known, unexpected = [], []
    for r in stragglers:
        if allow_upstream and r["col"] in UPSTREAM_LIMITED:
            r = {**r, "reason": UPSTREAM_LIMITED[r["col"]]}
            known.append(r)
        else:
            unexpected.append(r)
    ok = len(unexpected) == 0
    report = {
        "panel": str(panel), "panel_max_month": str(panel_max.date()), "max_lag": max_lag,
        "n_cols": len(value_cols), "n_ok": len(ok_cols),
        "n_stragglers": len(unexpected), "n_upstream_limited": len(known),
        "n_sparse_ignored": len(sparse),
        "stragglers": unexpected, "upstream_limited": known, "sparse_or_empty": sparse,
    }
    return ok, report


def format_report(report: dict) -> str:
    lines = [
        f"panel={Path(report['panel']).name}  max_month={report['panel_max_month']}  "
        f"max_lag={report['max_lag']}",
        f"columns: {report['n_ok']}/{report['n_cols']} reach the frontier; "
        f"{report['n_stragglers']} stragglers; {report['n_sparse_ignored']} sparse/empty (ignored)",
    ]
    if report.get("upstream_limited"):
        lines.append("UPSTREAM-LIMITED (source data ends early -- not a pipeline problem):")
        for r in report["upstream_limited"]:
            lines.append(f"  {r['col']:24} last={r['last_month']}  lag={r['lag_months']}mo"
                         f"  -- {r['reason']}")
    if report["stragglers"]:
        lines.append("STRAGGLERS (healthy mid-panel, then die early -- the pin-cap shape):")
        for r in report["stragglers"]:
            lines.append(f"  {r['col']:24s} last={r['last_month']}  "
                         f"lag={r['lag_months']}mo  cov@last={r['coverage_at_last']:.0%}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="ours-panel column-coverage guard")
    # Default to the configured build, not a mode name inherited from the reference
    # engine -- a public run has INPUT_MODE = "stage1" and no main_panel_ours.parquet.
    ap.add_argument("--panel", type=Path,
                    default=cfg.PANEL_DIR / f"main_panel_{cfg.INPUT_MODE}.parquet")
    ap.add_argument("--max-lag", type=int, default=1,
                    help="max months a healthy column may end before the panel max (default 1: "
                         "allows forward-difference terminal-edge columns like lib/libd)")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="also fail on columns limited by upstream data (UPSTREAM_LIMITED)")
    args = ap.parse_args()
    ok, report = check_coverage(args.panel, args.max_lag, allow_upstream=not args.strict)
    print(format_report(report))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=1))
    print("[coverage] overall:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

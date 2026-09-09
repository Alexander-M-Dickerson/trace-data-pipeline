# -*- coding: utf-8 -*-
"""
smoke_assertions.py
===================
Cross-stage invariant checks for a smoke run of the pipeline.

Run after run_smoke_test.sh has produced a small stage0 -> stage1 chain, or point it at
any existing output tree. Exits non-zero if any check fails, so it can gate a commit.

These assert against the pipeline's PUBLISHED contract -- the column lists and the
db_type codes as documented in stage1/DATA_DICTIONARY.md -- not against whatever the
code currently happens to do. That distinction is the point: a check derived from the
implementation cannot catch the implementation being wrong.

Usage
-----
    python3 tests/smoke_assertions.py --root smoke
    python3 tests/smoke_assertions.py --root smoke --members enhanced 144a

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# THE PUBLISHED CONTRACT -- keep in step with stage1/DATA_DICTIONARY.md
# ---------------------------------------------------------------------------

# "Source TRACE database: 1=Enhanced, 2=Standard, 3=144A"
# Published in stage1/DATA_DICTIONARY.md, stage1/README_stage1.md and README.md, and
# downstream research keys on db_type == 3 meaning 144A. These codes are a contract:
# a member's code must NEVER depend on how many other members happen to be configured.
CANONICAL_DB_TYPE = {"enhanced": 1, "standard": 2, "144a": 3}

STAGE0_COLUMNS = [
    "cusip_id", "trd_exctn_dt", "prc_ew", "prc_vw", "prc_vw_par", "prc_first",
    "prc_last", "prc_hi", "prc_lo", "trade_count", "time_ew", "time_last",
    "qvolume", "dvolume", "prc_bid", "bid_last", "bid_time_ew", "bid_time_last",
    "prc_ask", "bid_count", "ask_count",
]

STAGE1_COLUMNS = [
    "cusip_id", "permno", "permco", "gvkey", "trd_exctn_dt", "pr", "prfull",
    "acclast", "accpmt", "accall", "ytm", "mod_dur", "mac_dur", "convexity",
    "bond_maturity", "credit_spread", "prc_ew", "prc_vw_par", "prc_first",
    "prc_last", "prc_hi", "prc_lo", "trade_count", "time_ew", "time_last",
    "qvolume", "dvolume", "prc_bid", "bid_last", "bid_time_ew", "bid_time_last",
    "prc_ask", "bid_count", "ask_count", "db_type", "ff12num", "ff17num",
    "ff30num", "bond_age", "bond_amt_outstanding", "sp_rating", "mdy_rating",
    "spc_rating", "mdc_rating",
]

# Standard TRACE is kept only for dates AFTER the last Enhanced date, and the trailing
# DATE_CUT_OFF then falls before that -- so db_type 2 can never reach the stage1 output.
# See CHANGELOG 2.1.0, "Investigated -- no change".
NEVER_SURVIVES_STAGE1 = {"standard"}


class Results:
    """Collects check outcomes and prints one table at the end."""

    def __init__(self) -> None:
        self.rows = []

    def check(self, name, passed, detail=""):
        self.rows.append((name, bool(passed), detail))
        return bool(passed)

    def fail(self, name, detail):
        return self.check(name, False, detail)

    @property
    def ok(self):
        return all(p for _, p, _ in self.rows)

    def report(self):
        width = max((len(n) for n, _, _ in self.rows), default=10)
        print()
        print("=" * (width + 62))
        print("SMOKE TEST RESULTS")
        print("=" * (width + 62))
        for name, passed, detail in self.rows:
            mark = "PASS" if passed else "FAIL"
            print(f"[{mark}] {name:<{width}}  {detail}")
        n_fail = sum(1 for _, p, _ in self.rows if not p)
        print("-" * (width + 62))
        print(f"{len(self.rows)} checks, {n_fail} failed")
        print("=" * (width + 62))


def _latest(paths):
    """The file with the highest YYYYMMDD stamp, mirroring how stage1 resolves it."""
    dated = [(p.stem.split("_")[-1], p) for p in paths]
    dated = [(s, p) for s, p in dated if s.isdigit() and len(s) == 8]
    return max(dated)[1] if dated else None


def find_stage0_daily(root, member):
    """The daily panel, not its siblings.

    trace_{member}_*.parquet also matches trace_{member}_fisd_*.parquet, whose key
    column is complete_cusip rather than cusip_id. Keep only the main file --
    exactly trace_{member}_{YYYYMMDD} -- the same rule
    _stage1_settings.get_latest_stage0_date applies.
    """
    prefix = f"trace_{member}_"
    hits = [
        p for p in (root / "stage0" / member).glob(f"{prefix}*.parquet")
        if p.stem[len(prefix):].isdigit() and len(p.stem[len(prefix):]) == 8
    ]
    return _latest(hits)


def find_stage1(root):
    return _latest(list((root / "stage1" / "data").glob("stage1_*.parquet")))


def check_stage0(root, members, r):
    frames = {}
    for m in members:
        path = find_stage0_daily(root, m)
        if path is None:
            r.fail(f"stage0[{m}] output exists",
                   f"no trace_{m}_*.parquet under {root}/stage0/{m}")
            continue
        df = pd.read_parquet(path)
        frames[m] = df
        r.check(f"stage0[{m}] output exists", True, f"{path.name}  {len(df):,} rows")
        r.check(f"stage0[{m}] non-empty", len(df) > 0, f"{len(df):,} rows")

        # By NAME, not by position. Stage 1 reads columns by name, and the engines
        # genuinely differ in order: the standard/144A aggregation emits bid_time_ew
        # and bid_time_last after ask_count, where enhanced puts them before prc_ask.
        # That is a long-standing wart, not a correctness problem -- so note it, do
        # not fail on it.
        got, want = list(df.columns), STAGE0_COLUMNS
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        same_names = not missing and not extra
        detail = "exact match" if got == want else (
            "same 21 columns, different order (known: standard/144A order bid/ask differently)"
            if same_names else f"missing={missing} extra={extra}")
        r.check(f"stage0[{m}] has the 21 published columns", same_names, detail)

        if m == "enhanced":
            fisd = list((root / "stage0" / m).glob(f"trace_{m}_fisd_*.parquet"))
            r.check("stage0[enhanced] FISD universe file exists", bool(fisd),
                    fisd[0].name if fisd else
                    "MISSING -- stage1 step 3 and the reports job both need it")
    return frames


def check_stage0_audit(root, members, r):
    """The reports job rebuilds within-chunk filter order from ROW ORDER, via
    groupby(chunk).cumcount(). If audit rows are ever written out of order -- which is
    exactly what naive concurrency would do -- every filter-cascade plot in the data
    report is silently mislabelled. So audit ordering is a correctness property."""
    for m in members:
        path = _latest(list((root / "stage0" / m).glob(f"drr_filters_audit_{m}_*.parquet")))
        if path is None:
            r.fail(f"stage0[{m}] audit file exists", f"no drr_filters_audit_{m}_*.parquet")
            continue
        a = pd.read_parquet(path)
        r.check(f"stage0[{m}] audit file exists", True, f"{path.name}  {len(a):,} rows")
        if "chunk" not in a.columns:
            r.fail(f"stage0[{m}] audit carries a chunk column", f"columns={list(a.columns)}")
            continue
        runs = int((a["chunk"] != a["chunk"].shift()).cumsum().nunique())
        distinct = int(a["chunk"].nunique())
        r.check(f"stage0[{m}] audit rows grouped by chunk, not interleaved",
                runs == distinct, f"{distinct} chunks in {runs} contiguous runs")
        r.check(f"stage0[{m}] audit chunk ids ascend",
                bool(a["chunk"].drop_duplicates().is_monotonic_increasing),
                f"first={a['chunk'].iloc[0]} last={a['chunk'].iloc[-1]}")


def check_stage1(root, members, stage0, min_rows, r):
    path = find_stage1(root)
    if path is None:
        r.fail("stage1 output exists", f"no stage1_*.parquet under {root}/stage1/data")
        return
    df = pd.read_parquet(path)
    r.check("stage1 output exists", True, f"{path.name}  {len(df):,} rows")
    r.check("stage1 non-empty", len(df) > 0, f"{len(df):,} rows")

    got, want = list(df.columns), STAGE1_COLUMNS
    r.check("stage1 schema is the 44 published columns", got == want,
            f"{len(got)} columns, exact match" if got == want else
            f"missing={sorted(set(want) - set(got))} extra={sorted(set(got) - set(want))}")

    # ---- the severe one -----------------------------------------------------
    # db_type must be the member's CANONICAL code. Assigning it by POSITION in
    # TRACE_MEMBERS means dropping a member silently renumbers the others -- turning
    # 144A into 2, which the overlap clip then deletes as if it were Standard.
    if "db_type" in df.columns:
        present = set(int(v) for v in pd.unique(df["db_type"].dropna()))
        expected = {CANONICAL_DB_TYPE[m] for m in members if m not in NEVER_SURVIVES_STAGE1}
        unknown = present - set(CANONICAL_DB_TYPE.values())
        r.check("stage1 db_type values are all canonical codes", not unknown,
                f"present={sorted(present)}" if not unknown
                else f"UNKNOWN codes {sorted(unknown)}")
        r.check("stage1 db_type matches the configured members", present == expected,
                f"present={sorted(present)} expected={sorted(expected)}"
                + ("" if present == expected else
                   "  <-- db_type looks positional in TRACE_MEMBERS, not per-member"))
        # Retention, not an absolute count. A member being quietly gutted is the failure
        # mode we care about, and a fixed row floor only detects it at one sample size --
        # this harness is meant to run at 2 chunks, 5 chunks or the full universe.
        # Stage 1's filters legitimately remove some rows, so the bar is deliberately
        # low; what it catches is a member losing MOST of its data.
        for m in sorted(members):
            if m in NEVER_SURVIVES_STAGE1:
                continue
            code = CANONICAL_DB_TYPE[m]
            n = int((df["db_type"] == code).sum())
            r.check(f"stage1 has data for {m} (db_type={code})", n >= min_rows,
                    f"{n:,} rows (floor {min_rows:,})")
            src = stage0.get(m)
            if src is not None and len(src):
                kept = 100.0 * n / len(src)
                r.check(f"stage1 retained most of {m} from stage0", kept >= 40.0,
                        f"{kept:.1f}% of {len(src):,} stage0 rows"
                        + ("" if kept >= 40.0 else
                           "  <-- most of this member was dropped; check db_type and the overlap clip"))

    # ---- structural invariants ----------------------------------------------
    dupes = int(df.duplicated(subset=["cusip_id", "trd_exctn_dt"]).sum())
    r.check("stage1 has no duplicate (cusip_id, trd_exctn_dt)", dupes == 0,
            f"{dupes:,} duplicates")

    if "ff12num" in df.columns:
        n_ff12 = int(df["ff12num"].nunique(dropna=True))
        r.check("stage1 ff12num is not degenerate", n_ff12 > 1,
                f"{n_ff12} distinct values" + ("" if n_ff12 > 1 else
                "  <-- Siccodes12 parse failed; every bond fell into the Other bucket"))

    # Equity-link coverage is checked PER MEMBER, because it differs by an order of
    # magnitude between them and a pooled figure just measures the sample's mix.
    # Measured: Enhanced ~80%, 144A ~25% -- 144A issues are private placements and
    # Rule 144A offerings, which far less often have a listed equity parent. Pooling
    # them made this check fail on a 144A-heavy sample for no good reason.
    # What we actually want to catch is a linker merge that produced nothing, or one
    # that matched everything (which would mean the dated window was ignored).
    if "permno" in df.columns and "db_type" in df.columns:
        PERMNO_BAND = {"enhanced": (50.0, 99.0), "standard": (30.0, 99.0),
                       "144a": (2.0, 90.0)}
        for m in sorted(members):
            if m in NEVER_SURVIVES_STAGE1:
                continue
            sub = df[df["db_type"] == CANONICAL_DB_TYPE[m]]
            if not len(sub):
                continue
            cov = float(sub["permno"].notna().mean()) * 100.0
            lo, hi = PERMNO_BAND.get(m, (2.0, 99.0))
            r.check(f"stage1 permno coverage plausible for {m}", lo <= cov <= hi,
                    f"{cov:.1f}% of {len(sub):,} rows (expected {lo:.0f}-{hi:.0f}%)")

    if "credit_spread" in df.columns:
        cov = float(df["credit_spread"].notna().mean()) * 100.0
        r.check("stage1 credit_spread is mostly populated", cov >= 50.0,
                f"{cov:.1f}% non-null" + ("" if cov >= 50.0 else
                "  <-- the treasury curve may end before the panel"))

    # ---- cross-stage ---------------------------------------------------------
    if stage0:
        s0 = set()
        for d in stage0.values():
            s0 |= set(d["cusip_id"].astype(str).unique())
        s1 = set(df["cusip_id"].astype(str).unique())
        missing = s1 - s0
        r.check("stage1 CUSIPs are a subset of stage0's", not missing,
                f"{len(s1):,} of {len(s0):,}" if not missing else
                f"{len(missing):,} CUSIPs in stage1 that stage0 never produced")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-stage smoke assertions.")
    ap.add_argument("--root", default="smoke", help="root holding stage0/ and stage1/")
    ap.add_argument("--members", nargs="+", default=None,
                    help="TRACE members expected (default: read from config.py)")
    ap.add_argument("--min-rows", type=int, default=1000,
                    help="floor for per-db_type row counts (raise for a full run)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    members = args.members
    if members is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config import TRACE_MEMBERS
        members = list(TRACE_MEMBERS)

    print(f"root    : {root}")
    print(f"members : {members}")
    print(f"min rows: {args.min_rows:,} per db_type")

    r = Results()
    if not root.exists():
        r.fail("pipeline root exists", str(root))
        r.report()
        return 1

    stage0 = check_stage0(root, members, r)
    check_stage0_audit(root, members, r)
    check_stage1(root, members, stage0, args.min_rows, r)
    r.report()
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

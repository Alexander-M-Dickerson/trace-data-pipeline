# -*- coding: utf-8 -*-
"""
test_merge_keys.py
==================
Every lookup stage 1 joins onto the bond-day panel must have ONE ROW PER KEY.

This exists because of a specific, expensive failure. On 2026-09-09 a full run lost
~1.5 hours of stage-1 work to this:

    RuntimeError: 268 duplicate (cusip_id, trd_exctn_dt) rows after the linker merge.

The linker was innocent. `fisd.fisd_mergedissue` is one row per ISSUE, not per CUSIP,
and CUSIP 29357JAC0 carries two issue records -- an ABS row and a CDEB row with the same
issuer, maturity and coupon. Step 6 left-joined that frame on `cusip_id` to fill missing
offering amounts, which duplicated every one of that bond's 268 surviving bond-days. The
panel is not re-checked for uniqueness between step 2 and step 7, so the abort landed
five steps and an hour and a half downstream of its cause, pointing at the wrong thing.

The smoke test cannot catch this. It asserts panel uniqueness and passed 28/28 -- a
four-chunk sample simply contains no multi-issue CUSIP. The property that matters is not
"is the output unique" but "is every lookup keyed the way the join assumes", and that is
checkable in seconds against the source tables.

    python3 tests/test_merge_keys.py                 # needs WRDS
    python3 tests/test_merge_keys.py --offline       # skips the WRDS checks

Author: Open Source Bond Asset Pricing
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage1"))

FAILURES = []
SKIPPED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name:<58} {detail}")
    if not cond:
        FAILURES.append(name)


def note(name, detail, skipped=False):
    """Report something worth knowing that is NOT a failure."""
    print(f"[note] {name:<58} {detail}")
    if skipped:
        SKIPPED.append(name)


# ---------------------------------------------------------------- offline checks
def check_stage0_artifact():
    """The stage-0 FISD universe must be one row per CUSIP -- steps 3, 4 and 6 join it."""
    found = sorted(ROOT.glob("**/stage0/enhanced/trace_enhanced_fisd_*.parquet"))
    if not found:
        note("stage0 FISD artifact", "none on disk; skipped (run stage 0 or the smoke test)",
             skipped=True)
        return
    f = found[-1]
    d = pd.read_parquet(f, columns=["complete_cusip"])
    dup = int(d["complete_cusip"].duplicated().sum())
    check("stage0 FISD artifact is one row per CUSIP", dup == 0,
          f"{len(d):,} rows, {dup} duplicate(s) -- {f.name}")


def _load_fn(name, extra_globals=None):
    """Pull one function out of the shipped source and make it callable.

    stage1_pipeline.py cannot be imported on its own -- create_daily_stage1.py injects
    LOG_DIR and friends into its globals before exec'ing it. Rather than test a copy of
    the function, lift the real definition out of the file with ast and exec just that.
    A copy would drift; this cannot.
    """
    import ast
    src = (ROOT / "stage1" / "stage1_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"pd": pd}
            ns.update(extra_globals or {})
            exec(compile(ast.Module([node], []), f"<{name}>", "exec"), ns)
            return ns[name]
    return None


def check_merge_helper():
    """_merge_1to1 must refuse a duplicated lookup, and name the key."""
    fn = _load_fn("_merge_1to1")
    if fn is None:
        check("_merge_1to1 exists in stage1_pipeline.py", False, "not found")
        return

    class SP:
        _merge_1to1 = staticmethod(fn)

    panel = pd.DataFrame({"cusip_id": ["A", "A", "B"], "v": [1, 2, 3]})
    good = pd.DataFrame({"cusip_id": ["A", "B"], "x": [10, 20]})
    bad = pd.DataFrame({"cusip_id": ["A", "A", "B"], "x": [10, 11, 20]})

    out = SP._merge_1to1(panel, good, on="cusip_id", label="clean lookup")
    check("_merge_1to1 preserves row count on a clean lookup", len(out) == len(panel),
          f"{len(panel)} -> {len(out)}")

    try:
        SP._merge_1to1(panel, bad, on="cusip_id", label="dupe lookup")
        check("_merge_1to1 rejects a duplicated lookup", False, "no exception raised")
    except ValueError as e:
        msg = str(e)
        check("_merge_1to1 rejects a duplicated lookup", True, type(e).__name__)
        check("  ...and names the offending key in the message", "'A'" in msg or '"A"' in msg,
              msg[:70])


def check_the_actual_bug(issues_for_amt, panel_days=304):
    """Reproduce the 2026-09-09 failure on real FISD rows, then show the fix kills it.

    This is the check that would have caught the bug. It does not need a pipeline run --
    only the lookup frame and a panel of one bond's trading days.
    """
    dupd = issues_for_amt["cusip_id"].duplicated(keep=False)
    if not dupd.any():
        note("fan-out reproduction", "no multi-issue CUSIP in this FISD vintage; skipped")
        return
    cusip = issues_for_amt.loc[dupd, "cusip_id"].iloc[0]
    n_records = int((issues_for_amt["cusip_id"] == cusip).sum())

    # A panel standing in for final_df: one row per bond-day, as stage 0 produces.
    panel = pd.DataFrame({
        "cusip_id": [cusip] * panel_days,
        "trd_exctn_dt": pd.date_range("2023-01-02", periods=panel_days, freq="B"),
    })

    # BEFORE: the shipped merge, straight onto the issue-keyed frame.
    before = panel.merge(issues_for_amt[["cusip_id", "offering_amt"]],
                         on="cusip_id", how="left")
    check("the old merge DOES fan out (bug reproduced)",
          len(before) == panel_days * n_records,
          f"{cusip}: {panel_days} bond-days x {n_records} issue records -> {len(before):,} rows")

    # AFTER: the shipped collapse + guarded merge.
    collapse = _load_fn("_offering_amt_by_cusip")
    merge1to1 = _load_fn("_merge_1to1")
    if collapse is None or merge1to1 is None:
        check("the fix is present in stage1_pipeline.py", False, "helper(s) not found")
        return
    after = merge1to1(panel, collapse(issues_for_amt), on="cusip_id", label="offering_amt")
    check("the fix leaves the row count unchanged", len(after) == panel_days,
          f"{panel_days} -> {len(after)}")
    check("  ...and still assigns an offering_amt", after["offering_amt"].notna().all(),
          f"{int(after['offering_amt'].notna().sum())}/{len(after)} populated")


# ---------------------------------------------------------------- WRDS checks
def _connect_wrds_or_none(user):
    """Connect WITHOUT ever prompting, and report a failure in one line.

    ❗The wrds package answers a failed connect by calling input() for a username and
    password. On a login node that does not raise -- it silently consumes whatever is
    next in the terminal's paste buffer, which is usually the next command you typed.
    That produced, verbatim:

        FATAL: PAM authentication failed for user
               "[phd18ad1@wrds-sbd-cloud-login-05-w ~]$ cd ~ && rm -rf trace-da"

    followed by ~120 lines of SQLAlchemy traceback. So stdin is closed for the duration
    of the connect: the prompt then hits EOF and raises, and we say what to check.
    """
    import contextlib, io, wrds
    saved, devnull = sys.stdin, open(os.devnull)
    chatter = io.StringIO()          # "Loading library list...", and the prompt text
    try:
        sys.stdin = devnull
        with contextlib.redirect_stdout(chatter):
            return wrds.Connection(wrds_username=user)
    except Exception as e:
        first = (str(e).strip().splitlines() or [type(e).__name__])[0][:110]
        note("WRDS checks", "SKIPPED -- could not connect", skipped=True)
        print(f"       {first}")
        print(f"       Check ~/.pgpass has a line for "
              f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{user} and is chmod 600.")
        return None
    finally:
        sys.stdin = saved
        devnull.close()


def check_wrds():
    """The source tables, keyed as each join assumes. Aggregates only -- seconds."""
    user = os.environ.get("WRDS_USERNAME", "")
    if not user or user == "your_wrds_username":
        note("WRDS checks", "WRDS_USERNAME not set; skipped (--offline to silence)",
             skipped=True)
        return
    db = _connect_wrds_or_none(user)
    if db is None:
        return
    try:
        # Joined on issue_id: the ratings->cusip map, and the call dummies.
        for table, key in (("fisd_mergedissue", "issue_id"),
                           ("fisd_mergedredemption", "issue_id")):
            r = db.raw_sql(f"""
                SELECT COUNT(*) AS n, COUNT(DISTINCT {key}) AS k
                FROM fisd.{table} WHERE {key} IS NOT NULL
            """).iloc[0]
            check(f"fisd.{table} is one row per {key}", int(r["n"]) == int(r["k"]),
                  f"{int(r['n']):,} rows / {int(r['k']):,} keys")

        # The CUSIPs carrying more than one issue record. This is the whole input to the
        # reproduction below, and it is a handful of rows -- the previous version pulled
        # all 773,423 rows of the table to find them, which is why this test was slow.
        dups = db.raw_sql("""
            SELECT complete_cusip AS cusip_id, COUNT(*) AS n_issues
            FROM fisd.fisd_mergedissue
            WHERE complete_cusip IS NOT NULL
            GROUP BY complete_cusip HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
        """)
        note("CUSIPs with >1 FISD issue record",
             f"{len(dups)} (handled: the offering_amt lookup collapses them)")
        if len(dups) == 0:
            note("fan-out reproduction", "nothing to reproduce in this FISD vintage")
            return

        # Fetch ONLY those CUSIPs' rows -- two rows, not three quarters of a million.
        cusips = tuple(dups["cusip_id"].tolist())
        issues_for_amt = db.raw_sql(
            "SELECT complete_cusip AS cusip_id, offering_amt "
            "FROM fisd.fisd_mergedissue WHERE complete_cusip IN %(c)s",
            params={"c": cusips})
        check_the_actual_bug(issues_for_amt)
    finally:
        try:
            db.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip the WRDS checks")
    args = ap.parse_args()

    check_merge_helper()
    check_stage0_artifact()
    if not args.offline:
        check_wrds()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    if SKIPPED:
        # A skip is not a pass. Say so, so nobody reads a green line as coverage.
        print(f"merge-key checks passed, but {len(SKIPPED)} SKIPPED: {SKIPPED}")
        return 0
    print("all merge-key checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

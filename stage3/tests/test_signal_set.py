"""The signals every section sorts are the spec's Cluster rows, and nothing else ever is.

    python -m pytest stage3/tests/test_signal_set.py -q

Why this exists. Until 2026-09-27 the Section 3 census took every panel column that was NOT on a
hand-written list of identifiers. The 2026 panel gained five Treasury benchmark returns
(`tret_bns`, `tret_cfm`, `tret_gprs`, `tret_cls`, `tret_mat`) that the list did not name, so the
census sorted 113 "signals" and Table B.1 counted five Treasury returns among them. Nothing
failed. These tests make the same mistake impossible to repeat quietly: the signal list is read
from one place, every other list must equal it, an unclassified column stops the run, and no
stage 3 code may choose its signals by leaving columns out.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

STAGE3 = Path(__file__).resolve().parents[1]
REPO = STAGE3.parent
for p in (STAGE3, STAGE3 / "s4_zoo", STAGE3 / "s3_nse"):
    sys.path.insert(0, str(p))

import signal_set as SIG  # noqa: E402


def test_the_signals_are_the_specs_cluster_rows_and_every_list_agrees():
    import clusters as NSE
    import zoo_engine as Z
    assert len(SIG.SIGNALS) == 108
    assert set(Z.CLUSTER_OF) == set(SIG.SIGNALS), "the factor zoo's cluster map"
    assert set(NSE.ALL_SIGNALS) == set(SIG.SIGNALS), "Section 5's cluster map"
    stage4 = REPO / "stage4" / "spec" / "factors.json"
    if stage4.exists():
        names = json.loads(stage4.read_text(encoding="utf-8"))["signals"]["names"]
        assert set(names) == set(SIG.SIGNALS), "stage 4's signal list"


def test_no_return_benchmark_or_identifier_is_a_signal():
    returns = {c for c in SIG.NOT_SIGNALS | set(SIG.SIGNALS)
               if c.startswith(("tret", "ret_", "rf")) or c in ("cusip", "date", "permno")}
    assert {"tret", "tret_bns", "tret_cfm", "tret_gprs", "tret_cls", "tret_mat"} <= returns
    assert not (returns & set(SIG.SIGNALS)), sorted(returns & set(SIG.SIGNALS))


def test_every_panel_column_is_classified():
    sys.path.insert(0, str(REPO / "stage2"))
    try:
        from lib.contract import PANEL_COLUMNS
    except ImportError:
        pytest.skip("stage 2 is not beside stage 3 in this checkout")
    assert set(PANEL_COLUMNS) == set(SIG.SIGNALS) | SIG.NOT_SIGNALS
    chosen = SIG.select(PANEL_COLUMNS)
    assert len(chosen) == 108 and not [c for c in chosen if c.startswith("tret")]


def test_an_unclassified_column_stops_the_run():
    cols = list(SIG.SIGNALS) + sorted(SIG.NOT_SIGNALS)
    assert len(SIG.select(cols)) == 108
    with pytest.raises(ValueError, match="does not classify"):
        SIG.select(cols + ["tret_new_benchmark"])
    with pytest.raises(ValueError, match="lacks 1"):
        SIG.select([c for c in cols if c != "cs"])


def test_a_census_that_is_not_the_signals_is_refused():
    good = [s + "*" if i % 7 == 0 else s for i, s in enumerate(SIG.SIGNALS)]
    SIG.check_census(good, what="a census")
    # the names a real census writes: the swapped-in twin, the within-firm suffix, the flip
    real = [f"{s}_mmn_wf*" if i % 3 == 0 else f"{s}_wf" for i, s in enumerate(SIG.SIGNALS)]
    SIG.check_census(real, what="a within-firm census")
    assert SIG.signal_of("cs_mmn*") == "cs" and SIG.signal_of("age_wf") == "age"
    with pytest.raises(ValueError, match="not signals"):
        SIG.check_census(good + ["tret_bns*"], what="a census with a benchmark")
    with pytest.raises(ValueError, match="missing"):
        SIG.check_census(good[1:], what="a census short of one")


def test_no_stage3_code_chooses_signals_by_leaving_columns_out():
    """The pattern that let the benchmarks in: signals = the columns NOT on some list."""
    bad = []
    for f in STAGE3.rglob("*.py"):
        if "tests" in f.parts or f.name == "signal_set.py":
            continue
        text = f.read_text(encoding="utf-8")
        if re.search(r"\bID_COLUMNS\b", text) or re.search(
                r"for \w+ in \w+\.columns\s+if \w+ not in [A-Z_]{3,}", text):
            bad.append(str(f.relative_to(STAGE3)))
    assert not bad, (
        f"choose signals with signal_set.select(), never by excluding a list: {bad}")

# -*- coding: utf-8 -*-
"""
test_linker_window.py
=====================
Stage 1 attaches permno / permco / gvkey from the bond-firm linker. This pins WHICH of the
linker's two dated windows it joins, and what the join does at the edges.

Why it exists. Until 2026-09 Stage 1 joined the EVIDENCE window (w0/w1) while Stage 2 joined
the IDENTITY window (i0/i1). Same linker, different question, and the daily and monthly panels
named a different firm on about 2% of bond-months. Nothing failed. Both stages now join the
identity window, and these tests fail if Stage 1 drifts back.

No WRDS, no network: every frame here is built by hand.

    python -m pytest tests/test_linker_window.py -q

What this does NOT check: that the linker's content is right. That is the linker's own
release verification. This checks the JOIN.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "stage1"))

import _linker_join as lj  # noqa: E402

IDENTITY = ("i0", "i1")
EVIDENCE = ("w0", "w1")


def _linker() -> pd.DataFrame:
    """Two bonds.

    AAA: one owner. Provable 2005-01-01..2010-06-30 (the firm delisted), but nothing
         contradicts the label, so the identity window is open-ended either side.
    BBB: two owners. Firm 111 until 2012-03-15, firm 222 from 2012-03-16.
    """
    return pd.DataFrame({
        "cusip9": ["AAA", "BBB", "BBB"],
        "w0": ["2005-01-01", "2006-01-01", "2012-03-16"],
        "w1": ["2010-06-30", "2012-03-15", "2020-12-31"],
        "i0": ["1900-01-01", "1900-01-01", "2012-03-16"],
        "i1": ["2099-12-31", "2012-03-15", "2099-12-31"],
        "permno": [10001, 111, 222],
        "permco": [501, 511, 522],
        "gvkey": ["001234", "005678", "009999"],
        "confidence": ["high", "high", "medium"],
        "window_src": ["fisd_legal"] * 3,
        "rung": ["R1_cusip6_overlap"] * 3,
    })


def _panel(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["cusip_id", "trd_exctn_dt"])
    df["trd_exctn_dt"] = pd.to_datetime(df["trd_exctn_dt"])
    df["pr"] = 100.0
    return df


def _join(panel, window=IDENTITY):
    out, n = lj.attach_firm_ids(panel, lj.prepare_linker(_linker(), window), window)
    return out.sort_values(["cusip_id", "trd_exctn_dt"]).reset_index(drop=True), n


def test_the_default_window_is_identity():
    sys.path.insert(0, str(ROOT))
    import importlib
    s = importlib.import_module("_stage1_settings")
    assert tuple(s.LINKER_WINDOW) == IDENTITY, (
        "Stage 1 must join the identity window, the same one Stage 2 joins "
        "(stage2/_stage2_settings.LINKER_WINDOW). Joining w0/w1 here brings back the "
        "daily/monthly permno disagreement.")


def test_stage1_and_stage2_declare_the_same_window():
    import ast
    def declared(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "LINKER_WINDOW" for t in node.targets):
                return tuple(ast.literal_eval(node.value))
        raise AssertionError(f"no LINKER_WINDOW in {path}")
    s1 = declared(ROOT / "stage1" / "_stage1_settings.py")
    s2 = declared(ROOT / "stage2" / "_stage2_settings.py")
    assert s1 == s2, f"stage 1 joins {s1} and stage 2 joins {s2}: the panels will disagree"


def test_a_bond_that_outlives_its_firm_keeps_its_label():
    out, _ = _join(_panel([("AAA", "2004-06-01"), ("AAA", "2008-01-02"), ("AAA", "2015-09-30")]))
    assert out["permno"].tolist() == [10001, 10001, 10001]
    assert out["permco"].tolist() == [501, 501, 501]


def test_the_evidence_window_would_have_dropped_those_labels():
    """The property the change exists for, stated as a contrast."""
    out, n = _join(_panel([("AAA", "2004-06-01"), ("AAA", "2008-01-02"), ("AAA", "2015-09-30")]),
                   window=EVIDENCE)
    assert out["permno"].isna().tolist() == [True, False, True]
    assert n == 1                      # the 2015 row matched a window that had closed


def test_identity_never_changes_an_id_the_evidence_join_gives():
    days = [("AAA", "2008-01-02"), ("BBB", "2007-05-05"), ("BBB", "2012-03-15"),
            ("BBB", "2012-03-16"), ("BBB", "2019-01-01")]
    ev, _ = _join(_panel(days), window=EVIDENCE)
    idw, _ = _join(_panel(days), window=IDENTITY)
    have = ev["permno"].notna()
    assert (ev.loc[have, "permno"] == idw.loc[have, "permno"]).all()
    assert idw["permno"].notna().sum() >= have.sum()


def test_two_owners_do_not_fan_out_and_the_date_picks_the_owner():
    days = [("BBB", "2007-05-05"), ("BBB", "2012-03-15"), ("BBB", "2012-03-16"),
            ("BBB", "2030-01-01")]
    out, _ = _join(_panel(days))
    assert len(out) == len(days)
    assert out["permno"].tolist() == [111, 111, 222, 222]


def test_a_date_before_every_window_gets_no_id():
    lk = _linker()
    lk.loc[lk["cusip9"] == "AAA", "i0"] = "2005-01-01"
    out, _ = lj.attach_firm_ids(_panel([("AAA", "2004-12-31"), ("AAA", "2005-01-01")]),
                                lj.prepare_linker(lk, IDENTITY), IDENTITY)
    out = out.sort_values("trd_exctn_dt")
    assert out["permno"].isna().tolist() == [True, False]


def test_a_bond_absent_from_the_linker_gets_no_id():
    out, _ = _join(_panel([("ZZZ", "2010-01-04")]))
    assert out["permno"].isna().all()
    assert len(out) == 1


def test_a_linker_without_the_identity_window_is_refused():
    """The July 2026 bundle had no i0/i1. Falling back to w0/w1 would be silent and wrong."""
    old = _linker().drop(columns=["i0", "i1"])
    with pytest.raises(ValueError, match="i0"):
        lj.prepare_linker(old, IDENTITY)


def test_overlapping_windows_within_a_bond_are_refused():
    lk = pd.concat([_linker(), _linker().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        lj.prepare_linker(lk, IDENTITY)


def test_category_keys_and_mixed_datetime_resolution_still_match():
    """A category `by` key against a str matches nothing, and [us] vs [ns] raises."""
    p = _panel([("AAA", "2008-01-02"), ("BBB", "2013-01-02")])
    p["cusip_id"] = p["cusip_id"].astype("category")
    p["trd_exctn_dt"] = p["trd_exctn_dt"].astype("datetime64[us]")
    out, _ = _join(p)
    assert out["permno"].tolist() == [10001, 222]


def test_the_audit_columns_are_not_carried_into_the_panel():
    out, _ = _join(_panel([("AAA", "2008-01-02")]))
    for c in ("confidence", "window_src", "rung", "i0", "i1", "w0", "w1"):
        assert c not in out.columns
    assert out["gvkey"].tolist() == [1234]

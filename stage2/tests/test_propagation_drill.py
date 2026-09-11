# -*- coding: utf-8 -*-
"""
test_propagation_drill.py
=========================
Adding a variable has to reach all three panels, or fail loudly. This is the drill: for each
step of that route, SKIP it and prove the matching gate fails.

The failure this exists for is measured, not imagined. `main_panel_ours_plus.parquet` carries
four variables beyond the golden 140 -- `ff12num`, `hm_hs`, `hm_hs_inst`, `basrev` -- and **zero
of them reach the shipped 1973-current panel**. Adding a variable propagated to one dataset out
of three, silently, because nothing checked.

    step skipped                                   gate that must fail
    ---------------------------------------------  --------------------------------
    added to the panel, not to the contract        assert_panel_contract
    added to the contract, not to the panel        assert_panel_contract
    added in the wrong POSITION                    assert_panel_contract
    not carried across to the pre-TRACE engine     assert_pre_trace_coverage
    price/return signal shipped without its twin   assert_mmn_twins
    a fixed-end panel's frontier moved             assert_frontier_policy

Two gates are not unit-testable and live elsewhere: `engine_drift.py` in the private repo
(proved by injecting a new function and a changed body) and `certify_vintage.py`, which needs
the real multi-GB panels. `tools/audit_panels.py` runs both.

The negative cases matter as much: a `trace_only` column empty before 2002 is CORRECT, and a
gate that fires on it would be muted within a month.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STAGE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE2))

from lib import contract as C  # noqa: E402

SEAM = "2002-07-31"


def _panel(columns, *, pre_rows: int = 6, post_rows: int = 6) -> pd.DataFrame:
    """A frame with the given columns spanning both sides of the seam."""
    dates = list(pd.date_range("2000-01-31", periods=pre_rows, freq="ME")) + \
        list(pd.date_range("2003-01-31", periods=post_rows, freq="ME"))
    n = len(dates)
    columns = list(dict.fromkeys(columns))          # `date` is already in PANEL_COLUMNS
    if "date" not in columns:
        columns = ["date"] + columns
    df = pd.DataFrame({"date": dates})
    for c in columns:
        if c == "date":
            continue
        df[c] = np.arange(n, dtype="float64") if c not in ("cusip", "issuer_cusip",
                                                           "ret_type", "country") \
            else [f"{i:09d}" for i in range(n)]
    return df[list(columns)]


# --------------------------------------------------------------------------- the classification
def test_the_two_classes_partition_the_contract():
    assert len(C.TRACE_ONLY_COLUMNS) + len(C.ALL_PANEL_COLUMNS) == len(C.PANEL_COLUMNS)
    assert not (set(C.ALL_PANEL_COLUMNS) & C.TRACE_ONLY_COLUMNS)


def test_every_trace_only_column_has_a_stated_reason():
    for c in C.TRACE_ONLY_COLUMNS:
        assert C.trace_only_reason(c), f"{c} is trace_only with no reason on record"


def test_every_trace_only_column_is_in_the_contract():
    """A classification naming a column the panel does not have is drift, not documentation."""
    unknown = sorted(C.TRACE_ONLY_COLUMNS - set(C.PANEL_COLUMNS))
    assert not unknown, f"classified but not in the panel: {unknown}"


# --------------------------------------------------------------------------- step 1: the contract
def test_a_column_added_to_the_panel_but_not_the_contract_fails():
    df = _panel(list(C.PANEL_COLUMNS) + ["brand_new_signal"])
    with pytest.raises(AssertionError):
        C.assert_panel_contract(df)


def test_a_column_added_to_the_contract_but_not_the_panel_fails():
    df = _panel([c for c in C.PANEL_COLUMNS if c != "b_defb"])
    with pytest.raises(AssertionError):
        C.assert_panel_contract(df)


def test_the_right_columns_in_the_wrong_order_fails():
    cols = list(C.PANEL_COLUMNS)
    i = cols.index("b_defb")
    cols[i], cols[i + 1] = cols[i + 1], cols[i]        # the b_defb/b_termb swap, exactly
    with pytest.raises(AssertionError):
        C.assert_panel_contract(_panel(cols))


def test_the_real_contract_passes_itself():
    C.assert_panel_contract(_panel(list(C.PANEL_COLUMNS)))


# --------------------------------------------------------------------------- step 2: transplant
def test_an_all_panels_column_with_no_pre_trace_values_fails():
    """The subtle one: it exists everywhere and is empty before 2002."""
    df = _panel(list(C.ALL_PANEL_COLUMNS))
    df.loc[pd.to_datetime(df["date"]) <= SEAM, "b_defb"] = np.nan
    with pytest.raises(AssertionError, match="b_defb"):
        C.assert_pre_trace_coverage(df)


def test_the_message_says_where_to_look():
    df = _panel(list(C.ALL_PANEL_COLUMNS))
    df.loc[pd.to_datetime(df["date"]) <= SEAM, "b_defb"] = np.nan
    try:
        C.assert_pre_trace_coverage(df)
    except AssertionError as exc:
        msg = str(exc)
    assert "engine_drift" in msg, "the fix is a transplant; say so"
    assert "TRACE_ONLY" in msg, "the other fix is to classify it; say so"


def test_a_trace_only_column_empty_before_the_seam_is_FINE():
    """The negative case. Firing here would mute the gate within a month."""
    df = _panel(list(C.PANEL_COLUMNS))
    for c in C.TRACE_ONLY_COLUMNS:
        df.loc[pd.to_datetime(df["date"]) <= SEAM, c] = np.nan
    C.assert_pre_trace_coverage(df)


def test_a_panel_with_no_pre_trace_rows_is_not_flagged():
    """Panel 1 is TRACE-era only; the gate is about the COMBINED panel."""
    df = _panel(list(C.ALL_PANEL_COLUMNS), pre_rows=0)
    C.assert_pre_trace_coverage(df)


# --------------------------------------------------------------------------- step 3: the MMN twin
def test_a_signal_without_its_unadjusted_twin_fails():
    sidecar = [f"{c}_mmn" for c in C.MMN_TWINNED if c != "str"]
    with pytest.raises(AssertionError, match="str_mmn"):
        C.assert_mmn_twins(sidecar)


def test_a_complete_sidecar_passes():
    C.assert_mmn_twins([f"{c}_mmn" for c in C.MMN_TWINNED])


def test_every_twinned_signal_is_a_real_panel_column():
    unknown = sorted(C.MMN_TWINNED - set(C.PANEL_COLUMNS))
    assert not unknown, f"twinned but not in the panel: {unknown}"


# --------------------------------------------------------------------------- step 4: the frontier
def test_a_fixed_end_panel_that_moved_fails():
    df = pd.DataFrame({"date": pd.to_datetime(["1973-01-31", "2023-06-30"])})
    with pytest.raises(AssertionError, match="2023-01-31"):
        C.assert_frontier_policy(df, "pre_trace")


def test_a_fixed_end_panel_that_held_passes():
    df = pd.DataFrame({"date": pd.to_datetime(["1973-01-31", C.PRE_TRACE_FIXED_END])})
    C.assert_frontier_policy(df, "pre_trace")


@pytest.mark.parametrize("dataset", ["trace", "combined"])
def test_a_vintage_tracking_panel_is_never_pinned(dataset):
    """Panels 1 and 2 move every year; pinning them would fail every vintage."""
    df = pd.DataFrame({"date": pd.to_datetime(["2002-08-31", "2099-12-31"])})
    C.assert_frontier_policy(df, dataset)


# --------------------------------------------------------------------------- naming
def test_external_names_roll_with_the_data():
    assert C.external_name("combined", "standard", "1973-01-31", "2025-11-30") == \
        "lehman_ice_trace_1973_2025_std.parquet"
    assert C.external_name("combined", "standard", "1973-01-31", "2026-11-30") == \
        "lehman_ice_trace_1973_2026_std.parquet"


def test_both_return_spellings_give_the_same_tag():
    assert C.external_name("pre_trace", "duration_adj", "1973-01-31", "2023-01-31") == \
        C.external_name("pre_trace", "dur_adj", "1973-01-31", "2023-01-31")


def test_only_the_trace_panel_is_published():
    published = [k for k, v in C.DATASETS.items() if v["published"]]
    assert published == ["trace"], "the other two carry firm ids derived from private data"

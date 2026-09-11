# -*- coding: utf-8 -*-
"""
test_propagation_drill.py
=========================
Adding a column to the panel touches four things: the code that computes it, the frozen
contract, the `_mmn` sidecar, and the dictionary. Skip any one and the build should fail. This
drill skips each in turn and proves the matching gate fires.

    step skipped                                   gate that must fail
    ---------------------------------------------  --------------------------------
    added to the panel, not to the contract        assert_panel_contract
    added to the contract, not to the panel        assert_panel_contract
    added in the wrong POSITION                    assert_panel_contract
    price-based signal shipped without its twin    assert_mmn_twins

The position case is not hypothetical. Collapsing the degenerate DEF and TERM regressions into
one two-factor model swapped `b_defb` and `b_termb` in the output, because emission follows the
model's `keep` order -- the same 140 columns in a different file, which nothing would have
caught until a downstream positional read broke.

`test_column_contract.py` covers the fourth step (panel = report = dictionary); it needs a built
panel, so most of it skips on a fresh clone.

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


def _panel(columns) -> pd.DataFrame:
    """A frame carrying exactly `columns`, in that order."""
    columns = list(dict.fromkeys(columns))
    if "date" not in columns:
        columns = ["date"] + columns
    dates = pd.date_range("2003-01-31", periods=8, freq="ME")
    n = len(dates)
    data = {"date": dates}
    for c in columns:
        if c == "date":
            continue
        data[c] = ([f"{i:09d}" for i in range(n)]
                   if c in ("cusip", "issuer_cusip", "ret_type", "country")
                   else np.arange(n, dtype="float64"))
    return pd.DataFrame(data)[columns]


# --------------------------------------------------------------- what a column needs
def test_the_two_groups_partition_the_contract():
    assert len(C.NEEDS_TRADE_LEVEL_DATA) + len(C.FROM_MONTH_END_ONLY) == len(C.PANEL_COLUMNS)
    assert not (set(C.FROM_MONTH_END_ONLY) & C.NEEDS_TRADE_LEVEL_DATA)


def test_every_classified_column_says_what_it_needs():
    for c in C.NEEDS_TRADE_LEVEL_DATA:
        assert C.column_input(c), f"{c} is classified with no input on record"


def test_every_classified_column_is_in_the_contract():
    """A classification naming a column the panel does not have is drift, not documentation."""
    unknown = sorted(C.NEEDS_TRADE_LEVEL_DATA - set(C.PANEL_COLUMNS))
    assert not unknown, f"classified but not in the panel: {unknown}"


def test_a_month_end_column_reports_no_special_input():
    assert C.column_input("ret_vw") is None
    assert C.column_input("ytm") is None


def test_the_groups_do_not_overlap_each_other():
    seen: set[str] = set()
    for what, cols in C.COLUMN_INPUTS.items():
        clash = seen & set(cols)
        assert not clash, f"{what} repeats {sorted(clash)}"
        seen |= set(cols)


# --------------------------------------------------------------- step: the contract
def test_a_column_added_to_the_panel_but_not_the_contract_fails():
    with pytest.raises(AssertionError):
        C.assert_panel_contract(_panel(list(C.PANEL_COLUMNS) + ["brand_new_signal"]))


def test_a_column_added_to_the_contract_but_not_the_panel_fails():
    with pytest.raises(AssertionError):
        C.assert_panel_contract(_panel([c for c in C.PANEL_COLUMNS if c != "b_defb"]))


def test_the_right_columns_in_the_wrong_order_fails():
    cols = list(C.PANEL_COLUMNS)
    i = cols.index("b_defb")
    cols[i], cols[i + 1] = cols[i + 1], cols[i]        # the b_defb/b_termb swap, exactly
    with pytest.raises(AssertionError):
        C.assert_panel_contract(_panel(cols))


def test_the_real_contract_passes_itself():
    C.assert_panel_contract(_panel(list(C.PANEL_COLUMNS)))


# --------------------------------------------------------------- step: the MMN twin
def test_a_signal_without_its_unadjusted_twin_fails():
    sidecar = [f"{c}_mmn" for c in C.MMN_TWINNED if c != "str"]
    with pytest.raises(AssertionError, match="str_mmn"):
        C.assert_mmn_twins(sidecar)


def test_a_complete_sidecar_passes():
    C.assert_mmn_twins([f"{c}_mmn" for c in C.MMN_TWINNED])


def test_every_twinned_signal_is_a_real_panel_column():
    unknown = sorted(C.MMN_TWINNED - set(C.PANEL_COLUMNS))
    assert not unknown, f"twinned but not in the panel: {unknown}"


def test_the_message_names_what_is_missing():
    try:
        C.assert_mmn_twins([f"{c}_mmn" for c in C.MMN_TWINNED if c != "cs"])
    except AssertionError as exc:
        msg = str(exc)
    assert "cs_mmn" in msg
    assert "ret_vw_bgn" in msg, "say which return the twin pairs with"

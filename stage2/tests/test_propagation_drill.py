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


def test_an_undeclared_twin_in_the_sidecar_fails():
    """D25: a NEW price-based signal nobody added to MMN_TWINNED must not slip past.

    The twin gate iterates the declared set, so a name outside it is invisible to it. This is
    the other direction: the sidecar is checked against the set, not only the set against the
    sidecar.
    """
    sidecar = [f"{c}_mmn" for c in C.MMN_TWINNED] + ["brandnew_mmn"]
    with pytest.raises(AssertionError, match="brandnew"):
        C.assert_mmn_twins_are_declared(sidecar)


def test_a_sidecar_of_only_declared_twins_passes():
    C.assert_mmn_twins_are_declared([f"{c}_mmn" for c in C.MMN_TWINNED])


# ----------------------------------------------- step: WHICH form is in the main panel
def _mmn_pair(panel_values, sidecar_values, col="cs"):
    """One twinned signal in both frames, keyed so the assertion can join them."""
    n = len(panel_values)
    keys = {"cusip": [f"{i:09d}" for i in range(n)],
            "date": pd.date_range("2010-01-31", periods=n, freq="ME")}
    return (pd.DataFrame({**keys, col: panel_values}),
            pd.DataFrame({**keys, f"{col}_mmn": sidecar_values}))


def test_the_unadjusted_form_in_the_main_panel_fails():
    """D20: `assert_mmn_twins` passes happily when the WRONG form is in the panel.

    This is the basrev v1 failure -- AR(1) -0.05 adjusted vs -0.22 unadjusted, four fifths of
    the raw reversal being bid-ask bounce -- and it produces a complete, normal-looking panel.
    """
    vals = [0.01 * i for i in range(2000)]
    panel, sidecar = _mmn_pair(vals, vals)          # the panel IS the sidecar's column
    with pytest.raises(AssertionError, match="UNADJUSTED"):
        C.assert_main_panel_is_adjusted(panel, sidecar)


def test_the_adjusted_form_passes_even_when_many_rows_coincide():
    """Coincidence is normal and must NOT fire: `lix` matches on 23.8% of real rows."""
    n = 2000
    panel_v = [0.01 * i for i in range(n)]
    side_v = [v if i % 4 else v + 0.5 for i, v in enumerate(panel_v)]   # 75% identical
    panel, sidecar = _mmn_pair(panel_v, side_v)
    C.assert_main_panel_is_adjusted(panel, sidecar)


def test_comparing_nothing_is_not_a_pass():
    """A gate over an empty population must fail, not report green."""
    panel = pd.DataFrame({"cusip": ["x"], "date": pd.to_datetime(["2010-01-31"])})
    with pytest.raises(AssertionError, match="nothing was compared"):
        C.assert_main_panel_is_adjusted(panel, panel)


def test_the_message_names_what_is_missing():
    try:
        C.assert_mmn_twins([f"{c}_mmn" for c in C.MMN_TWINNED if c != "cs"])
    except AssertionError as exc:
        msg = str(exc)
    assert "cs_mmn" in msg
    assert "ret_vw_bgn" in msg, "say which return the twin pairs with"

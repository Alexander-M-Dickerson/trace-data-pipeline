# -*- coding: utf-8 -*-
"""
test_holding_period.py
======================
`hprd` is the number of NYSE sessions between the two trades a month-end return is computed
from: `dt_s`, the bond's last trade in the final five sessions of month t-1, and `dt_e`, the
same in month t.

Until 2026-09-21 it was something else. The reference implementation counted sessions from
`dt_s` to the CALENDAR month-end, a date the return does not use, and every definition (the
data dictionary, the data report, the paper's Table IA.VIII) called it "calendar days". The
column overstated the window (mean 21.8 against a true 21.0) and nothing failed, because
nothing compared it with the two dates printed beside it.

`contract.assert_holding_period_is_the_window` makes that comparison at the end of every
build. These tests prove it fires on the old definition and passes on the new one.

A window LONGER than a month is not an error. The start trade may sit up to four sessions
before the last session of month t-1 and still be inside its five-session window, so a
23-session month can give a 27-session return. About 5% of returns are like that.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

STAGE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE2))

from lib import contract, nyse_calendar  # noqa: E402


@pytest.fixture(scope="module")
def lut() -> pd.DataFrame:
    return nyse_calendar.cal_lut_frame()


def _sessions(lut, a, b) -> int:
    s = lut.set_index(pd.to_datetime(lut["cday"]))["cum_before"]
    return int(s[pd.Timestamp(b)] - s[pd.Timestamp(a)])


def _frame(lut, rows, to="trade"):
    """rows = [(dt_s, dt_e, calendar month-end)]. `to` picks which end hprd is measured to."""
    out = []
    for s, e, m in rows:
        end = e if to == "trade" else m
        out.append({"dt_s": pd.Timestamp(s), "dt_e": pd.Timestamp(e), "hprd": _sessions(lut, s, end)})
    return pd.DataFrame(out)


ROWS = [
    ("2025-09-30", "2025-10-31", "2025-10-31"),   # last session to last session: 23
    ("2025-09-24", "2025-10-31", "2025-10-31"),   # start trade 4 sessions early: 27
    ("2025-09-30", "2025-10-27", "2025-10-31"),   # end trade 4 sessions early: 19
]


def test_october_2025_has_the_windows_the_docstring_says(lut):
    assert _frame(lut, ROWS)["hprd"].tolist() == [23, 27, 19]


def test_the_real_window_passes(lut):
    contract.assert_holding_period_is_the_window(_frame(lut, ROWS), lut)


def test_the_old_definition_is_caught(lut):
    """Measured to the calendar month-end, the third row reads 23 where the window is 19."""
    old = _frame(lut, ROWS, to="calendar")
    assert old["hprd"].tolist() == [23, 27, 23]
    with pytest.raises(AssertionError, match="differs from the dt_s -> dt_e session count on 1 of 3"):
        contract.assert_holding_period_is_the_window(old, lut)


def test_calendar_days_are_caught(lut):
    days = _frame(lut, ROWS)
    days["hprd"] = (days["dt_e"] - days["dt_s"]).dt.days
    with pytest.raises(AssertionError, match="session count"):
        contract.assert_holding_period_is_the_window(days, lut)


def test_an_empty_population_is_not_a_pass(lut):
    empty = _frame(lut, ROWS)
    empty["hprd"] = float("nan")
    with pytest.raises(AssertionError, match="nothing was compared"):
        contract.assert_holding_period_is_the_window(empty, lut)


def test_rows_without_a_start_trade_are_left_alone(lut):
    """The first month of a bond has no dt_s and no hprd. That is not a violation."""
    df = pd.concat([_frame(lut, ROWS),
                    pd.DataFrame([{"dt_s": pd.NaT, "dt_e": pd.Timestamp("2025-10-31"),
                                   "hprd": float("nan")}])], ignore_index=True)
    contract.assert_holding_period_is_the_window(df, lut)


def test_a_missing_column_is_refused(lut):
    with pytest.raises(AssertionError, match="need"):
        contract.assert_holding_period_is_the_window(_frame(lut, ROWS).drop(columns="dt_e"), lut)

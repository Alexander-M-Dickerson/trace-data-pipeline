# -*- coding: utf-8 -*-
"""
test_frontier.py
================
The last month of a panel is the one most likely to be wrong, and the least likely to be
looked at.

Stage 1 pools TRACE Enhanced with 144A, which report on different lags, and resolves its
`auto:-Nmo` cut-off against the pooled maximum -- 144A's. When that lands in a month where
Enhanced has only a few days, the monthly panel still emits the month, but the only bonds
with a month-end price are 144A issues. The 2026-09-09 production run is exactly this: the
final month held 1,914 bonds against a trailing median of 10,702, and was 100% 144A where
every other month is around 22%.

Nothing failed. The build succeeded, the column contract passed, coverage passed -- and the
panel would have shipped with a final cross-section that is not a cross-section.

These tests cover the detector and the shape of what it reports.

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

from lib import frontier  # noqa: E402


def _panel(counts: list[int], share_144a: list[float] | None = None) -> pd.DataFrame:
    """One row per (bond, month), with `counts[i]` bonds in month i."""
    months = pd.date_range("2022-01-31", periods=len(counts), freq="ME")
    rows = []
    for i, (m, n) in enumerate(zip(months, counts)):
        s = share_144a[i] if share_144a else 0.2
        for b in range(n):
            rows.append((f"{b:09d}", m, 1 if b < int(n * s) else 0))
    return pd.DataFrame(rows, columns=["cusip", "date", "144a"])


def test_a_healthy_panel_passes():
    df = _panel([1000 + (i % 7) * 10 for i in range(24)])
    assert frontier.degenerate_tail_months(df).empty
    frontier.assert_frontier_is_a_cross_section(df)


def test_a_collapsed_final_month_is_caught():
    counts = [1000] * 23 + [180]          # the production failure, in miniature
    df = _panel(counts)
    bad = frontier.degenerate_tail_months(df)
    assert len(bad) == 1
    assert str(bad.index[-1])[:10] == str(df["date"].max())[:10]
    assert bad["coverage"].iloc[0] < 0.25


def test_the_gate_raises_and_names_the_month():
    df = _panel([1000] * 23 + [180])
    with pytest.raises(AssertionError, match="not a usable cross-section"):
        frontier.assert_frontier_is_a_cross_section(df)


def test_the_message_carries_the_evidence():
    df = _panel([1000] * 23 + [180], share_144a=[0.2] * 23 + [1.0])
    try:
        frontier.assert_frontier_is_a_cross_section(df)
    except AssertionError as exc:
        msg = str(exc)
    assert "180 bonds" in msg
    assert "144A" in msg, "the 144A share is the diagnostic; it must be in the message"


def test_a_mild_dip_is_not_flagged():
    """Real months move by a few percent. Only a collapse should fail."""
    df = _panel([1000] * 23 + [900])
    assert frontier.degenerate_tail_months(df).empty


def test_only_the_frontier_is_examined():
    """An old thin month is history, not a publication problem."""
    counts = [1000] * 6 + [150] + [1000] * 17
    df = _panel(counts)
    assert frontier.degenerate_tail_months(df).empty


def test_the_start_of_the_panel_is_not_flagged():
    """A ramp-up at inception has no trailing history to be judged against."""
    df = _panel([50, 120, 400, 800] + [1000] * 8)
    bad = frontier.degenerate_tail_months(df)
    assert bad.empty


def test_report_shape():
    df = _panel([1000] * 12)
    rep = frontier.frontier_report(df)
    for col in ("bonds", "share_144a", "trailing_median", "coverage"):
        assert col in rep.columns
    assert len(rep) == 12


def test_it_works_without_a_144a_column():
    df = _panel([1000] * 23 + [180]).drop(columns=["144a"])
    bad = frontier.degenerate_tail_months(df)
    assert len(bad) == 1
    assert "share_144a" not in bad.columns

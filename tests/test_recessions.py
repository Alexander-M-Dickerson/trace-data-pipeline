# -*- coding: utf-8 -*-
"""
test_recessions.py
==================
Every time-series figure the pipeline draws shades the NBER recessions, from ONE definition
(recessions.py at the repository root). This pins the definition, what the shading does at a
panel's edges, and that stage 2's data report and stage 3's figures both use it.

Why it exists. The submitted DRR paper shaded two of its three time-series figures, stage 3
had dropped the shading everywhere, and the stage 2 report never had it. A figure that forgets
it looks fine, so nothing else would notice.

No data on disk: the axes are built by hand.

    python -m pytest tests/test_recessions.py -q

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import recessions  # noqa: E402

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_the_two_recessions_in_the_trace_sample():
    assert [(a[:7], b[:7]) for a, b, _ in recessions.NBER] == [("2007-12", "2009-06"),
                                                               ("2020-02", "2020-04")]


def _spans(ax):
    import matplotlib.dates as mdates
    return [tuple(mdates.num2date(x).strftime("%Y-%m-%d") for x in (p.get_x(), p.get_x() + p.get_width()))
            for p in ax.patches if p.get_label() == "_recession"]


def test_shade_draws_only_the_recessions_inside_the_panel():
    """A panel that starts in 2014 shades COVID only and is not stretched back to 2007."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    dates = pd.date_range("2014-07-31", "2025-11-30", freq="ME")
    ax.plot(dates, range(len(dates)))
    assert recessions.shade(ax, dates, color="gray") == 1
    assert _spans(ax) == [("2020-02-01", "2020-04-30")]
    assert matplotlib.dates.num2date(ax.get_xlim()[0]).year >= 2014
    plt.close(fig)


def test_shade_clips_a_recession_at_the_panel_edge():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    dates = pd.date_range("2008-06-30", "2012-12-31", freq="ME")
    assert recessions.shade(ax, dates) == 1
    assert _spans(ax) == [("2008-06-30", "2009-06-30")]
    plt.close(fig)


def test_stage3_uses_the_one_definition():
    sys.path.insert(0, str(ROOT / "stage3"))
    import figstyle
    assert figstyle.NBER is recessions.NBER


def _functions(path: Path) -> dict[str, str]:
    src = path.read_text(encoding="utf-8")
    return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef)}


def test_every_stage2_date_axis_is_shaded_and_says_so():
    """Each report plot that formats a date axis shades every such panel, and its caption
    carries the sentence that says the shading is there."""
    fns = _functions(ROOT / "stage2" / "_report_helpers.py")
    dated = {name: body for name, body in fns.items() if "DateFormatter(" in body}
    assert len(dated) >= 5, sorted(dated)
    for name, body in dated.items():
        axes = body.count("xaxis.set_major_formatter(DateFormatter")
        assert body.count("recessions.shade(") == axes, f"{name}: {axes} date panels"
        assert "recessions.CAPTION" in body, f"{name}: its caption does not mention the shading"

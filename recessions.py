"""recessions.py -- the NBER recessions every time-series figure shades, defined once.

Stage 2's data report (_report_helpers.py) and stage 3's figures (figstyle.py) both read them:
the Great Recession (2007:12 to 2009:06) and COVID-19 (2020:02 to 2020:04), NBER peak month to
trough month, shaded from the first day of the peak month to the last day of the trough month.
These are the two inside the TRACE sample.

`shade(ax, dates)` draws the recessions that fall inside the dates a panel plots, clipped to
them, so a panel that starts in 2014 is not stretched back to 2007 to reach one.

Not shaded: the per-bond price plots of stages 0 and 1 (one bond's trades around a flagged
error), where a recession band would say nothing about the filter being shown.
"""
from __future__ import annotations

NBER = (("2007-12-01", "2009-06-30", "Great Recession"),
        ("2020-02-01", "2020-04-30", "COVID-19"))

CAPTION = "Shaded areas mark NBER recessions."


def shade(ax, dates, **style) -> int:
    """Shade the recessions inside `dates` (the dates the panel plots) on a date axis, behind
    the data and with no legend entry. Returns how many it drew."""
    import pandas as pd
    d = pd.to_datetime(pd.Series(list(dates))).dropna()
    if d.empty:
        return 0
    first, last = d.min(), d.max()
    n = 0
    for start, end, _ in NBER:
        a, b = max(pd.Timestamp(start), first), min(pd.Timestamp(end), last)
        if a < b:
            ax.axvspan(a, b, lw=0, zorder=0, label="_recession", **style)
            n += 1
    return n

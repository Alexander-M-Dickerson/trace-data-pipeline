"""frontier.py -- refuse to publish a final month that is not a real cross-section.

Stage 1 pools two sources with different reporting lags: TRACE Enhanced (db_type 1) and
144A/BTDS (db_type 3). 144A reaches closer to today. Stage 1's `DATE_CUT_OFF = "auto:-Nmo"`
is resolved against the LAST trade date in the pooled data, which is 144A's -- so the cut
can land in a month where Enhanced has only a handful of days.

The monthly panel then still produces that month. It just is not a cross-section any more:
a bond needs a trade near month-end to get a month-end price, Enhanced bonds have stopped
appearing by then, and what survives is the 144A universe alone.

Measured on the 2026-09-09 production run, which is exactly this case:

    2025-11-30    10,548 bonds    21.6% 144A
    2025-12-31     1,914 bonds   100.0% 144A     <- Enhanced ends 2025-12-04

Sorting on that month gives a portfolio of Rule 144A issues and nothing else. The
published 2025 vintage does not have this -- it ends at 2025-03-31 with 11,248 bonds and
22.8% 144A -- so it is a defect of a particular run, not a standing property, which is why
it needs a check rather than a fixed cut-off.

What this does NOT do: judge whether a month is economically unusual. A genuine market
event that halves trading would look similar. It compares a month against its own recent
history and says the coverage collapsed; a human decides what to do about it.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import pandas as pd

# A month whose bond count falls below this share of the trailing median is not a
# cross-section. Real months sit within a few percent of each other; the failure mode
# this catches is an order-of-magnitude collapse, so the threshold is not delicate.
MIN_COVERAGE_SHARE = 0.75
TRAILING_MONTHS = 12          # history each candidate month is judged against
CHECK_LAST = 6                # only the frontier can be truncated this way


def frontier_report(df: pd.DataFrame, *, date_col: str = "date",
                    id_col: str = "cusip") -> pd.DataFrame:
    """Per-month bond count, 144A share, and coverage against the trailing median."""
    cols = [date_col, id_col] + (["144a"] if "144a" in df.columns else [])
    g = df[cols].groupby(date_col, observed=True).agg(
        bonds=(id_col, "nunique"),
        **({"share_144a": ("144a", "mean")} if "144a" in df.columns else {}))
    g["trailing_median"] = (g["bonds"].shift(1)
                            .rolling(TRAILING_MONTHS, min_periods=3).median())
    g["coverage"] = g["bonds"] / g["trailing_median"]
    return g


def degenerate_tail_months(df: pd.DataFrame, *, date_col: str = "date",
                           id_col: str = "cusip",
                           min_share: float = MIN_COVERAGE_SHARE) -> pd.DataFrame:
    """The trailing months whose coverage collapsed. Empty is the healthy answer."""
    g = frontier_report(df, date_col=date_col, id_col=id_col)
    tail = g.tail(CHECK_LAST)
    return tail[tail["coverage"].notna() & (tail["coverage"] < min_share)]


def describe(bad: pd.DataFrame) -> str:
    lines = []
    for dt, r in bad.iterrows():
        share = (f", {100 * r['share_144a']:.1f}% 144A"
                 if "share_144a" in bad.columns and pd.notna(r["share_144a"]) else "")
        lines.append(f"    {str(dt)[:10]}: {int(r['bonds']):,} bonds vs a trailing median "
                     f"of {int(r['trailing_median']):,} "
                     f"({100 * r['coverage']:.0f}% coverage{share})")
    return "\n".join(lines)


def assert_frontier_is_a_cross_section(df: pd.DataFrame, *, what: str = "panel",
                                       date_col: str = "date", id_col: str = "cusip",
                                       min_share: float = MIN_COVERAGE_SHARE) -> None:
    """Raise if the last months of `df` are not real cross-sections."""
    bad = degenerate_tail_months(df, date_col=date_col, id_col=id_col,
                                 min_share=min_share)
    if bad.empty:
        return
    raise AssertionError(
        f"{what}: {len(bad)} month(s) at the frontier are not a usable cross-section:\n"
        + describe(bad)
        + "\n\n  The usual cause is Stage 1's date cut-off landing where TRACE Enhanced "
          "has\n  only a few days but 144A has the full month, so the survivors are 144A "
          "alone.\n  Truncate the panel to the last healthy month before publishing, or "
          "rebuild\n  Stage 1 with a cut-off inside BOTH sources."
    )

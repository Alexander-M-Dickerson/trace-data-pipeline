"""nyse_calendar.py -- the ONE NYSE session calendar for the monthly port.

Faithful to the reference implementation's `streamline_data`: sessions come from
`pandas_market_calendars.get_calendar('NYSE')` over a fixed range, and business-day gaps are
`np.busday_count(start, end, holidays=<non-session days>)`, i.e. the number of NYSE sessions in
[start, end). We expose that same quantity relationally:

    day_gap(s, e) = cum_before(e) - cum_before(s)

where `cum_before(d)` = number of NYSE sessions strictly before calendar day `d`. The identity holds
for ANY calendar days (bond trades can print on non-NYSE sessions, e.g. Veterans Day).

The calendar is materialized once to `data/nyse_calendar.parquet` (one row per CALENDAR day:
`day DATE, is_session BOOLEAN, cum_before INT32`) and read by every step. Rebuilt only if absent or
the configured range changed (the range is embedded in the parquet's key row span).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import _stage2_settings as cfg

CALENDAR_PARQUET = cfg.DATA_DIR / "nyse_calendar.parquet"


def build_calendar_frame(start: str = cfg.CAL_START, end: str = cfg.CAL_END) -> pd.DataFrame:
    """One row per calendar day in [start, end]: (day, is_session, cum_before).

    `cum_before` counts NYSE sessions strictly before `day` -- the prefix sum that turns
    np.busday_count into a join. Uses pandas_market_calendars, same as the upstream golden.
    """
    import pandas_market_calendars as mcal

    sched = mcal.get_calendar(cfg.CALENDAR_NAME).schedule(start_date=start, end_date=end)
    sessions = pd.DatetimeIndex(sched.index).normalize()
    all_days = pd.date_range(start=start, end=end, freq="D")
    is_session = all_days.isin(sessions)
    cum_before = np.concatenate([[0], np.cumsum(is_session[:-1])]).astype("int32")
    return pd.DataFrame({"day": all_days, "is_session": is_session, "cum_before": cum_before})


def ensure_calendar(path: Path = CALENDAR_PARQUET) -> Path:
    """Materialize the calendar parquet if absent or stale (range mismatch); return its path."""
    if path.exists():
        head = pd.read_parquet(path, columns=["day"])
        if (str(head["day"].min().date()) == cfg.CAL_START
                and str(head["day"].max().date()) == cfg.CAL_END):
            return path
    cfg.ensure_dirs()
    df = build_calendar_frame()
    df.to_parquet(path, index=False)
    return path


def non_business_days(start: str = cfg.CAL_START, end: str = cfg.CAL_END) -> np.ndarray:
    """The `holidays` array for np.busday_count, exactly as upstream builds it (weekends + holidays)."""
    df = build_calendar_frame(start, end)
    return df.loc[~df["is_session"], "day"].values.astype("datetime64[D]")


def cal_lut_frame() -> pd.DataFrame:
    """Calendar-day LUT for SQL joins: cday, cum_before (sessions < cday), prev_session
    (last session strictly < cday -- the CustomBusinessDay(-1) roll for any calendar day)."""
    cal = build_calendar_frame()
    sessions = cal.loc[cal["is_session"], "day"].to_numpy()
    idx = np.searchsorted(sessions, cal["day"].to_numpy(), side="left") - 1
    prev_session = np.where(idx >= 0, sessions[np.maximum(idx, 0)], np.datetime64("NaT"))
    return pd.DataFrame({"cday": cal["day"], "cum_before": cal["cum_before"],
                         "prev_session": prev_session})

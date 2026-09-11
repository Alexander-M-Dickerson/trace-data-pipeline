"""month_boundaries.py -- the per-month boundary table for step 1 (upstream steps 6-7 of
the reference implementation), computed exactly from the NYSE session calendar.

Upstream builds these with pandas CustomBusinessDay/CustomBusinessMonthEnd offsets; we compute the
identical dates positionally on the session array (proven equivalent in tests/test_month_boundaries.py):

  first_bday          first NYSE session >= calendar month start   (month_begin + 0*bday)
  impl_floor          first_bday + (imp_gap-1) sessions            (imp_gap=1 -> first_bday itself)
  cut_off_begin       first_bday + (business_day_gap-1) sessions   (the 5th session of the month)
  bus_month_end       last NYSE session of the month               (CustomBusinessMonthEnd)
  cut_off_end         bus_month_end - (business_day_gap-1) sessions
  date_end_bus_lag    previous month's bus_month_end (shift over the months PRESENT IN THE DATA)
  month_end_cal       calendar month end (the panel date)

Note the upstream "internal" offsets: the user-facing knob business_day_gap=5 means "within the first/
last 5 sessions", implemented as first_bday + 4*bday (in the reference implementation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _stage2_settings as cfg
from lib import nyse_calendar


def build_month_bounds(month_starts: pd.Series | np.ndarray) -> pd.DataFrame:
    """One row per month in `month_starts` (datetime64 month-start values, as found in the data).

    Returns columns: month_start, first_bday, impl_floor, cut_off_begin, bus_month_end,
    cut_off_end, date_end_bus_lag, month_end_cal. All datetime64[ns].
    """
    cal = nyse_calendar.build_calendar_frame()
    sessions = cal.loc[cal["is_session"], "day"].to_numpy()          # sorted datetime64[ns]

    ms = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(month_starts).normalize())))
    month_end_cal = ms + pd.offsets.MonthEnd(0)

    gap = cfg.BUSINESS_DAY_GAP - 1                                    # upstream *_internal offsets
    impl = max(0, cfg.IMP_GAP - 1)

    first_idx = np.searchsorted(sessions, ms.values, side="left")     # first session >= month start
    last_idx = np.searchsorted(sessions, month_end_cal.values, side="right") - 1  # last session <= month end

    if (first_idx + gap >= len(sessions)).any() or (last_idx - gap < 0).any():
        raise ValueError("session calendar does not cover the data's month range -- extend CAL_START/END")

    out = pd.DataFrame({
        "month_start": ms,
        "first_bday": sessions[first_idx],
        "impl_floor": sessions[first_idx + impl],
        "cut_off_begin": sessions[first_idx + gap],
        "bus_month_end": sessions[last_idx],
        "cut_off_end": sessions[last_idx - gap],
        "month_end_cal": month_end_cal,
    })
    # previous month's business end, shifted over the months present (upstream dti_end.shift(1))
    out["date_end_bus_lag"] = out["bus_month_end"].shift(1)
    return out

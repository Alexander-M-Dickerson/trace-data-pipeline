"""Prove the positional month-boundary construction == upstream pandas CustomBusinessDay/CBME offsets
(process_bond_data steps 6-7 with the *_internal offsets)."""
import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import CustomBusinessDay, CustomBusinessMonthEnd, MonthEnd

import _stage2_settings as cfg
from lib import month_boundaries, nyse_calendar


@pytest.fixture(scope="module")
def bounds_vs_pandas():
    months = pd.date_range("2002-06-01", "2025-03-01", freq="MS")
    ours = month_boundaries.build_month_bounds(months)

    nbd = nyse_calendar.build_calendar_frame()
    holidays = nbd.loc[~nbd["is_session"], "day"]
    bday = CustomBusinessDay(holidays=holidays)
    cbme = CustomBusinessMonthEnd(holidays=holidays)
    gap = cfg.BUSINESS_DAY_GAP - 1
    impl = max(0, cfg.IMP_GAP - 1)

    ref = pd.DataFrame({"month_start": months})
    rolled = ref["month_start"] + 0 * bday
    ref["first_bday"] = rolled
    ref["impl_floor"] = rolled + impl * bday
    ref["cut_off_begin"] = rolled + gap * bday
    ref["bus_month_end"] = rolled + cbme
    ref["cut_off_end"] = ref["bus_month_end"] - gap * bday
    ref["date_end_bus_lag"] = ref["bus_month_end"].shift(1)
    ref["month_end_cal"] = ref["bus_month_end"] + MonthEnd(0)
    return ours, ref


@pytest.mark.parametrize("col", ["first_bday", "impl_floor", "cut_off_begin",
                                 "bus_month_end", "cut_off_end", "date_end_bus_lag",
                                 "month_end_cal"])
def test_boundary_col_matches_pandas_offsets(bounds_vs_pandas, col):
    ours, ref = bounds_vs_pandas
    a, b = ours[col], ref[col]
    mism = (a.values != b.values) & ~(pd.isna(a).values & pd.isna(b).values)
    assert not mism.any(), f"{col}: {mism.sum()} mismatches, first at {ours.loc[mism.argmax(), 'month_start']}"

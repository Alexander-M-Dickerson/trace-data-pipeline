"""The session-calendar identity: cum_before(e) - cum_before(s) == np.busday_count(s, e, holidays),
with holidays built exactly as upstream streamline_data builds them."""
import numpy as np
import pandas as pd

from lib import nyse_calendar as nc


def test_cum_before_matches_busday_count():
    start, end = "2001-07-01", "2007-12-31"             # covers 9/11 closures + Reagan funeral 2004-06-11
    frame = nc.build_calendar_frame(start, end)
    lut = frame.set_index("day")["cum_before"]

    holidays = frame.loc[~frame["is_session"], "day"].values.astype("datetime64[D]")
    rng = np.random.default_rng(11)
    days = pd.date_range(start, "2007-06-30", freq="D")
    starts = days[rng.integers(0, len(days), 500)]
    gaps = rng.integers(0, 40, 500)
    for s, g in zip(starts, gaps):
        e = s + pd.Timedelta(days=int(g))
        expected = int(np.busday_count(s.date(), e.date(), holidays=holidays))
        got = int(lut[e] - lut[s])
        assert got == expected, f"{s.date()}..{e.date()}: got {got}, busday_count {expected}"


def test_known_nyse_closures_are_non_sessions():
    frame = nc.build_calendar_frame("2001-07-01", "2013-01-31").set_index("day")
    for closed in ["2001-09-11", "2001-09-12", "2004-06-11",   # 9/11, Reagan funeral
                   "2012-10-29", "2012-10-30"]:                # Sandy
        assert not frame.loc[pd.Timestamp(closed), "is_session"], closed
    for open_day in ["2001-09-17", "2012-10-31"]:              # reopen days
        assert frame.loc[pd.Timestamp(open_day), "is_session"], open_day

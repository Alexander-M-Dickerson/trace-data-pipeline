# -*- coding: utf-8 -*-
"""
test_cut_off_basis.py
=====================
`DATE_CUT_OFF = "auto:-3mo"` must be measured from the LEAST current source.

Stage 1 pools TRACE Enhanced with 144A/BTDS. They are different populations reported on
different lags, and 144A runs months ahead. The spec used to be resolved against the
POOLED last trade date, which is 144A's, so the sample end landed where 144A was complete
and Enhanced had stopped months earlier.

From the 2026-09-10 production log, verbatim:

    Last trade date in the stage0 data: 2026-06-05 00:00:00      <- 144A
    Enhanced max date: 2025-12-04 00:00:00
    Resolved DATE_CUT_OFF auto:-3mo -> 2026-03-31

Everything from 2025-12 to 2026-03 is therefore a month in which no TRACE Enhanced bond
can have a month-end price. The monthly panel still emits those months; they are just not
cross-sections any more. The final one held 1,914 bonds, 100% 144A, against a trailing
median of ~10,700 at ~22% 144A.

Measured from Enhanced instead, the same spec gives 2025-09-30 -- three months of real
cross-sections traded for four months of 144A-only ones.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1"))
with warnings.catch_warnings():           # module-level stage0 auto-detect may warn
    warnings.simplefilter("ignore")
    import _stage1_settings as st  # noqa: E402

ENHANCED, STANDARD, A144 = 1, 2, 3


def _tape(rows: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame({"db_type": [r[0] for r in rows],
                         "trd_exctn_dt": pd.to_datetime([r[1] for r in rows])})


def test_the_production_case():
    """The exact dates from the 2026-09-10 run."""
    tape = _tape([(ENHANCED, "2025-12-04"), (A144, "2026-06-05")])
    assert st.cut_off_basis(tape) == pd.Timestamp("2025-12-04")
    assert st.resolve_date_cut_off("auto:-3mo", st.cut_off_basis(tape)) == "2025-09-30"


def test_the_pooled_max_is_what_went_wrong():
    """Pinned so the regression is legible: the old basis gave four 144A-only months."""
    tape = _tape([(ENHANCED, "2025-12-04"), (A144, "2026-06-05")])
    assert st.resolve_date_cut_off("auto:-3mo", tape["trd_exctn_dt"].max()) == "2026-03-31"


def test_a_single_source_run_is_unchanged():
    tape = _tape([(ENHANCED, "2025-12-04"), (ENHANCED, "2025-06-30")])
    assert st.cut_off_basis(tape) == tape["trd_exctn_dt"].max()


def test_standard_extends_enhanced_rather_than_bounding_it():
    """db_type 2 is the same public tape, clipped to start where Enhanced ends."""
    tape = _tape([(ENHANCED, "2025-12-04"), (STANDARD, "2026-02-20"), (A144, "2026-06-05")])
    assert st.cut_off_basis(tape) == pd.Timestamp("2026-02-20")


def test_the_frontiers_are_reported_per_population():
    tape = _tape([(ENHANCED, "2025-12-04"), (STANDARD, "2026-02-20"), (A144, "2026-06-05")])
    fr = st.source_frontiers(tape)
    assert fr == {"TRACE (Enhanced/Standard)": pd.Timestamp("2026-02-20"),
                  "144A/BTDS": pd.Timestamp("2026-06-05")}


def test_an_empty_population_is_omitted_not_null():
    """A NaT would silently poison the min and cut the sample to nothing."""
    tape = _tape([(A144, "2026-06-05")])
    assert list(st.source_frontiers(tape)) == ["144A/BTDS"]
    assert st.cut_off_basis(tape) == pd.Timestamp("2026-06-05")


def test_whichever_source_is_behind_wins():
    """It is not 'Enhanced always'; it is the least current, whichever that is."""
    tape = _tape([(ENHANCED, "2026-06-05"), (A144, "2025-12-04")])
    assert st.cut_off_basis(tape) == pd.Timestamp("2025-12-04")


def test_an_unknown_db_type_falls_back_to_the_pooled_max():
    tape = _tape([(9, "2025-12-04"), (9, "2026-06-05")])
    assert st.cut_off_basis(tape) == pd.Timestamp("2026-06-05")


@pytest.mark.parametrize("spec", [None, "2025-06-30"])
def test_an_explicit_cut_off_is_the_users_choice(spec):
    tape = _tape([(ENHANCED, "2025-12-04"), (A144, "2026-06-05")])
    assert st.resolve_date_cut_off(spec, st.cut_off_basis(tape)) == spec


# ---------------------------------------------------------------------------
# auto:complete -- the default. See `last_complete_month` in _stage1_settings.py.
#
# `auto:-3mo` measures back from a date that is itself mid-month, so it discards
# complete months. On the 2026-09-10 run it cut at 2025-09-30 and threw away October
# and November, both ordinary months: 11,848 and 11,612 Enhanced bonds at 10.9 and 11.1
# trades per bond-day, against a 2025 range of 11.3-11.9k bonds and 10-12 trades. Only
# December is unusable, because Enhanced stops on the 4th and no bond trades in its last
# seven days. `auto:complete` keeps exactly the usable months.
# ---------------------------------------------------------------------------


def _sessions(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _tape_with_sessions(spans: dict[int, tuple[str, str]]) -> pd.DataFrame:
    """One row per (db_type, session) for each population's span."""
    rows = []
    for db, (a, b) in spans.items():
        for dt in _sessions(a, b):
            rows.append((db, dt))
    return pd.DataFrame(rows, columns=["db_type", "trd_exctn_dt"])


def test_auto_complete_on_the_production_frontier():
    """Enhanced stops 2025-12-04; 144A runs to month end. November is the answer."""
    tape = _tape_with_sessions({ENHANCED: ("2024-01-01", "2025-12-04"),
                                A144: ("2024-01-01", "2025-12-31")})
    assert st.last_complete_month(tape) == "2025-11-30"
    assert st.resolve_cut_off_from_data("auto:complete", tape) == "2025-11-30"


def test_the_old_spec_would_have_discarded_two_good_months():
    tape = _tape_with_sessions({ENHANCED: ("2024-01-01", "2025-12-04"),
                                A144: ("2024-01-01", "2025-12-31")})
    assert st.resolve_cut_off_from_data("auto:-3mo", tape) == "2025-09-30"


def test_a_month_whose_last_session_is_the_28th_still_counts():
    """November 2025's last trading day is the 28th. A calendar-free rule must see that."""
    tape = _tape_with_sessions({ENHANCED: ("2025-01-01", "2025-11-28"),
                                A144: ("2025-01-01", "2025-11-28")})
    assert st.last_complete_month(tape) == "2025-11-30"


def test_a_partial_frontier_month_is_not_called_complete():
    """Both sources stopping on the same mid-month day must not pass as a whole month."""
    tape = _tape_with_sessions({ENHANCED: ("2024-01-01", "2025-12-05"),
                                A144: ("2024-01-01", "2025-12-05")})
    assert st.last_complete_month(tape) == "2025-11-30"


def test_the_least_current_source_still_decides():
    tape = _tape_with_sessions({ENHANCED: ("2024-01-01", "2025-08-15"),
                                A144: ("2024-01-01", "2025-12-31")})
    assert st.last_complete_month(tape) == "2025-07-31"


def test_a_single_source_run_keeps_its_last_whole_month():
    tape = _tape_with_sessions({ENHANCED: ("2024-01-01", "2025-12-04")})
    assert st.last_complete_month(tape) == "2025-11-30"


@pytest.mark.parametrize("spec", [None, "2024-12-31"])
def test_an_explicit_cut_off_passes_through_the_data_resolver(spec):
    tape = _tape_with_sessions({ENHANCED: ("2024-01-01", "2025-12-04")})
    assert st.resolve_cut_off_from_data(spec, tape) == spec


def test_the_default_is_auto_complete():
    assert st.DATE_CUT_OFF == "auto:complete"

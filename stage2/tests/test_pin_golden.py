"""Golden-faithfulness of the pin: run the ACTUAL upstream `streamline_data` (imported from the
golden tree) on a 200-cusip sample of the golden daily input, and require our DuckDB pin to match
it column-for-column (floats to 1e-6, counters exactly).

Skips cleanly when the golden tree (or its import deps) is unavailable -- this is the integration
test; the unit tests (test_validate_core / test_nyse_calendar) never skip.
"""
import sys

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import _stage2_settings as cfg

# This test diffs our daily projection against the ORIGINAL implementation's, which only
# exists in the development tree. Point STAGE2_REFERENCE_ROOT at that tree to run it;
# without it the test skips, which is the normal case for anyone outside development.
import os
_REF = os.environ.get("STAGE2_REFERENCE_ROOT")
GOLDEN_STAGE2_SRC = Path(_REF) / "stage2" if _REF else None

pytestmark = pytest.mark.skipif(
    GOLDEN_STAGE2_SRC is None,
    reason="set STAGE2_REFERENCE_ROOT to diff the pin against the reference implementation",
)
N_CUSIPS = 200
FLOAT_ATOL = 1e-6

# upstream streamline output -> our pin column name
COLMAP = {
    "cusip_id": "cusip_id", "trd_exctn_dt": "dt",
    "pr": "pr", "prc_1st": "prc_1st", "prc_lst": "prc_lst", "prc_hi": "prc_hi",
    "prc_lo": "prc_lo", "prc_bid": "prc_bid", "prc_ask": "prc_ask",
    "dvol": "dvol", "ao": "ao",
    "day_gap": "day_gap", "ret_d": "ret_d", "ret_c": "ret_c",
    "ret_d_lag": "ret_d_lag", "ret_c_lag": "ret_c_lag", "dvol_lag": "dvol_lag",
    "Nret": "Nret", "Npair": "Npair", "lst_txn": "lst_txn",
}
EXACT_COLS = {"day_gap", "Nret", "Npair", "lst_txn"}


@pytest.fixture(scope="module")
def upstream():
    if not (GOLDEN_STAGE2_SRC / "illiq_helper_functions.py").exists():
        pytest.skip("golden stage2 tree not available")
    sys.path.insert(0, str(GOLDEN_STAGE2_SRC))
    try:
        import illiq_helper_functions as up
    except ImportError as e:  # e.g. wrds/mcal missing in a foreign env
        pytest.skip(f"cannot import upstream module: {e}")
    return up


@pytest.fixture(scope="module")
def sample_cusips():
    import duckdb
    daily = cfg.GOLDEN_DAILY_INPUT.as_posix()
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT cusip_id FROM read_parquet('{daily}') ORDER BY cusip_id LIMIT {N_CUSIPS}"
    ).fetchall()
    return [r[0] for r in rows]


def test_pin_matches_upstream_streamline(upstream, sample_cusips, tmp_path):
    import duckdb
    from lib import pin as pinlib

    daily = cfg.GOLDEN_DAILY_INPUT.as_posix()
    con = duckdb.connect()

    # --- upstream reference on the same sample -----------------------------------------------
    keep = ["cusip_id", "trd_exctn_dt", "pr", "prfull", "acclast", "accpmt", "accall",
            "prc_first", "prc_last", "prc_hi", "prc_lo", "prc_bid", "prc_ask",
            "qvolume", "dvolume", "bond_amt_outstanding", "sp_rating", "mdy_rating"]
    placeholders = ",".join(f"'{c}'" for c in sample_cusips)
    ref_in = con.execute(
        f"SELECT {', '.join(keep)} FROM read_parquet('{daily}') "
        f"WHERE cusip_id IN ({placeholders})").df()
    ref_in["trd_exctn_dt"] = pd.to_datetime(ref_in["trd_exctn_dt"])
    ref = upstream.streamline_data(ref_in, calendar_name=cfg.CALENDAR_NAME,
                                   max_day_gap=cfg.BUSINESS_DAY_GAP)

    # --- our pin on the same sample -----------------------------------------------------------
    out = pinlib.build_pin(con, mode="golden", limit_cusips=N_CUSIPS, force=True)
    ours = pd.read_parquet(out)

    # duplicate (cusip, date) twins have arbitrary within-tie order; drop those cusips from both
    dup_cusips = set(ref.loc[ref.duplicated(["cusip_id", "trd_exctn_dt"], keep=False),
                             "cusip_id"].astype(str))
    if dup_cusips:
        ref = ref[~ref["cusip_id"].astype(str).isin(dup_cusips)]
        ours = ours[~ours["cusip_id"].astype(str).isin(dup_cusips)]

    ref = ref.sort_values(["cusip_id", "trd_exctn_dt"]).reset_index(drop=True)
    ours = ours.sort_values(["cusip_id", "dt"]).reset_index(drop=True)
    assert len(ref) == len(ours), f"row counts differ: upstream {len(ref)} vs pin {len(ours)}"
    assert len(ref) > 50_000, "sample unexpectedly small -- trivial pass guard"
    assert ref["ret_d"].notna().sum() > 10_000, "no returns in sample -- trivial pass guard"

    # month_year (Period[M]) vs our month_start (DATE)
    assert (ref["month_year"].dt.start_time.values ==
            pd.to_datetime(ours["month_start"]).values).all(), "month key mismatch"

    failures = []
    for ref_col, our_col in COLMAP.items():
        if ref_col in ("cusip_id", "trd_exctn_dt"):
            a = ref[ref_col].astype(str) if ref_col == "cusip_id" else ref[ref_col]
            b = ours[our_col].astype(str) if ref_col == "cusip_id" else pd.to_datetime(ours[our_col])
            if not (a.values == b.values).all():
                failures.append(f"{ref_col}: key column mismatch")
            continue
        a = ref[ref_col].astype("float64").to_numpy(na_value=np.nan)
        b = ours[our_col].astype("float64").to_numpy(na_value=np.nan)
        an, bn = np.isnan(a), np.isnan(b)
        if (an != bn).any():
            failures.append(f"{ref_col}: {(an != bn).sum()} NaN mismatches")
            continue
        both = ~an
        if not both.any():
            continue
        d = np.abs(a[both] - b[both])
        tol = 0.0 if ref_col in EXACT_COLS else FLOAT_ATOL
        if d.max() > tol:
            failures.append(f"{ref_col}: max|d|={d.max():.3g} > {tol}")
    assert not failures, "pin diverges from upstream streamline_data:\n  " + "\n  ".join(failures)

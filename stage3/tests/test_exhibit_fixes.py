"""test_exhibit_fixes.py -- four defects a cold read of the revised paper found (2026-09-28).

  * the LAB sample span comes from the months that hold data, not the frame's empty first row;
  * Figures IA.5 and IA.6 draw each baseline mark on the boxes' sign, so it lies inside its box;
  * Tables IA.V and IA.VI count extreme returns the same way at every sample size;
  * Table IA.IV prints every return in percent, the latent implementation bias included.

    python -m pytest tests/test_exhibit_fixes.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGE3 = Path(__file__).resolve().parents[1]
for p in (STAGE3, STAGE3 / "s0_data", STAGE3 / "s2_lab", STAGE3 / "s3_nse"):
    sys.path.insert(0, str(p))


def test_mua_marks_take_the_boxes_sign():
    """db_mkt's case: units signed on their own means, the box on the VW_Dp mean."""
    import nse_engine as E
    base = pd.DataFrame({"signal": ["x"] * 3, "baseline_tstat": [1.0, 2.0, 3.0],
                         "sign_mult_premia": [1, -1, 1]})
    # stored as signed per unit: raw t is (1, -2, 3); the box flips the signal (Dp mean < 0)
    mua = pd.DataFrame({"signal": ["x"], "spec_id": [E.FLIP_BASELINE], "mean_ret": [-0.1]})
    got = E.mua_baseline_values(base, "baseline_tstat", mua)["x"]
    assert np.isclose(got, np.mean([-1.0, 2.0, -3.0]))
    assert np.isclose(E.dua_baseline_values(base, "baseline_tstat")["x"], 2.0)   # DUA: as stored


def test_lab_span_starts_where_the_data_does():
    import lab_engine as L
    idx = pd.date_range("2002-08-31", periods=4, freq="ME")
    frame = pd.DataFrame({"f": [np.nan, 0.01, 0.02, -0.01]}, index=idx)
    cell = {L._series_key(leg, var): frame for leg in L.LEGS for var in ("wins", "base", "bias")}
    spec = L.LabSpec(return_type="standard", rating="All", tail="left")
    out = L.lab_stats(cell, pd.Series(0.0, index=idx), spec, factors=("f",))
    assert set(out["first"]) == {"2002-09-30"}


def test_extreme_counts_do_not_depend_on_sample_size():
    """A return stored at exactly -20% must count the same in a 2-million-row column (which
    pandas would hand to numexpr) as in a small one."""
    import data_engine as DE
    edge = np.float32(-0.2)
    big = np.full(2_000_000, 0.01, dtype=np.float32)
    big[:5] = edge
    df = pd.DataFrame({"ret_vw": big, "ret_vw_bgn": big,
                       "spc_rat": np.where(np.arange(big.size) % 2 == 0, 5, 15).astype("float64"),
                       "date": pd.Timestamp("2010-01-31")})
    counts = DE.extreme_stats(df)["counts"].set_index("Direction")
    assert counts.loc["<20", "End_All"] == counts.loc["<20", "End_IG"] + counts.loc["<20", "End_NIG"]
    per_year = DE.time_concentration(df)
    assert int(per_year["End_neg_20"].sum()) == int(counts.loc["<20", "End_All"])


def test_monthly_rows_in_percent_say_so():
    """Table IA.IV printed the latent implementation bias as a decimal (SD 0.02) beside
    returns in percent (SD 4.88), so its mean and median read 0.00. A row is scaled to
    percent exactly when its label says (%)."""
    import data_engine as DE
    assert DE.MONTHLY_SCALE.get("lib") == 100
    for v, label in DE.MONTHLY_STAT_VARS:
        assert (DE.MONTHLY_SCALE.get(v, 1) == 100) == label.endswith("(%)"), (v, label)

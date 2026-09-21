# -*- coding: utf-8 -*-
"""
test_release_redaction.py
=========================
The panel that is BUILT is not the panel that is PUBLISHED.

`permco` and `gvkey` are proprietary identifiers and the agency ratings are licensed, so
`make_release.redact_for_publication` nulls the two identifiers and collapses `spc_rat` /
`mdc_rat` to investment grade (1) against non-investment grade and default (11). The
released 2025 vintage does exactly this, and it was verified against that file:
`permco` and `gvkey` are 100% null there and the ratings take only the values {1, 11},
while the raw build has them at 86.7% / 86.5% populated and ratings 1..22.

This is the one place in the pipeline where a silent failure means distributing data we
have no right to distribute, so it is tested rather than trusted -- both that the
redaction does what it says, and that the gate in front of the packager actually catches
an un-redacted frame.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STAGE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE2))

import make_release as mr  # noqa: E402


def _raw() -> pd.DataFrame:
    """A frame shaped like the build: identifiers populated, ratings across the scale."""
    return pd.DataFrame({
        "cusip": [f"{i:09d}" for i in range(8)],
        "date": pd.date_range("2020-01-31", periods=8, freq="ME"),
        "permno": [10001, 10002, np.nan, 10004, 10005, 10006, 10007, np.nan],
        "permco": [20001, 20002, np.nan, 20004, 20005, 20006, 20007, np.nan],
        "gvkey": [1001.0, 1002.0, np.nan, 1004.0, 1005.0, 1006.0, 1007.0, np.nan],
        "spc_rat": [1, 5, 10, 11, 15, 21, 22, np.nan],
        "mdc_rat": [2, 8, 10, 11, 16, 20, 22, np.nan],
        "ret_vw": np.linspace(-0.02, 0.03, 8),
    })


def test_the_two_identifiers_are_nulled():
    out = mr.redact_for_publication(_raw())
    for col in ("permco", "gvkey"):
        assert out[col].notna().sum() == 0, f"{col} survived redaction"


def test_permno_and_the_data_are_untouched():
    raw = _raw()
    out = mr.redact_for_publication(raw)
    pd.testing.assert_series_equal(out["permno"], raw["permno"])
    pd.testing.assert_series_equal(out["ret_vw"], raw["ret_vw"])
    pd.testing.assert_series_equal(out["cusip"], raw["cusip"])


def test_ratings_collapse_at_the_investment_grade_boundary():
    out = mr.redact_for_publication(_raw())
    # 1..10 -> 1 (investment grade); 11 and above, default included -> 11
    assert out["spc_rat"].tolist()[:7] == [1, 1, 1, 11, 11, 11, 11]
    assert out["mdc_rat"].tolist()[:7] == [1, 1, 1, 11, 11, 11, 11]


def test_a_missing_rating_stays_missing():
    """A bond with no rating must not be silently labelled investment grade."""
    out = mr.redact_for_publication(_raw())
    assert pd.isna(out["spc_rat"].iloc[-1])
    assert pd.isna(out["mdc_rat"].iloc[-1])


def test_redaction_does_not_mutate_its_input():
    raw = _raw()
    before = raw.copy()
    mr.redact_for_publication(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_the_gate_accepts_a_redacted_frame():
    mr.assert_publishable(mr.redact_for_publication(_raw()), "test frame")


def test_the_gate_rejects_surviving_identifiers():
    bad = mr.redact_for_publication(_raw())
    bad.loc[0, "gvkey"] = 1234.0
    with pytest.raises(AssertionError, match="gvkey"):
        mr.assert_publishable(bad, "test frame")


def test_the_gate_rejects_raw_ratings():
    bad = mr.redact_for_publication(_raw())
    bad.loc[0, "spc_rat"] = 7
    with pytest.raises(AssertionError, match="spc_rat"):
        mr.assert_publishable(bad, "test frame")


def test_the_gate_rejects_a_completely_unredacted_panel():
    """The case that matters: someone packages the build output by mistake."""
    with pytest.raises(AssertionError):
        mr.assert_publishable(_raw(), "the raw build")


def test_a_frame_without_the_columns_is_accepted():
    """The additional artifacts carry no identifiers or ratings; they are publishable."""
    mr.assert_publishable(
        pd.DataFrame({"cusip": ["0" * 9], "date": [pd.Timestamp("2020-01-31")],
                      "b_mktb": [1.0]}),
        "betas_x")


# =============================================================================================
# THE DAILY PANEL
#
# On 2026-09-18 Stage 1's daily file was uploaded as it stood: 44 columns, the agency ratings
# on about 30 million rows, permco and gvkey. It was copied with SELECT * by a step that had
# no notion of a public layout. `make_release.py --what daily` is that notion, and these tests
# are what stands between the raw file and a public link.
# =============================================================================================

STAGE1_COLUMNS = (
    "cusip_id", "permno", "permco", "gvkey", "trd_exctn_dt", "pr", "prfull", "acclast",
    "accpmt", "accall", "ytm", "mod_dur", "mac_dur", "convexity", "bond_maturity",
    "credit_spread", "prc_ew", "prc_vw_par", "prc_first", "prc_last", "prc_hi", "prc_lo",
    "trade_count", "time_ew", "time_last", "qvolume", "dvolume", "prc_bid", "bid_last",
    "bid_time_ew", "bid_time_last", "prc_ask", "bid_count", "ask_count", "db_type", "ff12num",
    "ff17num", "ff30num", "bond_age", "bond_amt_outstanding", "sp_rating", "mdy_rating",
    "spc_rating", "mdc_rating")


def _daily(columns=STAGE1_COLUMNS, n: int = 6) -> pd.DataFrame:
    out = {}
    for i, c in enumerate(columns):
        if c == "cusip_id":
            out[c] = [f"{k:09d}" for k in range(n)]
        elif c == "trd_exctn_dt":
            out[c] = pd.date_range("2025-01-02", periods=n, freq="B")
        else:
            out[c] = np.arange(n, dtype="float64") + i
    return pd.DataFrame(out)


def _write(tmp_path: Path, df: pd.DataFrame, name: str = "daily.parquet") -> Path:
    p = tmp_path / name
    df.to_parquet(p, index=False)
    return p


def test_every_stage1_column_is_classified_once():
    """Public or withheld, never both, never neither. 32 + 12 = the 44 Stage 1 writes."""
    pub, held = set(mr.DAILY_PUBLIC_COLUMNS), set(mr.DAILY_WITHHELD)
    assert not (pub & held)
    assert pub | held == set(STAGE1_COLUMNS)
    assert len(mr.DAILY_PUBLIC_COLUMNS) == 32 and len(held) == 12
    assert set(mr.DAILY_LICENSED) == {"permco", "gvkey", "sp_rating", "mdy_rating",
                                      "spc_rating", "mdc_rating"}
    assert "permno" in pub                                   # permno is published


def test_the_daily_gate_accepts_the_public_layout(tmp_path):
    p = _write(tmp_path, _daily(mr.DAILY_PUBLIC_COLUMNS))
    mr.assert_daily_publishable(p, source_rows=6)


def test_the_daily_gate_rejects_the_raw_stage1_file(tmp_path):
    """The case that happened."""
    with pytest.raises(AssertionError, match="licensed"):
        mr.assert_daily_publishable(_write(tmp_path, _daily()))


@pytest.mark.parametrize("col", ["permco", "gvkey", "sp_rating", "mdy_rating",
                                 "spc_rating", "mdc_rating"])
def test_the_daily_gate_rejects_each_licensed_column_on_its_own(tmp_path, col):
    df = _daily(mr.DAILY_PUBLIC_COLUMNS + (col,))
    with pytest.raises(AssertionError, match=col):
        mr.assert_daily_publishable(_write(tmp_path, df))


def test_the_daily_gate_rejects_a_licensed_looking_name_it_has_never_seen(tmp_path):
    df = _daily(mr.DAILY_PUBLIC_COLUMNS + ("fitch_rating",))
    with pytest.raises(AssertionError, match="fitch_rating"):
        mr.assert_daily_publishable(_write(tmp_path, df))


def test_the_daily_gate_rejects_anything_outside_the_layout(tmp_path):
    with pytest.raises(AssertionError, match="outside the public layout"):
        mr.assert_daily_publishable(_write(tmp_path, _daily(mr.DAILY_PUBLIC_COLUMNS + ("time_ew",))))
    with pytest.raises(AssertionError, match="lacks public"):
        mr.assert_daily_publishable(_write(tmp_path, _daily(mr.DAILY_PUBLIC_COLUMNS[:-1])))
    with pytest.raises(AssertionError, match="different order"):
        mr.assert_daily_publishable(_write(tmp_path, _daily(mr.DAILY_PUBLIC_COLUMNS[::-1])))
    with pytest.raises(AssertionError, match="rows"):
        mr.assert_daily_publishable(_write(tmp_path, _daily(mr.DAILY_PUBLIC_COLUMNS)), source_rows=7)


def test_release_daily_writes_the_public_layout_and_says_where_it_came_from(tmp_path, monkeypatch):
    import json
    import pyarrow.parquet as pq
    src = _write(tmp_path, _daily(), "stage1_20990101.parquet")
    monkeypatch.setattr(mr.cfg, "daily_input", lambda mode=None: src)
    assert mr.release_daily(tmp_path / "out", "2099") == 0

    stage = tmp_path / "out" / "osbap_daily_data_2099"
    target = stage / "stage1_daily_panel_2099.parquet"
    assert tuple(pq.ParquetFile(target).schema.names) == mr.DAILY_PUBLIC_COLUMNS
    got, want = pd.read_parquet(target), _daily()[list(mr.DAILY_PUBLIC_COLUMNS)]
    pd.testing.assert_frame_equal(got, want, check_dtype=False)          # values untouched

    prov = json.loads((stage / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert prov["source_sha256"] == mr._sha256(src) and prov["rows"] == 6
    assert set(prov["withheld"]) == set(mr.DAILY_WITHHELD)
    text = (stage / "PROVENANCE.json").read_text(encoding="utf-8") + (stage / "README.md").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and ":\\" not in text, "a local path reached a public file"


def test_release_daily_refuses_a_column_nobody_has_classified(tmp_path, monkeypatch, capsys):
    """Stage 1 gains a column. It is withheld until someone decides, and nothing is written."""
    src = _write(tmp_path, _daily(STAGE1_COLUMNS + ("fitch_grade",)), "stage1_20990101.parquet")
    monkeypatch.setattr(mr.cfg, "daily_input", lambda mode=None: src)
    assert mr.release_daily(tmp_path / "out", "2099") == 1
    assert "fitch_grade" in capsys.readouterr().out
    assert not (tmp_path / "out" / "osbap_daily_data_2099").exists()


def test_a_published_path_is_never_absolute(tmp_path):
    inside = Path(mr.cfg.ROOT_PATH) / "stage2" / "output" / "panel" / "main_panel_stage1.parquet"
    assert mr._public_path(inside) == "stage2/output/panel/main_panel_stage1.parquet"
    assert mr._public_path(tmp_path / "somewhere" / "f.parquet") == "f.parquet"

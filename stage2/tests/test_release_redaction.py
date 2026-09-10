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

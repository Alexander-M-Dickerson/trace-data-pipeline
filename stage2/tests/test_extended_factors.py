"""test_extended_factors.py -- the published extended-BBW seam normalizes both layouts.

The published parquet ships `date` as a DatetimeIndex; a cached copy ships it as a column. Both must
normalize to the same ['date'] + 7-factor frame, and a dateless frame must fail loudly (the M11 corrupt
cache was exactly that -- index=False dropped the DatetimeIndex at cache-write)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lib.extended_factors import COLS, _normalize


def _frame() -> pd.DataFrame:
    dates = pd.to_datetime(["1973-02-28", "1973-03-31", "2002-07-31"])
    return pd.DataFrame({c: np.arange(3, dtype=float) + i for i, c in enumerate(COLS)},
                        index=pd.DatetimeIndex(dates, name="date"))


def test_normalize_date_indexed_and_date_column_agree():
    indexed = _normalize(_frame())
    columned = _normalize(_frame().reset_index())
    pd.testing.assert_frame_equal(indexed, columned)
    assert list(indexed.columns) == ["date"] + COLS
    assert indexed["date"].dt.is_month_end.all()


def test_normalize_coerces_mid_month_dates_to_month_end():
    df = _frame().reset_index()
    df["date"] = pd.to_datetime(["1973-02-01", "1973-03-15", "2002-07-31"])
    out = _normalize(df)
    assert out["date"].tolist() == pd.to_datetime(["1973-02-28", "1973-03-31", "2002-07-31"]).tolist()


def test_normalize_rejects_dateless_frame():
    corrupt = _frame().reset_index(drop=True)      # the corrupt-cache case: dates dropped entirely
    with pytest.raises(ValueError, match="date"):
        _normalize(corrupt)


def test_normalize_rejects_missing_factor_columns():
    with pytest.raises(ValueError, match="MKTBx"):
        _normalize(_frame().drop(columns=["MKTBx"]))

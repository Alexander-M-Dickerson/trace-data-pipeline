# -*- coding: utf-8 -*-
"""
test_daily_input_guard.py
=========================
Stage 2 cannot run on the Stage 1 file published for download: that file withholds the
licensed columns (the agency ratings, permco and gvkey), and Stage 2 keeps only bond-days
carrying a rating. The guard used to report it as "missing 6 column(s)", which the docs
explained as an older release. It now names the download.

Author: Open Source Bond Asset Pricing
"""
from __future__ import annotations

import pandas as pd
import pytest

import _stage2_settings as cfg
import make_release as mr


def _write(tmp_path, columns):
    p = tmp_path / "stage1_20260921.parquet"
    pd.DataFrame({c: [1] for c in columns}).to_parquet(p, index=False)
    return p


def test_the_public_download_is_named_as_the_public_download(tmp_path):
    p = _write(tmp_path, mr.DAILY_PUBLIC_COLUMNS)
    with pytest.raises(ValueError, match="published for download") as e:
        cfg._validate_daily_panel(p)
    assert "sp_rating" in str(e.value)


def test_a_file_missing_an_ordinary_column_is_called_an_older_release(tmp_path):
    cols = [c for c in cfg.REQUIRED_DAILY_COLUMNS if c != "prc_hi"]
    with pytest.raises(ValueError, match="missing 1 column") as e:
        cfg._validate_daily_panel(_write(tmp_path, cols))
    assert "published for download" not in str(e.value)

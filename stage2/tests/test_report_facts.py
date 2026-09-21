# -*- coding: utf-8 -*-
"""
test_report_facts.py
====================
The Stage 2 data report printed two things that looked like data errors and were not.

    Table 1 showed fewer PRICES than RETURNS. The row labelled "Price (VW)" was 100/bbtm, and
    the main panel's `bbtm` is the price on the SIGNAL date, which a month with no earlier
    trade does not have. The price a return is computed from is the month-end one, and every
    return has it. The report now prints that, and the signal-date price on its own row.

    Its prose carried typed numbers ("the average signal gap is approximately 1.68"), which
    outlive the data they came from. They are computed from the panel the report describes.

These tests hold the two fixes, and the one invariant behind the first: a row with a
month-end return and no month-end price stops the report.

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

import _build_data_report as bdr  # noqa: E402
import _report_helpers as rpt  # noqa: E402


def _panel() -> pd.DataFrame:
    d = pd.to_datetime(["2025-09-30", "2025-10-31", "2025-11-30"])
    return pd.DataFrame({
        "cusip": pd.Categorical(["A", "A", "B"]),
        "date": d,
        "ret_vw": [0.01, 0.02, np.nan],
        "bbtm": [1.00, np.nan, 0.80],          # the October row has no signal-date price
        "hprd": [21, 25, np.nan],
        "sig_gap": [1, np.nan, 3],
    })


def _sidecar(tmp_path: Path, mode: str, bbtm_mmn) -> None:
    folder = tmp_path / mode
    folder.mkdir(parents=True)
    p = _panel()
    pd.DataFrame({"cusip": p["cusip"], "date": p["date"].astype("datetime64[us]"),
                  "bbtm_mmn": np.asarray(bbtm_mmn, dtype="float32")}
                 ).to_parquet(folder / "mmn_price_based_signals_20990101.parquet")


def test_price_vw_is_the_month_end_price(tmp_path, monkeypatch):
    monkeypatch.setattr(bdr.cfg, "BLOCKS_DIR", tmp_path)
    _sidecar(tmp_path, "m", [1.25, 0.5, 2.0])
    out = bdr.attach_prices(_panel(), "m")
    assert out["pr"].tolist() == [80.0, 200.0, 50.0]          # 100 / bbtm_mmn, every row
    assert out["pr_sig"].iloc[0] == 100.0 and np.isnan(out["pr_sig"].iloc[1])
    # the thing the old table got wrong: prices can never be fewer than returns
    assert out["pr"].notna().sum() >= out["ret_vw"].notna().sum()
    assert out["pr_sig"].notna().sum() < out["pr"].notna().sum()


def test_a_return_without_a_month_end_price_stops_the_report(tmp_path, monkeypatch):
    monkeypatch.setattr(bdr.cfg, "BLOCKS_DIR", tmp_path)
    _sidecar(tmp_path, "m", [1.25, np.nan, 2.0])
    with pytest.raises(AssertionError, match="month-end return and no month-end price"):
        bdr.attach_prices(_panel(), "m")


def test_a_missing_sidecar_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bdr.cfg, "BLOCKS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="mmn_price_based_signals"):
        bdr.attach_prices(_panel(), "nothing_here")


def test_the_prose_numbers_come_from_the_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(bdr.cfg, "BLOCKS_DIR", tmp_path)
    _sidecar(tmp_path, "m", [1.25, 0.5, 2.0])
    f = bdr.window_facts(bdr.attach_prices(_panel(), "m"))
    assert (f["HPRD_MIN"], f["HPRD_MAX"], f["HPRD_MEAN"]) == ("21", "25", "23.00")
    assert f["HPRD_GT23"] == "50.0"
    assert (f["SIGGAP_MEAN"], f["SIGGAP_MEDIAN"], f["SIGGAP_MAX"]) == ("2.00", "2.00", "3")
    assert f["N_RET"] == "2" and f["N_RET_NO_SIGNAL_PRICE"] == "1"
    assert f["ADJ_WINDOW"] == str(bdr.cfg.ADJ_WINDOW)
    assert "21 & 1 & 50.00" in f["WINDOW_TABLE"] and "25 & 1 & 50.00" in f["WINDOW_TABLE"]


def test_an_unfilled_number_is_refused_not_printed(tmp_path, monkeypatch):
    with pytest.raises(KeyError, match="HPRD_MAX"):
        rpt.build_latex_document(tables={}, fig_filenames=[], vintage="2026")
    monkeypatch.setattr(bdr.cfg, "BLOCKS_DIR", tmp_path)
    _sidecar(tmp_path, "m", [1.25, 0.5, 2.0])
    facts = bdr.window_facts(bdr.attach_prices(_panel(), "m"))
    doc = rpt.build_latex_document(tables={}, fig_filenames=[], vintage="2026", facts=facts)
    assert "@@" not in doc
    assert "runs from 21 to 25 sessions" in doc
    assert "approximately 1.68" not in doc            # the typed number is gone

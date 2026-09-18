# -*- coding: utf-8 -*-
"""
test_release_bbw.py
===================
The BBW four-factor bundle, `make_release.py --what bbw`.

Two claims travel with that zip and both are tested rather than trusted. First, that the
extended series equals the TRACE series from 2002-08 -- step 4 keeps the TRACE-native
factors after the cutoff and splices the Lehman-ICE backfill only before it, so any
difference means the bundle was built from blocks of different runs. Second, that the file
shipped as the authors' original series IS that file: it is pinned by sha256, and a changed
copy is refused.

The gates are exercised on small synthetic blocks written to a temporary directory, with
the real pinned original, so the test needs no build on disk.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import make_release as mr  # noqa: E402


# ------------------------------------------------------------------ fixtures

def _months(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _write_blocks(root: Path, *, poison: str | None = None) -> Path:
    """Synthetic step-3 blocks (2002-08 on) and a factors_merged with a 1990-01 backfill."""
    rng = np.random.default_rng(0)
    trace_dates = _months("2002-08-31", 36)
    ext_dates = _months("1990-01-31", 36 + 151)          # 1990-01 .. 2005-07
    assert ext_dates[151] == trace_dates[0]

    cols = list(mr.bbw_columns())
    trace_vals = {pub: rng.normal(0, 0.01, len(trace_dates)) for pub, _, _ in cols}

    std = pd.DataFrame({t: trace_vals[p] for p, t, _ in cols if t in
                        ("MKTB", "DRF", "CRF", "LRF", "MKTBx", "DRFx", "CRFx", "LRFx")},
                       index=pd.Index(trace_dates, name="date"))
    std["TERM"] = 0.0; std["DEFB"] = 0.0; std["TERMB"] = 0.0
    bns = pd.DataFrame({t: trace_vals[p] for p, t, _ in cols if t.endswith("_bns")},
                       index=pd.Index(trace_dates, name="date"))
    cls = pd.DataFrame({t: trace_vals[p] for p, t, _ in cols if t.endswith("_cls")},
                       index=pd.Index(trace_dates, name="date"))

    merged = pd.DataFrame({"date": ext_dates})
    for pub, _, ext_col in cols:
        pre = rng.normal(0, 0.01, 151)
        if pub.startswith("lrf_"):
            pre = np.full(151, np.nan)                     # no LRF before TRACE
        merged[ext_col] = np.concatenate([pre, trace_vals[pub]])
    merged["mktrf"] = 0.0                                  # a macro column, as the real file has

    if poison == "shifted":
        merged.loc[merged["date"] >= "2002-08-31", "drf_bns"] += 1e-6   # the block name
    if poison == "lrf_pre":
        merged.loc[merged["date"] < "2002-08-31", "lrf"] = 0.001

    root.mkdir(parents=True, exist_ok=True)
    std.to_parquet(root / "bbw_factors.parquet")
    bns.to_parquet(root / "bbw_factors_bns.parquet")
    cls.to_parquet(root / "bbw_factors_cls.parquet")
    merged.to_parquet(root / "factors_merged.parquet", index=False)
    return root


# ------------------------------------------------------------------- the map

def test_the_sixteen_columns_map_every_block_column_once():
    cols = mr.bbw_columns()
    assert len(cols) == 16
    pubs = [p for p, _, _ in cols]
    assert pubs[:4] == ["mktb_exc", "drf_exc", "crf_exc", "lrf_exc"]
    assert pubs[4:8] == ["mktb_dur", "drf_dur", "crf_dur", "lrf_dur"]
    assert len(set(pubs)) == 16
    assert {t for _, t, _ in cols} >= {"MKTB", "MKTBx", "MKTB_bns", "MKTB_cls", "LRF_cls"}
    assert {e for _, _, e in cols} >= {"mktb", "mktbx", "mktb_bns", "lrf_cls"}


# ---------------------------------------------------------------- the gates

def test_clean_blocks_pass_and_the_extended_series_carries_the_backfill(tmp_path):
    blocks = _write_blocks(tmp_path / "blocks")
    trace = mr._bbw_trace(blocks)
    ext = mr._bbw_extended(blocks)
    mr._gate_bbw(trace, ext)
    assert len(trace) == 36 and list(trace.columns)[0] == "date" and trace.shape[1] == 17
    assert len(ext) == 36 + 151
    pre = ext[ext["date"] < "2002-08-31"]
    assert pre["mktb_exc"].notna().all() and pre["lrf_exc"].isna().all()


def test_an_extended_series_that_differs_after_the_cutoff_is_refused(tmp_path):
    """❗One millionth on one column in one era. This is what blocks from two different runs
    look like, and the bundle must not ship it as one history."""
    blocks = _write_blocks(tmp_path / "blocks", poison="shifted")
    with pytest.raises(SystemExit, match="differs from the TRACE series"):
        mr._gate_bbw(mr._bbw_trace(blocks), mr._bbw_extended(blocks))


def test_lrf_with_pre_trace_values_is_refused(tmp_path):
    blocks = _write_blocks(tmp_path / "blocks", poison="lrf_pre")
    with pytest.raises(SystemExit, match="LRF carries pre-2002-08"):
        mr._gate_bbw(mr._bbw_trace(blocks), mr._bbw_extended(blocks))


def test_blocks_on_different_axes_are_refused(tmp_path):
    blocks = _write_blocks(tmp_path / "blocks")
    bns = pd.read_parquet(blocks / "bbw_factors_bns.parquet").iloc[:-1]
    bns.to_parquet(blocks / "bbw_factors_bns.parquet")
    with pytest.raises(SystemExit, match="do not share one date axis"):
        mr._bbw_trace(blocks)


def test_the_original_series_is_pinned_and_present():
    """The file shipped as the authors' original must hash as pinned, and the pin must be
    the sha256 of a real file with the authors' columns and their 209 months."""
    assert mr.BBW_ORIGINAL.exists(), mr.BBW_ORIGINAL
    assert hashlib.sha256(mr.BBW_ORIGINAL.read_bytes()).hexdigest() == mr.BBW_ORIGINAL_SHA256
    o = pd.read_csv(mr.BBW_ORIGINAL, parse_dates=["date"])
    assert list(o.columns) == ["date", "MKTbond", "DRF", "CRF", "LRF"]
    assert len(o) == 209 and str(o["date"].min())[:10] == "2004-08-31"
    assert str(o["date"].max())[:10] == "2021-12-31"


def test_a_changed_original_is_refused(tmp_path, monkeypatch):
    fake = tmp_path / "bbw_factors_original_2004_2021.csv"
    fake.write_text(mr.BBW_ORIGINAL.read_text(encoding="utf-8").replace("0.0181", "0.0182"),
                    encoding="utf-8")
    monkeypatch.setattr(mr, "BBW_ORIGINAL", fake)
    blocks = _write_blocks(tmp_path / "blocks")
    with pytest.raises(SystemExit, match="does not hash as pinned"):
        mr._gate_bbw(mr._bbw_trace(blocks), mr._bbw_extended(blocks))


# ------------------------------------------------------------- the bundle

def test_release_bbw_writes_a_verifiable_bundle(tmp_path, monkeypatch):
    blocks = _write_blocks(tmp_path / "blocks" / "unit")
    monkeypatch.setattr(mr.cfg, "BLOCKS_DIR", tmp_path / "blocks")
    out = tmp_path / "release"
    assert mr.release_bbw("unit", out, "2099") == 0
    zpath = out / "osbap_bbw_factors_2099.zip"
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        assert names == {"bbw_factors_trace_2099.parquet", "bbw_factors_trace_2099.csv",
                         "bbw_factors_extended_2099.parquet", "bbw_factors_extended_2099.csv",
                         "bbw_factors_original_2004_2021.csv", "README.md",
                         "PROVENANCE.json", "MANIFEST.json"}
        import json
        man = json.loads(z.read("MANIFEST.json"))
        for name, info in man["members"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == info["sha256"], name
        assert z.read("bbw_factors_original_2004_2021.csv") == mr.BBW_ORIGINAL.read_bytes()
        readme = z.read("README.md").decode("utf-8")
        assert "2099 vintage" in readme and "Turan Bali" in readme
        prov = json.loads(z.read("PROVENANCE.json"))
        assert prov["original"]["source"] == "see README"
        assert "Bali" not in json.dumps(prov)


def test_what_all_includes_the_bundle():
    src = (Path(mr.__file__)).read_text(encoding="utf-8")
    assert 'if args.what in ("all", "bbw"):' in src, "--what all no longer builds the BBW bundle"

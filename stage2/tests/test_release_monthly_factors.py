# -*- coding: utf-8 -*-
"""
test_release_monthly_factors.py
===============================
The monthly factors in one file, `make_release.py --what monthly_factors`.

The file is step 4's factor matrix (`factors_merged.parquet`), the factors the betas are
estimated on. Its claim is that every value equals the block it came from, and that the bond
factors before 2002-08 are the Lehman-ICE series. The matrix here is built by step 4's own
`build_factor_matrix` from small synthetic blocks, so a clean build must pass. Each poison
changes one thing a stale or mixed build would change, and each must be refused.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import make_release as mr  # noqa: E402
from lib import extended_factors  # noqa: E402

BOND = ["MKTB", "DRF", "CRF", "LRF", "MKTBx", "DRFx", "CRFx", "LRFx", "TERM", "DEFB", "TERMB"]
TWIN = ["MKTB", "DRF", "CRF", "LRF", "TERM"]
LIQ = ["PSB", "AMD", "LIX", "ILLIQ", "ROLL", "SPRD", "CSS", "ARS", "FHTS", "VOV", "RVOL", "RSJ"]
# The Lehman-ICE series as lib/extended_factors returns it, with the BBW names (step 4 renames).
EXT = ["MKTB", "DRF", "CRF", "MKTBx", "DRFx", "CRFx", "TERM", "DEFB", "TERMB"] + [
    f"{f}_{bm}" for bm in ("bns", "cls") for f in ("MKTB", "DRF", "CRF", "TERM")]


def _months(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _write_blocks(root: Path, monkeypatch) -> tuple[Path, pd.DataFrame]:
    """Synthetic step-2 and step-3 blocks from 2002-08, a factor input from 1990-01 with the
    stock market columns and the old bond copies, and a Lehman-ICE series 1990-01 to 2003-12."""
    rng = np.random.default_rng(1)
    all_dates = _months("1990-01-31", 192)                  # 1990-01 .. 2005-12
    trace = _months("2002-08-31", 41)                       # 2002-08 .. 2005-12
    ext_dates = _months("1990-01-31", 168)                  # 1990-01 .. 2003-12
    root.mkdir(parents=True, exist_ok=True)

    fac = pd.DataFrame({"date": all_dates, "mktrf": rng.normal(0, .04, len(all_dates)),
                        "dvix": rng.normal(0, .01, len(all_dates)),
                        "vix": rng.uniform(.03, .1, len(all_dates))})
    for c in EXT:                                           # the copies step 4 drops unread
        fac[c.lower()] = 99.0
    fac.to_parquet(root / "factors.parquet", index=False)

    pd.DataFrame({c: rng.normal(0, .01, len(trace)) for c in BOND},
                 index=pd.Index(trace, name="date")).to_parquet(root / "bbw_factors.parquet")
    for bm in ("bns", "cls"):
        pd.DataFrame({f"{c}_{bm}": rng.normal(0, .01, len(trace)) for c in TWIN},
                     index=pd.Index(trace, name="date")).to_parquet(root / f"bbw_factors_{bm}.parquet")
    pd.DataFrame({"date": trace, **{c: rng.normal(0, 1, len(trace)) for c in LIQ}}).to_parquet(
        root / "illiq_factors.parquet", index=False)

    ext = pd.DataFrame({"date": ext_dates, **{c: rng.normal(0, .01, len(ext_dates)) for c in EXT}})
    monkeypatch.setattr(extended_factors, "load_extended_bbw", lambda *a, **k: ext.copy())
    fm = mr._factor_matrix(root)
    fm.to_parquet(root / "factors_merged.parquet", index=False)
    return root, ext


def _rewrite(root: Path, change) -> None:
    fm = pd.read_parquet(root / "factors_merged.parquet")
    change(fm)
    fm.to_parquet(root / "factors_merged.parquet", index=False)


def test_step4_matrix_passes_and_the_bundle_verifies(tmp_path, monkeypatch):
    root, ext = _write_blocks(tmp_path / "blocks" / "unit", monkeypatch)
    monkeypatch.setattr(mr.cfg, "BLOCKS_DIR", tmp_path / "blocks")
    out = tmp_path / "release"
    assert mr.release_monthly_factors("unit", out, "2099") == 0

    with zipfile.ZipFile(out / "osbap_monthly_factors_2099.zip") as z:
        assert set(z.namelist()) == {"monthly_factors_2099.parquet", "monthly_factors_2099.csv",
                                     "README.md", "PROVENANCE.json", "MANIFEST.json"}
        man = json.loads(z.read("MANIFEST.json"))
        for name, info in man["members"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == info["sha256"], name
        pub = pd.read_parquet(z.open("monthly_factors_2099.parquet"))
        readme = z.read("README.md").decode("utf-8")

    fm = pd.read_parquet(root / "factors_merged.parquet")
    assert not set(mr.MONTHLY_DERIVED) & set(pub.columns)
    assert set(pub.columns) == set(fm.columns) - set(mr.MONTHLY_DERIVED)
    for c in pub.columns:                                   # the file is the matrix, unchanged
        pd.testing.assert_series_equal(pub[c], fm[c], check_names=False)

    pre = pub["date"] < "2002-08-31"
    e = pub.loc[pre, ["date"]].merge(ext, on="date", how="left")
    for col, src in (("mktb", "MKTB"), ("crfx", "CRFx"), ("defb", "DEFB"), ("term_cls", "TERM_cls")):
        np.testing.assert_array_equal(pub.loc[pre, col].to_numpy(), e[src].to_numpy())
    assert pub.loc[pre, "mktb"].notna().all()               # 1990-01 .. 2002-07, all Lehman-ICE
    assert pub.loc[pre, ["lrf", "lrfx", "lrf_bns", "psb", "rsj"]].isna().all().all()
    assert pub.loc[pub["date"] > "2003-12-31", "mktb"].notna().all()   # TRACE after the ICE end
    assert "2099 vintage" in readme and "`mktb` is empty in" in readme and "`amd` from `ami`" in readme


@pytest.mark.parametrize("poison, expect", [
    ("vix", "factors.parquet"),
    ("mktb_after", "bbw_factors.parquet"),
    ("crf_bns_after", "bbw_factors_bns.parquet"),
    ("amd", "illiq_factors.parquet"),
    ("mktb_before", "Lehman-ICE"),
    ("lrf_before", "need TRACE prices"),
    ("no_backfill", "Lehman-ICE"),
])
def test_a_matrix_that_is_not_its_blocks_is_refused(tmp_path, monkeypatch, poison, expect):
    root, _ = _write_blocks(tmp_path / "blocks" / "unit", monkeypatch)
    monkeypatch.setattr(mr.cfg, "BLOCKS_DIR", tmp_path / "blocks")
    after, before = (lambda fm: fm["date"] >= "2002-08-31"), (lambda fm: fm["date"] < "2002-08-31")
    changes = {
        "vix": lambda fm: fm.__setitem__("vix", fm["vix"] + 1e-12),
        "mktb_after": lambda fm: fm.loc.__setitem__((after(fm), "mktb"), fm.loc[after(fm), "mktb"] * 1.0001),
        "crf_bns_after": lambda fm: fm.loc.__setitem__((after(fm), "crf_bns"), 0.0),
        "amd": lambda fm: fm.__setitem__("amd", fm["amd"].shift(1)),
        "mktb_before": lambda fm: fm.loc.__setitem__((before(fm), "mktb"), fm.loc[before(fm), "mktb"] + 1e-9),
        "lrf_before": lambda fm: fm.loc.__setitem__((before(fm), "lrf"), 0.001),
        "no_backfill": lambda fm: fm.loc.__setitem__((before(fm), "mktb"), np.nan),
    }
    _rewrite(root, changes[poison])
    with pytest.raises(SystemExit, match=expect):
        mr.release_monthly_factors("unit", tmp_path / "release", "2099")
    assert not (tmp_path / "release" / "osbap_monthly_factors_2099.zip").exists()


def test_a_column_no_group_describes_is_refused(tmp_path, monkeypatch):
    root, _ = _write_blocks(tmp_path / "blocks" / "unit", monkeypatch)
    monkeypatch.setattr(mr.cfg, "BLOCKS_DIR", tmp_path / "blocks")
    _rewrite(root, lambda fm: fm.__setitem__("newfactor", 0.0))
    with pytest.raises(SystemExit, match="no group describes"):
        mr.release_monthly_factors("unit", tmp_path / "release", "2099")


def test_every_group_column_is_named_once():
    cols = [c for _, cs in mr.MONTHLY_GROUPS for c in cs]
    assert len(cols) == len(set(cols))
    assert not set(cols) & set(mr.MONTHLY_DERIVED)


def test_what_all_includes_the_bundle():
    src = Path(mr.__file__).read_text(encoding="utf-8")
    assert 'if args.what in ("all", "monthly_factors"):' in src

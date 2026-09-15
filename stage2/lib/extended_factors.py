"""extended_factors.py -- the pre-2002-08 BBW factor backfill, as a PUBLIC fetched/cached input.

TRACE begins in 2002-07, so the BBW factor columns (mktb/drf/crf/mktbx/drfx/crfx/term) have no
TRACE-based history before 2002-08-31. They are backfilled from an extended series (1973-2023)
estimated on pre-TRACE bond data:

  * the **Lehman Brothers Fixed Income Data**, also known as the Warga Fixed Income data; and
  * the investment-grade and high-yield **Bank of America (BAML) constituent bonds** distributed
    by the **Intercontinental Exchange (ICE)**.

Those underlying bond data are licensed and cannot be redistributed. The finished monthly FACTOR
SERIES can be, and is published at openbondassetpricing.com -- so the backfill is available to
everyone even though the data behind it is not. It is the "modified" variant: no LRF leg, matching
the factor set this pipeline builds from TRACE.

This module is that seam. It loads the series from a local cache (under `data/`), falling back to
the published download (`cfg.BBW_EXTENDED_URL`), so a clone builds the full factor history from
public sources alone.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

import _stage2_settings as cfg

CACHE = cfg.BBW_EXTENDED_CACHE
COLS = ["MKTB", "DRF", "CRF", "MKTBx", "DRFx", "CRFx", "TERM"]
# DEFB/TERMB were added to the published series after the first release. They are read when
# present and skipped when not, so this module loads BOTH the seven-column file published
# through 2026-07 and the nine-column file that supersedes it.
OPTIONAL_COLS = ["DEFB", "TERMB"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """['date'] + the 7 factor columns, dates coerced to month-end -- whether `date` arrived as the
    index (the published parquet's layout) or as a column. Raises if dates or factors are missing."""
    if "date" not in df.columns:
        df = df.reset_index()                     # published member ships date as a DatetimeIndex
    if "date" not in df.columns:
        raise ValueError(f"extended-BBW series has no 'date' index/column; got {list(df.columns)}")
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"extended-BBW series is missing factor columns {missing}")
    have = [c for c in OPTIONAL_COLS if c in df.columns]
    # The benchmark twins ride through here too. This selection is a WHITELIST -- a column absent
    # from it is silently dropped even when the published file carries it, which is how a correct
    # download can still produce a series with no benchmark history.
    twins = [c for c in getattr(cfg, "BBW_BENCHMARK_COLS", ()) if c in df.columns]
    df = df[["date"] + COLS + have + twins].copy()
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    return df


def _missing_twins(df: pd.DataFrame) -> list[str]:
    """Benchmark twins the settings say this series must carry and it does not.

    ❗THE CACHE IS THE TRAP, exactly as it is in lib/quote.py. This module returns the cached
    parquet whenever one exists, so repointing BBW_EXTENDED_URL changes nothing on a machine that
    already holds the old file -- and the failure is invisible: the splice still works, the tret
    factors are all present, and only the ALTERNATIVE benchmarks silently lose their pre-2002
    history. Betas on them then start in 2003 instead of 1997 with nothing anywhere saying why.
    """
    if not getattr(cfg, "BBW_HAS_BENCHMARKS", False):
        return []
    return [c for c in cfg.BBW_BENCHMARK_COLS if c not in df.columns]


def load_extended_bbw(force_fetch: bool = False) -> pd.DataFrame:
    """The extended BBW factor series: ['date', MKTB, DRF, CRF, MKTBx, DRFx, CRFx, TERM] plus
    DEFB/TERMB when the published file carries them, one row per month-end (1973-02 .. 2023-01).

    Reads the cached parquet under data/. If absent, corrupt, or `force_fetch`, downloads the published
    zip from `cfg.BBW_EXTENDED_URL` and caches the NORMALIZED frame. The cache is written only AFTER
    normalization: the published member is date-INDEXED, so caching it raw with index=False silently
    drops every date.
    """
    if CACHE.exists() and not force_fetch:
        try:
            cached = _normalize(pd.read_parquet(CACHE))
        except ValueError:
            cached = None                          # corrupt/legacy cache -> refetch below
        if cached is not None:
            gone = _missing_twins(cached)
            if not gone:
                return cached
            print(f"[bbw] cached {CACHE.name} is missing {gone} -- refetching from "
                  f"BBW_EXTENDED_URL. The cache predates a settings change.", flush=True)

    url = getattr(cfg, "BBW_EXTENDED_URL", None)
    if not url or "TODO" in url:
        raise FileNotFoundError(
            f"No usable cached extended-BBW factors at {CACHE} and cfg.BBW_EXTENDED_URL is not set. "
            "Set the published OSBAP URL or restore the manifest-fingerprinted cache."
        )
    import requests
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        df = _normalize(pd.read_parquet(io.BytesIO(zf.read(cfg.BBW_EXTENDED_ZIPKEY))))
    gone = _missing_twins(df)
    if gone:
        # The download and the settings disagree. Stopping beats splicing a series that silently
        # has no benchmark history.
        raise RuntimeError(
            f"the file at BBW_EXTENDED_URL is missing {gone}.\n"
            f"  url : {url}\n"
            f"  If you meant the nine-column series, set BBW_HAS_BENCHMARKS = False alongside "
            f"reverting BBW_EXTENDED_URL.")
    cfg.ensure_dirs()
    df.to_parquet(CACHE, index=False)
    return df

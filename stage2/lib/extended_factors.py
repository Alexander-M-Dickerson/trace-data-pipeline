"""extended_factors.py -- the pre-2002-08 BBW factor backfill, as a PUBLIC fetched/cached input.

The BBW factor columns in the golden `factors.parquet` (mktb/drf/crf/mktbx/drfx/crfx/term) are, before
2002-08-31, the "modified" (no-LRF) extended BBW factors built in the PRIVATE `lehman-ice` repo from
licensed Lehman/Warga + ICE/BAML data (1973-2023). The raw data is not redistributable, but the finished
factor series IS freely licensed and is published to openbondassetpricing.com.

This module is the public seam: it loads the extended factor series from a local cache (under `data/`),
falling back to the published OSBAP download (`cfg.BBW_EXTENDED_URL`). This is what lets the monthly
factor build source its pre-2002-08 BBW backfill WITHOUT depending on the private golden
`factors.parquet` (see HANDOFF_FABLE.md §Wrap-up W1/W2, assumptions.md A18/A10).

Provenance + bit-for-bit reproduction: `lehman-ice/BBW_MODIFIED.md` + `lehman-ice/verify_bbw_modified.py`.
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
    df = df[["date"] + COLS + have].copy()
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    return df


def load_extended_bbw(force_fetch: bool = False) -> pd.DataFrame:
    """The extended BBW factor series: ['date', MKTB, DRF, CRF, MKTBx, DRFx, CRFx, TERM] plus
    DEFB/TERMB when the published file carries them, one row per month-end (1973-02 .. 2023-01).

    Reads the cached parquet under data/. If absent, corrupt, or `force_fetch`, downloads the published
    zip from `cfg.BBW_EXTENDED_URL` and caches the NORMALIZED frame. The cache is written only AFTER
    normalization: the published member is date-INDEXED, so caching it raw with index=False silently
    drops every date (debug.md M11).
    """
    if CACHE.exists() and not force_fetch:
        try:
            return _normalize(pd.read_parquet(CACHE))
        except ValueError:
            pass                                   # corrupt/legacy cache (M11) -> refetch below

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
    cfg.ensure_dirs()
    df.to_parquet(CACHE, index=False)
    return df

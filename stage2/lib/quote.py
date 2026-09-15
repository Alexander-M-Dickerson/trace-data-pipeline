"""quote.py -- the 1997-2002 quote-return panel (openbondassetpricing.com published data),
downloaded once and cached to data/quote_returns_quantlib.parquet. Used by steps 3/4/5 exactly as
upstream `load_bond_data_from_url` uses the live download.

❗THE CACHE IS THE TRAP. This module returns the cached copy whenever one exists, so repointing
`QUOTE_URL` at a new file changes NOTHING on a machine that already has the old one -- the build
keeps running on stale data and every number it produces looks entirely reasonable. That is why
`load_quote` validates what it loaded against `QUOTE_REQUIRED_COLS` / `QUOTE_HAS_BENCHMARKS` and
re-fetches, rather than trusting that the file on disk is the file the settings describe.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

import _stage2_settings as cfg

QUOTE_CACHE = cfg.DATA_DIR / cfg.QUOTE_ZIPKEY


def _missing(df: pd.DataFrame) -> list[str]:
    """Columns the settings say this file must have and it does not."""
    want = list(cfg.QUOTE_REQUIRED_COLS)
    if getattr(cfg, "QUOTE_HAS_BENCHMARKS", False):
        want += list(cfg.QUOTE_BENCHMARK_COLS)
    have = set(df.columns)
    return [c for c in want if c not in have]


def _fetch() -> pd.DataFrame:
    import requests

    resp = requests.get(cfg.QUOTE_URL, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        return pd.read_parquet(io.BytesIO(zf.read(cfg.QUOTE_ZIPKEY)))


def load_quote(force_fetch: bool = False) -> pd.DataFrame:
    """The quote-returns frame (cusip_id, date, ret_vw, tret, cs, bbtm, sze, + the tret_* block).

    A cached copy that does not match what the settings describe is treated as stale and replaced,
    because the alternative is a silently short pre-history.
    """
    df = None
    if QUOTE_CACHE.exists() and not force_fetch:
        df = pd.read_parquet(QUOTE_CACHE)
        gone = _missing(df)
        if gone:
            print(f"[quote] cached {QUOTE_CACHE.name} is missing {gone} -- refetching from "
                  f"QUOTE_URL. The cache predates a settings change.", flush=True)
            df = None

    if df is None:
        df = _fetch()
        gone = _missing(df)
        if gone:
            # The download and the settings disagree. Failing here is the point: continuing would
            # mean building on a file that is not the one this configuration claims to use.
            raise RuntimeError(
                f"the file at QUOTE_URL is missing {gone}.\n"
                f"  url  : {cfg.QUOTE_URL}\n"
                f"  If you meant to use the nine-column panel, set QUOTE_HAS_BENCHMARKS = False "
                f"in _stage2_settings.py alongside reverting QUOTE_URL.")
        cfg.ensure_dirs()
        df.to_parquet(QUOTE_CACHE, index=False)

    return df

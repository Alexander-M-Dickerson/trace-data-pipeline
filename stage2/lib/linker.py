"""linker.py -- the published bond->firm linker, and the WINDOW the panel joins it on.

Stage 1 attaches `permno`/`permco`/`gvkey` from this same file and Stage 2 has inherited those ids
through the pin ever since. That was fine while there was one window. There are two:

    w0 / w1   EVIDENCE window -- WHEN the mapping is provable.
              greatest(bond life start, crsp_begdt) .. least(bond life end, crsp_enddt).
              Right when you are joining EQUITY-side data: outside it there is no listed firm.

    i0 / i1   IDENTITY window -- WHOSE bond this is. Contains the evidence window and opens
              outward only where no dated successor or acquisition contradicts it.
              Right when the firm id is a LABEL: grouping, within-firm sorts, issuer fixed effects.

A bond-month panel LABELS firms. It does not join equity data. So this module joins `i0`/`i1`,
which is what the linker bundle's own SCHEMA.md and README.md name as the default for panel
work (both ship inside the zip at LINKER_URL).

❗WHY THIS MODULE EXISTS AT ALL. Until 2026-09 the identity window shipped only on an unpublished
file, and the published README showed `BETWEEN w0 AND w1` as the join. Stage 1 followed it; our own
relink used the identity window. Same linker, different question, and the two panels disagreed about
firm membership on 2.14% of TRACE-era bond-months -- 43,779 rows where one carried a firm id and the
other did not. Single sorts never touch `permno` and matched exactly; within-firm sorts group by it,
so 8,479 firm-months existed in one panel and not the other and every within-firm portfolio moved.

❗THE CACHE IS THE TRAP, exactly as in lib/quote.py. A cached linker that predates the identity
window would let this module fall back to the evidence window and reproduce the original bug
silently. It does not fall back: a cache without `i0`/`i1` is stale by definition and is replaced,
and a DOWNLOAD without them is a hard error -- because it would mean the URL points at a linker
built before FL-DR44 and no join here can be trusted.
"""
from __future__ import annotations

import io
import os
import zipfile

import pandas as pd

import _stage2_settings as cfg

LINKER_CACHE = cfg.DATA_DIR / "fl_linker.parquet"


def _missing(df: pd.DataFrame) -> list[str]:
    want = list(cfg.LINKER_REQUIRED_COLS) + list(cfg.LINKER_WINDOW)
    have = set(df.columns)
    return [c for c in want if c not in have]


def _fetch() -> pd.DataFrame:
    import requests

    resp = requests.get(cfg.LINKER_URL, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        return pd.read_parquet(io.BytesIO(zf.read(cfg.LINKER_ZIPKEY)))


def load_linker(force_fetch: bool = False) -> pd.DataFrame:
    """The linker, guaranteed to carry the window this panel joins on.

    `STAGE2_LINKER_FILE` points at a local parquet and wins over both cache and URL -- the same
    override convention the other stage-2 inputs use. It exists because a bundle published before
    the identity window shipped is correctly REFUSED below, which would otherwise make it
    impossible to build at all in the window between fixing the linker and uploading it.
    """
    override = os.environ.get("STAGE2_LINKER_FILE")
    if override:
        df = pd.read_parquet(override)
        gone = _missing(df)
        if gone:
            raise RuntimeError(f"STAGE2_LINKER_FILE={override} is missing {gone}")
        print(f"[linker] STAGE2_LINKER_FILE override: {override}", flush=True)
        return df

    df = None
    if LINKER_CACHE.exists() and not force_fetch:
        df = pd.read_parquet(LINKER_CACHE)
        gone = _missing(df)
        if gone:
            print(f"[linker] cached {LINKER_CACHE.name} is missing {gone} -- refetching from "
                  f"LINKER_URL. The cache predates the identity window (FL-DR44).", flush=True)
            df = None

    if df is None:
        df = _fetch()
        gone = _missing(df)
        if gone:
            raise RuntimeError(
                f"the linker at LINKER_URL is missing {gone}.\n"
                f"  url    : {cfg.LINKER_URL}\n"
                f"  window : {cfg.LINKER_WINDOW}  (SCHEMA.md inside the zip says which window is which)\n"
                f"  This build joins the IDENTITY window because a bond-month panel labels firms.\n"
                f"  A linker without it predates FL-DR44; falling back to w0/w1 would silently\n"
                f"  reintroduce the 2.14% firm-membership disagreement that gate exists to stop.")
        cfg.ensure_dirs()
        df.to_parquet(LINKER_CACHE, index=False)

    return df


def register(con, table: str = "_linker") -> tuple[str, str]:
    """Register the linker on `con` and return the declared window's (lo, hi) column names.

    The caller writes its own SQL -- step 1 must join identifiers into their OWN block, never
    through `t_src`, because adding columns there changes physical row order and several
    order-sensitive float32 illiquidity kernels downstream move with it.
    """
    df = load_linker()
    con.register(table, df)
    print(f"[linker] {len(df):,} rows, joining on {cfg.LINKER_WINDOW} "
          f"({'identity' if cfg.LINKER_WINDOW == ('i0', 'i1') else 'evidence'} window)", flush=True)
    return cfg.LINKER_WINDOW

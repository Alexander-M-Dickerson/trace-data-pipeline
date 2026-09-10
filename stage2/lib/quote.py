"""quote.py -- the 1997-2002 quote-return panel (openbondassetpricing.com published data),
downloaded once and cached to data/quote_returns_quantlib.parquet. Used by steps 3/4/5 exactly as
upstream `load_bond_data_from_url` uses the live download."""
from __future__ import annotations

import io
import zipfile

import pandas as pd

import _stage2_settings as cfg

QUOTE_CACHE = cfg.DATA_DIR / cfg.QUOTE_ZIPKEY


def load_quote(force_fetch: bool = False) -> pd.DataFrame:
    """The quote-returns frame (columns incl. cusip_id, date, ret_vw, tret, cs, bbtm, sze)."""
    if QUOTE_CACHE.exists() and not force_fetch:
        return pd.read_parquet(QUOTE_CACHE)
    import requests

    resp = requests.get(cfg.QUOTE_URL, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        df = pd.read_parquet(io.BytesIO(zf.read(cfg.QUOTE_ZIPKEY)))
    cfg.ensure_dirs()
    df.to_parquet(QUOTE_CACHE, index=False)
    return df

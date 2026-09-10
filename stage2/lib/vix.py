"""vix.py -- daily VIX from WRDS (cboe.cboe), faithful to upstream `_fetch_vix_from_wrds`:
vix_scaled = vix / sqrt(12) / 100. Fetched once, cached to data/cboe_vix.parquet (fingerprinted in
the manifest). No vintage truncation needed: the risk step only uses VIX at trade dates present in
the daily input (<= 2025-03-31), and the CBOE history is static.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _stage2_settings as cfg

VIX_CACHE = cfg.DATA_DIR / "cboe_vix.parquet"


def load_vix(force_fetch: bool = False) -> pd.DataFrame:
    """['date', 'vix'] with the upstream scaling (vix / sqrt(12) / 100), date-sorted."""
    if VIX_CACHE.exists() and not force_fetch:
        return pd.read_parquet(VIX_CACHE)

    import os

    import wrds

    _u = os.environ.get("WRDS_USERNAME", "")
    if not _u:
        raise RuntimeError("WRDS_USERNAME is not set. Stage 2 needs it only for its first run, to fetch and cache Treasury returns, Fama-French factors and VIX. Set it in config.py or as an environment variable.")
    username = _u
    db = wrds.Connection(wrds_username=username)
    try:
        vix = db.raw_sql("SELECT date, vix FROM cboe.cboe WHERE vix IS NOT NULL")
    finally:
        db.close()
    vix["date"] = pd.to_datetime(vix["date"], errors="coerce")
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
    vix = vix.dropna(subset=["date", "vix"]).sort_values("date").reset_index(drop=True)
    vix["vix"] = vix["vix"] / np.sqrt(12) / 100
    cfg.ensure_dirs()
    vix[["date", "vix"]].to_parquet(VIX_CACHE, index=False)
    return vix[["date", "vix"]]

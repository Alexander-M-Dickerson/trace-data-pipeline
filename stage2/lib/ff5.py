"""ff5.py -- Fama-French 5 factors + RF from WRDS (ff.fivefactors_monthly), fetch-once cached.
Faithful to upstream `process_bond_data.fetch_ff5_factors` (decimal units, MonthEnd dates)."""
from __future__ import annotations

import pandas as pd

import _stage2_settings as cfg

FF5_CACHE = cfg.DATA_DIR / "ff5_factors.parquet"


def load_ff5(force_fetch: bool = False) -> pd.DataFrame:
    """['date', 'mktrf', 'smb', 'hml', 'rmw', 'cma', 'rf'], date = calendar month-end."""
    if FF5_CACHE.exists() and not force_fetch:
        return pd.read_parquet(FF5_CACHE)

    import os

    import wrds
    from pandas.tseries.offsets import MonthEnd

    _u = os.environ.get("WRDS_USERNAME", "")
    if not _u:
        raise RuntimeError("WRDS_USERNAME is not set. Stage 2 needs it only for its first run, to fetch and cache Treasury returns, Fama-French factors and VIX. Set it in config.py or as an environment variable.")
    db = wrds.Connection(wrds_username=_u)
    try:
        ff5 = db.raw_sql("SELECT dateff, mktrf, smb, hml, rmw, cma, rf FROM ff.fivefactors_monthly")
    finally:
        db.close()
    ff5["date"] = pd.to_datetime(ff5["dateff"], errors="coerce") + MonthEnd(0)
    cols = ["mktrf", "smb", "hml", "rmw", "cma", "rf"]
    for c in cols:
        ff5[c] = pd.to_numeric(ff5[c], errors="coerce")
    ff5 = ff5.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)[["date"] + cols]
    cfg.ensure_dirs()
    ff5.to_parquet(FF5_CACHE, index=False)
    return ff5

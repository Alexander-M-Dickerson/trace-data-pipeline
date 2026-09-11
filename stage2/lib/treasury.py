"""treasury.py -- duration-matched Treasury returns (`tret`), faithful to upstream
the reference implementation's treasury load + interpolation.

Source series (upstream fetches live from WRDS every run):
  - CRSP Fixed Term Treasury index monthly returns: crsp.tfz_idx (FIXEDTERM family -> terms
    1,2,5,7,10,20,30y) joined to crsp.tfz_mth_ft (tmretadj/100)
  - Fama-French monthly rf (ff.factors_monthly) as the 1/12-year tenor
Wide frame: index=month-end date, columns=[1/12, 1, 2, 5, 7, 10, 20, 30].

We fetch ONCE and cache to data/crsp_treasury_returns.parquet (fingerprinted in the run manifest);
the series is historical and stable. `interpolate_tret` is a verbatim numpy port: flat below the
shortest / above the longest tenor, linear in between, one-sided fallback when a bracket is NaN.
mod_dur is rounded to 2dp by the CALLER with numpy (pandas .round semantics), never in SQL.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import _stage2_settings as cfg

TREASURY_CACHE = cfg.DATA_DIR / "crsp_treasury_returns.parquet"
_TERM_ORDER = [1 / 12, 1, 2, 5, 7, 10, 20, 30]


def fetch_treasury_returns(wrds_username: str | None = None) -> pd.DataFrame:
    """One-time WRDS fetch (upstream load_treasury_returns, verbatim queries). Long format out."""
    import os

    import wrds
    from pandas.tseries.offsets import MonthEnd

    # (WRDS auth: .pgpass holds the password)
    _u = os.environ.get("WRDS_USERNAME", "")
    if not _u:
        raise RuntimeError("WRDS_USERNAME is not set. Stage 2 needs it only for its first run, to fetch and cache Treasury returns, Fama-French factors and VIX. Set it in config.py or as an environment variable.")
    username = wrds_username or _u
    db = wrds.Connection(wrds_username=username)
    try:
        headers = db.raw_sql("SELECT * FROM crsp.tfz_idx")
        headers = headers[headers.tidxfam == "FIXEDTERM"].copy()
        headers["term"] = [1, 2, 5, 7, 10, 20, 30]
        headers = headers[["kytreasnox", "term"]]

        tr = db.raw_sql("SELECT kytreasnox, mcaldt, tmyearstm, tmretadj FROM crsp.tfz_mth_ft")
        tr.columns = ["kytreasnox", "date", "t_tmt", "t_ret"]
        tr = tr.merge(headers, on="kytreasnox", how="left")
        tr["kytreasnox"] = tr["kytreasnox"].astype(int)
        tr["date"] = pd.to_datetime(tr["date"]) + MonthEnd(0)
        tr["t_ret"] = tr["t_ret"] / 100

        ff = db.get_table(library="ff", table="factors_monthly", columns=["dateff", "rf"]).dropna()
        ff["date"] = pd.to_datetime(ff["dateff"]) + MonthEnd(0)
        ff = ff[["date", "rf"]].rename(columns={"rf": "t_ret"})
        ff["term"] = 1 / 12
    finally:
        db.close()

    long = pd.concat([tr[["date", "term", "t_ret"]], ff[["date", "term", "t_ret"]]],
                     ignore_index=True)
    return long


def load_tret_wide(force_fetch: bool = False, mode: str | None = None) -> pd.DataFrame:
    """The wide (date x term) Treasury return frame, from cache or a one-time WRDS fetch.

    Truncated at the mode-aware cutoff cfg.tret_max_date(mode): golden freezes to '2024-12-31' to
    reproduce the golden run's data vintage (its Dec-2025 WRDS fetch had CRSP tfz_mth_ft only through
    2024-12; golden tret is NULL after); ours resolves to None -> no cap -> use every cached/WRDS
    month (the cache holds terms through 2025-12 as of 2026-07). See _stage2_settings.MODE_PINS.
    """
    if TREASURY_CACHE.exists() and not force_fetch:
        long = pd.read_parquet(TREASURY_CACHE)
    else:
        cfg.ensure_dirs()
        long = fetch_treasury_returns()
        long.to_parquet(TREASURY_CACHE, index=False)
    cap = cfg.tret_max_date(mode)
    if cap is not None:
        long = long[long["date"] <= pd.Timestamp(cap)]
    wide = long.pivot_table(index="date", columns="term", values="t_ret")
    terms = [t for t in _TERM_ORDER if t in wide.columns]
    return wide[terms]


def interpolate_tret(unique_pairs: pd.DataFrame, dftret: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port of upstream interpolate_treasury_returns.

    unique_pairs: columns ['date', 'mod_dur'] (mod_dur ALREADY rounded to 2dp, numpy semantics).
    Returns ['date', 'mod_dur', 'tret'].
    """
    terms = sorted(dftret.columns.to_list())
    terms_arr = np.asarray(terms, dtype=float)

    result = unique_pairs.copy()
    result = result.merge(dftret[terms], left_on="date", right_index=True, how="left")

    mod_dur = result["mod_dur"].to_numpy(dtype=float)
    trets = result[terms].to_numpy(dtype=float)
    tret = np.full(mod_dur.shape, np.nan, dtype=float)

    valid = ~np.isnan(mod_dur) & ~np.all(np.isnan(trets), axis=1)
    if np.any(valid):
        md_v = mod_dur[valid]
        trets_v = trets[valid]
        low_mask = md_v <= terms_arr[0]
        high_mask = md_v >= terms_arr[-1]
        mid_mask = (~low_mask) & (~high_mask)
        tret_v = np.full(md_v.shape, np.nan, dtype=float)

        if np.any(low_mask):
            tret_v[low_mask] = trets_v[low_mask, 0]
        if np.any(high_mask):
            tret_v[high_mask] = trets_v[high_mask, -1]
        if np.any(mid_mask):
            md_mid = md_v[mid_mask]
            trets_mid = trets_v[mid_mask]
            upper_idx = np.searchsorted(terms_arr, md_mid, side="right")
            lower_idx = upper_idx - 1
            row_idx = np.arange(md_mid.shape[0])
            lower_ret = trets_mid[row_idx, lower_idx]
            upper_ret = trets_mid[row_idx, upper_idx]

            both = (~np.isnan(lower_ret)) & (~np.isnan(upper_ret))
            lower_only = (~np.isnan(lower_ret)) & np.isnan(upper_ret)
            upper_only = np.isnan(lower_ret) & (~np.isnan(upper_ret))
            tret_mid = np.full(md_mid.shape, np.nan, dtype=float)
            if np.any(both):
                denom = terms_arr[upper_idx[both]] - terms_arr[lower_idx[both]]
                w = (md_mid[both] - terms_arr[lower_idx[both]]) / denom
                tret_mid[both] = lower_ret[both] + w * (upper_ret[both] - lower_ret[both])
            if np.any(lower_only):
                tret_mid[lower_only] = lower_ret[lower_only]
            if np.any(upper_only):
                tret_mid[upper_only] = upper_ret[upper_only]
            tret_v[mid_mask] = tret_mid
        tret[valid] = tret_v

    result["tret"] = tret
    return result[["date", "mod_dur", "tret"]]


def attach_tret(df: pd.DataFrame, dftret: pd.DataFrame) -> pd.DataFrame:
    """Round mod_dur to 2dp (numpy), interpolate on unique (date, mod_dur) pairs, merge, drop mod_dur."""
    df = df.copy()
    df["mod_dur"] = df["mod_dur"].round(2)
    pairs = df[["date", "mod_dur"]].drop_duplicates().reset_index(drop=True)
    tret_df = interpolate_tret(pairs, dftret)
    df = df.merge(tret_df, on=["date", "mod_dur"], how="left")
    return df.drop(columns=["mod_dur"])

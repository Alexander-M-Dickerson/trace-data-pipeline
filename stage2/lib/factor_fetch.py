"""factor_fetch.py -- the PUBLIC factor fetchers behind steps/compute_factors (W3, closes A10).

Each fetcher ports its upstream `stage2/create_factors.py` counterpart verbatim (same source, same
transforms, same output columns) and is fetch-once: the normalized frame is cached under
`cfg.FACTOR_CACHE_DIR/<name>.parquet` with a sidecar `<name>.meta.json` (url, fetched_utc, rows,
span, sha256) so any manifest can prove which vintage a build consumed. `force=True` re-pulls --
that is how a repo user updates factors; bump the vintage URLs in `_stage2_settings` for sources that
version their filenames (HKM, Ludvigson).

A fresh pull does NOT bit-match the golden pinned vintage: FRED CPIAUCSL is re-seasonally-adjusted
and EPU back-renormalized every release; HKM/Ludvigson revise history (A10). See
context/factors_public_divergence.md. No pandas_datareader dependency: the French zip and FRED's
fredgraph CSV endpoint parse directly.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

import _stage2_settings as cfg

REQUEST_TIMEOUT_S = 300

# Ludvigson zip member stem -> output column (create_factors.py:523-527)
_LUDVIGSON_FILE_MAP = {
    "FinancialUncertaintyToCirculate": "uncf",
    "MacroUncertaintyToCirculate": "unc",
    "RealUncertaintyToCirculate": "uncr",
}
# EPU "Indices" sheet column -> output column (create_factors.py:636-640)
_EPU_COL_MAP = {
    "1. Economic Policy Uncertainty": "epu",
    "2. Monetary policy": "epum",
    "3. Taxes": "eput",
}


def _get(url: str) -> bytes:
    import requests
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.content


def _cached(name: str, url: str | Callable[[], str], builder: Callable[[str], pd.DataFrame],
            force: bool) -> pd.DataFrame:
    """Fetch-once framework: cache the normalized frame + a sidecar meta fingerprint.
    `url` may be a callable (lazy discovery for sources that rotate filenames); it resolves only on
    a cache miss and the RESOLVED url is what lands in the sidecar meta."""
    cache = cfg.FACTOR_CACHE_DIR / f"{name}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    url = url() if callable(url) else url
    df = builder(url)
    assert "date" in df.columns and df["date"].notna().all(), \
        f"{name}: fetched frame must be non-null on 'date'"
    if not df["date"].is_unique:
        # e.g. the 2026 HKM vintage ships 2025-01 twice; upstream keeps the frame verbatim and the
        # panel-level drop_duplicates(keep='first') resolves it (create_factors.py:876)
        n_dup = int(df["date"].duplicated().sum())
        print(f"[factor_fetch] {name}: {n_dup} duplicate month(s) kept verbatim "
              "(panel dedup keep-first resolves)")
    cfg.ensure_dirs()
    df.to_parquet(cache, index=False)
    from lib import manifest as mf
    meta = {"name": name, "url": url,
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "rows": len(df), "columns": list(df.columns),
            "span": [str(df["date"].min().date()), str(df["date"].max().date())],
            "sha256": mf.sha256_file(cache)}
    (cfg.FACTOR_CACHE_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
    return df


def _fred_csv(series: tuple[str, ...], start: str) -> pd.DataFrame:
    """One FRED fredgraph.csv pull -> ['date', *series] with NaN for '.', clipped to start."""
    url = cfg.FRED_CSV_URL.format(ids=",".join(series))
    df = pd.read_csv(io.BytesIO(_get(url)), na_values=".")
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    for c in series:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)


# --- the six upstream fetchers ----------------------------------------------------------------------

def fetch_ff5(force: bool = False) -> pd.DataFrame:
    """['date','mktrf','smb','hml','rf'] decimal, month-end -- Ken French 5-factor monthly table
    (create_factors.fetch_ff5_factors; rmw/cma dropped, /100)."""
    def build(url: str) -> pd.DataFrame:
        raw = _get(url)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
        rows = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if re.fullmatch(r"\d{6}", parts[0]):
                rows.append(parts[:7])
            elif rows:
                break                      # monthly table ended (the annual section follows)
        df = pd.DataFrame(rows, columns=["yyyymm", "mktrf", "smb", "hml", "rmw", "cma", "rf"])
        df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m") + MonthEnd(0)
        for c in ("mktrf", "smb", "hml", "rf"):
            df[c] = df[c].astype(float) / 100.0        # percent -> decimal
        return df[["date", "mktrf", "smb", "hml", "rf"]]
    return _cached("ff5_french", cfg.FF5_URL, build, force)


def fetch_vix_monthly(force: bool = False) -> pd.DataFrame:
    """['date','vix','dvix','dvixlag'] -- EOM VIX (VXO fallback pre-1990) scaled /100/sqrt(12);
    dvix = first diff (first month: intra-month last-first) (create_factors.fetch_vix_monthly)."""
    def build(url: str) -> pd.DataFrame:
        import os

        import wrds
        _u = os.environ.get("WRDS_USERNAME", "")
        if not _u:
            raise RuntimeError("WRDS_USERNAME is not set. Stage 2 needs it only for its first run, to fetch and cache Treasury returns, Fama-French factors and VIX. Set it in config.py or as an environment variable.")
        db = wrds.Connection(wrds_username=_u)
        try:
            df = db.raw_sql("SELECT date, vix, vxo FROM cboe.cboe "
                            "WHERE date >= '1986-01-02' ORDER BY date")
        finally:
            db.close()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["vix"] = pd.to_numeric(df["vix"], errors="coerce").fillna(
            pd.to_numeric(df["vxo"], errors="coerce"))
        df = df.dropna(subset=["date", "vix"]).sort_values("date").reset_index(drop=True)

        df["month"] = df["date"].dt.to_period("M")
        first = df[df["month"] == df["month"].min()]
        dvix_first_raw = first["vix"].iloc[-1] - first["vix"].iloc[0]

        vix = df.groupby("month", observed=True).last().reset_index()
        vix["date"] = vix["month"].dt.to_timestamp() + MonthEnd(0)
        vix["vix"] = vix["vix"] / 100 / np.sqrt(12)     # decimal, monthly
        vix = vix.sort_values("date").reset_index(drop=True)
        vix["dvix"] = vix["vix"].diff()
        vix.loc[0, "dvix"] = dvix_first_raw / 100 / np.sqrt(12)
        vix["dvixlag"] = vix["dvix"].shift(1)
        return vix[["date", "vix", "dvix", "dvixlag"]]
    return _cached("vix_monthly", "wrds:cboe.cboe", build, force)


def fetch_cpi_credit(force: bool = False) -> pd.DataFrame:
    """['date','dcpi','cpi_vol6','credit','dcredit','lvl','ysp'] -- FRED CPIAUCSL + AAA/BAA + DGS
    curve (create_factors.fetch_cpi_credit: CPI lag-then-diff + 6m vol; (BAA-AAA)/12; EW lvl; 5y-1y)."""
    def build(url: str) -> pd.DataFrame:
        cpi = _fred_csv(("CPIAUCSL",), "1947-01-01").rename(columns={"CPIAUCSL": "cpi"})
        cpi["date"] = cpi["date"] + MonthEnd(0)
        cpi = cpi.sort_values("date").reset_index(drop=True)
        cpi["dcpi"] = cpi["cpi"].shift(1).diff()        # 1-month-lagged CPI level, first diff
        cpi["cpi_vol6"] = cpi["dcpi"].rolling(window=6, min_periods=6).std()
        cpi = cpi[["date", "dcpi", "cpi_vol6"]]

        yld = _fred_csv(("AAA", "BAA"), "1960-01-01").rename(columns={"AAA": "aaa", "BAA": "baa"})
        yld["date"] = yld["date"] + MonthEnd(0)
        yld["credit"] = yld["baa"] - yld["aaa"]
        yld = yld.dropna(subset=["aaa", "baa", "credit"]).sort_values("date").reset_index(drop=True)
        yld["credit"] = yld["credit"] / 12              # annual %-points -> monthly
        yld["dcredit"] = yld["credit"].diff()
        yld = yld[["date", "credit", "dcredit"]]

        tsy = _fred_csv(cfg.FRED_TSY_SERIES, "1960-01-01").set_index("date")
        tsy = tsy.resample("ME").last().ffill().reset_index()
        tsy.columns = ["date", "y1", "y2", "y3", "y5", "y7", "y10", "y20", "y30"]
        tsy["lvl"] = tsy[["y1", "y2", "y3", "y5", "y7", "y10", "y20", "y30"]].mean(axis=1) / 100
        tsy["ysp"] = (tsy["y5"] - tsy["y1"]) / 100
        tsy = tsy[["date", "lvl", "ysp"]]

        out = cpi.merge(yld, on="date", how="outer").merge(tsy, on="date", how="outer")
        return out.sort_values("date").reset_index(drop=True)
    return _cached("fred_cpi_credit", cfg.FRED_CSV_URL.format(ids="CPIAUCSL,AAA,BAA,DGS*"),
                   build, force)


def fetch_intermediary_capital(force: bool = False) -> pd.DataFrame:
    """['date','cptl','cptlt'] -- He-Kelly-Manela intermediary capital factors
    (create_factors.fetch_intermediary_capital; cptlt has RF subtracted later, at the merge)."""
    def build(url: str) -> pd.DataFrame:
        df = pd.read_csv(io.BytesIO(_get(url)))
        df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m") + MonthEnd(0)
        df = df.rename(columns={
            "intermediary_capital_risk_factor": "cptl",
            "intermediary_value_weighted_investment_return": "cptlt"})
        return df[["date", "cptl", "cptlt"]].sort_values("date").reset_index(drop=True)
    return _cached("hkm_intermediary", _hkm_url(), build, force)


def _hkm_url() -> str:
    """The current He-Kelly-Manela monthly-factor URL.

    The authors publish under a DATED filename (..._250627.csv) and change it on each
    release, so a hard-coded URL silently pins the panel to an old vintage -- and because
    a factor series that stops short truncates every beta estimated on it, that shows up
    as columns dying early rather than as an error. Same discovery approach as Ludvigson:
    use the configured URL while it is live, otherwise find the current one on the data
    page.

    As of 2026-09-10 the configured URL IS the current one -- the authors have not
    published past 2025-05, so `b_cptlt` legitimately lags a later frontier.
    """
    import requests
    try:
        r = requests.head(cfg.HKM_URL, timeout=60, allow_redirects=True)
        if r.status_code == 200:
            return cfg.HKM_URL
    except requests.RequestException:
        pass
    from urllib.parse import urljoin
    html = _get(cfg.HKM_INDEX).decode("utf-8", errors="replace")
    m = re.search(r'href="([^"]*He_Kelly_Manela_Factors_monthly[^"]*\.csv[^"]*)"', html)
    if not m:
        raise FileNotFoundError(
            f"He-Kelly-Manela monthly factor CSV not found on {cfg.HKM_INDEX}. "
            f"The configured HKM_URL is also unreachable. Update HKM_URL by hand.")
    url = urljoin(cfg.HKM_INDEX, m.group(1))
    print(f"[factor_fetch] HKM vintage rotated; discovered {url}")
    return url


def _ludvigson_url() -> str:
    """The current uncertainty-zip URL: the configured vintage if still live, else discovered from
    the index page (the site ROTATES the filename every update -- the golden's 202508 zip now 404s)."""
    import requests
    try:
        r = requests.head(cfg.LUDVIGSON_URL, timeout=60, allow_redirects=True)
        if r.status_code == 200:
            return cfg.LUDVIGSON_URL
    except requests.RequestException:
        pass
    from urllib.parse import urljoin
    html = _get(cfg.LUDVIGSON_INDEX).decode("utf-8", errors="replace")
    m = re.search(r'href="([^"]*MacroFinanceUncertainty[^"]*\.zip[^"]*)"', html)
    if not m:
        raise FileNotFoundError("no MacroFinanceUncertainty zip link on the Ludvigson index page")
    url = urljoin(cfg.LUDVIGSON_INDEX, m.group(1))
    print(f"[factor_fetch] ludvigson vintage rotated; discovered {url}")
    return url


def fetch_uncertainty(force: bool = False) -> pd.DataFrame:
    """['date','uncf','unc','uncr','duncf','dunc','duncr','dunc3','dunc6'] -- Ludvigson macro/
    financial/real uncertainty + first differences + 3/6-month changes (create_factors.fetch_uncertainty)."""
    def build(url: str) -> pd.DataFrame:
        frames = []
        with zipfile.ZipFile(io.BytesIO(_get(url))) as zf:
            for stem, col in _LUDVIGSON_FILE_MAP.items():
                zname = next(n for n in zf.namelist() if stem in n and n.endswith(".xlsx"))
                with zf.open(zname) as f:
                    tmp = pd.read_excel(f, sheet_name=0, usecols=[0, 1])
                tmp.columns = ["date", col]
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce") + MonthEnd(0)
                frames.append(tmp.dropna(subset=["date"]))
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on="date", how="outer")
        out = out.sort_values("date").reset_index(drop=True)
        for col in ("uncf", "unc", "uncr"):
            out[f"d{col}"] = out[col].diff()
        out["dunc3"] = out["unc"].diff(3)
        out["dunc6"] = out["unc"].diff(6)
        return out[["date", "uncf", "unc", "uncr", "duncf", "dunc", "duncr", "dunc3", "dunc6"]]
    return _cached("ludvigson_uncertainty", _ludvigson_url, build, force)


def fetch_epu(force: bool = False) -> pd.DataFrame:
    """['date','epu','epum','eput'] -- policyuncertainty.com categorical EPU indices, /1000
    (create_factors.fetch_epu; last sheet row is a text footnote, dropped)."""
    def build(url: str) -> pd.DataFrame:
        df = pd.read_excel(io.BytesIO(_get(url)), sheet_name="Indices")
        df = df.iloc[:-1].copy()                        # trailing text footnote row
        df["date"] = pd.to_datetime(df["Year"].astype(int).astype(str) + "-"
                                    + df["Month"].astype(int).astype(str) + "-01") + MonthEnd(0)
        out = df[["date"] + list(_EPU_COL_MAP)].rename(columns=_EPU_COL_MAP)
        for c in ("epu", "epum", "eput"):
            out[c] = out[c].astype(float) / 1000.0
        return out.sort_values("date").reset_index(drop=True)
    return _cached("epu_categorical", cfg.EPU_URL, build, force)

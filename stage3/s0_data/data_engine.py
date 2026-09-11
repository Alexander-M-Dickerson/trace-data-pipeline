r"""data_engine.py -- the data-appendix engine: the descriptive tables (IA.I-IA.VII).

One load per data source, one statistics pass per exhibit family; the drivers are
formatters and never recompute.

  * daily (IA.I, IA.II) read Stage 1's bond-day panel. The heavy statistics run as
    ONE grouped aggregate in DuckDB rather than a 30M-row pandas groupby.
    `quantile_cont` matches pandas' linear interpolation and `stddev_samp` matches
    ddof=1, so the two routes agree; the DuckDB one finishes.
  * monthly (IA.III-IA.VII) read Stage 2's monthly panel.

Conventions that shape what is printed -- each one asserted, not assumed:

  * IA.IV prints 17 of its 20 candidate variables, because `pr`, `mod_dur` and `conv`
    are not columns of the monthly panel. Absent variables are SKIPPED, not zero-filled.
  * rating buckets: IG = [1, 10], NIG = (10, 21], Default == 22. Percent-missing is
    against the BUCKET row count, not the panel.
  * ret_vwx = ret_vw - tret, the duration-adjusted return, computed here at report time.
  * scales: ytm / credit spread / returns / spd_rel are x100. Pooled statistics scale
    the SERIES before computing; cross-sectional ones scale the AVERAGED statistic
    after. For linear statistics that is the same number, and the distinction is kept
    so the rounding matches.
  * IA.V tail percentiles are quantiles of the DECIMAL series then x100, while the
    moments are unscaled. P0.01 means quantile(0.0001), not quantile(0.01).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _p in (str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths                # noqa: E402


DAILY_AVAIL_VARS = [
    ("pr", "Price (VW)"), ("prc_bid", "Price (Bid)"), ("prc_ask", "Price (Ask)"),
    ("credit_spread", "Spread"), ("sp_rating", "Rating (SP)"),
    ("mdy_rating", "Rating (MD)"), ("permno", "PERMNO"),
]
DAILY_STAT_VARS = [
    ("pr", "Price (VW)"), ("prc_ew", "Price (EW)"), ("prc_vw_par", "Price (ParW)"),
    ("prc_bid", "Price (Bid)"), ("prc_ask", "Price (Ask)"), ("prfull", "Price (Full)"),
    ("ytm", "YTM"), ("credit_spread", "Spread"),
    ("mac_dur", "Duration (Macaulay)"), ("mod_dur", "Duration (Modified)"),
    ("bond_maturity", "Bond Maturity"), ("bond_age", "Bond Age"),
    ("convexity", "Convexity"), ("dvolume", "Volume (Dollar)"),
    ("qvolume", "Volume (Par)"), ("bid_count", "Bid Count"),
    ("ask_count", "Ask Count"), ("sp_rating", "Rating (SP)"),
    ("mdy_rating", "Rating (MD)"),
]
DAILY_SCALE100 = ("ytm", "credit_spread")          # decimals -> percent

MONTHLY_AVAIL_VARS = [
    ("pr", "Price (VW)"), ("ret_vw", "Month-End Return"),
    ("ret_vw_bgn", "Month-Begin Return"), ("ytm", "YTM"), ("cs", "Spread"),
    ("spc_rat", "Composite Rating (SP)"), ("mdc_rat", "Composite Rating (MD)"),
    ("permno", "PERMNO"),
]
MONTHLY_STAT_VARS = [
    ("ret_vw", "Total End Return (%)"), ("ret_vw_bgn", "Total Begin Return (%)"),
    ("ret_vwx", "Dur. Adj. End Return (%)"), ("ret_vwx_bgn", "Dur. Adj. Begin Return (%)"),
    ("lib", "Latent Imp. Bias"), ("hprd", "End Holding Period"),
    ("hprd_bgn", "Begin Holding Period"), ("igap_bgn", "Implementation Gap"),
    ("sig_gap", "Signal Gap"), ("pr", "Price (VW)"), ("ytm", "YTM (%)"),
    ("cs", "Spread (%)"), ("mod_dur", "Duration (Modified)"),
    ("tmat", "Bond Maturity"), ("age", "Bond Age"), ("conv", "Convexity"),
    ("sze", "Market Cap."), ("spc_rat", "Composite Rating (SP)"),
    ("mdc_rat", "Composite Rating (MD)"), ("spd_rel", "Bid-Ask Spread (%)"),
]
MONTHLY_SCALE = {"ret_vw": 100, "ret_vw_bgn": 100, "ret_vwx": 100,
                 "ret_vwx_bgn": 100, "ytm": 100, "cs": 100, "spd_rel": 100}
STAT_COLS = ["Mean", "Median", "SD", "P1", "P5", "P95", "P99"]

RATING_BUCKETS = ["All", "IG", "NIG", "Def"]


def _bucket_where(col: str) -> dict[str, str]:
    return {"All": "TRUE",
            "IG": f"{col} >= 1 AND {col} <= 10",
            "NIG": f"{col} > 10 AND {col} <= 21",
            "Def": f"{col} = 22"}


def _bucket_mask(df: pd.DataFrame, col: str) -> dict[str, pd.Series]:
    return {"All": pd.Series(True, index=df.index),
            "IG": (df[col] >= 1) & (df[col] <= 10),
            "NIG": (df[col] > 10) & (df[col] <= 21),
            "Def": df[col] == 22}


# ---------------------------------------------------------------------------
# Daily (IA.I, IA.II) -- all heavy lifting in DuckDB against the parquet.
# ❗Shape matters (measured, learnings L13): 95 separate ungrouped
# quantile_cont aggregates in one query ran at ~2 cores and breached the
# watchdog; ONE list-parameter quantile per variable, one query per variable,
# runs the whole table in ~40 s (grouped 0.6 s / pooled 1.4 s per variable).
# ---------------------------------------------------------------------------
_QLIST = "[0.01, 0.05, 0.95, 0.99]"
_QNAMES = ["P1", "P5", "P95", "P99"]


def _con():
    """An in-memory DuckDB connection sized to this machine.

    ❗This is the heaviest read in Stage 3 -- a full scan of a 31-million-row daily panel,
    once pooled and once grouped by date, for each of nineteen variables. It is also the
    SECOND step of forty, so a machine that cannot take it fails before anything else has
    run.

    So the limit is explicit and a spill directory is given. An uncapped connection takes
    DuckDB's default share of RAM with nowhere to spill, which on a small machine is the
    difference between slow and killed. Override with STAGE3_MEMORY_LIMIT (e.g. "6GB").
    """
    import os

    import duckdb
    con = duckdb.connect()
    con.execute(f"SET threads={max(2, (os.cpu_count() or 4))}")
    limit = os.environ.get("STAGE3_MEMORY_LIMIT")
    if not limit:
        try:                                   # psutil is optional; fall back politely
            import psutil
            limit = f"{max(2, int(psutil.virtual_memory().available / 2**30 * 0.6))}GB"
        except Exception:                      # noqa: BLE001
            limit = "4GB"
    con.execute(f"SET memory_limit='{limit}'")
    spill = paths.CACHE / "duckdb"
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{spill.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con, paths.DAILY.as_posix()


def _end_clause(end: str | None) -> str:
    """A pushdown-friendly cut at the stats layer (measurement windows)."""
    return f"trd_exctn_dt <= TIMESTAMP '{end} 00:00:00'" if end else "TRUE"


def daily_date_range(end: str | None = None) -> tuple[str, str]:
    con, p = _con()
    lo, hi = con.sql(f"SELECT min(trd_exctn_dt), max(trd_exctn_dt) "
                     f"FROM read_parquet('{p}') WHERE {_end_clause(end)}").fetchone()
    return str(lo)[:10], str(hi)[:10]


def daily_availability(end: str | None = None) -> pd.DataFrame:
    """IA.I: one row per (variable, bucket): observations + pct_missing."""
    con, p = _con()
    rows = []
    for bucket, cond in _bucket_where("spc_rating").items():
        sel = ", ".join(
            [f"count({v}) AS obs_{v}, count(*) - count({v}) AS miss_{v}"
             for v, _ in DAILY_AVAIL_VARS] + ["count(*) AS total"])
        r = con.sql(f"SELECT {sel} FROM read_parquet('{p}') "
                    f"WHERE ({cond}) AND {_end_clause(end)}").df().iloc[0]
        for v, label in DAILY_AVAIL_VARS:
            rows.append({"bucket": bucket, "variable": label,
                         "observations": int(r[f"obs_{v}"]),
                         "pct_missing": (100.0 * r[f"miss_{v}"] / r["total"])
                         if r["total"] else 0.0})
    return pd.DataFrame(rows)


def daily_pooled(end: str | None = None) -> pd.DataFrame:
    """IA.II Panel A: pooled stats (dropna; ytm/credit_spread x100; round 2)."""
    con, p = _con()
    rows = []
    for v, label in DAILY_STAT_VARS:
        m, md, sd, q = con.sql(
            f"SELECT avg({v}), median({v}), stddev_samp({v}), "
            f"quantile_cont({v}, {_QLIST}) FROM read_parquet('{p}') "
            f"WHERE {_end_clause(end)}").fetchone()
        scale = 100.0 if v in DAILY_SCALE100 else 1.0
        stats = {"Mean": m, "Median": md, "SD": sd,
                 **dict(zip(_QNAMES, q))}
        rows.append({"Variable": label,
                     **{c: round(float(stats[c]) * scale, 2) for c in STAT_COLS}})
    return pd.DataFrame(rows)


def daily_cross_sectional(end: str | None = None) -> pd.DataFrame:
    """IA.II Panel B: per-day stats (one grouped DuckDB query per variable),
    then time-series means. ❗Scale AFTER averaging here, unlike the pooled panel
    above, which scales the series first."""
    con, p = _con()
    rows = []
    for v, label in DAILY_STAT_VARS:
        # ❗ORDER BY matters here, and not for tidiness. The time-series mean below
        # sums floats in array order, so an unordered group output changes the last
        # bits of the sum -- which once moved Convexity's P1 across the 1.605
        # rounding boundary and changed a printed digit.
        d = con.sql(
            f"SELECT trd_exctn_dt, avg({v}) AS m, median({v}) AS md, "
            f"stddev_samp({v}) AS sd, quantile_cont({v}, {_QLIST}) AS q "
            f"FROM read_parquet('{p}') WHERE {_end_clause(end)} "
            f"GROUP BY 1 ORDER BY 1").df()
        # a day with the variable entirely NULL yields NULL aggregates; the
        # the time-series mean skips empty days rather than treating them as zero
        qs = d["q"].dropna()
        qm = np.nanmean(np.vstack(qs.to_numpy()), axis=0)
        scale = 100.0 if v in DAILY_SCALE100 else 1.0
        stats = {"Mean": pd.to_numeric(d["m"]).mean(),
                 "Median": pd.to_numeric(d["md"]).mean(),
                 "SD": pd.to_numeric(d["sd"]).mean(), **dict(zip(_QNAMES, qm))}
        rows.append({"Variable": label,
                     **{c: round(float(stats[c]) * scale, 2) for c in STAT_COLS}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monthly (IA.III-IA.VII) -- the Stage 2 panel
# ---------------------------------------------------------------------------
def load_monthly(end: str | None = None) -> pd.DataFrame:
    cols = sorted({v for v, _ in MONTHLY_AVAIL_VARS if v != "pr"}
                  | {v for v, _ in MONTHLY_STAT_VARS if v not in ("pr", "mod_dur",
                                                                 "conv", "ret_vwx",
                                                                 "ret_vwx_bgn")}
                  | {"cusip", "date", "bbtm", "tret"})
    df = pd.read_parquet(paths.PANEL, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    if end:
        df = df[df["date"] <= end].copy()
    # the duration-adjusted return is computed at report time, not stored
    df["ret_vwx"] = df["ret_vw"] - df["tret"]
    df["ret_vwx_bgn"] = df["ret_vw_bgn"] - df["tret"]
    return df


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """build_month_end_panel, vectorised: a contiguous month-end skeleton per
    cusip between its first and last observed month, original rows merged in."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]) + pd.offsets.MonthEnd(0)
    per = out["date"].dt.to_period("M")
    bounds = (pd.DataFrame({"cusip": out["cusip"], "p": per})
                .groupby("cusip", observed=True)["p"].agg(first="min", last="max"))
    n = (bounds["last"] - bounds["first"]).apply(lambda d: d.n) + 1
    cusips = np.repeat(bounds.index.to_numpy(), n.to_numpy())
    offsets = np.concatenate([np.arange(k) for k in n.to_numpy()])
    periods = np.repeat(bounds["first"].to_numpy(), n.to_numpy()) + offsets
    skeleton = pd.DataFrame({"cusip": cusips,
                             "date": pd.PeriodIndex(periods).to_timestamp("M")})
    res = skeleton.merge(out, on=["cusip", "date"], how="left")
    # ❗bbtm is 100/price, so the price is 100/bbtm -- NOT bbtm*100. Created on the
    # resampled frame, after the month-end pick.
    return res


def monthly_availability(res: pd.DataFrame) -> pd.DataFrame:
    """IA.III: Total row + one row per (variable, bucket)."""
    rows = []
    masks = _bucket_mask(res, "spc_rat")
    for bucket in RATING_BUCKETS:
        sub = res[masks[bucket]]
        rows.append({"bucket": bucket, "variable": "Total",
                     "observations": len(sub), "pct_missing": None})
        for v, label in MONTHLY_AVAIL_VARS:
            nn = int(sub[v].notna().sum()) if v in sub.columns else 0
            pct = (100.0 * (len(sub) - nn) / len(sub)) if len(sub) else 0.0
            rows.append({"bucket": bucket, "variable": label,
                         "observations": nn, "pct_missing": pct})
    return pd.DataFrame(rows)


def monthly_pooled(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v, label in MONTHLY_STAT_VARS:
        if v not in df.columns:
            continue                        # the IA.IV 17-row rule
        s = df[v].dropna()
        if len(s) == 0:
            continue
        s = s * MONTHLY_SCALE.get(v, 1)
        rows.append({"Variable": label,
                     "Mean": round(s.mean(), 2), "Median": round(s.median(), 2),
                     "SD": round(s.std(), 2), "P1": round(s.quantile(.01), 2),
                     "P5": round(s.quantile(.05), 2), "P95": round(s.quantile(.95), 2),
                     "P99": round(s.quantile(.99), 2)})
    return pd.DataFrame(rows)


def monthly_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v, label in MONTHLY_STAT_VARS:
        if v not in df.columns:
            continue
        g = df.groupby("date", observed=True)[v]
        stats = {"Mean": g.mean().mean(), "Median": g.median().mean(),
                 "SD": g.std().mean(), "P1": g.quantile(.01).mean(),
                 "P5": g.quantile(.05).mean(), "P95": g.quantile(.95).mean(),
                 "P99": g.quantile(.99).mean()}
        scale = MONTHLY_SCALE.get(v, 1)
        rows.append({"Variable": label,
                     **{k: round(x * scale, 2) for k, x in stats.items()}})
    return pd.DataFrame(rows)


EXTREME_PCTLS = [("P0.01", 0.0001), ("P0.05", 0.0005), ("P0.10", 0.001),
                 ("P99.90", 0.999), ("P99.95", 0.9995), ("P99.99", 0.9999)]
EXTREME_THRESH = [0.20, 0.50, 1.00]


def extreme_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """IA.V: tail percentiles (x100), moments (unscaled), threshold counts --
    per (return column, rating bucket)."""
    from scipy import stats as sps
    masks = _bucket_mask(df, "spc_rat")
    series = {(d, b): df.loc[masks[b], c].dropna()
              for c, d in [("ret_vw", "End"), ("ret_vw_bgn", "Begin")]
              for b in RATING_BUCKETS}

    tail = [{"Statistic": name,
             **{f"{d}_{b}": float(series[(d, b)].quantile(q)) * 100
                for d in ("End", "Begin") for b in RATING_BUCKETS}}
            for name, q in EXTREME_PCTLS]
    mom = [{"Statistic": nm,
            **{f"{d}_{b}": float(fn(series[(d, b)]))
               for d in ("End", "Begin") for b in RATING_BUCKETS}}
           for nm, fn in [("Skewness", sps.skew),
                          ("Excess Kurtosis", lambda x: sps.kurtosis(x, fisher=True))]]
    counts = []
    for t in EXTREME_THRESH:
        for sign, direction in [("<", f"neg{int(t*100)}"), (">", f"pos{int(t*100)}")]:
            counts.append({"Direction": f"{sign}{int(t*100)}",
                           **{f"{d}_{b}": int((series[(d, b)] < -t).sum() if sign == "<"
                                              else (series[(d, b)] > t).sum())
                              for d in ("End", "Begin") for b in RATING_BUCKETS}})
    counts.append({"Direction": ">500",
                   **{f"{d}_{b}": int((series[(d, b)] > 5.0).sum())
                      for d in ("End", "Begin") for b in RATING_BUCKETS}})
    return {"tail": pd.DataFrame(tail), "moments": pd.DataFrame(mom),
            "counts": pd.DataFrame(counts)}


def time_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """IA.VI: per-year N + threshold counts for both return columns (All bonds)."""
    w = df.copy()
    w["_year"] = w["date"].dt.year
    rows = []
    for year, ydf in w.groupby("_year"):
        row = {"Year": int(year)}
        for c, d in [("ret_vw", "End"), ("ret_vw_bgn", "Begin")]:
            s = ydf[c].dropna()
            row[f"N_{d}"] = len(s)
            for t in (0.20, 0.95):
                row[f"{d}_neg_{int(t*100)}"] = int((s < -t).sum())
                row[f"{d}_pos_{int(t*100)}"] = int((s > t).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def annual_stats(df: pd.DataFrame) -> pd.DataFrame:
    """IA.VII: per-year N/mean/SD/median/min/max (x100) + matched-pair rho."""
    w = df.copy()
    w["_year"] = w["date"].dt.year
    rows = []
    for year, ydf in w.groupby("_year"):
        row = {"Year": int(year)}
        for c, d in [("ret_vw", "End"), ("ret_vw_bgn", "Begin")]:
            s = ydf[c].dropna()
            row[f"N_{d}"] = len(s)
            row[f"{d}_mean"] = s.mean() * 100
            row[f"{d}_std"] = s.std() * 100
            row[f"{d}_med"] = s.median() * 100
            row[f"{d}_min"] = s.min() * 100
            row[f"{d}_max"] = s.max() * 100
        valid = ydf[["ret_vw", "ret_vw_bgn"]].dropna()
        row["rho"] = valid["ret_vw"].corr(valid["ret_vw_bgn"]) if len(valid) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def annual_total_row(stats_df: pd.DataFrame) -> dict:
    """The printed Total row: N-weighted means, simple-average SD and rho,
    median of medians, and the overall min/max."""
    def wavg(col, wcol):
        v = stats_df[[col, wcol]].dropna()
        return (v[col] * v[wcol]).sum() / v[wcol].sum()
    out = {"Year": "Total"}
    for d in ("End", "Begin"):
        out[f"N_{d}"] = int(stats_df[f"N_{d}"].sum())
        out[f"{d}_mean"] = wavg(f"{d}_mean", f"N_{d}")
        out[f"{d}_std"] = stats_df[f"{d}_std"].mean()
        out[f"{d}_med"] = stats_df[f"{d}_med"].median()
        out[f"{d}_min"] = stats_df[f"{d}_min"].min()
        out[f"{d}_max"] = stats_df[f"{d}_max"].max()
    out["rho"] = stats_df["rho"].mean()
    return out

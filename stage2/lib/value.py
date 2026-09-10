"""value.py -- VERBATIM value-signal machinery from stage2/process_bond_data.py
(lines 5002-5700, 5906-6051): _apply_dts_quintile_overlay, compute_value, within_firm_demean,
compute_lagged_values, make_value_signals, build_d_spreads. Arbitrated by the G6/G7 validators."""
import gc
import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from numba import njit

_HAS_NUMBA = True

if _HAS_NUMBA:
    @njit(cache=True)
    def _roll_mean_skip1_nan(x, starts, ends, window):
        n = x.shape[0]
        out = np.empty(n, dtype=np.float64)
        out[:] = np.nan
        for g in range(starts.shape[0]):
            s0 = starts[g]
            e0 = ends[g]
            for i in range(s0, e0):
                s = 0.0
                c = 0
                # exclude current => look back 1..window rows
                for k in range(1, window + 1):
                    j = i - k
                    if j < s0:
                        break
                    v = x[j]
                    if not np.isnan(v):
                        s += v
                        c += 1
                out[i] = s / c if c >= 6 else np.nan
        return out




def _apply_dts_quintile_overlay(
    panel: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    cs_col: str,
    dur_col: str,
    sig_col: str,
    out_overlay_col: str,
    n_bins: int = 5,
    q_col: str = "Q_dts",
    mean_col: str = "sig_mean_dtsQ",
    return_details: bool = False,
) -> pd.DataFrame:
    """Within-date DtS quintiles; demean sig within (date,Q)."""
    w = panel[[id_col, date_col, cs_col, dur_col, sig_col]].copy()
    w[date_col] = pd.to_datetime(w[date_col])

    w["dts"] = w[dur_col] * w[cs_col]

    w["_rk"] = w.groupby(date_col)["dts"].rank(method="first")
    w[q_col] = (
        w.groupby(date_col)["_rk"]
         .transform(lambda s: pd.qcut(s, q=n_bins, labels=False, duplicates="drop"))
         .astype("float")
         .add(1.0)
         .astype("Int64")
    )
    w = w.drop(columns=["_rk"])

    w[mean_col] = w.groupby([date_col, q_col])[sig_col].transform("mean")
    w[out_overlay_col] = w[sig_col] - w[mean_col]

    keep = [id_col, date_col, sig_col, out_overlay_col]
    if return_details:
        keep += ["dts", q_col, mean_col]
    return w[keep]


def compute_value(
    df: pd.DataFrame,
    *,
    model_type: str = "hz",
    id_col: str = "cusip",
    date_col: str = "date",
    cs_col: str = "cs",
    rating_col: str = "spc_rat",
    # ALL non-rating / non-industry regressors go here:
    x_cols: Sequence[str],
    # Naming
    out_col: Optional[str] = None,          # default: f"val_{model_type}"
    suffix_dts: str = "_dts",               # short suffix for DtS overlay
    suffix_firm: str = "_wi",               # short suffix for firm overlay
    # Signal definition
    denom: str = "fitted",                  # "fitted"|"actual"|"resid"
    as_percent: bool = False,
    min_obs: int = 50,
    # Rating control
    rating_mode: str = "dummies",           # "dummies"|"numeric"
    rating_levels: Optional[Sequence[int]] = None,
    base_rating: Optional[int] = None,
    # Industry dummies
    industry_col: Optional[str] = None,
    industry_levels: Optional[Sequence[int]] = None,
    base_industry: Optional[int] = None,
    # Overlays
    dur_col: Optional[str] = None,
    overlay_bins: int = 5,
    firm_col: Optional[str] = None,         # e.g. "permco", "gvkey", "issuer_cusip"
    out_firm_col: Optional[str] = None,
    out_overlay_col: Optional[str] = None,
    return_overlay_details: bool = False,
    # Log option
    y_transform: str = "level",             # "level"|"log"
    retransform: str = "lognormal",         # "lognormal"|"smearing"
    log_min_cs: float = 1e-12,
) -> pd.DataFrame:
    """
    Monthly cross-sectional regression; output a value signal.

    Regression each month t:
      - y is cs (level) or log(cs) (log)
      - controls are: rating (dummies or numeric), optional industry dummies, and x_cols.

    denom:
      - "fitted":  (cs - cs_hat) / cs_hat
      - "actual":  (cs - cs_hat) / cs
      - "resid":   regression residual (levels or logs), no scaling

    Overlays (only if dur_col is provided):
      - {val}{suffix_dts}: DtS-demeaned signal (date × DtS quintile)
      - {val}{suffix_firm}: firm-within-date demeaned signal (single-issue firms unchanged)

    firm_col == "issuer_cusip":
      - creates "issuer_cusip" from first 6 chars of `id_col` if missing.
    """
    if denom not in {"fitted", "actual", "resid"}:
        raise ValueError("denom must be 'fitted', 'actual', or 'resid'.")
    if rating_mode not in {"dummies", "numeric"}:
        raise ValueError("rating_mode must be 'dummies' or 'numeric'.")
    if y_transform not in {"level", "log"}:
        raise ValueError("y_transform must be 'level' or 'log'.")
    if retransform not in {"lognormal", "smearing"}:
        raise ValueError("retransform must be 'lognormal' or 'smearing'.")
    x_cols = list(x_cols)
    if len(x_cols) == 0:
        raise ValueError("x_cols must contain at least one regressor (e.g., ['tmat','dcs3']).")

    base = out_col or f"val_{model_type}"
    out_dts = out_overlay_col or f"{base}{suffix_dts}"
    out_fl = out_firm_col or f"{base}{suffix_firm}"

    # Required columns for regression + optional overlays
    req = [id_col, date_col, cs_col, rating_col] + x_cols
    if industry_col is not None:
        req.append(industry_col)
    if dur_col is not None:
        req.append(dur_col)
    if firm_col is not None and firm_col != "issuer_cusip":
        req.append(firm_col)

    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    use = df.loc[:, req].copy()
    use[date_col] = pd.to_datetime(use[date_col])

    # Create issuer_cusip if requested
    if firm_col == "issuer_cusip" and "issuer_cusip" not in use.columns:
        use["issuer_cusip"] = use[id_col].astype(str).str.strip().str.slice(0, 6)

    # Regression requires these non-missing
    reg_drop = [id_col, date_col, cs_col, rating_col] + x_cols
    if industry_col is not None:
        reg_drop.append(industry_col)

    use = use.dropna(subset=reg_drop)
    if y_transform == "log":
        use = use[use[cs_col] > log_min_cs].copy()

    # Rating handling
    if rating_mode == "dummies":
        if rating_levels is None:
            rating_levels = sorted(use[rating_col].astype(int).unique().tolist())
        rating_levels = [int(x) for x in rating_levels]
        if len(rating_levels) < 2:
            raise ValueError("Need at least 2 distinct rating levels for dummies.")
        if base_rating is None:
            base_rating = rating_levels[-1]
        base_rating = int(base_rating)
        if base_rating not in set(rating_levels):
            raise ValueError("base_rating must be in rating_levels.")
        rating_dummy_levels = [r for r in rating_levels if r != base_rating]
        rating_pos = {lvl: j for j, lvl in enumerate(rating_dummy_levels)}
        Lr = len(rating_dummy_levels)
    else:
        rating_pos = {}
        Lr = 1  # numeric rating column

    # Industry handling
    if industry_col is not None:
        if industry_levels is None:
            industry_levels = sorted(use[industry_col].astype(int).unique().tolist())
        industry_levels = [int(x) for x in industry_levels]
        if len(industry_levels) < 2:
            raise ValueError("Need at least 2 distinct industry levels if industry_col is used.")
        if base_industry is None:
            base_industry = industry_levels[-1]
        base_industry = int(base_industry)
        if base_industry not in set(industry_levels):
            raise ValueError("base_industry must be in industry_levels.")
        ind_dummy_levels = [g for g in industry_levels if g != base_industry]
        ind_pos = {lvl: j for j, lvl in enumerate(ind_dummy_levels)}
        Li = len(ind_dummy_levels)
    else:
        ind_pos = {}
        Li = 0

    Kx = len(x_cols)

    # X order: [1, rating_part (Lr), industry_dums (Li), x_cols (Kx)]
    p = 1 + Lr + Li + Kx
    ind_base = 1 + Lr
    x_base = 1 + Lr + Li

    use = use.sort_values([date_col, id_col], kind="mergesort").reset_index(drop=True)

    cs_all = use[cs_col].to_numpy(dtype=np.float64)
    x_all = use.loc[:, x_cols].to_numpy(dtype=np.float64)  # Kx >= 1 by construction

    sig = np.full(len(use), np.nan, dtype=np.float64)

    dates = use[date_col].to_numpy()
    _, start_idx = np.unique(dates, return_index=True)
    start_idx = np.sort(start_idx)
    end_idx = np.r_[start_idx[1:], len(use)]

    for s, e in zip(start_idx, end_idx):
        n = e - s
        if n < min_obs:
            continue

        X = np.zeros((n, p), dtype=np.float64)
        X[:, 0] = 1.0

        # rating block
        if rating_mode == "dummies":
            rr = use.loc[s:e - 1, rating_col].astype(int).to_numpy()
            for i in range(n):
                j = rating_pos.get(int(rr[i]))
                if j is not None:
                    X[i, 1 + j] = 1.0
        else:
            X[:, 1] = use.loc[s:e - 1, rating_col].to_numpy(dtype=np.float64)

        # industry block
        if industry_col is not None:
            ii = use.loc[s:e - 1, industry_col].astype(int).to_numpy()
            for i in range(n):
                j = ind_pos.get(int(ii[i]))
                if j is not None:
                    X[i, ind_base + j] = 1.0

        # generic RHS regressors
        X[:, x_base:x_base + Kx] = x_all[s:e, :]

        cs = cs_all[s:e]

        if y_transform == "level":
            y = cs
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            cs_hat = X @ beta

            if denom == "resid":
                v = y - cs_hat
            else:
                den = cs_hat if denom == "fitted" else y
                v = (y - cs_hat) / den

        else:  # log
            y = np.log(cs)
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            xb = X @ beta
            resid = y - xb

            if denom == "resid":
                v = resid
            else:
                if retransform == "lognormal":
                    sig2 = float(np.mean(resid * resid))
                    cs_hat = np.exp(xb + 0.5 * sig2)
                else:
                    smear = float(np.mean(np.exp(resid)))
                    cs_hat = np.exp(xb) * smear

                den = cs_hat if denom == "fitted" else cs
                v = (cs - cs_hat) / den

        v = v.astype(np.float64, copy=False)
        v[~np.isfinite(v)] = np.nan
        if as_percent and denom != "resid":
            v *= 100.0

        sig[s:e] = v

    base_out = use[[id_col, date_col]].copy()
    base_out[base] = sig

    if dur_col is None:
        return base_out[[id_col, date_col, base]]

    # DtS overlay
    base_out[cs_col] = use[cs_col].to_numpy()
    base_out[dur_col] = use[dur_col].to_numpy()

    dts_block = _apply_dts_quintile_overlay(
        base_out,
        id_col=id_col,
        date_col=date_col,
        cs_col=cs_col,
        dur_col=dur_col,
        sig_col=base,
        out_overlay_col=out_dts,
        n_bins=overlay_bins,
        mean_col=f"{base}_m_dtsQ",
        return_details=return_overlay_details,
    )

    # Firm overlay
    firm_work = base_out[[id_col, date_col, base]].copy()
    if firm_col is None:
        firm_work["_firm_id"] = pd.NA
    else:
        firm_work["_firm_id"] = use[firm_col]  # issuer_cusip exists if requested

    g = firm_work.groupby([date_col, "_firm_id"])[base]
    firm_mean = g.transform("mean")
    firm_cnt = g.transform("size")

    firm_adj = (firm_work[base] - firm_mean).where(firm_cnt >= 2, firm_work[base])
    firm_work[out_fl] = firm_adj

    out = dts_block.merge(
        firm_work[[id_col, date_col, out_fl]],
        on=[id_col, date_col],
        how="inner",
    )

    core = [id_col, date_col, base, out_dts, out_fl]
    if return_overlay_details:
        extra = [c for c in out.columns if c not in core]
        return out[core + extra]
    return out[core]


def within_firm_demean(
    df: pd.DataFrame,
    *,
    id_col: str = "cusip",
    date_col: str = "date",
    firm_col: str = "issuer_cusip",   # can be a real column, or "issuer_cusip" to create from cusip
    cols: Iterable[str] = (),
    suffix: str = "_wf",
    keep_single_issue_raw: bool = True,
) -> pd.DataFrame:
    """
    For each col in `cols`, create col{suffix} = col - mean(col | date, firm),
    with optional override: if firm has only one bond that date, keep raw col.
    Returns a copy of df with the new columns added.
    """
    cols = list(cols)
    if not cols:
        raise ValueError("cols must be a non-empty iterable of column names.")

    missing = [c for c in [id_col, date_col] + cols if c not in df.columns]
    if firm_col != "issuer_cusip":
        missing += [firm_col] if firm_col not in df.columns else []
    if missing:
        raise KeyError(f"Missing columns in df: {sorted(set(missing))}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    # Firm id
    if firm_col == "issuer_cusip":
        if "issuer_cusip" not in out.columns:
            out["issuer_cusip"] = out[id_col].astype(str).str.strip().str.slice(0, 6)
        fcol = "issuer_cusip"
    else:
        fcol = firm_col

    g = out.groupby([date_col, fcol], sort=False)

    for c in cols:
        mu = g[c].transform("mean")
        if keep_single_issue_raw:
            n = g[c].transform("size")
            out[f"{c}{suffix}"] = (out[c] - mu).where(n >= 2, out[c])
        else:
            out[f"{c}{suffix}"] = out[c] - mu

    return out


def compute_lagged_values(
    df: pd.DataFrame,
    value_cols: list,
    lag: int,
    *,
    id_col: str = "cusip",
    date_col: str = "date",
    bandwidth: int = 1,
) -> pd.DataFrame:
    """
    Compute lagged values using calendar-month lookup with bandwidth search.

    Unlike simple shift(n), this function:
    1. Uses calendar month keys (year*12 + month) for accurate lag computation
    2. If exact lag month is missing, searches +/- bandwidth months
    3. Prefers earlier months (lag n+1) over later months (lag n-1) when both exist

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with id_col, date_col, and value columns
    value_cols : list
        List of column names to compute lags for
    lag : int
        Number of months to lag (e.g., 3 for 3-month lag)
    id_col : str, default "cusip"
        Identifier column name
    date_col : str, default "date"
        Date column name
    bandwidth : int, default 1
        Search bandwidth (+/- months) if exact lag is missing

    Returns
    -------
    pd.DataFrame
        DataFrame with id_col, date_col, and lagged value columns (suffix _lagN)
    """
    if bandwidth < 0:
        raise ValueError("bandwidth must be >= 0")

    # Build offset search order: exact first, then earlier month preferred
    # offset 0: exact month (m - lag)
    # offset -1: month (m - lag - 1) = lag + 1 [preferred - earlier]
    # offset +1: month (m - lag + 1) = lag - 1
    offsets = [0]
    for j in range(1, bandwidth + 1):
        offsets.extend([-j, +j])

    # Prepare data with calendar month key
    d = df[[id_col, date_col] + value_cols].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([id_col, date_col], kind="mergesort")
    dt = d[date_col].dt
    d["_m"] = (dt.year.astype(np.int32) * 12 + dt.month.astype(np.int32)).astype(np.int32)

    # Create lookup table: (id, month) -> values
    # Keep last observation if multiple per month
    lk_cols = [id_col, "_m"] + value_cols
    lk = d[lk_cols].drop_duplicates([id_col, "_m"], keep="last").copy()
    # Rename value columns for lookup
    rename_map = {c: f"{c}__lk" for c in value_cols}
    lk = lk.rename(columns=rename_map)

    # Target month for lag
    m0 = d["_m"].to_numpy(np.int32, copy=False) - np.int32(lag)

    # Initialize lagged arrays
    lagged = {c: np.full(len(d), np.nan, dtype=np.float64) for c in value_cols}

    # Search with bandwidth
    tmp = d[[id_col]].copy()
    for off in offsets:
        tmp["_mt"] = (m0 + np.int32(off)).astype(np.int32, copy=False)

        got = tmp.merge(
            lk,
            left_on=[id_col, "_mt"],
            right_on=[id_col, "_m"],
            how="left",
            sort=False,
            copy=False,
        )

        # Fill in values where still NaN
        all_filled = True
        for c in value_cols:
            lk_col = f"{c}__lk"
            vals = got[lk_col].to_numpy(dtype=np.float64, copy=False)
            lagged[c] = np.where(np.isnan(lagged[c]), vals, lagged[c])
            if np.isnan(lagged[c]).any():
                all_filled = False

        if all_filled:
            break

    # Build output DataFrame
    out = d[[id_col, date_col]].copy()
    for c in value_cols:
        out[f"{c}_lag{lag}"] = lagged[c]

    return out


def make_value_signals(
    end_signals: pd.DataFrame,
    adj_signals: pd.DataFrame,
    all_returns: pd.DataFrame,
    verbose: bool = True,
) -> tuple:
    """
    Compute value signals for cross-sectional regressions.

    Parameters
    ----------
    end_signals : pd.DataFrame
        End-of-month signals with columns: cusip, date, cs, spc_rat,
        tmat, age, cpn, call, ff17num, ff30num, etc.
    adj_signals : pd.DataFrame
        Adjusted signals with columns: cusip, date, cs_adj, etc.
    all_returns : pd.DataFrame
        Bond returns with columns: cusip, date, ret_vw, tret

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (signals_std, signals_adj) - standard and adjusted value signals
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("MAKE VALUE SIGNALS - START")
        logger.info("=" * 60)

    # =========================================================================
    # Step A: Merge rating/characteristic columns from end_signals to adj_signals
    # =========================================================================
    if verbose:
        logger.info("[Step A] Merging columns from end_signals to adj_signals...")

    merge_cols = ['cusip', 'date', 'spc_rat', 'tmat', 'age', 'cpn', 'call', 'ff17num', 'ff30num']
    merge_cols = [c for c in merge_cols if c in end_signals.columns]

    adj = adj_signals.copy()
    end = end_signals.copy()

    # Ensure date is datetime
    adj['date'] = pd.to_datetime(adj['date'])
    end['date'] = pd.to_datetime(end['date'])

    # Left merge to adj_signals
    adj = adj.merge(end[merge_cols], on=['cusip', 'date'], how='left')

    if verbose:
        logger.info("  Merged columns: %s", merge_cols[2:])
        logger.info("  adj_signals shape: %s", adj.shape)

    # =========================================================================
    # Step B: Compute lagged credit spreads (3-month lag with bandwidth search)
    # =========================================================================
    if verbose:
        logger.info("[Step B] Computing lagged credit spreads (calendar-month lookup, bandwidth=1)...")

    # Use compute_lagged_values for accurate calendar-month based lag with bandwidth search
    # This handles missing months by searching +/- 1 month around the target lag

    # Compute 3-month lagged cs_adj
    adj_lag = compute_lagged_values(
        adj[['cusip', 'date', 'cs_adj']],
        value_cols=['cs_adj'],
        lag=3,
        bandwidth=1,
    )
    adj = adj.merge(adj_lag, on=['cusip', 'date'], how='left')
    adj['dcs3_adj'] = adj['cs_adj'] - adj['cs_adj_lag3']
    adj = adj.drop(columns=['cs_adj_lag3'])
    del adj_lag

    # Compute 3-month lagged cs
    end_lag = compute_lagged_values(
        end[['cusip', 'date', 'cs']],
        value_cols=['cs'],
        lag=3,
        bandwidth=1,
    )
    end = end.merge(end_lag, on=['cusip', 'date'], how='left')
    end['dcs3'] = end['cs'] - end['cs_lag3']
    end = end.drop(columns=['cs_lag3'])
    del end_lag

    gc.collect()

    if verbose:
        n_valid_adj = adj['dcs3_adj'].notna().sum()
        n_valid_end = end['dcs3'].notna().sum()
        logger.info("  dcs3_adj valid: %s", f"{n_valid_adj:,}")
        logger.info("  dcs3 valid: %s", f"{n_valid_end:,}")

    # =========================================================================
    # Step C: Compute rolling 12-month volatility of duration-adjusted returns
    # =========================================================================
    if verbose:
        logger.info("[Step C] Computing rolling 12-month volatility of ret_vwx...")

    # Compute ret_vwx = ret_vw - tret
    ret = all_returns[['cusip', 'date', 'ret_vw', 'tret']].copy()
    ret['date'] = pd.to_datetime(ret['date'])
    ret['ret_vwx'] = ret['ret_vw'] - ret['tret']

    # Sort for rolling computation
    ret = ret.sort_values(['cusip', 'date']).reset_index(drop=True)

    # Compute rolling 12-month std of ret_vwx within each cusip
    # Use groupby().rolling() directly (no lambda)
    ret['vol12_x'] = (
        ret.groupby('cusip', observed=True)['ret_vwx']
        .rolling(window=12, min_periods=6)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Keep only cusip, date, vol12_x for merge
    vol_df = ret[['cusip', 'date', 'vol12_x']]
    del ret
    gc.collect()

    if verbose:
        n_valid_vol = vol_df['vol12_x'].notna().sum()
        logger.info("  vol12_x valid: %s", f"{n_valid_vol:,}")

    # Merge vol12_x to both adj and end
    adj = adj.merge(vol_df, on=['cusip', 'date'], how='left')
    end = end.merge(vol_df, on=['cusip', 'date'], how='left')

    del vol_df
    gc.collect()

    if verbose:
        logger.info("  Merged vol12_x to end and adj")

    # =========================================================================
    # Step D: Compute value signals using compute_value()
    # =========================================================================
    if verbose:
        logger.info("[Step D] Computing value signals...")

    # Compute value signals for end (standard)
    val_end = compute_value(
        end,
        id_col="cusip",
        date_col="date",
        model_type="hz",
        cs_col="cs",
        industry_col="ff17num",
        x_cols=['dcs3', 'call'],
        dur_col='md_dur',
        firm_col='issuer_cusip',
        y_transform="log",
        retransform="lognormal"
    )

    end['log_md_dur'] = np.log(end['md_dur'])
    val_end1 = compute_value(
        end,
        id_col="cusip",
        date_col="date",
        model_type="ipr",
        cs_col="cs",
        rating_mode="numeric",
        industry_col="ff17num",
        x_cols=['call', "log_md_dur", "vol12_x"],
        dur_col='md_dur',
        firm_col='issuer_cusip',
        denom="resid",
        y_transform="log",
        retransform="lognormal"
    )

    val_end = val_end.merge(val_end1, on=['cusip', 'date'], how='inner')
    del val_end1

    if verbose:
        logger.info("  val_end shape: %s", val_end.shape)

    # Compute value signals for adj (adjusted)
    val_adj = compute_value(
        adj,
        id_col="cusip",
        date_col="date",
        model_type="hz",
        cs_col="cs_adj",
        industry_col="ff17num",
        x_cols=['dcs3_adj', 'call'],
        dur_col='md_dur_adj',
        firm_col='issuer_cusip',
        y_transform="log",
        retransform="lognormal"
    )

    adj['log_md_dur'] = np.log(adj['md_dur_adj'])
    val_adj1 = compute_value(
        adj,
        id_col="cusip",
        date_col="date",
        model_type="ipr",
        cs_col="cs_adj",
        rating_mode="numeric",
        industry_col="ff17num",
        x_cols=['call', "log_md_dur", "vol12_x"],
        dur_col='md_dur_adj',
        firm_col='issuer_cusip',
        denom="resid",
        y_transform="log",
        retransform="lognormal"
    )

    val_adj = val_adj.merge(val_adj1, on=['cusip', 'date'], how='inner')
    del val_adj1

    # Rename adj value columns to have _adj suffix
    val_cols = [c for c in val_adj.columns if c.startswith('val_')]
    val_adj.rename(columns={c: f"{c}_adj" for c in val_cols}, inplace=True)

    if verbose:
        logger.info("  val_adj shape: %s", val_adj.shape)

    gc.collect()

    if verbose:
        logger.info("=" * 60)
        logger.info("MAKE VALUE SIGNALS - COMPLETE")
        logger.info("=" * 60)
        logger.info("  val_end columns: %s", list(val_end.columns))
        logger.info("  val_adj columns: %s", list(val_adj.columns))

    return val_end, val_adj
def build_d_spreads(
    end_signals_ext: pd.DataFrame,
    adj_signals_ext: pd.DataFrame,
    *,
    id_col: str = "cusip",
    date_col: str = "date",
    cs_col: str = "cs",
    bbtm_col: str = "bbtm",
    cs_adj_col: str = "cs_adj",
    bbtm_adj_col: str = "bbtm_adj",
    lags: tuple[int, ...] = (3, 6),
    bandwidth: int = 1,     # exact, then prefer (n+1) over (n-1): -1,+1,-2,+2,...
    mu_window: int = 12,
    use_numba_roll: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Very fast calendar-month lags via integer-month joins + (optional) numba rolling means.

    IMPORTANT preference:
    If lag n is missing and both (n+1) and (n-1) exist, choose (n+1).
    With month-keys, (n+1) corresponds to an *earlier* month relative to (m - n),
    so the fallback search order is: 0, -1, +1, -2, +2, ...

    Returns (d_spreads_std, d_spreads_adj).
    """

    # set to category inside (helps joins/grouping)
    if id_col in end_signals_ext.columns:
        end_signals_ext[id_col] = end_signals_ext[id_col].astype("category")
    if id_col in adj_signals_ext.columns:
        adj_signals_ext[id_col] = adj_signals_ext[id_col].astype("category")

    if bandwidth < 0:
        raise ValueError("bandwidth must be >= 0")

    # OFFSETS: exact, then earlier month first => (n+1) before (n-1)
    offsets = [0]
    for j in range(1, bandwidth + 1):
        offsets.extend([-j, +j])

    def _safe_log(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float64, copy=False)
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        m = arr > 0
        out[m] = np.log(arr[m])
        return out

    def _prep(df: pd.DataFrame, cs: str, bbtm: str) -> pd.DataFrame:
        d = df[[id_col, date_col, cs, bbtm]].copy()
        d[date_col] = pd.to_datetime(d[date_col])
        d = d.sort_values([id_col, date_col], kind="mergesort")
        dt = d[date_col].dt
        d["_m"] = (dt.year.astype(np.int32) * 12 + dt.month.astype(np.int32)).astype(np.int32)
        return d

    def _make_lookup(d: pd.DataFrame, cs: str, bbtm: str) -> pd.DataFrame:
        lk = d[[id_col, "_m", cs, bbtm]].drop_duplicates([id_col, "_m"], keep="last")
        return lk.rename(columns={cs: f"{cs}__lk", bbtm: f"{bbtm}__lk"})

    def _fill_lag(
        base: pd.DataFrame,
        lk: pd.DataFrame,
        cs: str,
        bbtm: str,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # target month for n-month lag is (m - n)
        m0 = base["_m"].to_numpy(np.int32, copy=False) - np.int32(n)

        cs_lag = np.full(len(base), np.nan, dtype=np.float64)
        bbtm_lag = np.full(len(base), np.nan, dtype=np.float64)

        tmp = base[[id_col]].copy()
        for off in offsets:
            # off=-1 => month (m-n-1) => lag (n+1)  [preferred]
            # off=+1 => month (m-n+1) => lag (n-1)
            tmp["_mt"] = (m0 + np.int32(off)).astype(np.int32, copy=False)

            got = tmp.merge(
                lk,
                left_on=[id_col, "_mt"],
                right_on=[id_col, "_m"],
                how="left",
                sort=False,
                copy=False,
            )

            a = got[f"{cs}__lk"].to_numpy(dtype=np.float64, copy=False)
            b = got[f"{bbtm}__lk"].to_numpy(dtype=np.float64, copy=False)

            cs_lag = np.where(np.isnan(cs_lag), a, cs_lag)
            bbtm_lag = np.where(np.isnan(bbtm_lag), b, bbtm_lag)

            if not np.isnan(cs_lag).any() and not np.isnan(bbtm_lag).any():
                break

        return cs_lag, bbtm_lag

    def _rolling_mean_fast(d: pd.DataFrame, col: str) -> np.ndarray:
        x = d[col].to_numpy(dtype=np.float64, copy=False)

        if use_numba_roll and _HAS_NUMBA:
            ids = d[id_col].to_numpy()
            # d is sorted by (id, date) so group blocks are contiguous
            change = np.empty(len(d), dtype=np.bool_)
            change[0] = True
            change[1:] = ids[1:] != ids[:-1]
            starts = np.flatnonzero(change).astype(np.int64)
            ends = np.empty_like(starts)
            ends[:-1] = starts[1:]
            ends[-1] = len(d)
            return _roll_mean_skip1_nan(x, starts, ends, mu_window)

        s = d.groupby(id_col, sort=False)[col].shift(1)
        return (
            s.groupby(d[id_col], sort=False)
             .rolling(mu_window, min_periods=6).mean()
             .reset_index(level=0, drop=True)
             .to_numpy(dtype=np.float64, copy=False)
        )

    def _build_one(df: pd.DataFrame, cs: str, bbtm: str, suffix: str) -> pd.DataFrame:
        d = _prep(df, cs, bbtm)
        lk = _make_lookup(d, cs, bbtm)

        cs_now = d[cs].to_numpy(dtype=np.float64, copy=False)
        bbtm_now = d[bbtm].to_numpy(dtype=np.float64, copy=False)

        log_cs_now = _safe_log(cs_now)
        log_bbtm_now = _safe_log(bbtm_now)

        out = d[[id_col, date_col]].copy()

        for n in lags:
            cs_lag, bbtm_lag = _fill_lag(d, lk, cs, bbtm, n=n)
            out[f"dcs{n}{suffix}"] = _safe_log(cs_lag) - log_cs_now
            out[f"dbbtm{n}{suffix}"] = _safe_log(bbtm_lag) - log_bbtm_now

        out[f"cs_mu{mu_window}_1{suffix}"] = _rolling_mean_fast(d, cs)
        out[f"bbtm_mu{mu_window}_1{suffix}"] = _rolling_mean_fast(d, bbtm)

        return out.reset_index(drop=True)

    d_spreads_std = _build_one(end_signals_ext, cs_col, bbtm_col, suffix="")
    d_spreads_adj = _build_one(adj_signals_ext, cs_adj_col, bbtm_adj_col, suffix="_adj")
    return d_spreads_std, d_spreads_adj

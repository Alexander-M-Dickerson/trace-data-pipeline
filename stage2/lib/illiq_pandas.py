"""illiq_pandas.py -- verbatim pandas/numba ports of the FLOAT32-ORDER-SENSITIVE illiquidity metrics.

Why these four live in pandas while the rest of step 2 is SQL (see debug.md M9): upstream computes
pi / amihud / ilq-roll with float32 Series arithmetic and float32 groupby SUMS whose value depends on
in-group accumulation order, and the risk kernel divides by near-singular denominators that amplify
any reassociation noise. DuckDB sums in parallel double precision -- close, but the near-singular
tail (pi up to 0.187, b_dvixd up to 4.9) cannot meet tolerance. These ports reproduce the upstream
arithmetic operation-for-operation (source: stage2/illiq_helper_functions.py, functions of the same
name); the only residual is numpy-version libm ulps.

All functions take the pin-derived frame in UPSTREAM column names:
  cusip_id (category), trd_exctn_dt, month_year (Period[M]), ret_c, ret_c_lag, ret_d, dvol,
  dvol_lag, day_gap, Nret, lst_txn  -- float32/Int dtypes as produced by lib/pin (bit-faithful).
Outputs: per-(cusip_id, date[month-end]) frames, as upstream.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from numba import njit

MIN_OBS_DEFAULT = 5


def _is_sorted_by(df: pd.DataFrame, id_col: str, date_col: str) -> bool:
    """O(n) check that df is (id, date)-block-sorted -- the pin pull already ships this order, so
    the risk function's mergesorts are identity permutations we can skip (W5 speed_up/02)."""
    ids = pd.factorize(df[id_col], sort=False)[0]
    d = df[date_col].to_numpy()
    di = np.diff(ids)
    return bool((di >= 0).all() and ((di > 0) | (d[1:] >= d[:-1])).all())


# ----------------------------------------------------------------------------------------------
# monthly_pi_fast (upstream lines 232-355)
# ----------------------------------------------------------------------------------------------
def monthly_pi_fast(df, id_col="cusip_id", date_col="trd_exctn_dt", month_col="month_year",
                    ret_col="ret_c", lag_ret_col="ret_c_lag", vol_col="dvol_lag",
                    lst_col="lst_txn", min_obs=MIN_OBS_DEFAULT):
    """Price impact: OLS re ~ 1 + lr_lag + svol_lag per bond-month (Cramer), pi = -beta2."""
    def _pi_from_sums(sums_df):
        n = sums_df['n'].to_numpy(dtype=float)
        s1, s2 = sums_df['sx1'].to_numpy(), sums_df['sx2'].to_numpy()
        S11, S22, S12 = sums_df['sx1x1'].to_numpy(), sums_df['sx2x2'].to_numpy(), sums_df['sx1x2'].to_numpy()
        Y0, Y1, Y2 = sums_df['sy'].to_numpy(), sums_df['sx1y'].to_numpy(), sums_df['sx2y'].to_numpy()
        detM = n * (S11 * S22 - S12**2) - s1 * (s1 * S22 - s2 * S12) + s2 * (s1 * S12 - s2 * S11)
        detM2 = n * (S11 * Y2 - S12 * Y1) - s1 * (s1 * Y2 - s2 * Y1) + Y0 * (s1 * S12 - s2 * S11)
        with np.errstate(divide='ignore', invalid='ignore'):
            beta2 = detM2 / detM
        beta2[(n < min_obs) | np.isclose(detM, 0)] = np.nan
        return beta2

    w = df[[id_col, date_col, month_col, ret_col, lag_ret_col, vol_col, lst_col]].copy()
    w['lr'] = np.log(1 + w[ret_col]) * 100
    w['lr_lag'] = np.log(1 + w[lag_ret_col]) * 100
    w['mkt'] = w.groupby(date_col, observed=True)['lr'].transform('mean')
    w['re'] = w['lr'] - w['mkt']
    w['mkt_lag'] = w.groupby(date_col, observed=True)['lr_lag'].transform('mean')
    w['re_lag'] = w['lr_lag'] - w['mkt_lag']
    w.drop(columns=['mkt', 'mkt_lag'], inplace=True)
    w['svol_lag'] = np.sign(w['re_lag']) * w[vol_col]
    w = w.dropna(subset=['re', 'lr_lag', 'svol_lag'])

    w['x1_sq'] = w['lr_lag'] ** 2
    w['x2_sq'] = w['svol_lag'] ** 2
    w['x1x2'] = w['lr_lag'] * w['svol_lag']
    w['x1y'] = w['lr_lag'] * w['re']
    w['x2y'] = w['svol_lag'] * w['re']
    agg_cols = {
        'n': ('re', 'size'), 'sy': ('re', 'sum'),
        'sx1': ('lr_lag', 'sum'), 'sx2': ('svol_lag', 'sum'),
        'sx1x1': ('x1_sq', 'sum'), 'sx2x2': ('x2_sq', 'sum'), 'sx1x2': ('x1x2', 'sum'),
        'sx1y': ('x1y', 'sum'), 'sx2y': ('x2y', 'sum'),
    }
    sums = w.groupby([month_col, id_col], observed=True).agg(**agg_cols).reset_index()
    sums['pi'] = _pi_from_sums(sums)
    agg = sums[[id_col, month_col, 'pi']].copy()
    del sums

    wa = w[w[lst_col] == 1].copy()
    del w
    sums_adj = wa.groupby([month_col, id_col], observed=True).agg(**agg_cols).reset_index()
    sums_adj['pi_adj'] = _pi_from_sums(sums_adj)
    agg_adj = sums_adj[[id_col, month_col, 'pi_adj']].copy()
    del wa, sums_adj

    out = agg.merge(agg_adj, on=[id_col, month_col], how='left')
    out['pi'] = -out['pi']
    out['pi_adj'] = -out['pi_adj']
    out['date'] = out[month_col].dt.to_timestamp('M').dt.normalize()
    out = out[[id_col, 'date', 'pi', 'pi_adj']].sort_values([id_col, 'date']).reset_index(drop=True)
    gc.collect()
    return out


# ----------------------------------------------------------------------------------------------
# compute_monthly_amihud (upstream lines 358-469)
# ----------------------------------------------------------------------------------------------
def compute_monthly_amihud(df, id_col="cusip_id", month_col="month_year", ret_col="ret_d",
                           vol_col="dvol", prc_lst_col="prc_lst", prc_hi_col="prc_hi",
                           prc_lo_col="prc_lo", lst_col="lst_txn", min_obs=MIN_OBS_DEFAULT):
    """Amihud ratio (mean + std) and LIX per bond-month, full + _adj."""
    keep = [id_col, month_col, ret_col, vol_col, prc_lst_col, prc_hi_col, prc_lo_col, lst_col]
    w = df[keep].copy()
    w['_ac'] = np.log(1 + w[ret_col]).abs() / w[vol_col].replace(0, np.nan)
    hl_range = (w[prc_hi_col] - w[prc_lo_col]).replace(0, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        w['_lix'] = -np.log10((w[vol_col] * w[prc_lst_col]) / hl_range)
    del hl_range

    g = w.groupby([id_col, month_col], observed=True)
    agg = g.agg(ami=('_ac', 'mean'), ami_v=('_ac', 'std'), lix=('_lix', 'mean'),
                _n=('_ac', 'count')).reset_index()
    agg.loc[agg['_n'] < min_obs, ['ami', 'ami_v', 'lix']] = np.nan
    agg.drop(columns='_n', inplace=True)
    del g

    wa = w[w[lst_col] == 1].copy()
    del w
    wa['_v'] = wa['_ac'].notna().astype('int8')
    wa['_nret'] = wa.groupby([id_col, month_col], observed=True)['_v'].transform('sum')
    wa = wa[wa['_nret'] >= min_obs]
    ga = wa.groupby([id_col, month_col], observed=True)
    agg_adj = ga.agg(ami_adj=('_ac', 'mean'), ami_v_adj=('_ac', 'std'),
                     lix_adj=('_lix', 'mean')).reset_index()
    del wa, ga

    out = agg.merge(agg_adj, on=[id_col, month_col], how='left')
    out['date'] = out[month_col].dt.to_timestamp('M').dt.normalize()
    out = out[[id_col, 'date', 'ami', 'ami_v', 'lix', 'ami_adj', 'ami_v_adj', 'lix_adj']]
    out = out.sort_values([id_col, 'date']).reset_index(drop=True)
    gc.collect()
    return out


# ----------------------------------------------------------------------------------------------
# compute_monthly_illiq_roll_fast (upstream lines 472-580)
# ----------------------------------------------------------------------------------------------
def compute_monthly_illiq_roll_fast(df, id_col="cusip_id", month_col="month_year",
                                    ret_col="ret_c", ret_lag_col="ret_c_lag",
                                    lst_col="lst_txn", min_obs=MIN_OBS_DEFAULT):
    """ilq = -Cov(lr, lr_lag) (the upstream mixed-n sample cov), roll = 2*sqrt(ilq) if >0 else 0."""
    def _ilq_roll(agg_df):
        num = agg_df['sxy'] - agg_df['sx'] * agg_df['sy'] / agg_df['n']
        denom = agg_df['n'] - 1
        ilq = -(num / denom)
        ilq[denom == 0] = np.nan
        with np.errstate(invalid='ignore'):
            roll = np.where(ilq > 0, 2 * np.sqrt(ilq), 0)
        return ilq, roll

    w = df[[id_col, month_col, ret_col, ret_lag_col, lst_col]].copy()
    w['lr'] = np.log(1 + w[ret_col]) * 100
    w['lr_lag'] = np.log(1 + w[ret_lag_col]) * 100
    w['prod'] = w['lr'] * w['lr_lag']
    w['_vp'] = (w['lr'].notna() & w['lr_lag'].notna()).astype('int8')

    w['_np'] = w.groupby([id_col, month_col], observed=True)['_vp'].transform('sum')
    wf = w[w['_np'] >= min_obs].copy()
    agg = wf.groupby([id_col, month_col], observed=True).agg(
        sx=('lr', 'sum'), sy=('lr_lag', 'sum'), sxy=('prod', 'sum'), n=('lr', 'size')).reset_index()
    ilq, roll = _ilq_roll(agg)
    agg['ilq'], agg['roll'] = ilq, roll
    agg = agg[[id_col, month_col, 'ilq', 'roll']]
    del wf

    wa = w[w[lst_col] == 1].copy()
    del w
    wa['_np'] = wa.groupby([id_col, month_col], observed=True)['_vp'].transform('sum')
    wa = wa[wa['_np'] >= min_obs]
    agg_adj = wa.groupby([id_col, month_col], observed=True).agg(
        sx=('lr', 'sum'), sy=('lr_lag', 'sum'), sxy=('prod', 'sum'), n=('lr', 'size')).reset_index()
    ilq_a, roll_a = _ilq_roll(agg_adj)
    agg_adj['ilq_adj'], agg_adj['roll_adj'] = ilq_a, roll_a
    agg_adj = agg_adj[[id_col, month_col, 'ilq_adj', 'roll_adj']]
    del wa

    out = agg.merge(agg_adj, on=[id_col, month_col], how='left')
    out['date'] = out[month_col].dt.to_timestamp('M').dt.normalize()
    out = out[[id_col, 'date', 'ilq', 'roll', 'ilq_adj', 'roll_adj']]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.sort_values([id_col, 'date']).reset_index(drop=True)
    gc.collect()
    return out


# ----------------------------------------------------------------------------------------------
# compute_within_month_risk + numba kernel (upstream lines 698-1115)
# ----------------------------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _group_stats_realized_vix_numba(y, f, vix, dvix, g, n_groups, min_obs):
    out = np.empty((n_groups, 13), dtype=np.float64)
    out[:] = np.nan
    n_total = y.shape[0]
    start = 0
    for gid in range(n_groups):
        s = start
        while start < n_total and g[start] == gid:
            start += 1
        e = start
        if e <= s:
            continue
        cnt = e - s
        if cnt < min_obs:
            continue
        nfloat = float(cnt)

        sum_y = 0.0; sum_f = 0.0; sum_y2 = 0.0; sum_f2 = 0.0; sum_yf = 0.0
        sum_r2 = 0.0; sum_r2_pos = 0.0; sum_r2_neg = 0.0; sum_r3 = 0.0; sum_r4 = 0.0
        n_v = 0.0; sum_v = 0.0; sum_v2 = 0.0; sum_y_v = 0.0; sum_yv = 0.0
        n_d = 0.0; sum_d = 0.0; sum_d2 = 0.0; sum_y_d = 0.0; sum_yd = 0.0
        for i in range(s, e):
            r = y[i]; fi = f[i]; vi = vix[i]; di = dvix[i]
            sum_y += r; sum_f += fi
            sum_y2 += r * r; sum_f2 += fi * fi; sum_yf += r * fi
            r2 = r * r
            sum_r2 += r2
            if r > 0.0:
                sum_r2_pos += r2
            elif r < 0.0:
                sum_r2_neg += r2
            sum_r3 += r2 * r
            sum_r4 += r2 * r2
            if not np.isnan(vi):
                n_v += 1.0; sum_v += vi; sum_v2 += vi * vi; sum_y_v += r; sum_yv += r * vi
            if not np.isnan(di):
                n_d += 1.0; sum_d += di; sum_d2 += di * di; sum_y_d += r; sum_yd += r * di

        mean_y = sum_y / nfloat
        mean_f = sum_f / nfloat
        rv = sum_r2
        if rv <= 0.0:
            continue
        rvol = np.sqrt(rv)
        rvp = np.sqrt(sum_r2_pos) if sum_r2_pos > 0.0 else 0.0
        rvn = np.sqrt(sum_r2_neg) if sum_r2_neg > 0.0 else 0.0
        rsj = (rvp - rvn) / rv
        denom_rsk = rv ** 1.5
        rsk = (np.sqrt(nfloat) * sum_r3) / denom_rsk if denom_rsk != 0.0 else np.nan
        denom_rkt = rv * rv
        rkt = (nfloat * sum_r4) / denom_rkt if denom_rkt != 0.0 else np.nan

        beta_vix = np.nan
        if n_v >= min_obs:
            den_v = n_v * sum_v2 - sum_v * sum_v
            if den_v != 0.0:
                beta_vix = (n_v * sum_yv - sum_v * sum_y_v) / den_v
        beta_dvix = np.nan
        if n_d >= min_obs:
            den_d = n_d * sum_d2 - sum_d * sum_d
            if den_d != 0.0:
                beta_dvix = (n_d * sum_yd - sum_d * sum_y_d) / den_d

        if cnt > 1:
            var_y = (sum_y2 - nfloat * mean_y * mean_y) / (nfloat - 1.0)
            var_f = (sum_f2 - nfloat * mean_f * mean_f) / (nfloat - 1.0)
            cov_yf = (sum_yf - nfloat * mean_y * mean_f) / (nfloat - 1.0)
        else:
            var_y = 0.0; var_f = 0.0; cov_yf = 0.0

        m2c = 0.0; m3c = 0.0; m4c = 0.0
        for i in range(s, e):
            dy = y[i] - mean_y
            d2 = dy * dy
            m2c += d2; m3c += d2 * dy; m4c += d2 * d2

        if var_y <= 0.0:
            vol = np.nan; skew = np.nan; kurt_ex = np.nan
        else:
            sd_y = np.sqrt(var_y)
            skew = (m3c / nfloat) / (sd_y ** 3)
            kurt_ex = (m4c / nfloat) / (sd_y ** 4) - 3.0
            vol = sd_y

        if var_f <= 0.0:
            beta_mkt = np.nan; vol_sys = np.nan; vol_idio = np.nan
        else:
            sd_f = np.sqrt(var_f)
            beta_mkt = cov_yf / var_f
            vol_sys = abs(beta_mkt) * sd_f
            var_idio = var_y - beta_mkt * beta_mkt * var_f
            if var_idio < 0.0:
                var_idio = 0.0
            vol_idio = np.sqrt(var_idio)

        out[gid, 0] = nfloat
        out[gid, 1] = vol; out[gid, 2] = skew; out[gid, 3] = kurt_ex
        out[gid, 4] = beta_mkt; out[gid, 5] = vol_sys; out[gid, 6] = vol_idio
        out[gid, 7] = rvol; out[gid, 8] = rsj; out[gid, 9] = rsk; out[gid, 10] = rkt
        out[gid, 11] = beta_vix; out[gid, 12] = beta_dvix
    return out


def compute_within_month_risk(df, vix_df, id_col="cusip_id", date_col="trd_exctn_dt",
                              month_col="month_year", ret_col="ret_d", day_gap_col="day_gap",
                              lst_col="lst_txn", min_obs=MIN_OBS_DEFAULT, mkt_col="mkt"):
    """Within-month risk stats + realized moments + VIX betas (full + _adj). vix_df required."""
    stat_cols = ["dvol", "dskew", "dkurt", "db_mkt", "dvol_sys", "dvol_idio",
                 "rvol", "rsj", "rsk", "rkt", "b_vix", "b_dvixd"]
    adj_cols = [c + "_adj" for c in stat_cols]
    out_cols = [id_col, "date"] + stat_cols + adj_cols

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if mkt_col not in df.columns:
        df[mkt_col] = df.groupby(date_col, observed=True)[ret_col].transform("mean")

    vix_df = vix_df.copy()
    vix_df["date"] = pd.to_datetime(vix_df["date"], errors="coerce")
    vix_df["vix"] = pd.to_numeric(vix_df["vix"], errors="coerce")
    vix_df = vix_df.dropna(subset=["date", "vix"])

    df = df.merge(vix_df.rename(columns={"date": date_col}), on=date_col, how="left")
    if not _is_sorted_by(df, id_col, date_col):        # identity permutation in the pipeline; the
        df = df.sort_values([id_col, date_col], kind="mergesort")   # sort stays as the fallback
    df["dvix"] = df.groupby(id_col, observed=True)["vix"].diff()
    if day_gap_col in df.columns:
        df.loc[df[day_gap_col] > 5, "dvix"] = np.nan

    keep = [id_col, month_col, ret_col, mkt_col, "vix", "dvix", lst_col]
    mask = df[ret_col].notna() & df[mkt_col].notna()
    df = df.loc[mask, keep].copy()
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    def _run_stats(sub):
        if sub.empty:
            return pd.DataFrame(columns=[id_col, month_col] + stat_cols)
        id_cat = sub[id_col].astype("category")
        m_cat = sub[month_col].astype("category")
        id_codes = id_cat.cat.codes.to_numpy(dtype=np.int64)
        m_codes = m_cat.cat.codes.to_numpy(dtype=np.int64)
        n_month = int(m_codes.max()) + 1
        pair_code = id_codes * n_month + m_codes
        g_codes, g_uniques = pd.factorize(pair_code, sort=False)
        g_codes = g_codes.astype(np.int64)
        n_groups = g_uniques.shape[0]
        if (np.diff(g_codes) >= 0).all():
            take = None                    # already group-contiguous ascending: skip the identity
        else:                              # argsort AND the four 29M-row gathers (W5 speed_up/02)
            take = np.argsort(g_codes, kind="mergesort")

        def _arr(col):
            a = sub[col].to_numpy(dtype=np.float64)
            return a if take is None else a[take]

        res = _group_stats_realized_vix_numba(
            y=_arr(ret_col), f=_arr(mkt_col), vix=_arr("vix"), dvix=_arr("dvix"),
            g=g_codes if take is None else g_codes[take],
            n_groups=n_groups, min_obs=int(min_obs))
        id_code_u = (g_uniques // n_month).astype(np.int64)
        m_code_u = (g_uniques % n_month).astype(np.int64)
        out_df = pd.DataFrame(res, columns=["n"] + stat_cols)
        out_df[id_col] = np.asarray(id_cat.cat.categories)[id_code_u]
        out_df[month_col] = np.asarray(m_cat.cat.categories)[m_code_u]
        out_df = out_df[out_df["n"].notna()].drop(columns=["n"])
        return out_df[[id_col, month_col] + stat_cols]

    full = _run_stats(df)
    adj = _run_stats(df[df[lst_col] == 1])
    del df
    gc.collect()
    if full.empty:
        return pd.DataFrame(columns=out_cols)

    out = full.merge(adj.rename(columns={c: c + "_adj" for c in stat_cols}),
                     on=[id_col, month_col], how="left")
    del full, adj
    out["date"] = pd.PeriodIndex(out[month_col]).to_timestamp("M")
    out = out.drop(columns=[month_col])
    out = out[[id_col, "date"] + stat_cols + adj_cols].reset_index(drop=True)
    gc.collect()
    return out

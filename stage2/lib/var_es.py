"""var_es.py -- rolling historical VaR/ES, verbatim port of upstream
the reference implementation's rolling VaR/ES kernels (loss-side, empirical quantile
floor(p*(T-1)) over an expanding-then-36m window, min 12 obs). Used by step 3 (var_95/var_95x) and
step 6 (var_90/es_90/var_95)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from numba import njit


@njit
def _rolling_var_es_numba(rets, ids_codes, min_obs, window, alphas):
    n = rets.shape[0]
    k = alphas.shape[0]
    var_res = np.full((n, k), np.nan)
    es_res = np.full((n, k), np.nan)
    if n == 0:
        return var_res, es_res

    group_starts = np.empty(n, dtype=np.int64)
    group_ends = np.empty(n, dtype=np.int64)
    gcount = 0
    cur_id = ids_codes[0]
    start = 0
    for i in range(1, n):
        if ids_codes[i] != cur_id:
            group_starts[gcount] = start
            group_ends[gcount] = i
            gcount += 1
            cur_id = ids_codes[i]
            start = i
    group_starts[gcount] = start
    group_ends[gcount] = n
    gcount += 1

    for gg in range(gcount):
        s = group_starts[gg]
        e = group_ends[gg]
        if e - s < min_obs:
            continue
        for i in range(s, e):
            end_i = i + 1
            beg_i = s if end_i - s <= window else end_i - window
            length = end_i - beg_i
            if length < min_obs:
                continue
            w = np.empty(length, dtype=np.float64)
            idx = 0
            for t in range(beg_i, end_i):
                w[idx] = rets[t]
                idx += 1
            w_sorted = w.copy()
            w_sorted.sort()
            for j in range(k):
                tail_prob = 1.0 - alphas[j]
                q_idx = int(np.floor(tail_prob * (length - 1)))
                if q_idx < 0:
                    q_idx = 0
                elif q_idx >= length:
                    q_idx = length - 1
                q = w_sorted[q_idx]
                var_val = -q
                sum_tail = 0.0
                cnt_tail = 0
                for t in range(length):
                    if w[t] <= q:
                        sum_tail += w[t]
                        cnt_tail += 1
                es_val = var_val if cnt_tail == 0 else -(sum_tail / cnt_tail)
                var_res[end_i - 1, j] = var_val
                es_res[end_i - 1, j] = es_val
    return var_res, es_res


def compute_rolling_var_es(df: pd.DataFrame, *, id_col: str = "cusip_id", date_col: str = "date",
                           ret_col: str = "ret_vw", min_obs: int = 12, window: int = 36,
                           alphas: Sequence[float] = (0.90, 0.95)) -> pd.DataFrame:
    """Per-id rolling VaR/ES; returns df_clean keys + var_<a>/es_<a> columns (NaN rets purged)."""
    alphas = tuple(float(a) for a in alphas)
    df_clean = df.dropna(subset=[ret_col]).loc[:, [id_col, date_col, ret_col]].copy()
    df_clean = df_clean.sort_values([id_col, date_col], kind="mergesort").reset_index(drop=True)
    rets = df_clean[ret_col].to_numpy(dtype=np.float64)
    codes, _ = pd.factorize(df_clean[id_col], sort=False)
    var_res, es_res = _rolling_var_es_numba(rets, codes.astype(np.int64), min_obs, window,
                                            np.asarray(alphas, dtype=np.float64))
    out = df_clean.copy()
    for j, a in enumerate(alphas):
        suffix = f"{int(round(a * 100))}"
        out[f"var_{suffix}"] = var_res[:, j]
        out[f"es_{suffix}"] = es_res[:, j]
    out.drop(columns=[ret_col], inplace=True)
    return out

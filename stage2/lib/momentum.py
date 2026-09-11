"""momentum.py -- VERBATIM momentum/LTR machinery from the reference implementation
(lines 6227-6356, 6360-6559): numba window helpers + build_mom_ltr_and_industry. Arbitrated by G6."""
import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit

_HAS_NUMBA = True
logger = logging.getLogger(__name__)

if _HAS_NUMBA:
    @njit(cache=True)
    def _group_bounds(gcodes):
        n = gcodes.shape[0]
        ng = 1
        for i in range(1, n):
            if gcodes[i] != gcodes[i - 1]:
                ng += 1
        starts = np.empty(ng, dtype=np.int64)
        ends   = np.empty(ng, dtype=np.int64)
        gi = 0
        starts[0] = 0
        for i in range(1, n):
            if gcodes[i] != gcodes[i - 1]:
                ends[gi] = i
                gi += 1
                starts[gi] = i
        ends[gi] = n
        return starts, ends

    @njit(cache=True)
    def _signals_fixed_fullwindow(lr, starts, ends, L, S):
        """
        Fixed (L,S) signals with YOUR convention:
          - S excludes the most-recent S months INCLUDING current month.
          - window months are t-(L-1) .. t-S (inclusive), length W=L-S.
        Full-window only: require at least L obs (available returns) before first signal.
        """
        n = lr.shape[0]
        K = L.shape[0]
        out = np.empty((n, K), dtype=np.float64)
        out[:] = np.nan

        for g in range(starts.shape[0]):
            s0 = starts[g]
            e0 = ends[g]
            m = e0 - s0

            pref = np.empty(m + 1, dtype=np.float64)
            pref[0] = 0.0
            for i in range(m):
                pref[i + 1] = pref[i] + lr[s0 + i]

            for ii in range(m):
                for k in range(K):
                    # end index (within group) is ii - S
                    end_i = ii - S[k]
                    if end_i < 0:
                        out[s0 + ii, k] = np.nan
                        continue

                    # full-window condition collapses to ii+1 >= L
                    # start index is ii - L + 1
                    start_i = ii - L[k] + 1
                    if start_i < 0:
                        out[s0 + ii, k] = np.nan
                        continue

                    ssum = pref[end_i + 1] - pref[start_i]
                    out[s0 + ii, k] = np.expm1(ssum)
        return out

    @njit(cache=True)
    def _signals_ltr_ramped(lr, starts, ends, L_star, S_star, L0, S0):
        """
        Ramped LTR only (reduces NaNs):
          - starts producing signals once ii+1 >= L0 (default L0=12).
          - effective L grows from L0 up to L_star as more history arrives:
                L_eff = min(L_star, ii+1) but at least L0 (and only defined if ii+1>=L0)
          - effective S grows from S0 up to S_star as L_eff increases:
                S_eff = S0 + floor((L_eff-L0)*(S_star-S0)/(L_star-L0)), capped in [S0,S_star]
          - uses YOUR convention: exclude most-recent S_eff months INCLUDING current.
          - full-window by construction (since L_eff <= ii+1).
        """
        n = lr.shape[0]
        K = L_star.shape[0]
        out = np.empty((n, K), dtype=np.float64)
        out[:] = np.nan

        for g in range(starts.shape[0]):
            s0 = starts[g]
            e0 = ends[g]
            m = e0 - s0

            pref = np.empty(m + 1, dtype=np.float64)
            pref[0] = 0.0
            for i in range(m):
                pref[i + 1] = pref[i] + lr[s0 + i]

            for ii in range(m):
                obs = ii + 1
                if obs < L0:
                    continue

                for k in range(K):
                    Ls = L_star[k]
                    Ss = S_star[k]

                    # L_eff ramps up to L*
                    Le = obs
                    if Le > Ls:
                        Le = Ls
                    if Le < L0:
                        Le = L0

                    # S_eff ramps from S0 to S*
                    Se = S0
                    if Ss <= S0 or Ls == L0:
                        Se = Ss
                    else:
                        num = (Le - L0) * (Ss - S0)
                        den = (Ls - L0)
                        Se = S0 + (num // den)
                        if Se > Ss:
                            Se = Ss
                        if Se < S0:
                            Se = S0

                    # end and start indices 
                    end_i = ii - Se
                    if end_i < 0:
                        continue
                    start_i = ii - Le + 1
                    if start_i < 0:
                        continue

                    ssum = pref[end_i + 1] - pref[start_i]
                    out[s0 + ii, k] = np.expm1(ssum)

        return out
def build_mom_ltr_and_industry(
    all_returns: pd.DataFrame,
    *,
    id_col: str = "cusip",
    date_col: str = "date",
    ind_col: str = "sic_code",
    ret_col: str = "ret_vw",
    ret_col_alt: str = "ret_vwx",
    ltr_ramp_L0: int = 12,
    ltr_ramp_S0: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extreme-speed panel signals (Numba required). Outputs:
      Asset momentum: mom3_1, mom6_1, mom9_1, mom12_1, mom12_7
      Asset LTR (ramped to reduce NaNs): ltr48_12, ltr30_6, ltr24_3
      Industry momentum: imom1, imom3_1, imom12_1   (equal-weight industry returns)
      Industry LTR (ramped): iltr48_12, iltr30_6, iltr24_3

    Naming convention (IMPORTANT):
      For X_L_S, S excludes the most-recent S months INCLUDING the current month.
      Window is months t-(L-1) .. t-S (inclusive), length W=L-S.

    Ramping applies ONLY to LTR specs listed above (asset + industry), per your request.
    """

    if not _HAS_NUMBA:
        raise RuntimeError("Numba not available. Install numba to use ultrafast version.")

    # ---- specs (L,S)  ----
    # Momentum
    MOM = [
        ("mom3_1", 3, 1),
        ("mom6_1", 6, 1),
        ("mom9_1", 9, 1),
        ("mom12_1", 12, 1),
        ("mom12_7", 12, 7),
    ]
    # LTR (ramped)
    LTR = [
        ("ltr48_12", 48, 12),
        ("ltr30_6", 30, 6),
        ("ltr24_3", 24, 3),
    ]

    # Industry momentum (imom1 = last month industry return => (L=2,S=1))
    IMOM = [
        ("imom1", 2, 1),
        ("imom3_1", 3, 1),
        ("imom12_1", 12, 1),
    ]
    # Industry LTR (ramped)
    ILTR = [
        ("iltr48_12", 48, 12),
        ("iltr30_6", 30, 6),
        ("iltr24_3", 24, 3),
    ]

    mom_names = [x[0] for x in MOM]
    mom_L = np.array([x[1] for x in MOM], dtype=np.int64)
    mom_S = np.array([x[2] for x in MOM], dtype=np.int64)

    ltr_names = [x[0] for x in LTR]
    ltr_Ls = np.array([x[1] for x in LTR], dtype=np.int64)
    ltr_Ss = np.array([x[2] for x in LTR], dtype=np.int64)

    imom_names = [x[0] for x in IMOM]
    imom_L = np.array([x[1] for x in IMOM], dtype=np.int64)
    imom_S = np.array([x[2] for x in IMOM], dtype=np.int64)

    iltr_names = [x[0] for x in ILTR]
    iltr_Ls = np.array([x[1] for x in ILTR], dtype=np.int64)
    iltr_Ss = np.array([x[2] for x in ILTR], dtype=np.int64)

    # ---- base arrays ----
    d = all_returns[[id_col, date_col, ind_col, ret_col, ret_col_alt]].copy()
    d[date_col] = pd.to_datetime(d[date_col])

    d[id_col] = d[id_col].astype("category")
    d[ind_col] = d[ind_col].astype("category")

    cusip_codes = d[id_col].cat.codes.to_numpy(np.int32, copy=False)
    sic_codes = d[ind_col].cat.codes.to_numpy(np.int32, copy=False)

    dt = d[date_col].dt
    m = (dt.year.to_numpy(np.int32, copy=False) * 12 + dt.month.to_numpy(np.int32, copy=False)).astype(np.int32, copy=False)
    m_min = int(m.min())
    m_off = (m - m_min).astype(np.uint32, copy=False)

    # row-level industry key (sic, month)
    row_key = (sic_codes.astype(np.int64) << 32) | m_off.astype(np.int64)

    def _one(retname: str) -> pd.DataFrame:
        r = d[retname].to_numpy(np.float64, copy=False)

        out = pd.DataFrame({
            id_col: d[id_col],
            date_col: d[date_col],
            ind_col: d[ind_col],
        })

        # =========================
        # Asset signals (mom fixed, ltr ramped)
        # =========================
        ok = ~np.isnan(r)
        if ok.any():
            idx = np.nonzero(ok)[0].astype(np.int64)
            cc = cusip_codes[ok]
            mm = m[ok]
            rr = r[ok]

            order = np.lexsort((mm, cc))  # cusip, then month
            idx_s = idx[order]
            cc_s = cc[order].astype(np.int64)
            rr_s = rr[order]

            lr = np.log1p(rr_s).astype(np.float64, copy=False)

            starts, ends = _group_bounds(cc_s)

            mom_sig = _signals_fixed_fullwindow(lr, starts, ends, mom_L, mom_S)
            ltr_sig = _signals_ltr_ramped(
                lr, starts, ends,
                ltr_Ls, ltr_Ss,
                np.int64(ltr_ramp_L0), np.int64(ltr_ramp_S0),
            )

            for j, nm in enumerate(mom_names):
                col = np.full(len(d), np.nan, dtype=np.float64)
                col[idx_s] = mom_sig[:, j]
                out[nm] = col

            for j, nm in enumerate(ltr_names):
                col = np.full(len(d), np.nan, dtype=np.float64)
                col[idx_s] = ltr_sig[:, j]
                out[nm] = col
        else:
            for nm in mom_names + ltr_names:
                out[nm] = np.nan

        # =========================
        # Industry equal-weight returns per (sic, month)
        # =========================
        ok2 = ok & (sic_codes >= 0)
        if ok2.any():
            sic2 = sic_codes[ok2]
            mo2 = m_off[ok2].astype(np.uint32, copy=False)
            rr2 = r[ok2]

            ikey = (sic2.astype(np.int64) << 32) | mo2.astype(np.int64)

            o2 = np.argsort(ikey, kind="mergesort")
            ikey_s = ikey[o2]
            rr2_s = rr2[o2]

            change = np.empty(len(ikey_s), dtype=np.bool_)
            change[0] = True
            change[1:] = ikey_s[1:] != ikey_s[:-1]
            gpos = np.nonzero(change)[0].astype(np.int64)

            sums = np.add.reduceat(rr2_s, gpos).astype(np.float64, copy=False)
            ones = np.ones(len(rr2_s), dtype=np.int64)
            cnts = np.add.reduceat(ones, gpos).astype(np.int64, copy=False)
            ind_ret = sums / cnts

            ind_key = ikey_s[gpos]  # sorted unique keys
            ind_sic = (ind_key >> 32).astype(np.int64)
            lr_ind = np.log1p(ind_ret).astype(np.float64, copy=False)

            starts_i, ends_i = _group_bounds(ind_sic)

            imom_sig = _signals_fixed_fullwindow(lr_ind, starts_i, ends_i, imom_L, imom_S)
            iltr_sig = _signals_ltr_ramped(
                lr_ind, starts_i, ends_i,
                iltr_Ls, iltr_Ss,
                np.int64(ltr_ramp_L0), np.int64(ltr_ramp_S0),
            )

            pos = np.searchsorted(ind_key, row_key)
            hit = (pos < ind_key.shape[0]) & (ind_key[pos] == row_key)

            for j, nm in enumerate(imom_names):
                col = np.full(len(d), np.nan, dtype=np.float64)
                col[hit] = imom_sig[pos[hit], j]
                out[nm] = col

            for j, nm in enumerate(iltr_names):
                col = np.full(len(d), np.nan, dtype=np.float64)
                col[hit] = iltr_sig[pos[hit], j]
                out[nm] = col
        else:
            for nm in imom_names + iltr_names:
                out[nm] = np.nan

        return out

    df_vw = _one(ret_col)
    df_vwx = _one(ret_col_alt)
    df_vw.drop(columns = ['sic_code'], inplace = True)
    df_vwx.drop(columns = ['sic_code'], inplace = True)
    return df_vw, df_vwx

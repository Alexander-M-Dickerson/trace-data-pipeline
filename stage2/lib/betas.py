"""betas.py -- rolling-beta / systematic-momentum / iskew orchestrators (upstream
the reference implementation). The numba kernels (lib/rolling_kernels) and the model
spec (BETA_MODELS) are verbatim.

The fast path: the old code re-merged, re-dropna'd and re-mergesorted the 2.3M-row
panel for EVERY model x return-type (~76 times) and combined results through ~76 outer merges. Now
one merged (cusip,date)-sorted base is built per return column (`_PanelBase`); each model takes a
NaN-mask view of it -- PROVABLY the same row sequence the old merge+dropna+sort produced, because
the base is block-sorted and masking preserves order -- and results scatter into base-aligned
arrays (union-of-masks presence == the old outer-merge key union). Kernel calls fan out on a thread
pool (the kernels are nogil). Values are bit-identical on keys; only the output ROW ORDER changed
(now fully (cusip,date)-sorted). Faithfulness is arbitrated by the beta validator
(betas_x 2,282,733 x 53). `compute_rolling_betas_panel` is the earlier reference path, kept as the
arbitration baseline."""
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from lib.rolling_kernels import (
    _panel_rolling_ols_k1, _panel_rolling_ols_k1_with_mom, _panel_rolling_ols_kgt1,
    _panel_rolling_ols_kgt1_with_mom, _panel_rolling_skew,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------
# BENCHMARK VARIANTS
#
# A duration-adjusted beta is not just "the same regression on a different y". The four bond-market
# factors are swapped for twins estimated on the SAME excess return -- step 3 builds them by
# re-running the BBW double-sorts. So a variant is a (return column, factor swap) pair, and adding
# a benchmark means adding both halves, not just the left-hand side.
#
# `tret` is the incumbent and its twins are the historical `*x` names. Any other benchmark uses
# `<factor>_<benchmark>`, which is what steps 3 emits for it.
# ----------------------------------------------------------------------------------------------
SWAPPED_FACTORS = ("mktb", "drf", "crf", "lrf")
FACTOR_SWAP_TRET = {f: f + "x" for f in SWAPPED_FACTORS}

# TERM is MKTB_raw - MKTBx, so it moves with the benchmark as well -- and `dcapm`, `psbm` and
# `amdm` all regress on `term` ALONGSIDE `mktbx`, so leaving it behind would put two different
# benchmarks on one right-hand side. The tret variant needs no entry for it: there was never a
# `termx`, because the historical `term` IS the tret-adjusted one.
BENCHMARK_FACTORS = SWAPPED_FACTORS + ("term",)


def factor_swap(benchmark: str | None) -> dict:
    """The factor twins to use for a benchmark. None means the raw factors (the `std` side)."""
    if benchmark is None:
        return {}
    if benchmark == "tret":
        return dict(FACTOR_SWAP_TRET)
    return {f: f"{f}_{benchmark}" for f in BENCHMARK_FACTORS}


# (key, return column, benchmark) -- the default reproduces the historical (betas_std, betas_x).
DEFAULT_VARIANTS = (("std", "ret_vw", None), ("x", "ret_vwx", "tret"))

# thread fan-out for the per-model kernel calls (kernels are nogil; pandas glue is brief)
N_JOBS = min(12, os.cpu_count() or 1)
RIDGE_DEFAULT = 1e-12   # K>1 rolling-OLS regularization (upstream compute_rolling_betas_panel default)
# group-aligned slices per model run: without chunking, the wall clock is pinned by the largest
# single model (the K=7 ring-buffer kernels run ~40-60 s alone); every kernel resets ALL state at a
# group boundary, so running the SAME kernel on group-aligned slices is value-identical
N_CHUNKS = N_JOBS


def _group_chunks(gid: np.ndarray, n_chunks: int) -> list[tuple[int, int]]:
    """~Equal-row [(start, end)) slices that NEVER split a gid group (kernel state resets there)."""
    n = len(gid)
    if n == 0:
        return [(0, 0)]
    starts = np.flatnonzero(np.diff(gid)) + 1          # every group start except row 0
    cuts = [0]
    for k in range(1, n_chunks):
        target = k * n // n_chunks
        j = np.searchsorted(starts, target)
        cut = int(starts[j]) if j < len(starts) else n
        if cuts[-1] < cut < n:
            cuts.append(cut)
    cuts.append(n)
    return list(zip(cuts[:-1], cuts[1:]))


class _PanelBase:
    """One merged (cusip,date)-sorted returns x factors panel; models take NaN-masked views.

    Replaces the per-model merge+dropna+factorize+mergesort: `combined_returns` ships
    (cusip,date)-sorted, a how='left' merge preserves left order, and first-appearance factorize
    codes are then non-decreasing -- so the old per-model sort was an identity permutation. The
    assert guards that invariant loudly instead of paying it 76 times.
    """

    def __init__(self, combined_returns: pd.DataFrame, factors: pd.DataFrame,
                 ret_col: str, factor_cols: list[str]):
        rdf = combined_returns[["cusip", "date", ret_col]].copy()
        rdf["date"] = pd.to_datetime(rdf["date"])
        fdf = factors[["date"] + factor_cols].copy()
        fdf["date"] = pd.to_datetime(fdf["date"])
        merged = rdf.merge(fdf, on="date", how="left", sort=False)
        merged = merged[merged[ret_col].notna()].reset_index(drop=True)

        gid = pd.factorize(merged["cusip"], sort=False)[0].astype(np.int32)
        dates = merged["date"].to_numpy()
        dg = np.diff(gid)
        assert (dg >= 0).all() and ((dg > 0) | (dates[1:] >= dates[:-1])).all(), \
            "base panel is not (cusip,date)-block-sorted -- the masked fast path would be wrong"

        self.cusip = merged["cusip"].to_numpy()
        self.date = dates
        self.gid = gid
        self.y = merged[ret_col].to_numpy(dtype=np.float64)
        self.fac = {c: merged[c].to_numpy(dtype=np.float64) for c in factor_cols}
        self.n = len(merged)

    def view(self, fac_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        """(row_idx, gid, y, [x per factor]) for rows where y and ALL fac_cols are non-NaN --
        the exact row set the old dropna(subset=[ret]+facs) kept, in the same order."""
        m = np.ones(self.n, dtype=bool)
        for c in fac_cols:
            m &= ~np.isnan(self.fac[c])
        idx = np.flatnonzero(m)
        return idx, self.gid[idx], self.y[idx], [self.fac[c][idx] for c in fac_cols]

def compute_rolling_iskew(
    returns_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    betas_df: pd.DataFrame,
    id_col: str = "cusip",
    date_col: str = "date",
    ret_col: str = "ret_vw",
    factor_cols: list = None,
    window: int = 36,
    min_obs: int = 12,
) -> pd.DataFrame:
    """
    Compute rolling idiosyncratic skewness (skewness of residuals).

    Uses numba-optimized rolling skewness computation for speed.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns data with id_col, date_col, ret_col
    factor_df : pd.DataFrame
        Factor data with date_col and factor columns
    betas_df : pd.DataFrame
        Beta estimates with id_col, date_col, and beta_{factor} columns
    id_col : str
        ID column name
    date_col : str
        Date column name
    ret_col : str
        Return column name
    factor_cols : list
        List of factor column names (must match beta columns)
    window : int
        Rolling window size
    min_obs : int
        Minimum observations required

    Returns
    -------
    pd.DataFrame
        DataFrame with id_col, date_col, and iskew column
    """
    if factor_cols is None:
        raise ValueError("factor_cols must be specified")

    # Merge returns with factors
    rdf = returns_df[[id_col, date_col, ret_col]].copy()
    rdf[date_col] = pd.to_datetime(rdf[date_col])

    fdf = factor_df[[date_col] + factor_cols].copy()
    fdf[date_col] = pd.to_datetime(fdf[date_col])

    merged = pd.merge(rdf, fdf, on=date_col, how='left')

    # Merge with betas
    beta_cols = [f"beta_{f}" for f in factor_cols]
    betas_sub = betas_df[[id_col, date_col] + [c for c in beta_cols if c in betas_df.columns]].copy()
    betas_sub[date_col] = pd.to_datetime(betas_sub[date_col])

    merged = pd.merge(merged, betas_sub, on=[id_col, date_col], how='left')
    merged = merged.dropna(subset=[ret_col] + factor_cols).reset_index(drop=True)
    merged = merged.sort_values([id_col, date_col]).reset_index(drop=True)

    # Compute residuals: ret - alpha - sum(beta_k * factor_k)
    # Note: we include intercept in beta (beta_0 is alpha)
    # But compute_rolling_betas_panel stores betas separately without alpha by default
    # So residual = ret - sum(beta_k * factor_k)
    resid = merged[ret_col].values.copy()
    for f in factor_cols:
        beta_col = f"beta_{f}"
        if beta_col in merged.columns:
            resid -= merged[beta_col].fillna(0).values * merged[f].values

    # Factorize id to int codes for numba
    gid, _ = pd.factorize(merged[id_col], sort=False)
    gid = gid.astype(np.int32)

    # Compute rolling skewness using numba
    iskew = _panel_rolling_skew(gid, resid, int(window), int(min_obs))

    return pd.DataFrame({
        id_col: merged[id_col].values,
        date_col: merged[date_col].values,
        'iskew': iskew,
    })


# ============================================================
# Single public function: K=1 or K>1 with insane speed
# ============================================================
def compute_rolling_betas_panel(
    returns_df,
    factor_df,
    id_col="permno",
    date_col="date",
    ret_col="rets",
    window=36,
    min_obs=24,
    factor_cols=None,
    model_name="CAPM",
    ridge=1e-12,
    include_alpha=False,
):
    """
    Ultra-fast rolling betas on a panel.
    - K=1: closed-form moments (fastest)
    - K>1: rolling-moment updates + per-row small linear solve
    """
    model_suffix = str(model_name).replace(" ", "")

    # normalize factor_df to have date as a column
    if date_col in getattr(factor_df.index, "names", [None]):
        fdf = factor_df.reset_index()
    else:
        fdf = factor_df.copy()

    if factor_cols is None:
        factor_cols = [c for c in fdf.columns if c != date_col]

    # minimal columns
    rdf = returns_df[[id_col, date_col, ret_col]].copy()
    rdf[date_col] = pd.to_datetime(rdf[date_col])
    fdf[date_col] = pd.to_datetime(fdf[date_col])

    # avoid collisions
    rename_map = {}
    fac_cols = []
    for c in factor_cols:
        if c == ret_col:
            rename_map[c] = c + "_fac"
            fac_cols.append(c + "_fac")
        else:
            fac_cols.append(c)
    if rename_map:
        fdf = fdf.rename(columns=rename_map)

    # merge + drop NaNs (so numba sees clean arrays)
    merged = pd.merge(
        rdf,
        fdf[[date_col] + fac_cols],
        on=date_col,
        how="left",
        sort=False,
    )
    merged = merged.dropna(subset=[ret_col] + fac_cols).reset_index(drop=True)

    # factorize id to int codes, then sort by (_gid, date)
    gid, _ = pd.factorize(merged[id_col], sort=False)
    merged["_gid"] = gid.astype(np.int32, copy=False)
    merged = merged.sort_values(["_gid", date_col], kind="mergesort").reset_index(drop=True)

    gid_arr = merged["_gid"].to_numpy(dtype=np.int32, copy=False)
    y = merged[ret_col].to_numpy(dtype=np.float64, copy=False)

    K = len(fac_cols)

    if K == 1:
        x = merged[fac_cols[0]].to_numpy(dtype=np.float64, copy=False)
        a, b, sig_tot, sig_idi, adj_r2 = _panel_rolling_ols_k1(
            gid_arr, y, x, int(window), int(min_obs)
        )

        out = pd.DataFrame(
            {
                id_col: merged[id_col].to_numpy(copy=False),
                date_col: merged[date_col].to_numpy(copy=False),
                f"beta_{fac_cols[0]}": b,
                "sigma_total": sig_tot,
                f"sigma_idio_{model_suffix}": sig_idi,
                f"adj_R2_{model_suffix}": adj_r2,
            }
        )
        if include_alpha:
            out[f"alpha_{model_suffix}"] = a
        return out

    # K > 1
    X = merged[fac_cols].to_numpy(dtype=np.float64, copy=False)

    betas, sig_tot, sig_idi, adj_r2 = _panel_rolling_ols_kgt1(
        gid_arr, y, X, int(window), int(min_obs), float(ridge)
    )

    cols = {
        id_col: merged[id_col].to_numpy(copy=False),
        date_col: merged[date_col].to_numpy(copy=False),
        "sigma_total": sig_tot,
        f"sigma_idio_{model_suffix}": sig_idi,
        f"adj_R2_{model_suffix}": adj_r2,
    }

    # drop intercept by default (matches your earlier output)
    for j, c in enumerate(fac_cols):
        cols[f"beta_{c}"] = betas[:, j + 1]

    out = pd.DataFrame(cols)

    if include_alpha:
        out[f"alpha_{model_suffix}"] = betas[:, 0]

    return out


def compute_all_betas(
    combined_returns: pd.DataFrame,
    factors: pd.DataFrame,
    window: int = 36,
    min_obs: int = 12,
    verbose: bool = True,
    variants: tuple = DEFAULT_VARIANTS,
) -> tuple:
    """
    Compute rolling betas for all factor models.

    Parameters
    ----------
    combined_returns : pd.DataFrame
        Bond returns with columns: cusip, date, ret_vw, ret_vwx
    factors : pd.DataFrame
        Factor data with date column and all required factors
    window : int, default 36
        Rolling window size (months)
    min_obs : int, default 12
        Minimum observations required
    verbose : bool, default True
        If True, log progress
    variants : tuple, default DEFAULT_VARIANTS
        One (key, return column, benchmark) triple per set of betas to produce. The benchmark
        selects the factor twins via `factor_swap`; None means the raw factors. Pass a single
        triple to compute one set and pay a fraction of the default's cost.

    Returns
    -------
    tuple of pd.DataFrame, one per variant, in the order given
        With the default variants this is (betas_std, betas_x), unchanged.
    """
    # Beta model configuration
    BETA_MODELS = [
        # Models with model suffix (multiple outputs or conflicts)
        {"name": "mkt", "factors": ["mktrf", "mktb"], "keep": None, "sum": None, "ivol": True,
         "out": {"mktrf": "mktrf_mkt", "mktb": "mktb_mkt"}},
        {"name": "bbw", "factors": ["mktb", "drf", "crf", "lrf"], "keep": [], "sum": None, "ivol": True,
         "out": {}},
        {"name": "dcapm", "factors": ["mktbx", "term"], "keep": None, "sum": None, "ivol": False,
         "out": {"mktbx": "mktbx_dcapm", "term": "term_dcapm"}},
        {"name": "volam", "factors": ["mktrf", "smb", "hml", "vix", "dvix", "dvixlag", "amd"],
         "keep": ["dvix"], "sum": ("dvix", ["dvix", "dvixlag"]), "ivol": False,
         "out": {"dvix": "dvix_va"}},
        {"name": "volpsb", "factors": ["mktrf", "smb", "hml", "vix", "dvix", "dvixlag", "psb"],
         "keep": ["dvix"], "sum": ("dvix", ["dvix", "dvixlag"]), "ivol": True,
         "out": {"dvix": "dvix_vp"}},
        {"name": "psbm", "factors": ["mktrf", "smb", "hml", "mktbx", "term", "psb"],
         "keep": ["psb"], "sum": None, "ivol": False,
         "out": {"psb": "psb_m"}},
        {"name": "amdm", "factors": ["mktrf", "smb", "hml", "mktbx", "term", "amd"],
         "keep": ["amd"], "sum": None, "ivol": False,
         "out": {"amd": "amd_m"}},
        # Models with simple naming (unique output)
        {"name": "vix", "factors": ["mktb", "mktrf", "dvix", "dvixlag"],
         "keep": ["dvix"], "sum": ("dvix", ["dvix", "dvixlag"]), "ivol": False,
         "out": {"dvix": "dvix"}},
        {"name": "inflv", "factors": ["mktb", "cpi_vol6"], "keep": ["cpi_vol6"], "sum": None, "ivol": False,
         "out": {"cpi_vol6": "cpi_vol6"}},
        {"name": "unc", "factors": ["mktb", "dunc"], "keep": ["dunc"], "sum": None, "ivol": False,
         "out": {"dunc": "dunc"}},
        {"name": "uncl", "factors": ["mktb", "unc"], "keep": ["unc"], "sum": None, "ivol": False,
         "out": {"unc": "unc"}},
        {"name": "unc3", "factors": ["mktb", "dunc3"], "keep": ["dunc3"], "sum": None, "ivol": False,
         "out": {"dunc3": "dunc3"}},
        {"name": "unc6", "factors": ["mktb", "dunc6"], "keep": ["dunc6"], "sum": None, "ivol": False,
         "out": {"dunc6": "dunc6"}},
        {"name": "uncr", "factors": ["mktb", "duncr"], "keep": ["duncr"], "sum": None, "ivol": False,
         "out": {"duncr": "duncr"}},
        {"name": "uncf", "factors": ["mktb", "duncf"], "keep": ["duncf"], "sum": None, "ivol": False,
         "out": {"duncf": "duncf"}},
        {"name": "credd", "factors": ["mktb", "dcredit"], "keep": ["dcredit"], "sum": None, "ivol": False,
         "out": {"dcredit": "dcredit"}},
        {"name": "credl", "factors": ["mktb", "credit"], "keep": ["credit"], "sum": None, "ivol": False,
         "out": {"credit": "credit"}},
        {"name": "infl", "factors": ["mktb", "dcpi"], "keep": ["dcpi"], "sum": None, "ivol": False,
         "out": {"dcpi": "dcpi"}},
        {"name": "hkm", "factors": ["mktrf", "cptlt"], "keep": ["cptlt"], "sum": None, "ivol": False,
         "out": {"cptlt": "cptlt"}},
        {"name": "rvol", "factors": ["mktb", "rvol"], "keep": ["rvol"], "sum": None, "ivol": False,
         "out": {"rvol": "rvol"}},
        {"name": "rsj", "factors": ["mktb", "rsj"], "keep": ["rsj"], "sum": None, "ivol": False,
         "out": {"rsj": "rsj"}},
        {"name": "psb", "factors": ["mktb", "psb"], "keep": ["psb"], "sum": None, "ivol": False,
         "out": {"psb": "psb"}},
        {"name": "amd", "factors": ["mktb", "amd"], "keep": ["amd"], "sum": None, "ivol": False,
         "out": {"amd": "amd"}},
        {"name": "illiq", "factors": ["mktb", "illiq"], "keep": ["illiq"], "sum": None, "ivol": False,
         "out": {"illiq": "illiq"}},
        # Asymmetric VIX model (control for mktrf)
        {"name": "dvix_asym", "factors": ["mktrf", "dvix_down", "dvix_up"], "keep": ["dvix_down", "dvix_up"], "sum": None, "ivol": False,
         "out": {"dvix_down": "dvix_dn", "dvix_up": "dvix_up"}},
        # Coskewness model (mktb + mktb^2) - coskew is beta on mktb_sq
        {"name": "coskew", "factors": ["mktb", "mktb_sq"], "keep": ["mktb_sq"], "sum": None, "ivol": False,
         "out": {"mktb_sq": "coskew"}, "iskew": True},
        # Univariate models
        # DEF / TERM -- Gebhardt, Hvidkjaer & Swaminathan (2005); factors as in Fama-French
        # (1993). One two-factor regression yields both loadings:
        #
        #     r = a + b_term * TERMB + b_def * DEFB + e
        #
        # TERMB = long-term government return - risk free; DEFB = long-term corporate
        # return - long-term government return (built in step3_bbw).
        #
        # ! These were previously TWO UNIVARIATE models -- `def` on mktbx and `term` on
        #   term -- and neither estimated what its name claimed. No `defb` series existed;
        #   "defb" was only an output rename of the mktbx loading. Worse, in the
        #   duration-adjusted world FACTOR_SWAP rewrites mktb -> mktbx, which made the
        #   `def` and `mktb` models the SAME regression: b_defb and b_mktb came out
        #   bit-identical on every row. See the DEF/TERM entry in DATA_DICTIONARY.md.
        # `keep` fixes the OUTPUT column order and `factors` the design matrix, so keeping
        # defb first here preserves the panel's historical b_defb/b_termb ordering while
        # leaving the regression itself untouched (bit-identical betas).
        {"name": "defterm", "factors": ["termb", "defb"], "keep": ["defb", "termb"],
         "sum": None, "ivol": False, "out": {"defb": "defb", "termb": "termb"}},
        {"name": "drf", "factors": ["drf"], "keep": None, "sum": None, "ivol": False,
         "out": {"drf": "drf"}},
        {"name": "crf", "factors": ["crf"], "keep": None, "sum": None, "ivol": False,
         "out": {"crf": "crf"}},
        {"name": "lrf", "factors": ["lrf"], "keep": None, "sum": None, "ivol": False,
         "out": {"lrf": "lrf"}},
        {"name": "mktb", "factors": ["mktb"], "keep": None, "sum": None, "ivol": False,
         "out": {"mktb": "mktb"}},
        {"name": "lvl", "factors": ["lvl"], "keep": None, "sum": None, "ivol": False,
         "out": {"lvl": "lvl"}},
        {"name": "ysp", "factors": ["ysp"], "keep": None, "sum": None, "ivol": False,
         "out": {"ysp": "ysp"}},
        # Asymmetric market model
        {"name": "mktb_asym", "factors": ["mktb_down", "mktb_up"], "keep": None, "sum": None, "ivol": False,
         "out": {"mktb_down": "mktb_dn", "mktb_up": "mktb_up"}},
        # EPU models
        {"name": "epu", "factors": ["mktb", "epu"], "keep": ["epu"], "sum": None, "ivol": False,
         "out": {"epu": "epu"}},
        {"name": "epum", "factors": ["mktb", "epum"], "keep": ["epum"], "sum": None, "ivol": False,
         "out": {"epum": "epum"}},
        {"name": "eput", "factors": ["mktb", "eput"], "keep": ["eput"], "sum": None, "ivol": False,
         "out": {"eput": "eput"}},
    ]

    swaps = {key: factor_swap(bm) for key, _ret, bm in variants}

    # every factor column any model can touch, raw + every variant's twins (one base carries them
    # all, so the merge is paid once however many variants are asked for)
    factor_cols_all = sorted({f for m in BETA_MODELS for f in m["factors"]}
                             | {sw.get(f, f) for sw in swaps.values()
                                for m in BETA_MODELS for f in m["factors"]})
    missing = [c for c in factor_cols_all if c not in factors.columns]
    if missing:
        raise KeyError(
            f"the factor panel is missing {missing}. A benchmark variant needs its OWN bond-market "
            f"factor twins -- step 3 emits them as bbw_factors_<benchmark>.parquet. Computing it "
            f"against the wrong twins would produce plausible, wrong betas.")

    def _submit_model(ex: ThreadPoolExecutor, base: _PanelBase, model: dict, swap: dict):
        """Stage A: mask the base for this model and submit one kernel task per group-aligned
        chunk. Returns (idx, facs_use, futures) for _finish_model."""
        facs = model["factors"]
        facs_use = [swap.get(f, f) for f in facs]
        idx, gid, y, xs = base.view(facs_use)
        chunks = _group_chunks(gid, N_CHUNKS)
        if len(facs_use) == 1:
            x = xs[0]
            futs = [ex.submit(_panel_rolling_ols_k1, gid[s:e], y[s:e], x[s:e],
                              int(window), int(min_obs)) for s, e in chunks]
        else:
            X = np.column_stack(xs)
            futs = [ex.submit(_panel_rolling_ols_kgt1, gid[s:e], y[s:e], X[s:e],
                              int(window), int(min_obs), RIDGE_DEFAULT) for s, e in chunks]
        return idx, facs_use, futs

    def _finish_model(base: _PanelBase, model: dict, swap: dict, ret_col: str,
                      idx: np.ndarray, facs_use: list[str], futs: list):
        """Stage B: concatenate the chunk outputs (value-identical to one full-pass kernel call --
        every kernel resets at group boundaries) and apply the model's naming spec."""
        parts = [f.result() for f in futs]
        if len(facs_use) == 1:
            beta = np.concatenate([p[1] for p in parts])
            sig_idi = np.concatenate([p[3] for p in parts])
            beta_by_fac = {facs_use[0]: beta}
        else:
            betas_arr = np.concatenate([p[0] for p in parts], axis=0)
            sig_idi = np.concatenate([p[2] for p in parts])
            beta_by_fac = {c: betas_arr[:, j + 1] for j, c in enumerate(facs_use)}

        # SUM operation (e.g. dvix = dvix + dvixlag); overwrites the regression coefficient of the
        # same name, exactly like the old `betas[f"beta_{out_name}"] = beta_sum`
        sum_op = model["sum"]
        if sum_op is not None:
            out_name, sum_cols = sum_op
            sum_cols_use = [swap.get(c, c) for c in sum_cols]
            acc = beta_by_fac[sum_cols_use[0]].copy()
            for c in sum_cols_use[1:]:
                acc = acc + beta_by_fac[c]          # NaN propagates == sum(skipna=False)
            beta_by_fac[out_name] = acc

        facs, keep, out_map, mname = model["factors"], model["keep"], model["out"], model["name"]
        named: dict[str, np.ndarray] = {}
        if keep is None:
            for f_orig, f_use in zip(facs, facs_use):
                named[f"b_{out_map.get(f_orig, f_orig)}"] = beta_by_fac[f_use]
        else:
            for f_orig in keep:
                f_use = swap.get(f_orig, f_orig)
                named[f"b_{out_map.get(f_orig, f_orig)}"] = beta_by_fac[f_use]
        if sum_op is not None:
            out_name_sum, _ = sum_op
            named[f"b_{out_map.get(out_name_sum, out_name_sum)}"] = beta_by_fac[out_name_sum]

        if model["ivol"]:
            if out_map:
                ivol_name = out_map.get(list(out_map.keys())[0], mname).split("_")[-1] \
                    if "_" in list(out_map.values())[0] else mname
            else:
                ivol_name = mname
            named[f"ivol_{ivol_name}"] = sig_idi

        if model.get("iskew", False):
            bdf = pd.DataFrame({"cusip": base.cusip[idx], "date": base.date[idx]})
            for c in facs_use:
                bdf[f"beta_{c}"] = beta_by_fac[c]
            iskew_df = compute_rolling_iskew(
                returns_df=combined_returns, factor_df=factors, betas_df=bdf,
                id_col="cusip", date_col="date", ret_col=ret_col,
                factor_cols=facs_use, window=window, min_obs=min_obs)
            aligned = bdf[["cusip", "date"]].merge(iskew_df, on=["cusip", "date"], how="left")
            named["iskew"] = aligned["iskew"].to_numpy()

        return idx, named

    ret_of = {key: ret for key, ret, _bm in variants}
    missing_ret = [r for r in ret_of.values() if r not in combined_returns.columns]
    if missing_ret:
        raise KeyError(f"combined_returns is missing {missing_ret}")
    bases = {key: _PanelBase(combined_returns, factors, ret_of[key], factor_cols_all)
             for key, _ret, _bm in variants}

    tasks = [(model, key) for model in BETA_MODELS for key, _r, _b in variants]
    if verbose:
        logger.info("  Computing %d model runs x %d chunks on %d threads (%d variant(s): %s)...",
                    len(tasks), N_CHUNKS, N_JOBS, len(variants),
                    ", ".join(k for k, _r, _b in variants))
    with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
        submitted = [(model, key, _submit_model(ex, bases[key], model, swaps[key]))
                     for model, key in tasks]
        outputs = [_finish_model(bases[key], model, swaps[key], ret_of[key],
                                 idx, facs_use, futs)
                   for model, key, (idx, facs_use, futs) in submitted]

    # scatter-assembly: union-of-masks presence + base-aligned columns reproduce the old outer-merge
    # chain exactly (values on keys identical; row order now fully (cusip,date)-sorted)
    frames: dict[str, pd.DataFrame] = {}
    for side, _ret, _bm in variants:
        base = bases[side]
        present = np.zeros(base.n, dtype=bool)
        cols: dict[str, np.ndarray] = {}
        for (model, key), (idx, named) in zip(tasks, outputs):
            if key != side:
                continue
            present[idx] = True
            for name, vals in named.items():
                assert name not in cols, f"duplicate beta output column {name!r}"
                arr = np.full(base.n, np.nan)
                arr[idx] = vals
                cols[name] = arr
        keep_rows = np.flatnonzero(present)
        frame = pd.DataFrame({"cusip": base.cusip[keep_rows], "date": base.date[keep_rows]})
        for name, arr in cols.items():
            frame[name] = arr[keep_rows]
        frames[side] = frame

    out = tuple(frames[key] for key, _r, _b in variants)
    if verbose:
        for (key, _r, _b), frame in zip(variants, out):
            logger.info("  betas_%s shape: %s", key, frame.shape)

    return out


def compute_sys_momentum(
    combined_returns: pd.DataFrame,
    factors: pd.DataFrame,
    factor_cols: list = None,
    window: int = 36,
    min_obs: int = 12,
    ridge: float = 1e-12,
    verbose: bool = True,
    variants: tuple = DEFAULT_VARIANTS,
) -> tuple:
    """
    Compute systematic and idiosyncratic momentum using rolling factor model.

    Momentum signals are computed INSIDE the rolling OLS loop: at each position
    where we have valid alpha/beta, we use the CURRENT alpha/beta to compute
    fitted values for ALL positions in the rolling window, then sum to get
    momentum. This ensures momentum is available exactly when beta is available.

    Parameters
    ----------
    combined_returns : pd.DataFrame
        Bond returns with columns: cusip, date, ret_vw, ret_vwx
    factors : pd.DataFrame
        Factor data with date column and factor columns
    factor_cols : list, optional
        List of factor columns to use. Default: ["mktb"]
        Examples: ["mktb"], ["mktb", "drf", "crf"]
    window : int, default 36
        Rolling window size (months)
    min_obs : int, default 12
        Minimum observations required
    ridge : float, default 1e-12
        Ridge regularization for multi-factor models
    verbose : bool, default True
        If True, log progress

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (mom_std, mom_x) - momentum signals from normal and duration-adjusted returns
        Each contains: cusip, date, sysmom3_1, sysmom6_1, sysmom12_1,
                       idimom3_1, idimom6_1, idimom12_1
    """
    if factor_cols is None:
        factor_cols = ["mktb"]

    results: dict = {}

    for _key, ret_col, _bm in variants:
        facs_use = [factor_swap(_bm).get(f, f) for f in factor_cols]

        if verbose:
            fac_str = ", ".join(facs_use)
            logger.info("  Computing momentum for %s with factors: [%s]", ret_col, fac_str)

        # Prepare data
        rdf = combined_returns[["cusip", "date", ret_col]].copy()
        rdf["date"] = pd.to_datetime(rdf["date"])

        fdf = factors[["date"] + facs_use].copy()
        fdf["date"] = pd.to_datetime(fdf["date"])

        merged = pd.merge(rdf, fdf, on="date", how="left")
        merged = merged.dropna(subset=[ret_col] + facs_use).reset_index(drop=True)

        # Factorize cusip and sort
        gid, _ = pd.factorize(merged["cusip"], sort=False)
        merged["_gid"] = gid.astype(np.int32)
        merged = merged.sort_values(["_gid", "date"], kind="mergesort").reset_index(drop=True)

        gid_arr = merged["_gid"].to_numpy(dtype=np.int32)
        y = merged[ret_col].to_numpy(dtype=np.float64)

        K = len(facs_use)

        if K == 1:
            x = merged[facs_use[0]].to_numpy(dtype=np.float64)
            (alpha, beta,
             sysmom3_1, sysmom6_1, sysmom12_1,
             idimom3_1, idimom6_1, idimom12_1) = _panel_rolling_ols_k1_with_mom(
                gid_arr, y, x, int(window), int(min_obs)
            )
        else:
            X = merged[facs_use].to_numpy(dtype=np.float64)
            (betas,
             sysmom3_1, sysmom6_1, sysmom12_1,
             idimom3_1, idimom6_1, idimom12_1) = _panel_rolling_ols_kgt1_with_mom(
                gid_arr, y, X, int(window), int(min_obs), float(ridge)
            )

        # Build output dataframe
        mom_df = pd.DataFrame({
            "cusip": merged["cusip"].values,
            "date": merged["date"].values,
            "sysmom3_1": sysmom3_1,
            "sysmom6_1": sysmom6_1,
            "sysmom12_1": sysmom12_1,
            "idimom3_1": idimom3_1,
            "idimom6_1": idimom6_1,
            "idimom12_1": idimom12_1,
        })

        results[_key] = mom_df

    # The historical contract is a (std, x) pair. For any other variant set, return the frames in
    # the order asked for -- the caller knows which benchmark it requested.
    if tuple(k for k, _r, _b in variants) == ("std", "x"):
        mom_std = results.get("std", pd.DataFrame())
        mom_x = results.get("x", pd.DataFrame())
    else:
        return tuple(results[k] for k, _r, _b in variants)

    if verbose:
        logger.info("  mom_std shape: %s", mom_std.shape)
        logger.info("  mom_x shape: %s", mom_x.shape)

    return mom_std, mom_x

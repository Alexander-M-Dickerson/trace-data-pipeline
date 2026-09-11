"""zoo_engine.py -- the ONE computation behind every S4 (factor-zoo) exhibit.

Tables IA.IX/IA.X/IA.XI and the two inline breakdowns all read the same four
stats frames -- (vw, ew) x (single, wf) -- computed once from the zoo sort CSVs
run_zoo_sorts.py produces:

  data    the two zoo CSVs -- SIGN-CORRECTED series (the '*' is the receipt, and
          the flip is decided per weighting on the sample that was sorted),
          truncated to [2002-08-31, 2024-12-31] at use.
  stats   per factor after dropna: T, date range, NW mean t (floor(T^0.25)
          lags), std, SR; CAPMB alpha/t from HAC OLS on MKTB; IR = alpha /
          resid_std(ddof=2); annualized x12 (mean, alpha) and xsqrt(12)
          (std, SR, IR); printed x100 at 2dp.
  FDR     p = 2*(1 - t.cdf(|t_alpha|, df=n-2)); Benjamini-Hochberg at 5% over
          each frame's own factors (m = 108 per panel); printed rows =
          t_alpha > 1.96 (one-sided ON the sign-corrected series), sorted by
          alpha descending; BH survivors shaded.
  filter  min_factor_start = 2003-12-31 (keep factors whose first obs is on or
          before it) -- bites ZERO factors here, implemented for fidelity.

"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/

import drrlib as D          # noqa: E402
import paths                # noqa: E402

DATE_START = "2002-08-31"
DATE_END = "2024-12-31"
MIN_FACTOR_START = "2003-12-31"
N_FACTORS = 108
SIG_T = 1.96
FDR_Q = 0.05

CSV = {"single": "bond_single_sort_all_p10_h1.csv",
       "wf": "bond_within_firm_all_p2_h1.csv"}

# the 9 signal clusters; 9+5+21+13+16+9+4+13+18 = 108. The grouping follows
# Stage 2's data dictionary, and `run_zoo_sorts.py` asserts the panel's signal
# columns are exactly this set -- a panel with more or fewer stops the run.
CLUSTERS = {
    "Spreads, Yields, Size": [
        "tmat", "age", "ytm", "cs", "md_dur", "convx", "sze", "dcs6", "cs_mu12_1"],
    "Value": ["bbtm", "val_hz", "val_hz_dts", "val_ipr", "val_ipr_dts"],
    "Momentum & Reversal": [
        "mom3_1", "mom6_1", "mom9_1", "mom12_1", "mom12_7", "sysmom3_1",
        "sysmom6_1", "sysmom12_1", "idimom3_1", "idimom6_1", "idimom12_1",
        "imom1", "imom3_1", "imom12_1", "ltr24_3", "ltr30_6", "ltr48_12",
        "iltr24_3", "iltr30_6", "iltr48_12", "str"],
    "Illiquidity": ["pi", "ami", "ami_v", "roll", "ilq", "spd_rel", "spd_abs",
                    "cs_sprd", "ar_sprd", "p_zro", "p_fht", "vov", "lix"],
    "Volatility & Risk": [
        "dvol", "dskew", "dkurt", "rvol", "rsj", "rsk", "rkt", "var_90",
        "var_95", "es_90", "dvol_sys", "dvol_idio", "ivol_mkt", "ivol_bbw",
        "ivol_vp", "iskew"],
    "Market Risk": ["b_mktrf_mkt", "b_mktb_mkt", "b_mktb", "b_mktbx_dcapm",
                    "b_term_dcapm", "b_mktb_dn", "b_mktb_up", "b_termb", "db_mkt"],
    "Credit & Default Betas": ["b_drf", "b_crf", "b_lrf", "b_defb"],
    "Vol. & Liq. Betas": [
        "b_dvix", "b_dvix_va", "b_dvix_vp", "b_dvix_dn", "b_dvix_up", "b_psb",
        "b_psb_m", "b_amd_m", "b_amd", "b_coskew", "b_vix", "b_dvixd", "b_illiq"],
    "Macro & Other Betas": [
        "b_dunc", "b_duncr", "b_duncf", "b_unc", "b_dunc3", "b_dunc6", "b_dcpi",
        "b_cpi_vol6", "b_dcredit", "b_credit", "b_cptlt", "b_rvol", "b_rsj",
        "b_lvl", "b_ysp", "b_epu", "b_epum", "b_eput"],
}
CLUSTER_OF = {sig: c for c, sigs in CLUSTERS.items() for sig in sigs}
SPECS = [("vw", "single"), ("vw", "wf"), ("ew", "single"), ("ew", "wf")]


def zoo_csv(sort: str, root: Path | None = None) -> Path:
    """The zoo sort CSV run_zoo_sorts.py writes, for one sort design."""
    return Path(root or (paths.SORTS / "zoo")) / CSV[sort]


def base_of(factor: str) -> str:
    """'b_amd_m_wf*' -> 'b_amd_m' (strip the flip star and the _wf decoration)."""
    f = factor.rstrip("*")
    return f[:-3] if f.endswith("_wf") else f


def load_wide(sort: str, weighting: str, root: Path | None = None,
              end: str = DATE_END) -> pd.DataFrame:
    """ls-leg returns AS STORED (sign-corrected), wide date x decorated-factor,
    truncated to the zoo window (`end` overrides it).
    All-NaN columns (the other weighting's flip twin) are dropped -- exactly
    the n<12 gate drops."""
    import duckdb

    p = zoo_csv(sort, root).as_posix()
    df = duckdb.sql(
        f"SELECT date, factor, \"return\" AS r FROM read_csv_auto('{p}') "
        f"WHERE leg = 'ls' AND weighting = '{weighting}' "
        f"AND date BETWEEN DATE '{DATE_START}' AND DATE '{end}'").df()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="factor", values="r").sort_index()
    wide.columns.name = None
    return wide.dropna(axis=1, how="all")


def zoo_stats(wide: pd.DataFrame, mktb: pd.Series) -> pd.DataFrame:
    """Per-factor summary statistics, then Benjamini-Hochberg, then the printed rows."""
    from scipy.stats import t as t_dist

    rows = []
    for col in wide.columns:
        s = wide[col].dropna()
        n = len(s)
        if n < 12:
            continue
        mean, t_mu, _ = _nw_mean(s)
        std = float(s.std())
        sr = mean / std if std > 0 else np.nan
        common = s.index.intersection(mktb.index)
        alpha = alpha_t = ir = np.nan
        if len(common) >= 12:
            y = s.loc[common].to_numpy(float)
            x = mktb.loc[common].to_numpy(float)
            alpha, alpha_t, resid_std = _hac_alpha(y, x)
            if resid_std > 0 and not np.isnan(alpha):
                ir = alpha / resid_std
        rows.append({"factor": col, "n": n,
                     "date_start": s.index.min(), "date_end": s.index.max(),
                     "mean": mean * 12, "std": std * np.sqrt(12), "t_stat": t_mu,
                     "sr": sr * np.sqrt(12), "alpha": alpha * 12,
                     "alpha_t": alpha_t, "ir": ir * np.sqrt(12)})
    df = pd.DataFrame(rows)

    # the full-sample min_factor_start filter: keep factors whose first observation is
    # on or before it, so a factor that only exists late cannot enter the count
    df = df[df["date_start"] <= pd.Timestamp(MIN_FACTOR_START)].reset_index(drop=True)

    # p-values, BH over the WHOLE frame, the t>1.96 selection flag
    df["alpha_p"] = [2 * (1 - t_dist.cdf(abs(t), df=n - 2))
                     if pd.notna(t) and n > 2 else np.nan
                     for t, n in zip(df["alpha_t"], df["n"])]
    df["t_mu_p"] = [2 * (1 - t_dist.cdf(abs(t), df=n - 2))
                    if pd.notna(t) and n > 2 else np.nan
                    for t, n in zip(df["t_stat"], df["n"])]
    df["bh_pass"] = _bh_mask(df["alpha_p"])
    df["bh_pass_mu"] = _bh_mask(df["t_mu_p"])
    df["sig_196"] = df["alpha_t"] > SIG_T
    df["sig_196_mu"] = df["t_stat"] > SIG_T
    return df


def _nw_mean(s: pd.Series) -> tuple[float, float, float]:
    import statsmodels.api as sm
    y = s.to_numpy(float)
    T = len(y)
    res = sm.OLS(y, np.ones((T, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(np.floor(T ** 0.25))})
    return float(res.params[0]), float(res.tvalues[0]), float(res.bse[0])


def _hac_alpha(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    import statsmodels.api as sm
    T = len(y)
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit(cov_type="HAC",
                           cov_kwds={"maxlags": int(np.floor(T ** 0.25))})
    return (float(res.params[0]), float(res.tvalues[0]),
            float(np.std(res.resid, ddof=2)))


def _bh_mask(p: pd.Series, q: float = FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg at q over the valid p-values."""
    mask = np.zeros(len(p), dtype=bool)
    valid = p.notna().to_numpy()
    pv = p[p.notna()].to_numpy(float)
    m = len(pv)
    if m == 0:
        return mask
    order = np.argsort(pv)
    passing = pv[order] <= np.arange(1, m + 1) / m * q
    if passing.any():
        k = int(np.max(np.where(passing)[0]))
        sub = np.zeros(m, dtype=bool)
        sub[order[:k + 1]] = True
        mask[np.where(valid)[0]] = sub
    return mask


def build(mktb: pd.Series | None = None, root: Path | None = None) -> dict:
    """{(weighting, sort): stats frame} for all four specifications."""
    m = mktb if mktb is not None else D.load_mktb(
        paths.BBW, start=DATE_START, end=DATE_END)
    out = {}
    for weighting, sort in SPECS:
        wide = load_wide(sort, weighting, root=root)
        df = zoo_stats(wide, m)
        if len(df) != N_FACTORS:
            raise AssertionError(
                f"{weighting}/{sort}: {len(df)} factors after the gates, "
                f"expected {N_FACTORS} -- the BH m would be wrong")
        out[(weighting, sort)] = df
    return out


def printed_view(df: pd.DataFrame) -> pd.DataFrame:
    """The rows a zoo longtable prints: t(alpha) > 1.96, sorted by alpha desc."""
    sub = df[df["sig_196"]].sort_values("alpha", ascending=False, na_position="last")
    return sub.reset_index(drop=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    frames = build()
    for (w, s), df in frames.items():
        pv = printed_view(df)
        print(f"{w}/{s}: {len(df)} factors, sig {int(df['sig_196'].sum())}, "
              f"BH {int(df['bh_pass'].sum())}, sig_mu {int(df['sig_196_mu'].sum())}, "
              f"BH_mu {int(df['bh_pass_mu'].sum())}")
        print("   top:", ", ".join(pv["factor"].head(5)))

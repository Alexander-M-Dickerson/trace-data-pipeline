"""step3_bbw.py -- Bai-Bali-Wen 4-factor construction (MKTB/DRF/CRF/LRF + duration-adjusted + TERM).

Ground truth: the reference implementation's `prep_bbw_data` + `create_bbw_factors`
(2563-3019), called by the runner with rating_col='composite', signal_cols=(var_95, ilq_adj,
str1_adj) / (var_95x, ilq_adj, str1_adjx), n_portf=5x5, and NO fisd_path (the USA filter is
SKIPPED for BBW, unlike the illiq factors). This step is pandas end-to-end -- the double sorts run
through PyBondLab exactly as upstream, on small (2M-row) monthly panels; the heavy lifting was
already done by steps 1-2.

Inputs: step-1 blocks (all_returns, end_returns, end_signals, adj_signals) + step-2
illiq_signals_adj + the cached quote panel (lib/quote) + FF5 rf (lib/ff5).
Output: blocks/<mode>/bbw_factors.parquet (date + MKTB DRF CRF LRF MKTBx DRFx CRFx LRFx TERM).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _stage2_settings as cfg
from lib import ff5, quote, treasury, var_es

BBW_SIGNALS = ("var_95", "ilq_adj", "str1_adj")
BBW_SIGNALS_X = ("var_95x", "ilq_adj", "str1_adjx")
FACTOR_NAME_MAP = {"var_95": "DRF", "ilq_adj": "LRF"}
FACTOR_NAME_MAP_X = {"var_95x": "DRFx", "ilq_adj": "LRFx"}


def _prep_bbw_panel(blocks_dir: Path) -> pd.DataFrame:
    """Upstream prep_bbw_data: quote+all_returns VaR, end_returns base, signal merges. No USA filter."""
    # -- 1. quote (pre-2002-07) + all_returns; rolling VaR on ret_vw and ret_vwx ------------------
    q = quote.load_quote()
    q.columns = q.columns.str.lower()
    q = q.rename(columns={"cusip_id": "cusip"})[["cusip", "date", "ret_vw", "tret"]].copy()
    q["date"] = pd.to_datetime(q["date"])
    q = q[q["date"] < "2002-07-01"].copy()

    all_ret = pd.read_parquet(blocks_dir / "all_returns.parquet",
                              columns=["cusip", "date", "ret_vw", "tret"])
    all_ret["cusip"] = all_ret["cusip"].astype(str)
    all_ret["date"] = pd.to_datetime(all_ret["date"])

    combined = pd.concat([q, all_ret], ignore_index=True)
    combined = combined.drop_duplicates(subset=["cusip", "date"], keep="last")
    combined = combined.sort_values(["cusip", "date"]).reset_index(drop=True)
    combined["ret_vwx"] = combined["ret_vw"] - combined["tret"]
    combined = combined.rename(columns={"cusip": "cusip_id"})

    var_df = var_es.compute_rolling_var_es(combined, id_col="cusip_id", ret_col="ret_vw")
    var_df = var_df[["cusip_id", "date", "var_95"]].copy()
    var_x = var_es.compute_rolling_var_es(combined, id_col="cusip_id", ret_col="ret_vwx")
    var_x = var_x[["cusip_id", "date", "var_95"]].rename(columns={"var_95": "var_95x"})
    var_df = var_df.merge(var_x, on=["cusip_id", "date"], how="left")
    var_df = var_df.rename(columns={"cusip_id": "cusip"})
    del combined, q, all_ret, var_x

    # -- 3-5. base panel from end_returns + end_signals + adj_signals ------------------------------
    out = pd.read_parquet(blocks_dir / "end_returns.parquet", columns=["cusip", "date", "ret_vw", "tret"])
    out["cusip"] = out["cusip"].astype(str)
    out["date"] = pd.to_datetime(out["date"])
    out["ret_vwx"] = out["ret_vw"] - out["tret"]

    sig = pd.read_parquet(blocks_dir / "end_signals.parquet",
                          columns=["cusip", "date", "mcap_s", "mcap_e", "sp_rat", "mdy_rat",
                                   "tmat"])
    sig["cusip"] = sig["cusip"].astype(str)
    sig["date"] = pd.to_datetime(sig["date"])
    out = out.merge(sig, on=["cusip", "date"], how="outer")

    adj = pd.read_parquet(blocks_dir / "adj_signals.parquet",
                          columns=["cusip", "date", "str1_adj", "str2_adj"])
    adj["cusip"] = adj["cusip"].astype(str)
    adj["date"] = pd.to_datetime(adj["date"])
    out = out.merge(adj, on=["cusip", "date"], how="outer")

    out["str1_adjx"] = out["str1_adj"] - out["tret"]
    out["str2_adjx"] = out["str2_adj"] - out["tret"]
    out.drop(columns=["tret"], inplace=True)

    # -- 6. illiq_signals_adj (ilq_adj) -------------------------------------------------------------
    ilq = pd.read_parquet(blocks_dir / "illiq_signals_adj.parquet",
                          columns=["cusip_id", "date", "ilq_adj"]).rename(columns={"cusip_id": "cusip"})
    ilq["cusip"] = ilq["cusip"].astype(str)
    ilq["date"] = pd.to_datetime(ilq["date"])
    out = out.merge(ilq, on=["cusip", "date"], how="outer")

    # -- 7. VaR (left) + 9. keep return rows or 2002-07-31 -----------------------------------------
    out = out.merge(var_df, on=["cusip", "date"], how="left")
    keep_mask = out["ret_vw"].notna() | (out["date"] == "2002-07-31")
    out = out[keep_mask].sort_values(["cusip", "date"]).reset_index(drop=True)
    return out


def _double_sort_factors(df: pd.DataFrame, signals: tuple, ret_field: str,
                         keep_map: dict, str_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Upstream create_bbw_factors steps 7/7b: PyBondLab 5x5 double sorts per signal.
    Returns (core factors renamed via keep_map, CRF leg = mean over signal-bucket rating spreads)."""
    import PyBondLab as pbl

    factors = pd.DataFrame()
    factors_crf = pd.DataFrame()
    n1 = cfg.N_PORTF_1
    n2 = cfg.N_PORTF_2
    for sig in signals:
        data = df[["date", "ID", "VW", "RATING_NUM", sig, ret_field]].copy()
        if ret_field != "ret":
            keep_mask = data[ret_field].notna() | (data["date"] == "2002-07-31")
            data = data[keep_mask].rename(columns={ret_field: "ret"}).reset_index(drop=True)
        ds = pbl.DoubleSort(holding_period=1, sort_var="RATING_NUM", sort_var2=sig,
                            num_portfolios=n1, num_portfolios2=n2,
                            how="unconditional", auto_match_signals=True)
        res = pbl.StrategyFormation(data, strategy=ds, rating=None,
                                    dynamic_weights=True, turnover=False).fit()
        sorts = res.get_ptf()[1]

        top, bot = f"{sig.upper()}{n2}", f"{sig.upper()}1"
        subs = {f"RATING_NUM{i}_DIFF": sorts[f"RATING_NUM{i}_{top}"] - sorts[f"RATING_NUM{i}_{bot}"]
                for i in range(1, n1 + 1)}
        factors = pd.concat([factors, pd.DataFrame(subs).mean(axis=1).to_frame(name=sig)], axis=1)

        subs = {f"RATING_NUM{j}_DIFF":
                sorts[f"RATING_NUM{n1}_{sig.upper()}{j}"] - sorts[f"RATING_NUM1_{sig.upper()}{j}"]
                for j in range(1, n2 + 1)}
        factors_crf = pd.concat(
            [factors_crf, pd.DataFrame(subs).mean(axis=1).to_frame(name=f"{sig}_crf")], axis=1)
        del data, res, sorts

    if str_col in factors.columns:
        factors[str_col] = -factors[str_col]        # short-term reversal sign flip
    crf = factors_crf.mean(axis=1)
    core = factors[[c for c in keep_map if c in factors.columns]].rename(columns=keep_map)
    return core, crf


def build(con=None, mode: str | None = None, limit_cusips: int | None = None) -> dict[str, Path]:
    """Build the bbw_factors block. `con`/`limit_cusips` unused (pandas step; kept for API parity)."""
    mode = mode or cfg.INPUT_MODE
    t0 = time.time()
    blocks_dir = cfg.BLOCKS_DIR / mode
    out_dir = blocks_dir

    df = _prep_bbw_panel(blocks_dir)

    # -- create_bbw_factors steps 2-5: dedup, unrated drop, composite rating, renames --------------
    n0 = len(df)
    df = df.drop_duplicates(subset=["cusip", "date"], keep="first").reset_index(drop=True)
    df = df.sort_values(["cusip", "date"]).reset_index(drop=True)
    df = df[~(df["sp_rat"].isna() & df["mdy_rat"].isna())].reset_index(drop=True)
    df["RATING_NUM"] = np.nanmean(np.column_stack([df["sp_rat"], df["mdy_rat"]]), axis=1)
    df = df.rename(columns={"cusip": "ID", "ret_vw": "ret", "ret_vwx": "ret_x", "mcap_e": "VW"})
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)

    rf_df = ff5.load_ff5()[["date", "rf"]].set_index("date")["rf"]

    core_std, crf_std = _double_sort_factors(df, BBW_SIGNALS, "ret", FACTOR_NAME_MAP, "str1_adj")
    core_x, crf_x = _double_sort_factors(df, BBW_SIGNALS_X, "ret_x", FACTOR_NAME_MAP_X, "str1_adjx")
    CRF_std = crf_std.to_frame(name="CRF")
    CRF_x = crf_x.to_frame(name="CRFx")

    # -- MKTB / MKTBx / TERM (VW by mcap_s over non-null return rows) ------------------------------
    mkt = df[["date", "ID", "mcap_s", "ret", "ret_x"]].dropna(subset=["ret", "mcap_s"]).copy()
    mkt["vw_w"] = mkt["mcap_s"] / mkt.groupby("date", observed=True)["mcap_s"].transform("sum")
    mkt["ret_w"] = mkt["ret"] * mkt["vw_w"]
    MKTB_raw = mkt.groupby("date", observed=True)["ret_w"].sum().to_frame(name="MKTB_raw")

    mkt_x = df[["date", "ID", "mcap_s", "ret_x"]].dropna(subset=["ret_x", "mcap_s"]).copy()
    mkt_x["vw_w"] = mkt_x["mcap_s"] / mkt_x.groupby("date", observed=True)["mcap_s"].transform("sum")
    mkt_x["ret_w"] = mkt_x["ret_x"] * mkt_x["vw_w"]
    MKTBx_df = mkt_x.groupby("date", observed=True)["ret_w"].sum().to_frame(name="MKTBx")

    rf_df.index = pd.to_datetime(rf_df.index) + pd.offsets.MonthEnd(0)
    MKT_all = MKTB_raw.join(rf_df.rename("RF"), how="inner")
    MKT_all["MKTB"] = MKT_all["MKTB_raw"] - MKT_all["RF"]
    MKT_all = MKT_all.join(MKTBx_df, how="inner")
    MKT_all["TERM"] = MKT_all["MKTB_raw"] - MKT_all["MKTBx"]

    # -- DEFB / TERMB: the default and term premia (Fama-French 1993; Gebhardt, Hvidkjaer &
    #    Swaminathan 2005) ---------------------------------------------------------------------
    #
    #      DEFB  = long-term CORPORATE total return  -  long-term GOVERNMENT total return
    #      TERMB = long-term government total return -  risk-free rate
    #
    # The corporate leg is the value-weighted return of panel bonds with at least
    # DEF_CORP_MIN_MATURITY years to maturity; the government leg is the CRSP key-rate
    # Treasury return at DEF_GOVT_TENOR. Both legs are TOTAL returns, so the difference is
    # the credit premium a long-maturity bond earns over a matched government bond.
    lt = (df.loc[df["tmat"] >= cfg.DEF_CORP_MIN_MATURITY, ["date", "ID", "mcap_s", "ret"]]
            .dropna(subset=["ret", "mcap_s"]).copy())
    lt["vw_w"] = lt["mcap_s"] / lt.groupby("date", observed=True)["mcap_s"].transform("sum")
    lt["ret_w"] = lt["ret"] * lt["vw_w"]
    LTCORP = lt.groupby("date", observed=True)["ret_w"].sum().to_frame(name="LTCORP")

    tw = treasury.load_tret_wide(mode=mode)
    if cfg.DEF_GOVT_TENOR not in tw.columns:
        raise KeyError(f"Treasury tenor {cfg.DEF_GOVT_TENOR} not in {list(tw.columns)}")
    LTGOVT = tw[cfg.DEF_GOVT_TENOR].rename("LTGOVT").to_frame()
    LTGOVT.index = pd.to_datetime(LTGOVT.index) + pd.offsets.MonthEnd(0)

    MKT_all = MKT_all.join(LTCORP, how="left").join(LTGOVT, how="left")
    MKT_all["DEFB"] = MKT_all["LTCORP"] - MKT_all["LTGOVT"]
    MKT_all["TERMB"] = MKT_all["LTGOVT"] - MKT_all["RF"]

    bbw = (core_std.join(CRF_std, how="inner").join(core_x, how="inner")
           .join(CRF_x, how="inner").join(MKT_all[["MKTB", "MKTBx", "TERM", "DEFB", "TERMB"]], how="inner")
           .reset_index().rename(columns={"index": "date"}).set_index("date")
           .loc[:, ["MKTB", "DRF", "CRF", "LRF", "MKTBx", "DRFx", "CRFx", "LRFx", "TERM",
                   "DEFB", "TERMB"]]
           .sort_index())

    out_path = out_dir / "bbw_factors.parquet"
    bbw.to_parquet(out_path, index=True)
    (out_dir / "step3_meta.json").write_text(json.dumps(
        {"wall_s": round(time.time() - t0, 2), "mode": mode, "rows": len(bbw),
         "panel_rows": int(n0)}, indent=1))
    return {"bbw_factors": out_path}

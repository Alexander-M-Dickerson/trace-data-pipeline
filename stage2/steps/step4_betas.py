"""step4_betas.py -- upstream step 4: the merged factor matrix, combined returns, rolling 36-month
betas (min 12 obs) for ~38 factor models x 2 return types, and systematic/idiosyncratic momentum.

Ground truth: `_debug_stage2.py::step4_betas` (factor merging incl. the extended-BBW pre-2002-08-31
backfill and the derived asymmetric factors) + `process_bond_data.{concat_data, compute_all_betas,
compute_sys_momentum}` (verbatim in lib/betas). Inputs: the pinned factors.parquet (G4/A10), our
validated bbw_factors (G3) and illiq_factors (G2), all_returns (G1), the cached quote panel, and the
PUBLISHED extended-BBW series (lib/extended_factors -- the W2 public seam).

Outputs under blocks/<mode>/: betas_x.parquet (the G5 golden target, 2,282,733 x 53),
betas_std.parquet (feeds the final panel at G7), factors_merged.parquet (feeds steps 5-6 + rfret).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _stage2_settings as cfg
from lib import betas as betalib
from lib import extended_factors, quote
from lib.phase_timer import PhaseTimer

# Pre-2002-08-31 backfill: BBW factor column -> its extended-series name. Upstream fills these from
# the lowercase LHM/ICE columns inside factors.parquet (_debug_stage2.py:541-556); we fill them from
# the PUBLISHED extended series (== those columns at max|d|=0, assumptions A18), which removes the
# private-pin dependency for the BBW piece (HANDOFF W2). Post-cutoff stays our pure-TRACE G3 build.
ICE_MAP = {"MKTB": "mktb", "DRF": "drf", "CRF": "crf",
           "MKTBx": "mktbx", "DRFx": "drfx", "CRFx": "crfx", "TERM": "term",
           # DEFB/TERMB are spliced the same way IF the published extended series carries
           # them. It does not yet -- the pre-2002 long-term-corporate return needed for a
           # default premium lives in the private pre-TRACE panel and has never been
           # published. Until it is, these two fall through the `if ... in factors.columns`
           # guard below and b_defb/b_termb are TRACE-era only. Adding the column upstream
           # is all that is needed; nothing here changes.
           "DEFB": "defb", "TERMB": "termb"}
ICE_CUTOFF = pd.Timestamp("2002-08-31")


def build_factor_matrix(blocks_dir: Path) -> pd.DataFrame:
    """Runner step 4.1-4.3: factors panel + BBW (outer) with the extended-BBW pre-2002-08 splice +
    illiq (outer), lowercase, dedup dates, asymmetric mktb/dvix columns. Grain: one row per month."""
    factors = pd.read_parquet(blocks_dir / "factors.parquet")
    factors["date"] = pd.to_datetime(factors["date"])
    # The pinned panel's own copies of the extended-BBW columns are dropped UNREAD -- the backfill
    # below comes from the published series instead (W2).
    factors = factors.drop(columns=[c for c in ICE_MAP.values() if c in factors.columns])

    bbw = pd.read_parquet(blocks_dir / "bbw_factors.parquet")
    if "date" not in bbw.columns:
        bbw = bbw.reset_index()
    bbw["date"] = pd.to_datetime(bbw["date"])
    factors = factors.merge(bbw, on="date", how="outer")

    ext = extended_factors.load_extended_bbw().rename(columns=ICE_MAP)
    missing = set(ext["date"]) - set(factors["date"])
    assert not missing, f"extended-BBW dates absent from the factor panel: {sorted(missing)[:5]}"
    factors = factors.merge(ext, on="date", how="left")

    mask = factors["date"] < ICE_CUTOFF
    for bbw_col, ice_col in ICE_MAP.items():
        if bbw_col in factors.columns and ice_col in factors.columns:
            factors.loc[mask, bbw_col] = factors.loc[mask, ice_col]
    factors = factors.drop(columns=[c for c in ICE_MAP.values() if c in factors.columns])

    illiq = pd.read_parquet(blocks_dir / "illiq_factors.parquet")
    if "date" not in illiq.columns:
        illiq = illiq.reset_index()
    illiq["date"] = pd.to_datetime(illiq["date"])
    factors = factors.merge(illiq, on="date", how="outer")

    factors = factors.sort_values("date").reset_index(drop=True)
    factors.columns = factors.columns.str.lower()
    factors = factors.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    mktb = factors["mktb"].to_numpy(dtype=float, na_value=np.nan)
    factors["mktb_down"] = np.where(mktb < 0, mktb, 0)
    factors["mktb_up"] = np.where(mktb > 0, mktb, 0)
    factors["mktb_sq"] = mktb ** 2
    dvix = factors["dvix"].to_numpy(dtype=float, na_value=np.nan)
    factors["dvix_down"] = np.where(np.isnan(dvix), np.nan, np.where(dvix < 0, dvix, 0))
    factors["dvix_up"] = np.where(np.isnan(dvix), np.nan, np.where(dvix > 0, dvix, 0))
    return factors


def build_combined_returns(blocks_dir: Path) -> pd.DataFrame:
    """concat_data: quote panel (pre-2002-07) + all_returns, end rows win; + ret_vwx."""
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
    combined = combined[["cusip", "date", "ret_vw", "tret"]]
    combined["ret_vwx"] = combined["ret_vw"] - combined["tret"]
    return combined


def build(con=None, mode: str | None = None, limit_cusips: int | None = None) -> dict[str, Path]:
    """Build betas_x / betas_std / factors_merged blocks (pandas+numba step; `con` unused)."""
    mode = mode or cfg.INPUT_MODE
    t0 = time.time()
    pt = PhaseTimer()
    blocks_dir = cfg.BLOCKS_DIR / mode

    with pt("factor_matrix"):
        factors = build_factor_matrix(blocks_dir)
    with pt("combined_returns"):
        combined = build_combined_returns(blocks_dir)

    with pt("compute_all_betas"):
        betas_std, betas_x = betalib.compute_all_betas(
            combined_returns=combined, factors=factors,
            window=cfg.BETA_WINDOW, min_obs=cfg.BETA_MIN_OBS, verbose=True)
    with pt("sys_momentum"):
        mom_std, mom_x = betalib.compute_sys_momentum(
            combined_returns=combined, factors=factors, factor_cols=["mktb"],
            window=cfg.BETA_WINDOW, min_obs=cfg.BETA_MIN_OBS, verbose=True)
    with pt("merge_mom"):
        betas_std = betas_std.merge(mom_std, on=["cusip", "date"], how="outer")
        betas_x = betas_x.merge(mom_x, on=["cusip", "date"], how="outer")

    out: dict[str, Path] = {}
    with pt("write_blocks"):
        for name, frame in (("betas_x", betas_x), ("betas_std", betas_std),
                            ("factors_merged", factors)):
            p = blocks_dir / f"{name}.parquet"
            frame.to_parquet(p, index=False)
            out[name] = p
    (blocks_dir / "step4_meta.json").write_text(json.dumps(
        {"wall_s": round(time.time() - t0, 2), "mode": mode, "phases": pt.phases,
         "rows": {n: len(f) for n, f in
                  (("betas_x", betas_x), ("betas_std", betas_std), ("factors_merged", factors))}},
        indent=1))
    return out

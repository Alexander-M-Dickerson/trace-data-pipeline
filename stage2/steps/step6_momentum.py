"""step6_momentum.py -- upstream step 6: momentum / long-term reversal / industry momentum + rolling
VaR/ES, on the 1997-extended return panel.

Ground truth: the reference implementation's return/risk step -- FISD sic_code merge, ret_vwx = ret_vw - tret,
build_mom_ltr_and_industry (verbatim in lib/momentum), compute_rolling_var_es on ret_vw (-> mom_ret)
and ret_vwx (-> mom_retx), keeping var_90/es_90/var_95.

Outputs under blocks/<mode>/: mom_retx (the momentum validation target, 2,317,538 x 19) + mom_ret (G7 input).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import _stage2_settings as cfg
from lib import momentum as momlib
from lib import var_es

VAR_COLS = ["cusip", "date", "var_90", "es_90", "var_95"]


def build(con=None, mode: str | None = None, limit_cusips: int | None = None,
          tret_col: str = "tret") -> dict[str, Path]:
    mode = mode or cfg.INPUT_MODE
    t0 = time.time()
    blocks_dir = cfg.BLOCKS_DIR / mode

    are = pd.read_parquet(blocks_dir / "all_returns_ext.parquet")
    are["cusip"] = are["cusip"].astype(str)
    are["date"] = pd.to_datetime(are["date"])

    fisd = pd.read_parquet(cfg.AUX["fisd"], columns=["complete_cusip", "sic_code"])
    fisd = fisd.drop_duplicates(subset=["complete_cusip"]).rename(columns={"complete_cusip": "cusip"})
    are = are.merge(fisd[["cusip", "sic_code"]], on="cusip", how="left")
    are["ret_vwx"] = are["ret_vw"] - are[tret_col]

    mom_ret, mom_retx = momlib.build_mom_ltr_and_industry(are)

    var_df = var_es.compute_rolling_var_es(are, id_col="cusip", ret_col="ret_vw")[VAR_COLS]
    var_df_adj = var_es.compute_rolling_var_es(are, id_col="cusip", ret_col="ret_vwx")[VAR_COLS]
    mom_ret = mom_ret.merge(var_df, on=["cusip", "date"], how="left")
    mom_retx = mom_retx.merge(var_df_adj, on=["cusip", "date"], how="left")

    out: dict[str, Path] = {}
    for name, frame in (("mom_retx", mom_retx), ("mom_ret", mom_ret)):
        p = blocks_dir / f"{name}.parquet"
        frame.to_parquet(p, index=False)
        out[name] = p
    (blocks_dir / "step6_meta.json").write_text(json.dumps(
        {"wall_s": round(time.time() - t0, 2), "mode": mode,
         "rows": {"mom_retx": len(mom_retx), "mom_ret": len(mom_ret)}}, indent=1))
    return out

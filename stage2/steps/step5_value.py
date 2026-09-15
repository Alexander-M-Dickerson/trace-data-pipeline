"""step5_value.py -- upstream step 5: value signals (credit-spread cross-sectional residuals ->
val_hz / val_ipr families) + d-spread momentum-of-spread signals.

Ground truth: the reference implementation's value step -- prep_value_signal_inputs (quote extension to
1997), FISD call-dummy merge, make_value_signals, build_d_spreads(lags=(6,12), bandwidth=1,
mu_window=12) with the runner's column drops, left-merged into the signal frames. Machinery is
verbatim in lib/value.py.

Outputs under blocks/<mode>/: value_signals_std, value_signals_adj, all_returns_ext (for step 6),
end_signals_ext + adj_signals_ext (for the final wrangle).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _stage2_settings as cfg
from lib import quote
from lib import value as valuelib

# runner drops these build_d_spreads outputs before merging (step5, the reference implementation)
DROP_STD = ["dbbtm6", "dcs12", "dbbtm12", "bbtm_mu12_1"]
DROP_ADJ = ["dbbtm6_adj", "dcs12_adj", "dbbtm12_adj", "bbtm_mu12_1_adj"]


def _prep_inputs(blocks_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """prep_value_signal_inputs: extend end/adj signals + all_returns back to 1997 with quote data."""
    q = quote.load_quote()
    q.columns = q.columns.str.lower()
    q = q.rename(columns={"cusip_id": "cusip"})
    q["date"] = pd.to_datetime(q["date"])
    q = q[q["date"] < "2002-07-01"].copy()

    def _read(name: str, cols=None) -> pd.DataFrame:
        df = pd.read_parquet(blocks_dir / f"{name}.parquet", columns=cols)
        df["cusip"] = df["cusip"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # The benchmark columns ride along so step 6 can roll momentum and VaR on any of them.
    # They change nothing here: make_value_signals reads ret_vw and tret by name, and value
    # signals are deliberately NOT duration-adjusted -- both panels share value_signals_std.
    _RET = ["cusip", "date", "ret_vw", "tret", "tret_bns", "tret_cls"]
    all_ret = _read("all_returns", _RET)
    all_returns_ext = pd.concat([q[_RET], all_ret], ignore_index=True)
    all_returns_ext = all_returns_ext.drop_duplicates(subset=["cusip", "date"], keep="last")
    all_returns_ext = all_returns_ext.sort_values(["cusip", "date"]).reset_index(drop=True)

    end_sub = _read("end_signals")
    quote_end = pd.DataFrame({"cusip": q["cusip"].values, "date": q["date"].values,
                              "cs": q["cs"].values, "bbtm": q["bbtm"].values})
    end_ext = pd.concat([quote_end, end_sub], ignore_index=True)
    end_ext = end_ext.drop_duplicates(subset=["cusip", "date"], keep="last")
    end_ext = end_ext.sort_values(["cusip", "date"]).reset_index(drop=True)

    adj_sub = _read("adj_signals")
    quote_adj = pd.DataFrame({"cusip": q["cusip"].values, "date": q["date"].values,
                              "cs_adj": q["cs"].values, "bbtm_adj": q["bbtm"].values})
    adj_ext = pd.concat([quote_adj, adj_sub], ignore_index=True)
    adj_ext = adj_ext.drop_duplicates(subset=["cusip", "date"], keep="last")
    adj_ext = adj_ext.sort_values(["cusip", "date"]).reset_index(drop=True)
    return end_ext, adj_ext, all_returns_ext


def build(con=None, mode: str | None = None, limit_cusips: int | None = None) -> dict[str, Path]:
    mode = mode or cfg.INPUT_MODE
    t0 = time.time()
    blocks_dir = cfg.BLOCKS_DIR / mode

    end_ext, adj_ext, all_returns_ext = _prep_inputs(blocks_dir)

    # FISD call dummy onto end_signals_ext (make_value_signals merges it to adj internally)
    fisd = pd.read_parquet(cfg.AUX["fisd"], columns=["complete_cusip", "issue_id"])
    fisd_call = pd.read_parquet(cfg.AUX["call"])
    fisd = fisd.merge(fisd_call, on="issue_id", how="left")
    fisd["callable"] = np.where(fisd["callable"].isnull(), 0, fisd["callable"])
    fisd = fisd.rename(columns={"complete_cusip": "cusip", "callable": "call"}).drop(columns=["issue_id"])
    fisd["call"] = fisd["call"].astype("int8")
    end_ext = end_ext.merge(fisd, on="cusip", how="left")
    end_ext["call"] = end_ext["call"].fillna(0).astype("int8")

    signals_std, signals_adj = valuelib.make_value_signals(
        end_signals=end_ext, adj_signals=adj_ext, all_returns=all_returns_ext, verbose=True)

    d_std, d_adj = valuelib.build_d_spreads(end_ext, adj_ext, lags=cfg.DSPREAD_LAGS,
                                            bandwidth=cfg.DSPREAD_BANDWIDTH,
                                            mu_window=cfg.DSPREAD_MU_WINDOW)
    d_std = d_std.drop(columns=DROP_STD)
    d_adj = d_adj.drop(columns=DROP_ADJ)
    signals_std = signals_std.merge(d_std, on=["cusip", "date"], how="left")
    signals_adj = signals_adj.merge(d_adj, on=["cusip", "date"], how="left")

    out: dict[str, Path] = {}
    for name, frame in (("value_signals_std", signals_std), ("value_signals_adj", signals_adj),
                        ("all_returns_ext", all_returns_ext),
                        ("end_signals_ext", end_ext), ("adj_signals_ext", adj_ext)):
        p = blocks_dir / f"{name}.parquet"
        frame.to_parquet(p, index=False)
        out[name] = p
    (blocks_dir / "step5_meta.json").write_text(json.dumps(
        {"wall_s": round(time.time() - t0, 2), "mode": mode,
         "rows": {n: len(f) for n, f in (("value_signals_std", signals_std),
                                         ("value_signals_adj", signals_adj),
                                         ("all_returns_ext", all_returns_ext))}}, indent=1))
    return out

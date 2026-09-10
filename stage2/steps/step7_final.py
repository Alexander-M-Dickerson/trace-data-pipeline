"""step7_final.py -- upstream step 7 (step_final_wrangle): the final left-join merge chain that
produces the 140-column monthly panel.

Ground truth: `_debug_stage2.py::step_final_wrangle` line-for-line -- wrangle_returns (+ OSBAP
linker), wrangle_signals, build_main_panel, str = ret_vw, runner column drops, str1_adj -> str_adj,
swap_adj_signals (which also writes the mmn_price_based_signals block), sig_dt/sig_gap merge, rfret
from the pinned factor panel, FISD 144a/country/call, reorder_panel_cols. Machinery is verbatim in
lib/wrangle.py.

Outputs: blocks/<mode>/{returns_alt_final, mmn_price_based_signals}.parquet and
panel/main_panel_<mode>.parquet -- the deliverable, validated against main_panel_20251126.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _stage2_settings as cfg
from lib import wrangle
from lib.phase_timer import PhaseTimer

RUNNER_DROP_COLS = ["val_hz_wi", "val_ipr_wi", "trn"]   # deprecated/redundant (runner lines 1174+)


def _read(blocks_dir: Path, name: str, cusip_col: str = "cusip") -> pd.DataFrame:
    df = pd.read_parquet(blocks_dir / f"{name}.parquet")
    if cusip_col in df.columns:
        df[cusip_col] = df[cusip_col].astype(str)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def build(con=None, mode: str | None = None, limit_cusips: int | None = None) -> dict[str, Path]:
    mode = mode or cfg.INPUT_MODE
    t0 = time.time()
    pt = PhaseTimer()
    blocks_dir = cfg.BLOCKS_DIR / mode
    cfg.PANEL_DIR.mkdir(parents=True, exist_ok=True)

    with pt("read_blocks"):
        end_returns = _read(blocks_dir, "end_returns")
        bgn_returns = _read(blocks_dir, "bgn_returns")
        end_signals = _read(blocks_dir, "end_signals")
        adj_signals = _read(blocks_dir, "adj_signals")

    # Firm identifiers. Stage 1 attaches permno/permco/gvkey at BOND level using dated
    # windows, and step 1 carries them through, so by default we use those directly.
    #
    # The alternative is the historical path: re-derive them here by merging a separate
    # issuer-level linker keyed on the first six CUSIP characters and forward-filled by
    # month. That is strictly worse -- issuer-level keys cannot distinguish two issuers
    # sharing a CUSIP6, and a forward fill back-casts a link across a period where it did
    # not hold. It stays reachable (set STAGE2_LINKER_FILE) only to reproduce a panel
    # published before the change.
    use_linker_file = cfg.AUX.get("linker") is not None
    with pt("wrangle_returns"):
        returns_main, returns_alt, signals_out = wrangle.wrangle_returns(
            end_returns=end_returns, bgn_returns=bgn_returns, end_signals=end_signals,
            linker_url="local" if use_linker_file else None,
            linker_zipkey=(Path(cfg.AUX["linker"]).name if use_linker_file else None),
            local_dir=str(cfg.DATA_DIR), verbose=True)

    if not use_linker_file:
        with pt("attach_stage1_firm_ids"):
            firm_ids = _read(blocks_dir, "firm_ids")
            before = len(returns_main)
            returns_main["issuer_cusip"] = returns_main["cusip"].astype(str).str[:6]
            returns_main = returns_main.merge(firm_ids, on=["cusip", "date"], how="left",
                                              validate="1:1")
            assert len(returns_main) == before, (
                f"firm_ids merge changed the row count ({before:,} -> {len(returns_main):,}); "
                f"it must be one row per (cusip, date)")
            matched = int(returns_main["permno"].notna().sum())
            print(f"  Stage 1 firm ids: {matched:,} of {before:,} rows "
                  f"({100 * matched / max(before, 1):.1f}%) carry a permno")
            # wrangle_returns normalises dtypes internally, before this merge, so these
            # columns would otherwise escape it. Match the shape the linker path produced
            # so the panel's schema is unchanged and only the VALUES differ.
            returns_main["issuer_cusip"] = returns_main["issuer_cusip"].astype("category")
            for c in ("permno", "permco", "gvkey"):
                returns_main[c] = returns_main[c].astype("float64")

    with pt("wrangle_signals"):
        value_std = _read(blocks_dir, "value_signals_std")
        value_adj = _read(blocks_dir, "value_signals_adj")
        spreads_value, spreads_value_adj = wrangle.wrangle_signals(
            signals_std=signals_out, adj_signals=adj_signals,
            value_signals_std=value_std, value_signals_adj=value_adj, verbose=True)

    with pt("build_main_panel"):
        illiq_signals = _read(blocks_dir, "illiq_signals", cusip_col="cusip_id")
        illiq_signals_adj = _read(blocks_dir, "illiq_signals_adj", cusip_col="cusip_id")
        betas_std = _read(blocks_dir, "betas_std")
        mom_ret = _read(blocks_dir, "mom_ret")

        main_panel, main_panel_adj = wrangle.build_main_panel(
            returns_main=returns_main, spreads_value=spreads_value,
            spreads_value_adj=spreads_value_adj, illiq_signals=illiq_signals,
            illiq_signals_adj=illiq_signals_adj, betas_std=betas_std, mom_ret=mom_ret, verbose=True)

    # runner: str = prior-month return signal; deprecated column drops; str1_adj -> str_adj
    main_panel["str"] = main_panel["ret_vw"]
    for col in RUNNER_DROP_COLS:
        if col in main_panel.columns:
            main_panel.drop(columns=[col], inplace=True)
        if f"{col}_adj" in main_panel_adj.columns:
            main_panel_adj.drop(columns=[f"{col}_adj"], inplace=True)
    if "str2_adj" in main_panel_adj.columns:
        main_panel_adj.drop(columns=["str2_adj"], inplace=True)
    if "str1_adj" in main_panel_adj.columns:
        main_panel_adj.rename(columns={"str1_adj": "str_adj"}, inplace=True)

    with pt("swap_adj_signals"):
        if cfg.SWAP_ADJ_SIGNALS:
            main_panel = wrangle.swap_adj_signals(
                main_panel=main_panel, main_panel_adj=main_panel_adj,
                output_dir=blocks_dir, date_stamp=cfg.DATE_STAMP, compress=True, verbose=True)

    main_panel = main_panel.merge(adj_signals[["cusip", "date", "sig_dt", "sig_gap"]],
                                  on=["cusip", "date"], how="left")

    factors = pd.read_parquet(blocks_dir / "factors.parquet", columns=["date", "rf"])
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.rename(columns={"rf": "rfret"})
    main_panel = main_panel.merge(factors, on="date", how="left")
    main_panel_adj = main_panel_adj.merge(factors, on="date", how="left")

    fisd = pd.read_parquet(cfg.AUX["fisd"],
                           columns=["complete_cusip", "issue_id", "rule_144a", "country_domicile"])
    fisd = fisd.rename(columns={"complete_cusip": "cusip", "rule_144a": "144a",
                                "country_domicile": "country"})
    fisd_call = pd.read_parquet(cfg.AUX["call"])
    fisd = fisd.merge(fisd_call, on="issue_id", how="left")
    fisd["callable"] = np.where(fisd["callable"].isnull(), 0, fisd["callable"])
    fisd = fisd.rename(columns={"callable": "call"}).drop(columns=["issue_id"])
    fisd["144a"] = (fisd["144a"] == "Y").astype("int8")
    fisd["country"] = fisd["country"].astype("category")
    fisd["call"] = fisd["call"].astype("int8")
    main_panel = main_panel.merge(fisd, on="cusip", how="left")
    main_panel_adj = main_panel_adj.merge(fisd, on="cusip", how="left")

    with pt("reorder"):
        main_panel = wrangle.reorder_panel_cols(main_panel, verbose=True)
        sig_gap_idx = list(main_panel.columns).index("sig_gap")
        signal_order = list(main_panel.columns[sig_gap_idx + 1:])
        main_panel_adj = wrangle.reorder_panel_cols(main_panel_adj, signal_order=signal_order,
                                                    verbose=True)
        main_panel["cusip"] = main_panel["cusip"].astype("category")

    panel_path = cfg.PANEL_DIR / f"main_panel_{mode}.parquet"
    with pt("save_panel"):
        wrangle.save_parquet(main_panel, panel_path, compress=True)
    with pt("save_alt"):
        alt_path = blocks_dir / "returns_alt_final.parquet"
        wrangle.save_parquet(returns_alt, alt_path, compress=True)
    mmn_path = blocks_dir / f"mmn_price_based_signals_{cfg.DATE_STAMP}.parquet"

    (blocks_dir / "step7_meta.json").write_text(json.dumps(
        {"wall_s": round(time.time() - t0, 2), "mode": mode, "phases": pt.phases,
         "rows": len(main_panel), "cols": len(main_panel.columns)}, indent=1))
    return {"main_panel": panel_path, "returns_alt_final": alt_path,
            "mmn_price_based_signals": mmn_path}

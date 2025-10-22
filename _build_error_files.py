# -*- coding: utf-8 -*-
"""
_build_error_files
========================
Created TRACE data reports

Author : Alex Dickerson
Created: 2025-10-20
"""

from __future__ import annotations
import _error_plot_helpers as HLP
from pathlib import Path

# ---------------------------
# Editable defaults 
# ---------------------------
DATE           = "" # Date of the data run
output_figures = True # Set to True for all the error plots #
# Either only e.g., 'enhanced', or list of the databases, e.g. ['enhanced','144a']
DATA_TYPES     = ['enhanced','standard', '144a'] 

# Can leave these blank for execution directory, e.g., ""
# Or, put your path, e.g., for Windows: C:\Users\proj\
IN_DIR         = "" #Path(r"C:\Users\ASUS\Dropbox\01_trace\data\stage0\2025_20_10")
OUT_DIR        = "" #Path(r"C:\Users\ASUS\Dropbox\01_trace\data\stage0\2025_20_10")

SUBPLOT_DIM    = (4, 2)# Leave this alone
USE_LATEX_FONTS= False # Change to True, but your plots will take ages to render

# ---------------------------
# Helpers 
# ---------------------------

def _is_blank_path(p) -> bool:
    # Accept None, "", Path(""), or Path() as "blank"
    try:
        if p is None:
            return True
        if isinstance(p, str) and p.strip() == "":
            return True
        if isinstance(p, Path) and (str(p).strip() == "" or p == Path()):
            return True
        return False
    except Exception:
        return True

def _as_path_or_cwd(p) -> Path:
    # Resolve user input to a real path, defaulting to the execution directory (CWD)
    if _is_blank_path(p):
        return Path.cwd()
    return Path(p).expanduser().resolve()

# Plot style (use_latex wired to toggle above)
PLOT_STYLE = HLP.PlotParams(
    orientation="auto",
    use_latex=USE_LATEX_FONTS,
    base_font=10, title_size=10, label_size=10, tick_size=9, legend_size=9,
    x_spacing="rank",
    all_color="orange", all_alpha=0.75, all_lw=1.0,
    filtered_color="blue", filtered_lw=1.3,
    show_flagged=True, flagged_size=16, flagged_edgecolor="red",
    # defaults for speed
    export_format="pdf",         # fastest to include in LaTeX
    figure_dpi=150,              # used only if you switch to PNG/JPG
    transparent=False,           # force opaque (smaller/faster)
    jpeg_quality=85,             # if you ever set export_format="jpg"
)

# ---------------------------
# Imports
# ---------------------------
import argparse
import sys
import logging
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
import wrds
import gc

from _trace_settings import get_config
import create_daily_enhanced_trace as EDT 
import create_daily_standard_trace as SDT 

# ---------------------------
# Logging
# ---------------------------
def _setup_logger(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"_build_error_files_{ts}.log"

    # Build handlers explicitly and clear any existing ones
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Nuke any pre-existing handlers (e.g., from imported libs / SGE env)
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)

    fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
                            )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    root.addHandler(fh)
    root.addHandler(sh)

    # (Optional) quieten noisy libs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info("=== _build_error_files.py started ===")
    logging.info("Log file: %s", log_path)

def _log_runtime_environment() -> None:
    """Log versions, paths, and selected env vars to root logger (for WRDS debugging)."""
    import sys, os, platform
    try: import pandas as pd
    except Exception: pd = None
    try: import numpy as np
    except Exception: np = None
    try: import pyarrow as pa
    except Exception: pa = None
    try: import matplotlib as mpl
    except Exception: mpl = None
    try: import pandas_market_calendars as mcal
    except Exception: mcal = None
    try: import wrds
    except Exception: wrds = None

    logging.info("=== Runtime environment ===")
    logging.info("Python %s @ %s", platform.python_version(), sys.executable)
    if pd:   logging.info("pandas %s (%s)", pd.__version__, getattr(pd, "__file__", "?"))
    if np:   logging.info("numpy  %s (%s)", np.__version__, getattr(np, "__file__", "?"))
    if pa:   logging.info("pyarrow %s (%s)", pa.__version__, getattr(pa, "__file__", "?"))
    if mpl:  logging.info("matplotlib %s (%s)", mpl.__version__, getattr(mpl, "__file__", "?"))
    if mcal: logging.info("pandas_market_calendars %s (%s)", mcal.__version__, getattr(mcal, "__file__", "?"))
    if wrds: logging.info("wrds   %s (%s)", wrds.__version__, getattr(wrds, "__file__", "?"))
    logging.info("CWD: %s", os.getcwd())
    logging.info("PATH: %s", os.environ.get("PATH", ""))
    logging.info("MPLBACKEND: %s", os.environ.get("MPLBACKEND", ""))
    logging.info("===========================")


# ---------------------------
# Helpers
# ---------------------------
def _load_cusip_list(parquet_path: Path, col_name: str = "cusip_id") -> pd.Index:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if col_name in df.columns:
        s = df[col_name]
    else:
        cand = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        if not cand:
            raise ValueError(f"No string-like '{col_name}' column found in {parquet_path}.")
        s = df[cand[0]]
    cusips = (
        s.astype(str)
         .str.strip()
         .dropna()
         .loc[lambda x: x.str.len() > 0]
         .unique()
    )
    return pd.Index(cusips, dtype="object")

def _split_into_chunks(x: pd.Index, n: int) -> list[list[str]]:
    n = max(int(n), 1)
    if len(x) == 0:
        return []
    arr = np.array(x, dtype=object)
    parts = np.array_split(arr, int(np.ceil(len(arr) / n)))
    return [p.astype(str).tolist() for p in parts if len(p)]

def _bootstrap_auditors(dtype: str):
    if dtype == 'enhanced':
        if not hasattr(EDT, "audit_records"):
            EDT.audit_records = []
        if not hasattr(EDT, "ct_audit_records"):
            EDT.ct_audit_records = []
    else:
        if not hasattr(SDT, "audit_records"):
            SDT.audit_records = []
        if not hasattr(SDT, "ct_audit_records"):
            SDT.ct_audit_records = []


# ---------------------------
# Argument resolution
# ---------------------------
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-run cleaning/figures and build reports.")
    p.add_argument("--date", help="YYYYmmdd, e.g., 20251017")
    p.add_argument("--in-dir", help="Folder with *_filters_*.parquet inputs")
    p.add_argument("--out-dir", help="Folder to write outputs")
    p.add_argument("--wrds-username", help="WRDS username (overrides settings if supplied)")
    p.add_argument("--fisd-parquet", help="Optional parquet with ['cusip_id','offering_amt','maturity']")
    p.add_argument("--data-type", action="append",
                   help="One or more of enhanced,standard,144a. Repeat flag or use comma-separated. Use 'all' for all three.")
    return p.parse_args()


def _resolve_args() -> dict:
    ns = _parse_cli()

    cfg = dict(
        date = str(DATE).strip() if DATE else "",
        in_dir = IN_DIR,
        out_dir = OUT_DIR,
        data_types = list(DATA_TYPES),  # default from top-of-file
    )

    if getattr(ns, "date", None):
        cfg["date"] = (ns.date or "").strip()
    if getattr(ns, "in_dir", None):
        cfg["in_dir"] = ns.in_dir
    if getattr(ns, "out_dir", None):
        cfg["out_dir"] = ns.out_dir
    if getattr(ns, "wrds_username", None):
        cfg["wrds_username"] = (ns.wrds_username or "").strip()
    if getattr(ns, "fisd_parquet", None):
        s = (ns.fisd_parquet or "").strip()
        cfg["fisd_parquet"] = Path(s).expanduser().resolve() if s else None

    # Parse data types from CLI if provided
    if getattr(ns, "data_type", None):
        raw = []
        for item in ns.data_type:
            raw.extend([t.strip().lower() for t in item.split(",") if t.strip()])
        if "all" in raw:
            cfg["data_types"] = ['enhanced', 'standard', '144a']
        else:
            allowed = {'enhanced', 'standard', '144a'}
            chosen = [t for t in raw if t in allowed]
            if not chosen:
                raise SystemExit(f"No valid --data-type provided (got {raw}).")
            cfg["data_types"] = chosen

    # ---- Robust path resolution ----
    in_dir_base  = _as_path_or_cwd(cfg["in_dir"])
    out_dir_base = _as_path_or_cwd(cfg["out_dir"])

    # All outputs must live under OUT_DIR/data_reports
    report_dir = (out_dir_base / "data_reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    cfg["date"]      = cfg["date"] or datetime.now().strftime("%Y%m%d")
    cfg["in_dir"]    = in_dir_base
    cfg["out_dir"]   = report_dir
    return cfg


# ---------------------------
# Main
# ---------------------------
def main():
    args          = _resolve_args()
    date_override = (args["date"] or "").strip()   # empty --> auto per dtype
    in_dir        = Path(args["in_dir"])
    base_out      = Path(args["out_dir"])
    wrds_user     = args.get("wrds_username") or ""
    data_types    = args["data_types"]
    
    _setup_logger(base_out)
    _log_runtime_environment()
    date_disp = date_override if date_override else "(auto: RUN_STAMP)"
    logging.info("Resolved inputs -> date=%s | in_dir=%s | report_root=%s | types=%s",
                 date_disp, in_dir, base_out, ",".join(data_types))



    db = None
    try:
        for dtype in data_types:
            logging.info("========== Processing data_type = %s ==========", dtype)

            # Per-type out dir
            out_dir = base_out / dtype
            out_dir.mkdir(parents=True, exist_ok=True)
            subfolder  = "enhanced" if dtype == "enhanced" else ("144a" if dtype == "144a" else "standard")
            in_dir_t   = in_dir / subfolder

            _bootstrap_auditors(dtype)
            cfg = get_config(dtype)
            
            # Resolve the effective date tag for this dtype            
            if date_override:
                eff_date_tag = date_override
                eff_src = "override"
            else:
                if dtype == "enhanced":
                    eff_date_tag = EDT.RUN_STAMP  # from create_daily_enhanced_trace.py
                    eff_src = "EDT.RUN_STAMP"
                else:
                    eff_date_tag = SDT.RUN_STAMP  # from create_daily_standard_trace.py (standard/144a)
                    eff_src = "SDT.RUN_STAMP"
            logging.info("[%s] Effective date tag: %s (source: %s)", dtype, eff_date_tag, eff_src)

            # Per-type config
            cfg = get_config(dtype)
            if wrds_user:
                cfg["wrds_username"] = wrds_user
            wrds_user_eff = cfg.get("wrds_username", "")

            # Parameter dicts -> neat tables
            filters_dict = dict(cfg.get("filters", {})) or {}
            ds_params    = dict(cfg.get("ds_params", {}))  or {}
            bb_params    = dict(cfg.get("bb_params", {}))  or {}
            fisd_params  = dict(cfg.get("fisd_params", {})) or {}

            filters_df = HLP._filters_to_df(filters_dict)
            ds_df      = HLP._dict_to_df(ds_params, key_header="Decimal-Shift Parameter", val_header="Value")
            bb_df      = HLP._dict_to_df(bb_params, key_header="Bounce-Back Parameter",   val_header="Value")

            # Audit / FISD filter tables (must exist from Stage-0)
            fn_fisd = in_dir_t / f"fisd_filters_{dtype}_{eff_date_tag}.parquet"
            fn_dn   = in_dir_t / f"dick_nielsen_filters_audit_{dtype}_{eff_date_tag}.parquet"
            fn_dr   = in_dir_t / f"drr_filters_audit_{dtype}_{eff_date_tag}.parquet"

            logging.info("Loading audit tables for %s .", dtype)
            dn = pd.read_parquet(fn_dn)
            dr = pd.read_parquet(fn_dr)
            df = pd.read_parquet(fn_fisd)

            # Build summary tables
            filters_table_dr = HLP.build_filter_summary(dr, decimal_stage="decimal_shift")
            total_start      = int(dr.loc[dr["stage"] == "start", "rows_before"].sum())
            if total_start == 0:
                total_start = int(dr.loc[dr["stage"] == "dick_nielsen_filter", "rows_before"].sum())
            filters_table_dn = HLP.build_dn_summary(dn, total_start=total_start)
            filters_table_fi = HLP.filters_df_to_summary(df)

            # Optional figure build
            pages_made_ds = []
            pages_made_bb = []

            if output_figures:
                # Check that required CUSIP files exist
                fn_bb = in_dir_t / f"bounce_back_cusips_{dtype}_{eff_date_tag}.parquet"
                fn_ds = in_dir_t / f"decimal_shift_cusips_{dtype}_{eff_date_tag}.parquet"

                missing_files = []
                if not fn_bb.exists():
                    missing_files.append(str(fn_bb))
                if not fn_ds.exists():
                    missing_files.append(str(fn_ds))

                if missing_files:
                    error_msg = (
                        f"[{dtype}] Cannot generate figures: required CUSIP files are missing.\n"
                        f"Missing files:\n  " + "\n  ".join(missing_files) + "\n"
                        f"These files must exist when output_figures=True.\n"
                        f"Please run the data processing step that generates these CUSIP files first."
                    )
                    logging.error(error_msg)
                    raise FileNotFoundError(error_msg)
                else:
                    logging.info("[%s] Loading CUSIP lists: %s, %s", dtype, fn_bb.name, fn_ds.name)
                    bb = _load_cusip_list(fn_bb)
                    ds = _load_cusip_list(fn_ds)
                    cusips_union = pd.Index(sorted(pd.Index(bb).union(ds)))
                    logging.info("[%s] Unique CUSIPs in union: %s", dtype, f"{len(cusips_union):,}")

                    if len(cusips_union) == 0:
                        logging.warning("[%s] No CUSIPs in union. Skipping figure build.", dtype)
                    else:
                        chunk_size   = int(cfg.get("chunk_size", 250))
                        cusip_chunks = _split_into_chunks(cusips_union, n=chunk_size)
                        
                        # For Enhanced - this helps us not read-in too much data
                        if len(cusip_chunks) > 10:
                            new_chunk_size = max(1, chunk_size // 4)
                            if new_chunk_size != chunk_size:
                                logging.info(
                                    "[%s] Too many chunks (%s). Halving chunk_size %s -> %s and re-partitioning.",
                                    dtype, f"{len(cusip_chunks):,}", chunk_size, new_chunk_size
                                )
                                chunk_size   = new_chunk_size
                                cusip_chunks = _split_into_chunks(cusips_union, n=chunk_size)
                        
                        logging.info("[%s] Prepared %s chunk(s) using chunk_size=%s",
                                     dtype, f"{len(cusip_chunks):,}", chunk_size)
                                            
                        # WRDS connection (lazy)
                        if db is None:
                            logging.info("Connecting to WRDS as '%s' ...", wrds_user_eff)
                            db = wrds.Connection(wrds_username=wrds_user_eff) if wrds_user_eff else wrds.Connection()
                            logging.info("WRDS connection established.")

                        filters = dict(cfg.get("filters", {})) or {}
                        ds_params_uncleaned = {**cfg.get("ds_params", {}), "output_type": "uncleaned"}
                        gc.collect()
                        logging.info("[%s] Starting clean_trace_data() on restricted CUSIP universe .", dtype)
                        if dtype == 'enhanced':
                            dfds, dfbb, bb_cusips_all, dec_shift_cusips_all = EDT.error_checks(
                                db=db,
                                cusip_chunks=cusip_chunks,
                                clean_agency=cfg.get("clean_agency"),
                                volume_filter=cfg.get("volume_filter", ("par", 10000)),
                                trade_times=cfg.get("trade_times", ["00:00:00", "23:59:59"]),
                                calendar_name=cfg.get("calendar_name", "NYSE"),
                                ds_params=ds_params_uncleaned,
                                bb_params=cfg.get("bb_params", {}),
                                filters=filters,
                            )
                        else:
                            dfds, dfbb, bb_cusips_all, dec_shift_cusips_all = SDT.error_checks(
                                db=db,
                                cusip_chunks=cusip_chunks,
                                data_type=dtype,
                                start_date=cfg.get("start_date"),
                                volume_filter=cfg.get("volume_filter", ("par", 10000)),
                                trade_times=cfg.get("trade_times", ["00:00:00", "23:59:59"]),
                                calendar_name=cfg.get("calendar_name", "NYSE"),
                                ds_params=ds_params_uncleaned,
                                bb_params=cfg.get("bb_params", {}),
                                filters=filters,
                            )
                        gc.collect()                                                   
                        # --- dtype normalization + pre-sort for plotting speed ---
                        for name, _df in (("dfds", dfds), ("dfbb", dfbb)):
                            if not pd.api.types.is_datetime64_any_dtype(_df["trd_exctn_dt"]):
                                _df["trd_exctn_dt"] = pd.to_datetime(_df["trd_exctn_dt"], errors="coerce")
                            # Ensure stable within-CUSIP chronological order (so panels don't sort)
                            _df.sort_values(["cusip_id", "trd_exctn_dt"], inplace=True, kind="mergesort")
                        
                        # Build fast index maps once per table
                        idx_map_ds = dfds.groupby("cusip_id", sort=False).indices
                        idx_map_bb = dfbb.groupby("cusip_id", sort=False).indices

                                                                        
                        # Plot pages
                        def batched(seq: Sequence[str], n: int):
                            for i in range(0, len(seq), n):
                                yield seq[i:i+n], (i//n + 1)

                        rows, cols = SUBPLOT_DIM
                        per_page = rows * cols
                        
                        
                        for chunk, page_idx in batched(dec_shift_cusips_all, per_page):
                            stub = f"{dtype}_fig_page_{page_idx:03d}"
                            png_path = HLP.make_panel(
                                df_out=dfds,
                                error_cusips=chunk,
                                subplot_dim=SUBPLOT_DIM,
                                export_dir=out_dir,
                                filename_stub=stub,
                                params=PLOT_STYLE,
                                idx_map=idx_map_ds,    
                                error_type="decimal_shift",
                            )
                            pages_made_ds.append(png_path.name)
                            logging.info("[%s] Saved DS page %03d: %s", dtype, page_idx, png_path.name)

                        for chunk, page_idx in batched(bb_cusips_all, per_page):
                            stub = f"{dtype}_fig_page_{page_idx:03d}"
                            png_path = HLP.make_panel(
                                df_out=dfbb,
                                error_cusips=chunk,
                                subplot_dim=SUBPLOT_DIM,
                                export_dir=out_dir,
                                filename_stub=stub,
                                params=PLOT_STYLE,
                                error_type="bounce_back",
                                idx_map=idx_map_bb,         
                            )
                            pages_made_bb.append(png_path.name)
                            logging.info("[%s] Saved BB page %03d: %s", dtype, page_idx, png_path.name)

            # Build LaTeX report #
            kwargs = dict(
                out_dir=out_dir,
                data_type=dtype,
                filters_df=filters_df,
                ds_df=ds_df,
                bb_df=bb_df,
                fisd_params=fisd_params,
                filters_table_dr=filters_table_dr,
                filters_table_dn=filters_table_dn,
                filters_table_fi=filters_table_fi,
                output_figures=output_figures,
            )
            if output_figures:
                kwargs["pages_made_ds"] = pages_made_ds
                kwargs["pages_made_bb"] = pages_made_bb

            HLP.build_data_report_tex(**kwargs)
            logging.info("========== Finished data_type = %s ==========", dtype)

    finally:
        if db is not None:
            try:
                db.close()
                logging.info("WRDS connection closed.")
            except Exception:
                pass

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        logging.exception("SystemExit encountered (exit code %s).", getattr(e, "code", "?"))
        raise
    except Exception:
        logging.exception("FATAL: Unhandled exception in _build_error_files.py")
        sys.exit(1)
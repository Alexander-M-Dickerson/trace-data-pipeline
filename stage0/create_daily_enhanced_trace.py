# -*- coding: utf-8 -*-
"""
create_daily_enhanced_trace
========================
Pulls, cleans, and saves TRACE Enhanced data in daily-frequency panels.

Author : Alex Dickerson
Created: 2025-10-22
"""

# -------------------------------------------------------------------------
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import List, Dict, Sequence, Mapping, Any, Optional, Tuple
import pandas as pd
import numpy as np
import time
import wrds
import gc
from functools import reduce
import pyarrow as pa
import pandas_market_calendars as mcal
RUN_STAMP = pd.Timestamp.today().strftime("%Y%m%d")
# -------------------------------------------------------------------------
def _configure_root_logger(level: int = logging.INFO) -> None:    
    root = logging.getLogger()
    if root.handlers:                       
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
# -------------------------------------------------------------------------
def log_filter(df_before: pd.DataFrame,
               df_after:  pd.DataFrame,
               stage: str,
               chunk_id: int,
               *,
               replace: bool = False,
               n_rows_replaced: int = 0) -> None:
    rows_before = len(df_before)
    rows_after  = len(df_after)

    removed = (n_rows_replaced if replace else (rows_before - rows_after))

    audit_records.append(
        dict(
            chunk       = chunk_id,
            stage       = stage,
            rows_before = rows_before,
            rows_after  = rows_after,
            removed     = int(removed),
        )
    )

    if replace:
        logging.info(
            f"[chunk {chunk_id:03}] {stage:<30} "
            f"kept {rows_after:,} (replaced {int(removed):,})"
        )
    else:
        logging.info(
            f"[chunk {chunk_id:03}] {stage:<30} "
            f"kept {rows_after:,} (-{rows_before - rows_after:,})"
        )

# Filter with a boolean mask --------------------------
def filter_with_log(df: pd.DataFrame,
                    mask: pd.Series,
                    stage: str,
                    chunk_id: int) -> pd.DataFrame:
    before = df
    after  = df.loc[mask].copy()          
    log_filter(before, after, stage, chunk_id)
    return after
# -------------------------------------------------------------------------
def log_fisd_filter(df_before: pd.DataFrame,
                    df_after:  pd.DataFrame,
                    stage: str) -> None:
    """Append one audit row for the FISD cleaning step `stage`."""
    fisd_audit_records.append(
        dict(stage       = stage,
             rows_before = len(df_before),
             rows_after  = len(df_after),
             removed     = len(df_before) - len(df_after))
    )
    logging.info(f"[FISD] {stage:<35} "
                 f"kept {len(df_after):,} "
                 f"(-{len(df_before)-len(df_after):,})")
# -------------------------------------------------------------------------       
def log_ct_filter(before, after, stage, chunk_id):
    """Append an audit row for clean_trace_chunk-level filters."""
    ct_audit_records.append(
        dict(chunk       = chunk_id,
             stage       = stage,
             rows_before = len(before),
             rows_after  = len(after),
             removed     = len(before) - len(after))
    )
# -------------------------------------------------------------------------     
def _normalize_volume_filter(v) -> Tuple[str, float]:
    """
    Accept either a scalar threshold (legacy: dollar) or a (kind, threshold) tuple.
    Returns (kind, threshold) with kind in {"dollar","par"} (lowercased).
    """
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return ("dollar", float(v))
    if isinstance(v, (tuple, list)) and len(v) == 2:
        kind, thr = v[0], v[1]
        kind = str(kind).strip().lower()
        if kind not in {"dollar", "par"}:
            raise ValueError("volume_filter kind must be 'dollar' or 'par'")
        try:
            thr = float(thr)
        except Exception:
            raise ValueError("volume_filter threshold must be numeric")
        return (kind, thr)
    raise ValueError("volume_filter must be a number or a 2-tuple ('dollar'|'par', threshold)")
# -------------------------------------------------------------------------
def time_to_seconds(time_str):
    """
    Convert time string 'HH:MM:SS' to seconds since midnight (int32).

    This is ~50% more RAM-efficient than storing as string.

    Parameters
    ----------
    time_str : str or pd.Series
        Time in format 'HH:MM:SS'

    Returns
    -------
    int or pd.Series
        Seconds since midnight (0-86399)
    """
    # Check if Series FIRST (before pd.isna which returns Series of bools)
    if isinstance(time_str, pd.Series):
        # Vectorized operation for Series
        return time_str.apply(lambda x: time_to_seconds(x) if pd.notna(x) else np.nan)

    if pd.isna(time_str):
        return np.nan

    # Parse HH:MM:SS
    parts = time_str.split(':')
    if len(parts) != 3:
        return np.nan

    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return np.nan
# -------------------------------------------------------------------------
def compute_trace_all_metrics(trace):
    """
    Aggregate TRACE trades to daily (cusip_id, trd_exctn_dt) panel with
    price, volume, and customer-side bid/ask summaries.

    Parameters
    ----------
    trace : pandas.DataFrame
        Must contain: cusip_id, trd_exctn_dt, rptd_pr, entrd_vol_qt, 
        rpt_side_cd, cntra_mp_id

    Returns
    -------
    pandas.DataFrame
        Daily panel with columns: cusip_id, trd_exctn_dt, prc_ew, prc_vw, prc_vw_par,
        prc_first, prc_last, trade_count, qvolume, dvolume, prc_bid, prc_ask,
        bid_count, ask_count
    """
        
    # Ensure we have dollar_vol
    if 'dollar_vol' not in trace.columns:
        trace['dollar_vol'] = trace['rptd_pr'] * trace['entrd_vol_qt'] / 100

    # Convert trd_exctn_tm from string to seconds (int32) for RAM efficiency
    if 'trd_exctn_tm' in trace.columns:
        trace['trd_exctn_tm_sec'] = time_to_seconds(trace['trd_exctn_tm']).astype('Int32')
    else:
        trace['trd_exctn_tm_sec'] = pd.NA

    # Split into bid and ask dataframes
    # Customer Only #
    _bid = trace[((trace['rpt_side_cd'] == 'B')&(trace['cntra_mp_id'] == 'C'))].copy()
    _ask = trace[((trace['rpt_side_cd'] == 'S')&(trace['cntra_mp_id'] == 'C'))].copy()
    
    # Note this may mean we have a valid prc but not bid or ask because 
    # rpt_side_cd or cntra_mp_id is missing.
        
    #--------------------------------------------------------------------------
    # 1. Compute PricesAll (Equal-weighted, volume-weighted prices and trade count)
    #--------------------------------------------------------------------------
    
    # Precompute the weighted products
    trace['dollar_weighted_price'] = trace['rptd_pr'] * trace['dollar_vol']
    trace['volume_weighted_price'] = trace['rptd_pr'] * trace['entrd_vol_qt']

    agg_dict = {
        'rptd_pr': ['mean', 'first', 'last', 'max', 'min', 'count'],
        'dollar_vol': 'sum',
        'entrd_vol_qt': 'sum',
        'dollar_weighted_price': 'sum',
        'volume_weighted_price': 'sum',
        'trd_exctn_tm_sec': ['mean', 'last']
    }

    
    results = trace.groupby(['cusip_id', 'trd_exctn_dt']).agg(agg_dict)
    
    # Flatten column names
    results.columns = ['_'.join(col).strip() for col in results.columns.values]
    
    # Calculate weighted prices
    results['prc_vw'] = results['dollar_weighted_price_sum'] / results['dollar_vol_sum']
    results['prc_vw_par'] = results['volume_weighted_price_sum'] / results['entrd_vol_qt_sum']
    
    # Create final PricesAll dataframe
    PricesAll = results.rename(columns={
        'rptd_pr_mean': 'prc_ew',
        'rptd_pr_first': 'prc_first',
        'rptd_pr_last': 'prc_last',
        'rptd_pr_max': 'prc_hi',
        'rptd_pr_min': 'prc_lo',
        'rptd_pr_count': 'trade_count',
        'trd_exctn_tm_sec_mean': 'time_ew',
        'trd_exctn_tm_sec_last': 'time_last'
    }).reset_index()


    # Select columns
    PricesAll = PricesAll[['cusip_id', 'trd_exctn_dt', 'prc_ew', 'prc_vw', 'prc_vw_par',
                           'prc_first', 'prc_last', 'prc_hi', 'prc_lo', 'trade_count',
                           'time_ew', 'time_last']]
   
    #--------------------------------------------------------------------------
    # 2. Compute VolumesAll 
    #--------------------------------------------------------------------------
    
    # 
    VolumesAll = trace.groupby(['cusip_id', 'trd_exctn_dt']).agg({
        'entrd_vol_qt': 'sum',
        'dollar_vol': 'sum'
    }).reset_index()
    
    # Volume is in millions #
    VolumesAll['entrd_vol_qt'] = VolumesAll['entrd_vol_qt'] / 1000000
    VolumesAll['dollar_vol']   = VolumesAll['dollar_vol']   / 1000000
    
    # 
    VolumesAll = VolumesAll.rename(columns={
        'entrd_vol_qt': 'qvolume',
        'dollar_vol': 'dvolume'
    })
    
    #--------------------------------------------------------------------------
    # 3. Compute Bid-Ask metrics
    #--------------------------------------------------------------------------
    
    # Initialize an empty result dataframe
    prc_BID_ASK = pd.DataFrame(columns=['cusip_id', 'trd_exctn_dt', 'prc_bid',
                                        'bid_last', 'bid_time_ew', 'bid_time_last',
                                        'prc_ask', 'bid_count', 'ask_count'])
    
    # Process bid data
    if not _bid.empty:
        # Sort by cusip and execution date
        _bid = _bid.sort_values(['cusip_id', 'trd_exctn_dt'])

        # Calculate dollar volume and value weights (fully vectorized)
        _bid['dollar_volume'] = _bid['entrd_vol_qt'] * _bid['rptd_pr'] / 100
        bid_dollar_vol_sums = _bid.groupby(['cusip_id', 'trd_exctn_dt'])['dollar_volume'].transform('sum')
        _bid['value_weights'] = _bid['dollar_volume'] / bid_dollar_vol_sums
        
        # Calculate weighted price products
        _bid['weighted_price'] = _bid['rptd_pr'] * _bid['value_weights']

        # Group and aggregate
        bid_agg = _bid.groupby(['cusip_id', 'trd_exctn_dt']).agg({
            'rptd_pr': ['count', 'last'],
            'weighted_price': 'sum',
            'trd_exctn_tm_sec': ['mean', 'last']
        })

        # Flatten multi-level columns
        bid_agg.columns = ['bid_count', 'bid_last', 'prc_bid', 'bid_time_ew', 'bid_time_last']
        
        # Reset index for merging
        prc_BID = bid_agg.reset_index()
        
        # Initialize with bid data
        prc_BID_ASK = prc_BID.copy()
        
        # Add empty ask columns
        if 'prc_ask' not in prc_BID_ASK.columns:
            prc_BID_ASK['prc_ask'] = np.nan
        if 'ask_count' not in prc_BID_ASK.columns:
            prc_BID_ASK['ask_count'] = 0
    
    # Process ask data
    if not _ask.empty:
        # Sort by cusip and execution date
        _ask = _ask.sort_values(['cusip_id', 'trd_exctn_dt'])

        # Calculate dollar volume and value weights (fully vectorized)
        _ask['dollar_volume'] = _ask['entrd_vol_qt'] * _ask['rptd_pr'] / 100
        ask_dollar_vol_sums = _ask.groupby(['cusip_id', 'trd_exctn_dt'])['dollar_volume'].transform('sum')
        _ask['value_weights'] = _ask['dollar_volume'] / ask_dollar_vol_sums
        
        # Calculate weighted price products
        _ask['weighted_price'] = _ask['rptd_pr'] * _ask['value_weights']
        
        # Group and aggregate
        ask_agg = _ask.groupby(['cusip_id', 'trd_exctn_dt']).agg({
            'rptd_pr': 'count',
            'weighted_price': 'sum'
        })
        
        # Rename columns for clarity
        ask_agg.columns = ['ask_count', 'prc_ask']
        
        # Reset index for merging
        prc_ASK = ask_agg.reset_index()
        
        if prc_BID_ASK.empty:
            # If we have no bid data, initialize with ask data
            prc_BID_ASK = prc_ASK.copy()
            
            # Add empty bid columns
            if 'prc_bid' not in prc_BID_ASK.columns:
                prc_BID_ASK['prc_bid'] = np.nan
            if 'bid_last' not in prc_BID_ASK.columns:
                prc_BID_ASK['bid_last'] = np.nan
            if 'bid_time_ew' not in prc_BID_ASK.columns:
                prc_BID_ASK['bid_time_ew'] = pd.NA
            if 'bid_time_last' not in prc_BID_ASK.columns:
                prc_BID_ASK['bid_time_last'] = pd.NA
            if 'bid_count' not in prc_BID_ASK.columns:
                prc_BID_ASK['bid_count'] = 0
        else:
            # If we already have bid data, merge with ask data
            prc_BID_ASK = prc_BID_ASK.merge(
                prc_ASK, 
                how="outer", 
                on=['cusip_id', 'trd_exctn_dt']
            )
            
            # Fix column names if needed after merge
            if 'prc_ask_x' in prc_BID_ASK.columns:
                # This means we had overlapping column names
                prc_BID_ASK['prc_ask'] = prc_BID_ASK['prc_ask_y'].fillna(prc_BID_ASK['prc_ask_x'])
                prc_BID_ASK['ask_count'] = prc_BID_ASK['ask_count_y'].fillna(prc_BID_ASK['ask_count_x'])
                prc_BID_ASK = prc_BID_ASK.drop(['prc_ask_x', 'prc_ask_y', 'ask_count_x', 'ask_count_y'], axis=1)
    
    # Ensure all required columns exist
    if not prc_BID_ASK.empty:
        # Select and order columns
        prc_BID_ASK = prc_BID_ASK[['cusip_id', 'trd_exctn_dt', 'prc_bid',
                                  'bid_last', 'bid_time_ew', 'bid_time_last',
                                  'prc_ask', 'bid_count', 'ask_count']]
    
    # ------------------------------------------------------------------ #
    # 4. Merge everything - FULL OUTER JOIN                              #
    # ------------------------------------------------------------------ #
    dfs           = [PricesAll, VolumesAll, prc_BID_ASK]
    dfs_non_empty = [df for df in dfs if not df.empty]

    if not dfs_non_empty:
        # unlikely, but keeps type-safety
        return pd.DataFrame(columns=['cusip_id','trd_exctn_dt'])

    merged = reduce(
        lambda left, right: pd.merge(
            left, right, on=['cusip_id','trd_exctn_dt'], how='outer'),
        dfs_non_empty
    )

    # sort rows for tidy output
    merged = merged.sort_values(['cusip_id','trd_exctn_dt']).reset_index(drop=True)
    return merged    
# -------------------------------------------------------------------------
def clean_trace_data(
    db,
    cusip_chunks,
    fisd_off,
    clean_agency: bool = True,
    volume_filter: float | tuple[str, float] = ("dollar", 10000.0),
    trade_times: list[str] | None = None,
    calendar_name: str | None = None,
    *,
    fetch_fn=None,
    ds_params: dict | None = None,
    bb_params: dict | None = None,
    init_error_params: dict | None = None,
    filters: dict | None = None
                    ):
    
    if fetch_fn is None:
        fetch_fn = lambda sql, params=None: db.raw_sql(sql, params=params)

    # --- Filter Defaults -----------------
    FILTER_DEFAULTS = dict(
        dick_nielsen            = True,
        decimal_shift_corrector = True,
        trading_time            = False,
        trading_calendar        = True,
        price_filters           = True,
        volume_filter_toggle    = True,
        bounce_back_filter      = True,
        yld_price_filter        = True,
        amtout_volume_filter    = True,
        trd_exe_mat_filter      = True,
        flag_initial_price_errors = True,
    )
    f = {**FILTER_DEFAULTS, **(filters or {})}
    """
    Fetch, clean, and aggregate Enhanced TRACE data in CUSIP chunks, then build
    daily price/volume/bid-ask metrics. Also applies decimal-shift correction and
    bounce-back (price change) filtering prior to aggregation.

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection; must support .raw_sql(...).
    cusip_chunks : list[list[str]]
        Lists of CUSIPs to process per chunk.
    fisd_off : pandas.DataFrame
        Issue-level subset with at least ['cusip_id','offering_amt','maturity'] used
        for volume vs outstanding and maturity date filters.
    clean_agency : bool, default True
        Apply Dick-Nielsen style agency de-duplication within the cleaning step..
    volume_filter : float | tuple[str, float], default ("dollar", 10000.0)
        If float/int: legacy dollar-volume threshold.
        If tuple: ("dollar", x) uses dollar_vol >= x; ("par", x) uses entrd_vol_qt >= x.
    trade_times : list[str] | None, default None
        Optional inclusive intraday window ["HH:MM:SS","HH:MM:SS"] to retain trades
        by execution time.
    calendar_name : str | None, default None
        Optional pandas_market_calendars name (e.g., "NYSE") to retain trades whose
        dates fall on valid sessions.
    ds_params : dict | None, default None
        Keyword overrides forwarded to decimal_shift_corrector(...), e.g.
        factors, tol_pct_good, tol_abs_good, tol_pct_bad, low_pr, high_pr,
        anchor, window, improvement_frac, par_snap, par_band, output_type.
    bb_params : dict | None, default None
        Keyword overrides forwarded to flag_price_change_errors(...), e.g.
        threshold_abs, lookahead, max_span, window, back_to_anchor_tol,
        candidate_slack_abs, reassignment_margin_abs, use_unique_trailing_median,
        par_spike_heuristic, par_level, par_equal_tol, par_min_run,
        par_cooldown_after_flag.
    init_error_params : dict | None, default None
        Keyword overrides forwarded to flag_initial_price_errors(...), e.g.
        abs_change, n_transactions.

    Returns
    -------
    tuple[pandas.DataFrame, list[str], list[str], list[str]]
        final_df :
            Daily, per-(cusip_id, trd_exctn_dt) metrics frame combining prices,
            volumes and bid/ask summaries (sorted and ready for export).
        bb_cusips_all :
            Unique CUSIPs that had at least one row flagged by the
            bounce-back/price-change filter in any processed chunk.
        dec_shift_cusips_all :
            Unique CUSIPs that had at least one row corrected by the
            decimal-shift corrector in any processed chunk.
        init_price_cusips_all :
            Unique CUSIPs that had at least one row flagged by the
            initial price error filter in any processed chunk.
    """
    
    all_super_list       = []
    bb_cusips_all        = []
    dec_shift_cusips_all = []
    init_price_cusips_all = []

    sort_cols = ["cusip_id","trd_exctn_dt","trd_exctn_tm",
                 "trd_rpt_dt","trd_rpt_tm","msg_seq_nb"]
    
    for i in range(0, len(cusip_chunks)):  
        start_time = time.time()  # Start timer
        logging.info(f"Processing chunk {i+1} of {len(cusip_chunks)}")        
        temp_list = cusip_chunks[i] 
        temp_tuple = tuple(temp_list)
        parm = {'cusip_id': temp_tuple}
        
        # Load data from WRDS per chunk
        trace = fetch_fn('''
            SELECT cusip_id, bond_sym_id, trd_exctn_dt, trd_exctn_tm, days_to_sttl_ct,
                   lckd_in_ind, wis_fl, sale_cndtn_cd, msg_seq_nb, trc_st,
                   trd_rpt_dt, trd_rpt_tm, entrd_vol_qt, rptd_pr, yld_pt,
                   asof_cd, orig_msg_seq_nb, rpt_side_cd, cntra_mp_id
            FROM trace.trace_enhanced
            WHERE cusip_id IN %(cusip_id)s
              AND cusip_id IS NOT NULL
              AND TRIM(cusip_id) != ''
        ''', params=parm)
       
        logging.info(f"Chunk {i+1}: Retrieved {len(trace)} rows from WRDS")
        
        if len(trace) == 0:
            continue
              
        trace["rptd_pr"] = trace["rptd_pr"].astype("float64").round(6)
        trace = trace.drop(columns=["index"], errors="ignore").reset_index(drop=True)
        
        # Initial log for cleaning
        log_filter(trace, trace, "start", i)
        
        # Filter 1: Dick-Nielsen
        if f["dick_nielsen"]:
            clean_chunk = clean_trace_chunk(
                trace,
                chunk_id      = i,
                clean_agency  = clean_agency,
                logger        = log_ct_filter
            )
            log_filter(trace, clean_chunk, "dick_nielsen_filter", i)
            trace = clean_chunk.copy()
            del clean_chunk
        else:
            # skip DN cleaner; pass raw chunk through
            log_filter(trace, trace, "dick_nielsen_filter (skipped)", i)
        gc.collect()
               
        # Pre decimal sort #
        trace = trace.sort_values(sort_cols, kind="mergesort", ignore_index=True)
       
        # Filter 2: Decimal Correction 
        if f["decimal_shift_corrector"]:
            _ds_defaults = dict(
                id_col="cusip_id",
                date_col="trd_exctn_dt",
                time_col="trd_exctn_tm",
                price_col="rptd_pr",
                factors=(0.1, 0.01, 10.0, 100.0),
                tol_pct_good=0.02,
                tol_abs_good=8.0,
                tol_pct_bad=0.05,
                low_pr=5.0,
                high_pr=300.0,
                anchor="rolling",
                window=5,
                improvement_frac=0.2,
                par_snap=True,
                par_band=15.0,
                output_type="cleaned",
            )
            _ds = {**_ds_defaults, **(ds_params or {})}
        
            trace, n_rows_replaced, replace_cusips = decimal_shift_corrector(
                trace,**_ds
            )
            if replace_cusips:
                dec_shift_cusips_all.extend([str(c) for c in replace_cusips])
        
            log_filter(trace, trace, "decimal_shift", i, replace=True, n_rows_replaced=n_rows_replaced)
        else:
            log_filter(trace, trace, "decimal_shift (skipped)", i, replace=True, n_rows_replaced=0)
        gc.collect()
                       
        # Filter 3: Trading Time                
        if f["trading_time"]:
            before_time = trace
            trace = filter_by_trade_time(
                df=trace,
                trade_times=trade_times,
                time_col="trd_exctn_tm",
                keep_missing=False,
            )
            log_filter(before_time, trace, "trading_time_filter", i)
            del before_time
            gc.collect()
        else:
            log_filter(trace, trace, "trading_time_filter (skipped)", i)

        # Filter 4: Trading Calendar       
        if f["trading_calendar"]:
            before_calr = trace
            trace = filter_by_calendar(
                df=trace,
                calendar_name=calendar_name,
                date_col="trd_exctn_dt",
                start_date="2002-07-01",
                end_date=None,
                keep_missing=False,
            )
            log_filter(before_calr, trace, "calendar_filter", i)
            del before_calr
            gc.collect()
        else:
            log_filter(trace, trace, "calendar_filter (skipped)", i)

        
        # Filter 5: Prices                      
        if f["price_filters"]:
            trace = filter_with_log(trace, trace['rptd_pr'] > 0,     "neg_price_filter",   i)
            trace = filter_with_log(trace, trace['rptd_pr'] <= 1000, "large_price_filter", i)
        else:
            log_filter(trace, trace, "neg_price_filter (skipped)",   i)
            log_filter(trace, trace, "large_price_filter (skipped)", i)

        # Filter 6: Trading Volume    
        # Compute dollar volume
        # entrd_vol_qt is in DOLLARS
        # https://wrds-www.wharton.upenn.edu/documents/1241/TRACE_Enhanced_Corporate_and_Agency_Historic_Data_File_Layout_post_2_6_12_10252024v.pdf
        # Applies PRE and POST 2012, see:
        # https://wrds-www.wharton.upenn.edu/documents/1240/TRACE_Enhanced_Corporate_and_Agency_Historic_Data_File_Layout_pre_2_6_12_09092021v.pdf            
        trace['dollar_vol'] = (trace['entrd_vol_qt'] * trace['rptd_pr'] / 100)  # always compute
        
        if f["volume_filter_toggle"]:
            vkind, vthr = _normalize_volume_filter(volume_filter)  # NEW
            if vkind == "dollar":
                mask = trace['dollar_vol'] >= vthr
                stage_name = "volume_filter[dollar]"
            else:  # "par"
                # keep rows with entered volume (par units) >= threshold
                mask = trace['entrd_vol_qt'] >= vthr
                stage_name = "volume_filter[par]"
        
            trace = filter_with_log(trace, mask, stage_name, i)
        else:
            log_filter(trace, trace, "volume_filter (skipped)", i)
       
        
        # Pre BB sort #
        trace = trace.sort_values(sort_cols, kind="mergesort", ignore_index=True)
        
        # Filter 7: Bounce-Back
        if f["bounce_back_filter"]:            
            _bb_defaults = dict(
                id_col="cusip_id",
                date_col="trd_exctn_dt",
                time_col="trd_exctn_tm",
                price_col="rptd_pr",
                threshold_abs=35.0,
                lookahead=5,
                max_span=5,
                window=5,
                back_to_anchor_tol=0.25,
                candidate_slack_abs=1.0,
                reassignment_margin_abs=5.0,
                use_unique_trailing_median=True,
                par_spike_heuristic=True,
                par_level=100.0,
                par_equal_tol=1e-8,
                par_min_run=3,
                par_cooldown_after_flag=2,
            )
            _bb = {**_bb_defaults, **(bb_params or {})}
            trace_bb = flag_price_change_errors(
                trace,
                **_bb
            )
            
            # Collect BB CUSIPs (audit/export only)
            bb_cusips = (
                trace_bb.loc[trace_bb.get("filtered_error", 0).eq(1), "cusip_id"]
                        .astype(str).str.strip().unique().tolist()
            )
                    
            if bb_cusips:
                bb_cusips_all.extend([str(c) for c in bb_cusips])
            trace = filter_with_log(trace_bb, trace_bb['filtered_error'] == 0, "bounce_back_filter", i)
            trace.drop(['delta_rptd_pr', 'baseline_trailing', 'filtered_error'], inplace=True, axis = 1)
            gc.collect()
        else:
            log_filter(trace, trace, "bounce_back_filter (skipped)", i)
        gc.collect()    
                        
        # Filter 8: Yield != Price
        if f["yld_price_filter"]:
            mask = (trace["rptd_pr"] != trace["yld_pt"]) | trace["yld_pt"].isna()
            trace = filter_with_log(trace, mask, "price_yld_filter", i)
        else:
            log_filter(trace, trace, "price_yld_filter (skipped)", i)

        # Filter 9: Amount-outstanding vs. volume filter            
        trace = trace.merge(fisd_off, how="left", left_on='cusip_id', right_on='cusip_id')
        if f["amtout_volume_filter"]:
            trace = filter_with_log(
                trace,
                trace['entrd_vol_qt'] < trace['offering_amt']*1000*0.50,
                "volume_offamt_filter",
                i
            )
        else:
            log_filter(trace, trace, "volume_offamt_filter (skipped)", i)

        # Filter 10: Execution date cannot exceed maturity
        if f["trd_exe_mat_filter"]:
            trace = filter_with_log(
                trace,
                trace['trd_exctn_dt'] <= trace['maturity'],
                "exctn_mat_dt_filter",
                i
            )
        else:
            log_filter(trace, trace, "exctn_mat_dt_filter (skipped)", i)

        # Filter 11: Initial Price Errors
        if f["flag_initial_price_errors"]:
            _ie_defaults = dict(
                id_col="cusip_id",
                date_col="trd_exctn_dt",
                price_col="rptd_pr",
                abs_change=50.0,
                n_transactions=3,
            )
            _ie = {**_ie_defaults, **(init_error_params or {})}
            trace_ie = flag_initial_price_errors(trace, **_ie)

            # Collect Init Price Error CUSIPs (audit/export only)
            init_price_cusips = (
                trace_ie.loc[trace_ie.get("initial_error_flag", 0).eq(1), "cusip_id"]
                        .astype(str).str.strip().unique().tolist()
            )

            if init_price_cusips:
                init_price_cusips_all.extend([str(c) for c in init_price_cusips])
            trace = filter_with_log(trace_ie, trace_ie['initial_error_flag'] == 0, "init_price_error_filter", i)
            trace.drop(['initial_error_flag'], inplace=True, axis=1)
            gc.collect()
        else:
            log_filter(trace, trace, "init_price_error_filter (skipped)", i)
        gc.collect()

        #* ************************************** */
        #* DAILY AGGREGATION                      */
        #* ************************************** */ 
     
        # Calculate all metrics with a single function call
        AllData   = compute_trace_all_metrics(trace)
        
        # Free memory from the trace object and other large dataframes
        del trace
        
        # Run garbage collection to reclaim memory
        gc.collect()
                
        all_super_list.append(AllData)
               
        elapsed_time = round(time.time() - start_time, 2)
        logging.info(f"Chunk {i+1}: took {elapsed_time} seconds")
        logging.info("-" * 50)  
            
    if all_super_list:
        final_df = pd.concat(all_super_list, ignore_index=True)
        return final_df, bb_cusips_all, dec_shift_cusips_all, init_price_cusips_all
    else:
        return pd.DataFrame(), bb_cusips_all, dec_shift_cusips_all, init_price_cusips_all
# -------------------------------------------------------------------------
def decimal_shift_corrector(
    df: pd.DataFrame,
    *,
    id_col: str = "cusip_id",
    date_col: str = "trd_exctn_dt",
    time_col: str | None = "trd_exctn_tm",
    price_col: str = "rptd_pr",
    factors=(0.1, 0.01, 10.0, 100.0),
    tol_pct_good: float = 0.02,
    tol_abs_good: float = 8.0,
    tol_pct_bad: float = 0.05,
    low_pr: float = 5.0,
    high_pr: float = 300.0,
    anchor: str = "rolling",
    window: int = 5,
    improvement_frac: float = 0.2,
    par_snap: bool = True,
    par_band: float = 15.0,
    output_type: str = "uncleaned"
):
        
    """
    Detect and (optionally) correct decimal-shift price errors within each CUSIP's
    time series by testing multiplicative scale factors against a robust anchor
    (rolling unique-median). A candidate replacement is accepted only if it brings
    the observation much closer to the anchor and passes absolute/relative gates.

    Parameters
    ----------
    df : pandas.DataFrame
        Input panel with at least [id_col, date_col, price_col]; time_col is optional.
    id_col : str, default "cusip_id"
        Bond identifier column.
    date_col : str, default "trd_exctn_dt"
        Trade date column (used for sorting and grouping).
    time_col : str | None, default "trd_exctn_tm"
        Optional trade time column (included in sort if present in `df`).
    price_col : str, default "rptd_pr"
        Price column to evaluate and potentially correct.
    factors : iterable of float, default (0.1, 0.01, 10.0, 100.0)
        Candidate decimal-shift multipliers to test against each observation.
    tol_pct_good : float, default 0.02
        Relative error threshold for accepting a corrected price (e.g., 2%).
    tol_abs_good : float, default 8.0
        Absolute distance threshold (price points) for acceptance.
    tol_pct_bad : float, default 0.05
        Minimum raw relative error needed to consider a decimal-shift (e.g., 5%).
    low_pr, high_pr : float, defaults 5.0 and 300.0
        Plausible price bounds; help gate clearly implausible observations.
    anchor : str, default "rolling"
        Anchor type. Currently supports "rolling" (rolling unique-median).
    window : int, default 5
        Rolling half-window size for the anchor (effective window = 2*window+1).
    improvement_frac : float, default 0.2
        Required proportional improvement vs raw relative error (e.g., 20%).
    par_snap : bool, default True
        Enable relaxed acceptance for observations near par=100.
    par_band : float, default 15.0
        Par proximity band (|price-100| <= par_band) for the par snap rule.
    output_type : {"uncleaned","cleaned"}, default "uncleaned"
        - "uncleaned": Return the input frame (sorted) with three added columns:
            * dec_shift_flag   (int8)    1 if corrected candidate accepted
            * dec_shift_factor (float)   chosen factor (1.0 if no change)
            * suggested_price  (float)   corrected price proposal
        - "cleaned": Apply `suggested_price` where flagged and return a triplet:
            (cleaned_df, n_corrected, affected_cusips).

    Returns
    -------
    If output_type == "uncleaned":
        pandas.DataFrame
            Sorted copy of `df` with added columns:
            ["dec_shift_flag", "dec_shift_factor", "suggested_price"].
    If output_type == "cleaned":
        tuple[pandas.DataFrame, int, list[str]]
            cleaned_df :
                Copy of `df` with `price_col` overwritten where flagged.
            n_corrected :
                Count of rows where a correction was applied.
            affected_cusips :
                Sorted unique list of CUSIPs with at least one correction.

    Notes
    -----
    - Sorting is by [id_col, date_col] and includes time_col if present in `df`.
    - The rolling anchor uses unique values to reduce the impact of rapid repeats.
    - Choose `output_type="uncleaned"` for audit/debugging; use "cleaned" to
      directly obtain a corrected price series.
    """
    
    eps = 1e-12
    original_cols = list(df.columns)
    out = df.copy()
    
    def _rolling_meds_series(s: pd.Series, w: int) -> pd.DataFrame:
        s = s.astype(float)
        med_center = s.rolling(window=2*w+1, center=True, min_periods=w+1).median()
        med_fwd    = s[::-1].rolling(window=w+1, min_periods=1).median()[::-1]
        med_back   = s.rolling(window=w+1, min_periods=1).median()
        return pd.DataFrame({
            "anchor_med_center": med_center,
            "anchor_med_fwd":    med_fwd,
            "anchor_med_back":   med_back,
        }, index=s.index).astype(float)

    def _compose_anchor(df_meds: pd.DataFrame, s: pd.Series) -> pd.Series:
        anchor_s = df_meds["anchor_med_center"].copy()
        na = anchor_s.isna()
        if na.any():
            anchor_s[na] = df_meds.loc[na, "anchor_med_fwd"]
        na = anchor_s.isna()
        if na.any():
            anchor_s[na] = df_meds.loc[na, "anchor_med_back"]
        if anchor_s.isna().any():
            anchor_s = anchor_s.fillna(float(np.nanmedian(s.astype(float))))
        return anchor_s.astype(float)

    if anchor == "rolling":
        work = out.drop_duplicates(subset=[id_col, date_col, price_col], keep="first").copy()
        meds = (
            work.groupby(id_col, observed=True)[price_col]
                .apply(lambda s: _rolling_meds_series(s, window))
        )
        if isinstance(meds.index, pd.MultiIndex):
            meds.index = meds.index.droplevel(0)
        work = work.join(meds, how="left")
        work["anchor_price_calc"] = _compose_anchor(
            work[["anchor_med_center", "anchor_med_fwd", "anchor_med_back"]], work[price_col]
        )
        merge_cols = [id_col, date_col, price_col,
                      "anchor_price_calc", "anchor_med_center", "anchor_med_fwd", "anchor_med_back"]
        out = out.merge(
            work[merge_cols].rename(columns={"anchor_price_calc": "anchor_price"}),
            on=[id_col, date_col, price_col],
            how="left",
            validate="m:1"
        )
        na_mask = out["anchor_price"].isna()
        if na_mask.any():
            out.loc[na_mask, "anchor_price"] = (
                out.groupby([id_col, date_col])[price_col].transform("median")
            ).astype(float)[na_mask]
    else:
        out["anchor_price"] = (
            out.groupby([id_col, date_col], observed=True)[price_col].transform("median").astype(float)
        )

    price  = out[price_col].astype(float)
    anchor_vals = out["anchor_price"].astype(float)
    raw_rel = (price.sub(anchor_vals).abs() / anchor_vals).replace([np.inf, -np.inf], np.nan)

    best_relerr = pd.Series(np.nan, index=out.index, dtype="float64")
    best_factor = pd.Series(np.nan, index=out.index, dtype="float64")
    best_price  = pd.Series(np.nan, index=out.index, dtype="float64")

    for f in factors:
        cand_price = price * f
        plausible  = (cand_price >= low_pr) & (cand_price <= high_pr)
        relerr     = ((cand_price - anchor_vals).abs() / anchor_vals).where(plausible, np.nan)
        take = relerr.notna() & (best_relerr.isna() | (relerr < best_relerr))
        best_relerr = best_relerr.where(~take, relerr)
        best_factor = best_factor.where(~take, f)
        best_price  = best_price.where(~take, cand_price)

    abs_good = (best_price.sub(anchor_vals).abs() <= tol_abs_good + eps)
    near_par_anchor = anchor_vals.sub(100.0).abs() <= par_band
    near_par_best   = best_price.sub(100.0).abs() <= par_band

    par_ok = (near_par_anchor & near_par_best) if par_snap else pd.Series(False, index=out.index)

    dec_flag = (
        (raw_rel > tol_pct_bad - eps) &
        (
            (best_relerr <= tol_pct_good + eps) |
            abs_good |
            par_ok
        ) &
        (best_relerr <= improvement_frac * raw_rel + eps)
    ).astype("int8")

    out["dec_shift_flag"]   = dec_flag
    out["dec_shift_factor"] = np.where(out["dec_shift_flag"].eq(1), best_factor, 1.0)
    out["suggested_price"]  = np.where(out["dec_shift_flag"].eq(1), best_price, price)

    if output_type.lower() == "uncleaned":
        out["suggested_price"]  = out["suggested_price"].astype(float)
        out["dec_shift_factor"] = out["dec_shift_factor"].astype(float)
        return out

    corrected = out.copy()
    n_corrected = int(corrected["dec_shift_flag"].sum())
    mask = corrected["dec_shift_flag"].eq(1)
    corrected.loc[mask, price_col] = corrected.loc[mask, "suggested_price"].values
    cleaned_df = corrected[original_cols].copy()

    # QoL addition: list of unique cusips affected (no logic change)
    affected_cusips = sorted(cleaned_df.loc[mask, id_col].dropna().astype(str).unique().tolist())

    # Return triplet for output_type="cleaned"
    return cleaned_df, n_corrected, affected_cusips
# -------------------------------------------------------------------------
def flag_price_change_errors(
    df: pd.DataFrame,
    *,
    id_col: str = "cusip_id",
    date_col: str = "trd_exctn_dt",
    time_col: Optional[str] = "trd_exctn_tm",
    price_col: str = "rptd_pr",
    threshold_abs: float = 35.0,
    lookahead: int = 5,
    max_span: int = 5,
    window: int = 5,
    back_to_anchor_tol: float = 0.25,
    candidate_slack_abs: float = 1.0,
    reassignment_margin_abs: float = 5.0,
    use_unique_trailing_median: bool = True,
    par_spike_heuristic: bool = True,
    par_level: float = 100.0,
    par_equal_tol: float = 1e-8,
    par_min_run: int = 3,
    par_cooldown_after_flag: int = 2,
) -> pd.DataFrame:
    """
    Flag likely price-entry errors using a bounce-back logic around large changes
    relative to a backward-looking robust anchor. Designed for intraday TRACE-like
    panels and robust to repeated  transactions and par-level plateaus.

    Core idea
    ---------
    A candidate  transaction price error is a large one-step price change (absolute change greater
    than or equal to threshold_abs) that is followed, within a limited number of
    rows, by an opposite-signed move that returns part of the way toward a
    trailing anchor. Path length is capped by max_span. The decision uses a
    backward-looking anchor (unique-median option), a small slack around the
    anchor, and a back-to-anchor consistency check.

    Workflow (per id, time-sorted)
    ------------------------------
    1) Sort by [id_col, date_col] and include time_col when present.
    2) Build a strictly backward-looking anchor:
       - If use_unique_trailing_median is True, use a trailing unique median
         with window = window (effective 1..window rows back).
    3) Open a candidate when the absolute one-step price change is large
       (greater than or equal to threshold_abs) and the price is sufficiently
       displaced from the anchor (candidate_slack_abs).
    4) Bounce-back gate:
       - Search forward up to lookahead rows (and total path length no more
         than max_span) for an opposite-signed move that returns toward the
         anchor by at least back_to_anchor_tol times the pre-jump displacement.
    5) Reassignment margin:
       - Prefer flags where the chosen tick is more extreme than nearby
         alternatives by at least reassignment_margin_abs to avoid flagging
         the wrong row in multi-move sequences.
    6) Par-specific heuristic (optional):
       - If par_spike_heuristic is True, apply special handling for prices
         at or near par_level (within par_equal_tol), and avoid flagging
         short par-only runs where the run length is less than par_min_run.
    7) Cooldown:
       - After a flag, suppress further flags for the next
         par_cooldown_after_flag rows within the same id group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input panel with at least [id_col, date_col, price_col]. time_col is
        optional but recommended for intraday ordering.
    id_col : str, default "cusip_id"
        Security identifier column.
    date_col : str, default "trd_exctn_dt"
        Trade date column used for sorting and grouping.
    time_col : str or None, default "trd_exctn_tm"
        Optional trade time column; used in sorting if present in df.
    price_col : str, default "rptd_pr"
        Price column evaluated by the filter.
    threshold_abs : float, default 35.0
        Minimum absolute one-step price change that opens a candidate.
    lookahead : int, default 5
        Maximum number of rows ahead to search for the bounce.
    max_span : int, default 5
        Maximum total path length from candidate start to resolution.
    window : int, default 5
        Backward window length for the trailing median anchor.
    back_to_anchor_tol : float, default 0.25
        Fraction of the initial displacement that must be recovered toward
        the anchor to count as a bounce-back.
    candidate_slack_abs : float, default 1.0
        Small absolute slack around the anchor when opening a candidate.
    reassignment_margin_abs : float, default 5.0
        Tie-break margin to decide which tick to flag in multi-move clusters.
    use_unique_trailing_median : bool, default True
        If True, compute the anchor using unique values to reduce duplicate-print bias.
    par_spike_heuristic : bool, default True
        Enable special handling near par-level prints.
    par_level : float, default 100.0
        Numerical par level used by the heuristic.
    par_equal_tol : float, default 1e-8
        Absolute tolerance to treat a price as exactly par_level.
    par_min_run : int, default 3
        Minimum length of a contiguous par-only run to be considered a par block.
    par_cooldown_after_flag : int, default 2
        Number of subsequent rows to skip from flagging after a flag is issued.

    Returns
    -------
    pandas.DataFrame
        A sorted copy of df with at least one added column:
          - filtered_error (int8): 1 if the row is flagged as an error, else 0
        Implementations may add diagnostics such as deltas and anchors.

    Notes
    -----
    - Grouping is performed internally by id_col; ensure each group has at least
      two rows.
    - The anchor is strictly backward-looking to avoid look-ahead bias.
    - Typical usage is within a groupby-apply over ids, followed by aggregation
      or export steps that drop flagged rows or track flagged ids.
    """
    
    eps = 1e-12

    def rolling_unique_median(series: pd.Series, window: int) -> pd.Series:
        def uniq_med(x):
            x = x[~np.isnan(x)]
            if x.size == 0:
                return np.nan
            return float(np.median(np.unique(x)))
        # preserve the original index
        s = series.astype(float)
        out = pd.Series(s.to_numpy(), index=s.index)\
                .rolling(window=window, min_periods=1)\
                .apply(uniq_med, raw=True)
        return out.shift(1).astype(float)

    out = df.copy()

    # Differences
    out["delta_rptd_pr"] = out.groupby(id_col, observed=True)[price_col].diff().astype(float)

    # Build backward-looking baseline
    if use_unique_trailing_median:
        out["baseline_trailing"] = (
            out.groupby(id_col, observed=True)[price_col]
               .transform(lambda s: rolling_unique_median(s, window=window+1))
        )
    else:
        out["baseline_trailing"] = (
            out.groupby(id_col, observed=True)[price_col]
               .transform(lambda s: s.rolling(window=window+1, min_periods=1).median())
               .shift(1)
               .astype(float)
        )

    n = len(out)
    filtered = np.zeros(n, dtype=np.int8)
    thr_lo = max(0.0, threshold_abs - float(candidate_slack_abs))
    back_tol_abs = back_to_anchor_tol * threshold_abs

    # Main scan per id
    for _, gidx in out.groupby(id_col, observed=True).groups.items():
        idxs = np.asarray(gidx)
        P = out.loc[idxs, price_col].to_numpy(float)
        D = out.loc[idxs, "delta_rptd_pr"].to_numpy(float)
        B = out.loc[idxs, "baseline_trailing"].to_numpy(float)

        i = 0
        m = len(idxs)
        par_cooldown_until = -1  # local index; skip non-par flags until this index after a par-run
        while i < m:
            # If within cooldown and current is non-par, skip any new non-par flags
            if i <= par_cooldown_until and (abs(P[i] - par_level) > par_equal_tol):
                i += 1
                continue

            cond_jump     = (not np.isnan(D[i])) and (abs(D[i])        >= thr_lo - eps)
            cond_far_prev = (not np.isnan(B[i])) and (abs(P[i] - B[i]) >= thr_lo - eps)

            cond_par = False
            if par_spike_heuristic and not np.isnan(P[i]) and abs(P[i] - par_level) <= par_equal_tol:
                if (not np.isnan(B[i])) and (abs(P[i] - B[i]) >= back_tol_abs - eps):
                    cond_par = True

            par_only = cond_par and not cond_jump  # triggered by par heuristic but not by big jump

            if cond_jump or cond_far_prev or cond_par:
                j_lim    = min(m - 1, i + lookahead)
                j_match  = None
                k_return = None

                # IMPORTANT: If par-only, we *do not* use quick-correction path;
                # only persistent par-run can cause flags.
                if not par_only:
                    for j in range(i + 1, j_lim + 1):
                        # Opposite big move
                        if (not np.isnan(D[i])) and (not np.isnan(D[j])) and (np.sign(D[j]) == -np.sign(D[i])) and (abs(D[j]) >= thr_lo - eps):
                            j_match = j
                            break
                        # Return to the pre-move baseline (at i)
                        if not np.isnan(B[i]) and (abs(P[j] - B[i]) <= back_tol_abs + eps):
                            k_return = j
                            break

                par_start = cond_par

                # Standard quick-correction case (not available for par-only)
                if (not par_only) and ((j_match is not None) or (k_return is not None)):
                    stop_at = j_match if j_match is not None else k_return
                    flag_start = i

                    # Blame reassignment if the prior row deviates more from *its* baseline
                    prev = i - 1
                    if prev >= 0:
                        B_prev = B[prev]
                        dev_prev = abs(P[prev] - B_prev) if not np.isnan(B_prev) else np.nan
                        dev_curr = abs(P[i]   - B[i])    if not np.isnan(B[i])    else np.nan
                        if (not np.isnan(dev_prev)) and (not np.isnan(dev_curr)):
                            if (dev_prev - dev_curr) >= reassignment_margin_abs - eps and (dev_prev >= back_tol_abs - eps):
                                flag_start = prev

                    # Flag the start
                    if (not par_start) or abs(P[flag_start] - par_level) <= par_equal_tol:
                        filtered[idxs[flag_start]] = 1

                    # Extend plateau flags until stop_at, respecting par/non-par logic
                    B_start = B[flag_start]
                    span_end = min(stop_at, flag_start + max_span)
                    for k in range(flag_start + 1, span_end + 1):
                        if par_start:
                            if abs(P[k] - par_level) <= par_equal_tol:
                                filtered[idxs[k]] = 1
                        else:
                            if not np.isnan(B_start) and abs(P[k] - B_start) >= back_tol_abs - eps:
                                filtered[idxs[k]] = 1
                            else:
                                break

                    if par_start:
                        par_cooldown_until = max(par_cooldown_until, stop_at + par_cooldown_after_flag)

                    i = stop_at + 1
                    continue

                # Persistent par block (with no quick-correction): require run_len >= par_min_run
                if par_start:
                    run_end = i
                    while run_end + 1 < m and abs(P[run_end + 1] - par_level) <= par_equal_tol:
                        run_end += 1
                    run_len = run_end - i + 1
                    if run_len >= par_min_run:
                        for k in range(i, run_end + 1):
                            filtered[idxs[k]] = 1
                        par_cooldown_until = max(par_cooldown_until, run_end + par_cooldown_after_flag)
                        i = run_end + 1
                        continue

            i += 1

    out["filtered_error"] = filtered.astype(np.int8)
    return out
# -------------------------------------------------------------------------
def flag_initial_price_errors(
    df: pd.DataFrame,
    *,
    id_col: str = "cusip_id",
    date_col: str = "trd_exctn_dt",
    price_col: str = "rptd_pr",
    abs_change: float = 50.00,
    n_transactions: int = 3
) -> pd.DataFrame:
    """
    Flag erroneous initial prices that jump to correct levels.

    Logic:
    ------
    For each CUSIP:
    1. Examine first n_transactions (default: 3)
    2. Find first large price jump (abs change > abs_change)
    3. Flag all rows BEFORE the jump as errors

    Example:
    --------
    CUSIP has prices: 0.360, 0.403, 106.000, 106.500
    - Rows 0-1: prices are 0.360, 0.403 (suspiciously low)
    - Row 2: price jumps to 106.000 (jump = 105.597 > 50)
    - Flag rows 0-1 as errors, keep row 2 onwards

    Rationale:
    - Prices before a large jump are likely data entry errors
    - Price after the jump is likely correct
    - Only checks first n_transactions to avoid false positives

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with bond price data
    id_col : str, default "cusip_id"
        Column name for bond identifier
    date_col : str, default "trd_exctn_dt"
        Column name for date
    price_col : str, default "rptd_pr"
        Column name for price (in % of par)
    abs_change : float, default 50.0
        Minimum absolute price change to trigger flagging
        Default 50.0 = $50 change for a $1000 par bond
    n_transactions : int, default 3
        Number of initial transactions to scan for each CUSIP

    Returns
    -------
    pd.DataFrame
        Copy of input with added 'initial_error_flag' column:
        - 0: Keep observation (no initial error detected)
        - 1: Flag for removal (erroneous initial price)

    Notes
    -----
    - Only examines first n_transactions for each CUSIP
    - Flags all rows BEFORE the first large jump
    - Fast implementation using groupby + numpy arrays
    """
    out = df.copy()

    # Sort by CUSIP and date
    out = out.sort_values([id_col, date_col]).reset_index(drop=True)

    # Calculate price differences within each CUSIP (vectorized)
    out['_price_diff'] = out.groupby(id_col, observed=True)[price_col].diff().abs()

    # Initialize flag array
    n = len(out)
    flagged = np.zeros(n, dtype=np.int8)

    # Process each CUSIP group
    for _, gidx in out.groupby(id_col, observed=True).groups.items():
        idxs = np.asarray(gidx)
        m = len(idxs)

        # Only scan first n_transactions
        scan_len = min(m, n_transactions)

        if scan_len > 1:
            # Get price differences for first n transactions
            diffs = out.loc[idxs[:scan_len], '_price_diff'].to_numpy(float)

            # Find first large jump (skip position 0 which has NaN diff)
            for i in range(1, scan_len):
                if not np.isnan(diffs[i]) and diffs[i] > abs_change:
                    # Flag all positions BEFORE this jump
                    flagged[idxs[:i]] = 1
                    break  # Only process first large jump

    # Add flag column
    out['initial_error_flag'] = flagged

    # Drop temporary column
    out = out.drop(columns=['_price_diff'])

    return out
# -------------------------------------------------------------------------
def _hms_to_seconds(x: str) -> float:
    """
    Convert an HH:MM:SS string (zero-padded or not) to seconds since midnight.
    Returns np.nan on parse failure.
    """
    try:
        s = str(x).strip()
        if not s:
            return np.nan
        parts = s.split(":")
        if len(parts) != 3:
            return np.nan
        h = int(parts[0])  # accepts "4" or "04"
        m = int(parts[1])
        sec = float(parts[2])  # allow "22" or "22.0"
        if not (0 <= h <= 23 and 0 <= m <= 59 and 0.0 <= sec < 60.0):
            return np.nan
        return h * 3600 + m * 60 + sec
    except Exception:
        return np.nan


def filter_by_trade_time(
    df: pd.DataFrame,
    trade_times: list[str] | tuple[str, str] | None,
    time_col: str = "trd_exctn_tm",
    keep_missing: bool = False,           
) -> pd.DataFrame:
    
    if not trade_times or len(trade_times) != 2:
        return df

    start_s = _hms_to_seconds(trade_times[0])
    end_s   = _hms_to_seconds(trade_times[1])

    # If either bound is invalid, do nothing
    if np.isnan(start_s) or np.isnan(end_s):
        return df

    tsec = df[time_col].astype(str).map(_hms_to_seconds)
    valid = ~tsec.isna()

    if end_s >= start_s:
        in_win = (tsec >= start_s) & (tsec <= end_s)
    else:
        # Wrap-around: keep t >= start OR t <= end
        in_win = (tsec >= start_s) | (tsec <= end_s)

    if keep_missing:
        mask = in_win | ~valid
    else:
        mask = valid & in_win
                    
    return df.loc[mask].copy()
# -------------------------------------------------------------------------
def add_filter_flags(group):
    # Calculate logarithmic price changes within this CUSIP group
    group['log_price']        = np.log(group['rptd_pr'])
    group['log_price_change'] = group['log_price'].diff()
    
    # Calculate the product of consecutive log price changes
    group['next_log_price_change'] = group['log_price_change'].shift(-1)
    group['log_price_change_product'] = group['log_price_change'] * group['next_log_price_change']
    
    # Filter out rows where the condition is met, but keep NaN values
    filtered_group = group[(group['log_price_change_product'] > -0.25) |\
                           (pd.isna(group['log_price_change_product']))]
    
    # Drop the temporary columns we created
    columns_to_drop = ['log_price', 'log_price_change', 
                       'next_log_price_change', 'log_price_change_product']
    filtered_group = filtered_group.drop(columns=columns_to_drop)
    
    return filtered_group
# -------------------------------------------------------------------------
def filter_by_calendar(
    df: pd.DataFrame,
    calendar_name: str | None,
    date_col: str = "trd_exctn_dt",
    start_date: str = "2002-07-01",
    end_date: str | None = None,
    keep_missing: bool = False,            
) -> pd.DataFrame:
    """
    Keep only rows whose date_col is a valid session date in the selected
    pandas_market_calendars calendar. Inclusive check.

    Parameters
    ----------
    calendar_name : str or None
        Example: "NYSE". If None or empty, returns df unchanged.
    date_col : str
        Column containing trade dates. Will be parsed with pandas.to_datetime.
    start_date : str
        Lower bound for calendar schedule construction. Per your spec this is fixed.
    end_date : str or None
        Upper bound for calendar schedule construction. If None, uses today().
    keep_missing : bool
        If True, keep rows with missing/unparsable dates. If False, drop them.

    Returns
    -------
    A filtered copy of df.
    """
    if not calendar_name:
        return df

    try:
        import pandas_market_calendars as mcal
    except Exception as e:
        raise RuntimeError(
            "pandas_market_calendars is required for filter_by_calendar but is not available."
        ) from e

    # End date defaults to today (no hardcoding)
    if end_date is None:
        end_date = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    # Build schedule and set of valid session dates (date-only)
    cal = mcal.get_calendar(calendar_name)
    sched = cal.schedule(start_date=start_date, end_date=end_date)
    # Normalize to date (no time, no tz)
    valid_dates = pd.Index(sched.index.tz_localize(None).normalize().date)

    # Parse input date column to date-only
    d_parsed = pd.to_datetime(df[date_col], errors="coerce").dt.normalize().dt.date
    is_valid = d_parsed.isin(valid_dates)
    has_date = d_parsed.notna()

    if keep_missing:
        mask = is_valid | (~has_date)
    else:
        mask = has_date & is_valid
                    
    return df.loc[mask].copy()
# -------------------------------------------------------------------------
def clean_trace_chunk(trace, *, chunk_id=None, clean_agency=True, logger=None):
    """
    Clean one chunk of Enhanced TRACE trades and returns a filtered DataFrame.
    This routine applies date cutoffs, status and condition filters, duplicate
    handling, and optional agency de-duplication. It is designed to be called
    inside a loop over CUSIP chunks prior to any decimal-shift or bounce-back
    logic.

    What it does
    ------------
    - Ensures datetime types for report and execution timestamps.
    - Drops rows with missing or empty cusip_id.
    - Splits the data at the regulatory change on 2012-02-06:
      * pre segment: applies legacy filters and explicit cancel and correction handling.
      * post segment: applies the updated filters using the newer status fields.
    - Reconciles cancels and corrections within the pre segment using key-based matching.
    - Optionally removes agency duplicate prints if clean_agency is True.
    - Re-combines pre and post segments and returns the cleaned result.

    Parameters
    ----------
    trace : pandas.DataFrame
        Raw Enhanced TRACE rows for a set of CUSIPs. Expected columns follow the
        standard TRACE schema, for example:
        cusip_id, trd_exctn_dt, trd_exctn_tm, trd_rpt_dt, trd_rpt_tm,
        msg_seq_nb, orig_msg_seq_nb, trc_st, asof_cd, sale_cndtn_cd,
        lckd_in_ind, wis_fl, rpt_side_cd, entrd_vol_qt, rptd_pr, yld_pt.
        Additional columns are passed through unchanged.
    chunk_id : int or None, optional
        Identifier used in logs and audit trails for this chunk.
    clean_agency : bool, default True
        If True, apply the agency de-duplication pass at the end.
    logger : callable or None, optional
        Function used to append one audit row per stage. It will be called as:
            logger(df_before, df_after, stage="stage_name", chunk_id=chunk_id)
        Suggested stage names include:
            "base_filters", "pre_rules", "post_rules",
            "cancels_corrections", "agency_cleaning".

    Returns
    -------
    pandas.DataFrame
        Cleaned TRACE rows for the input chunk. The return is a new DataFrame
        and the input is not modified in place.

    Notes
    -----
    - Sorting and index are not guaranteed; callers commonly re-sort by
      cusip_id, trd_exctn_dt, trd_exctn_tm after downstream merges.
    - If the input is empty, an empty DataFrame is returned.
    - The exact list of filters mirrors the commonly used academic cleaning
      conventions around the 2012-02-06 transition date.
    """

    # Convert date strings to datetime objects
    cutoff_date = pd.to_datetime('2012-02-06')
    
    trace['trd_exctn_dt'] = pd.to_datetime(trace['trd_exctn_dt'])
    trace['trd_rpt_dt'] = pd.to_datetime(trace['trd_rpt_dt'])
        
    # Split data into pre and post 2012/02/06 based on reporting date
    post = trace[trace['trd_rpt_dt'] >= cutoff_date].copy()
    pre  = trace[trace['trd_rpt_dt'] < cutoff_date].copy()
    
    # Apply additional filters for pre-2012-02-06 data
    if not pre.empty:
        # Convert indicators to string
        pre['days_to_sttl_ct'] = pre['days_to_sttl_ct'].astype(str)
        pre['wis_fl'] = pre['wis_fl'].astype(str)
        pre['lckd_in_ind'] = pre['lckd_in_ind'].astype(str)
        pre['sale_cndtn_cd'] = pre['sale_cndtn_cd'].astype(str)
        
        # Replace NaN strings with 'None'
        pre['days_to_sttl_ct'] = pre['days_to_sttl_ct'].replace('nan', 'None')
        pre['wis_fl'] = pre['wis_fl'].replace('nan', 'None')
        pre['lckd_in_ind'] = pre['lckd_in_ind'].replace('nan', 'None')
        pre['sale_cndtn_cd'] = pre['sale_cndtn_cd'].replace('nan', 'None')
        
        # 1) <= 2-day settlement ---------------------------------------------
        before = pre
        pre = pre[pre['days_to_sttl_ct'].isin({'000','001','002','None'})]
        if logger:
            logger(before, pre, "pre_settle_<=2d", chunk_id)
    
        # 2) Exclude when-issued (wis_fl == 'Y') -----------------------------
        before = pre
        pre = pre[pre['wis_fl'] != 'Y']
        if logger:
            logger(before, pre, "pre_exclude_WIS", chunk_id)
    
        # 3) Exclude locked-in (lckd_in_ind == 'Y') -------------------------
        before = pre
        pre = pre[pre['lckd_in_ind'] != 'Y']
        if logger:
            logger(before, pre, "pre_exclude_locked_in", chunk_id)
    
        # 4) Exclude special conditions (sale_cndtn_cd not in {None, @}) ----
        before = pre
        pre = pre[pre['sale_cndtn_cd'].isin({'None','@'})]
        if logger:
            logger(before, pre, "pre_exclude_special_cond", chunk_id)
    
    # Clean Post 2012/02/06 data
    clean_post = clean_post_20120206(post,
      chunk_id=chunk_id,logger=log_ct_filter) if not post.empty else pd.DataFrame()
    # 38628-37057
    # Clean Pre 2012/02/06 data
    clean_pre = clean_pre_20120206(pre,
      chunk_id=chunk_id,logger=log_ct_filter ) if not pre.empty else pd.DataFrame()
    
    # Combine pre and post data
    clean_combined = pd.concat([clean_pre, clean_post], ignore_index=True)
    
    # Apply agency transaction cleaning conditionally
    if clean_agency:
        clean_final = clean_agency_transactions(clean_combined) if not clean_combined.empty else pd.DataFrame()
        if logger:
            logger(clean_combined, clean_final,
                   stage="agency_cleaning", chunk_id=chunk_id)      
    else:
        clean_final = clean_combined  
    
    return clean_final
# -------------------------------------------------------------------------
def clean_post_20120206(post, chunk_id=None, logger=None):
    """
    Clean Enhanced TRACE data reported on or after February 6, 2012.
    
    Removes cancellations (C, X), corrections, and reversals (Y) from post-2012 
    TRACE Enhanced data using the updated regulatory reporting structure. This 
    function implements the two-stage cleaning procedure described in 
    Dick-Nielsen (2009, 2014) for the modern TRACE reporting system.
    
    Regulatory Context
    ------------------
    On February 6, 2012, FINRA updated the TRACE reporting system with new 
    status codes:
    - 'T': Trade Report (original transaction)
    - 'R': Trade Report (used with reversals in some contexts)
    - 'X', 'C': Cancellation and Correction (same MSG_SEQ_NB as original)
    - 'Y': Reversal (ORIG_MSG_SEQ_NB points to original trade's MSG_SEQ_NB)
    
    Cleaning Procedure
    ------------------
    **Step 1: Remove Cancellations and Corrections (C, X)**
    Matches using 8 keys:
        - cusip_id, trd_exctn_dt, trd_exctn_tm, rptd_pr, entrd_vol_qt
        - rpt_side_cd, cntra_mp_id, msg_seq_nb
    
    Key insight: C and X records show the SAME msg_seq_nb as the original 
    trade they cancel or correct. Matched T/R records are removed.
    
    **Step 2: Remove Reversals (Y)**
    Matches using 7 keys + asymmetric msg_seq_nb:
        - cusip_id, trd_exctn_dt, trd_exctn_tm, rptd_pr, entrd_vol_qt
        - rpt_side_cd, cntra_mp_id
        - clean_post1.msg_seq_nb = post_Y.orig_msg_seq_nb (asymmetric)
    
    Key insight: Y records contain orig_msg_seq_nb pointing to the original 
    trade's msg_seq_nb. This creates an asymmetric join condition. Matched 
    T/R records are removed.
    
    Parameters
    ----------
    post : pandas.DataFrame
        Enhanced TRACE records with trd_rpt_dt >= '2012-02-06'. Must contain:
        - cusip_id : str
            CUSIP identifier
        - trd_exctn_dt : datetime
            Trade execution date
        - trd_exctn_tm : str or time
            Trade execution time
        - trd_rpt_dt : datetime
            Trade report date (used for date filtering upstream)
        - rptd_pr : float
            Reported price
        - entrd_vol_qt : float or int
            Entered volume quantity
        - rpt_side_cd : str
            Reporting side code ('B' or 'S')
        - cntra_mp_id : str
            Contra-party market participant ID ('C' or 'D')
        - msg_seq_nb : str or int
            Message sequence number (unique trade identifier)
        - orig_msg_seq_nb : str or int
            Original message sequence number (for Y reversals)
        - trc_st : str
            Transaction status ('T', 'R', 'X', 'C', 'Y')
                   
    Returns
    -------
    pandas.DataFrame
        Cleaned dataset with cancellations, corrections, and reversals removed.
        Contains only valid trade reports (T/R records that were not matched
        to any C/X/Y records). Empty DataFrame returned if input is empty.
        
        The returned DataFrame:
        - Preserves all columns from the input
        - May have fewer rows due to cancellation/reversal removal
        - Is not sorted (downstream code should sort as needed)
        - Has reset index (ignore_index=True used in operations)        
    """
    
    if post.empty:
        return pd.DataFrame()
    
    # Store original for logging
    original_post = post['cusip_id'].copy()
    
    # Split data based on trc_st
    post_TR = post[post['trc_st'].isin(['T', 'R'])].copy()
    post_XC = post[post['trc_st'].isin(['X', 'C'])].copy()
    post_Y  = post[post['trc_st'] == 'Y'].copy()
        
    # Step 1.1: Remove Cancellation and Correction
    # Match Cancellation and Correction using 7 keys + msg_seq_nb:
    # cusip_id, Execution Date and Time, Quantity, Price, Buy/Sell Indicator, 
    # Contra Party, msg_seq_nb
    # C and X records show the same MSG_SEQ_NB as the original record
    if not post_XC.empty and not post_TR.empty:
        # Create a merge key for matching
        post_TR['merge_key'] = (post_TR['cusip_id'] + '_' + 
                                post_TR['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' +
                                post_TR['trd_exctn_tm'].astype(str) + '_' + 
                                post_TR['rptd_pr'].astype(str) + '_' +
                                post_TR['entrd_vol_qt'].astype(str) + '_' + 
                                post_TR['rpt_side_cd'] + '_' +
                                post_TR['cntra_mp_id'] + '_' + 
                                post_TR['msg_seq_nb'].astype(str))
        
        post_XC['merge_key'] = (post_XC['cusip_id'] + '_' + 
                                post_XC['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' +
                                post_XC['trd_exctn_tm'].astype(str) + '_' + 
                                post_XC['rptd_pr'].astype(str) + '_' +
                                post_XC['entrd_vol_qt'].astype(str) + '_' + 
                                post_XC['rpt_side_cd'] + '_' +
                                post_XC['cntra_mp_id'] + '_' + 
                                post_XC['msg_seq_nb'].astype(str))
        
        # Keep records that don't have matches in post_XC
        clean_post1 = post_TR[~post_TR['merge_key'].isin(post_XC['merge_key'])].copy()
        clean_post1.drop(columns=['merge_key'], inplace=True)
    else:
        clean_post1 = post_TR.copy()
    
    if logger:              
        logger(original_post, clean_post1, "post_cancel_corr", chunk_id)
        
    # Step 1.2: Remove Reversals
    # Match Reversal using the same 7 keys but:
    # - Use msg_seq_nb from clean_post1
    # - Use orig_msg_seq_nb from post_Y (not msg_seq_nb!)
    if not post_Y.empty and not clean_post1.empty:
        # Create merge keys 
        # For clean_post1: use msg_seq_nb
        clean_post1['merge_key'] = (clean_post1['cusip_id'] + '_' + 
                                    clean_post1['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' +
                                    clean_post1['trd_exctn_tm'].astype(str) + '_' + 
                                    clean_post1['rptd_pr'].astype(str) + '_' +
                                    clean_post1['entrd_vol_qt'].astype(str) + '_' + 
                                    clean_post1['rpt_side_cd'] + '_' +
                                    clean_post1['cntra_mp_id'] + '_' + 
                                    clean_post1['msg_seq_nb'].astype(str))  # <-- msg_seq_nb
        
        # For post_Y: use orig_msg_seq_nb (not msg_seq_nb!)
        post_Y['merge_key'] = (post_Y['cusip_id'] + '_' + 
                               post_Y['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' +
                               post_Y['trd_exctn_tm'].astype(str) + '_' + 
                               post_Y['rptd_pr'].astype(str) + '_' +
                               post_Y['entrd_vol_qt'].astype(str) + '_' + 
                               post_Y['rpt_side_cd'] + '_' +
                               post_Y['cntra_mp_id'] + '_' + 
                               post_Y['orig_msg_seq_nb'].astype(str))  # <-- orig_msg_seq_nb!
        
        # Keep records that don't have matches in post_Y
        clean_post2 = clean_post1[~clean_post1['merge_key'].isin(post_Y['merge_key'])].copy()
        clean_post2.drop(columns=['merge_key'], inplace=True)
    else:
        clean_post2 = clean_post1.copy()
        
    if logger:              
        logger(clean_post1, clean_post2, "post_reversals", chunk_id)
    
    return clean_post2
# -------------------------------------------------------------------------
def clean_pre_20120206(pre, *, chunk_id=None, logger=None):
    """
    Clean Enhanced TRACE data reported before February 6, 2012.
    
    Removes cancellations (C), corrections (W), and reversals (asof_cd='R') from 
    pre-2012 TRACE Enhanced data using the legacy reporting structure. This function 
    implements the three-stage cleaning procedure described in Dick-Nielsen (2009, 2014) 
    for the original TRACE reporting system with special handling for correction chains.
    
    Regulatory Context
    ------------------
    Before February 6, 2012, FINRA used a different TRACE reporting structure:
    - 'T': Trade Report (original transaction)
    - 'C': Cancellation (ORIG_MSG_SEQ_NB points to original trade's MSG_SEQ_NB)
    - 'W': Correction (can form chains where W corrects W corrects T)
    - asof_cd='R': Reversal indicator (requires sequence-based matching)
    - asof_cd='D': Delayed dissemination (excluded from matching)
    - asof_cd='X': Delayed reversal (excluded from matching)
                
    Returns
    -------
    pandas.DataFrame
        Cleaned dataset with cancellations, corrections, and reversals removed.
        For corrections, matched T records are replaced with their corresponding
        W records (the corrected versions). Empty DataFrame returned if input is empty.
        
        The returned DataFrame:
        - Preserves all columns from the input
        - Contains corrected W records in place of original T records
        - Has fewer rows due to cancellation/reversal removal
        - Is not sorted (downstream code should sort as needed)
        - Has reset index from concatenation operations
    
    References
    ----------
    - Dick-Nielsen, J. (2009). "Liquidity Biases in TRACE." Journal of Fixed 
      Income, 19(2), 43-55.
    - Dick-Nielsen, J. (2014). "How to Clean Enhanced TRACE Data." Working Paper.
    - FINRA TRACE Historical Reporting Guide
    
    See Also
    --------
    clean_post_20120206 : Cleaning function for post-2012 data (simpler logic)
    clean_agency_transactions : Removes inter-dealer duplicate reporting
    """
    if pre.empty:
        return pd.DataFrame()
    
    # Store original for logging
    original_pre = pre['cusip_id'].copy()
    
    # Split data based on trc_st
    pre_C = pre[pre['trc_st'] == 'C'].copy()
    pre_W = pre[pre['trc_st'] == 'W'].copy()
    pre_T = pre[pre['trc_st'] == 'T'].copy()
    
    # Step 2.1: Remove Cancellation Cases (C)
    if not pre_C.empty and not pre_T.empty:
        # Create keys for matching
        pre_T['cancel_key'] = (pre_T['cusip_id'] + '_' + 
                              pre_T['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' +
                              pre_T['trd_exctn_tm'].astype(str) + '_' + 
                              pre_T['rptd_pr'].astype(str) + '_' +
                              pre_T['entrd_vol_qt'].astype(str) + '_' + 
                              pre_T['trd_rpt_dt'].dt.strftime('%Y%m%d') + '_' +
                              pre_T['msg_seq_nb'].astype(str))
        
        pre_C['cancel_key'] = (pre_C['cusip_id'] + '_' + 
                              pre_C['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' +
                              pre_C['trd_exctn_tm'].astype(str) + '_' + 
                              pre_C['rptd_pr'].astype(str) + '_' +
                              pre_C['entrd_vol_qt'].astype(str) + '_' + 
                              pre_C['trd_rpt_dt'].dt.strftime('%Y%m%d') + '_' +
                              pre_C['orig_msg_seq_nb'].astype(str))
        
        # Keep records that don't have matches in pre_C
        clean_pre1 = pre_T[~pre_T['cancel_key'].isin(pre_C['cancel_key'])].copy()
        clean_pre1.drop(columns=['cancel_key'], inplace=True)
    else:
        clean_pre1 = pre_T.copy()
        
    if logger:                                       
        logger(original_pre, clean_pre1, "pre_cancel_C", chunk_id)
    
    # Step 2.2: Remove Correction Cases (W) - keeping your existing logic as it looks correct
    if not pre_W.empty and not clean_pre1.empty:
        # [Keeping your existing W correction logic as is - it appears correct]
        w_msg = pre_W[['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb']].copy()
        w_msg['flag'] = 'msg'
        
        w_omsg = pre_W[['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'orig_msg_seq_nb']].copy()
        w_omsg = w_omsg.rename(columns={'orig_msg_seq_nb': 'msg_seq_nb'})
        w_omsg['flag'] = 'omsg'
        
        w_combined = pd.concat([w_msg, w_omsg], ignore_index=True)
        
        w_napp = w_combined.groupby(['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 
                                     'trd_exctn_tm', 'msg_seq_nb']).size().reset_index(name='napp')
        
        w_mult = w_combined.drop_duplicates(['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 
                                             'trd_exctn_tm', 'msg_seq_nb', 'flag'])
        w_ntype = w_mult.groupby(['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 
                                  'trd_exctn_tm', 'msg_seq_nb']).size().reset_index(name='ntype')
        
        w_comb = pd.merge(w_napp, w_ntype, 
                         on=['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb'], 
                         how='left')
        
        w_keep = w_comb[(w_comb['napp'] == 1) | ((w_comb['napp'] > 1) & (w_comb['ntype'] == 1))]
        w_keep = pd.merge(w_keep, w_combined, 
                         on=['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb'], 
                         how='inner')
        
        w_keep['npair'] = w_keep.groupby(['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm']).transform('size') / 2
        
        w_keep1 = w_keep[w_keep['npair'] == 1].copy()
        w_keep1_pivot = w_keep1.pivot_table(
            index=['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm'], 
            columns='flag', values='msg_seq_nb', aggfunc='first'
        ).reset_index()
        w_keep1_pivot = w_keep1_pivot.rename(columns={'msg': 'msg_seq_nb', 'omsg': 'orig_msg_seq_nb'})
        
        w_keep2 = w_keep[(w_keep['npair'] > 1) & (w_keep['flag'] == 'msg')].copy()
        w_keep2 = pd.merge(
            w_keep2[['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb']], 
            pre_W[['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb', 'orig_msg_seq_nb']], 
            on=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb'], 
            how='left'
        )
        
        w_clean = pd.concat([
            w_keep1_pivot, 
            w_keep2[['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 
                    'msg_seq_nb', 'orig_msg_seq_nb']]
        ], ignore_index=True)
        w_clean = w_clean.drop(columns=['bond_sym_id'])
        
        w_clean_full = pd.merge(w_clean, pre_W.drop(columns=['orig_msg_seq_nb']), 
                               on=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb'], 
                               how='left')
        
        clean_pre1['correction_key'] = (clean_pre1['cusip_id'] + '_' + 
                                        clean_pre1['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' +
                                        clean_pre1['msg_seq_nb'].astype(str))
        
        w_clean_full['correction_key'] = (w_clean_full['cusip_id'] + '_' + 
                                          w_clean_full['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' +
                                          w_clean_full['orig_msg_seq_nb'].astype(str))
        
        clean_pre2 = clean_pre1[~clean_pre1['correction_key'].isin(w_clean_full['correction_key'])].copy()
        
        matched_t_keys = clean_pre1['correction_key'][
            clean_pre1['correction_key'].isin(w_clean_full['correction_key'])
        ].tolist()
        w_to_add = w_clean_full[w_clean_full['correction_key'].isin(matched_t_keys)].copy()
        
        w_to_add = w_to_add.drop_duplicates(
            ['cusip_id', 'trd_exctn_dt', 'msg_seq_nb', 'orig_msg_seq_nb', 'rptd_pr', 'entrd_vol_qt']
        )
        
        clean_pre2.drop(columns=['correction_key'], inplace=True)
        w_to_add.drop(columns=['correction_key'], inplace=True)
        clean_pre3 = pd.concat([clean_pre2, w_to_add], ignore_index=True)
    else:
        clean_pre3 = clean_pre1.copy()
    
    if logger:
        logger(clean_pre1, clean_pre3, "pre_correction_W", chunk_id)
    
    # Step 2.3: Reversal Case - CORRECTED TO MATCH SAS EXACTLY
    # Extract reversal records
    rev_header = clean_pre3[clean_pre3['asof_cd'] == 'R'][
        ['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 
         'trd_rpt_dt', 'trd_rpt_tm', 'entrd_vol_qt', 'rptd_pr', 
         'rpt_side_cd', 'cntra_mp_id']
    ].copy()
    
    # Remove records that are R (reversal) D (Delayed dissemination) and X (delayed reversal)
    clean_pre4 = clean_pre3[~clean_pre3['asof_cd'].isin(['R', 'X', 'D'])].copy()
    
    if rev_header.empty or clean_pre4.empty:
        clean_pre5 = clean_pre4.copy()
    else:
        # Prepare header for clean_pre4
        clean_pre4_header = clean_pre4[
            ['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 
             'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 'cntra_mp_id', 
             'trd_rpt_dt', 'trd_rpt_tm', 'msg_seq_nb']
        ].copy()
        
        # =========================
        # Option A: 7-key matching (including execution time)
        # =========================
        
        # Sort for 7-key sequence numbering
        rev_header7 = rev_header.sort_values([
            'cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 
            'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 'cntra_mp_id', 
            'trd_rpt_dt', 'trd_rpt_tm'
        ]).copy()
        
        # Add sequence number for 7-key groups
        rev_header7['seq'] = rev_header7.groupby([
            'cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm',
            'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 'cntra_mp_id'
        ]).cumcount() + 1
        
        # Sort clean_pre4_header for 7-key sequence numbering
        clean_pre4_header7 = clean_pre4_header.sort_values([
            'cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm',
            'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 'cntra_mp_id',
            'trd_rpt_dt', 'trd_rpt_tm', 'msg_seq_nb'
        ]).copy()
        
        clean_pre4_header7['seq7'] = clean_pre4_header7.groupby([
            'cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm',
            'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 'cntra_mp_id'
        ]).cumcount() + 1
        
        # =========================
        # Option B: 6-key matching (excluding execution time)
        # =========================
        
        # Sort for 6-key sequence numbering (note different sort order!)
        rev_header6 = rev_header.sort_values([
            'cusip_id', 'bond_sym_id', 'trd_exctn_dt',  # <-- no trd_exctn_tm here
            'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 'cntra_mp_id',
            'trd_exctn_tm', 'trd_rpt_dt', 'trd_rpt_tm'  # <-- trd_exctn_tm comes after the groupby keys
        ]).copy()
        
        # Add sequence number for 6-key groups
        rev_header6['seq'] = rev_header6.groupby([
            'cusip_id', 'bond_sym_id', 'trd_exctn_dt',  # <-- no trd_exctn_tm in groupby
            'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 'cntra_mp_id'
        ]).cumcount() + 1
        
        # Sort clean_pre4_header for 6-key sequence numbering
        clean_pre4_header6 = clean_pre4_header.sort_values([
            'cusip_id', 'bond_sym_id', 'trd_exctn_dt',  # <-- no trd_exctn_tm here
            'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 'cntra_mp_id',
            'trd_exctn_tm', 'trd_rpt_dt', 'trd_rpt_tm', 'msg_seq_nb'
        ]).copy()
        
        clean_pre4_header6['seq6'] = clean_pre4_header6.groupby([
            'cusip_id', 'bond_sym_id', 'trd_exctn_dt',  # <-- no trd_exctn_tm in groupby
            'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 'cntra_mp_id'
        ]).cumcount() + 1
        
        # =========================
        # Perform the joins (matching SAS SQL logic)
        # =========================
        
        # First do 7-key join
        clean_pre5_header = pd.merge(
            clean_pre4_header7,
            rev_header7[['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 
                        'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 
                        'cntra_mp_id', 'seq']].rename(columns={'seq': 'rev_seq7'}),
            left_on=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm',
                    'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 
                    'cntra_mp_id', 'seq7'],
            right_on=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm',
                     'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd',
                     'cntra_mp_id', 'rev_seq7'],
            how='left'
        )
        
        # Then do 6-key join - need to add seq6 from clean_pre4_header6 first
        # Merge clean_pre5_header with clean_pre4_header6 to get seq6 column
        clean_pre5_header = pd.merge(
            clean_pre5_header,
            clean_pre4_header6[['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm',
                               'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd',
                               'cntra_mp_id', 'msg_seq_nb', 'trd_rpt_dt', 'trd_rpt_tm', 'seq6']],
            on=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm',
                'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd',
                'cntra_mp_id', 'msg_seq_nb', 'trd_rpt_dt', 'trd_rpt_tm'],
            how='left'
        )
        
        # Now do the 6-key join with reversals
        clean_pre5_header = pd.merge(
            clean_pre5_header,
            rev_header6[['cusip_id', 'trd_exctn_dt',  # no trd_exctn_tm in join
                        'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd',
                        'cntra_mp_id', 'seq']].rename(columns={'seq': 'rev_seq6'}),
            left_on=['cusip_id', 'trd_exctn_dt',  # no trd_exctn_tm in join
                    'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd',
                    'cntra_mp_id', 'seq6'],
            right_on=['cusip_id', 'trd_exctn_dt',  # no trd_exctn_tm in join
                     'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd',
                     'cntra_mp_id', 'rev_seq6'],
            how='left'
        )
        
        # As per SAS: use 6-key results (better match rate)
        # Keep only records where rev_seq6 is null (i.e., no match found)
        clean_pre5_header = clean_pre5_header[clean_pre5_header['rev_seq6'].isna()].copy()
        clean_pre5_header = clean_pre5_header.drop(columns=['rev_seq6', 'rev_seq7', 'seq7'], errors='ignore')
        
        # Join back to get all columns from clean_pre4
        # Using all the identifying columns for the join
        clean_pre5 = pd.merge(
            clean_pre4,
            clean_pre5_header[['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm',
                              'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd',
                              'cntra_mp_id', 'msg_seq_nb', 'trd_rpt_dt', 'trd_rpt_tm']],
            on=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm',
                'entrd_vol_qt', 'rptd_pr', 'rpt_side_cd', 
                'cntra_mp_id', 'msg_seq_nb', 'trd_rpt_dt', 'trd_rpt_tm'],
            how='inner'
        )
    
    if logger:
        logger(clean_pre3, clean_pre5, "pre_reversal", chunk_id)
    
    return clean_pre5
# -------------------------------------------------------------------------
def clean_agency_transactions(clean_combined, 
                              remove_all_interdealer_buys=False                            
                              ):
    """
    Remove double-counting from inter-dealer trades reported by both buy and sell sides.
    
    Agency transactions (trades between dealers) can be reported by both the buyer
    and seller, leading to double-counting in aggregated statistics. This function
    identifies matched inter-dealer trades and removes the buy-side reports to 
    eliminate duplication while preserving all customer-dealer trades.
    
    Problem Context
    ---------------
    When two dealers trade with each other:
    - Dealer A (buyer) reports : rpt_side_cd='B', cntra_mp_id='D'
    - Dealer B (seller) reports: rpt_side_cd='S', cntra_mp_id='D'
    
    Both reports describe the same transaction, causing double-counting in volume
    and trade count statistics.
    
    Solution Strategy
    -----------------
    Keep all sell-side inter-dealer trades (S, D) as the canonical records.
    For buy-side inter-dealer trades (B, D):
        - Match buy records to sell records using 4 keys (NO execution time)
        - Remove matched buy records (duplicates)
        - Keep unmatched buy records (unique information)
    
    Keep all customer trades (C) regardless of reporting side.
    
    Final dataset composition:
        = All customer trades (cntra_mp_id='C')
        + All sell-side inter-dealer trades (rpt_side_cd='S', cntra_mp_id='D')
        + Unmatched buy-side inter-dealer trades (rpt_side_cd='B', cntra_mp_id='D')
    
    Matching Logic
    --------------
    Buy and sell records are matched using 4 keys:
        1. cusip_id: Same bond
        2. trd_exctn_dt: Same execution date
        3. rptd_pr: Same price
        4. entrd_vol_qt: Same volume
    
    **Critical: Execution time (trd_exctn_tm) is EXCLUDED from matching.**
    
    Rationale: Execution time is self-reported by each dealer and may differ
    slightly due to clock synchronization issues or reporting delays. Two trades
    on the same day with the same CUSIP, price, and volume are considered the
    same transaction even if reported at different times.
    
    Parameters
    ----------
    clean_combined : pandas.DataFrame
        Combined cleaned TRACE data after cancellation, correction, and reversal
        removal. Must contain:
        - cusip_id : str
        - trd_exctn_dt : datetime
        - rptd_pr : float
        - entrd_vol_qt : float or int
        - rpt_side_cd : str ('B'=buy, 'S'=sell)
        - cntra_mp_id : str ('C'=customer, 'D'=dealer)
        
        Additional columns are preserved in output.
        
    remove_all_interdealer_buys : bool, optional (default=False)
        Controls the aggressiveness of inter-dealer buy removal:
        
        - False (DEFAULT): Remove only matched inter-dealer buys.
            Standard SAS behavior. Keeps unmatched buy-side reports that may 
            contain unique information. Conservative approach.
            
        - True (AGGRESSIVE): Remove ALL inter-dealer buys regardless of matching.
            Implements alternative approach: "Some calls to remove all inter-dealer 
            buys completely." Use when maximum consistency is needed over completeness.
    
    Returns
    -------
    pandas.DataFrame
        Cleaned data with inter-dealer double-counting removed:
        - All customer trades (cntra_mp_id='C')
        - All sell-side inter-dealer trades
        - Unmatched buy-side inter-dealer trades (default mode) OR
          No buy-side inter-dealer trades (aggressive mode)
        
        Empty DataFrame returned if input is empty. Index is reset.
      
    Examples
    --------
    >>> # Basic usage (conservative mode)
    >>> trace_clean = clean_agency_transactions(trace_combined)
    >>> print(f"Removed: {len(trace_combined) - len(trace_clean)}")
    
    >>> # Aggressive mode
    >>> trace_clean = clean_agency_transactions(
    ...     trace_combined, 
    ...     remove_all_interdealer_buys=True
    ... )
    
    >>> # Example: Inter-dealer trades on 2024-01-15
    >>> # Sell: CUSIP='123', Price=101.5, Vol=100K, Time=10:30:15
    >>> # Buy:  CUSIP='123', Price=101.5, Vol=100K, Time=10:30:18
    >>> # Result: Buy removed (3-second time difference ignored)
    
    References
    ----------
    - Dick-Nielsen, J. (2009). "Liquidity Biases in TRACE." Journal of Fixed 
      Income, 19(2), 43-55.
    - Dick-Nielsen, J. (2014). "How to Clean Enhanced TRACE Data." Working Paper.
    
    See Also
    --------
    clean_post_20120206 : Post-2012 cancellation/correction/reversal removal
    clean_pre_20120206 : Pre-2012 cancellation/correction/reversal removal
    clean_trace_enhanced_chunk : Main orchestration function for Enhanced TRACE
    """
    
    if clean_combined.empty:
        return pd.DataFrame()
    
    # Split data based on rpt_side_cd and cntra_mp_id
    agency_s = clean_combined[(clean_combined['rpt_side_cd'] == 'S') & 
                              (clean_combined['cntra_mp_id'] == 'D')].copy()
    
    agency_b = clean_combined[(clean_combined['rpt_side_cd'] == 'B') & 
                              (clean_combined['cntra_mp_id'] == 'D')].copy()
    
    customer = clean_combined[clean_combined['cntra_mp_id'] == 'C'].copy()
    
    # if logger:
    logging.info(f"Customer trades: {len(customer)}")
    logging.info(f"Agency sells (inter-dealer): {len(agency_s)}")  
    logging.info(f"Agency buys (inter-dealer): {len(agency_b)}")
    
    if remove_all_interdealer_buys:
        # Aggressive approach: Remove ALL inter-dealer buys       
        logging.info("Removing ALL inter-dealer buy records (aggressive approach)")
        logging.info(f"Inter-dealer buys removed: {len(agency_b)}")
        
        clean_final = pd.concat([customer, agency_s], ignore_index=True)
        
    else:
        # Conservative approach: Remove only matched inter-dealer buys (SAS default)
        if not agency_b.empty and not agency_s.empty:
            # Perform the anti-join to find unmatched agency buys
            # This matches the actual SAS SQL logic
            agency_b_merged = pd.merge(
                agency_b,
                agency_s[['cusip_id', 'trd_exctn_dt', 'rptd_pr', 'entrd_vol_qt']].drop_duplicates(),
                on=['cusip_id', 'trd_exctn_dt', 'rptd_pr', 'entrd_vol_qt'],
                how='left',
                indicator='_merge'
            )
            
            # Keep only unmatched buy records (equivalent to "having b.rpt_side_cd = ''" in SAS)
            agency_b_nodup = agency_b_merged[agency_b_merged['_merge'] == 'left_only'].copy()
            agency_b_nodup = agency_b_nodup.drop(columns=['_merge'])
            
            
            n_matched = len(agency_b) - len(agency_b_nodup)
            logging.info(f"Matched inter-dealer buys removed: {n_matched}")
            logging.info(f"Unmatched inter-dealer buys kept: {len(agency_b_nodup)}")
            
            clean_final = pd.concat([customer, agency_s, agency_b_nodup], ignore_index=True)
            
        else:
            # If either agency_s or agency_b is empty, combine what we have
            clean_final = pd.concat([customer, agency_s, agency_b], ignore_index=True)        
    
    return clean_final
# -------------------------------------------------------------------------
def build_fisd(db, params: dict | None = None):
    """
    Build FISD bond universe with switchable screens.

    Parameters
    ----------
    db : wrds.Connection
    params : dict or None
        Switchboard + knobs. See defaults below.

    Returns
    -------
    fisd     : pd.DataFrame  (filtered FISD issue-level table)
    fisd_off : pd.DataFrame  (['cusip_id','offering_amt','maturity'])
    """
    import pandas as pd
    import gc

    # ---- Defaults mirror _trace_settings.FISD_PARAMS ---------------------
    p = {
        "currency_usd_only": True,
        "fixed_rate_only": True,
        "non_convertible_only": True,
        "non_asset_backed_only": True,
        "exclude_bond_types": True,
        "valid_coupon_frequency_only": True,
        "require_accrual_fields": True,
        "principal_amt_eq_1000_only": True,
        "exclude_equity_index_linked": True,
        "enforce_tenor_min": True,
        "invalid_coupon_freq": [-1, 13, 14, 15, 16],
        "excluded_bond_types": [
            "TXMU","CCOV","CPAS","MBS","FGOV","USTC","USBD","USNT","USSP","USSI",
            "FGS","USBL","ABS","O30Y","O10Y","O5Y","O3Y","O4W","O13W","O26W","O52W",
            "CCUR","ADEB","AMTN","ASPZ","EMTN","ADNT","ARNT","TPCS","CPIK","PS","PSTK"
        ],
        "tenor_min_years": 1.0,
    }
    p.update(params or {})

    # ---- 1) Pull raw FISD tables ---------------------------------
    qry_issuer = """
        SELECT issuer_id, country_domicile, sic_code
        FROM   fisd.fisd_mergedissuer
    """
    qry_issue = """
        SELECT complete_cusip, issue_id, issue_name,
               issuer_id, foreign_currency,
               coupon_type, coupon, convertible,
               asset_backed, rule_144a,
               bond_type, private_placement,
               interest_frequency, dated_date,
               day_count_basis, offering_date,
               maturity, principal_amt, offering_amt
        FROM   fisd.fisd_mergedissue
    """
    fisd_issuer = db.raw_sql(qry_issuer)
    fisd_issue  = db.raw_sql(qry_issue)
    fisd        = pd.merge(fisd_issue, fisd_issuer, on="issuer_id", how="left").copy()

    # ---- 2) Start log -----------------------------------------------------
    log_fisd_filter(fisd, fisd, "start")

    # ---- 3) Currency (foreign_currency == 'N') ---------------------------
    if p["currency_usd_only"]:
        before = fisd
        fisd = fisd.loc[(fisd["foreign_currency"] == "N")].copy()
        log_fisd_filter(before, fisd, "USD currency")
    else:
        log_fisd_filter(fisd, fisd, "USD currency (skipped)")

    # ---- 4) Fixed-rate ----------------------------------------------------
    if p["fixed_rate_only"]:
        before = fisd
        fisd = fisd.loc[fisd["coupon_type"] != "V"].copy()
        log_fisd_filter(before, fisd, "fixed rate")
    else:
        log_fisd_filter(fisd, fisd, "fixed rate (skipped)")

    # ---- 5) Non-convertible ----------------------------------------------
    if p["non_convertible_only"]:
        before = fisd
        fisd = fisd.loc[fisd["convertible"] == "N"].copy()
        log_fisd_filter(before, fisd, "non convertible")
    else:
        log_fisd_filter(fisd, fisd, "non convertible (skipped)")

    # ---- 6) Non-asset-backed ---------------------------------------------
    if p["non_asset_backed_only"]:
        before = fisd
        fisd = fisd.loc[fisd["asset_backed"] == "N"].copy()
        log_fisd_filter(before, fisd, "non asset backed")
    else:
        log_fisd_filter(fisd, fisd, "non asset backed (skipped)")

    del before
    gc.collect()

    # ---- 7) Exclude bond types -------------------------------------------
    if p["exclude_bond_types"]:
        before = fisd
        exclude_btypes = set(p["excluded_bond_types"])
        fisd = fisd.loc[~fisd["bond_type"].isin(exclude_btypes)].copy()
        log_fisd_filter(before, fisd, "exclude gov muni ABS types")
    else:
        log_fisd_filter(fisd, fisd, "exclude gov muni ABS types (skipped)")

    # ---- 9) Valid coupon frequency ---------------------------------------
    # Convert interest_frequency to int before filtering
    # Handle NA values by converting to numeric, filling NA with -1, then converting to int
    fisd["interest_frequency"] = pd.to_numeric(fisd["interest_frequency"], errors='coerce').fillna(-1).astype(int)

    if p["valid_coupon_frequency_only"]:
        before = fisd
        invalid_freq = set(p["invalid_coupon_freq"])
        fisd = fisd.loc[~fisd["interest_frequency"].isin(invalid_freq)].copy()
        log_fisd_filter(before, fisd, "valid coupon frequency")
    else:
        log_fisd_filter(fisd, fisd, "valid coupon frequency (skipped)")

    # ---- 10) Complete accrual fields -------------------------------------
    if p["require_accrual_fields"]:
        before = fisd
        date_cols = ["offering_date", "dated_date"]
        fisd[date_cols] = fisd[date_cols].apply(pd.to_datetime, errors="coerce")
        req_cols = date_cols + ["interest_frequency", "day_count_basis", "coupon_type", "coupon"]
        fisd = fisd.dropna(subset=req_cols).copy()
        log_fisd_filter(before, fisd, "complete accrual fields")
    else:
        log_fisd_filter(fisd, fisd, "complete accrual fields (skipped)")

    # ---- 11) principal_amt == 1000 ---------------------------------------
    if p["principal_amt_eq_1000_only"]:
        before = fisd
        fisd = fisd.loc[fisd["principal_amt"] == 1000].copy()
        log_fisd_filter(before, fisd, "principal_amt == 1,000")
    else:
        log_fisd_filter(fisd, fisd, "principal_amt == 1,000 (skipped)")

    # ---- 12) Exclude equity/index-linked ---------------------------------
    if p["exclude_equity_index_linked"]:
        before = fisd
        fisd["equity_linked"] = fisd["issue_name"].str.contains(
            r"EQUITY\-LINKED|EQUITY LINKED|EQUITYLINKED|INDEX\-LINKED|INDEX LINKED|INDEXLINKED",
            case=False, na=False
        ).astype(int)
        fisd = fisd[fisd["equity_linked"] == 0].drop(columns="equity_linked").copy()
        log_fisd_filter(before, fisd, "exclude equity and index linked")
    else:
        log_fisd_filter(fisd, fisd, "exclude equity and index linked (skipped)")

    # ---- 12b) Tenor >= min years -----------------------------------------
    if p["enforce_tenor_min"]:
        before = fisd
        fisd["maturity"] = pd.to_datetime(fisd["maturity"], errors="coerce")
        fisd["offering_date"] = pd.to_datetime(fisd["offering_date"], errors="coerce")
        fisd["tenor"] = (fisd["maturity"] - fisd["offering_date"]).dt.days / 365.25
        fisd = fisd.loc[fisd["tenor"] >= float(p["tenor_min_years"])].copy()
        log_fisd_filter(before, fisd, f"tenor >= {p['tenor_min_years']} year(s)")
    else:
        log_fisd_filter(fisd, fisd, f"tenor >= {p['tenor_min_years']} year(s) (skipped)")

    # ---- 13) Housekeeping + fisd_off -------------------------------------
    fisd = fisd.reset_index(drop=True)
    fisd["index"] = range(1, len(fisd) + 1)

    fisd_off = fisd[["complete_cusip", "offering_amt", "maturity"]].copy()
    fisd_off.rename(columns={"complete_cusip": "cusip_id"}, inplace=True)

    return fisd, fisd_off
# -------------------------------------------------------------------------
def error_checks(
    db,
    cusip_chunks,
    fisd_off: pd.DataFrame,
    clean_agency: bool = True,
    volume_filter: float | tuple[str, float] = ("dollar", 10000.0),
    trade_times: list[str] | None = None,
    calendar_name: str | None = None,
    *,
    ds_params: dict | None = None,
    bb_params: dict | None = None,
    init_error_params: dict | None = None,
    filters: dict | None = None
                    ):
    # --- Filter Defaults -----------------
    FILTER_DEFAULTS = dict(
        dick_nielsen            = True,
        decimal_shift_corrector = True,
        trading_time            = False,
        trading_calendar        = True,
        price_filters           = True,
        volume_filter_toggle    = True,
        bounce_back_filter      = True,
        yld_price_filter        = True,
        amtout_volume_filter    = True,
        trd_exe_mat_filter      = True,
        flag_initial_price_errors = True,
    )
    f = {**FILTER_DEFAULTS, **(filters or {})}
    """
    Slimline filtering function for the plots of filtered data

    """

    all_super_list         = []
    all_super_listbb       = []
    all_super_list_ie      = []
    bb_cusips_all          = []
    dec_shift_cusips_all   = []
    init_price_cusips_all  = []
    
    sort_cols = ["cusip_id","trd_exctn_dt","trd_exctn_tm",
                 "trd_rpt_dt","trd_rpt_tm","msg_seq_nb"]
      
    for i in range(0, len(cusip_chunks)):  
        start_time = time.time()  # Start timer
        logging.info(f"Processing chunk {i+1} of {len(cusip_chunks)}")
        temp_list = cusip_chunks[i] 
        temp_tuple = tuple(temp_list)
        parm = {'cusip_id': temp_tuple}
        
        # Load data from WRDS per chunk
        trace = db.raw_sql('''
            SELECT cusip_id, bond_sym_id, trd_exctn_dt, trd_exctn_tm, days_to_sttl_ct, 
                   lckd_in_ind, wis_fl, sale_cndtn_cd, msg_seq_nb, trc_st, 
                   trd_rpt_dt, trd_rpt_tm, entrd_vol_qt, rptd_pr, yld_pt, 
                   asof_cd, orig_msg_seq_nb, rpt_side_cd, cntra_mp_id 
            FROM trace.trace_enhanced 
            WHERE cusip_id in %(cusip_id)s
        ''', params=parm)
        
        trace["rptd_pr"] = trace["rptd_pr"].astype("float64").round(6)
        trace = trace.drop(columns=["index"], errors="ignore").reset_index(drop=True)
                       
        logging.info(f"Chunk {i+1}: Retrieved {len(trace)} rows from WRDS")
        
        if len(trace) == 0:
            continue
        
        # Initial log for cleaning
        log_filter(trace, trace, "start", i)
        
        # Filter 1: Dick-Nielsen
        if f["dick_nielsen"]:
            clean_chunk = clean_trace_chunk(
                trace,
                chunk_id      = i,
                clean_agency  = clean_agency,
                logger        = log_ct_filter
            )
            log_filter(trace, clean_chunk, "dick_nielsen_filter", i)
            trace = clean_chunk.copy()
            del clean_chunk
        else:
            # skip DN cleaner; pass raw chunk through
            log_filter(trace, trace, "dick_nielsen_filter (skipped)", i)
        gc.collect()
        
        
        # Pre decimal sort #
        trace = trace.sort_values(sort_cols, kind="mergesort", ignore_index=True)
       
        # Filter 2: Decimal Correction 
        if f["decimal_shift_corrector"]:
            _ds_defaults = _ds_defaults = dict(
                id_col="cusip_id",
                date_col="trd_exctn_dt",
                time_col="trd_exctn_tm",
                price_col="rptd_pr",
                factors=(0.1, 0.01, 10.0, 100.0),
                tol_pct_good=0.02,
                tol_abs_good=8.0,
                tol_pct_bad=0.05,
                low_pr=5.0,
                high_pr=300.0,
                anchor="rolling",
                window=5,
                improvement_frac=0.2,
                par_snap=True,
                par_band=15.0,
                output_type="cleaned",
            )
            _ds = {**_ds_defaults, **(ds_params or {})}
        
            trace = decimal_shift_corrector(
                trace,**_ds
            )
            
            # Collect DS CUSIPs (for checking purposes)
            ds_cusips = (
                trace.loc[trace.get('dec_shift_flag', 0).eq(1), "cusip_id"]
                        .astype(str).str.strip().unique().tolist()
            )
                      
            dec_shift_cusips_all.extend([str(c) for c in ds_cusips])
            
            # Keep only columns needed for decimal-shift plotting (memory optimization)
            DS_COLS = ["cusip_id", "trd_exctn_dt", "rptd_pr", "dec_shift_flag", "suggested_price"]
            traceds = (
                trace[DS_COLS]
                .sort_values(["cusip_id", "trd_exctn_dt"], kind="mergesort", ignore_index=True)
                .copy()
            )

        else:
            log_filter(trace, trace, "decimal_shift (skipped)", i, replace=True, n_rows_replaced=0)
        gc.collect()
        
        # Need to correct the prices now #
        trace["rptd_pr"] = np.where(trace['dec_shift_flag'] == 1, trace['suggested_price'],
                                    trace["rptd_pr"])
                       
        # Filter 3: Trading Time                
        if f["trading_time"]:
            before_time = trace
            trace = filter_by_trade_time(
                df=trace,
                trade_times=trade_times,
                time_col="trd_exctn_tm",
                keep_missing=False,
            )
            log_filter(before_time, trace, "trading_time_filter", i)
            del before_time
            gc.collect()
        else:
            log_filter(trace, trace, "trading_time_filter (skipped)", i)

        # Filter 4: Trading Calendar       
        if f["trading_calendar"]:
            before_calr = trace
            trace = filter_by_calendar(
                df=trace,
                calendar_name=calendar_name,
                date_col="trd_exctn_dt",
                start_date="2002-07-01",
                end_date=None,
                keep_missing=False,
            )
            log_filter(before_calr, trace, "calendar_filter", i)
            del before_calr
            gc.collect()
        else:
            log_filter(trace, trace, "calendar_filter (skipped)", i)

        
        # Filter 5: Prices                      
        if f["price_filters"]:
            trace = filter_with_log(trace, trace['rptd_pr'] > 0,     "neg_price_filter",   i)
            trace = filter_with_log(trace, trace['rptd_pr'] <= 1000, "large_price_filter", i)
        else:
            log_filter(trace, trace, "neg_price_filter (skipped)",   i)
            log_filter(trace, trace, "large_price_filter (skipped)", i)

        # Filter 6: Trading Volume    
        # Compute dollar volume
        # entrd_vol_qt is in DOLLARS
        # https://wrds-www.wharton.upenn.edu/documents/1241/TRACE_Enhanced_Corporate_and_Agency_Historic_Data_File_Layout_post_2_6_12_10252024v.pdf
        # Applies PRE and POST 2012, see:
        # https://wrds-www.wharton.upenn.edu/documents/1240/TRACE_Enhanced_Corporate_and_Agency_Historic_Data_File_Layout_pre_2_6_12_09092021v.pdf            
        trace['dollar_vol'] = (trace['entrd_vol_qt'] * trace['rptd_pr'] / 100)  # always compute
        
        if f["volume_filter_toggle"]:
            vkind, vthr = _normalize_volume_filter(volume_filter)  
            if vkind == "dollar":
                mask = trace['dollar_vol'] >= vthr
                stage_name = "volume_filter[dollar]"
            else:  # "par"
                # keep rows with entered volume (par units) >= threshold
                mask = trace['entrd_vol_qt'] >= vthr
                stage_name = "volume_filter[par]"
        
            trace = filter_with_log(trace, mask, stage_name, i)
        else:
            log_filter(trace, trace, "volume_filter (skipped)", i)
       
        
        # Pre BB sort #
        trace = trace.sort_values(sort_cols, kind="mergesort", ignore_index=True)
        
        # Filter 7: Bounce-Back
        if f["bounce_back_filter"]:
            _bb_defaults = _bb_defaults = dict(
                id_col="cusip_id",
                date_col="trd_exctn_dt",
                time_col="trd_exctn_tm",
                price_col="rptd_pr",
                threshold_abs=35.0,
                lookahead=5,
                max_span=5,
                window=5,
                back_to_anchor_tol=0.25,
                candidate_slack_abs=1.0,
                reassignment_margin_abs=5.0,
                use_unique_trailing_median=True,
                par_spike_heuristic=True,
                par_level=100.0,
                par_equal_tol=1e-8,
                par_min_run=3,
                par_cooldown_after_flag=2,
            )
            _bb = {**_bb_defaults, **(bb_params or {})}

            trace_bb = flag_price_change_errors(
                trace,
                **_bb
            )

            # Collect BB CUSIPs (audit/export only)
            bb_cusips = (
                trace_bb.loc[trace_bb.get("filtered_error", 0).eq(1), "cusip_id"]
                        .astype(str).str.strip().unique().tolist()
            )

            bb_cusips_all.extend([str(c) for c in bb_cusips])

            gc.collect()
        else:
            trace_bb = trace.copy()
            trace_bb["filtered_error"] = 0
            log_filter(trace, trace, "bounce_back_filter (skipped)", i)
        gc.collect()

        # ============================================================
        # From here onwards, work on trace_ie (copy of trace_bb)
        # trace_bb stays untouched for bounce-back plotting
        # IMPORTANT: Remove rows flagged by bounce-back filter before
        # applying subsequent filters
        # ============================================================
        trace_ie = trace_bb[trace_bb['filtered_error'] == 0].copy()

        # Filter 8: Yield != Price
        if f["yld_price_filter"]:
            mask = (trace_ie["rptd_pr"] != trace_ie["yld_pt"]) | trace_ie["yld_pt"].isna()
            trace_ie = filter_with_log(trace_ie, mask, "price_yld_filter", i)
        else:
            log_filter(trace_ie, trace_ie, "price_yld_filter (skipped)", i)

        # Filter 9: Amount-outstanding vs. volume filter
        trace_ie = trace_ie.merge(fisd_off, how="left", left_on='cusip_id', right_on='cusip_id')
        if f["amtout_volume_filter"]:
            trace_ie = filter_with_log(
                trace_ie,
                trace_ie['entrd_vol_qt'] < trace_ie['offering_amt']*1000*0.50,
                "volume_offamt_filter",
                i
            )
        else:
            log_filter(trace_ie, trace_ie, "volume_offamt_filter (skipped)", i)

        # Filter 10: Execution date cannot exceed maturity
        if f["trd_exe_mat_filter"]:
            trace_ie = filter_with_log(
                trace_ie,
                trace_ie['trd_exctn_dt'] <= trace_ie['maturity'],
                "exctn_mat_dt_filter",
                i
            )
        else:
            log_filter(trace_ie, trace_ie, "exctn_mat_dt_filter (skipped)", i)

        # Filter 11: Initial Price Errors
        if f["flag_initial_price_errors"]:
            _ie_defaults = dict(
                id_col="cusip_id",
                date_col="trd_exctn_dt",
                price_col="rptd_pr",
                abs_change=50.0,
                n_transactions=3,
            )
            _ie = {**_ie_defaults, **(init_error_params or {})}
            trace_ie = flag_initial_price_errors(trace_ie, **_ie)

            # Collect Init Price Error CUSIPs (audit/export only)
            ie_cusips = (
                trace_ie.loc[trace_ie.get("initial_error_flag", 0).eq(1), "cusip_id"]
                        .astype(str).str.strip().unique().tolist()
            )

            if ie_cusips:
                init_price_cusips_all.extend([str(c) for c in ie_cusips])
            gc.collect()
        else:
            log_filter(trace_ie, trace_ie, "init_price_error_filter (skipped)", i)
            # Ensure column exists even when skipped (for plotting consistency)
            trace_ie["initial_error_flag"] = 0
        gc.collect()

        # Keep only columns needed for plotting (memory optimization)
        BB_COLS = ["cusip_id", "trd_exctn_dt", "rptd_pr", "filtered_error"]
        IE_COLS = ["cusip_id", "trd_exctn_dt", "rptd_pr", "initial_error_flag"]

        trace_bb = (
            trace_bb[BB_COLS]
            .sort_values(["cusip_id", "trd_exctn_dt"], kind="mergesort", ignore_index=True)
        )
        trace_ie = (
            trace_ie[IE_COLS]
            .sort_values(["cusip_id", "trd_exctn_dt"], kind="mergesort", ignore_index=True)
        )

        # Run garbage collection to reclaim memory
        del trace
        gc.collect()

        # Append here (now with minimal columns only)
        all_super_list.append(traceds)
        all_super_listbb.append(trace_bb)
        all_super_list_ie.append(trace_ie)

        del(traceds, trace_bb, trace_ie)

        # CUSIP check (only for bb and ds - ie is checked separately)
        merged_cusips = pd.unique(pd.Series(bb_cusips + ds_cusips)).tolist()
        
        # If cusip_chunks only has one list, always use index 0
        chunk_index = 0 if len(cusip_chunks) == 1 else i
        chunk_cusips = set(cusip_chunks[chunk_index])
        missing = [c for c in merged_cusips if c not in chunk_cusips]
        
        logging.info(f"[CUSIP CHECK] Chunk {i}: merged={len(merged_cusips)} | "
                     f"chunk_size={len(chunk_cusips)} | missing={len(missing)}")
        if missing:
            logging.info(f"[CUSIP CHECK] Chunk {i}: missing (first {min(25, len(missing))}): "
                         + ", ".join(missing[:25]))
        
        # Optionally log if we're using the single-chunk fallback
        if len(cusip_chunks) == 1 and i > 0:
            logging.info("[CUSIP CHECK] Note: Using single cusip_chunks[0] for all iterations")
              
        elapsed_time = round(time.time() - start_time, 2)
        logging.info(f"Chunk {i}: took {elapsed_time} seconds")

        # Memory tracking after each chunk
        try:
            import psutil
            mem_gb = psutil.Process().memory_info().rss / (1024**3)
            logging.info(f"[MEMORY] Chunk {i}: {mem_gb:.2f} GB")
        except ImportError:
            pass

        logging.info("-" * 50)

    gc.collect()

    final_df_ds = pd.concat(all_super_list,   ignore_index=True)
    final_df_bb = pd.concat(all_super_listbb, ignore_index=True)
    final_df_ie = pd.concat(all_super_list_ie, ignore_index=True)

    return final_df_ds, final_df_bb, final_df_ie, bb_cusips_all, dec_shift_cusips_all, init_price_cusips_all
# -------------------------------------------------------------------------
# Export helper function #
def export_trace_dataframes(
    all_data: pd.DataFrame,
    fisd_df: pd.DataFrame,
    ct_audit_records: Sequence[Mapping[str, Any]],
    audit_records:    Sequence[Mapping[str, Any]],
    *,
    output_format: str = "csv",   # "csv" or "parquet"
    out_dir: str | Path = ".",
    bounce_back_cusips: list[str] | None = None,
    decimal_shift_cusips: list[str] | None = None,
    init_price_cusips: list[str] | None = None,
    fisd_audit_records: list[dict] | None = None,
    stamp: str | None = None,
) -> None:
    """
    Export six TRACE-related dataframes.

    Parameters
    ----------
    prices_df, volumes_df, illiquidity_df, fisd_df : pandas.DataFrame
        Core dataframes ready for export.

    ct_audit_records, audit_records : Sequence[Mapping]
        Raw audit logs (e.g. list of dicts). They are converted to DataFrames
        
    output_format : {"csv", "parquet"}, default "csv"
        * "csv": write `.csv.gzip` files (pandas + gzip)  
        * "parquet": write `.parquet` files (pyarrow + snappy)

    out_dir : str or pathlib.Path, default "."
        Destination directory for all exported files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if stamp is None:
        stamp = pd.Timestamp.today().strftime("%Y%m%d")


    # --- Convert audit records to DataFrames and adjust 'chunk' -------------
    clean_trace_audit_df = pd.DataFrame(ct_audit_records)
    audit_df             = pd.DataFrame(audit_records)    
    fisd_filters_df      = pd.DataFrame(fisd_audit_records)
                   
            
    def _uniq_df(vals: list[str] | None) -> pd.DataFrame:
        vals = [] if vals is None else vals
        uniq = sorted(set(map(str, vals)))
        return pd.DataFrame({"cusip_id": uniq})

    df_bb = _uniq_df(bounce_back_cusips)
    df_ds = _uniq_df(decimal_shift_cusips)
    df_ie = _uniq_df(init_price_cusips)

    # --- Map each dataframe to a filename -----------------------------------
    files: dict[str, pd.DataFrame] = {
        "trace_enhanced"                     : all_data,
        "trace_enhanced_fisd"                : fisd_df,
        "fisd_filters_enhanced"              : fisd_filters_df,
        "dick_nielsen_filters_audit_enhanced": clean_trace_audit_df,
        "drr_filters_audit_enhanced"         : audit_df,
        "bounce_back_cusips_enhanced"        : df_bb,
        "decimal_shift_cusips_enhanced"      : df_ds,
        "init_price_cusips_enhanced"         : df_ie,
    }

    output_format = output_format.lower()
    if output_format not in {"csv", "parquet"}:
        raise ValueError("output_format must be 'csv' or 'parquet'")

    # --- Write the files -----------------------------------------------------
    for base_name, df in files.items():
        if output_format == "csv":
            file_path = out_dir / f"{base_name}_{stamp}.csv.gzip"
            df.to_csv(file_path, index=False, compression="gzip")
        else:  # parquet
            file_path = out_dir / f"{base_name}_{stamp}.parquet"
            df.to_parquet(file_path, engine="pyarrow", compression="snappy")
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -----------------------  MAIN CLASS -------------------------------------
# -------------------------------------------------------------------------
class ProcessEnhancedTRACE:
    """
    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection used to query TRACE data.
    cusip_chunks : list[list[str]]
        Lists of CUSIPs to process per chunk. Each sublist is one chunk.
    fisd_off : pandas.DataFrame
        Issue-level data used for filters or merges during aggregation.
        Must contain at least columns such as cusip_id and any offering or maturity fields you rely on.
    clean_agency : bool, default True
        If True, apply agency de-duplication after pre and post cleaning.
    volume_filter : float, default 10000.0
        Minimum dollar volume threshold. Dollar volume is size times price over 100.
    trade_times : list[str] or None, default None
        Optional inclusive intraday window as [HH:MM:SS, HH:MM:SS] to filter trades by execution time.
    calendar_name : str or None, default None
        Optional market calendar name for trading day validation.
    ds_params : dict or None, default None
        Keyword overrides passed to decimal_shift_corrector.
    bb_params : dict or None, default None
        Keyword overrides passed to flag_price_change_errors.
    export_dir : str or pathlib.Path or None, default None
        If provided, daily results and audit artifacts may be written here.
    export_format : str, default "parquet"
        File format for exports. Examples: "parquet", "csv".
    compress : str or None, default "gzip"
        Compression for exports. Examples: "gzip", "snappy", or None.
    log : logging.Logger or None, default None
        Optional logger. If None, the class will create a module-level logger.

    Attributes
    ----------
    audit_records : list[dict]
        Per-stage audit entries appended by the internal logger hook.
    bb_cusips_all : set[str]
        Set of CUSIPs flagged at least once by the price change filter.
    dec_shift_cusips_all : set[str]
        Set of CUSIPs corrected at least once by the decimal-shift corrector.
    final_df : pandas.DataFrame | None
        Aggregated daily output after run completes, or None before run.

    Methods
    -------
    run() -> tuple[pd.DataFrame, list[str], list[str]]
        Execute the full pipeline over all chunks and return the final daily frame
        plus the two CUSIP lists.
    export_outputs(...) -> dict[str, str]
        Write outputs such as daily frame and CUSIP lists to export_dir if set.
    _log_filter(df_before, df_after, stage, chunk_id) -> None
        Internal audit logger hook. Appends one row to audit_records.
    _load_chunk(cusips: list[str]) -> pd.DataFrame
        Query TRACE for a given list of CUSIPs.
    _process_chunk(trace: pd.DataFrame, chunk_id: int) -> pd.DataFrame
        Apply cleaning, agency pass, decimal shift, and price change flags for one chunk.
    _aggregate_daily(trace: pd.DataFrame) -> pd.DataFrame
        Build daily metrics per cusip_id and date.
    """
    # ---------------------------------------------------------------------
    def __init__(
        self,
        wrds_username: str,
        *,
        output_format: str = "csv",
        chunk_size: int = 250,
        clean_agency: bool = True,
        volume_filter: float | tuple[str, float] = ("dollar", 10000.0),
        trade_times: list[str] | None = None,
        calendar_name: str | None = None,
        out_dir: str | Path = ".",
        log_level: int = logging.INFO,
        ds_params: dict | None = None,
        bb_params: dict | None = None,
        init_error_params: dict | None = None,
        filters: dict | None = None,
        fisd_params: dict | None = None
    ) -> None:
        # user options
        self.wrds_username = wrds_username
        self.output_format = output_format.lower()
        self.chunk_size    = int(chunk_size)
        self.clean_agency  = clean_agency
        vkind, vthr = _normalize_volume_filter(volume_filter)
        self.volume_filter: tuple[str, float] = (vkind, vthr)  # canonical tuple
        self.volume_filter_kind: str = vkind                   # convenience
        self.volume_filter_threshold: float = vthr             # convenience
        self.trade_times   = trade_times
        self.calendar_name = calendar_name
        self.out_dir       = Path(out_dir)
        self.ds_params = ds_params or {}
        self.bb_params = bb_params or {}
        self.init_error_params = init_error_params or {}
        self.filters    = filters or {}
        self.fisd_params = fisd_params or {}

        self.out_dir = Path(out_dir).expanduser()     

        # handle "" or Path("")  
        if not self.out_dir.as_posix():      # empty string check
            self.out_dir = Path.cwd()

        # turn it into an absolute path
        self.out_dir = self.out_dir.resolve()

        # sanity checks
        if self.output_format not in {"csv", "parquet"}:
            raise ValueError("output_format must be 'csv' or 'parquet'")

        # logging
        self.logger              = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # runtime state
        self.db: wrds.Connection | None = None
        self.audit_records:      List[Dict] = []
        self.fisd_audit_records: List[Dict] = []
        self.ct_audit_records:   List[Dict] = []
        self.bounce_back_cusips_all: list[str] = []
        self.decimal_shift_cusips_all: list[str] = []
        self.init_price_cusips_all: list[str] = []  

    # ---------------------------------------------------------------------
    # -------------- MAIN --------------
    # ---------------------------------------------------------------------
    def CreateDailyEnhancedTRACE(self):
        """Run the full pipeline and return the three core DataFrames."""
        try:
            self._connect_wrds()
            fisd, fisd_off = self._build_fisd()
            cusip_chunks   = self._make_cusip_chunks(fisd)
            # TODO
            # cusip_chunks = cusip_chunks[:20]
            all_data = self._run_clean_trace(cusip_chunks, fisd_off)
            self._export(all_data, fisd)
            return all_data
        finally:
            self._disconnect_wrds()

    # ---------------------------------------------------------------------
    # -------------  helpers (underscored) -------------------------
    # ---------------------------------------------------------------------
    def _connect_wrds(self) -> None:
        self.logger.info("Connecting to WRDS ...")
        self.db = wrds.Connection(wrds_username=self.wrds_username)

        # expose shared audit lists to helper functions that expect globals
        global audit_records, fisd_audit_records, ct_audit_records
        audit_records       = self.audit_records
        fisd_audit_records  = self.fisd_audit_records
        ct_audit_records    = self.ct_audit_records

    def _disconnect_wrds(self) -> None:
        if self.db is not None:
            self.db.close()
            self.logger.info("WRDS session closed.")

    def _build_fisd(self):
        self.logger.info("Filtering FISD universe ...")
        return build_fisd(self.db, params=self.fisd_params)  # uses global log helpers

    def _make_cusip_chunks(self, fisd: pd.DataFrame):
        self.logger.info("Creating CUSIP batches ...")
        cusips = list(fisd["complete_cusip"].unique())

        def divide_chunks(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i : i + n]

        return list(divide_chunks(cusips, self.chunk_size))

    def _run_clean_trace(self, cusip_chunks, fisd_off):
        self.logger.info("Running TRACE cleaning loop ...")
        all_data, bb_list, ds_list, ie_list = clean_trace_data(
            self.db,
            cusip_chunks,
            fisd_off,
            clean_agency=self.clean_agency,
            fetch_fn=self._raw_sql_with_retry,
            volume_filter=self.volume_filter,
            trade_times=self.trade_times,
            calendar_name=self.calendar_name,
            ds_params=self.ds_params,
            bb_params=self.bb_params,
            init_error_params=self.init_error_params,
            filters=self.filters
        )

        # accumulate across all chunks/runs
        if bb_list:
            self.bounce_back_cusips_all.extend(bb_list)
        if ds_list:
            self.decimal_shift_cusips_all.extend(ds_list)
        if ie_list:
            self.init_price_cusips_all.extend(ie_list)

        return all_data

    def _export(self, all_data: pd.DataFrame, fisd_df: pd.DataFrame):
        self.logger.info("Exporting results ...")
        # Route Enhanced outputs into a dedicated subfolder
        out_sub = self.out_dir / "enhanced"
        out_sub.mkdir(parents=True, exist_ok=True)

        export_trace_dataframes(
            all_data,
            fisd_df,
            self.ct_audit_records,
            self.audit_records,
            output_format=self.output_format,
            out_dir=out_sub,
            bounce_back_cusips=self.bounce_back_cusips_all,
            decimal_shift_cusips=self.decimal_shift_cusips_all,
            init_price_cusips=self.init_price_cusips_all,
            fisd_audit_records=self.fisd_audit_records,
            stamp=RUN_STAMP,
        )

        
    def _reconnect_wrds(self):
        try:
            if self.db is not None:
                try:
                    self.db.close()
                except Exception:
                    pass
            self.logger.info("Reconnecting to WRDS ...")
            self.db = wrds.Connection(wrds_username=self.wrds_username)
        except Exception as e:
            self.logger.exception("WRDS reconnect failed")
            raise
    
    def _raw_sql_with_retry(self, sql: str, params=None, *, max_retries=3, base_sleep=2.0):
        """
        Execute SQL with automatic reconnection on transient connection errors.
        Retries on psycopg2.OperationalError and SQLAlchemy OperationalError.
        """
        import time
        from sqlalchemy.exc import OperationalError as SAOperationalError
        try:
            import psycopg2
            from psycopg2 import OperationalError as PGOperationalError
        except Exception:
            PGOperationalError = tuple()  # best effort
    
        attempt = 0
        while True:
            try:
                # quick keepalive ping before heavy query (optional, cheap)
                try:
                    _ = self.db.raw_sql("SELECT 1")
                except Exception:
                    # if ping itself failed, reconnect then continue
                    self._reconnect_wrds()
                return self.db.raw_sql(sql, params=params)
            except (SAOperationalError, PGOperationalError) as e:
                attempt += 1
                # only retry on "connection closed"/"SSL connection" patterns
                msg = str(e).lower()
                transient = ("ssl connection has been closed" in msg or
                             "server closed the connection" in msg or
                             "connection not open" in msg or
                             "terminating connection" in msg or
                             "connection reset" in msg)
                if not transient or attempt > max_retries:
                    self.logger.exception("DB query failed (attempt %s/%s)", attempt, max_retries)
                    raise
                sleep_s = base_sleep * (2 ** (attempt - 1))
                self.logger.warning("DB connection issue (%s). Reconnecting and retrying in %.1fs ...", e.__class__.__name__, sleep_s)
                time.sleep(sleep_s)
                self._reconnect_wrds()




# -------------------------------------------------------------------------
# -------------- _RUN ------------------------------
# -------------------------------------------------------------------------
def CreateDailyEnhancedTRACE(
    wrds_username: str,
    **kwargs,
):
    """
    Functional wrapper
    """
    
    import logging

    _configure_root_logger(level=logging.INFO)

    import sys, platform
    import pandas as pd
    import numpy as np
    
    logging.info("Python %s @ %s", platform.python_version(), sys.executable)
    logging.info("pandas %s (%s)", pd.__version__, pd.__file__)
    logging.info("numpy  %s (%s)", np.__version__, np.__file__)
    logging.info("wrds   %s (%s)", wrds.__version__, wrds.__file__)
    logging.info("pyarrow %s (%s)", pa.__version__, pa.__file__)
    logging.info("pandas_market_calendars %s (%s)", mcal.__version__, mcal.__file__)

    return ProcessEnhancedTRACE(wrds_username, **kwargs).CreateDailyEnhancedTRACE()

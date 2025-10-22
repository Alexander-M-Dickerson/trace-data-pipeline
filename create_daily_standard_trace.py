# -*- coding: utf-8 -*-
"""
create_daily_standard_trace
========================
Pulls, cleans, and saves TRACE Standard/144A data in daily-frequency panels.

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

# Convenience wrapper:  filter with a boolean mask --------------------------
def filter_with_log(df: pd.DataFrame,
                    mask: pd.Series,
                    stage: str,
                    chunk_id: int) -> pd.DataFrame:
    before = df
    after  = df.loc[mask].copy()          # copy avoids chained-assignment traps
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
def add_seq(df: pd.DataFrame, by: list, seq_name: str) -> pd.DataFrame:
    """Add 1-based sequence number within groups (like SAS BY-group + FIRST.)"""
    df[seq_name] = (
        df
          .groupby(by, sort=False)      # keep original order, avoids full sort
          .cumcount()
          .add(1)                       # 1, 2, 3...  (matches SAS)
    )
    return df
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
def clean_reversal(clean4: pd.DataFrame) -> pd.DataFrame:
    """
    Remove reversal trades in TRACE Standard data (SAS Step 4 from WRDS) and keep only
    unmatched non-reversal trades that correspond to surviving headers (SAS Step 5 from WRDS).

    What it does
    ------------
    A) Build a reversal header table from rows with asof_cd == "R", keyed on:
       cusip_id, bond_sym_id, trd_exctn_dt, entrd_vol_qt, rptd_pr,
       rpt_side_cd, contra_party_type, plus trd_exctn_tm and msg_seq_nb.
       It also assigns a within-group sequence number to replicate the SAS
       6-key plus sequence matching.

    B) Build the non-reversal body by excluding rows with asof_cd in ["R","X","D"].
       Create a matching header view and assign the same within-group sequence
       to enable a 6-key plus sequence match.

    C) Mark header rows that have a matching reversal partner using the
       6-key plus sequence. Retain only header rows that do not have a match.

    D) Keep final trades whose 8-key
       [cusip_id, trd_exctn_dt, trd_exctn_tm, entrd_vol_qt, rptd_pr,
        rpt_side_cd, contra_party_type, msg_seq_nb]
       appears in the surviving header set. Drop duplicates to mirror the
       SAS DISTINCT behavior.

    Expected input columns
    ----------------------
    Required:
      - cusip_id, bond_sym_id, trd_exctn_dt, trd_exctn_tm
      - entrd_vol_qt, rptd_pr, rpt_side_cd, contra_party_type
      - asof_cd, msg_seq_nb
    Extra columns are preserved and passed through.

    Parameters
    ----------
    clean4 : pandas.DataFrame
        TRACE rows after prior cleaning steps but before reversal removal.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with reversals removed and only unmatched non-reversal
        trades retained. Returns an empty DataFrame if the input is empty.

    Notes
    -----
    - trd_exctn_dt should be a date or datetime dtype upstream. If it is a string,
      ensure consistent formatting before calling this function.
    - trd_exctn_tm is used as a key; it should be a consistently formatted string.
    """
    # ------------------------------------------------------------------
    # 0)  Column sets used repeatedly
    # ------------------------------------------------------------------
    keys6 = ['cusip_id','bond_sym_id','trd_exctn_dt',
             'entrd_vol_qt','rptd_pr','rpt_side_cd','contra_party_type']
    
    header_cols = keys6 + ['trd_exctn_tm','msg_seq_nb']          # -> _clean5_header
    join8       = ['cusip_id','trd_exctn_dt','trd_exctn_tm',
                   'entrd_vol_qt','rptd_pr','rpt_side_cd',
                   'contra_party_type','msg_seq_nb']             # final filter

    # ------------------------------------------------------------------
    # Step 4-A: reversal header   (~ DATA _rev_header)
    # ------------------------------------------------------------------
    rev_header = (
        clean4
          .loc[clean4['asof_cd'] == 'R',
               keys6 + ['trd_exctn_tm']]         # keep only needed columns
          .copy()
    )
    
    # Sort reversal records before adding sequence
    rev_header = rev_header.sort_values(
        keys6 + ['trd_exctn_tm']  # Sort by 6 keys + time
    ).copy()
    
    rev_header = add_seq(rev_header, keys6, 'seq')

    # ------------------------------------------------------------------
    # Step 4-B: non-reversal body  (~ _clean5 / _clean5_header)
    # ------------------------------------------------------------------
    clean5 = clean4.loc[~clean4['asof_cd'].isin(['R','X','D'])].copy()

    clean5_header = clean5[header_cols].copy()
   
    # Sort non-reversal records before adding sequence
    clean5_header = clean5_header.sort_values(
        keys6 + ['trd_exctn_tm', 'msg_seq_nb']  # Sort by 6 keys + time + msg_seq
    ).copy()
    
    clean5_header = add_seq(clean5_header, keys6, 'seq6')

    # ------------------------------------------------------------------
    # Step 4-C: mark header rows that have a matching reversal (6-key + seq)
    # ------------------------------------------------------------------
    # Align column names for a vectorised merge
    rev_header6 = rev_header.rename(columns={'seq': 'seq6'})

    merge_keys = keys6 + ['seq6']                 # 6-key match in SAS

    clean5_header = clean5_header.merge(
        rev_header6[merge_keys],                  # RHS slimmed to keys
        on=merge_keys,
        how='left',
        indicator='rev_match'                     # keeps track of hits
    )

    # Keep ONLY rows that did *not* find a reversal partner
    clean5_header = clean5_header[clean5_header['rev_match'] == 'left_only']
    clean5_header = clean5_header.drop(columns='rev_match')

    # ------------------------------------------------------------------
    # Step 4-D: final SAS proc SQL "distinct a. join
    #           – we need rows in clean5 whose 8-key appears in header
    #           – SQL DISTINCT -> drop_duplicates()
    # ------------------------------------------------------------------
    #   Drop dup keys in header first: identical to SAS LEFT JOIN DISTINCT
    clean5_header_keys = clean5_header[join8].drop_duplicates()

    #   Build a multi-index for O(1) membership checks
    mask = (
        pd.MultiIndex.from_frame(clean5[join8])
        .isin(pd.MultiIndex.from_frame(clean5_header_keys))
    )
    
    clean6 = clean5.loc[mask].copy()
    clean6 = clean6.drop_duplicates().reset_index(drop=True)   # SQL DISTINCT

    return clean6
# -------------------------------------------------------------------------
def compute_trace_all_metrics(trace):
    """
    Aggregate TRACE trades to a daily (cusip_id, trd_exctn_dt) panel with
    price, volume, and customer-side bid/ask summaries.

    Summary
    -------
    - Prices: equal-weighted, value-weighted by dollar volume, par-value-weighted,
      earliest trade of the day, latest trade of the day, and a trade count.
    - Volumes: daily totals for quantity and dollar volume, reported in millions.
    - Bid/Ask: customer-side value-weighted prices based on customer buys (bid)
      and customer sells (ask).

    Expected input columns
    ----------------------
    Required
      - 'cusip_id', 'trd_exctn_dt', 'rptd_pr', 'entrd_vol_qt'
    For bid/ask construction
      - 'rpt_side_cd'           # 'B' for customer buy, 'S' for customer sell
      - 'contra_party_type'     # 'C' indicates customer
    Notes
      - If 'dollar_vol' is missing, it is created as: rptd_pr * entrd_vol_qt / 100.
      - If no customer-side trades exist for a day, bid/ask fields return NaN.

    Returns
    -------
    pandas.DataFrame
        One row per (cusip_id, trd_exctn_dt) with:
        - Keys:
            'cusip_id', 'trd_exctn_dt'
        - Prices:
            'prc_ew'       : equal-weighted price
            'prc_vw'       : value-weighted price (weights proportional to dollar_vol)
            'prc_vw_par'   : par-value-weighted price (weights proportional to entrd_vol_qt)
            'prc_earliest' : earliest trade price in the day
            'prc_latest'   : latest trade price in the day
            'trade_count'  : number of trades in the day
        - Volumes (in millions):
            'qvolume'      : sum(entrd_vol_qt) / 1e6
            'dvolume'      : sum(dollar_vol)   / 1e6
        - Bid/Ask (customer only, value-weighted):
            'prc_bid', 'prc_ask', 'bid_count', 'ask_count'

    Implementation notes
    --------------------
    - Vectorized groupby aggregations; no per-row loops.
    - Outputs from price, volume, and bid/ask blocks are merged and sorted by
      ['cusip_id', 'trd_exctn_dt'].
    - If the input is empty, returns an empty DataFrame with the expected columns.
    """
        
    # Ensure we have dollar_vol
    if 'dollar_vol' not in trace.columns:
        trace['dollar_vol'] = trace['rptd_pr'] * trace['entrd_vol_qt'] / 100
           
    # Split into bid and ask dataframes
    # Customer Only #
    _bid = trace[((trace['rpt_side_cd'] == 'B')&(trace['contra_party_type'] == 'C'))].copy()
    _ask = trace[((trace['rpt_side_cd'] == 'S')&(trace['contra_party_type'] == 'C'))].copy()
        
    #--------------------------------------------------------------------------
    # 1. Compute PricesAll (Equal-weighted, volume-weighted prices and trade count)
    #--------------------------------------------------------------------------
    
    # Precompute the weighted products
    trace['dollar_weighted_price'] = trace['rptd_pr'] * trace['dollar_vol']
    trace['volume_weighted_price'] = trace['rptd_pr'] * trace['entrd_vol_qt']
    
    # Group and aggregate in one step
    agg_dict = {
        'rptd_pr': ['mean', 'first', 'last', 'max', 'min', 'count'],
        'dollar_vol': 'sum',
        'entrd_vol_qt': 'sum',
        'dollar_weighted_price': 'sum',
        'volume_weighted_price': 'sum'
    }
    
    results = trace.groupby(['cusip_id', 'trd_exctn_dt']).agg(agg_dict)
    
    # Flatten the column names
    results.columns = ['_'.join(col).strip() for col in results.columns.values]
    
    # Calculate the weighted prices
    results['prc_vw'] = results['dollar_weighted_price_sum'] / results['dollar_vol_sum']
    results['prc_vw_par'] = results['volume_weighted_price_sum'] / results['entrd_vol_qt_sum']
    
    # Create the final PricesAll dataframe
    PricesAll = results.rename(columns={
        'rptd_pr_mean': 'prc_ew',
        'rptd_pr_first': 'prc_first',
        'rptd_pr_last': 'prc_last',
        'rptd_pr_max': 'prc_hi',
        'rptd_pr_min': 'prc_lo',
        'rptd_pr_count': 'trade_count'
    }).reset_index()
    
    # Select columns
    PricesAll = PricesAll[['cusip_id', 'trd_exctn_dt', 'prc_ew', 'prc_vw', 'prc_vw_par',
                           'prc_first', 'prc_last', 'prc_hi', 'prc_lo', 'trade_count']]
    
    #--------------------------------------------------------------------------
    # 2. Compute VolumesAll - Simple re-computation approach
    #--------------------------------------------------------------------------
    
    # Simple aggregation for volumes
    VolumesAll = trace.groupby(['cusip_id', 'trd_exctn_dt']).agg({
        'entrd_vol_qt': 'sum',
        'dollar_vol': 'sum'
    }).reset_index()
    
    # Scale volumes by 1 million
    VolumesAll['entrd_vol_qt'] = VolumesAll['entrd_vol_qt'] / 1000000
    VolumesAll['dollar_vol']   = VolumesAll['dollar_vol']   / 1000000
    
    # Rename columns
    VolumesAll = VolumesAll.rename(columns={
        'entrd_vol_qt': 'qvolume',
        'dollar_vol': 'dvolume'
    })
    
    #--------------------------------------------------------------------------
    # 3. Compute Bid-Ask metrics
    #--------------------------------------------------------------------------
    
    # Initialize an empty result dataframe
    prc_BID_ASK = pd.DataFrame(columns=['cusip_id', 'trd_exctn_dt', 'prc_bid', 
                                        'prc_ask', 'bid_count', 'ask_count'])
    
    # Process bid data
    if not _bid.empty:
        # Sort by cusip and execution date
        _bid = _bid.sort_values(['cusip_id', 'trd_exctn_dt'])
        
        # Compute sum of volumes per group for normalization
        bid_vol_sums = _bid.groupby(['cusip_id', 'trd_exctn_dt'])['entrd_vol_qt'].transform('sum')
        
        # Calculate value weights directly (vectorized operation)
        _bid['value_weights'] = _bid['entrd_vol_qt'] / bid_vol_sums
        
        # Calculate weighted price products
        _bid['weighted_price'] = _bid['rptd_pr'] * _bid['value_weights']
        
        # Group and aggregate
        bid_agg = _bid.groupby(['cusip_id', 'trd_exctn_dt']).agg({
            'rptd_pr': 'count',
            'weighted_price': 'sum'
        })
        
        # Rename columns for clarity
        bid_agg.columns = ['bid_count', 'prc_bid']
        
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
        
        # Compute sum of volumes per group for normalization
        ask_vol_sums = _ask.groupby(['cusip_id', 'trd_exctn_dt'])['entrd_vol_qt'].transform('sum')
        
        # Calculate value weights directly (vectorized operation)
        _ask['value_weights'] = _ask['entrd_vol_qt'] / ask_vol_sums
        
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

    # optional: sort rows for tidy output
    merged = merged.sort_values(['cusip_id','trd_exctn_dt']).reset_index(drop=True)
    return merged    
# -------------------------------------------------------------------------
def clean_trace_data(
    db,
    cusip_chunks,
    fisd_off,
    *,
    fetch_fn=None,
    clean_agency: bool = True,
    start_date: str | None = None,
    data_type: str = "standard",
    volume_filter: float | tuple[str, float] = ("dollar", 10000.0),
    trade_times: list[str] | None = None,  
    calendar_name: str | None = None,
    ds_params: dict | None = None,
    bb_params: dict | None = None,
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
    )
    f = {**FILTER_DEFAULTS, **(filters or {})}
    """
    Fetch, clean, and aggregate TRACE Standard style data in CUSIP chunks, then
    build daily metrics. Optionally applies decimal shift correction and bounce
    back price change filtering before aggregation.

    What it does
    ------------
    - Loads intraday TRACE rows for each chunk of CUSIPs from the database.
    - Applies pre and post 2012 cleaning rules and optional agency de-duplication.
    - Applies volume and optional intraday time window filters.
    - Optionally snaps to a market calendar if a calendar name is provided.
    - Optionally runs decimal_shift_corrector and flag_price_change_errors.
    - Aggregates to one row per cusip_id and trade date with daily metrics.
    - Tracks two CUSIP lists: those with bounce back flags and those with
      decimal shift corrections.

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection. Must support raw_sql or an equivalent query method.
    cusip_chunks : list of list of str
        Lists of CUSIPs to process per chunk.
    start_date : str or None, default None
        Optional lower bound for trade dates when querying. Use ISO format
        YYYY-MM-DD. If None, no explicit lower bound filter is applied.
    data_type : str, default "standard"
        Logical dataset selector. Examples: "standard" for TRACE Standard or
        "144a" if your code routes to the Rule 144A table. The function body
        is expected to route queries and minor schema differences based on this.
    clean_agency : bool, default True
        Apply agency de-duplication during cleaning.
    volume_filter : float | tuple[str, float], default ("dollar", 10000.0)
        If float/int: legacy dollar-volume threshold.
        If tuple: ("dollar", x) uses dollar_vol >= x; ("par", x) uses entrd_vol_qt >= x.
    trade_times : list[str] or None, default None
        Optional inclusive intraday window as ["HH:MM:SS", "HH:MM:SS"] used to
        retain trades within the specified time of day.
    calendar_name : str or None, default None
        Optional market calendar name used to keep only valid trading sessions
        if your code supports calendar filtering.
    ds_params : dict or None, default None
        Keyword overrides forwarded to decimal_shift_corrector.
        Examples include factors, tol_pct_good, tol_abs_good, tol_pct_bad,
        low_pr, high_pr, anchor, window, improvement_frac, par_snap, par_band,
        output_type.
    bb_params : dict or None, default None
        Keyword overrides forwarded to flag_price_change_errors.
        Examples include threshold_abs, lookahead, max_span, window,
        back_to_anchor_tol, candidate_slack_abs, reassignment_margin_abs,
        use_unique_trailing_median, par_spike_heuristic, par_level,
        par_equal_tol, par_min_run, par_cooldown_after_flag.

    Returns
    -------
    tuple[pandas.DataFrame, list[str], list[str]]
        final_df
            Daily metrics DataFrame per cusip_id and date, sorted and ready to export.
        bb_cusips_all
            Unique CUSIPs that had at least one row flagged by the bounce back
            price change filter in any processed chunk.
        dec_shift_cusips_all
            Unique CUSIPs that had at least one row corrected by the decimal
            shift corrector in any processed chunk.

    Notes
    -----
    - The function should sort within groups by date and, when present, time.
    - The implementation is expected to branch on data_type for table names and
      minor field differences, but the interface is consistent across types.
    - If no rows are returned for a chunk, the implementation should log and
      continue. The function should return empty outputs if all chunks are empty.
    """
    # ---------- 0. validate data_type ----------
    data_type = data_type.lower()
    if data_type == "standard":
        table_name = "trace.trace"
    elif data_type == "144a":
        table_name = "trace.trace_btds144a"
    else:
        raise ValueError("data_type must be 'standard' or '144a'")

    all_super_list       = []
    bb_cusips_all        = []
    dec_shift_cusips_all = []
    
    sort_cols = ["cusip_id","trd_exctn_dt","trd_exctn_tm", "msg_seq_nb"]

    # ---------- 1. chunk loop ----------
    for i, temp_list in enumerate(cusip_chunks, start=1):
        start_time = time.time()
        logging.info(f"Processing chunk {i} of {len(cusip_chunks)}")
        print(f"Processing chunk {i} of {len(cusip_chunks)}")

        temp_tuple = tuple(temp_list)

        # ---------- 2. build query ----------
        sql_query = f'''
            SELECT cusip_id, bond_sym_id, bsym, trd_exctn_dt, trd_exctn_tm,
                   msg_seq_nb, trc_st, wis_fl, cmsn_trd, ascii_rptd_vol_tx,
                   rptd_pr, yld_pt, asof_cd, side, diss_rptg_side_cd,
                   orig_msg_seq_nb, orig_dis_dt, rptg_party_type,
                   contra_party_type
            FROM {table_name}
            WHERE cusip_id IN %(cusip_id)s
              AND cusip_id IS NOT NULL
              AND TRIM(cusip_id) != ''
        '''
        
        params = {"cusip_id": temp_tuple}
        if start_date:
            sql_query += " AND trd_exctn_dt >= %(start_date)s"
            params["start_date"] = start_date

        # ---------- 3. fetch ----------
        trace = fetch_fn(sql_query, params=params)
                
        logging.info(f"Chunk {i}: Retrieved {len(trace)} rows from WRDS")
        
        if len(trace) == 0:
            continue
        
        trace["rptd_pr"] = trace["rptd_pr"].astype("float64").round(6)
        trace = trace.drop(columns=["index"], errors="ignore").reset_index(drop=True)
        
        # Initial log for cleaning
        log_filter(trace, trace, "start", i)
        
        # Filter 1: Dick-Nielsen
        if f["dick_nielsen"]:
            clean_chunk = clean_trace_standard_chunk(
                trace,
                chunk_id     = i,
                logger       = log_ct_filter
            )
            log_filter(trace, clean_chunk, "dick_nielsen_filter", i)
            trace = clean_chunk.copy()
            del clean_chunk
        else:
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
        
            trace, n_rows_replaced, replace_cusips = decimal_shift_corrector(
                trace.sort_values(["cusip_id", "trd_exctn_dt", "trd_exctn_tm"]),
                **_ds
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
                start_date=start_date,
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
                mask = trace['entrd_vol_qt'] >= vthr
                stage_name = "volume_filter[par]"
            trace = filter_with_log(trace, mask, stage_name, i)
        else:
            log_filter(trace, trace, "volume_filter (skipped)", i)

        # Pre BB sort #
        trace = trace.sort_values(sort_cols, kind="mergesort", ignore_index=True)
        # Filter 7: Bounce-back    
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
        
            trace = flag_price_change_errors(
                trace.sort_values(["cusip_id", "trd_exctn_dt", "trd_exctn_tm"]),
                **_bb
            )
            bb_cusips = list(trace[trace['filtered_error'] == 1]['cusip_id'].unique())
            if bb_cusips:
                bb_cusips_all.extend([str(c) for c in bb_cusips])
        
            trace = filter_with_log(trace, trace['filtered_error'] == 0, "bounce_back_filter", i)
            trace.drop(['delta_rptd_pr', 'baseline_trailing', 'filtered_error'], axis=1, inplace=True)
            gc.collect()
        else:
            log_filter(trace, trace, "bounce_back_filter (skipped)", i)                          
        
        # Filter 8: Bounce-back    
        if f["yld_price_filter"]:
            mask = (trace["rptd_pr"] != trace["yld_pt"]) | trace["yld_pt"].isna()
            trace = filter_with_log(trace, mask, "price_yld_filter", i)
        else:
            log_filter(trace, trace, "price_yld_filter (skipped)", i)

        
        # Filter 9: Amount-outstanding vs volume filter                 
        trace = trace.merge(fisd_off, how="left", on="cusip_id")
        if f["amtout_volume_filter"]:
            trace = filter_with_log(
                trace,
                trace['entrd_vol_qt'] < trace['offering_amt']*1000*0.50,
                "volume_offamt_filter",
                i
            )
        else:
            log_filter(trace, trace, "volume_offamt_filter (skipped)", i)

        # Filter 10: Trade execution date <= maturity filter                   
        if f["trd_exe_mat_filter"]:
            trace = filter_with_log(
                trace,
                trace['trd_exctn_dt'] <= trace['maturity'],
                "exctn_mat_dt_filter",
                i
            )
        else:
            log_filter(trace, trace, "exctn_mat_dt_filter (skipped)", i)
            
        #* ************************************** */
        #* DAILY AGGREGATION                      */
        #* ************************************** */ 
     
        # Calculate all metrics with a single function call
        # PricesAll, VolumesAll, prc_BID_ASK = compute_trace_all_metrics(trace)
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
        return final_df, bb_cusips_all, dec_shift_cusips_all               
    else:
        return pd.DataFrame(), bb_cusips_all, dec_shift_cusips_all                    
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
            * dec_shift_flag   (int8)   - 1 if corrected candidate accepted
            * dec_shift_factor (float)  - chosen factor (1.0 if no change)
            * suggested_price  (float)  - corrected price proposal
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
    sort_cols = [id_col, date_col] + ([time_col] if time_col and time_col in df.columns else [])
    out = df.sort_values(sort_cols).reset_index(drop=True).copy()

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

    # End date defaults to today 
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
def clean_trace_standard_chunk(trace, *, chunk_id=None, logger=None):
    """
    Clean one chunk of TRACE Standard trades and return a filtered DataFrame.
    Applies base filters, legacy reversal removal, and optional agency
    de-duplication. Designed to run inside a loop over CUSIP chunks prior to
    decimal-shift and bounce-back steps.

    What it does
    ------------
    - Normalizes key dtypes for dates, times, and numeric fields.
    - Drops rows with missing or empty cusip_id.
    - Applies standard status and condition code filters common in academic use.
    - Removes reversals using clean_reversal (SAS Step 4 and Step 5 logic).
    - Optionally removes agency duplicate prints via clean_agency_transactions.
    - Returns the cleaned result for the chunk.

    Expected input columns
    ----------------------
    Required:
      cusip_id, bond_sym_id, trd_exctn_dt, trd_exctn_tm,
      entrd_vol_qt, rptd_pr, rpt_side_cd, contra_party_type,
      asof_cd, msg_seq_nb
    Extra columns are preserved and passed through.

    Parameters
    ----------
    trace : pandas.DataFrame
        Raw TRACE Standard rows for a set of CUSIPs in this chunk.
    chunk_id : int or None, optional
        Identifier used for logs and audit entries.
    clean_agency : bool, default True
        If True, apply the agency de-duplication pass after reversal cleaning.
    logger : callable or None, optional
        Function used to append audit rows. It is called as:
            logger(df_before, df_after, stage, chunk_id)
        Suggested stage names:
            "std_base_filters", "std_reversals", "std_agency_cleaning".

    Returns
    -------
    pandas.DataFrame
        Cleaned TRACE Standard rows for the input chunk. A new DataFrame is
        returned; the input is not modified in place. If the input is empty,
        an empty DataFrame is returned.

    Notes
    -----
    - Downstream code typically re-sorts by cusip_id, trd_exctn_dt, trd_exctn_tm.
    - Ensure trd_exctn_dt is date or datetime and trd_exctn_tm is a consistent
      string format upstream to avoid key-matching issues.
    - This function is the Standard counterpart to the Enhanced cleaner and is
      intended to be followed by aggregation to daily metrics.
    """
    # Filter out empty CUSIPs
    # Store original for logging
    original_trace = trace['cusip_id'].copy()
     
    # Convert date strings to datetime objects
    trace['trd_exctn_dt'] = pd.to_datetime(trace['trd_exctn_dt'])
    if 'orig_dis_dt' in trace.columns:
        trace['orig_dis_dt'] = pd.to_datetime(trace['orig_dis_dt'])
    
    # Step 1: Reassign Volume and Other Values
    trace['ascii_rptd_vol_tx'] = trace['ascii_rptd_vol_tx'].replace('5MM+', '5000000')
    trace['ascii_rptd_vol_tx'] = trace['ascii_rptd_vol_tx'].replace('1MM+', '1000000')
    trace['entrd_vol_qt'] = pd.to_numeric(trace['ascii_rptd_vol_tx'], errors='coerce')
    
    # Convert Different TRC_ST values to be uniform
    trace['trc_st'] = trace['trc_st'].replace(['G', 'M'], 'T')
    trace['trc_st'] = trace['trc_st'].replace(['H', 'N'], 'C')
    trace['trc_st'] = trace['trc_st'].replace(['I', 'O'], 'W')
    
    # Rename DISS_RPTG_SIDE_CD to RPT_SIDE_CD
    if 'diss_rptg_side_cd' in trace.columns:
        trace = trace.rename(columns={'diss_rptg_side_cd': 'rpt_side_cd'})
    
    # Split data based on trc_st
    trace_C = trace[trace['trc_st'] == 'C'].copy()
    trace_W = trace[trace['trc_st'] == 'W'].copy()
    trace_T = trace[trace['trc_st'] == 'T'].copy()
    
    # Step 2: Remove Cancellation Cases (C)
    # Composite merge (current solution)
    if not trace_C.empty and not trace_T.empty:
        before = trace_T
        trace_T['cancel_key'] = trace_T['cusip_id'] + '_' + trace_T['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' + \
                               trace_T['trd_exctn_tm'].astype(str) + '_' + trace_T['rptd_pr'].astype(str) + '_' + \
                               trace_T['entrd_vol_qt'].astype(str) + '_' + trace_T['msg_seq_nb'].astype(str)
        trace_C['cancel_key'] = trace_C['cusip_id'] + '_' + trace_C['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' + \
                               trace_C['trd_exctn_tm'].astype(str) + '_' + trace_C['rptd_pr'].astype(str) + '_' + \
                               trace_C['entrd_vol_qt'].astype(str) + '_' + trace_C['orig_msg_seq_nb'].astype(str)
        clean2 = trace_T[~trace_T['cancel_key'].isin(trace_C['cancel_key'])].copy()
        clean2.drop(columns=['cancel_key'], inplace=True)
        if logger: logger(original_trace, clean2, "remove_cancellations", chunk_id)
    else:
        clean2 = trace_T.copy()
    ##########################
    # Merge based solution #    
    # if not trace_C.empty and not trace_T.empty:
    #     before = trace_T
        
    #     # Perform a proper anti-join using merge
    #     trace_T_merged = pd.merge(
    #         trace_T,
    #         trace_C[['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'rptd_pr', 
    #                  'entrd_vol_qt', 'orig_msg_seq_nb']],
    #         left_on=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'rptd_pr', 
    #                  'entrd_vol_qt', 'msg_seq_nb'],
    #         right_on=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'rptd_pr', 
    #                   'entrd_vol_qt', 'orig_msg_seq_nb'],
    #         how='left',
    #         indicator='_merge',
    #         suffixes=('', '_cancel')  # Add suffixes parameter
    #     )
        
    #     # Keep only records that didn't match (equivalent to trc_st_c = '')
    #     clean2 = trace_T_merged[trace_T_merged['_merge'] == 'left_only'].copy()
    #     clean2 = clean2.drop(columns=['_merge', 'orig_msg_seq_nb_cancel'])  # Drop the _cancel version
        
    #     if logger: logger(before, clean2, "remove_cancellations", chunk_id)
    # else:
    #     clean2 = trace_T.copy()
    ##########################        
    # Step 3: Remove Correction Cases (W)
    if not trace_W.empty:
        before = clean2
        w_msg = trace_W[['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb']].copy()
        w_msg['flag'] = 'msg'
        w_omsg = trace_W[['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'orig_msg_seq_nb']].copy()
        w_omsg = w_omsg.rename(columns={'orig_msg_seq_nb': 'msg_seq_nb'})
        w_omsg['flag'] = 'omsg'
        w_combined = pd.concat([w_msg, w_omsg], ignore_index=True)
        w_napp = w_combined.groupby(['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb']).size().reset_index(name='napp')
        w_mult = w_combined.drop_duplicates(['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb', 'flag'])
        w_ntype = w_mult.groupby(['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb']).size().reset_index(name='ntype')
        w_comb = pd.merge(w_napp, w_ntype, on=['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb'], how='left')
        w_keep = w_comb[(w_comb['napp'] == 1) | ((w_comb['napp'] > 1) & (w_comb['ntype'] == 1))]
        w_keep = pd.merge(w_keep, w_combined, on=['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb'], how='inner')
        w_keep['npair'] = w_keep.groupby(['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm']).transform('size') / 2
        w_keep1 = w_keep[w_keep['npair'] == 1].copy()
        w_keep1_pivot = w_keep1.pivot_table(index=['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm'], 
                                          columns='flag', values='msg_seq_nb', aggfunc='first').reset_index()
        w_keep1_pivot = w_keep1_pivot.rename(columns={'msg': 'msg_seq_nb', 'omsg': 'orig_msg_seq_nb'})
        w_keep2 = w_keep[(w_keep['npair'] > 1) & (w_keep['flag'] == 'msg')].copy()
        w_keep2 = pd.merge(w_keep2[['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb']], 
                          trace_W[['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb', 'orig_msg_seq_nb']], 
                          on=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb'], how='left')
        w_clean = pd.concat([w_keep1_pivot, w_keep2[['cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb', 'orig_msg_seq_nb']]], 
                           ignore_index=True)
        w_clean = w_clean.drop(columns=['bond_sym_id'])
        w_clean_full = pd.merge(w_clean, trace_W.drop(columns=['orig_msg_seq_nb']), 
                               on=['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'msg_seq_nb'], how='left')
        
        # Composite Keys (chosen method for now):      
        clean2['correction_key'] = clean2['cusip_id'] + '_' + clean2['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' + \
                                  clean2['msg_seq_nb'].astype(str)
                                  
        w_clean_full['correction_key'] = w_clean_full['cusip_id'] + '_' + w_clean_full['trd_exctn_dt'].dt.strftime('%Y%m%d') + '_' + \
                                        w_clean_full['orig_msg_seq_nb'].astype(str)
                                        
        clean3 = clean2[~clean2['correction_key'].isin(w_clean_full['correction_key'])].copy()
        ######################             
        # OR Left Merge with drop #
        # I have left this here for reference #              
        # clean2_merged = pd.merge(
        #     clean2,
        #     w_clean_full[['cusip_id', 'trd_exctn_dt', 'orig_msg_seq_nb']].rename(
        #         columns={'orig_msg_seq_nb': 'w_orig_msg_seq_nb'}  # Rename to avoid conflict
        #     ),
        #     left_on=['cusip_id', 'trd_exctn_dt', 'msg_seq_nb'],
        #     right_on=['cusip_id', 'trd_exctn_dt', 'w_orig_msg_seq_nb'],
        #     how='left',
        #     indicator='_merge'
        # )
        
        # clean3 = clean2_merged[clean2_merged['_merge'] == 'left_only'].copy()
        # clean3 = clean3.drop(columns=['_merge', 'w_orig_msg_seq_nb'])
        ######################
                
        # Step 3.10 (SAS): Replace with W Records 
        matched_t_keys = clean2['correction_key'][clean2['correction_key'].isin(w_clean_full['correction_key'])].tolist()
        
        w_to_add = w_clean_full[w_clean_full['correction_key'].isin(matched_t_keys)].copy()
        w_to_add = w_to_add.drop_duplicates(['cusip_id', 'trd_exctn_dt', 'msg_seq_nb', 'orig_msg_seq_nb', 'rptd_pr', 'entrd_vol_qt'])
        
        clean3.drop(columns=['correction_key'], inplace=True)
        w_to_add.drop(columns=['correction_key'], inplace=True)
        ######################
        
        # Step 3.10 (SAS): Replace with W Records 
        # I have left this here for reference #
        # First, identify which T records were matched (deleted)
        
        # clean2_matched = clean2[clean2['correction_key'].isin(w_clean_full['correction_key'])].copy()
        
        # Now find W records that match these deleted T records
        # Join w_clean_full to clean2_matched on the matching keys
        
        # w_to_add = pd.merge(
        #     w_clean_full,
        #     clean2_matched[['cusip_id', 'trd_exctn_dt', 'msg_seq_nb']].rename(
        #         columns={'msg_seq_nb': 'matched_msg_seq_nb'}
        #     ),
        #     left_on=['cusip_id', 'trd_exctn_dt', 'orig_msg_seq_nb'],
        #     right_on=['cusip_id', 'trd_exctn_dt', 'matched_msg_seq_nb'],
        #     how='inner'
        # ).drop(columns=['matched_msg_seq_nb'])
        
        # w_to_add = w_to_add.drop_duplicates(
        #     ['cusip_id', 'trd_exctn_dt', 'msg_seq_nb', 'orig_msg_seq_nb', 'rptd_pr', 'entrd_vol_qt']
        # )
        
        ###############
        # Step 3.11: Final Combination
        clean4 = pd.concat([clean3, w_to_add], ignore_index=True)
        if logger: logger(before, clean4, "remove_corrections", chunk_id)
        
    else:
        clean4 = clean2.copy()
    
    # Step 4: Remove Reversal Cases
    before = clean4
    clean6 = clean_reversal(clean4)
    if logger: logger(before, clean6, "remove_reversals", chunk_id)
    
    # Step 5: Clean Agency Transaction
    if 'side' in clean6.columns:
        before = clean6
        clean6['rpt_side_cd'] = clean6['rpt_side_cd'].fillna(clean6['side'])
        if logger: logger(before, clean6, "agency_fill_side", chunk_id)
    
    return clean6

# -------------------------------------------------------------------------
def build_fisd(db, params: dict | None = None, *, data_type: str = "standard"):
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

    # ---- 1) Pull raw FISD tables ------------------------------------
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
    fisd        = pd.merge(fisd_issue, fisd_issuer, on="issuer_id", how="left")

    # ---- 2) Start log -----------------------------------------------------
    log_fisd_filter(fisd, fisd, "start")

    # ---- 3) Currency (foreign_currency == 'N') ---------------------------
    if p["currency_usd_only"]:
        before = fisd
        fisd = fisd.loc[(fisd["foreign_currency"] == "N")]
        log_fisd_filter(before, fisd, "USD currency")
    else:
        log_fisd_filter(fisd, fisd, "USD currency (skipped)")

    # ---- 4) Fixed-rate ----------------------------------------------------
    if p["fixed_rate_only"]:
        before = fisd
        fisd = fisd.loc[fisd["coupon_type"] != "V"]
        log_fisd_filter(before, fisd, "fixed rate")
    else:
        log_fisd_filter(fisd, fisd, "fixed rate (skipped)")

    # ---- 5) Non-convertible ----------------------------------------------
    if p["non_convertible_only"]:
        before = fisd
        fisd = fisd.loc[fisd["convertible"] == "N"]
        log_fisd_filter(before, fisd, "non convertible")
    else:
        log_fisd_filter(fisd, fisd, "non convertible (skipped)")

    # ---- 6) Non-asset-backed ---------------------------------------------
    if p["non_asset_backed_only"]:
        before = fisd
        fisd = fisd.loc[fisd["asset_backed"] == "N"]
        log_fisd_filter(before, fisd, "non asset backed")
    else:
        log_fisd_filter(fisd, fisd, "non asset backed (skipped)")

    del before
    gc.collect()

    # ---- 7) Exclude bond types -------------------------------------------
    if p["exclude_bond_types"]:
        before = fisd
        exclude_btypes = set(p["excluded_bond_types"])
        fisd = fisd.loc[~fisd["bond_type"].isin(exclude_btypes)]
        log_fisd_filter(before, fisd, "exclude gov muni ABS types")
    else:
        log_fisd_filter(fisd, fisd, "exclude gov muni ABS types (skipped)")

    # ---- 9) Valid coupon frequency ---------------------------------------
    if p["valid_coupon_frequency_only"]:
        before = fisd
        invalid_freq = set(p["invalid_coupon_freq"])
        fisd = fisd.loc[~fisd["interest_frequency"].isin(invalid_freq)]
        log_fisd_filter(before, fisd, "valid coupon frequency")
    else:
        log_fisd_filter(fisd, fisd, "valid coupon frequency (skipped)")

    # ---- 10) Complete accrual fields -------------------------------------
    if p["require_accrual_fields"]:
        before = fisd
        date_cols = ["offering_date", "dated_date"]
        fisd[date_cols] = fisd[date_cols].apply(pd.to_datetime, errors="coerce")
        req_cols = date_cols + ["interest_frequency", "day_count_basis", "coupon_type", "coupon"]
        fisd = fisd.dropna(subset=req_cols)
        log_fisd_filter(before, fisd, "complete accrual fields")
    else:
        log_fisd_filter(fisd, fisd, "complete accrual fields (skipped)")

    # ---- 11) principal_amt == 1000 ---------------------------------------
    if p["principal_amt_eq_1000_only"]:
        before = fisd
        fisd = fisd.loc[fisd["principal_amt"] == 1000]
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
        fisd = fisd[fisd["equity_linked"] == 0].drop(columns="equity_linked")
        log_fisd_filter(before, fisd, "exclude equity and index linked")
    else:
        log_fisd_filter(fisd, fisd, "exclude equity and index linked (skipped)")

    # ---- 12b) Tenor >= min years -----------------------------------------
    if p["enforce_tenor_min"]:
        before = fisd
        fisd["maturity"] = pd.to_datetime(fisd["maturity"], errors="coerce")
        fisd["offering_date"] = pd.to_datetime(fisd["offering_date"], errors="coerce")
        fisd["tenor"] = (fisd["maturity"] - fisd["offering_date"]).dt.days / 365.25
        fisd = fisd.loc[fisd["tenor"] >= float(p["tenor_min_years"])]
        log_fisd_filter(before, fisd, f"tenor >= {p['tenor_min_years']} year(s)")
    else:
        log_fisd_filter(fisd, fisd, f"tenor >= {p['tenor_min_years']} year(s) (skipped)")
        
        
    # ---- 13) 144A Only or Not?  ------------------------------------
    before = fisd
    if str(data_type).lower() == "144a":
        fisd = fisd.loc[(fisd["rule_144a"] == "Y") | (fisd["private_placement"] == "Y")]
        log_fisd_filter(before, fisd, "Only keep 144A bonds")
  
    # ---- 14) Housekeeping + fisd_off  ------------------------------
    fisd = fisd.reset_index(drop=True)
    fisd["index"] = range(1, len(fisd) + 1)

    fisd_off = fisd[["complete_cusip", "offering_amt", "maturity"]].copy()
    fisd_off.rename(columns={"complete_cusip": "cusip_id"}, inplace=True)

    return fisd, fisd_off
# -------------------------------------------------------------------------
def error_checks(
    db,
    cusip_chunks,
    start_date: str | None = None,
    data_type: str = "standard",
    volume_filter: float | tuple[str, float] = ("dollar", 10000.0),
    trade_times: list[str] | None = None,  
    calendar_name: str | None = None,
    *,
    ds_params: dict | None = None,
    bb_params: dict | None = None,
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
    )
    f = {**FILTER_DEFAULTS, **(filters or {})}
    """
    Slimline filtering function for the plots of filtered data 

    """
    
    # ---------- 0. validate data_type ----------
    data_type = data_type.lower()
    if data_type == "standard":
        table_name = "trace.trace"
    elif data_type == "144a":
        table_name = "trace.trace_btds144a"
    else:
        raise ValueError("data_type must be 'standard' or '144a'")
    
    all_super_list         = []
    all_super_listbb       = []
    bb_cusips_all          = []
    dec_shift_cusips_all   = []
    
    sort_cols = ["cusip_id","trd_exctn_dt","trd_exctn_tm", "msg_seq_nb"]
    
    # ---------- 1. chunk loop ----------
    for i, temp_list in enumerate(cusip_chunks, start=1):
        start_time = time.time()
        logging.info(f"Processing chunk {i} of {len(cusip_chunks)}")
        temp_tuple = tuple(temp_list)

        # ---------- 2. build query ----------
        sql_query = f'''
            SELECT cusip_id, bond_sym_id, bsym, trd_exctn_dt, trd_exctn_tm,
                   msg_seq_nb, trc_st, wis_fl, cmsn_trd, ascii_rptd_vol_tx,
                   rptd_pr, yld_pt, asof_cd, side, diss_rptg_side_cd,
                   orig_msg_seq_nb, orig_dis_dt, rptg_party_type,
                   contra_party_type
            FROM {table_name}
            WHERE cusip_id in %(cusip_id)s
        '''

        params = {"cusip_id": temp_tuple}
        if start_date:
            sql_query += " AND trd_exctn_dt >= %(start_date)s"
            params["start_date"] = start_date

        # ---------- 3. fetch ----------
        trace = db.raw_sql(sql_query, params=params)
        logging.info(f"Chunk {i}: Retrieved {len(trace)} rows from WRDS")
        
        trace["rptd_pr"] = trace["rptd_pr"].astype("float64").round(6)
        trace = trace.drop(columns=["index"], errors="ignore").reset_index(drop=True)
                               
        if len(trace) == 0:
            continue
        
        # Initial log for cleaning
        log_filter(trace, trace, "start", i)
        
        # Filter 1: Dick-Nielsen
        if f["dick_nielsen"]:
            clean_chunk = clean_trace_standard_chunk(
                trace,
                chunk_id      = i,
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
            
            traceds = trace.sort_values(sort_cols, kind="mergesort", ignore_index=True).copy()                        
                    
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
            log_filter(trace, trace, "bounce_back_filter (skipped)", i)
        gc.collect()    
        
        trace_bb = trace_bb.sort_values(sort_cols, kind="mergesort", ignore_index=True)
                       
        # Run garbage collection to reclaim memory
        del trace
        gc.collect()
                
        # Append here #        
        all_super_list.append(traceds)
        all_super_listbb.append(trace_bb)
        
        del(traceds,trace_bb)
        
        # CUSIP check 
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
        logging.info("-" * 50)
    
    gc.collect()                                           
    final_df_ds = pd.concat(all_super_list,   ignore_index=True)
    final_df_bb = pd.concat(all_super_listbb, ignore_index=True)
        
    return final_df_ds, final_df_bb, bb_cusips_all, dec_shift_cusips_all                           
# -------------------------------------------------------------------------
def export_trace_dataframes(
    all_data: pd.DataFrame,
    fisd_df: pd.DataFrame,
    ct_audit_records: Sequence[Mapping[str, Any]],
    audit_records:    Sequence[Mapping[str, Any]],
    *,
    data_type: str = "standard",        
    output_format: str = "csv",
    out_dir: str | Path = ".",
    bounce_back_cusips: list[str] | None = None,
    decimal_shift_cusips: list[str] | None = None,
    fisd_audit_records: list[dict] | None = None,
    stamp: str | None = None,
) -> None:
    """
    Export TRACE-related DataFrames with names that reflect `data_type`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if stamp is None:
        stamp = pd.Timestamp.today().strftime("%Y%m%d")


    # ---------- 0. audit frames ----------
    clean_trace_audit_df = pd.DataFrame(ct_audit_records)
    audit_df             = pd.DataFrame(audit_records)
    fisd_filters_df      = pd.DataFrame(fisd_audit_records)
    
    for df in (clean_trace_audit_df, audit_df):
        if "chunk" in df.columns:
            df["chunk"] = df["chunk"] + 1
            
    def _uniq_df(vals: list[str] | None) -> pd.DataFrame:
        vals = [] if vals is None else vals
        uniq = sorted(set(map(str, vals)))
        return pd.DataFrame({"cusip_id": uniq})

    df_bb = _uniq_df(bounce_back_cusips)
    df_ds = _uniq_df(decimal_shift_cusips)

    # ---------- 1. file-name suffix ----------
    dt = data_type.lower()
    
    if dt not in {"standard", "144a"}:
        raise ValueError("data_type must be 'standard' or '144a'")
        
    suffix = "_standard" if dt == "standard" else "_144a"

    # ---------- 2. mapping ----------
    files: dict[str, pd.DataFrame] = {
        f"trace{suffix}"                     : all_data,
        f"trace_fisd{suffix}"                : fisd_df,
        f"fisd_filters{suffix}"              : fisd_filters_df, 
        f"dick_nielsen_filters_audit{suffix}": clean_trace_audit_df,
        f"drr_filters_audit{suffix}"         : audit_df,
        f"bounce_back_cusips{suffix}"        : df_bb,
        f"decimal_shift_cusips{suffix}"      : df_ds,
    }

    # ---------- 3. write ----------
    output_format = output_format.lower()
    
    if output_format not in {"csv", "parquet"}:
        raise ValueError("output_format must be 'csv' or 'parquet'")

    for base_name, df in files.items():
        if output_format == "csv":
            df.to_csv(out_dir / f"{base_name}_{stamp}.csv.gzip",
                      index=False, compression="gzip")
        else:
            df.to_parquet(out_dir / f"{base_name}_{stamp}.parquet",
                          engine="pyarrow", compression="snappy")

# -------------------------------------------------------------------------
# -----------------------  MAIN CLASS -------------------------------------
# -------------------------------------------------------------------------
class ProcessStandardTRACE:
    """
    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection used to query TRACE data.
    cusip_chunks : list[list[str]]
        Lists of CUSIPs to process per chunk. Each inner list is one chunk.
    start_date : str or None, default None
        Optional lower bound for trade dates in ISO format YYYY-MM-DD. If None,
        do not apply a lower bound in the query.
    data_type : str, default "standard"
        Dataset selector or routing hint. Example values include "standard"
        and "144a". The implementation may branch on this for table names and
        minor schema differences.
    clean_agency : bool, default True
        If True, apply the agency de-duplication pass after reversal cleaning.
    volume_filter : float, default 10000.0
        Minimum dollar volume threshold used to drop small trades. Dollar volume
        is size times price divided by 100.
    trade_times : list[str] or None, default None
        Optional inclusive intraday window as ["HH:MM:SS", "HH:MM:SS"] used to
        retain trades within a time-of-day range.
    calendar_name : str or None, default None
        Optional market calendar name used to keep only valid trading sessions.
    ds_params : dict or None, default None
        Keyword overrides forwarded to decimal_shift_corrector.
    bb_params : dict or None, default None
        Keyword overrides forwarded to flag_price_change_errors.
    export_dir : str or pathlib.Path or None, default None
        Directory where outputs may be written. If None, skip on-disk export.
    export_format : str, default "parquet"
        Output format for daily files. Examples: "parquet", "csv".
    compress : str or None, default "gzip"
        Compression option for exports. Examples: "gzip", "snappy", or None.
    log : logging.Logger or None, default None
        Optional logger. If None, a module-level logger may be created.

    Attributes
    ----------
    audit_records : list[dict]
        Per-stage audit entries appended by the internal logger hook.
    bb_cusips_all : set[str]
        CUSIPs flagged at least once by the bounce-back filter.
    dec_shift_cusips_all : set[str]
        CUSIPs corrected at least once by the decimal-shift corrector.
    final_df : pandas.DataFrame or None
        Aggregated daily output after run completes, or None before run.

    Methods
    -------
    run() -> tuple[pandas.DataFrame, list[str], list[str]]
        Execute the full pipeline over all chunks and return the final daily frame
        and the two CUSIP lists.
    export_outputs(...) -> dict[str, str]
        Write outputs such as the daily frame and CUSIP lists to export_dir if set.
    _log_filter(df_before, df_after, stage, chunk_id) -> None
        Internal audit logger hook. Appends one entry to audit_records.
    _load_chunk(cusips: list[str]) -> pandas.DataFrame
        Query TRACE for a given list of CUSIPs, honoring start_date and data_type.
    _process_chunk(trace: pandas.DataFrame, chunk_id: int) -> pandas.DataFrame
        Apply reversal cleaning, optional agency pass, and any additional filters.
    _aggregate_daily(trace: pandas.DataFrame) -> pandas.DataFrame
        Aggregate intraday trades to daily metrics per cusip_id and date.

    Notes
    -----
    - The class mirrors the Enhanced pipeline structure but uses Standard-specific
      cleaning steps such as clean_reversal before any optional agency pass.
    - If a chunk query returns no rows, downstream steps should handle an empty
      DataFrame and continue. The final outputs are empty if all chunks are empty.
    """

    # ---------------------------------------------------------------------
    def __init__(
        self,
        wrds_username: str,
        *,
        output_format: str = "csv",
        chunk_size: int = 250,
        clean_agency: bool = True,
        out_dir: str | Path = ".",
        start_date: str | None = None,     
        log_level: int = logging.INFO,
        data_type: str = "standard",
        volume_filter: float | tuple[str, float] = ("dollar", 10000.0),
        trade_times: list[str] | None = None,
        calendar_name: str | None = None,       
        ds_params: dict | None = None,
        bb_params: dict | None = None,
        filters: dict | None = None,
        fisd_params: dict | None = None
    ) -> None:
        # user options
        self.wrds_username = wrds_username
        self.output_format = output_format.lower()
        self.chunk_size    = int(chunk_size)
        self.clean_agency  = clean_agency
        self.out_dir       = Path(out_dir)
        self.start_date    = start_date
        self.data_type     = data_type.lower()       
        vkind, vthr = _normalize_volume_filter(volume_filter)
        self.volume_filter: tuple[str, float] = (vkind, vthr)  # canonical tuple
        self.volume_filter_kind: str = vkind                   # convenience
        self.volume_filter_threshold: float = vthr             # convenience
        self.trade_times   = trade_times
        self.calendar_name = calendar_name        
        self.ds_params = ds_params or {}
        self.bb_params = bb_params or {}
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

    # ---------------------------------------------------------------------
    # -------------- MAIN --------------
    # ---------------------------------------------------------------------
    def CreateDailyStandardTRACE(self):
        """Run the full pipeline and return the three core DataFrames."""
        try:
            self._connect_wrds()
            fisd, fisd_off = self._build_fisd()
            cusip_chunks   = self._make_cusip_chunks(fisd)
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
        return build_fisd(
            self.db,
            params=self.fisd_params,
            data_type=self.data_type,      
        )

    def _make_cusip_chunks(self, fisd: pd.DataFrame):
        self.logger.info("Creating CUSIP batches ...")
        cusips = list(fisd["complete_cusip"].unique())

        def divide_chunks(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i : i + n]

        return list(divide_chunks(cusips, self.chunk_size))

    def _run_clean_trace(self, cusip_chunks, fisd_off):
        self.logger.info("Running TRACE cleaning loop ...")
        
        all_data, bb_list, ds_list = clean_trace_data(
            self.db,
            cusip_chunks,
            fisd_off,
            clean_agency=self.clean_agency,
            fetch_fn=self._raw_sql_with_retry,
            start_date=self.start_date,
            data_type=self.data_type,
            volume_filter=self.volume_filter,            
            trade_times=self.trade_times,
            calendar_name=self.calendar_name,
            ds_params=self.ds_params,
            bb_params=self.bb_params,
            filters=self.filters
        )
           
        if bb_list:
            self.bounce_back_cusips_all.extend(bb_list)
        if ds_list:
            self.decimal_shift_cusips_all.extend(ds_list)
    
        return all_data
                               

    def _export(self, all_data: pd.DataFrame, fisd_df: pd.DataFrame):
        self.logger.info("Exporting results ...")
        # Route Standard/144A outputs into dedicated subfolders
        subfolder = "144a" if str(self.data_type).lower() == "144a" else "standard"
        out_sub = self.out_dir / subfolder
        out_sub.mkdir(parents=True, exist_ok=True)
    
        export_trace_dataframes(
            all_data,
            fisd_df,
            self.ct_audit_records,
            self.audit_records,
            data_type=self.data_type,
            output_format=self.output_format,
            out_dir=out_sub,
            bounce_back_cusips=self.bounce_back_cusips_all,
            decimal_shift_cusips=self.decimal_shift_cusips_all,
            fisd_audit_records=self.fisd_audit_records,
            stamp=RUN_STAMP,
        )

        
    def _reconnect_wrds(self):
        try:
            if getattr(self, "db", None) is not None:
                try:
                    self.db.close()
                except Exception:
                    pass
            self.logger.info("Reconnecting to WRDS ...")
            import wrds  # import here to avoid module import at top if not used
            self.db = wrds.Connection(wrds_username=self.wrds_username)
        except Exception:
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
            PGOperationalError = tuple()  # best-effort if psycopg2 isn't importable
    
        attempt = 0
        while True:
            try:
                # Cheap ping before heavy query. If it fails, reconnect.
                try:
                    _ = self.db.raw_sql("SELECT 1")
                except Exception:
                    self._reconnect_wrds()
                return self.db.raw_sql(sql, params=params)
            except (SAOperationalError, PGOperationalError) as e:
                attempt += 1
                msg = str(e).lower()
                transient = (
                    "ssl connection has been closed" in msg
                    or "server closed the connection" in msg
                    or "connection not open" in msg
                    or "terminating connection" in msg
                    or "connection reset" in msg
                )
                if not transient or attempt > max_retries:
                    self.logger.exception("DB query failed (attempt %s/%s)", attempt, max_retries)
                    raise
                sleep_s = base_sleep * (2 ** (attempt - 1))
                self.logger.warning(
                    "DB connection issue (%s). Reconnecting and retrying in %.1fs ...",
                    e.__class__.__name__,
                    sleep_s,
                )
                time.sleep(sleep_s)
                self._reconnect_wrds()



# -------------------------------------------------------------------------
# -------------- _RUN ------------------------------
# -------------------------------------------------------------------------
def CreateDailyStandardTRACE(wrds_username: str, *, 
                             data_type: str = "standard", **kwargs):
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
    return ProcessStandardTRACE(wrds_username,data_type=data_type,**kwargs).CreateDailyStandardTRACE()
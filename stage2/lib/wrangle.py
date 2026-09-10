"""wrangle.py -- VERBATIM final-wrangle machinery from stage2/process_bond_data.py
(lines 3022-3286, 6566-7478): extend_and_ffill_linker, load_linker, wrangle_returns, wrangle_signals,
build_main_panel, swap_adj_signals, reorder_panel_cols. Arbitrated by the G7 validator
(main_panel 1,859,546 x 140). load_bond_data_from_url is shimmed to the LOCAL fingerprinted
OSBAP linker (cfg.AUX[linker]) -- same published artifact upstream downloads."""
import gc
import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pathlib import Path

import _stage2_settings as cfg


def save_parquet(df, path, cusip_col='cusip', compress=True):
    """Upstream pbd.save_parquet (lines 50-80): cusip -> category, zstd parquet. The compression
    LEVEL is ours (cfg.PANEL_ZSTD_LEVEL; upstream hardcoded 9) -- an encoding knob, values identical
    (W5 speed_up/05)."""
    if cusip_col in df.columns and df[cusip_col].dtype != 'category':
        df = df.copy()
        df[cusip_col] = df[cusip_col].astype('category')
    if compress:
        df.to_parquet(path, engine='pyarrow', compression='zstd',
                      compression_level=cfg.PANEL_ZSTD_LEVEL, index=False, use_dictionary=True)
    else:
        df.to_parquet(path, index=False)

logger = logging.getLogger(__name__)


def load_bond_data_from_url(url, zipkey, local_dir='data', verbose=True):
    """Shim: the pipeline consumes the LOCAL fingerprinted OSBAP linker parquet (cfg.AUX)."""
    return pd.read_parquet(cfg.AUX['linker']).copy()

def extend_and_ffill_linker(
    dfl: pd.DataFrame,
    ffill_date: pd.Timestamp,
    *,
    group_col: str = "issuer_cusip",
    date_col: str = "date",
    id_cols: tuple = ("gvkey", "permno", "permco"),
    verbose: bool = True
) -> tuple:
    """
    Forward-extend the linker from its max date to ffill_date.

    Only extends for issuers that have all id_cols non-missing at max_link_date.

    Parameters
    ----------
    dfl : pd.DataFrame
        Linker DataFrame with issuer_cusip, date, and equity ID columns
    ffill_date : pd.Timestamp
        Target date to extend to (will use month-end)
    group_col : str, default "issuer_cusip"
        Column to group by (issuer identifier)
    date_col : str, default "date"
        Date column name
    id_cols : tuple, default ("gvkey", "permno", "permco")
        ID columns that must be present for extension
    verbose : bool, default True
        If True, log progress messages

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        Extended linker DataFrame and stats dict with extension details
    """
    stats = {
        'link_max_date': None,
        'ffill_date': None,
        'n_months_extended': 0,
        'n_issuers_extended': 0,
        'extended': False
    }

    if dfl.empty:
        return dfl.copy(), stats

    out = dfl.copy()

    # Work with month-end dates
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    me_ffill = pd.to_datetime(ffill_date) + pd.offsets.MonthEnd(0)
    me_link_max = out[date_col].max()

    if pd.isna(me_link_max):
        return out, stats

    me_link_max = me_link_max + pd.offsets.MonthEnd(0)

    stats['link_max_date'] = me_link_max
    stats['ffill_date'] = me_ffill

    # Nothing to do if target is not beyond current max
    if me_ffill <= me_link_max:
        if verbose:
            logger.info("  Linker already covers target date, no extension needed")
        return out, stats

    # Filter id_cols to only those that exist in the data
    existing_id_cols = [c for c in id_cols if c in out.columns]
    if not existing_id_cols:
        if verbose:
            logger.warning("  No ID columns found in linker, cannot extend")
        return out, stats

    # Snapshot at max_link_date (month-end aligned)
    snap = out.loc[
        (out[date_col] + pd.offsets.MonthEnd(0)) == me_link_max,
        [group_col] + existing_id_cols
    ].copy()

    # Eligible issuers: all existing IDs present (non-missing)
    elig_mask = snap[existing_id_cols].notna().all(axis=1)
    snap = snap.loc[elig_mask].drop_duplicates(subset=[group_col])

    if snap.empty:
        if verbose:
            logger.info("  No eligible issuers to extend (all have missing IDs)")
        return out, stats

    # Build month-end range to append
    new_months = pd.date_range(me_link_max + pd.offsets.MonthEnd(1), me_ffill, freq="ME")
    if len(new_months) == 0:
        return out, stats

    stats['n_months_extended'] = len(new_months)
    stats['n_issuers_extended'] = len(snap)
    stats['extended'] = True

    # Cartesian product: eligible issuers x new months
    ext_dates = pd.DataFrame({date_col: new_months})
    ext = snap.merge(ext_dates, how="cross")

    # Populate helper columns if present in input
    if "yyyymm" in out.columns:
        ext["yyyymm"] = (ext[date_col].dt.year * 100 + ext[date_col].dt.month).astype("Int64")

    # Ensure all original columns exist in ext (fill with NA if not derivable)
    for c in out.columns:
        if c not in ext.columns:
            ext[c] = pd.NA

    # Reorder to match original columns
    ext = ext[out.columns]

    # Concatenate and return
    result = pd.concat([out, ext], ignore_index=True)
    result.sort_values([group_col, date_col], inplace=True, kind="mergesort")
    result.reset_index(drop=True, inplace=True)

    return result, stats


def load_linker(
    linker_url: str,
    linker_zipkey: str,
    local_dir: str = "data",
    cols: list = None,
    ffill_date: pd.Timestamp = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load and clean the OSBAP linker file for equity identifier mapping.

    This function loads the bond-equity linker file with internet fallback,
    cleans it, optionally extends it via forward-fill, and returns a
    DataFrame ready to merge with bond data.

    Parameters
    ----------
    linker_url : str
        URL to download the linker ZIP file from
    linker_zipkey : str
        Name of the parquet file inside the ZIP
    local_dir : str, default "data"
        Directory for local file fallback
    cols : list, optional
        Columns to keep from linker. Default: ['issuer_cusip', 'permno', 'permco', 'gvkey']
    ffill_date : pd.Timestamp, optional
        If provided, forward-fill extend the linker to this date
    verbose : bool, default True
        If True, log progress messages

    Returns
    -------
    pd.DataFrame
        Cleaned linker with columns: issuer_cusip, date, permno, permco, gvkey
        Ready to merge on ['issuer_cusip', 'date'] or ['issuer_cusip', 'year_month']
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("LOAD LINKER - START")
        logger.info("=" * 60)

    # Default columns to keep
    if cols is None:
        cols = ['issuer_cusip', 'permno', 'permco', 'gvkey']

    # Load linker file with internet fallback
    dfl = load_bond_data_from_url(
        url=linker_url,
        zipkey=linker_zipkey,
        local_dir=local_dir,
        verbose=verbose
    )

    # Lowercase column names
    dfl.columns = dfl.columns.str.lower()

    if verbose:
        logger.info("  Raw linker shape: %s", dfl.shape)
        logger.info("  Columns: %s", list(dfl.columns))

    # Convert yyyymm to date (month-end)
    dfl['date'] = pd.to_datetime(dfl['yyyymm'], format='%Y%m', errors='coerce')
    dfl['date'] = dfl['date'] + pd.offsets.MonthEnd(0)

    # Process equity identifiers with proper dtypes
    # Handle permno
    if 'permno' in dfl.columns:
        dfl['permno'] = pd.to_numeric(dfl['permno'], errors='coerce').astype('Int64')
        if verbose:
            n_permno = dfl['permno'].notna().sum()
            logger.info("  permno: %s non-null values", f"{n_permno:,}")
    else:
        if verbose:
            logger.warning("  permno column not found in linker file - will be NA")
        dfl['permno'] = pd.NA

    # Handle permco
    if 'permco' in dfl.columns:
        dfl['permco'] = pd.to_numeric(dfl['permco'], errors='coerce').astype('Int64')
        if verbose:
            n_permco = dfl['permco'].notna().sum()
            logger.info("  permco: %s non-null values", f"{n_permco:,}")
    else:
        if verbose:
            logger.warning("  permco column not found in linker file - will be NA")
        dfl['permco'] = pd.NA

    # Handle gvkey
    if 'gvkey' in dfl.columns:
        dfl['gvkey'] = pd.to_numeric(dfl['gvkey'].round(0), errors='coerce').astype('Int32')
        if verbose:
            n_gvkey = dfl['gvkey'].notna().sum()
            logger.info("  gvkey: %s non-null values", f"{n_gvkey:,}")
    else:
        if verbose:
            logger.warning("  gvkey column not found in linker file - will be NA")
        dfl['gvkey'] = pd.NA

    # Keep only needed columns
    keep_cols = ['issuer_cusip', 'date'] + [c for c in cols if c != 'issuer_cusip']
    keep_cols = [c for c in keep_cols if c in dfl.columns]
    dfl = dfl[keep_cols].copy()

    if verbose:
        logger.info("  Cleaned linker shape: %s", dfl.shape)
        logger.info("  Date range: %s to %s", dfl['date'].min(), dfl['date'].max())

    # Forward-fill extend linker if ffill_date provided
    if ffill_date is not None:
        if verbose:
            logger.info("[Extension] Forward-filling linker to %s...", ffill_date)

        # Determine which ID columns exist for extension
        id_cols_for_ext = tuple(c for c in ['permno', 'permco', 'gvkey'] if c in dfl.columns)

        dfl, ext_stats = extend_and_ffill_linker(
            dfl,
            ffill_date=ffill_date,
            group_col='issuer_cusip',
            date_col='date',
            id_cols=id_cols_for_ext,
            verbose=verbose
        )

        if verbose:
            if ext_stats['extended']:
                logger.info("  Linker extended:")
                logger.info("    Original max date: %s", ext_stats['link_max_date'].strftime('%Y-%m-%d'))
                logger.info("    Extended to: %s", ext_stats['ffill_date'].strftime('%Y-%m-%d'))
                logger.info("    Months extended: %d", ext_stats['n_months_extended'])
                logger.info("    Issuers extended: %s", f"{ext_stats['n_issuers_extended']:,}")
                logger.info("  Extended linker shape: %s", dfl.shape)
            else:
                logger.info("  No extension needed (linker already covers target date)")

    # Convert to category for memory efficiency
    dfl['issuer_cusip'] = dfl['issuer_cusip'].astype('category')

    if verbose:
        logger.info("=" * 60)
        logger.info("LOAD LINKER - COMPLETE")
        logger.info("=" * 60)

    return dfl
def wrangle_returns(
    end_returns: pd.DataFrame,
    bgn_returns: pd.DataFrame,
    end_signals: pd.DataFrame = None,
    linker_url: str = None,
    linker_zipkey: str = None,
    local_dir: str = "data",
    end_cols: list = None,
    bgn_cols: list = None,
    alt_cols: list = None,
    signal_char_cols: list = None,
    verbose: bool = True
) -> tuple:
    """
    Wrangle return DataFrames into final output format.

    Creates output DataFrames:
    1. returns_main: Primary return panel (end_returns + bgn_returns + signals merged)
       with equity identifiers from linker and bond characteristics from end_signals
    2. returns_alt: Alternative return measures
    3. end_signals (trimmed): If end_signals provided, returns trimmed version

    Parameters
    ----------
    end_returns : pd.DataFrame
        Month-end returns from process_bond_data()
    bgn_returns : pd.DataFrame
        Within-month returns from process_bond_data()
    end_signals : pd.DataFrame, optional
        End-of-month signals from process_bond_data(). If provided, bond
        characteristics are merged to returns_main and end_signals is trimmed.
    linker_url : str, optional
        URL to download linker ZIP file. If None, linker merge is skipped.
    linker_zipkey : str, optional
        Name of parquet file inside linker ZIP
    local_dir : str, default "data"
        Directory for local file fallback
    end_cols : list, optional
        Columns to keep from end_returns for returns_main.
        Default: ['cusip', 'date', 'dt_s', 'dt_e', 'ret_vw', 'lib', 'ret_type',
                  'spc_rat', 'mdyc_rat', 'tret']
    bgn_cols : list, optional
        Columns to keep from bgn_returns for returns_main.
        Default: ['cusip', 'date', 'dt_s', 'dt_e', 'ret_vw', 'hprd', 'igap']
    alt_cols : list, optional
        Columns to keep for returns_alt.
        Default: ['cusip', 'date', 'ret_vwp', 'ret_ew', 'ret_1st', 'ret_lst',
                  'ret_bid', 'tret']
    signal_char_cols : list, optional
        Bond characteristic columns to merge from end_signals.
        Default: ['tmat', 'age', 'fce_val', 'cpn', 'call', 'ff17num', 'ff30num',
                  'mcap_s', 'mcap_e']
    verbose : bool, default True
        If True, log progress messages

    Returns
    -------
    tuple
        If end_signals is None: (returns_main, returns_alt)
        If end_signals provided: (returns_main, returns_alt, end_signals_trimmed)
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("WRANGLE RETURNS - START")
        logger.info("=" * 60)

    # Default column selections
    if end_cols is None:
        end_cols = ['cusip', 'date', 'dt_s', 'dt_e', 'ret_vw', 'hprd', 'lib', 'libd', 'ret_type',
                    'spc_rat', 'mdyc_rat', 'tret']

    if bgn_cols is None:
        bgn_cols = ['cusip', 'date', 'dt_s', 'dt_e', 'ret_vw', 'hprd', 'igap']

    if alt_cols is None:
        alt_cols = ['cusip', 'date', 'ret_vwp', 'ret_ew', 'ret_1st', 'ret_lst',
                    'ret_bid', 'tret']

    if signal_char_cols is None:
        signal_char_cols = ['tmat', 'age', 'fce_val', 'cpn', 'call', 'ff17num',
                            'ff30num', 'mcap_s', 'mcap_e']

    # =========================================================================
    # 1. Build returns_main
    # =========================================================================
    if verbose:
        logger.info("[Step 1] Building returns_main...")

    # Select end_returns columns (filter to those that exist)
    end_sel = [c for c in end_cols if c in end_returns.columns]
    ret_main = end_returns[end_sel].copy()

    # Rename mdyc_rat -> mdc_rat for consistency
    if 'mdyc_rat' in ret_main.columns:
        ret_main.rename(columns={'mdyc_rat': 'mdc_rat'}, inplace=True)

    # Select bgn_returns columns and rename with _bgn suffix
    bgn_sel = [c for c in bgn_cols if c in bgn_returns.columns]
    bgn_sub = bgn_returns[bgn_sel].copy()

    # Rename bgn columns (except cusip, date which are merge keys)
    bgn_rename = {c: f"{c}_bgn" for c in bgn_sel if c not in ['cusip', 'date']}
    bgn_sub.rename(columns=bgn_rename, inplace=True)

    # Left merge bgn_returns to end_returns
    ret_main = ret_main.merge(bgn_sub, on=['cusip', 'date'], how='left')

    if verbose:
        logger.info("  returns_main after bgn merge: %s", ret_main.shape)

    # =========================================================================
    # 2. Merge linker (equity identifiers)
    # =========================================================================
    if linker_url is not None and linker_zipkey is not None:
        if verbose:
            logger.info("[Step 2] Merging linker equity identifiers...")

        # Get max date from returns for linker extension
        max_ret_date = ret_main['date'].max()

        # Load linker (with forward-fill extension to max return date)
        linker = load_linker(
            linker_url=linker_url,
            linker_zipkey=linker_zipkey,
            local_dir=local_dir,
            cols=['issuer_cusip', 'permno', 'permco', 'gvkey'],
            ffill_date=max_ret_date,
            verbose=verbose
        )

        # Create issuer_cusip from cusip (first 6 chars)
        ret_main['issuer_cusip'] = ret_main['cusip'].astype(str).str[:6]

        # Merge on issuer_cusip and date
        ret_main = ret_main.merge(
            linker,
            on=['issuer_cusip', 'date'],
            how='left'
        )

        n_matched = ret_main['permno'].notna().sum()
        if verbose:
            logger.info("  Linker merge: %s rows with equity IDs (%.1f%%)",
                        f"{n_matched:,}", 100 * n_matched / len(ret_main))

        del linker
    else:
        if verbose:
            logger.info("[Step 2] Skipping linker merge (no URL provided)")

    # =========================================================================
    # 3. Merge bond characteristics from end_signals
    # =========================================================================
    if end_signals is not None:
        if verbose:
            logger.info("[Step 3] Merging bond characteristics from end_signals...")

        # Select columns to merge (cusip, date + signal_char_cols)
        sig_merge_cols = ['cusip', 'date'] + [c for c in signal_char_cols if c in end_signals.columns]
        sig_sub = end_signals[sig_merge_cols].copy()

        # Left merge to returns_main
        ret_main = ret_main.merge(sig_sub, on=['cusip', 'date'], how='left')

        if verbose:
            logger.info("  Merged columns: %s", [c for c in signal_char_cols if c in end_signals.columns])
            logger.info("  returns_main after signals merge: %s", ret_main.shape)

        del sig_sub
    else:
        if verbose:
            logger.info("[Step 3] Skipping signals merge (no end_signals provided)")

    # =========================================================================
    # 4. Build returns_alt
    # =========================================================================
    if verbose:
        logger.info("[Step 4] Building returns_alt...")

    alt_sel = [c for c in alt_cols if c in end_returns.columns]
    ret_alt = end_returns[alt_sel].copy()

    if verbose:
        logger.info("  returns_alt shape: %s", ret_alt.shape)

    # =========================================================================
    # 5. Optimize memory
    # =========================================================================
    if verbose:
        logger.info("[Step 5] Optimizing memory...")

    # returns_main dtypes
    dtype_map_main = {
        'cusip': 'category',
        'issuer_cusip': 'category',
        'permno': 'category',
        'permco': 'category',
        'gvkey': 'category',
        'ret_vw': 'float32',
        'ret_vw_bgn': 'float32',
        'hprd': 'Int16',
        'lib': 'float32',
        'libd': 'float32',
        'ret_type': 'category',
        'spc_rat': 'Int16',
        'mdc_rat': 'Int16',
        # Bond characteristics from end_signals
        'tmat': 'float32',
        'age': 'float32',
        'fce_val': 'Int64',
        'cpn': 'float32',
        'call': 'Int8',
        'ff17num': 'Int8',
        'ff30num': 'Int8',
        'mcap_s': 'float32',
        'mcap_e': 'float32',
        # Treasury and other
        'tret': 'float32',
        'hprd_bgn': 'Int16',
        'igap_bgn': 'Int8',
    }

    for col, dtype in dtype_map_main.items():
        if col in ret_main.columns:
            ret_main[col] = ret_main[col].astype(dtype)

    # returns_alt dtypes
    dtype_map_alt = {
        'cusip': 'category',
        'ret_vwp': 'float32',
        'ret_ew': 'float32',
        'ret_1st': 'float32',
        'ret_lst': 'float32',
        'ret_bid': 'float32',
        'tret': 'float32',
    }

    for col, dtype in dtype_map_alt.items():
        if col in ret_alt.columns:
            ret_alt[col] = ret_alt[col].astype(dtype)

    # =========================================================================
    # 6. Final ordering
    # =========================================================================
    # Order returns_main columns logically
    main_order = [
        # Identifiers (bond + equity)
        'cusip', 'issuer_cusip', 'permno', 'permco', 'gvkey', 'date',
        # Returns
        'ret_vw', 'ret_vw_bgn', 'hprd', 'lib', 'libd', 'ret_type',
        # Ratings
        'spc_rat', 'mdc_rat',
        # Bond characteristics (from end_signals)
        'tmat', 'age', 'fce_val', 'cpn', 'call', 'ff17num', 'ff30num',
        'mcap_s', 'mcap_e',
        # Treasury return
        'tret',
        # Date ranges (end then bgn)
        'dt_s', 'dt_e', 'dt_s_bgn', 'dt_e_bgn',
        # Within-month (bgn) other
        'hprd_bgn', 'igap_bgn',
    ]
    main_order = [c for c in main_order if c in ret_main.columns]
    ret_main = ret_main[main_order]

    # Order returns_alt columns
    alt_order = ['cusip', 'date', 'ret_vwp', 'ret_ew', 'ret_1st', 'ret_lst',
                 'ret_bid', 'tret']
    alt_order = [c for c in alt_order if c in ret_alt.columns]
    ret_alt = ret_alt[alt_order]

    # =========================================================================
    # 7. Trim end_signals (if provided)
    # =========================================================================
    sig_out = None
    if end_signals is not None:
        if verbose:
            logger.info("[Step 7] Trimming end_signals...")

        # Keep only required signal columns
        sig_keep_cols = ['cusip', 'date', 'ytm', 'cs', 'md_dur', 'convx', 'bbtm', 'sze']
        sig_keep_cols = [c for c in sig_keep_cols if c in end_signals.columns]
        sig_out = end_signals[sig_keep_cols].copy()

        # Optimize dtypes
        if 'cusip' in sig_out.columns:
            sig_out['cusip'] = sig_out['cusip'].astype('category')
        for col in ['ytm', 'cs', 'md_dur', 'convx', 'bbtm', 'sze']:
            if col in sig_out.columns:
                sig_out[col] = sig_out[col].astype('float32')

        if verbose:
            logger.info("  end_signals trimmed columns: %s", list(sig_out.columns))
            logger.info("  end_signals trimmed shape: %s", sig_out.shape)

    if verbose:
        logger.info("  returns_main columns: %s", list(ret_main.columns))
        logger.info("  returns_alt columns: %s", list(ret_alt.columns))
        logger.info("=" * 60)
        logger.info("WRANGLE RETURNS - COMPLETE")
        logger.info("=" * 60)

    # Return based on whether end_signals was provided
    if end_signals is not None:
        return ret_main, ret_alt, sig_out
    else:
        return ret_main, ret_alt


def wrangle_signals(
    signals_std: pd.DataFrame,
    adj_signals: pd.DataFrame,
    value_signals_std: pd.DataFrame,
    value_signals_adj: pd.DataFrame,
    verbose: bool = True
) -> tuple:
    """
    Wrangle signal DataFrames into final output format.

    Creates two output DataFrames:
    1. spreads_value: Merged standard signals + value signals
    2. spreads_value_adj: Merged adjusted signals + value signals

    Parameters
    ----------
    signals_std : pd.DataFrame
        Trimmed end_signals from wrangle_returns() with columns:
        cusip, date, ytm, cs, md_dur, convx, bbtm, sze
    adj_signals : pd.DataFrame
        Adjusted signals from process_bond_data() with columns:
        cusip, date, sig_dt, sig_gap, ytm_adj, md_dur_adj, convx_adj, cs_adj,
        str1_adj, str2_adj, bbtm_adj, sze_adj
    value_signals_std : pd.DataFrame
        Standard value signals from make_value_signals() + d_spreads
    value_signals_adj : pd.DataFrame
        Adjusted value signals from make_value_signals() + d_spreads

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (spreads_value, spreads_value_adj)
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("WRANGLE SIGNALS - START")
        logger.info("=" * 60)

    # =========================================================================
    # 1. Build spreads_value (standard)
    # =========================================================================
    if verbose:
        logger.info("[Step 1] Building spreads_value...")

    # Merge signals_std with value_signals_std
    spreads_val = signals_std.merge(
        value_signals_std,
        on=['cusip', 'date'],
        how='left'
    )

    if verbose:
        logger.info("  spreads_value shape after merge: %s", spreads_val.shape)

    # =========================================================================
    # 2. Build spreads_value_adj (adjusted)
    # =========================================================================
    if verbose:
        logger.info("[Step 2] Building spreads_value_adj...")

    # Merge adj_signals with value_signals_adj
    spreads_val_adj = adj_signals.merge(
        value_signals_adj,
        on=['cusip', 'date'],
        how='left'
    )

    if verbose:
        logger.info("  spreads_value_adj shape after merge: %s", spreads_val_adj.shape)

    # =========================================================================
    # 3. Optimize memory
    # =========================================================================
    if verbose:
        logger.info("[Step 3] Optimizing memory...")

    # spreads_value dtypes
    dtype_map_std = {
        'cusip': 'category',
        'ytm': 'float32',
        'cs': 'float32',
        'md_dur': 'float32',
        'convx': 'float32',
        'bbtm': 'float32',
        'sze': 'float32',
    }

    for col, dtype in dtype_map_std.items():
        if col in spreads_val.columns:
            spreads_val[col] = spreads_val[col].astype(dtype)

    # Convert all value and d_spread columns to float32
    for col in spreads_val.columns:
        if col.startswith('val_') or col.startswith('dcs') or col.startswith('bbtm_mu'):
            spreads_val[col] = spreads_val[col].astype('float32')

    # spreads_value_adj dtypes
    dtype_map_adj = {
        'cusip': 'category',
        'ytm_adj': 'float32',
        'cs_adj': 'float32',
        'md_dur_adj': 'float32',
        'convx_adj': 'float32',
        'bbtm_adj': 'float32',
        'sze_adj': 'float32',
        'str1_adj': 'float32',
        'str2_adj': 'float32',
    }

    for col, dtype in dtype_map_adj.items():
        if col in spreads_val_adj.columns:
            spreads_val_adj[col] = spreads_val_adj[col].astype(dtype)

    # Convert all value_adj and d_spread_adj columns to float32
    for col in spreads_val_adj.columns:
        if col.startswith('val_') or col.startswith('dcs') or col.startswith('bbtm_mu'):
            spreads_val_adj[col] = spreads_val_adj[col].astype('float32')

    # =========================================================================
    # 4. Final ordering
    # =========================================================================
    if verbose:
        logger.info("[Step 4] Ordering columns...")

    # Order spreads_value columns
    std_first = ['cusip', 'date', 'ytm', 'cs', 'md_dur', 'convx', 'bbtm', 'sze']
    std_first = [c for c in std_first if c in spreads_val.columns]
    std_rest = [c for c in spreads_val.columns if c not in std_first]
    spreads_val = spreads_val[std_first + std_rest]

    # Order spreads_value_adj columns (sig_dt, sig_gap come after cusip, date)
    adj_first = ['cusip', 'date', 'sig_dt', 'sig_gap', 'ytm_adj', 'md_dur_adj',
                 'convx_adj', 'cs_adj', 'str1_adj', 'str2_adj', 'bbtm_adj', 'sze_adj']
    adj_first = [c for c in adj_first if c in spreads_val_adj.columns]
    adj_rest = [c for c in spreads_val_adj.columns if c not in adj_first]
    spreads_val_adj = spreads_val_adj[adj_first + adj_rest]

    if verbose:
        logger.info("  spreads_value columns: %s", list(spreads_val.columns))
        logger.info("  spreads_value_adj columns: %s", list(spreads_val_adj.columns))
        logger.info("=" * 60)
        logger.info("WRANGLE SIGNALS - COMPLETE")
        logger.info("=" * 60)

    return spreads_val, spreads_val_adj


def build_main_panel(
    returns_main: pd.DataFrame,
    spreads_value: pd.DataFrame,
    spreads_value_adj: pd.DataFrame,
    illiq_signals: pd.DataFrame,
    illiq_signals_adj: pd.DataFrame,
    betas_std: pd.DataFrame,
    mom_ret: pd.DataFrame,
    verbose: bool = True
) -> tuple:
    """
    Build the main analysis panels by merging returns with signals and betas.

    Creates two panels:
    1. main_panel: returns_main + spreads_value + illiq_signals + betas_std + mom_ret
    2. main_panel_adj: returns_main + spreads_value_adj + illiq_signals_adj + betas_std + mom_ret

    All merges are left joins onto returns_main.

    Parameters
    ----------
    returns_main : pd.DataFrame
        Primary return panel from wrangle_returns()
    spreads_value : pd.DataFrame
        Standard signals + value signals from wrangle_signals()
    spreads_value_adj : pd.DataFrame
        Adjusted signals + value signals from wrangle_signals()
    illiq_signals : pd.DataFrame
        Standard illiquidity signals (cusip_id, date, signal columns)
    illiq_signals_adj : pd.DataFrame
        Adjusted illiquidity signals (cusip_id, date, *_adj columns)
    betas_std : pd.DataFrame
        Rolling betas from compute_all_betas()
    mom_ret : pd.DataFrame
        Momentum/LTR signals from build_mom_ltr_and_industry()
    verbose : bool, default True
        If True, log progress

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (main_panel, main_panel_adj)
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("BUILD MAIN PANEL - START")
        logger.info("=" * 60)

    # =========================================================================
    # 1. Prepare illiq signals (rename cusip_id -> cusip)
    # =========================================================================
    if verbose:
        logger.info("[Step 1] Preparing illiq signals...")

    illiq_std = illiq_signals.copy()
    if 'cusip_id' in illiq_std.columns:
        illiq_std = illiq_std.rename(columns={'cusip_id': 'cusip'})

    illiq_adj = illiq_signals_adj.copy()
    if 'cusip_id' in illiq_adj.columns:
        illiq_adj = illiq_adj.rename(columns={'cusip_id': 'cusip'})

    if verbose:
        logger.info("  illiq_std shape: %s", illiq_std.shape)
        logger.info("  illiq_adj shape: %s", illiq_adj.shape)

    # =========================================================================
    # 2. Check inputs for duplicates on merge keys
    # =========================================================================
    if verbose:
        logger.info("[Step 2] Checking inputs for duplicates on (cusip, date)...")

        # returns_main
        dup_ret = returns_main.duplicated(subset=['cusip', 'date']).sum()
        logger.info("  returns_main: %s rows, %d duplicates", f"{len(returns_main):,}", dup_ret)

        # spreads_value
        dup_sv = spreads_value.duplicated(subset=['cusip', 'date']).sum()
        logger.info("  spreads_value: %s rows, %d duplicates", f"{len(spreads_value):,}", dup_sv)

        # spreads_value_adj
        dup_sva = spreads_value_adj.duplicated(subset=['cusip', 'date']).sum()
        logger.info("  spreads_value_adj: %s rows, %d duplicates", f"{len(spreads_value_adj):,}", dup_sva)

        # illiq_std
        dup_illiq = illiq_std.duplicated(subset=['cusip', 'date']).sum()
        logger.info("  illiq_std: %s rows, %d duplicates", f"{len(illiq_std):,}", dup_illiq)

        # illiq_adj
        dup_illiq_adj = illiq_adj.duplicated(subset=['cusip', 'date']).sum()
        logger.info("  illiq_adj: %s rows, %d duplicates", f"{len(illiq_adj):,}", dup_illiq_adj)

        # betas_std
        dup_betas = betas_std.duplicated(subset=['cusip', 'date']).sum()
        logger.info("  betas_std: %s rows, %d duplicates", f"{len(betas_std):,}", dup_betas)

        # mom_ret
        dup_mom = mom_ret.duplicated(subset=['cusip', 'date']).sum()
        logger.info("  mom_ret: %s rows, %d duplicates", f"{len(mom_ret):,}", dup_mom)

    # =========================================================================
    # 3. Build main_panel (standard)
    # =========================================================================
    if verbose:
        logger.info("[Step 3] Building main_panel...")

    main_panel = returns_main.copy()
    n_before = len(main_panel)

    # Merge spreads_value
    main_panel = main_panel.merge(
        spreads_value,
        on=['cusip', 'date'],
        how='left'
    )
    n_after = len(main_panel)
    dup_after = main_panel.duplicated(subset=['cusip', 'date']).sum()
    if verbose:
        logger.info("  After spreads_value merge: %s -> %s rows (diff: %+d), dups: %d",
                    f"{n_before:,}", f"{n_after:,}", n_after - n_before, dup_after)
    n_before = n_after

    # Merge illiq_signals (standard)
    main_panel = main_panel.merge(
        illiq_std,
        on=['cusip', 'date'],
        how='left'
    )
    n_after = len(main_panel)
    dup_after = main_panel.duplicated(subset=['cusip', 'date']).sum()
    if verbose:
        logger.info("  After illiq_std merge: %s -> %s rows (diff: %+d), dups: %d",
                    f"{n_before:,}", f"{n_after:,}", n_after - n_before, dup_after)
    n_before = n_after

    # Merge betas_std
    main_panel = main_panel.merge(
        betas_std,
        on=['cusip', 'date'],
        how='left'
    )
    n_after = len(main_panel)
    dup_after = main_panel.duplicated(subset=['cusip', 'date']).sum()
    if verbose:
        logger.info("  After betas_std merge: %s -> %s rows (diff: %+d), dups: %d",
                    f"{n_before:,}", f"{n_after:,}", n_after - n_before, dup_after)
    n_before = n_after

    # Merge mom_ret (momentum/LTR signals)
    main_panel = main_panel.merge(
        mom_ret,
        on=['cusip', 'date'],
        how='left'
    )
    n_after = len(main_panel)
    dup_after = main_panel.duplicated(subset=['cusip', 'date']).sum()
    if verbose:
        logger.info("  After mom_ret merge: %s -> %s rows (diff: %+d), dups: %d",
                    f"{n_before:,}", f"{n_after:,}", n_after - n_before, dup_after)

    # =========================================================================
    # 4. Build main_panel_adj (adjusted)
    # =========================================================================
    if verbose:
        logger.info("[Step 4] Building main_panel_adj...")

    main_panel_adj = returns_main.copy()
    n_before = len(main_panel_adj)

    # Merge spreads_value_adj
    main_panel_adj = main_panel_adj.merge(
        spreads_value_adj,
        on=['cusip', 'date'],
        how='left'
    )
    n_after = len(main_panel_adj)
    dup_after = main_panel_adj.duplicated(subset=['cusip', 'date']).sum()
    if verbose:
        logger.info("  After spreads_value_adj merge: %s -> %s rows (diff: %+d), dups: %d",
                    f"{n_before:,}", f"{n_after:,}", n_after - n_before, dup_after)
    n_before = n_after

    # Merge illiq_signals_adj
    main_panel_adj = main_panel_adj.merge(
        illiq_adj,
        on=['cusip', 'date'],
        how='left'
    )
    n_after = len(main_panel_adj)
    dup_after = main_panel_adj.duplicated(subset=['cusip', 'date']).sum()
    if verbose:
        logger.info("  After illiq_adj merge: %s -> %s rows (diff: %+d), dups: %d",
                    f"{n_before:,}", f"{n_after:,}", n_after - n_before, dup_after)
    n_before = n_after

    # Merge betas_std
    main_panel_adj = main_panel_adj.merge(
        betas_std,
        on=['cusip', 'date'],
        how='left'
    )
    n_after = len(main_panel_adj)
    dup_after = main_panel_adj.duplicated(subset=['cusip', 'date']).sum()
    if verbose:
        logger.info("  After betas_std merge: %s -> %s rows (diff: %+d), dups: %d",
                    f"{n_before:,}", f"{n_after:,}", n_after - n_before, dup_after)
    n_before = n_after

    # Merge mom_ret (momentum/LTR signals)
    main_panel_adj = main_panel_adj.merge(
        mom_ret,
        on=['cusip', 'date'],
        how='left'
    )
    n_after = len(main_panel_adj)
    dup_after = main_panel_adj.duplicated(subset=['cusip', 'date']).sum()
    if verbose:
        logger.info("  After mom_ret merge: %s -> %s rows (diff: %+d), dups: %d",
                    f"{n_before:,}", f"{n_after:,}", n_after - n_before, dup_after)

    # =========================================================================
    # 5. Ensure identifier columns remain category dtype
    # =========================================================================
    if verbose:
        logger.info("[Step 5] Re-converting identifier columns to category...")

    id_cols = ['cusip', 'issuer_cusip', 'permno', 'permco', 'gvkey']

    for col in id_cols:
        if col in main_panel.columns:
            main_panel[col] = main_panel[col].astype('category')
        if col in main_panel_adj.columns:
            main_panel_adj[col] = main_panel_adj[col].astype('category')

    # =========================================================================
    # 6. Reorder columns
    # =========================================================================
    if verbose:
        logger.info("[Step 6] Reordering columns...")

    # Helper function to reorder columns
    def reorder_cols(df, move_cols, before_col):
        """Move specified columns to appear before a target column."""
        cols = list(df.columns)
        # Find columns that exist
        move_existing = [c for c in move_cols if c in cols]
        if not move_existing or before_col not in cols:
            return df
        # Remove move_cols from their current positions
        for c in move_existing:
            cols.remove(c)
        # Find position of before_col and insert move_cols there
        idx = cols.index(before_col)
        for i, c in enumerate(move_existing):
            cols.insert(idx + i, c)
        return df[cols]

    # Reorder main_panel: move tmat, age, fce_val, cpn before ytm
    char_cols = ['tmat', 'age', 'fce_val', 'cpn']
    if 'ytm' in main_panel.columns:
        main_panel = reorder_cols(main_panel, char_cols, 'ytm')

    # Reorder main_panel_adj: move tmat, age, fce_val, cpn before ytm_adj
    if 'ytm_adj' in main_panel_adj.columns:
        main_panel_adj = reorder_cols(main_panel_adj, char_cols, 'ytm_adj')

    # =========================================================================
    # 7. Summary
    # =========================================================================
    if verbose:
        logger.info("  main_panel: %s rows, %d cols", f"{len(main_panel):,}", len(main_panel.columns))
        logger.info("  main_panel_adj: %s rows, %d cols", f"{len(main_panel_adj):,}", len(main_panel_adj.columns))
        logger.info("=" * 60)
        logger.info("BUILD MAIN PANEL - COMPLETE")
        logger.info("=" * 60)

    del illiq_std, illiq_adj
    gc.collect()

    return main_panel, main_panel_adj


def swap_adj_signals(
    main_panel: pd.DataFrame,
    main_panel_adj: pd.DataFrame,
    output_dir: Path,
    date_stamp: str,
    compress: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Replace price-based signals in main_panel with adjusted versions from main_panel_adj.

    Signals measured at month-end may be stale if no trade occurred on the last day.
    The _adj versions use the most recent trade within 10 business days of month-end.
    This function swaps in those adjusted values for cleaner signal measurement.

    The original (non-adjusted) values are saved to a separate parquet file for
    researchers who prefer month-end measurement (e.g., Möllenhoff-Müller-Nagel style).

    Parameters
    ----------
    main_panel : pd.DataFrame
        Main panel with standard (month-end) signals
    main_panel_adj : pd.DataFrame
        Adjusted panel with _adj signal columns
    output_dir : Path
        Directory for output files
    date_stamp : str
        Date stamp for filename (e.g., '20251126')
    compress : bool, default True
        Use zstd compression for parquet export
    verbose : bool, default True
        Log progress

    Returns
    -------
    pd.DataFrame
        main_panel with _adj values swapped in (suffix removed)
    """
    if verbose:
        logger.info("[swap_adj_signals] Replacing month-end signals with adjusted versions...")

    # Identify _adj columns in main_panel_adj
    adj_cols = [c for c in main_panel_adj.columns if c.endswith('_adj')]

    if not adj_cols:
        if verbose:
            logger.warning("  No _adj columns found in main_panel_adj, skipping swap")
        return main_panel

    if verbose:
        logger.info("  Found %d _adj columns to swap: %s", len(adj_cols), adj_cols)

    # Get base column names (strip _adj suffix)
    base_cols = [c.replace('_adj', '') for c in adj_cols]

    # Check which base columns exist in main_panel
    existing_base = [c for c in base_cols if c in main_panel.columns]
    missing_base = [c for c in base_cols if c not in main_panel.columns]

    if missing_base and verbose:
        logger.warning("  Base columns not in main_panel (will skip): %s", missing_base)

    if not existing_base:
        if verbose:
            logger.warning("  No matching base columns found, skipping swap")
        return main_panel

    # Store original values before replacement
    original_signals = main_panel[['cusip', 'date'] + existing_base].copy()

    # Add _mmn suffix to signal columns
    rename_map = {col: f"{col}_mmn" for col in existing_base}
    original_signals.rename(columns=rename_map, inplace=True)

    if verbose:
        logger.info("  Saving original (month-end) signals to mmn_price_based_signals...")
        logger.info("  Renamed signals with _mmn suffix: %s", list(rename_map.values()))

    # Export original signals
    mmn_path = output_dir / f"mmn_price_based_signals_{date_stamp}.parquet"
    save_parquet(original_signals, mmn_path, compress=compress)

    if verbose:
        logger.info("  Saved: %s (%s rows, %d signal cols)",
                    mmn_path, f"{len(original_signals):,}", len(existing_base))

    del original_signals

    # Swap in adjusted values
    swapped = []
    for base_col in existing_base:
        adj_col = f"{base_col}_adj"
        if adj_col in main_panel_adj.columns:
            main_panel[base_col] = main_panel_adj[adj_col]
            swapped.append(base_col)

    if verbose:
        logger.info("  Swapped %d columns: %s", len(swapped), swapped)

    gc.collect()

    return main_panel


def reorder_panel_cols(
    df: pd.DataFrame,
    signal_order: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Reorder DataFrame columns with consistent ID/metadata prefix and signal order.

    Ensures both main_panel and main_panel_adj have consistent column ordering
    for the first ~31 columns (identifiers, returns, metadata up to sig_gap),
    with signal columns following in the specified order.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame (main_panel or main_panel_adj)
    signal_order : list, optional
        Canonical order for signal columns (base names without _adj).
        For each signal, checks if 'sig' or 'sig_adj' exists in df.
        If None, signals are appended in their original order.
    verbose : bool, default True
        Log reordering info

    Returns
    -------
    pd.DataFrame
        DataFrame with reordered columns
    """
    cols = list(df.columns)

    # Define canonical order for ID/metadata columns (up to and including sig_gap)
    id_prefix = [
        'cusip', 'date', 'issuer_cusip', 'permno', 'permco', 'gvkey',
        '144a', 'country', 'call',
    ]

    ret_prefix = [
        'ret_vw', 'ret_vw_bgn', 'hprd', 'lib', 'libd', 'ret_type',
        'spc_rat', 'mdc_rat', 'ff17num', 'ff30num', 'fce_val', 'mcap_s', 'mcap_e',
        'tret', 'rfret', 'dt_s', 'dt_e', 'dt_s_bgn', 'dt_e_bgn',
        'hprd_bgn', 'igap_bgn', 'sig_dt', 'sig_gap',
    ]

    # Build ordered list: prefix columns first (if they exist)
    ordered = []
    for c in id_prefix + ret_prefix:
        if c in cols:
            ordered.append(c)
            cols.remove(c)

    # Order signal columns
    if signal_order is not None:
        # Use provided signal order, checking for both base and _adj versions
        for sig in signal_order:
            if sig in cols:
                ordered.append(sig)
                cols.remove(sig)
            elif f"{sig}_adj" in cols:
                ordered.append(f"{sig}_adj")
                cols.remove(f"{sig}_adj")

    # Append any remaining columns (signals not in signal_order, or all signals if no order provided)
    ordered.extend(cols)

    if verbose:
        n_prefix = len(id_prefix) + len(ret_prefix)
        n_signals = len(ordered) - n_prefix
        logger.info("  Reordered columns: %d total, prefix: %d, signals: %d",
                    len(ordered), n_prefix, n_signals)

    return df[ordered]

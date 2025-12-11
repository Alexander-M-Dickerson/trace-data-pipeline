# -*- coding: utf-8 -*-
"""
stage1_pipeline.py
==================
Main processing pipeline for Stage 1 of the TRACE data pipeline.

This module contains all the core processing logic for Stage 1:
- Loading treasury yields and TRACE data
- Merging with FISD bond characteristics
- Computing bond analytics (duration, convexity, spreads)
- Merging credit ratings and external identifiers
- Applying ultra-distressed filters
- Generating comprehensive data quality reports

IMPORTANT: This module is designed to be configured dynamically.
All configuration variables (WRDS_USERNAME, ROOT_PATH, etc.) are set
by create_daily_stage1.py from _stage1_settings.py before execution.

Do not edit configuration values directly in this file - they will be
overwritten at runtime. Instead, edit _stage1_settings.py.

Author: Alex Dickerson
Created: 2025-11-04
Updated: 2025-11-17 
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import wrds
import gc
from pandas.tseries.offsets import MonthEnd
from tqdm import tqdm
import logging
import time

tqdm.pandas()

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================
# NOTE: These variables are set dynamically by create_daily_stage1.py
# Do not edit them here - they will be overwritten at runtime.
# To change configuration, edit _stage1_settings.py instead.
#
# The following variables must be set before this module executes:
# - WRDS_USERNAME, AUTHOR
# - ROOT_PATH, STAGE0_DIR, STAGE1_DIR, STAGE1_DATA, LOG_DIR
# - STAGE0_DATE_STAMP, TRACE_MEMBERS
# - N_CORES, N_CHUNKS, DATE_CUT_OFF
# - ULTRA_DISTRESSED_CONFIG, FINAL_FILTER_CONFIG
# - yld_type, liu_wu_url, LINKER_URL, LINKER_ZIPKEY
# ============================================================================

# ============================================================================
# LOGGING SETUP
# ============================================================================

timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"stage1_{timestamp_log}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# IMPORT HELPER FUNCTIONS
# ============================================================================

import helper_functions as hf
import _distressed_plot_helpers as dph

logger.info("=" * 80)
logger.info("Stage 1 Pipeline Initialized")
logger.info("Root path: %s", ROOT_PATH)
logger.info("Stage0 date stamp: %s", STAGE0_DATE_STAMP)
logger.info("TRACE members: %s", ", ".join(TRACE_MEMBERS))
logger.info("=" * 80)

# ============================================================================
# SYSTEM & PACKAGE VERSION LOGGING
# ============================================================================

logger.info("=" * 80)
logger.info("System and Package Versions")
logger.info("=" * 80)
logger.info("Python version: %s", sys.version.split()[0])
logger.info("NumPy version: %s", np.__version__)
logger.info("Pandas version: %s", pd.__version__)

# Check QuantLib version
try:
    import QuantLib as ql
    logger.info("QuantLib version: %s", ql.__version__)
except (ImportError, AttributeError):
    logger.warning("QuantLib version not available")

# Check joblib version
try:
    import joblib
    logger.info("joblib version: %s", joblib.__version__)
except (ImportError, AttributeError):
    logger.warning("joblib version not available")

logger.info("=" * 80)


# ============================================================================
# GLOBAL VARIABLES (will be populated by each step)
# ============================================================================

ylds           = None
final_df       = None
fisd           = None
trace_other    = None       # Non-processed columns (created in step4)
traced_out     = None       # Columns for parallel processing (created in step4)
sp_ratings     = None
moodys_ratings = None
call_dummy     = None
db             = None

# Filter tracking for Table 2 reporting
FILTER_COUNTS  = {}

# LaTeX table strings (set by step10a, used by step10)
table1_tex     = None
table2_tex     = None

# Fama-French industry mappings (set by step3, used by step10)
FF17_MAPPING   = None
FF30_MAPPING   = None

# ============================================================================
# STEP 1: LOAD TREASURY YIELDS
# ============================================================================

def step1_load_yields():
    """Load treasury yields from configured source (FRED or Liu-Wu)."""
    mem_start = hf.log_memory_usage("step1_load_yields_start")
    global ylds
    
    logger.info("=" * 80)
    logger.info(f"STEP 1: Loading Treasury Yields (Source: {yld_type})")
    logger.info("=" * 80)
    
    if yld_type == 'LIU_WU':
        logger.info("Using Liu-Wu zero-coupon yields from Google Sheets")
        ylds = hf.get_liu_wu_yields(url=liu_wu_url)
    elif yld_type == 'FRED':
        logger.info("Using FRED yields")
        ylds = hf.get_fred_yields()
    else:
        raise ValueError(f"Invalid yld_type: {yld_type}. Must be 'FRED' or 'LIU_WU'")
    
    logger.info("Yields loaded: %d rows", len(ylds))
    logger.info("Date range: %s to %s", ylds['trd_exctn_dt'].min(), ylds['trd_exctn_dt'].max())
    logger.info("Columns: %s", list(ylds.columns))
    
    print(f"\n[STEP 1 COMPLETE] {yld_type} yields loaded")
    print(f"Shape: {ylds.shape}")
   
    gc.collect()
    mem_end = hf.log_memory_usage("step1_load_yields_end")
    hf.log_memory_delta(mem_start, mem_end, "step1_load_yields")   
    return ylds


# ============================================================================
# STEP 2: LOAD TRACE DATA FROM STAGE0
# ============================================================================

def step2_load_trace_data():
    """Load TRACE data from stage0 outputs."""
    mem_start = hf.log_memory_usage("step2_load_trace_data_start")
    global final_df

    logger.info("=" * 80)
    logger.info("STEP 2: Loading TRACE Data from Stage0")
    logger.info("=" * 80)

    trace_parts = []

    for i, member in enumerate(TRACE_MEMBERS, start=1):
        # Construct path: stage0/enhanced/trace_enhanced_YYYYMMDD.parquet
        member_folder = STAGE0_DIR / member
        filename = f"trace_{member}_{STAGE0_DATE_STAMP}.parquet"
        filepath = member_folder / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Stage0 output not found: {filepath}\n"
                f"Expected: stage0/{member}/trace_{member}_{STAGE0_DATE_STAMP}.parquet"
            )

        logger.info("Loading %s...", filepath.name)
        df_i = hf.load_and_process_trace_file(filepath)
        df_i["db_type"] = i  # Tag by load order: 1=enhanced, 2=standard, 3=144a
        trace_parts.append(df_i)
        logger.info("  Loaded %s: %d rows", member, len(df_i))

    # Combine all TRACE datasets
    final_df = pd.concat(trace_parts, ignore_index=True)
    del trace_parts
    gc.collect()

    logger.info("Combined TRACE data: %d rows", len(final_df))

    # Handle overlaps: Only clip STANDARD to be after Enhanced max date
    # Keep ALL 144a data (do not clip)
    if len(TRACE_MEMBERS) > 1 and "enhanced" in TRACE_MEMBERS:
        # Find max enhanced date first
        max_enhanced_date = final_df.loc[final_df["db_type"] == 1, "trd_exctn_dt"].max()

        if pd.notna(max_enhanced_date):
            logger.info("Enhanced max date: %s", max_enhanced_date)

            # Create mask for rows to keep (more memory efficient than creating separate dfs)
            # Keep: all enhanced (db_type==1) + standard after max_enhanced_date (db_type==2) + all 144a (db_type==3)
            before = len(final_df)
            mask = (
                (final_df["db_type"] == 1) |  # Keep all enhanced
                (final_df["db_type"] == 3) |  # Keep all 144a
                ((final_df["db_type"] == 2) & (final_df["trd_exctn_dt"] > max_enhanced_date))  # Keep only standard after enhanced
            )
            final_df = final_df[mask].copy()
            after = len(final_df)
            logger.info("Clipped Standard to dates after Enhanced: -%d rows", before - after)

            del mask
            gc.collect()

    logger.info("After clipping: %d rows", len(final_df))

    # Apply date cutoff filter (in-place to save memory)
    cut_off_date = pd.to_datetime(DATE_CUT_OFF)
    before = len(final_df)
    final_df = final_df[final_df['trd_exctn_dt'] <= cut_off_date].copy()
    after = len(final_df)
    logger.info("Applied date cutoff (<= %s): -%d rows", DATE_CUT_OFF, before - after)

    # Drop duplicates with preference: Enhanced > Standard > 144a
    # db_type: 1=enhanced, 2=standard, 3=144a
    before = len(final_df)
    final_df = final_df.sort_values(["cusip_id", "trd_exctn_dt", "db_type"])
    final_df = final_df.drop_duplicates(subset=["cusip_id", "trd_exctn_dt"], keep="first").reset_index(drop=True)
    after = len(final_df)
    logger.info("Dropped duplicates (cusip_id, trd_exctn_dt): -%d rows", before - after)

    # Sort by cusip and date
    final_df = final_df.sort_values(["cusip_id", "trd_exctn_dt"]).reset_index(drop=True)

    # Convert to category for memory optimization
    if "cusip_id" in final_df.columns:
        final_df['cusip_id'] = final_df['cusip_id'].astype('category')
    if "issuer_cusip" in final_df.columns:
        final_df['issuer_cusip'] = final_df['issuer_cusip'].astype('category')

    print("\n[STEP 2 COMPLETE] TRACE data loaded")
    print(f"Shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")
    print(f"Date range: {final_df['trd_exctn_dt'].min()} to {final_df['trd_exctn_dt'].max()}")

    gc.collect()
    mem_end = hf.log_memory_usage("step2_load_trace_data_end")
    hf.log_memory_delta(mem_start, mem_end, "step2_load_trace_data")
    return final_df


# ============================================================================
# STEP 3: LOAD FISD DATA
# ============================================================================

def step3_load_fisd_data():
    """Load FISD bond characteristics from stage0/enhanced/."""
    mem_start = hf.log_memory_usage("step3_load_fisd_data_start")
    global fisd, db, FF17_MAPPING, FF30_MAPPING
    
    logger.info("=" * 80)
    logger.info("STEP 3: Loading FISD Data")
    logger.info("=" * 80)
    
    # FISD file is always in stage0/enhanced/
    fisd_path = STAGE0_DIR / "enhanced" / f"trace_enhanced_fisd_{STAGE0_DATE_STAMP}.parquet"
    
    if not fisd_path.exists():
        raise FileNotFoundError(
            f"FISD file not found: {fisd_path}\n\n"
            f"Expected location: stage0/enhanced/trace_enhanced_fisd_{STAGE0_DATE_STAMP}.parquet\n\n"
            f"The FISD file must be generated by stage0 processing.\n"
            f"Please ensure stage0 has been run with date stamp: {STAGE0_DATE_STAMP}"
        )
    
    logger.info("Loading FISD from local file: %s", fisd_path.name)
    fisd = pd.read_parquet(fisd_path)
    fisd.rename(columns = {'complete_cusip':'cusip_id'}, inplace = True)

    # Convert to category for memory optimization
    if "cusip_id" in fisd.columns:
        fisd['cusip_id'] = fisd['cusip_id'].astype('category')

    logger.info("FISD loaded from file: %d rows", len(fisd))

    # Add Fama-French 17 industry classification
    logger.info("Adding Fama-French 17 industry classifications...")
    fisd, FF17_MAPPING, FF30_MAPPING = hf.add_ff_industries(fisd, verbose=False)
    logger.info("FF17 and FF30 industries added: ff17num and ff30num columns created")
    logger.info("Stored FF17_MAPPING with %d industries", len(FF17_MAPPING))
    logger.info("Stored FF30_MAPPING with %d industries", len(FF30_MAPPING))
    
    print("\n[STEP 3 COMPLETE] FISD data loaded")
    print(f"Shape: {fisd.shape}")
    print(f"Columns: {list(fisd.columns)}")
    gc.collect()
    mem_end = hf.log_memory_usage("step3_load_fisd_data_end")   
    hf.log_memory_delta(mem_start, mem_end, "step3_load_fisd_data")
    return fisd


# ============================================================================
# STEP 4: MERGE FISD TO TRACE
# ============================================================================

def step4_merge_fisd():
    """Merge FISD characteristics to TRACE data and prepare for parallel processing."""
    mem_start = hf.log_memory_usage("step4_merge_fisd_start")
    global final_df, traced_out  # trace_other saved to disk, not kept in memory

    logger.info("=" * 80)
    logger.info("STEP 4: Merging FISD and Preparing for Parallel Processing")
    logger.info("=" * 80)

    # Record starting row count for Table 2 (this is "start")
    n_start = len(final_df)
    FILTER_COUNTS['start'] = n_start
    logger.info("Starting rows: %d", n_start)

    # Define FISD columns to merge
    fisd_cols = [
        "cusip_id", "offering_date", "dated_date", "interest_frequency",
        "coupon", "day_count_basis", "coupon_type", "maturity",
        "principal_amt"
    , "ff17num", "ff30num"
    ]


    # Check which columns exist in fisd
    available_cols = [col for col in fisd_cols if col in fisd.columns]
    missing_cols = [col for col in fisd_cols if col not in fisd.columns]

    if missing_cols:
        logger.warning("FISD missing columns: %s", missing_cols)

    logger.info("Merging FISD columns: %s", available_cols)

    # Merge FISD and calculate bond_maturity/bond_age
    traced_pre_filter = (
        final_df
        .merge(fisd[available_cols], on="cusip_id", how="left")
    )

    # Free memory from final_df as it's no longer needed
    del final_df
    gc.collect()

    # Calculate bond_maturity and bond_age without lambda (faster)
    traced_pre_filter["bond_maturity"] = (traced_pre_filter["maturity"] - traced_pre_filter["trd_exctn_dt"]).dt.days / 365.25
    traced_pre_filter["bond_age"] = (traced_pre_filter["trd_exctn_dt"] - traced_pre_filter["offering_date"]).dt.days / 365.25

    # Convert interest_frequency to int for filtering
    traced_pre_filter["interest_frequency"] = traced_pre_filter["interest_frequency"].astype(int)

    n_before_accrued = len(traced_pre_filter)

    # Apply valid_accrued_vars filter
    traced = (
        traced_pre_filter
        .query("bond_maturity > 0")
        .query("bond_age > 0")
        .query("dated_date.notna()")
        .query("interest_frequency not in [-1, 13, 16]")
        .rename(columns={"prc_vw": "pr"})
    )

    # Free memory from traced_pre_filter immediately
    del traced_pre_filter
    gc.collect()

    # Convert interest_frequency to string (avoid lambda)
    traced["interest_frequency"] = traced["interest_frequency"].astype(str)

    n_after_accrued = len(traced)

    # Record valid_accrued_vars filter counts for Table 2
    FILTER_COUNTS['valid_accrued_vars'] = (n_before_accrued, n_after_accrued)

    logger.info("After valid_accrued_vars filter: %d -> %d rows (-%d)",
                n_before_accrued, n_after_accrued, n_before_accrued - n_after_accrued)

    gc.collect()

    # ----- Split for parallel processing -------------------------------------
    logger.info("Splitting data for parallel processing...")

    # Columns that will NOT be processed in parallel (keep aside)
    selected_cols = [
        "pr", "offering_date", "dated_date",
        "maturity", "bond_maturity", "day_count_basis", "interest_frequency",
        "coupon_type"
    ]

    # All other columns (not for parallel processing)
    trace_other = traced.drop(columns=selected_cols, errors='ignore').copy()
    logger.info("trace_other shape: %s", trace_other.shape)

    # Subset for heavy processing (bond analytics)
    traced_out = traced.sort_values(["cusip_id", "trd_exctn_dt"])[
        [
            "cusip_id", "trd_exctn_dt", "pr", "offering_date", "dated_date",
            "maturity", "bond_maturity", "day_count_basis", "interest_frequency",
            "coupon", "coupon_type"
        ]
    ].copy()

    logger.info("traced_out shape: %s", traced_out.shape)

    del traced
    gc.collect()

    # Check for required columns
    logger.info("traced_out columns: %s", list(traced_out.columns))

    # ----- Optimize dtypes to reduce memory usage ----------------------------
    logger.info("Optimizing dtypes to reduce memory usage...")

    # Define categorical columns for trace_other (low-cardinality string columns)
    categorical_cols = [
        'cusip_id', 'rating_type', 'day_count_basis', 'interest_frequency',
        'coupon_type', 'bond_type', 'rating_class'
    ]
    trace_other = hf.optimize_dtypes(trace_other, categorical_cols=categorical_cols)

    # Optimize traced_out
    traced_out_categorical = ['cusip_id', 'day_count_basis', 'interest_frequency', 'coupon_type']
    traced_out = hf.optimize_dtypes(traced_out, categorical_cols=traced_out_categorical)

    # ----- Save trace_other as chunked parquet files -------------------------
    # Save in N_CHUNKS files to avoid redundant I/O during step5
    # Each chunk corresponds to the same chunk ranges used in step5
    logger.info("Saving trace_other as %d chunked parquet files to free memory...", N_CHUNKS)

    chunk_size = int(np.ceil(len(trace_other) / N_CHUNKS))
    for i in range(N_CHUNKS):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(trace_other))

        chunk_path = STAGE1_DATA / f"temp_trace_other_chunk_{i:03d}.parquet"
        trace_other_chunk = trace_other.iloc[start_idx:end_idx]
        trace_other_chunk.to_parquet(chunk_path, index=False, compression='snappy')
        logger.info("Saved chunk %d/%d: %s (rows %d-%d)", i+1, N_CHUNKS, chunk_path.name, start_idx, end_idx)

    logger.info("trace_other saved as %d chunk files in: %s", N_CHUNKS, STAGE1_DATA)

    # Free memory by deleting trace_other
    del trace_other
    gc.collect()
    logger.info("trace_other removed from memory")

    print("\n[STEP 4 COMPLETE] FISD merged and data split for parallel processing")
    print(f"trace_other saved as {N_CHUNKS} chunk files (freed from memory)")
    print(f"traced_out shape: {traced_out.shape} (for parallel analytics)")
    print(f"Filters applied: bond_maturity > 0, bond_age > 0, dated_date not null")

    gc.collect()
    mem_end = hf.log_memory_usage("step4_merge_fisd_end")
    hf.log_memory_delta(mem_start, mem_end, "step4_merge_fisd")
    return traced_out  # Only return traced_out now


# ============================================================================
# STEP 5: COMPUTE BOND ANALYTICS (PARALLEL)
# ============================================================================
def step5_compute_bond_analytics():
    """Compute bond analytics in sequential chunks with credit spreads.

    Memory optimization: Uses chunked merge-and-write pattern to avoid holding
    full analytics_df + trace_other + final_df in memory simultaneously.
    """
    mem_start = hf.log_memory_usage("step5_compute_bond_analytics_start")
    global final_df, traced_out

    logger.info("=" * 80)
    logger.info("STEP 5: Computing Bond Analytics (Sequential Chunks with Incremental Merge)")
    logger.info("=" * 80)

    columns1=['cusip_id', 'trd_exctn_dt', 'pr', 'prclean', 'prfull',
             'acclast', 'accpmt', 'accall', 'ytm', 'mod_dur','mac_dur', 'convexity',
             'bond_maturity']

    # ----- Chunk + parallel --------------------------------------------------
    if N_CHUNKS <= 0:
        raise ValueError("N_CHUNKS must be >= 1")

    chunk_size = int(np.ceil(len(traced_out) / N_CHUNKS))
    total_rows = len(traced_out)
    logger.info("Processing %d rows in %d chunks (chunk_size=%d)", total_rows, N_CHUNKS, chunk_size)

    # Verify trace_other chunk files exist (saved in step4)
    chunk_0_path = STAGE1_DATA / "temp_trace_other_chunk_000.parquet"
    if not chunk_0_path.exists():
        raise FileNotFoundError(
            f"trace_other chunk files not found. Expected {chunk_0_path} and similar. "
            f"Ensure step4 completed successfully."
        )

    # Path for incremental output
    final_output_path = STAGE1_DATA / "temp_final_merged.parquet"
    if final_output_path.exists():
        final_output_path.unlink()  # Remove if exists from previous run
        logger.info("Removed existing temp final output")

    logger.info("Will read trace_other from chunked parquet files in: %s", STAGE1_DATA)
    logger.info("Will write merged results to: %s", final_output_path)

    # ----- Process each chunk with incremental merge -------------------------
    for i in range(N_CHUNKS):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)

        logger.info("=" * 60)
        logger.info("Processing chunk %d/%d | rows %d-%d", i + 1, N_CHUNKS, start_idx, end_idx)

        # 1. Get traced_out chunk for analytics computation
        chunk = traced_out.iloc[start_idx:end_idx].copy()
        logger.info("Chunk shape for analytics: %s", chunk.shape)

        # 2. Process bond analytics using helper function
        logger.info("Computing bond analytics...")
        processed = hf.process_chunk(chunk, N_CORES)
        processed.columns = columns1
        gc.collect()

        # 3. Calculate credit spreads
        logger.info("Calculating credit spreads...")
        spreads = hf.calculate_credit_spreads(processed, ylds)

        # 4. Merge spreads back into processed chunk
        analytics_chunk = processed.merge(
            spreads[['cusip_id', 'trd_exctn_dt', 'credit_spread']],
            how="left",
            left_on=['cusip_id', 'trd_exctn_dt'],
            right_on=['cusip_id', 'trd_exctn_dt']
        )

        del chunk, spreads, processed
        gc.collect()

        # 5. Read corresponding trace_other chunk from pre-chunked parquet file
        # Each chunk was saved as a separate file in step4, avoiding redundant I/O
        chunk_path = STAGE1_DATA / f"temp_trace_other_chunk_{i:03d}.parquet"
        logger.info("Reading trace_other chunk from: %s", chunk_path.name)

        trace_other_chunk = pd.read_parquet(chunk_path)
        logger.info("trace_other_chunk shape: %s", trace_other_chunk.shape)

        # 6. Merge analytics with trace_other chunk
        logger.info("Merging analytics with trace_other columns...")
        merged_chunk = analytics_chunk.merge(
            trace_other_chunk,
            on=["cusip_id", "trd_exctn_dt"],
            how="left"
        )
        logger.info("Merged chunk shape: %s", merged_chunk.shape)

        # Optimize dtypes of merged chunk before writing
        merged_chunk = hf.optimize_dtypes(merged_chunk)

        del analytics_chunk, trace_other_chunk
        gc.collect()

        # 7. Append to output parquet (incremental write)
        if i == 0:
            # First chunk: create new file
            merged_chunk.to_parquet(final_output_path, index=False, compression='snappy')
            logger.info("Created output parquet with first chunk")
        else:
            # Subsequent chunks: append
            existing = pd.read_parquet(final_output_path)
            combined = pd.concat([existing, merged_chunk], ignore_index=True)
            combined.to_parquet(final_output_path, index=False, compression='snappy')
            del existing, combined
            logger.info("Appended chunk to output parquet")

        del merged_chunk
        gc.collect()

        logger.info("[OK] Chunk %d/%d complete", i + 1, N_CHUNKS)

    # ----- Load final merged result -------------------------------------------
    logger.info("=" * 80)
    logger.info("All chunks processed. Loading final merged dataset...")
    final_df = pd.read_parquet(final_output_path)
    logger.info("Final merged shape: %s", final_df.shape)

    # Clean up traced_out from memory
    del traced_out
    gc.collect()

    # Clean up temporary files
    logger.info("Cleaning up temporary files...")

    # Remove all trace_other chunk files
    chunk_files_removed = 0
    for i in range(N_CHUNKS):
        chunk_path = STAGE1_DATA / f"temp_trace_other_chunk_{i:03d}.parquet"
        if chunk_path.exists():
            chunk_path.unlink()
            chunk_files_removed += 1
    logger.info("Removed %d trace_other chunk files", chunk_files_removed)

    # Remove final merged temp file
    if final_output_path.exists():
        final_output_path.unlink()
        logger.info("Removed temp_final_merged.parquet")

    print("\n[STEP 5 COMPLETE] Bond analytics computed with chunked merge")
    print(f"Shape: {final_df.shape}")
    print(f"New analytics columns: ytm, mod_dur, convexity, credit_spread, etc.")
    print(f"Memory optimization: Avoided holding full datasets simultaneously")

    mem_end = hf.log_memory_usage("step5_compute_bond_analytics_end")
    hf.log_memory_delta(mem_start, mem_end, "step5_compute_bond_analytics")
    return final_df


# ============================================================================
# STEP 6: LOAD AND MERGE RATINGS
# ============================================================================
def step6_merge_ratings():
    """Load and merge Amount Outstanding, then S&P and Moody's ratings."""
    mem_start = hf.log_memory_usage("step6_merge_ratings_start")
    global final_df, sp_ratings, moodys_ratings, call_dummy, db
    
    logger.info("=" * 80)
    logger.info("STEP 6: Amount Outstanding & Ratings")
    logger.info("=" * 80)
    
    if db is None:
        logger.info("Connecting to WRDS...")
        db = wrds.Connection(wrds_username=WRDS_USERNAME)
        logger.info("WRDS connection established")
    
    # ========================================================================
    # PART 1: AMOUNT OUTSTANDING
    # ========================================================================
    logger.info("=== Amount Outstanding: download & merge started ===")
    
    # Download historical amount-outstanding and issue tables
    amt_out = db.get_table(library="fisd", table="fisd_amt_out_hist")
    issues_for_amt = db.get_table(
        library="fisd",
        table="fisd_mergedissue",
        columns=["issue_id", "complete_cusip", "offering_amt", "offering_date"]
    )
    
    logger.info("Amount outstanding rows: %d", len(amt_out))
    logger.info("Issues for offering_amt rows: %d", len(issues_for_amt))
    
    # --- Tidy ---
    amt_out.rename(columns={"amount_outstanding": "bond_amt_outstanding",
                            "effective_date": "amt_date"},
                   inplace=True)
    issues_for_amt.rename(columns={"complete_cusip": "cusip_id"}, inplace=True)
    
    mergent_amounts = amt_out.merge(
        issues_for_amt[["issue_id", "cusip_id", "offering_amt", "offering_date"]],
        on="issue_id",
        how="inner"
    )

    # Convert to category for memory optimization
    if "cusip_id" in mergent_amounts.columns:
        mergent_amounts['cusip_id'] = mergent_amounts['cusip_id'].astype('category')
    if "cusip_id" in issues_for_amt.columns:
        issues_for_amt['cusip_id'] = issues_for_amt['cusip_id'].astype('category')

    # Need a valid issue map (keeps only bonds appearing in the FISD reference file)
    fisd_ref = fisd[["issue_id"]].copy() if "issue_id" in fisd.columns else None
    if fisd_ref is not None:
        mergent_amounts = mergent_amounts.merge(
            fisd_ref[["issue_id"]],
            on="issue_id",
            how="left"
        )
    
    # --- Dates ---
    for col in ["amt_date", "offering_date"]:
        mergent_amounts[col] = pd.to_datetime(mergent_amounts[col])
    
    final_df["trd_exctn_dt"] = pd.to_datetime(final_df["trd_exctn_dt"])
    
    # --- Basic QC logs ---
    logger.info("Initial amount-outstanding records: %d", len(mergent_amounts))
    logger.info("Null CUSIPs: %d", mergent_amounts["cusip_id"].isna().sum())
    logger.info("Null amt_date: %d", mergent_amounts["amt_date"].isna().sum())
    
    # Drop bad rows and zero amounts
    mergent_amounts = mergent_amounts.dropna(subset=["cusip_id", "amt_date"])
    logger.info("After null-drop: %d", len(mergent_amounts))
    
    mergent_amounts = mergent_amounts[mergent_amounts["bond_amt_outstanding"] > 0]
    logger.info("After zero-amount filter: %d", len(mergent_amounts))
    
    # Keep latest entry per (cusip, amt_date)
    mergent_amounts = (
        mergent_amounts
        .sort_values(["cusip_id", "amt_date", "bond_amt_outstanding"])
        .drop_duplicates(subset=["cusip_id", "amt_date"], keep="last")
    )
    logger.info("After duplicate prune: %d", len(mergent_amounts))
    
    # --- Merge into final_df ---
    final_df = final_df.sort_values(["trd_exctn_dt"])
    mergent_amounts = mergent_amounts.sort_values(["amt_date"])
    
    final_df["cusip_id"] = final_df["cusip_id"].astype(str)
    mergent_amounts["cusip_id"] = mergent_amounts["cusip_id"].astype(str)
    
    logger.info("Merging %d TRACE rows with %d amount-outstanding rows...",
                len(final_df), len(mergent_amounts))
    
    final_df = pd.merge_asof(
        final_df,
        mergent_amounts[["cusip_id", "amt_date", "bond_amt_outstanding"]],
        left_on="trd_exctn_dt",
        right_on="amt_date",
        by="cusip_id",
        direction="backward"
    )
    
    # Fill remaining gaps with original offering amount
    final_df = final_df.merge(
        issues_for_amt[["cusip_id", "offering_amt"]],
        on="cusip_id",
        how="left"
    )
    
    final_df = final_df.sort_values(["cusip_id", "trd_exctn_dt"])
    
    final_df["bond_amt_outstanding"] = final_df["bond_amt_outstanding"].fillna(
        final_df["offering_amt"]
    )

    final_df.drop(columns=["amt_date", "offering_amt"], inplace=True, errors="ignore")

    # Re-convert to category after merge to maintain memory optimization
    if "cusip_id" in final_df.columns:
        final_df['cusip_id'] = final_df['cusip_id'].astype('category')

    amt_merge_rate = (final_df["bond_amt_outstanding"].notna().sum() / len(final_df)) * 100
    logger.info("Amount outstanding merge rate: %.2f%%", amt_merge_rate)
    logger.info("=== Amount Outstanding: merge complete ===")
    
    gc.collect()
    del amt_out, mergent_amounts, issues_for_amt
    
    # ========================================================================
    # PART 2: RATINGS
    # ========================================================================
    logger.info("=== Ratings: download & merge started ===")
    
    # --- Fetch ratings ---
    logger.info("Downloading S&P ratings (SPR)...")
    sp_ratings = db.raw_sql("""
        SELECT issue_id, rating_date, rating
        FROM   fisd.fisd_ratings
        WHERE  rating_type = 'SPR'
    """)
    
    logger.info("Downloading Moody's ratings (MR)...")
    moodys_ratings = db.raw_sql("""
        SELECT issue_id, rating_date, rating
        FROM   fisd.fisd_ratings
        WHERE  rating_type = 'MR'
    """)
    
    logger.info("Downloading issue map ...")
    issues = db.raw_sql("""
        SELECT issue_id, complete_cusip
        FROM   fisd.fisd_mergedissue
    """)
    issues.rename(columns={"complete_cusip": "cusip_id"}, inplace=True)
    
    logger.info("S&P rows: %d | Moody's rows: %d | Issue rows: %d",
                len(sp_ratings), len(moodys_ratings), len(issues))
    
    # --- Fetch call dummy variable (WRDS) ---
    logger.info("Downloading Bond Call Dummies...")
    fisd_r = db.raw_sql("""
        SELECT issue_id, callable
        FROM   fisd.fisd_mergedredemption
    """)
    
    fisd_r.dropna(inplace=True)
    fisd_r['issue_id'] = fisd_r['issue_id'].astype(int)
    fisd_r['callable'] = (fisd_r['callable'] == 'Y').astype(int)
    
    # --- Merge CUSIPs to ratings ---
    sp_ratings = sp_ratings.merge(
        issues[["issue_id", "cusip_id"]], 
        on="issue_id", 
        how="inner"
    ).rename(columns={
        "rating": "sp_rating",
        "rating_date": "sp_rating_date"
    })
    
    moodys_ratings = moodys_ratings.merge(
        issues[["issue_id", "cusip_id"]],
        on="issue_id",
        how="inner"
    ).rename(columns={
        "rating": "mdy_rating",
        "rating_date": "mdy_rating_date"
    })

    # Convert to category for memory optimization
    if "cusip_id" in sp_ratings.columns:
        sp_ratings['cusip_id'] = sp_ratings['cusip_id'].astype('category')
    if "cusip_id" in moodys_ratings.columns:
        moodys_ratings['cusip_id'] = moodys_ratings['cusip_id'].astype('category')

    # --- Convert dates ---
    sp_ratings["sp_rating_date"] = pd.to_datetime(sp_ratings["sp_rating_date"])
    moodys_ratings["mdy_rating_date"] = pd.to_datetime(moodys_ratings["mdy_rating_date"])
    
    # --- Drop NAs ---
    sp_ratings = sp_ratings.dropna(subset=["cusip_id", "sp_rating_date", "sp_rating"])
    moodys_ratings = moodys_ratings.dropna(subset=["cusip_id", "mdy_rating_date", "mdy_rating"])
    
    logger.info("S&P after QC: %d | Moody's after QC: %d", len(sp_ratings), len(moodys_ratings))
    
    # --- Add numeric and NAIC ratings ---
    sp_ratings["sp_rating_numeric"] = sp_ratings["sp_rating"].map(hf.convert_sp_to_numeric)
    sp_ratings["sp_naic_numeric"] = sp_ratings["sp_rating_numeric"].map(hf.numeric_to_naic)
    
    moodys_ratings["mdy_rating_numeric"] = moodys_ratings["mdy_rating"].map(hf.convert_moodys_to_numeric)
    moodys_ratings["mdy_naic_numeric"] = moodys_ratings["mdy_rating_numeric"].map(hf.numeric_to_naic)
    
    # --- Keep relevant columns ---
    sp_ratings = sp_ratings[[
        "cusip_id", "sp_rating_date",
        "sp_rating_numeric", "sp_naic_numeric"
    ]]
    
    moodys_ratings = moodys_ratings[[
        "cusip_id", "mdy_rating_date",
        "mdy_rating_numeric", "mdy_naic_numeric"
    ]]
    
    # --- Prep for merge_asof ---
    final_df["trd_exctn_dt"] = pd.to_datetime(final_df["trd_exctn_dt"])
    sp_ratings["sp_rating_date"] = pd.to_datetime(sp_ratings["sp_rating_date"])
    moodys_ratings["mdy_rating_date"] = pd.to_datetime(moodys_ratings["mdy_rating_date"])
    
    final_df["cusip_id"] = final_df["cusip_id"].astype(str)
    sp_ratings["cusip_id"] = sp_ratings["cusip_id"].astype(str)
    moodys_ratings["cusip_id"] = moodys_ratings["cusip_id"].astype(str)
    
    # --- Sort for merge_asof ---
    final_df = final_df.sort_values(["trd_exctn_dt"]).reset_index(drop=True)
    sp_ratings = sp_ratings.sort_values(["sp_rating_date"]).reset_index(drop=True)
    moodys_ratings = moodys_ratings.sort_values(["mdy_rating_date"]).reset_index(drop=True)
    
    # --- Merge S&P ratings ---
    logger.info("Merging S&P ratings with asof...")
    final_df = pd.merge_asof(
        final_df,
        sp_ratings,
        left_on="trd_exctn_dt",
        right_on="sp_rating_date",
        by="cusip_id",
        direction="backward"
    )
    
    # --- Merge Moody's ratings ---
    logger.info("Merging Moody's ratings with asof...")
    final_df = pd.merge_asof(
        final_df,
        moodys_ratings,
        left_on="trd_exctn_dt",
        right_on="mdy_rating_date",
        by="cusip_id",
        direction="backward"
    )
    
    # --- Merge Call Dummies to main panel ---
    logger.info("Merging call dummies...")
    
    final_df = final_df.merge(
        fisd[['issue_id','cusip_id']],
        how="left",
        left_on=['cusip_id'],
        right_on=['cusip_id']
    )
    
    final_df = final_df.merge(
        fisd_r,
        how="left",
        left_on=['issue_id'],
        right_on=['issue_id']
    )
    final_df.drop(['issue_id'], axis=1, inplace=True, errors="ignore")
    final_df['callable'] = final_df['callable'].fillna(0).astype(int)

    # Re-convert to category after merges to maintain memory optimization
    if "cusip_id" in final_df.columns:
        final_df['cusip_id'] = final_df['cusip_id'].astype('category')

    # --- Create composite ratings ---
    final_df["mdy_rating_numeric_adj"] = np.where(
        final_df["mdy_rating_numeric"] >= 21, 22, final_df["mdy_rating_numeric"]
    )
    
    final_df["sp_rating_composite"] = np.where(
        final_df["sp_rating_numeric"].isna(),
        final_df["mdy_rating_numeric_adj"],
        final_df["sp_rating_numeric"]
    )
    
    final_df["mdy_rating_composite"] = np.where(
        final_df["mdy_rating_numeric_adj"].isna(),
        final_df["sp_rating_numeric"],
        final_df["mdy_rating_numeric_adj"]
    )
    
    final_df["comp_rating"] = final_df[[
        "sp_rating_composite",
        "mdy_rating_composite"
    ]].mean(axis=1)
    
    # --- Rename columns ---
    final_df.rename(columns={
        "sp_rating_composite": "spc_rating",
        "mdy_rating_composite": "mdc_rating",
        "sp_naic_numeric": "sp_naic",
        "sp_rating_numeric":"sp_rating",
        "mdy_rating_numeric":"mdy_rating"
    }, inplace=True)
    
    final_df.drop(columns=["sp_rating_date",
                           "mdy_rating_date",
                           "mdy_rating_numeric_adj",
                           "sp_naic_numeric",
                           "mdy_naic_numeric"], errors="ignore",
                  inplace=True)
    
    logger.info("=== Ratings: merge complete ===")
    gc.collect()
    
    # Store call_dummy for return before deletion
    call_dummy = fisd_r.copy()
    
    del fisd_r
    
    print("\n[STEP 6 COMPLETE] Amount Outstanding & Ratings merged")
    print(f"Shape: {final_df.shape}")
    print(f"Amount outstanding merge rate: {amt_merge_rate:.2f}%")
    print(f"Rating columns: sp_rating, mdy_rating, spc_rating, mdc_rating, comp_rating")
    
    mem_end = hf.log_memory_usage("step6_merge_ratings_end")
    hf.log_memory_delta(mem_start, mem_end, "step6_merge_ratings")
    return final_df, sp_ratings, moodys_ratings, call_dummy


# ============================================================================
# STEP 7: MERGE EQUITY IDENTIFIERS (LINKER)
# ============================================================================
def step7_merge_linker():
    """Merge equity identifiers from linker file."""
    mem_start = hf.log_memory_usage("step7_merge_linker_start")
    global final_df

    logger.info("=" * 80)
    logger.info("STEP 7: Merging Equity Identifiers")
    logger.info("=" * 80)

    # Check if internet is available
    has_internet = hf._check_internet_connectivity()

    # Try to load linker file - either from internet or local file
    if has_internet:
        try:
            logger.info("Internet available - downloading OSBAP linker from URL")
            logger.info("Loading linker file...")
            dfl = hf.load_parquet_from_zip_url(LINKER_URL, LINKER_ZIPKEY).copy()
            logger.info("Successfully downloaded and extracted OSBAP linker")
        except Exception as e:
            logger.warning(f"Failed to download from internet: {e}")
            logger.info("Falling back to local file")
            has_internet = False  # Trigger local file fallback

    if not has_internet:
        # No internet - use local file
        from pathlib import Path
        local_file = f"data/{LINKER_ZIPKEY}"
        local_path = Path(local_file)

        if not local_path.exists():
            raise FileNotFoundError(
                f"No internet connection and local file not found: {local_file}\n"
                f"Please download the file manually:\n"
                f"  wget -O data/linker_file_2025.zip \"{LINKER_URL}\"\n"
                f"  unzip data/linker_file_2025.zip -d data/\n"
                f"Or run this from a machine with internet access."
            )

        logger.info(f"Internet not available - loading OSBAP linker from local file: {local_file}")
        dfl = pd.read_parquet(local_path).copy()
        logger.info(f"Successfully loaded OSBAP linker from {local_file}")

    dfl.columns = dfl.columns.str.lower()

    dfl["date"] = pd.to_datetime(dfl["yyyymm"], format="%Y%m", errors="coerce")
    dfl["year_month"] = dfl["date"].dt.to_period("M").astype(str)
    dfl["permno"] = pd.to_numeric(dfl["permno"], errors="coerce").astype("Int64")

    # Handle permco and gvkey - may not exist in all linker files
    if "permco" in dfl.columns:
        dfl["permco"] = pd.to_numeric(dfl["permco"], errors="coerce").astype("Int64")
        logger.info("permco column found and processed")
    else:
        logger.warning("permco column not found in linker file - will be NaN in output")

    if "gvkey" in dfl.columns:
        # Convert to integer, preserving missing values
        dfl['gvkey'] = pd.to_numeric(dfl['gvkey'].round(0), errors='coerce').astype('Int32')
        logger.info("gvkey column found and processed")
    else:
        logger.warning("gvkey column not found in linker file - will be NaN in output")

    # Extend linker with forward fill
    ffill_date = pd.to_datetime(final_df["trd_exctn_dt"].max()) + MonthEnd(0)
    dfl = hf.extend_and_ffill_linker(dfl, ffill_date)
    dfl = dfl.drop(columns=["date"], errors="ignore")

    # Prep keys
    final_df["issuer_cusip"] = final_df["cusip_id"].astype(str).str[:6]
    final_df["trd_exctn_dt"] = pd.to_datetime(final_df["trd_exctn_dt"], errors="coerce")
    final_df["year_month"] = final_df["trd_exctn_dt"].dt.to_period("M").astype(str)

    # Merge
    logger.info("Merging linker on issuer_cusip and year_month...")
    before = len(final_df)
    final_df = final_df.merge(
        dfl,
        on=["issuer_cusip", "year_month"],
        how="left"
    )
    after = len(final_df)
    logger.info("Linker merge: %d -> %d rows", before, after)

    final_df = final_df.drop(columns=["year_month", "yyyymm"], errors="ignore")

    # Re-convert to category after merge to maintain memory optimization
    if "cusip_id" in final_df.columns:
        final_df['cusip_id'] = final_df['cusip_id'].astype('category')
    if "issuer_cusip" in final_df.columns:
        final_df['issuer_cusip'] = final_df['issuer_cusip'].astype('category')

    # Check merge success
    merge_rate = (final_df['permno'].notna().sum() / len(final_df)) * 100
    logger.info("Merge success rate: %.2f%% of rows have equity IDs", merge_rate)

    # Reorder columns
    first_cols = ["cusip_id", "issuer_cusip", "permno"] + \
                  (["permco"] if "permco" in final_df.columns else []) + \
                  (["gvkey"] if "gvkey" in final_df.columns else []) + \
                  ["trd_exctn_dt"]
    rest = [c for c in final_df.columns if c not in first_cols]
    final_df = final_df[first_cols + rest]

    final_df = final_df.sort_values(["cusip_id", "trd_exctn_dt"])

    logger.info("Linker merge complete")

    print("\n[STEP 7 COMPLETE] Equity identifiers merged")
    print(f"Shape: {final_df.shape}")
    print(f"Equity ID columns: permno, permco, gvkey")
    print(f"Merge rate: {merge_rate:.2f}%")

    # Delete linker dataframe - no longer needed
    logger.info("Deleting linker dataframe (dfl) to free memory...")
    del dfl
    gc.collect()

    mem_end = hf.log_memory_usage("step7_merge_linker_end")
    hf.log_memory_delta(mem_start, mem_end, "step7_merge_linker")
    return final_df

# ============================================================================
# VARIABLE DROP & MEMORY OPTIMIZATION (between step7 and step8)
# ============================================================================

def variable_drop():
    """
    Drop unnecessary columns and optimize memory before step8.

    This function:
    1. Drops columns no longer needed: issuer_cusip, prclean, coupon,
       principal_amt, sp_naic, comp_rating, callable
    2. Optimizes data types for RAM efficiency
    3. Exports sp_ratings, moodys_ratings, call_dummy to parquet and
       deletes them from memory
    """
    mem_start = hf.log_memory_usage("variable_drop_start")
    global final_df, sp_ratings, moodys_ratings, call_dummy

    logger.info("=" * 80)
    logger.info("VARIABLE DROP & MEMORY OPTIMIZATION")
    logger.info("=" * 80)

    # ========================================================================
    # PART 1: Drop unnecessary columns from final_df
    # ========================================================================
    cols_to_drop = [
        'issuer_cusip', 'prclean', 'coupon', 'principal_amt',
        'sp_naic', 'comp_rating', 'callable'
    ]

    dropped = []
    for col in cols_to_drop:
        if col in final_df.columns:
            final_df.drop(columns=[col], inplace=True)
            dropped.append(col)

    if dropped:
        logger.info("Dropped %d columns: %s", len(dropped), ", ".join(dropped))
    else:
        logger.info("No columns to drop (already absent)")

    gc.collect()

    # ========================================================================
    # PART 2: Optimize data types for RAM efficiency
    # ========================================================================
    logger.info("Optimizing data types for RAM efficiency...")

    # Round float columns to reduce precision and save memory
    float_rounding = {
        'mod_dur': 3,
        'mac_dur': 3,
        'convexity': 3,
        'bond_maturity': 3,
        'bond_amt_outstanding': 0,
    }

    for col, decimals in float_rounding.items():
        if col in final_df.columns:
            original_dtype = final_df[col].dtype
            final_df[col] = final_df[col].round(decimals)
            logger.info("  Rounded %s to %d decimals (dtype: %s)", col, decimals, original_dtype)

    # Convert principal_amt to Int16 (if still present - may have been dropped)
    if 'principal_amt' in final_df.columns:
        try:
            original_dtype = final_df['principal_amt'].dtype
            final_df['principal_amt'] = final_df['principal_amt'].round(0).astype('Int16')
            logger.info("  principal_amt: %s -> Int16", original_dtype)
        except Exception as e:
            logger.warning("  Could not convert principal_amt to Int16: %s", e)

    # Define type conversions with guards
    type_conversions = {
        # IDs - Int32 (handles NaN, range up to ~2.1B)
        'permno': 'Int32',
        'permco': 'Int32',
        'gvkey': 'Int32',

        # Counts - Int16 (handles NaN, range up to 32,767)
        'trade_count': 'Int16',
        'bid_count': 'Int16',
        'ask_count': 'Int16',

        # Ratings - Int8 (handles NaN, range 1-22)
        'sp_rating': 'Int8',
        'sp_naic': 'Int8',
        'mdy_rating': 'Int8',
        'spc_rating': 'Int8',
        'mdc_rating': 'Int8',
        'comp_rating': 'Int8',

        # Binary flags - Int8 (handles NaN if present)
        'callable': 'Int8',
        'db_type': 'Int8',

        # Large amounts - Int64 (handles NaN)
        'bond_amt_outstanding': 'Int64',
    }

    # Apply conversions with guards
    n_optimized = 0
    for col, target_dtype in type_conversions.items():
        if col in final_df.columns:
            try:
                original_dtype = final_df[col].dtype
                final_df[col] = final_df[col].astype(target_dtype)
                n_optimized += 1
                logger.info("  %s: %s -> %s", col, original_dtype, target_dtype)
            except Exception as e:
                logger.warning("  Could not convert %s to %s: %s", col, target_dtype, e)

    logger.info("Optimized %d columns for RAM efficiency", n_optimized)
    gc.collect()

    # ========================================================================
    # PART 3: Export ratings and call_dummy, then delete from memory
    # ========================================================================
    logger.info("Exporting auxiliary objects to free memory...")

    timestamp = STAGE0_DATE_STAMP  # Use consistent timestamp

    # Export S&P ratings
    if sp_ratings is not None:
        out_file = STAGE1_DATA / f"sp_ratings_{timestamp}.parquet"
        sp_ratings.to_parquet(out_file, index=False)
        logger.info("[OK] S&P ratings saved early: %s", out_file)
        del sp_ratings
        sp_ratings = None
        logger.info("  Deleted sp_ratings from memory")

    # Export Moody's ratings
    if moodys_ratings is not None:
        out_file = STAGE1_DATA / f"moodys_ratings_{timestamp}.parquet"
        moodys_ratings.to_parquet(out_file, index=False)
        logger.info("[OK] Moody's ratings saved early: %s", out_file)
        del moodys_ratings
        moodys_ratings = None
        logger.info("  Deleted moodys_ratings from memory")

    # Export call dummy
    if call_dummy is not None:
        out_file = STAGE1_DATA / f"call_dummy_{timestamp}.parquet"
        call_dummy.to_parquet(out_file, index=False)
        logger.info("[OK] Call dummy saved early: %s", out_file)
        del call_dummy
        call_dummy = None
        logger.info("  Deleted call_dummy from memory")

    gc.collect()

    print("\n[VARIABLE DROP COMPLETE] Memory optimized before step8")
    print(f"Shape: {final_df.shape}")
    print(f"Columns dropped: {dropped}")
    print(f"Dtypes optimized: {n_optimized}")
    print(f"Auxiliary objects exported and deleted: sp_ratings, moodys_ratings, call_dummy")

    mem_end = hf.log_memory_usage("variable_drop_end")
    hf.log_memory_delta(mem_start, mem_end, "variable_drop")
    return final_df


# ============================================================================
# STEP 8: ULTRA DISTRESSED FILTERING
# ============================================================================

def step8_ultra_distressed():
    """Apply ultra distressed filtering to identify suspicious price observations.

    Memory optimization: Uses disk-backed chunking to avoid holding all processed
    chunks in memory simultaneously. Each chunk is written to a temporary parquet
    file after processing, then all files are read back and concatenated.
    """
    mem_start = hf.log_memory_usage("step8_ultra_distressed_start")
    global final_df

    logger.info("=" * 80)
    logger.info("STEP 8: Ultra Distressed Filtering")
    logger.info("=" * 80)

    # Get config
    cfg = ULTRA_DISTRESSED_CONFIG
    target_rows = cfg['target_rows_per_chunk']

    # Memory check before chunk creation
    mem_before_chunks = hf.log_memory_usage("step8_before_chunk_creation")

    # Create CUSIP-based chunks
    logger.info("Creating CUSIP-based chunks (target: ~%d rows per chunk)...", target_rows)

    # Get unique CUSIPs and their row counts
    cusip_counts = final_df.groupby('cusip_id', observed=True).size().reset_index(name='count')
    cusip_counts = cusip_counts.sort_values('cusip_id')

    # Create chunks by grouping CUSIPs to reach target size
    chunks = []
    current_chunk = []
    current_size = 0

    for _, row in cusip_counts.iterrows():
        cusip = row['cusip_id']
        count = row['count']

        # If adding this CUSIP would exceed target and we have some CUSIPs already, start new chunk
        if current_size + count > target_rows and current_chunk:
            chunks.append(current_chunk.copy())
            current_chunk = [cusip]
            current_size = count
        else:
            current_chunk.append(cusip)
            current_size += count

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    n_chunks = len(chunks)
    logger.info("Created %d chunks", n_chunks)
    for i, chunk_cusips in enumerate(chunks, 1):
        chunk_rows = cusip_counts[cusip_counts['cusip_id'].isin(chunk_cusips)]['count'].sum()
        logger.info("  Chunk %d: %d CUSIPs, %d rows", i, len(chunk_cusips), chunk_rows)

    # Memory check after chunk creation
    mem_after_chunks = hf.log_memory_usage("step8_after_chunk_creation")
    hf.log_memory_delta(mem_before_chunks, mem_after_chunks, "step8_chunk_creation")

    # Clean up cusip_counts
    del cusip_counts
    gc.collect()

    # ========================================================================
    # DISK-BACKED CHUNK PROCESSING
    # Write each processed chunk to disk to avoid memory accumulation
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Processing chunks with disk-backed storage...")
    logger.info("=" * 80)

    chunk_files = []

    for i, chunk_cusips in enumerate(chunks, 1):
        mem_chunk_start = hf.log_memory_usage(f"step8_chunk_{i}_start")
        logger.info("-" * 80)
        logger.info("Processing Chunk %d/%d", i, n_chunks)
        logger.info("-" * 80)

        # Extract chunk data
        chunk = final_df[final_df['cusip_id'].isin(chunk_cusips)].copy()
        logger.info("Chunk size: %d rows, %d CUSIPs", len(chunk), chunk['cusip_id'].nunique())

        # Apply filtering function
        start_time = time.time()
        chunk = hf.ultra_distressed_filter(
            chunk,
            price_col=cfg['price_col'],
            # Intraday inconsistency
            intraday_range_threshold=cfg['intraday_range_threshold'],
            intraday_price_threshold=cfg['intraday_price_threshold'],
            # Anomaly detection
            ultra_low_threshold=cfg['ultra_low_threshold'],
            min_normal_price_ratio=cfg['min_normal_price_ratio'],
            # Plateau detection
            plateau_ultra_low_threshold=cfg['plateau_ultra_low_threshold'],
            min_plateau_days=cfg['min_plateau_days'],
            # Round numbers
            suspicious_round_numbers=cfg['suspicious_round_numbers'],
            # Intraday consistency
            price_cols=cfg['price_cols'],
            high_spike_threshold=cfg['high_spike_threshold'],
            min_spike_ratio=cfg['min_spike_ratio'],
            recovery_ratio=cfg['recovery_ratio'],
            verbose=cfg['verbose']
        )
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info("Chunk %d processed in %.2f seconds", i, elapsed)

        # Write chunk to disk (disk-backed pattern)
        chunk_path = STAGE1_DATA / f"temp_ultra_chunk_{i:03d}.parquet"
        chunk.to_parquet(chunk_path, index=False, compression='snappy')
        chunk_files.append(chunk_path)
        logger.info("Chunk %d written to disk: %s", i, chunk_path.name)

        # Clear chunk from memory
        del chunk
        gc.collect()

        mem_chunk_end = hf.log_memory_usage(f"step8_chunk_{i}_end")
        hf.log_memory_delta(mem_chunk_start, mem_chunk_end, f"step8_chunk_{i}")

    # ========================================================================
    # DELETE ORIGINAL final_df BEFORE CONCAT
    # This is critical - we no longer need the original data
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Freeing original final_df before concatenation...")
    logger.info("=" * 80)
    mem_before_del = hf.log_memory_usage("step8_before_del_final_df")

    del final_df
    gc.collect()

    mem_after_del = hf.log_memory_usage("step8_after_del_final_df")
    hf.log_memory_delta(mem_before_del, mem_after_del, "step8_del_final_df")

    # ========================================================================
    # CONCATENATE ALL CHUNKS FROM DISK
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Concatenating all chunks from disk...")
    logger.info("=" * 80)
    mem_before_concat = hf.log_memory_usage("step8_before_concat")

    # Read all chunk files and concatenate
    chunk_dfs = []
    for chunk_path in chunk_files:
        chunk_dfs.append(pd.read_parquet(chunk_path))

    final_df = pd.concat(chunk_dfs, ignore_index=True)
    del chunk_dfs
    gc.collect()

    mem_after_concat = hf.log_memory_usage("step8_after_concat")
    hf.log_memory_delta(mem_before_concat, mem_after_concat, "step8_concat")

    # Clean up temporary chunk files
    logger.info("Cleaning up temporary chunk files...")
    for chunk_path in chunk_files:
        if chunk_path.exists():
            chunk_path.unlink()
    logger.info("Removed %d temporary chunk files", len(chunk_files))

    # Convert CUSIP columns to category dtype for memory efficiency
    logger.info("Converting CUSIP columns to category dtype...")
    if 'cusip_id' in final_df.columns:
        final_df['cusip_id'] = final_df['cusip_id'].astype('category')
        logger.info("  cusip_id converted to category (reduces memory)")
    if 'issuer_cusip' in final_df.columns:
        final_df['issuer_cusip'] = final_df['issuer_cusip'].astype('category')
        logger.info("  issuer_cusip converted to category (reduces memory)")

    # Sort by cusip and date
    final_df = final_df.sort_values(['cusip_id', 'trd_exctn_dt']).reset_index(drop=True)

    # Re-convert to category after concat to maintain memory optimization
    if "cusip_id" in final_df.columns:
        final_df['cusip_id'] = final_df['cusip_id'].astype('category')
    if "issuer_cusip" in final_df.columns:
        final_df['issuer_cusip'] = final_df['issuer_cusip'].astype('category')

    # Report filtering results
    flag_cols = ['flag_anomalous_price', 'flag_upward_spike', 'flag_plateau_sequence', 
                 'flag_intraday_inconsistent', 'flag_refined_any']
    
    logger.info("Ultra Distressed Filtering Results:")
    for col in flag_cols:
        if col in final_df.columns:
            count = final_df[col].sum()
            pct = 100 * count / len(final_df)
            logger.info("  %s: %d (%.2f%%)", col, count, pct)
    
    print("\n[STEP 8 COMPLETE] Ultra distressed filtering applied")
    print(f"Shape: {final_df.shape}")
    print(f"New columns: {flag_cols}")
    print(f"Total flagged observations: {final_df['flag_refined_any'].sum():,} ({100*final_df['flag_refined_any'].sum()/len(final_df):.2f}%)")

    # ========================================================================
    # Export CUSIPs impacted by ultra distressed filter
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Exporting CUSIPs impacted by ultra distressed filter...")
    logger.info("=" * 80)

    # Identify CUSIPs with any flagged observations
    flagged_cusips = final_df[final_df['flag_refined_any'] == 1]['cusip_id'].unique()
    n_flagged_cusips = len(flagged_cusips)
    total_cusips = final_df['cusip_id'].nunique()
    pct_flagged = 100 * n_flagged_cusips / total_cusips

    logger.info("CUSIPs impacted by ultra distressed filter: %d (%.2f%% of all CUSIPs)",
                n_flagged_cusips, pct_flagged)

    if n_flagged_cusips > 0:
        # Efficient approach: Filter once, then use groupby operations (fast for 30M+ rows)
        logger.info("Building CUSIP summary using efficient groupby operations...")

        # Filter to only flagged CUSIPs once (much faster than filtering in loop)
        flagged_df_subset = final_df[final_df['cusip_id'].isin(flagged_cusips)].copy()

        # Group by CUSIP and compute all statistics in one pass
        grouped = flagged_df_subset.groupby('cusip_id')

        # Total observations per CUSIP
        total_obs = grouped.size().rename('total_observations')

        # Flagged observations per CUSIP
        flagged_obs = grouped['flag_refined_any'].sum().rename('flagged_observations')

        # Individual flag counts (check column existence first)
        flag_stats = {}
        if 'flag_anomalous_price' in flagged_df_subset.columns:
            flag_stats['flag_anomalous_price'] = grouped['flag_anomalous_price'].sum()
        if 'flag_upward_spike' in flagged_df_subset.columns:
            flag_stats['flag_upward_spike'] = grouped['flag_upward_spike'].sum()
        if 'flag_plateau_sequence' in flagged_df_subset.columns:
            flag_stats['flag_plateau_sequence'] = grouped['flag_plateau_sequence'].sum()
        if 'flag_intraday_inconsistent' in flagged_df_subset.columns:
            flag_stats['flag_intraday_inconsistent'] = grouped['flag_intraday_inconsistent'].sum()

        # Date ranges per CUSIP
        date_ranges = grouped['trd_exctn_dt'].agg(['min', 'max'])
        date_ranges.columns = ['first_trade_date', 'last_trade_date']

        # Combine all statistics
        flagged_df = pd.concat([total_obs, flagged_obs] + list(flag_stats.values()), axis=1)
        flagged_df = flagged_df.join(date_ranges)
        flagged_df = flagged_df.reset_index()

        # Calculate percentage flagged
        flagged_df['pct_flagged'] = (100 * flagged_df['flagged_observations'] /
                                      flagged_df['total_observations']).round(2)

        # Reorder columns for better readability
        col_order = ['cusip_id', 'total_observations', 'flagged_observations', 'pct_flagged']
        col_order += [c for c in flag_stats.keys()]
        col_order += ['first_trade_date', 'last_trade_date']
        flagged_df = flagged_df[col_order]

        # Sort by flagged observations (descending)
        flagged_df = flagged_df.sort_values('flagged_observations', ascending=False)

        # Clean up temporary subset
        del flagged_df_subset
        gc.collect()

        # Export to CSV
        output_path = STAGE1_DATA / f"ultra_distressed_cusips_{STAGE0_DATE_STAMP}.csv"
        flagged_df.to_csv(output_path, index=False)

        logger.info("Exported flagged CUSIPs summary to: %s", output_path)
        logger.info("Summary statistics:")
        logger.info("  Total CUSIPs flagged: %d", n_flagged_cusips)
        logger.info("  Median flagged observations per CUSIP: %d", flagged_df['flagged_observations'].median())
        logger.info("  Mean flagged observations per CUSIP: %.1f", flagged_df['flagged_observations'].mean())
        logger.info("  Max flagged observations (single CUSIP): %d", flagged_df['flagged_observations'].max())

        print(f"\nExported {n_flagged_cusips:,} flagged CUSIPs to: {output_path.name}")
    else:
        logger.info("No CUSIPs flagged by ultra distressed filter (nothing to export)")

    gc.collect()

    # End of step 8
    mem_end = hf.log_memory_usage("step8_ultra_distressed_end")
    hf.log_memory_delta(mem_start, mem_end, "step8_ultra_distressed")
    return final_df


# ============================================================================
# STEP 8B: BUILD DISTRESSED REPORT
# ============================================================================

def step8b_build_distressed_report(
    output_figures: bool = True,
    subplot_dim: tuple = (4, 2),
    use_latex: bool = False,
    price_col: str = "pr",
):
    """
    Build LaTeX report documenting the ultra distressed filter results.

    Creates figures for all CUSIPs impacted by the filter and assembles
    a LaTeX document with summary statistics and price series plots.

    Parameters
    ----------
    output_figures : bool
        Whether to generate figure pages (default True)
    subplot_dim : tuple
        Grid dimensions for figure pages (default (4, 2) = 8 panels per page)
    use_latex : bool
        Whether to use LaTeX fonts in figures (default False for speed)
    price_col : str
        Column containing prices to plot (default 'pr')
    """
    mem_start = hf.log_memory_usage("step8b_build_distressed_report_start")
    global final_df

    logger.info("=" * 80)
    logger.info("STEP 8B: Building Ultra Distressed Filter Report")
    logger.info("=" * 80)

    # Create output directory
    report_dir = STAGE1_DATA / "data_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Report output directory: %s", report_dir)

    # Calculate summary statistics
    total_rows = len(final_df)
    total_cusips = final_df['cusip_id'].nunique()

    flag_col = 'flag_refined_any'
    if flag_col not in final_df.columns:
        logger.warning("Column '%s' not found. Run step8_ultra_distressed() first.", flag_col)
        return

    flagged_rows = int(final_df[flag_col].sum())
    flagged_cusips_list = final_df[final_df[flag_col] == 1]['cusip_id'].unique()
    flagged_cusips = len(flagged_cusips_list)

    logger.info("Summary statistics:")
    logger.info("  Total observations: %s", f"{total_rows:,}")
    logger.info("  Total CUSIPs: %s", f"{total_cusips:,}")
    logger.info("  Flagged observations: %s (%.2f%%)", f"{flagged_rows:,}", 100 * flagged_rows / total_rows)
    logger.info("  Flagged CUSIPs: %s (%.2f%%)", f"{flagged_cusips:,}", 100 * flagged_cusips / total_cusips)

    # Build flag breakdown
    flag_breakdown = {}
    for col in ['flag_anomalous_price', 'flag_upward_spike', 'flag_plateau_sequence', 'flag_intraday_inconsistent']:
        if col in final_df.columns:
            flag_breakdown[col] = int(final_df[col].sum())

    # Configure plot style
    plot_params = dph.PlotParams(
        use_latex=use_latex,
        orientation="auto",
        x_spacing="rank",
        base_font=10,
        title_size=10,
        label_size=10,
        tick_size=9,
        legend_size=9,
        all_color="orange",
        all_alpha=0.75,
        all_lw=1.0,
        filtered_color="blue",
        filtered_lw=1.3,
        show_flagged=True,
        flagged_size=16,
        flagged_edgecolor="red",
        export_format="pdf",
        figure_dpi=150,
        transparent=False,
    )

    pages_made = []

    if output_figures and flagged_cusips > 0:
        logger.info("Generating figure pages for %d flagged CUSIPs...", flagged_cusips)

        # Prepare data subset for plotting (only flagged CUSIPs)
        df_plot = final_df[final_df['cusip_id'].isin(flagged_cusips_list)].copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_plot['trd_exctn_dt']):
            df_plot['trd_exctn_dt'] = pd.to_datetime(df_plot['trd_exctn_dt'], errors='coerce')

        # Sort for plotting
        df_plot = df_plot.sort_values(['cusip_id', 'trd_exctn_dt'])

        # Build index map for fast slicing
        idx_map = df_plot.groupby('cusip_id', sort=False).indices

        # Convert flagged_cusips_list to sorted list
        cusips_to_plot = sorted(flagged_cusips_list)

        # Batch into pages
        rows, cols = subplot_dim
        per_page = rows * cols

        n_pages = (len(cusips_to_plot) + per_page - 1) // per_page
        logger.info("Creating %d figure pages (%d CUSIPs per page)", n_pages, per_page)

        for page_idx in range(n_pages):
            start_idx = page_idx * per_page
            end_idx = min(start_idx + per_page, len(cusips_to_plot))
            page_cusips = cusips_to_plot[start_idx:end_idx]

            stub = f"distressed_fig_page_{page_idx + 1:03d}"
            fig_path = dph.make_distressed_panel(
                df_out=df_plot,
                error_cusips=page_cusips,
                subplot_dim=subplot_dim,
                export_dir=report_dir,
                filename_stub=stub,
                params=plot_params,
                idx_map=idx_map,
                flag_col=flag_col,
                price_col=price_col,
            )
            pages_made.append(fig_path.name)
            logger.info("  Saved page %03d: %s", page_idx + 1, fig_path.name)

            # Periodic garbage collection to manage memory with many figures
            if (page_idx + 1) % 10 == 0:
                gc.collect()

        # Clean up
        del df_plot
        gc.collect()

    elif not output_figures:
        logger.info("Figure generation disabled (output_figures=False)")
    else:
        logger.info("No flagged CUSIPs to plot")

    # Build LaTeX document
    logger.info("Building LaTeX document...")
    tex_path = dph.build_distressed_report_tex(
        out_dir=report_dir,
        total_rows=total_rows,
        total_cusips=total_cusips,
        flagged_rows=flagged_rows,
        flagged_cusips=flagged_cusips,
        flag_breakdown=flag_breakdown,
        pages_made=pages_made,
        author=AUTHOR if 'AUTHOR' in globals() else None,
    )

    logger.info("LaTeX report written to: %s", tex_path)
    print(f"\n[STEP 8B COMPLETE] Ultra distressed report generated")
    print(f"  Report directory: {report_dir}")
    print(f"  LaTeX file: {tex_path.name}")
    print(f"  Figure pages: {len(pages_made)}")

    gc.collect()
    mem_end = hf.log_memory_usage("step8b_build_distressed_report_end")
    hf.log_memory_delta(mem_start, mem_end, "step8b_build_distressed_report")


# ============================================================================
# STEP 9: FINAL FILTERS
# ============================================================================

def step9_final_filters(price_threshold=300, dip_threshold=35):
    """
    Apply final filters to flag high prices and price dips.
    
    Creates two dummy variables:
    - prc_high: 1 if price > price_threshold, 0 otherwise
    - prc_dip: 1 if first price change in 2002-07 exceeds dip_threshold, 0 otherwise
    
    Args:
        price_threshold: Price threshold for prc_high flag (default: 300)
        dip_threshold: Absolute price change threshold for prc_dip flag (default: 35)
    """
    mem_start = hf.log_memory_usage("step9_final_filters_start")
    global final_df
    
    logger.info("=" * 80)
    logger.info("STEP 9: Applying Final Filters")
    logger.info("=" * 80)
    
    # 1. prc_high: Flag if price > threshold
    final_df['prc_high'] = (final_df['pr'] > price_threshold).astype(int)
    prc_high_count = final_df['prc_high'].sum()
    logger.info(f"prc_high: {prc_high_count:,} rows flagged (pr > {price_threshold})")
    
    # 2. prc_dip: Flag first price change in 2002-07 if exceeds threshold
    final_df['prc_dip'] = 0  # Initialize all to 0
    
    # Filter to 2002-07
    mask_july = (
        (final_df['trd_exctn_dt'].dt.year == 2002) & 
        (final_df['trd_exctn_dt'].dt.month == 7)
    )
    
    if mask_july.sum() > 0:
        logger.info(f"Processing {mask_july.sum():,} rows in 2002-07")
        
        # Work with subset but keep index
        df_july = final_df[mask_july].copy()
        df_july = df_july.sort_values(['cusip_id', 'trd_exctn_dt'])
        
        # Calculate price change within each cusip
        df_july['price_change'] = df_july.groupby('cusip_id')['pr'].diff()
        
        # For each CUSIP, get the FIRST non-null price change
        # Mark which observation is the second one for each CUSIP (has first price change)
        df_july['is_first_change'] = df_july.groupby('cusip_id').cumcount() == 1
        
        # Get observations with first price change
        first_changes = df_july[df_july['is_first_change']].copy()
        
        # Flag if absolute price change > threshold
        flagged_idx = first_changes[first_changes['price_change'].abs() > dip_threshold].index
        
        # Set prc_dip = 1 for these observations in main dataframe
        final_df.loc[flagged_idx, 'prc_dip'] = 1
        
        prc_dip_count = final_df['prc_dip'].sum()
        logger.info(f"prc_dip: {prc_dip_count:,} rows flagged")
    else:
        logger.info("No data found for 2002-07, prc_dip set to 0 for all rows")
    
    print("\n[STEP 9 COMPLETE] Final filters applied")
    print(f"Shape: {final_df.shape}")
    print(f"prc_high flagged: {final_df['prc_high'].sum():,}")
    print(f"prc_dip flagged: {final_df['prc_dip'].sum():,}")
    gc.collect()
    mem_end = hf.log_memory_usage("step9_final_filters_end")
    hf.log_memory_delta(mem_start, mem_end, "step9_final_filters")
    return final_df

# ============================================================================
# STEP 10A: BUILD FILTER TABLES AND APPLY FILTERS
# ============================================================================

def step10a_build_filter_tables():
    """
    Build Tables 1 and 2, apply all data filters, and save outputs.
    
    This function:
    1. Creates Table 1: Daily Data Filter Configuration
    2. Creates Table 2: TRACE Daily Filter Records (with sequential filtering)
    3. Applies all filters to final_df (rating, maturity, distressed errors, etc.)
    4. Calls save_outputs() to save filtered data
    5. Sets global variables table1_tex and table2_tex for use in step10
    """
    mem_start = hf.log_memory_usage("step10a_build_filter_tables_start")
    global final_df, table1_tex, table2_tex
    
    logger.info("=" * 80)
    logger.info("STEP 10A: Building Filter Tables and Applying Filters")
    logger.info("=" * 80)
    
    # Extract date range from data for table captions
    min_date_dt = final_df['trd_exctn_dt'].min()
    max_date_dt = final_df['trd_exctn_dt'].max()
    min_date_str = min_date_dt.strftime('%Y-%m-%d')
    max_date_str = max_date_dt.strftime('%Y-%m-%d')
    logger.info("Data date range: %s to %s", min_date_str, max_date_str)
    
    # ========================================================================
    # TABLE 1: Daily Data Filter Configuration
    # ========================================================================
    logger.info("Building Table 1: Daily Data Filter Configuration")
    
    # Extract ULTRA_DISTRESSED_CONFIG without target_rows_per_chunk
    config_dict = {k: v for k, v in ULTRA_DISTRESSED_CONFIG.items() 
                   if k != 'target_rows_per_chunk'}
    
    table1_tex = hf.make_inputs_table(
        config_dict=config_dict,
        date_cut_off=DATE_CUT_OFF,
        final_filter_config=FINAL_FILTER_CONFIG,
        min_date=min_date_str
    )
    
    # ========================================================================
    # TABLE 2: TRACE Daily Filter Records
    # ========================================================================
    logger.info("Building Table 2: TRACE Daily Filter Records")
    
    # Track filtering steps
    filter_records = []
    
    # Start - use the count from step4 (before any filtering)
    if 'start' in FILTER_COUNTS:
        n_start = FILTER_COUNTS['start']
    else:
        # Fallback if step4 wasn't run
        n_start = len(final_df)
        logger.warning("Using current row count as start (step4 filter tracking not found)")
    
    filter_records.append(("start", n_start, n_start, 0, 0.0))
    logger.info("Start: %d rows", n_start)
    
    # Filter 1: Valid accrued vars (from step4)
    n_after_filter1 = n_start  # default if filter 1 wasn't tracked
    if 'valid_accrued_vars' in FILTER_COUNTS:
        n_before, n_after = FILTER_COUNTS['valid_accrued_vars']
        n_after_filter1 = n_after  # Store for next filter
        removed = n_before - n_after
        pct = 100.0 * removed / n_start if n_start > 0 else 0.0
        filter_records.append(("valid_accrued_vars", n_before, n_after, removed, pct))
        logger.info("Valid accrued vars: -%d rows (%.3f%%)", removed, pct)
    
    # Filter 2: Valid rating (spc_rating OR mdc_rating present)
    # FIX: Use n_after from Filter 1 instead of len(final_df)
    n_before = n_after_filter1
    has_rating = (final_df['spc_rating'].notna() | final_df['mdc_rating'].notna())
    final_df = final_df[has_rating].copy()
    n_after = len(final_df)
    removed = n_before - n_after
    pct = 100.0 * removed / n_start if n_start > 0 else 0.0
    filter_records.append(("valid_rating", n_before, n_after, removed, pct))
    logger.info("Valid rating: -%d rows (%.3f%%)", removed, pct)
    
    # Filter 3: Valid maturity (bond_maturity >= 1)
    n_before = len(final_df)
    final_df = final_df[final_df['bond_maturity'] >= 1.0].copy()
    n_after = len(final_df)
    removed = n_before - n_after
    pct = 100.0 * removed / n_start if n_start > 0 else 0.0
    filter_records.append(("valid_maturity", n_before, n_after, removed, pct))
    logger.info("Valid maturity: -%d rows (%.3f%%)", removed, pct)
    
    # Filter 4: Distressed errors (flag_refined_any == 1)
    if 'flag_refined_any' in final_df.columns:
        n_before = len(final_df)
        final_df = final_df[final_df['flag_refined_any'] != 1].copy()
        n_after = len(final_df)
        removed = n_before - n_after
        pct = 100.0 * removed / n_start if n_start > 0 else 0.0
        filter_records.append(("distressed_errors", n_before, n_after, removed, pct))
        logger.info("Distressed errors: -%d rows (%.3f%%)", removed, pct)
    
    # Filter 5: 2002-07 filter (prc_dip == 1)
    if 'prc_dip' in final_df.columns:
        n_before = len(final_df)
        final_df = final_df[final_df['prc_dip'] != 1].copy()
        n_after = len(final_df)
        removed = n_before - n_after
        pct = 100.0 * removed / n_start if n_start > 0 else 0.0
        filter_records.append(("2002_07_filter", n_before, n_after, removed, pct))
        logger.info("2002-07 filter: -%d rows (%.3f%%)", removed, pct)
    
    # Filter 6: High price (prc_high == 1)
    if 'prc_high' in final_df.columns:
        n_before = len(final_df)
        final_df = final_df[final_df['prc_high'] != 1].copy()
        n_after = len(final_df)
        removed = n_before - n_after
        pct = 100.0 * removed / n_start if n_start > 0 else 0.0
        filter_records.append(("high_prc", n_before, n_after, removed, pct))
        logger.info("High price: -%d rows (%.3f%%)", removed, pct)
    
    # Overall row
    n_final = len(final_df)
    total_removed = n_start - n_final
    total_pct = 100.0 * total_removed / n_start if n_start > 0 else 0.0
    filter_records.append(("overall", n_start, n_final, total_removed, total_pct))
    
    table2_tex = hf.make_filter_records_table(filter_records)
    
    logger.info("Filter summary: %d rows removed (%.2f%%)", total_removed, total_pct)
    logger.info("Final row count after filtering: %d", n_final)
    
    # ========================================================================
    # Winsorize within-date for outlier variables
    # ========================================================================
    logger.info("Applying within-date winsorization...")
    winsor_vars = ['ytm', 'credit_spread']
    
    for var in winsor_vars:
        if var in final_df.columns:
            def winsorize_group(group):
                lower = group.quantile(0.005)
                upper = group.quantile(0.995)
                return group.clip(lower=lower, upper=upper)
            
            final_df[var] = final_df.groupby('trd_exctn_dt')[var].transform(winsorize_group)
            logger.info("  Winsorized %s", var)
    
    # ========================================================================
    # Drop filter/flag columns
    # ========================================================================
    logger.info("Dropping filter and flag columns...")
    
    # Note: flag_anomalous_price, flag_upward_spike, flag_plateau_sequence, 
    # flag_intraday_inconsistent, anomaly_type, spike_type, plateau_id
    # are already dropped in ultra_distressed_filter() to save RAM
    cols_to_drop = ['flag_refined_any', 'prc_dip', 'prc_high']
    
    existing_cols = [c for c in cols_to_drop if c in final_df.columns]
    if existing_cols:
        final_df = final_df.drop(columns=existing_cols)
        logger.info("Dropped %d filter columns: %s", len(existing_cols), ", ".join(existing_cols))
    
    # ========================================================================
    # Standardize data types (Float64 -> float64)
    # ========================================================================
    logger.info("Standardizing float dtypes...")
    final_df = hf.standardize_float_dtypes(final_df, verbose=False)
    
    # Count Float64 columns for logging
    float64_cols = [col for col in final_df.columns if final_df[col].dtype.name == 'Float64']
    if float64_cols:
        logger.warning("Still have %d Float64 columns: %s", len(float64_cols), ", ".join(float64_cols))
    else:
        logger.info("All Float64 columns standardized to float64")
    
    # ========================================================================
    # Optimize data types for RAM efficiency (~30M rows)
    # ========================================================================
    logger.info("Optimizing data types for RAM efficiency...")
    
    # Define type conversions
    type_conversions = {
        # IDs - Int32 (handles NaN, range up to ~2.1B)
        'permno': 'Int32',
        'permco': 'Int32',
        'gvkey': 'Int32',
        
        # Counts - Int16 (handles NaN, range up to 32,767)
        'trade_count': 'Int16',
        'bid_count': 'Int16',
        'ask_count': 'Int16',
        
        # Ratings - Int8 (handles NaN, range 1-22)
        'sp_rating': 'Int8',
        'sp_naic': 'Int8',
        'mdy_rating': 'Int8',
        'spc_rating': 'Int8',
        'mdc_rating': 'Int8',
        'comp_rating': 'Int8',
        
        # Binary flags - Int8 (handles NaN if present)
        'callable': 'Int8',
        'db_type': 'Int8',
        
        # Large amounts - Int64 (handles NaN)
        'bond_amt_outstanding': 'Int64',
    }
    
    # Apply conversions
    n_optimized = 0
    for col, target_dtype in type_conversions.items():
        if col in final_df.columns:
            try:
                original_dtype = final_df[col].dtype
                final_df[col] = final_df[col].astype(target_dtype)
                n_optimized += 1
                logger.info("  %s: %s -> %s", col, original_dtype, target_dtype)
            except Exception as e:
                logger.warning("  Could not convert %s to %s: %s", col, target_dtype, e)
    
    logger.info("Optimized %d columns for RAM efficiency", n_optimized)
    
    # ========================================================================
    # Save outputs (filtered and cleaned data)
    # ========================================================================
    logger.info("Saving filtered and cleaned outputs...")
    save_outputs()
    
    print("\n[STEP 10A COMPLETE] Filter tables built and data filtered")
    print(f"Filtered data shape: {final_df.shape}")
    print(f"Total filtered: {total_removed:,} rows ({total_pct:.2f}%)")
    print("Tables 1 and 2 stored in global variables for step10")
    
    gc.collect()
    mem_end = hf.log_memory_usage("step10a_build_filter_tables_end")
    hf.log_memory_delta(mem_start, mem_end, "step10a_build_filter_tables")


# ============================================================================
# STEP 10: GENERATE REPORTS AND CLEAN DATA
# ============================================================================

def step10_generate_reports():
    """
    Generate comprehensive LaTeX data report with descriptive statistics and figures.
    
    This function:
    1. Uses Tables 1 and 2 from global variables (set by step10a)
    2. Creates Table 3-8: TRACE Daily Descriptive Statistics and Trade metrics
    3. Generates all figures (time-series, concentration, heatmaps)
    4. Drops all filter/flag columns
    5. Saves LaTeX report to STAGE1_DIR/data_reports/
    
    Note: step10a_build_filter_tables() must be called BEFORE this function.
    """
    mem_start = hf.log_memory_usage("step10_generate_reports_start")
    global final_df, table1_tex, table2_tex, fisd
    
    logger.info("=" * 80)
    logger.info("STEP 10: Generating Reports and Cleaning Data")
    logger.info("=" * 80)
    
    # Check that step10a was called
    if 'table1_tex' not in globals() or 'table2_tex' not in globals():
        raise RuntimeError("step10a_build_filter_tables() must be called before step10_generate_reports()")
    
    # Create reports directory
    reports_dir = STAGE1_DIR / "data_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Create time-series data subdirectory for CSV exports
    ts_data_dir = reports_dir / "time_series_data"
    ts_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract date range from data for table captions
    min_date_dt = final_df['trd_exctn_dt'].min()
    max_date_dt = final_df['trd_exctn_dt'].max()
    min_date_str = min_date_dt.strftime('%Y-%m-%d')
    max_date_str = max_date_dt.strftime('%Y-%m-%d')
    logger.info("Data date range: %s to %s", min_date_str, max_date_str)
    
    logger.info("Using Tables 1 and 2 from step10a")
    
    
    # ========================================================================
    # TABLE 3: Data Availability by Rating Category
    # ========================================================================
    logger.info("Building Table 3: Data Availability by Rating Category")
    
    table3_tex = hf.make_data_availability_table(
        df=final_df,
        min_date=min_date_str,
        max_date=max_date_str
    )
    # ========================================================================
    # TABLE 4: TRACE Daily Descriptive Statistics (with winsorization)
    # ========================================================================
    logger.info("Building Table 4: TRACE Daily Descriptive Statistics")
    
    # Note: Data has already been winsorized in step10a
    
    # Define variables for descriptive stats
    stat_vars = [
        ('pr', 'Price (VW)'),
        ('prc_ew', 'Price (EW)'),
        ('prc_vw_par', 'Price (ParW)'),
        ('prc_bid', 'Price (Bid)'),
        ('prc_ask', 'Price (Ask)'),
        ('prfull', 'Price (Full)'),
        ('ytm', 'YTM'),
        ('credit_spread', 'Spread'),
        ('mac_dur', 'Duration (Macaulay)'),
        ('mod_dur', 'Duration (Modified)'),
        ('bond_maturity', 'Bond Maturity'),
        ('bond_age', 'Bond Age'),
        ('convexity', 'Convexity'),
        ('dvolume', 'Volume (Dollar)'),
        ('qvolume', 'Volume (Par)'),
        ('bid_count', 'Bid Count'),
        ('ask_count', 'Ask Count'),
        ('sp_rating', 'Rating (SP)'),
        ('mdy_rating', 'Rating (MD)'),
    ]
    
    # Panel A: Pooled statistics
    logger.info("Computing Panel A: Pooled statistics")
    panel_a_stats = hf.compute_pooled_stats_fixed(final_df, stat_vars)
    
    # Panel B: Cross-sectional statistics (time-series averages of daily stats)
    logger.info("Computing Panel B: Cross-sectional statistics")
    panel_b_stats = hf.compute_cross_sectional_stats_fixed(final_df, stat_vars)
    
    table4_tex = hf.make_descriptive_stats_table_fixed(
        panel_a=panel_a_stats,
        panel_b=panel_b_stats,
        min_date=min_date_str,
        max_date=max_date_str
    )
    
    # ========================================================================
    # TABLE 5: Investment Grade Bonds Descriptive Statistics
    # ========================================================================
    logger.info("Building Table 5: Investment Grade Bonds Descriptive Statistics")
    
    # Filter for investment grade bonds (spc_rating 1-10 inclusive)
    ig_df = final_df[(final_df['spc_rating'] >= 1) & (final_df['spc_rating'] <= 10)].copy()
    logger.info("Investment Grade bonds: %d rows (%.2f%%)", len(ig_df), 100 * len(ig_df) / len(final_df))
    
    # Compute stats for investment grade
    panel_a_ig = hf.compute_pooled_stats_fixed(ig_df, stat_vars)
    panel_b_ig = hf.compute_cross_sectional_stats_fixed(ig_df, stat_vars)
    
    table5_tex = hf.make_descriptive_stats_table_by_rating(
        panel_a=panel_a_ig,
        panel_b=panel_b_ig,
        min_date=min_date_str,
        max_date=max_date_str,
        table_number=5,
        title="Investment Grade Corporate Bonds",
        rating_range_text="Ratings 1-10 (AAA to BBB-)"
    )
    
    # ========================================================================
    # TABLE 6: Non-Investment Grade Bonds Descriptive Statistics
    # ========================================================================
    logger.info("Building Table 6: Non-Investment Grade Bonds Descriptive Statistics")
    
    # Filter for non-investment grade bonds (spc_rating 11-21 inclusive)
    nig_df = final_df[(final_df['spc_rating'] > 10) & (final_df['spc_rating'] <= 21)].copy()
    logger.info("Non-Investment Grade bonds: %d rows (%.2f%%)", len(nig_df), 100 * len(nig_df) / len(final_df))
    
    # Compute stats for non-investment grade
    panel_a_nig = hf.compute_pooled_stats_fixed(nig_df, stat_vars)
    panel_b_nig = hf.compute_cross_sectional_stats_fixed(nig_df, stat_vars)
    
    table6_tex = hf.make_descriptive_stats_table_by_rating(
        panel_a=panel_a_nig,
        panel_b=panel_b_nig,
        min_date=min_date_str,
        max_date=max_date_str,
        table_number=6,
        title="Non-Investment Grade Corporate Bonds",
        rating_range_text="Ratings 11-21 (BB+ to CCC-)"
    )
    
    # ========================================================================
    # TABLE 7: Defaulted Bonds Descriptive Statistics
    # ========================================================================
    logger.info("Building Table 7: Defaulted Bonds Descriptive Statistics")
    
    # Filter for defaulted bonds (spc_rating == 22)
    def_df = final_df[final_df['spc_rating'] == 22].copy()
    logger.info("Defaulted bonds: %d rows (%.2f%%)", len(def_df), 100 * len(def_df) / len(final_df))
    
    # Compute stats for defaulted bonds
    panel_a_def = hf.compute_pooled_stats_fixed(def_df, stat_vars)
    panel_b_def = hf.compute_cross_sectional_stats_fixed(def_df, stat_vars)
    
    table7_tex = hf.make_descriptive_stats_table_by_rating(
        panel_a=panel_a_def,
        panel_b=panel_b_def,
        min_date=min_date_str,
        max_date=max_date_str,
        table_number=7,
        title="Defaulted Corporate Bonds",
        rating_range_text="Rating 22 (D - Default)"
    )

    # ========================================================================
    # Memory optimization: Drop columns no longer needed after Table 7
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Memory optimization: Dropping columns no longer needed...")
    logger.info("=" * 80)
    mem_before_drop = hf.log_memory_usage("step10_drop_columns_start")

    # Columns to drop after Table 7 (descriptive statistics complete)
    # Note: 'prfull', 'ytm', 'mod_dur', 'sp_rating', and 'convexity' are preserved
    # for Figures 1-4 and will be dropped after the GENERATE FIGURES section
    cols_to_drop_after_table7 = [
        'prc_ew', 'prc_vw_par', 'mac_dur', 'bond_age', 'mdy_rating'
    ]

    # Drop only columns that exist
    existing_cols_to_drop = [col for col in cols_to_drop_after_table7 if col in final_df.columns]

    if existing_cols_to_drop:
        logger.info("Dropping %d columns no longer needed:", len(existing_cols_to_drop))
        logger.info("  Columns: %s", ", ".join(existing_cols_to_drop))
        final_df.drop(columns=existing_cols_to_drop, inplace=True)
        gc.collect()
        logger.info("Columns dropped successfully")
    else:
        logger.info("No additional columns to drop")

    mem_after_drop = hf.log_memory_usage("step10_drop_columns_end")
    hf.log_memory_delta(mem_before_drop, mem_after_drop, "step10_drop_columns")

    # ========================================================================
    # TABLE 8: Trading Concentration Metrics
    # ========================================================================
    logger.info("Building Table 8: Trading Concentration Metrics")
    
    # Compute statistics for each rating category
    stats_conc_ig = hf.compute_concentration_stats(final_df, 'investment_grade')
    stats_conc_nig = hf.compute_concentration_stats(final_df, 'non_investment_grade')
    stats_conc_def = hf.compute_concentration_stats(final_df, 'defaulted')
    
    table8_tex = hf.make_concentration_table(
        stats_ig=stats_conc_ig,
        stats_nig=stats_conc_nig,
        stats_def=stats_conc_def,
        min_date=min_date_str,
        max_date=max_date_str
    )
    
    # ========================================================================
    # GENERATE FIGURES (4x2 time-series plots) - 4 separate PDFs by rating
    # ========================================================================
    logger.info("Generating time-series figures...")
    
    timestamp = datetime.now().strftime("%Y%m%d")
    plot_params = hf.PlotParams()
    
    # Define the 4 rating categories to plot
    rating_configs = [
        ('all', 'All Bonds', f"stage1_figures_{timestamp}"),
        ('investment_grade', 'Investment Grade', f"stage1_figures_{timestamp}_investment_grade"),
        ('non_investment_grade', 'Non-Investment Grade', f"stage1_figures_{timestamp}_non_investment_grade"),
        ('defaulted', 'Defaulted', f"stage1_figures_{timestamp}_defaulted"),
    ]
    
    fig_filenames = []  # List of (filename, caption) tuples for LaTeX
    
    for rating_filter, caption_title, filename in rating_configs:
        try:
            fig_path, df_weekly = hf.create_time_series_plots(
                df=final_df,
                output_dir=reports_dir,
                filename=filename,
                params=plot_params,
                rating_filter=rating_filter,
            )
            logger.info("Saved %s figures: %s", caption_title, fig_path)
            
            # Export weekly time-series data to CSV
            csv_filename = filename.replace('stage1_figures', 'timeseries_data')
            csv_path = ts_data_dir / f"{csv_filename}.csv"
            df_weekly.to_csv(csv_path, index=False)
            logger.info("Exported %s time-series CSV: %s", caption_title, csv_path)
            
            # Store just the basename for LaTeX (relative path) with detailed caption
            # Get min and max dates from data
            min_date_str = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
            max_date_str = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
            
            # Adjust caption title and (F) panel description based on rating category
            if caption_title == 'All Bonds':
                caption_title_text = 'all bonds'
                panel_f_text = 'S\\&P rating (numeric 1-22)'
            elif caption_title == 'Defaulted':
                caption_title_text = 'defaulted bonds'
                panel_f_text = 'market capitalization in billions'
            else:
                caption_title_text = f'{caption_title.lower()} bonds'
                panel_f_text = 'S\\&P rating (numeric 1-22)'
            
            detailed_caption = (
                f'Time-series plots for {caption_title_text}. This figure displays eight panels '
                f'showing weekly averages (Monday week-start) of key bond variables: '
                f'(A) clean price (value-weighted), (B) dirty price (value-weighted), '
                f'(C) yield to maturity (in percent), (D) credit spread (in percent), '
                f'(E) modified duration (in years), '
                f'(F) {panel_f_text}, '
                f'(G) dollar volume (in millions), and (H) convexity. '
                f'Each time series is computed as the weekly mean across all bond-day observations '
                f'in each rating category. '
                f'The sample spans the period {min_date_str} to {max_date_str}.'
            )
            fig_filenames.append((fig_path.name, detailed_caption))
            
        except Exception as e:
            logger.warning("Could not generate %s figures: %s", caption_title, e)

    # ========================================================================
    # Drop columns preserved for Figures 1-4
    # ========================================================================
    logger.info("Dropping columns preserved for Figures 1-4...")
    cols_to_drop_after_figures = ['prfull', 'ytm', 'mod_dur', 'sp_rating', 'convexity']
    existing_cols_to_drop_figs = [col for col in cols_to_drop_after_figures if col in final_df.columns]

    if existing_cols_to_drop_figs:
        logger.info("  Dropping %d columns: %s", len(existing_cols_to_drop_figs), ", ".join(existing_cols_to_drop_figs))
        final_df.drop(columns=existing_cols_to_drop_figs, inplace=True)
        gc.collect()
        logger.info("  Columns dropped successfully")

    # ========================================================================
    # FIGURE 5 (Dynamics of Default) REMOVED - causes memory issues in stage1
    # This figure will be generated in stage2 instead.
    # ========================================================================

    # ========================================================================
    # GENERATE FIGURE 6: Trade Sparsity - Count (3x1 plot)
    # ========================================================================
    logger.info("Generating trade sparsity count figure...")
    
    try:
        sparsity_count_path = hf.create_trade_sparsity_count_plot(
            df=final_df,
            output_dir=reports_dir,
            filename=f"stage1_trade_sparsity_count_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved trade sparsity count figure: %s", sparsity_count_path)
        
        # Add to fig_filenames with caption
        min_date_str = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((sparsity_count_path.name, 
            'This figure shows the average number of days per month that bonds have valid price '
            'observations, by rating category. '
            'For each month, we compute the average number of days per bond with a valid price '
            '(black line), bid price (red line), and ask price (blue line). '
            'The dashed horizontal line shows the number of business days in each month. '
            'Lower values indicate less frequent trading or price updates. '
            'The $y$-axis represents the count of days per month (1-23), averaged across all bonds '
            'in each rating category. '
            'A bond is considered ``alive\'\'\' from its first observed trade date to its last '
            'observed trade date in TRACE, for the purposes of computing average trades within a month. '
            f'The sample spans the period {min_date_str} to {max_date_str}.'))
        
    except Exception as e:
        logger.warning("Could not generate trade sparsity count figure: %s", e)
    
    # ========================================================================
    # GENERATE FIGURE 7: Trade Sparsity - Probability (3x1 plot)
    # ========================================================================
    logger.info("Generating trade sparsity probability figure...")
    
    try:
        sparsity_prob_path = hf.create_trade_sparsity_probability_plot(
            df=final_df,
            output_dir=reports_dir,
            filename=f"stage1_trade_sparsity_probability_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved trade sparsity probability figure: %s", sparsity_prob_path)
        
        # Add to fig_filenames with caption
        min_date_str = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((sparsity_prob_path.name, 
            'This figure shows the historical probability of observing a valid price quote on any '
            'given business day, by rating category. '
            'The probability is computed as the ratio of average days traded per month (from '
            'Figure 6) to the number of business days in that month, expressed as a percentage. '
            'For example, a value of 50 percent means that on average, bonds in that category have '
            'a valid price on half of all business days in the month. '
            'The black line shows prices, the red line shows bid prices, and the blue line shows '
            'ask prices. '
            'Higher values indicate more liquid bonds with more frequent price observations. '
            'A bond is considered ``alive\'\'\' from its first observed trade date to its last '
            'observed trade date in TRACE, for the purposes of computing average trades within a month. '
            f'The sample spans the period {min_date_str} to {max_date_str}.'))
        
    except Exception as e:
        logger.warning("Could not generate trade sparsity probability figure: %s", e)
    
    # ========================================================================
    # GENERATE FIGURE 8: Trade Frequency Histogram (3x1 plot)
    # ========================================================================
    # logger.info("Generating trade frequency histogram...")
    
    # try:
    #     freq_hist_path = hf.create_trade_frequency_histogram(
    #         df=final_df,
    #         output_dir=reports_dir,
    #         filename=f"stage1_trade_frequency_hist_{timestamp}",
    #         params=plot_params,
    #     )
    #     logger.info("Saved trade frequency histogram: %s", freq_hist_path)
        
    #     # Add to fig_filenames with caption
    #     min_date_str = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
    #     max_date_str = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
    #     fig_filenames.append((freq_hist_path.name, 
    #         'This figure shows the cross-sectional distribution of the number of days per month '
    #         'that each bond trades (has a valid price observation). '
    #         'Each panel represents a different rating category. '
    #         'The $y$-axis shows the percentage of bond-month observations falling into each bin. '
    #         'The red dashed line indicates the median number of trading days per month. '
    #         'A bond is considered ``alive\'\'\' from its first observed trade date to its last '
    #         'observed trade date in TRACE, for the purposes of computing average trades within a month. '
    #         f'The sample spans the period {min_date_str} to {max_date_str}.'))
        
    # except Exception as e:
    #     logger.warning("Could not generate trade frequency histogram: %s", e)
    
    # ========================================================================
    # GENERATE FIGURE 9: Trading Concentration Over Time (3x1 plot)
    # ========================================================================
    logger.info("Generating trading concentration figure...")
    
    try:
        conc_time_path = hf.create_concentration_over_time_plot(
            df=final_df,
            output_dir=reports_dir,
            filename=f"stage1_concentration_time_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved trading concentration figure: %s", conc_time_path)
        
        # Add to fig_filenames with caption
        min_date_str = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((conc_time_path.name, 
            'This figure shows the evolution of trading concentration over time by rating category. '
            'Each panel displays three lines representing the percentage of bonds needed to account '
            'for 50 percent, 75 percent, and 90 percent of total dollar volume in each month. '
            'Lower values indicate higher concentration. '
            'For example, if the ``50 percent Volume\'\'\' line shows a value of 5 percent, this means '
            'that just 5 percent of all bonds account for half of the total trading volume in that '
            'month, indicating very high concentration. '
            'Conversely, if the value is 40 percent, then 40 percent of bonds are needed to account '
            'for half the volume, indicating lower concentration and more dispersed trading activity. '
            'The measure is computed monthly using dollar volume per bond. '
            f'The sample spans the period {min_date_str} to {max_date_str}.'))
        
    except Exception as e:
        logger.warning("Could not generate trading concentration figure: %s", e)
    
    # ========================================================================
    # GENERATE FIGURE 10: Trading Intensity Heatmap
    # ========================================================================
    logger.info("Generating trading intensity heatmap...")
    
    try:
        heatmap_path = hf.create_trading_intensity_heatmap(
            df=final_df,
            output_dir=reports_dir,
            filename=f"stage1_trading_intensity_heatmap_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved trading intensity heatmap: %s", heatmap_path)
        
        # Add to fig_filenames with caption
        min_date_str_fig10 = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str_fig10 = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((heatmap_path.name, 
            'This figure displays a heatmap showing the intensity of trading activity across different '
            'rating categories and time periods. '
            'The $y$-axis shows five rating groups (from highest rated AAA-A minus at the top to defaulted '
            'D at the bottom), while the $x$-axis shows time in monthly intervals. '
            'The color intensity represents the average probability of observing a valid price on any '
            'business day within that month for bonds in that rating category, expressed as a percentage. '
            'Darker colors indicate higher trading intensity (bonds trade more frequently), while lighter '
            'colors indicate lower trading intensity or illiquid periods. '
            'This visualization allows for quick identification of patterns in trading activity across '
            'rating categories and time, such as periods of market stress or differential liquidity '
            'between rating grades. '
            'A bond is considered ``alive\'\'\'  from its first observed trade date to its last observed '
            'trade date in TRACE, for the purposes of computing average trades within a month. '
            f'The sample spans the period {min_date_str_fig10} to {max_date_str_fig10}.'))
        
    except Exception as e:
        logger.warning("Could not generate trading intensity heatmap: %s", e)
    
    # ========================================================================
    # GENERATE FIGURE 11: Industry Market Cap Evolution (FF17)
    # ========================================================================
    logger.info("Generating FF17 industry market cap evolution figure...")
    
    try:
        industry_ff17_path = hf.create_industry_marketcap_evolution_plot(
            df=final_df,
            output_dir=reports_dir,
            ff_column='ff17num',
            industry_mapping=FF17_MAPPING,
            filename=f"stage1_industry_marketcap_evolution_ff17_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved FF17 industry market cap evolution figure: %s", industry_ff17_path)
        
        # Add to fig_filenames with caption
        min_date_str_fig11 = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str_fig11 = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((industry_ff17_path.name, 
            'This figure shows the evolution of corporate bond market composition across 17 '
            'Fama-French industries over time. '
            'The $y$-axis represents the percentage share of total market capitalization (0-100 percent), '
            'computed as the bond clean price multiplied by the par amount outstanding. '
            'Each colored area corresponds to one industry sector, with the stacked areas summing to '
            '100 percent at each point in time. '
            'Market capitalization is aggregated weekly and expressed as a percentage of the total '
            'market to show relative industry weights. '
            'Industries are classified using Standard Industrial Classification (SIC) codes matched to '
            'the Fama-French 17 industry groupings. '
            f'The sample spans the period {min_date_str_fig11} to {max_date_str_fig11}.'))
        
    except Exception as e:
        logger.warning("Could not generate FF17 industry market cap evolution figure: %s", e)
    
    # ========================================================================
    # GENERATE FIGURE 12: Industry Market Cap Evolution (FF30)
    # ========================================================================
    logger.info("Generating FF30 industry market cap evolution figure...")
    
    try:
        industry_ff30_path = hf.create_industry_marketcap_evolution_plot(
            df=final_df,
            output_dir=reports_dir,
            ff_column='ff30num',
            industry_mapping=FF30_MAPPING,
            filename=f"stage1_industry_marketcap_evolution_ff30_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved FF30 industry market cap evolution figure: %s", industry_ff30_path)
        
        # Add to fig_filenames with caption
        min_date_str_fig12 = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str_fig12 = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((industry_ff30_path.name, 
            'This figure shows the evolution of corporate bond market composition across 30 '
            'Fama-French industries over time. '
            'The $y$-axis represents the percentage share of total market capitalization (0-100 percent), '
            'computed as the bond clean price multiplied by the par amount outstanding. '
            'Each colored area corresponds to one industry sector, with the stacked areas summing to '
            '100 percent at each point in time. '
            'Market capitalization is aggregated weekly and expressed as a percentage of the total '
            'market to show relative industry weights. '
            'Industries are classified using Standard Industrial Classification (SIC) codes matched to '
            'the Fama-French 30 industry groupings. '
            f'The sample spans the period {min_date_str_fig12} to {max_date_str_fig12}.'))
        
    except Exception as e:
        logger.warning("Could not generate FF30 industry market cap evolution figure: %s", e)
    
    # ========================================================================
    # GENERATE FIGURE 13: Industry Dollar Volume Evolution (FF17)
    # ========================================================================
    logger.info("Generating FF17 industry dollar volume evolution figure...")
    
    try:
        dvolume_ff17_path = hf.create_industry_dvolume_evolution_plot(
            df=final_df,
            output_dir=reports_dir,
            ff_column='ff17num',
            industry_mapping=FF17_MAPPING,
            filename=f"stage1_industry_dvolume_evolution_ff17_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved FF17 industry dollar volume evolution figure: %s", dvolume_ff17_path)
        
        # Add to fig_filenames with caption
        min_date_str_fig13 = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str_fig13 = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((dvolume_ff17_path.name, 
            'This figure shows the evolution of corporate bond trading activity composition across 17 '
            'Fama-French industries over time. '
            'The $y$-axis represents the percentage share of total dollar volume (0-100 percent), computed '
            'as the sum of trade sizes multiplied by trade prices within each week. '
            'Each colored area corresponds to one industry sector, with the stacked areas summing to '
            '100 percent at each point in time. '
            'Dollar volume is aggregated weekly and expressed as a percentage of the total trading volume '
            'to show relative industry trading intensity. '
            'Industries are classified using Standard Industrial Classification (SIC) codes matched to '
            'the Fama-French 17 industry groupings. '
            f'The sample spans the period {min_date_str_fig13} to {max_date_str_fig13}.'))
        
    except Exception as e:
        logger.warning("Could not generate FF17 industry dollar volume evolution figure: %s", e)
    
    # ========================================================================
    # GENERATE FIGURE 14: Industry Dollar Volume Evolution (FF30)
    # ========================================================================
    logger.info("Generating FF30 industry dollar volume evolution figure...")
    
    try:
        dvolume_ff30_path = hf.create_industry_dvolume_evolution_plot(
            df=final_df,
            output_dir=reports_dir,
            ff_column='ff30num',
            industry_mapping=FF30_MAPPING,
            filename=f"stage1_industry_dvolume_evolution_ff30_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved FF30 industry dollar volume evolution figure: %s", dvolume_ff30_path)
        
        # Add to fig_filenames with caption
        min_date_str_fig14 = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str_fig14 = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((dvolume_ff30_path.name, 
            'This figure shows the evolution of corporate bond trading activity composition across 30 '
            'Fama-French industries over time. '
            'The $y$-axis represents the percentage share of total dollar volume (0-100 percent), computed '
            'as the sum of trade sizes multiplied by trade prices within each week. '
            'Each colored area corresponds to one industry sector, with the stacked areas summing to '
            '100 percent at each point in time. '
            'Dollar volume is aggregated weekly and expressed as a percentage of the total trading volume '
            'to show relative industry trading intensity. '
            'Industries are classified using Standard Industrial Classification (SIC) codes matched to '
            'the Fama-French 30 industry groupings. '
            f'The sample spans the period {min_date_str_fig14} to {max_date_str_fig14}.'))
        
    except Exception as e:
        logger.warning("Could not generate FF30 industry dollar volume evolution figure: %s", e)
    
    # ========================================================================
    # GENERATE FIGURE 15: Trade Size Distribution Over Time (3x1 plot)
    # ========================================================================
    logger.info("Generating trade size distribution figure...")
    
    try:
        trade_size_path = hf.create_trade_size_distribution_plot(
            df=final_df,
            output_dir=reports_dir,
            filename=f"stage1_trade_size_distribution_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved trade size distribution figure: %s", trade_size_path)
        
        # Add to fig_filenames with caption
        min_date_str_fig15 = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str_fig15 = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((trade_size_path.name, 
            'This figure shows the distribution of daily trading volume over time. '
            'The data contain one observation per trading day, where dollar volume represents the sum of '
            'all trades on that day. '
            'Each panel displays 13 cumulative lines showing the percentage of days with total daily '
            'volume below specified thresholds, ranging from \$0-5k to \$20M+. '
            'The $y$-axis represents the cumulative percentage of days (0-100 percent). '
            'Data are aggregated monthly. '
            f'The sample spans the period {min_date_str_fig15} to {max_date_str_fig15}.'))
        
    except Exception as e:
        logger.warning("Could not generate trade size distribution figure: %s", e)
    
    
    # ========================================================================
    # GENERATE FIGURE 16: Bond Characteristics Evolution (3x1 plot)
    # ========================================================================
    logger.info("Generating bond characteristics evolution figure...")
    
    try:
        characteristics_path, df_characteristics = hf.create_bond_characteristics_evolution_plot(
            df=final_df,
            fisd=fisd,
            output_dir=reports_dir,
            filename=f"stage1_bond_characteristics_evolution_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved bond characteristics evolution figure: %s", characteristics_path)
        
        # Export characteristics data to CSV
        csv_path = ts_data_dir / f"timeseries_data_bond_characteristics_{timestamp}.csv"
        df_characteristics.to_csv(csv_path, index=False)
        logger.info("Exported bond characteristics time-series CSV: %s", csv_path)
        
        # Add to fig_filenames with caption
        min_date_str_fig16 = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str_fig16 = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((characteristics_path.name, 
            'This figure shows the evolution of bond market composition by three key characteristics '
            'over time. '
            'Panel A displays bond type composition, showing the percentage of trades by the top 5 bond '
            'types (by trade count) plus an ``Other\'\'\'  category. '
            'Panel B shows country domicile composition, displaying the percentage of trades by bonds '
            'domiciled in the top 5 countries plus an ``Other\'\'\'  category. '
            'Panel C shows Rule 144A composition, displaying the percentage of trades in 144A bonds '
            '(private placements exempt from SEC registration) versus non-144A bonds. '
            'All percentages are based on trade counts aggregated weekly (Monday week-start), with stacked '
            'areas summing to 100 percent at each point in time. '
            'The top 5 categories in Panels A and B are determined by the unconditional mean percentage '
            'across the entire sample period. '
            f'The sample spans the period {min_date_str_fig16} to {max_date_str_fig16}.'))
        
    except Exception as e:
        logger.warning("Could not generate bond characteristics evolution figure: %s", e)
    
    # ========================================================================
    # GENERATE FIGURE 17: Rating & Maturity Categories Evolution (2x1 plot)
    # ========================================================================
    logger.info("Generating rating and maturity categories evolution figure...")
    
    try:
        rating_maturity_path, df_rating_maturity = hf.create_rating_maturity_evolution_plot(
            df=final_df,
            output_dir=reports_dir,
            filename=f"stage1_rating_maturity_evolution_{timestamp}",
            params=plot_params,
        )
        logger.info("Saved rating and maturity categories evolution figure: %s", rating_maturity_path)
        
        # Export rating/maturity data to CSV
        csv_path = ts_data_dir / f"timeseries_data_rating_maturity_{timestamp}.csv"
        df_rating_maturity.to_csv(csv_path, index=False)
        logger.info("Exported rating and maturity time-series CSV: %s", csv_path)
        
        # Add to fig_filenames with caption
        min_date_str_fig17 = final_df['trd_exctn_dt'].min().strftime('%Y-%m-%d')
        max_date_str_fig17 = final_df['trd_exctn_dt'].max().strftime('%Y-%m-%d')
        
        fig_filenames.append((rating_maturity_path.name, 
            'This figure shows the evolution of corporate bond market composition by credit rating '
            'and maturity categories over time. '
            'Panel A displays the distribution of trades across six NAIC rating categories: AAA+ to A$-$ '
            '(investment grade high quality), BBB+ to BBB$-$ (investment grade lower quality), BB+ to BB$-$ '
            '(non-investment grade high speculative), B+ to B$-$ (non-investment grade speculative), '
            'CCC+ to C (substantial credit risk), and D (default). '
            'Panel B shows the distribution of trades across four maturity buckets: 1 to 3 Year '
            '(short-term), 3 to 5 Year (intermediate-term), 5 to 10 Year (medium-term), and 10 Year '
            'Plus (long-term). '
            'All percentages are based on trade counts aggregated weekly (Monday week-start), with '
            'stacked areas summing to 100 percent at each point in time. '
            'Categories are ordered by their unconditional mean percentage across the entire sample '
            'period (highest to lowest). '
            f'The sample spans the period {min_date_str_fig17} to {max_date_str_fig17}.'))
        
    except Exception as e:
        logger.warning("Could not generate rating and maturity categories evolution figure: %s", e)
    # ========================================================================
    # Generate LaTeX document
    # ========================================================================
    logger.info("Generating LaTeX report...")
    
    timestamp = datetime.now().strftime("%Y%m%d")
    tex_filename = f"stage1_data_report_{timestamp}.tex"
    tex_path = reports_dir / tex_filename
    
    # Build complete LaTeX document (with optional figures)
    # Pass list of (filename, caption) tuples if figures were generated
    fig_list = fig_filenames if fig_filenames else None
    tex_content = hf.build_latex_document(
        table1_tex, table2_tex, table3_tex, table4_tex, table5_tex, table6_tex, table7_tex,
        table8_tex, fig_filenames=fig_list, author=AUTHOR
    )
    
    # Write .tex file
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    logger.info("Saved LaTeX report: %s", tex_path)
    
    # Write references.bib
    bib_path = reports_dir / "references.bib"
    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write(hf.get_references_bib())
    logger.info("Saved bibliography: %s", bib_path)
    
    print("\n[STEP 10 COMPLETE] Reports generated and data cleaned")
    print(f"Final shape: {final_df.shape}")
    print(f"LaTeX report: {tex_path}")
    print("Generated 9 tables:")
    print("  - Table 1: Daily Data Filter Configuration")
    print("  - Table 2: TRACE Daily Filter Records")
    print("  - Table 3: Data Availability by Rating Category")
    print("  - Table 4: TRACE Daily Descriptive Statistics (All Bonds)")
    print("  - Table 5: TRACE Daily Descriptive Statistics (Investment Grade)")
    print("  - Table 6: TRACE Daily Descriptive Statistics (Non-Investment Grade)")
    print("  - Table 7: TRACE Daily Descriptive Statistics (Defaulted)")
    print("  - Table 8: Trade Frequency Distribution by Rating Category")
    print("  - Table 9: Trading Concentration Metrics by Rating Category")
    if fig_filenames:
        print(f"Generated {len(fig_filenames)} figure sets")
    gc.collect()
    mem_end = hf.log_memory_usage("step10_generate_reports_end")
    hf.log_memory_delta(mem_start, mem_end, "step10_generate_reports")
    return final_df


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs():
    """Save all output files.

    Note: sp_ratings, moodys_ratings, and call_dummy are exported early in
    variable_drop() to free memory before step8. This function checks if they
    are None and skips re-exporting if already saved.
    """
    global final_df, sp_ratings, moodys_ratings, call_dummy

    logger.info("=" * 80)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d")

    # Main output
    out_file = STAGE1_DATA / f"stage1_{timestamp}.parquet"
    final_df.to_parquet(out_file, index=False)
    logger.info("[OK] Main output saved: %s", out_file)

    # Ratings outputs - check if already exported in variable_drop()
    if sp_ratings is not None:
        out_file = STAGE1_DATA / f"sp_ratings_{timestamp}.parquet"
        sp_ratings.to_parquet(out_file, index=False)
        logger.info("[OK] S&P ratings saved: %s", out_file)
    else:
        logger.info("[SKIP] S&P ratings already exported in variable_drop()")

    if moodys_ratings is not None:
        out_file = STAGE1_DATA / f"moodys_ratings_{timestamp}.parquet"
        moodys_ratings.to_parquet(out_file, index=False)
        logger.info("[OK] Moody's ratings saved: %s", out_file)
    else:
        logger.info("[SKIP] Moody's ratings already exported in variable_drop()")

    # Call dummy - check if already exported in variable_drop()
    if call_dummy is not None:
        out_file = STAGE1_DATA / f"call_dummy_{timestamp}.parquet"
        call_dummy.to_parquet(out_file, index=False)
        logger.info("[OK] Call dummy saved: %s", out_file)
    else:
        logger.info("[SKIP] Call dummy already exported in variable_drop()")

    print("\n[OUTPUTS SAVED]")
    print(f"Main file: stage1_{timestamp}.parquet")
    print(f"Location: {STAGE1_DATA}")

    # ========================================================================
    # Memory optimization: Drop unnecessary columns and objects
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Optimizing memory: Dropping unnecessary columns and objects...")
    logger.info("=" * 80)
    mem_before = hf.log_memory_usage("save_outputs_memory_optimization_start")

    # Keep only essential columns for report generation
    essential_cols = [
        'cusip_id', 'trd_exctn_dt', 'pr', 'prc_bid', 'prc_ask',
        'prc_ew', 'prc_vw_par', 'prfull', 'ytm', 'mac_dur', 'mod_dur',
        'bond_maturity', 'bond_age', 'convexity', 'credit_spread',
        'sp_rating', 'spc_rating', 'mdy_rating', 'permno',
        'bond_amt_outstanding', 'ff17num', 'ff30num',
        'dvolume', 'qvolume', 'bid_count', 'ask_count'
    ]

    # Keep only columns that exist in final_df
    cols_to_keep = [col for col in essential_cols if col in final_df.columns]
    cols_to_drop = [col for col in final_df.columns if col not in cols_to_keep]

    if cols_to_drop:
        logger.info("Dropping %d unnecessary columns from final_df:", len(cols_to_drop))
        logger.info("  Columns: %s", ", ".join(cols_to_drop))
        final_df = final_df[cols_to_keep]
    else:
        logger.info("No columns to drop from final_df")

    # Delete auxiliary dataframes to free memory
    objects_to_delete = []

    # Delete sp_ratings if it exists and is not None
    try:
        if sp_ratings is not None:
            objects_to_delete.append('sp_ratings')
            del sp_ratings
    except NameError:
        pass

    # Delete moodys_ratings if it exists and is not None
    try:
        if moodys_ratings is not None:
            objects_to_delete.append('moodys_ratings')
            del moodys_ratings
    except NameError:
        pass

    # Delete call_dummy if it exists and is not None
    try:
        if call_dummy is not None:
            objects_to_delete.append('call_dummy')
            del call_dummy
    except NameError:
        pass

    if objects_to_delete:
        logger.info("Deleted auxiliary objects: %s", ", ".join(objects_to_delete))
    else:
        logger.info("No auxiliary objects to delete")

    # Force garbage collection
    gc.collect()

    mem_after = hf.log_memory_usage("save_outputs_memory_optimization_end")
    hf.log_memory_delta(mem_before, mem_after, "save_outputs_memory_optimization")

    logger.info("Memory optimization complete")
    logger.info("Final dataframe shape: %s", final_df.shape)
    logger.info("Final columns (%d): %s", len(final_df.columns), ", ".join(final_df.columns))


# ============================================================================
# CLEANUP
# ============================================================================

def cleanup():
    """Close database connection."""
    global db
    
    if db is not None:
        db.close()
        logger.info("WRDS connection closed")
        print("\n[CLEANUP] WRDS connection closed")


# ============================================================================
# RUN ALL STEPS
# ============================================================================

def run_all_steps():
    """Execute all steps in sequence."""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("RUNNING ALL STEPS")
    logger.info("=" * 80)
    
    try:
        step1_load_yields()
        step2_load_trace_data()
        step3_load_fisd_data()
        step4_merge_fisd()              # Creates trace_other and traced_out
        step5_compute_bond_analytics()  # Uses ylds for credit_spread calculation
        step6_merge_ratings()
        step7_merge_linker()
        variable_drop()                 # Memory optimization before step8
        step8_ultra_distressed()
        step8b_build_distressed_report()
        step9_final_filters(
            price_threshold=FINAL_FILTER_CONFIG['price_threshold'],
            dip_threshold=FINAL_FILTER_CONFIG['dip_threshold']
        )
        step10a_build_filter_tables()   # Build tables 1-2, apply filters, save outputs
        step10_generate_reports()       # Build tables 3-8, generate figures, save LaTeX
        cleanup()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("ALL STEPS COMPLETED SUCCESSFULLY")
        logger.info("Duration: %s", str(duration))
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("STAGE 1 COMPLETE")
        print(f"Duration: {duration}")
        print(f"Final data shape: {final_df.shape}")
        print("=" * 80)
        
    except Exception as e:
        logger.exception("ERROR in pipeline: %s", e)
        cleanup()
        raise
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import os
import sys

# Import shared configuration from root-level config.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WRDS_USERNAME, AUTHOR, OUTPUT_FORMAT

# --- FISD universe build params --------------------------------------
FISD_PARAMS = {
    # Switches for each screen
    "currency_usd_only": True,                 # foreign_currency == 'N'
    "fixed_rate_only": True,                   # coupon_type != 'V'
    "non_convertible_only": True,              # convertible == 'N'
    "non_asset_backed_only": True,             # asset_backed == 'N'
    "exclude_bond_types": True,                # drop certain bond_type codes
    "valid_coupon_frequency_only": True,       # drop invalid interest_frequency
    "require_accrual_fields": True,            # offering_date/dated_date etc. non-null
    "principal_amt_eq_1000_only": True,        # principal_amt == 1000
    "exclude_equity_index_linked": True,       # name contains 'EQUITY-LINKED' / 'INDEX-LINKED'
    "enforce_tenor_min": True,                 # tenor >= tenor_min_years

    # Knobs/sets
    "invalid_coupon_freq": [-1, 13, 14, 15, 16],
    "excluded_bond_types": [
        "TXMU","CCOV","CPAS","MBS","FGOV","USTC","USBD","USNT","USSP","USSI",
        "FGS","USBL","ABS","O30Y","O10Y","O5Y","O3Y","O4W","O13W","O26W","O52W",
        "CCUR","ADEB","AMTN","ASPZ","EMTN","ADNT","ARNT","TPCS","CPIK","PS","PSTK"
    ],
    "tenor_min_years": 1.0,
}


# --- Filter switchboard (True = apply, False = skip) -----------------
FILTER_SWITCHES =  dict(
    dick_nielsen            = True,  # 1: clean_trace_chunk()
    decimal_shift_corrector = True,  # 2: decimal_shift_corrector() [note:see ds_params]
    trading_time            = False, # 3: filter_by_trade_time()
    trading_calendar        = True,  # 4: filter_by_calendar()
    price_filters           = True,  # 5: > 0 (not neg) & <= 1000 price screens
    volume_filter_toggle    = True,  # 6: dollar_vol >= threshold   [note: renamed key to disambiguate]
    bounce_back_filter      = True,  # 7: flag_price_change_errors() [note:see bb_params]
    yld_price_filter        = True,  # 8: rptd_pr != yld_pt
    amtout_volume_filter    = True,  # 9: entrd_vol_qt < 0.5*offamt*1000
    trd_exe_mat_filter      = True,  # 10: trd_exctn_dt <= maturity
    flag_initial_price_errors = True, # 11: flag_initial_price_errors() [note:see init_error_params]
)

# --- Decimal-shift corrector params ---------------------------------
DS_PARAMS = {
    "factors": (0.1, 0.01, 10.0, 100.0),
    "tol_pct_good": 0.02,
    "tol_abs_good": 8.0,
    "tol_pct_bad": 0.05,
    "low_pr": 5.0,
    "high_pr": 300.0,
    "anchor": "rolling",
    "window": 5,
    "improvement_frac": 0.2,
    "par_snap": True,
    "par_band": 15.0,
    "output_type": "cleaned",
}

# --- Bounce-back (price-change) filter params -------------------------
BB_PARAMS = {
    "threshold_abs": 35.0,
    "lookahead": 5,
    "max_span": 5,
    "window": 5,
    "back_to_anchor_tol": 0.25,
    "candidate_slack_abs": 1.0,
    "reassignment_margin_abs": 5.0,
    "use_unique_trailing_median": True,
    "par_spike_heuristic": True,
    "par_level": 100.0,
    "par_equal_tol": 1e-8,
    "par_min_run": 3,
    "par_cooldown_after_flag": 2,
}

# --- Initial price error filter params --------------------------------
INIT_ERROR = {
    "abs_change": 50.0,
    "n_transactions": 3,
}

# --- Dev/test chunk limit ---------------------------------------------
# Process only the first N CUSIP chunks. None = the full universe (production).
# Overridable from the environment so a smoke run needs no edit here:
#     STAGE0_LIMIT_CHUNKS=5 ./run_smoke_test.sh
# A limited run logs a loud warning; its output is NOT the full universe.
LIMIT_CHUNKS = int(os.environ.get("STAGE0_LIMIT_CHUNKS", "0")) or None

# --- Chunk sizing ------------------------------------------------------
# Work units are packed to about this many TRADE ROWS rather than a fixed count of
# CUSIPs. Activity is enormously skewed, so 250-CUSIP chunks ranged from 7,806 rows
# to 3,392,802 over the Enhanced universe -- and the memory a job must reserve is set
# by the worst chunk, not the average.
#
# Measured on the real universe (111,727 CUSIPs / 345,874,974 trades):
#   fixed 250 CUSIPs   447 chunks, max 3,392,802 rows (~2.7 GB raw frame)
#   packed to 750,000  474 chunks, max   749,992 rows (~0.6 GB raw frame)
# Nearly the same number of chunks, but a 4.5x smaller worst case -- which is what
# makes running several at once affordable.
#
# This does not change the cleaned data: every per-chunk filter groups by cusip_id
# and chunks are disjoint CUSIP sets, so regrouping cannot move a row. It does change
# the audit tables, whose chunk column follows the new grouping.
# Set to None to restore fixed chunk_size chunks.
# Overridable from the environment, which is how the smoke test keeps its chunks
# small enough to run in minutes while still exercising the packing path:
#     STAGE0_TARGET_ROWS=40000 ./run_smoke_test.sh
TARGET_ROWS_PER_CHUNK = int(os.environ.get("STAGE0_TARGET_ROWS", "0")) or 750_000

# --- WRDS connection budget -------------------------------------------
# WRDS publishes a limit of 5 CONCURRENT JOBS but does NOT publish a per-user limit
# on database CONNECTIONS. There is one, and hitting it is nasty: the wrds package's
# connect-failure path calls input() to re-prompt for a username, so in a batch job
# the error arrives as "EOFError: EOF when reading a line" -- a rate limit disguised
# as a keyboard error.
#
# MEASURED 2026-09-09 with tests/probe_wrds_connections.py (account phd18ad1):
#   * 7 connections held simultaneously; the 8th failed with exactly that EOFError.
#   * Opening 6 AT ONCE, with no stagger and no lock, succeeded -- so the ceiling is
#     on connections HELD, not on how fast they are opened.
#   * A connect costs about 5 s, so a pool of N costs ~5N seconds to start.
# Re-measure on your own account before changing these:
#     python3 tests/probe_wrds_connections.py --max 10
MAX_WRDS_CONNECTIONS = 7

# How many connections each stage0 job may hold. Enhanced and 144A run at the same
# time, so their sum is what must fit; Standard is held behind them and runs alone,
# so it may use the whole budget. One connection is deliberately left spare beneath
# MAX_WRDS_CONNECTIONS so a mid-run reconnect cannot be refused.
CONCURRENCY = {
    "enhanced": 5,
    "144a": 1,
    "standard": 6,
}


def validate_connection_budget(members) -> None:
    """Fail at submit time, not four hours into a run, if the budget is over the cap.

    Only Enhanced and 144A overlap; Standard is scheduled after them.
    """
    concurrent = [m for m in members if m in ("enhanced", "144a")]
    total = sum(CONCURRENCY.get(m, 1) for m in concurrent)
    if total > MAX_WRDS_CONNECTIONS - 1:
        raise ValueError(
            f"WRDS connection budget exceeded: {concurrent} would hold {total} "
            f"connections, but the measured ceiling is {MAX_WRDS_CONNECTIONS} and one "
            "is reserved for reconnects. Lower CONCURRENCY in stage0/_trace_settings.py, "
            "or re-measure with tests/probe_wrds_connections.py if your account differs."
        )


# --- Grid resource requests -------------------------------------------
# A serial job could live on the WRDS batch default (2 cores / 16 GB). A pool of five
# worker processes cannot, so each stage0 job now asks for what it will actually use.
#
# ❗m_mem_free is charged PER SLOT, not per job. Ask for 4 slots at 24G and you have
# asked for 96 GB, the scheduler can never satisfy it, and the job PENDS FOREVER --
# silently, with no error anywhere. WRDS hard caps are 8 cores and 48 GB TOTAL per
# job, so what must hold is:  slots x mem_per_slot <= 48.
#
# These are passed on the qsub COMMAND LINE, not written into the job scripts, so the
# request follows CONCURRENCY automatically instead of drifting away from it. The
# scripts' own #$ directives are untouched.
#
# Slots match the worker count: the parent mostly waits while workers do the fetching
# and cleaning. Memory per slot is sized for the peak inside decimal_shift_corrector,
# which copies a chunk and adds columns -- roughly 2.5x a ~750k-row frame, so ~1.5 GB
# per worker -- plus headroom for the parent, which holds every chunk's daily frame
# until the end and is the real consumer.
MAX_SLOTS_PER_JOB = 8
MAX_MEM_GB_PER_JOB = 48

MEM_PER_SLOT_GB = {
    "enhanced": 8,    # 5 slots x 8G = 40 GB
    "standard": 8,    # 6 slots x 8G = 48 GB, exactly the cap
    "144a":     16,   # 1 slot -- same 16 GB it has always had
}


def qsub_resources(member: str) -> str:
    """qsub flags for one stage0 member, derived from its worker count.

    Returns e.g. "-pe onenode 5 -l m_mem_free=8G". Raises rather than emit a request
    the scheduler can never satisfy, because that failure mode is an invisible
    permanent pend.
    """
    slots = max(1, int(CONCURRENCY.get(member, 1)))
    mem = int(MEM_PER_SLOT_GB.get(member, 16))
    if slots > MAX_SLOTS_PER_JOB:
        raise ValueError(
            f"{member}: {slots} slots exceeds the WRDS limit of {MAX_SLOTS_PER_JOB} "
            "cores per job. Lower CONCURRENCY in stage0/_trace_settings.py.")
    if slots * mem > MAX_MEM_GB_PER_JOB:
        raise ValueError(
            f"{member}: {slots} slots x {mem}G = {slots * mem} GB exceeds the WRDS "
            f"limit of {MAX_MEM_GB_PER_JOB} GB per job (m_mem_free is charged PER "
            "SLOT). Lower MEM_PER_SLOT_GB or CONCURRENCY in "
            "stage0/_trace_settings.py.")
    return f"-pe onenode {slots} -l m_mem_free={mem}G"


# --- Price-scale normalization ----------------------------------------
# TRACE rptd_pr is a PERCENT OF PAR for the standard $1,000-principal bond: at par
# it prints 100. Small-denomination issues -- retail and structured notes with a
# principal of $10, $25 or $100 -- are quoted in UNIT dollars, so a $10 note at par
# prints 10.00. Every filter downstream assumes percent of par (price bounds, the
# decimal-shift gates, the bounce-back point threshold, stage 1's ultra-distressed
# thresholds, dollar volume, QuantLib), so left alone those bonds read as deeply
# distressed with tenfold-understated volume.
#
# When on, each non-$1,000 CUSIP is rescaled by 100/principal_amt if that puts its
# MEDIAN price nearer par -- decided once per bond, so distressed prints cannot flip
# the regime. $1,000-principal bonds are never touched.
#
# This is a NO-OP under the default screens: FISD_PARAMS["principal_amt_eq_1000_only"]
# is True, so no such bond is in the universe and every factor resolves to 1.0. It
# earns its keep the moment you turn that screen off -- which is exactly when the
# tape fills with unit-quoted notes.
PRICE_NORM = {
    "normalize_nonpar1000": True,
}

# --- Arguments identical across all runners ---------------------------
COMMON_KWARGS = dict(
    wrds_username = WRDS_USERNAME,
    output_format = OUTPUT_FORMAT,  # Imported from shared config.py
    chunk_size    = 250,
    target_rows_per_chunk = TARGET_ROWS_PER_CHUNK,
    limit_chunks  = LIMIT_CHUNKS,   # dev/test only: process just the first N CUSIP
                                    # chunks (None = the full universe). Lets a config
                                    # change be checked in minutes rather than a ~4h run.
    clean_agency  = True,
    out_dir       = "",
    volume_filter = ("dollar", 10000),
    trade_times   = ["00:00:00", "23:59:59"],  # Filter switched off as default
    calendar_name = "NYSE",
    ds_params     = DS_PARAMS,
    bb_params     = BB_PARAMS,
    init_error_params = INIT_ERROR,
    filters       = FILTER_SWITCHES,
    fisd_params   = FISD_PARAMS,
    price_norm    = PRICE_NORM
)


# --- Per-dataset overrides (only where needed) ------------------------
# Concurrency can be forced from the environment, which is how the smoke test proves
# the pool path without editing settings:  STAGE0_WORKERS=1 ./run_smoke_test.sh
WORKERS_OVERRIDE = int(os.environ.get("STAGE0_WORKERS", "0")) or None

PER_DATASET = {
    # Enhanced is the long pole -- ~4 hours, and its chunk loop was strictly serial on
    # one connection while the job held a whole node. CONCURRENCY holds the budget.
    "enhanced": dict(n_workers=WORKERS_OVERRIDE or CONCURRENCY["enhanced"]),
    # Standard runs AFTER the other two, so it may use the whole budget. 144A is
    # small (136 chunks vs Enhanced's 485) and runs alongside Enhanced, so it takes
    # one connection -- see CONCURRENCY and validate_connection_budget above.
    "standard": dict(start_date="2024-10-01", data_type="standard",
                     n_workers=WORKERS_OVERRIDE or CONCURRENCY["standard"]),
    "144a":     dict(start_date="2002-07-01", data_type="144a",
                     n_workers=WORKERS_OVERRIDE or CONCURRENCY["144a"]),
}

def get_config(kind: str) -> dict:
    """
    Returns a full kwargs dict for CreateDaily*TRACE calls.
    kind in {"enhanced", "standard", "144a"}.
    """
    overrides = PER_DATASET.get(kind, {})
    return {**COMMON_KWARGS, **overrides}

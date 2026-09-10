# -*- coding: utf-8 -*-
"""
Stage 1 Configuration Settings
================================
Central configuration for Stage 1 TRACE processing pipeline.
Edit this file to customize your processing parameters.

Author: Alex Dickerson
Created: 2025-11-17
"""

from __future__ import annotations
from pathlib import Path
import os
import sys

# Import shared configuration from root-level config.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WRDS_USERNAME, AUTHOR, OUTPUT_FORMAT, TRACE_MEMBERS

# ============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# --- Paths Configuration ---
# ROOT_PATH is the parent directory containing both stage0/ and stage1/ folders
#
# Option 1: AUTO-DETECT (recommended - leave blank or use "")
# The code will automatically detect ROOT_PATH from the current working directory.
# If you run the script from ~/proj/stage1, ROOT_PATH will be set to ~/proj
ROOT_PATH = ""  # Auto-detect from current working directory

# Option 2: MANUAL OVERRIDE (uncomment and edit if auto-detect doesn't work)
# Uncomment ONE of the following lines if you need to manually specify ROOT_PATH:
# ROOT_PATH = Path("~/proj").expanduser()                          # Linux/Mac/WRDS
# ROOT_PATH = Path("C:\\Users\\YourName\\Documents\\trace_data")   # Windows
# ROOT_PATH = Path("/Users/yourname/Documents/trace_data")         # Mac

# --- Stage0 Input Configuration ---
# STAGE0_DATE_STAMP will be auto-detected from stage0 parquet files (see DERIVED PATHS section)
# NOTE: TRACE_MEMBERS is imported from config.py (shared across all stages)

# --- Execution Settings ---
# Number of worker processes for the joblib passes (bond analytics, credit spreads).
# Leave as None to pick a safe value automatically; set an integer to force one.
N_CORES = None    # Set to None for auto-detection, or specify a number (e.g., 10)
N_CHUNKS = 10      # Number of chunks for parallel operations

# Default worker count when N_CORES is not set explicitly.
#
# This used to be multiprocessing.cpu_count(), which reports the cores of the whole
# HOST -- WRDS Cloud compute nodes have up to 24 -- so it spawned one worker per host
# core regardless of what the job was actually granted, and joblib's process backend
# copies the data each worker touches.
#
# The resolution order below reads $NSLOTS, so it now tracks the grant: since v2.2.1
# run_stage1.sh requests `-pe onenode 4` with `-l m_mem_free=10G`, and N_CORES resolves
# to 4 to match.
#
# ❗m_mem_free is charged PER SLOT, so the total is slots x mem and must stay within the
# WRDS caps of 8 cores and 48 GB per job. 4 x 10G = 40 GB. Raising one without dividing
# the other does not error -- the job pends forever, silently.
STAGE1_DEFAULT_N_CORES = 4

if N_CORES is None:
    import multiprocessing
    # Honour an explicit override, then a scheduler-granted slot count, then the
    # conservative default. Never more than the machine actually has.
    N_CORES = (
        int(os.environ.get("STAGE1_N_CORES", 0))
        or int(os.environ.get("NSLOTS", 0))
        or STAGE1_DEFAULT_N_CORES
    )
    N_CORES = max(1, min(N_CORES, multiprocessing.cpu_count()))

# --- Date Filter ---
# Only include data on or before this date. Two forms are accepted:
#
#   "YYYY-MM-DD"  a fixed date, used exactly as given.
#   "auto:-Nmo"   the last day of the month N months before the last trade date
#                 in the stage0 data. Resolved at run time, so the sample end
#                 tracks the data instead of needing an edit every vintage.
#
# The default keeps a three-month trailing buffer. TRACE is revised after the
# fact -- cancellations, corrections and reversals arrive for weeks -- so the
# most recent months are not yet settled and are held out of the sample.
#
# Note this buffer means Standard TRACE (db_type=2) never survives: step 2 keeps
# Standard only for dates AFTER the last Enhanced date, and any trailing cutoff
# falls before that, so the two conditions cannot both hold. That was already
# true of the previous fixed date; it is stated here so it is not a surprise.
DATE_CUT_OFF = "auto:-3mo"

# --- Output Settings ---
# OUTPUT_FORMAT is imported from shared config.py
# NOTE: Stage 1 always generates comprehensive reports and figures.
# These outputs are essential for data quality assessment and cannot be disabled.

# ============================================================================
# ULTRA DISTRESSED FILTER CONFIGURATION
# ============================================================================

ULTRA_DISTRESSED_CONFIG = {
    'price_col': 'pr',

    # Intraday inconsistency (Step 4)
    'intraday_range_threshold': 0.75,  # % Within Day Move for Flag
    'intraday_price_threshold': 20,    # Kicks in below this % of par

    # Anomaly detection
    'ultra_low_threshold': 0.10,
    'min_normal_price_ratio': 3.0,

    # Plateau detection
    'plateau_ultra_low_threshold': 0.15,
    'min_plateau_days': 2,

    # Round numbers
    'suspicious_round_numbers': [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00],

    # Intraday consistency (high spike detection)
    'price_cols': ['prc_hi', 'prc_lo'],
    'high_spike_threshold': 5.0,
    'min_spike_ratio': 3.0,
    'recovery_ratio': 2.0,
    'verbose': True,

    # Chunking settings
    'target_rows_per_chunk': 500000,  # Target ~500k rows per chunk
}

# ============================================================================
# FINAL FILTERS CONFIGURATION
# ============================================================================

FINAL_FILTER_CONFIG = {
    'price_threshold': 300,    # Filter out prices above this value (% of par)
    'dip_threshold': 35,       # 2002-07 price dip threshold (% of par)
}

# ============================================================================
# YIELD DATA CONFIGURATION
# ============================================================================

# Yield data source: 'LIU_WU' (recommended) or 'FRED'
YLD_TYPE = 'LIU_WU'

# Liu-Wu zero-coupon yield curve URL
LIU_WU_URL = "https://docs.google.com/spreadsheets/d/11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t/edit?usp=sharing"

# ============================================================================
# EXTERNAL DATA URLS
# ============================================================================

# Bond-firm linker: bond (cusip9) -> PERMNO / PERMCO / GVKEY, with a validity window.
#
# The mapping is DATED. A bond outlives its issuer's independence -- firms are
# acquired, spun off and renamed -- so each row carries [w0, w1], the window over
# which that firm owned that bond. Joining without the window attributes a bond to a
# firm in years it did not own it, and it does not throw.
#
# ZIPKEY is the path INSIDE the archive, which is nested under a folder; the member is
# read by exact name, so the folder prefix is required. run_pipeline.sh extracts to
# stage1/data/, giving stage1/data/bond_firm_linker_2026/fl_linker.parquet -- the same
# path the offline fallback in step 7 builds from LINKER_ZIPKEY.
#
# The archive also ships fl_verdicts.parquet (every refusal and its reason -- how you
# tell a refusal from a gap) and firm_names.parquet (permno -> dated firm name).
# Stage 1 reads only fl_linker.parquet.
LINKER_URL = "https://openbondassetpricing.com/wp-content/uploads/2026/09/bond_firm_linker_2026.zip"
LINKER_ZIPKEY = "bond_firm_linker_2026/fl_linker.parquet"

# ============================================================================
# PRE-DOWNLOADED EXTERNAL FILES (DO NOT EDIT)
# ============================================================================
# WRDS compute nodes have no internet, so run_pipeline.sh fetches these on the
# login node into stage1/data/ before any job is submitted. validate_config()
# checks for them up front: without that check a missing file surfaces deep
# inside step 6/7, after the ~30-minute QuantLib pass, instead of immediately.
#
# Keyed off LINKER_ZIPKEY so the entry follows automatically when the linker
# release changes.
EXTERNAL_FILES = {
    "Liu-Wu treasury yields": "liu_wu_yields.xlsx",
    "Fama-French 12 industries": "Siccodes12.txt",
    "Fama-French 17 industries": "Siccodes17.txt",
    "Fama-French 30 industries": "Siccodes30.txt",
    "bond-firm linker": LINKER_ZIPKEY,
}

# ============================================================================
# DERIVED PATHS (DO NOT EDIT)
# ============================================================================

# Auto-detect ROOT_PATH if not set
if not ROOT_PATH or ROOT_PATH == "":
    # Get current working directory (where the script is being run from)
    # If running from ~/proj/stage1, this will be ~/proj/stage1
    current_dir = Path.cwd()

    # If current directory is named 'stage1', use parent directory as ROOT_PATH
    if current_dir.name == "stage1":
        ROOT_PATH = current_dir.parent
    else:
        # Otherwise, assume current directory IS the root path
        ROOT_PATH = current_dir
else:
    # User specified ROOT_PATH, convert to Path object if needed
    ROOT_PATH = Path(ROOT_PATH)
    if str(ROOT_PATH).startswith("~"):
        ROOT_PATH = ROOT_PATH.expanduser()

STAGE0_DIR = ROOT_PATH / "stage0"
STAGE1_DIR = ROOT_PATH / "stage1"
STAGE1_DATA = STAGE1_DIR / "data"
LOG_DIR = STAGE1_DIR / "logs"

# ============================================================================
# AUTO-DETECT STAGE0 DATE STAMP
# ============================================================================

def get_latest_stage0_date(stage0_dir: Path, trace_member: str = "enhanced") -> str:
    """
    Auto-detect the most recent STAGE0_DATE_STAMP by finding the latest parquet file.

    Searches for main TRACE parquet files with pattern: trace_{member}_YYYYMMDD.parquet
    Excludes files with additional suffixes (e.g., trace_enhanced_fisd_YYYYMMDD.parquet)

    Parameters
    ----------
    stage0_dir : Path
        Path to stage0 directory
    trace_member : str, default="enhanced"
        TRACE dataset type: "enhanced", "standard", or "144a"

    Returns
    -------
    str
        Date stamp in YYYYMMDD format (e.g., "20251022")

    Raises
    ------
    FileNotFoundError
        If stage0 output directory or parquet files don't exist
    ValueError
        If no valid date stamps can be extracted from filenames

    Examples
    --------
    >>> get_latest_stage0_date(Path("/path/to/stage0"), "enhanced")
    '20251022'
    """
    member_dir = stage0_dir / trace_member

    if not member_dir.exists():
        raise FileNotFoundError(
            f"Stage0 output directory not found: {member_dir}\n"
            f"Ensure stage0 has been run successfully and outputs exist."
        )

    # Look for main data file: trace_{member}_YYYYMMDD.parquet
    pattern = f"trace_{trace_member}_*.parquet"
    parquet_files = list(member_dir.glob(pattern))

    if not parquet_files:
        raise FileNotFoundError(
            f"No TRACE parquet files found in {member_dir}\n"
            f"Expected pattern: {pattern}\n"
            f"Ensure stage0 has been run successfully."
        )

    # Filter for main files only (exclude files with additional suffixes like _fisd_)
    # Main file format: trace_enhanced_20251022.parquet (3 parts when split by '_')
    # FISD file format: trace_enhanced_fisd_20251022.parquet (4 parts when split by '_')
    main_files = []
    for f in parquet_files:
        stem = f.stem  # e.g., "trace_enhanced_20251022"
        parts = stem.split("_")

        # Should be exactly 3 parts: ["trace", "enhanced", "20251022"]
        if len(parts) == 3 and parts[-1].isdigit() and len(parts[-1]) == 8:
            main_files.append(f)

    if not main_files:
        raise ValueError(
            f"No main TRACE parquet files found in {member_dir}\n"
            f"Found {len(parquet_files)} parquet files, but none match expected format:\n"
            f"  Expected: trace_{trace_member}_YYYYMMDD.parquet\n"
            f"  Found files: {[f.name for f in parquet_files[:5]]}"
        )

    # Extract dates and return most recent
    dates = [f.stem.split("_")[-1] for f in main_files]
    latest_date = max(dates)

    return latest_date


# Auto-detect STAGE0_DATE_STAMP from most recent parquet file
try:
    STAGE0_DATE_STAMP = get_latest_stage0_date(STAGE0_DIR, trace_member="enhanced")
    print(f"[AUTO-DETECT] STAGE0_DATE_STAMP = {STAGE0_DATE_STAMP} (from stage0/{TRACE_MEMBERS[0]}/ parquet files)")
except (FileNotFoundError, ValueError) as e:
    # If auto-detect fails, use a fallback or raise error
    import warnings
    warnings.warn(
        f"Could not auto-detect STAGE0_DATE_STAMP: {e}\n"
        f"Falling back to manual setting. Please check stage0 outputs.",
        UserWarning
    )
    # Fallback to manual setting (user can override)
    STAGE0_DATE_STAMP = "20251022"  # Manual fallback

# ============================================================================
# CONFIGURATION GETTER FUNCTION
# ============================================================================

def resolve_date_cut_off(value, raw_max):
    """Resolve a DATE_CUT_OFF spec against the last trade date in the data.

    A fixed "YYYY-MM-DD" (or None) is returned unchanged. "auto:-Nmo" returns the
    last day of the month N months before `raw_max`, e.g. raw_max 2025-06-30 with
    N=3 gives 2025-03-31.

    Parameters
    ----------
    value : str or None
        Either a literal date, None, or an "auto:-Nmo" spec.
    raw_max : anything date-like
        The maximum trade date observed in the stage0 data.

    Returns
    -------
    str or None
        A "YYYY-MM-DD" date, or the input unchanged when it was not an auto spec.
    """
    import datetime as dt
    import re

    if value is None or not (isinstance(value, str) and value.startswith("auto")):
        return value

    m = re.fullmatch(r"auto:-(\d+)mo", value.strip())
    if not m:
        raise ValueError(
            f"Invalid DATE_CUT_OFF spec: {value!r}. "
            "Expected a 'YYYY-MM-DD' date or an 'auto:-Nmo' spec (e.g. 'auto:-3mo')."
        )
    n = int(m.group(1))

    last = dt.date.fromisoformat(str(raw_max)[:10])
    # Step back n months, then take the last day of that month.
    total = (last.year * 12 + last.month - 1) - n
    year, month = divmod(total, 12)
    month += 1
    first_of_next = (dt.date(year + 1, 1, 1) if month == 12
                     else dt.date(year, month + 1, 1))
    return (first_of_next - dt.timedelta(days=1)).isoformat()


def get_config() -> dict:
    """
    Returns a full configuration dictionary for Stage 1 processing.
    This centralizes all settings in one place for easy modification.
    """
    config = {
        # User settings
        "wrds_username": WRDS_USERNAME,
        "author": AUTHOR,

        # Paths
        "root_path": ROOT_PATH,
        "stage0_dir": STAGE0_DIR,
        "stage1_dir": STAGE1_DIR,
        "stage1_data": STAGE1_DATA,
        "log_dir": LOG_DIR,

        # Input configuration
        "stage0_date_stamp": STAGE0_DATE_STAMP,
        "trace_members": TRACE_MEMBERS,

        # Execution settings
        "n_cores": N_CORES,
        "n_chunks": N_CHUNKS,

        # Filters and date range
        "date_cut_off": DATE_CUT_OFF,
        "ultra_distressed_config": ULTRA_DISTRESSED_CONFIG,
        "final_filter_config": FINAL_FILTER_CONFIG,

        # Yield data
        "yld_type": YLD_TYPE,
        "liu_wu_url": LIU_WU_URL,

        # External data
        "linker_url": LINKER_URL,
        "linker_zipkey": LINKER_ZIPKEY,
        "external_files": EXTERNAL_FILES,

        # Output settings
        "output_format": OUTPUT_FORMAT,
        # NOTE: Stage 1 always generates reports and figures (no config flags needed)
    }

    return config


def validate_config(config: dict) -> None:
    """
    Validates configuration settings and raises errors if invalid.
    """
    # Check WRDS username
    if not config["wrds_username"]:
        raise ValueError(
            "WRDS_USERNAME not set. Please set the environment variable or "
            "edit config.py (in project root) to include your WRDS username."
        )

    # Check stage0 directory exists
    if not config["stage0_dir"].exists():
        raise FileNotFoundError(
            f"Stage0 directory not found: {config['stage0_dir']}\n"
            f"Please ensure ROOT_PATH is correctly set in _stage1_settings.py"
        )

    # Check stage0 data files exist
    missing_files = []
    for member in config["trace_members"]:
        member_folder = config["stage0_dir"] / member
        filename = f"trace_{member}_{config['stage0_date_stamp']}.parquet"
        filepath = member_folder / filename
        if not filepath.exists():
            missing_files.append(str(filepath))

    if missing_files:
        raise FileNotFoundError(
            "Stage0 output files not found:\n" + "\n".join(missing_files) + "\n"
            "Please ensure stage0 has been run with the correct date stamp."
        )

    # Check the externally-downloaded inputs BEFORE the pipeline starts. These are
    # fetched by run_pipeline.sh on the login node (WRDS compute nodes have no
    # internet). Steps 1, 3 and 7 need them; without this check the run fails hours
    # in -- after the QuantLib pass -- for want of a file we could catch in the
    # first second.
    missing_external = []
    for label, filename in config.get("external_files", {}).items():
        if not (config["stage1_data"] / filename).exists():
            missing_external.append(f"  {filename:<45s} ({label})")

    if missing_external:
        raise FileNotFoundError(
            "Required external data files not found in "
            f"{config['stage1_data']}:\n" + "\n".join(missing_external) + "\n\n"
            "These are downloaded on the WRDS login node (compute nodes have no\n"
            "internet access). From the project ROOT, run:\n"
            "  ./run_pipeline.sh          # downloads them, then submits the jobs\n"
            "or fetch them by hand -- see the PRE-STAGE block in run_pipeline.sh."
        )

    # Validate output format
    valid_formats = ["parquet", "csv"]
    if config["output_format"] not in valid_formats:
        raise ValueError(
            f"Invalid output_format: {config['output_format']}. "
            f"Must be one of {valid_formats}"
        )

    # Validate yield type
    valid_yld_types = ["LIU_WU", "FRED"]
    if config["yld_type"] not in valid_yld_types:
        raise ValueError(
            f"Invalid yld_type: {config['yld_type']}. "
            f"Must be one of {valid_yld_types}"
        )


def print_config_summary(config: dict) -> None:
    """
    Prints a summary of the current configuration.
    Useful for logging and debugging.
    """
    print("=" * 80)
    print("STAGE 1 CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"WRDS Username:      {config['wrds_username']}")
    print(f"Root Path:          {config['root_path']}")
    print(f"Stage0 Date Stamp:  {config['stage0_date_stamp']}")
    print(f"TRACE Members:      {', '.join(config['trace_members'])}")
    print(f"Date Cut-off:       {config['date_cut_off']}")
    print(f"Yield Source:       {config['yld_type']}")
    print(f"Output Format:      {config['output_format']}")
    print(f"CPU Cores:          {config['n_cores']}")
    print(f"External files:     {', '.join(config['external_files'].values())}")
    print("NOTE: Stage 1 always generates comprehensive reports and figures")
    print("=" * 80)

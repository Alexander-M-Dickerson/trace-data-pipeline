# -*- coding: utf-8 -*-
"""
Stage 2 Configuration Settings
==============================
Central configuration for Stage 2 monthly panel construction.
Edit this file to customize your processing parameters.

Stage 2 turns Stage 1's daily bond-day panel into the monthly asset-pricing panel
(returns, characteristics, signals, factors, rolling betas).

WHERE THIS RUNS
---------------
Stage 2 runs on YOUR OWN MACHINE, not on the WRDS cluster. Download the stage0/ and
stage1/ output folders from WRDS first, then run stage 2 against them locally. It is
CPU- and memory-hungry (a full build is roughly 7 minutes on 24 cores / 128 GB) and it
needs no TRACE database access -- only the parquet files Stage 1 produced.

Processing knobs below are carried over verbatim from the reference implementation;
each cites its source so a change is always a deliberate one.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations
from pathlib import Path
import os
import sys

# Import shared configuration from root-level config.py.
# Stage 2 can also be run from a folder that has no config.py (e.g. a stand-alone
# extract), so fall back to environment variables rather than failing at import.
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from config import WRDS_USERNAME, AUTHOR
except ImportError:  # pragma: no cover - only hit outside the pipeline tree
    WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "")
    AUTHOR = os.environ.get("AUTHOR", "Open Source Bond Asset Pricing")

# ============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# --- Paths Configuration ---
# ROOT_PATH is the parent directory containing stage0/, stage1/ and stage2/.
#
# Option 1: AUTO-DETECT (recommended - leave blank or use "")
# If you run from ~/proj/stage2, ROOT_PATH becomes ~/proj.
ROOT_PATH = ""  # Auto-detect from current working directory

# Option 2: MANUAL OVERRIDE (uncomment and edit if auto-detect doesn't work)
# ROOT_PATH = Path("~/proj").expanduser()                          # Linux/Mac
# ROOT_PATH = Path("C:\\Users\\YourName\\Documents\\trace_data")   # Windows

# --- Release vintage ---
# Blank = derive from the Stage 1 stamp's year (20260909 -> "2026"), which is what makes
# the naming extensible: next year's run publishes itself. Set a value only to override.
RELEASE_VINTAGE = ""

# --- Stage 1 input ---
# Leave as None to use the newest stage1/data/stage1_YYYYMMDD.parquet.
# Set a path (or the STAGE2_DAILY_INPUT environment variable) to pin one explicitly.
DAILY_INPUT = None

# --- Factor panel source ---
# "public"  = assemble the monthly factor matrix from public sources (Ken French,
#             FRED, He-Kelly-Manela, Ludvigson, Policy Uncertainty) plus the published
#             extended BBW series. Self-contained; this is the default.
# "pinned"  = read a pre-built factors.parquet (set FACTORS_PINNED_FILE). Use this only
#             to reproduce a specific published vintage exactly -- data vendors revise
#             history (FRED re-seasonally-adjusts CPI, EPU back-renormalizes), so the
#             two sources agree closely but do not match bit-for-bit.
FACTOR_SOURCE = "public"
FACTORS_PINNED_FILE = None

# The factor panel each published vintage was built from, hosted so a published number can
# be reproduced. Public sources revise -- Ken French restates SMB/HML, FRED
# re-seasonally-adjusts CPI, Ludvigson re-estimates its history -- so a fresh build will
# NOT reproduce an older release. Set FACTOR_SOURCE = "pinned" to fetch and use one of
# these instead; it is cached under stage2/data/ like any other download.
FACTORS_PINNED_URL = {
    "2026": "https://openbondassetpricing.com/wp-content/uploads/2026/09/osbap_stage2_factors_2026.zip",
}
FACTORS_PINNED_ZIPKEY = "factors_{vintage}.parquet"

# --- Execution settings ---
# Worker processes and DuckDB threads. workers * threads should not exceed your cores.
WORKERS = 6
THREADS_PER = 2
DUCKDB_THREADS = int(os.environ.get("STAGE2_DUCKDB_THREADS", "0")) or None
DUCKDB_MEMORY_LIMIT = os.environ.get("STAGE2_DUCKDB_MEMORY_LIMIT", "")

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================
# Verbatim from the reference implementation's "Processing Parameters" block.
# Changing any of these changes the published panel -- do so deliberately.

IMP_GAP = 1                # business days between signal observation and portfolio formation
BUSINESS_DAY_GAP = 5       # max NYSE-session gap for contiguous returns
SIGNAL_LAG = 1             # min day gap between signal observation and month-end (adj signals)
ADJ_WINDOW = 10            # max days back within the month for the adjusted month-end signal
CALENDAR_NAME = "NYSE"     # market calendar for business-day math
DEFAULT_METHOD = "event_based"   # default-return adjustment method
SWAP_ADJ_SIGNALS = True    # replace month-end signals with adjusted versions in the final panel
START_DATE = "2002-07-31"  # first month-end of the panel
SIGNALS = ("ytm", "mod_dur", "convexity", "credit_spread")   # step-1 signals (lagged by IMP_GAP)

# step 2 illiquidity
ILLIQ_MIN_OBS = 5          # min valid obs per bond-month for estimated measures

# step 3 BBW factor sorts
N_PORTF_1 = 5              # single-sort quintiles
N_PORTF_2 = 5              # double-sort 5x5
INCLUDE_ICE = True         # published extended-BBW backfill for factor columns before 2002-08-31

# step 4 rolling betas / systematic momentum
BETA_WINDOW = 36           # rolling window (months)
BETA_MIN_OBS = 12          # min obs within window

# step 4 DEF / TERM factors (see stage2/DATA_DICTIONARY.md, "Model Specifications")
#
# The default premium is the difference between the total returns on long-term corporate
# bonds and long-term government bonds (Fama-French 1993; Gebhardt, Hvidkjaer &
# Swaminathan 2005):
#     defb  = VW return of bonds with tmat >= DEF_CORP_MIN_MATURITY  -  DEF_GOVT_TENOR treasury return
#     termb = DEF_GOVT_TENOR treasury return  -  risk-free rate
DEF_CORP_MIN_MATURITY = 10.0   # years; the long-term corporate leg
DEF_GOVT_TENOR = 20.0          # years; the long-term government leg (CRSP key-rate tenor)

# step 5 value / d-spreads
DSPREAD_LAGS = (6, 12)
DSPREAD_BANDWIDTH = 1      # search +/- 1 month around the target lag month if missing
DSPREAD_MU_WINDOW = 12

# --- NYSE session calendar ---
# Holidays are known in advance, so any end date at or beyond your data max behaves
# identically. Fixed rather than date.today() so builds are deterministic.
CAL_START = "2001-07-01"
CAL_END = "2030-12-31"

# --- Treasury series vintage cap ---
# None = use every month available. Set a date only to reproduce an older published
# vintage. A cap here silently truncates tret -> ret_vwx -> the BBW duration-adjusted
# factors -> every rolling beta: 42 panel columns. Leave it None unless reproducing.
TRET_MAX_DATE = None

# --- Validation tolerances ---
FLOAT_TOL = 1e-6           # prices / returns / most signals
RATE_TOL = 1e-4            # rate-like winsorized float32 cols (ytm, cs, betas)

# --- Parquet compression for the large writes ---
PANEL_ZSTD_LEVEL = 3       # level 3 writes ~5x faster than 9 for ~7% larger files, identical values

# ============================================================================
# EXTERNAL DATA SOURCES
# ============================================================================
# Published inputs, downloaded once and cached under stage2/data/.

# Pre-TRACE quote returns (1997-01 -> 2002-06). Lets rolling signals reach a common
# 2002-08 start.
#
# 2026-09: extended from 9 to 14 columns, adding the five Treasury benchmarks
# (tret_bns / tret_cfm / tret_gprs / tret_cls / tret_mat) so the rolling windows behind the
# duration-adjusted blocks have the same pre-history for those benchmarks that they already had
# for `tret`. The original nine columns are byte-identical -- the file was extended by a join
# against the Lehman-ICE panel, not rebuilt. The zip member keeps its old name, so QUOTE_ZIPKEY
# is unchanged and nothing that reads this file by name needs to move.
QUOTE_URL = "https://openbondassetpricing.com/wp-content/uploads/2026/09/quote_returns_quantlib_tret.zip"
QUOTE_ZIPKEY = "quote_returns_quantlib.parquet"

# The columns this file is REQUIRED to carry. Steps 3/4/5 each select an explicit subset of these
# by name, so a wider file cannot change any computation -- but a NARROWER one silently produces
# short rolling windows, which is the failure this pins.
QUOTE_REQUIRED_COLS = ("cusip_id", "date", "ret_vw", "tret", "cs", "bbtm", "sze")

# The benchmark block the 2026 file adds. QUOTE_HAS_BENCHMARKS declares what QUOTE_URL points at;
# lib/quote.py refuses a cached copy that disagrees with it rather than using the stale one. Set
# this False alongside reverting QUOTE_URL if you ever need the nine-column file back.
QUOTE_BENCHMARK_COLS = ("tret_bns", "tret_cfm", "tret_gprs", "tret_cls", "tret_mat")
QUOTE_HAS_BENCHMARKS = True

# Extended "modified" BBW factor series, used ONLY to backfill factor history before
# 2002-08-31 (rows from 2002-08 on are recomputed from TRACE and overwritten).
# Estimated on the Lehman Brothers (Warga) Fixed Income Data and the BAML investment-grade
# and high-yield constituent bonds distributed by ICE. Those bond data are licensed and
# cannot be redistributed; the finished factor series can be, and is. Do not replace this
# with a longer file: the build asserts every date in it exists in the factor panel.
# 2026-09: extended from 9 to 17 columns, adding the bond-market factor twins for the alternative
# Treasury benchmarks (MKTB/DRF/CRF/TERM for each of bns and cls). build_factor_matrix splices this
# series in before 2002-08-31, so without the twins a rolling beta on ret_vw - tret_bns has no
# factor history to roll over and starts in 2003-08 where the tret one reaches 1997-01. The
# original nine columns are unchanged, and the zip member keeps its name.
BBW_EXTENDED_URL = "https://openbondassetpricing.com/wp-content/uploads/2026/09/bbw_factors_extended_1973_2023_tret.zip"
BBW_EXTENDED_ZIPKEY = "bbw_factors_extended_1973_2023.parquet"

# The twins the 2026-09 file adds, and the flag declaring that QUOTE_URL's sibling points at it.
# Same contract as QUOTE_BENCHMARK_COLS / QUOTE_HAS_BENCHMARKS: lib/extended_factors.py refuses a
# cached copy that disagrees rather than quietly splicing a series with no benchmark history.
BBW_BENCHMARK_COLS = ("MKTB_bns", "DRF_bns", "CRF_bns", "TERM_bns",
                      "MKTB_cls", "DRF_cls", "CRF_cls", "TERM_cls")
BBW_HAS_BENCHMARKS = True

# Public factor sources (used when FACTOR_SOURCE == "public").
FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_CSV.zip")
# He-Kelly-Manela publish under a dated filename and change it each release. The fetcher
# uses this URL while it is live and otherwise discovers the current one from HKM_INDEX.
# Note: as of 2026-09-10 the authors had not published past 2025-05, so `b_cptlt`
# legitimately stops there on a later panel -- that is upstream, not a broken link.
HKM_URL = "https://zhiguohe.net/wp-content/uploads/2025/07/He_Kelly_Manela_Factors_monthly_250627.csv"
HKM_INDEX = ("https://zhiguohe.net/data-and-empirical-patterns/"
             "intermediary-capital-ratio-and-risk-factor/")
# Ludvigson rotates the zip filename on every update, so the fetcher tries this URL and
# then discovers the current link from the index page.
# Current vintage as of 2026-09-10 (data through 2026-06); the previous 202508 link 404s.
LUDVIGSON_URL = "https://www.sydneyludvigson.com/s/MacroFinanceUncertainty_202608Update-3.zip"
LUDVIGSON_INDEX = "https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes"
EPU_URL = "https://www.policyuncertainty.com/media/Categorical_EPU_Data.xlsx"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={ids}"
FRED_TSY_SERIES = ("DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30")

FACTORS_START_DATE = "1973-01-31"   # first factor month kept

# WRDS tables fetched once and cached (treasury returns, FF5, VIX). These need your
# WRDS credentials, but only on the first run.
WRDS_TABLES = ("crsp.tfz_idx", "crsp.tfz_mth_ft", "ff.fivefactors_monthly", "cboe.cboe")

# ============================================================================
# DERIVED PATHS (DO NOT EDIT)
# ============================================================================

if os.environ.get("STAGE2_ROOT"):
    ROOT_PATH = Path(os.environ["STAGE2_ROOT"])
elif not ROOT_PATH or ROOT_PATH == "":
    current_dir = Path.cwd()
    ROOT_PATH = current_dir.parent if current_dir.name == "stage2" else current_dir
else:
    ROOT_PATH = Path(ROOT_PATH)
    if str(ROOT_PATH).startswith("~"):
        ROOT_PATH = ROOT_PATH.expanduser()

STAGE0_DIR = ROOT_PATH / "stage0"
STAGE1_DIR = ROOT_PATH / "stage1"
STAGE1_DATA = STAGE1_DIR / "data"
STAGE2_DIR = ROOT_PATH / "stage2"
STAGE2_DATA = STAGE2_DIR / "data"          # cached external inputs (gitignored)
OUTPUT_DIR = STAGE2_DIR / "output"         # build artifacts (gitignored)
CACHE_DIR = OUTPUT_DIR / "_cache"          # fingerprinted layer cache
BLOCKS_DIR = OUTPUT_DIR / "blocks"         # per-step intermediate blocks
PANEL_DIR = OUTPUT_DIR / "panel"           # final panels
REPORT_DIR = STAGE2_DIR / "data_reports"   # LaTeX report + figures
LOG_DIR = STAGE2_DIR / "logs"
MANIFEST_DIR = STAGE2_DIR / "manifests"    # committed JSON run manifests
FACTOR_CACHE_DIR = STAGE2_DATA / "factor_cache"

# ============================================================================
# INPUT RESOLUTION
# ============================================================================


def _latest_stamped(directory: Path, prefix: str, suffix: str = ".parquet") -> Path | None:
    """Newest `<prefix>YYYYMMDD<suffix>` in `directory`, or None."""
    if not directory.exists():
        return None
    best = None
    for p in directory.glob(f"{prefix}*{suffix}"):
        stem = p.name[len(prefix):-len(suffix)] if suffix else p.name[len(prefix):]
        if len(stem) == 8 and stem.isdigit() and (best is None or stem > best[0]):
            best = (stem, p)
    return best[1] if best else None


def daily_input(mode: str | None = None) -> Path:
    """The Stage 1 daily panel Stage 2 reads.

    `mode` is accepted and ignored: the reference implementation carried several input
    modes, the public pipeline has one.

    Precedence: STAGE2_DAILY_INPUT env var > DAILY_INPUT setting > newest
    stage1/data/stage1_YYYYMMDD.parquet.
    """
    env = os.environ.get("STAGE2_DAILY_INPUT")
    if env:
        return Path(env)
    if DAILY_INPUT:
        return Path(DAILY_INPUT)
    found = _latest_stamped(STAGE1_DATA, "stage1_")
    if found is None:
        # Return the expected path so validate_config can report it clearly.
        return STAGE1_DATA / "stage1_YYYYMMDD.parquet"
    return found


def date_stamp() -> str:
    """The YYYYMMDD stamp Stage 2 carries on its outputs, taken from the input filename.

    Stage 2 outputs inherit the Stage 1 stamp so a panel is always traceable to the
    daily file it was built from.
    """
    name = daily_input().stem
    tail = name.split("_")[-1]
    return tail if len(tail) == 8 and tail.isdigit() else "unstamped"


def release_vintage() -> str:
    """The four-digit vintage a RELEASE is published under, e.g. "2026".

    Build artifacts carry the Stage 1 date stamp (20260909) so a panel is always traceable
    to the file it came from. Released artifacts carry the vintage YEAR instead, which is
    what users cite -- `main_panel_2026.parquet`. Stamped while building, vintaged when
    released.

    Derived from the stamp's year rather than hard-coded, so next year's run publishes
    itself. Override with RELEASE_VINTAGE in this file, or the STAGE2_RELEASE_VINTAGE
    environment variable, if a release ever needs to differ from its build year.
    """
    override = os.environ.get("STAGE2_RELEASE_VINTAGE") or RELEASE_VINTAGE
    if override:
        return str(override)
    stamp = date_stamp()
    if len(stamp) == 8 and stamp.isdigit():
        return stamp[:4]
    from datetime import date
    return str(date.today().year)


def released_name(kind: str, suffix: str = ".parquet") -> str:
    """The published filename for an artifact, e.g. released_name("main_panel") ->
    "main_panel_2026.parquet"."""
    return f"{kind}_{release_vintage()}{suffix}"


def fisd_file() -> Path | None:
    """Stage 0's FISD characteristics file (144a flag, domicile, SIC, issue_id).

    Honours STAGE2_FISD_FILE. It must: AUX below reads the same override, and when the two
    disagreed the validator reported a file as missing that the build then used happily --
    so `_run_stage2.py` refused to start on inputs `build_panel.py` accepted.
    """
    env = os.environ.get("STAGE2_FISD_FILE")
    if env:
        return Path(env)
    return _latest_stamped(STAGE0_DIR / "enhanced", "trace_enhanced_fisd_")


def call_dummy_file() -> Path | None:
    """Stage 1's callable-flag file. Honours STAGE2_CALL_FILE (see `fisd_file`)."""
    env = os.environ.get("STAGE2_CALL_FILE")
    if env:
        return Path(env)
    return _latest_stamped(STAGE1_DATA, "call_dummy_")


def ensure_dirs() -> None:
    """Create the gitignored output tree (idempotent)."""
    for d in (STAGE2_DATA, OUTPUT_DIR, CACHE_DIR, BLOCKS_DIR, PANEL_DIR,
              REPORT_DIR, LOG_DIR, MANIFEST_DIR, FACTOR_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION ASSEMBLY
# ============================================================================


def get_config() -> dict:
    """Return the full Stage 2 configuration dictionary."""
    return {
        # User settings
        "wrds_username": WRDS_USERNAME,
        "author": AUTHOR,

        # Paths
        "root_path": ROOT_PATH,
        "stage0_dir": STAGE0_DIR,
        "stage1_dir": STAGE1_DIR,
        "stage1_data": STAGE1_DATA,
        "stage2_dir": STAGE2_DIR,
        "stage2_data": STAGE2_DATA,
        "output_dir": OUTPUT_DIR,
        "cache_dir": CACHE_DIR,
        "blocks_dir": BLOCKS_DIR,
        "panel_dir": PANEL_DIR,
        "report_dir": REPORT_DIR,
        "log_dir": LOG_DIR,
        "manifest_dir": MANIFEST_DIR,
        "factor_cache_dir": FACTOR_CACHE_DIR,

        # Inputs
        "daily_input": daily_input(),
        "date_stamp": date_stamp(),
        "fisd_file": fisd_file(),
        "call_dummy_file": call_dummy_file(),

        # Factor panel
        "factor_source": FACTOR_SOURCE,
        "factors_pinned_file": FACTORS_PINNED_FILE,
        "factors_start_date": FACTORS_START_DATE,

        # Execution
        "workers": WORKERS,
        "threads_per": THREADS_PER,
        "duckdb_threads": DUCKDB_THREADS,
        "duckdb_memory_limit": DUCKDB_MEMORY_LIMIT,

        # Processing parameters
        "imp_gap": IMP_GAP,
        "business_day_gap": BUSINESS_DAY_GAP,
        "signal_lag": SIGNAL_LAG,
        "adj_window": ADJ_WINDOW,
        "calendar_name": CALENDAR_NAME,
        "default_method": DEFAULT_METHOD,
        "swap_adj_signals": SWAP_ADJ_SIGNALS,
        "start_date": START_DATE,
        "signals": SIGNALS,
        "illiq_min_obs": ILLIQ_MIN_OBS,
        "n_portf_1": N_PORTF_1,
        "n_portf_2": N_PORTF_2,
        "include_ice": INCLUDE_ICE,
        "beta_window": BETA_WINDOW,
        "beta_min_obs": BETA_MIN_OBS,
        "def_corp_min_maturity": DEF_CORP_MIN_MATURITY,
        "def_govt_tenor": DEF_GOVT_TENOR,
        "dspread_lags": DSPREAD_LAGS,
        "dspread_bandwidth": DSPREAD_BANDWIDTH,
        "dspread_mu_window": DSPREAD_MU_WINDOW,
        "cal_start": CAL_START,
        "cal_end": CAL_END,
        "tret_max_date": TRET_MAX_DATE,

        # Tolerances / output
        "float_tol": FLOAT_TOL,
        "rate_tol": RATE_TOL,
        "panel_zstd_level": PANEL_ZSTD_LEVEL,

        # External sources
        "quote_url": QUOTE_URL,
        "quote_zipkey": QUOTE_ZIPKEY,
        "bbw_extended_url": BBW_EXTENDED_URL,
        "bbw_extended_zipkey": BBW_EXTENDED_ZIPKEY,
    }


# Columns Stage 2 reads from the Stage 1 daily panel. Kept here (not buried in SQL) so a
# missing upstream column is reported before a build starts rather than mid-run.
REQUIRED_DAILY_COLUMNS = (
    "cusip_id", "trd_exctn_dt", "pr", "prfull", "acclast", "accpmt", "accall",
    "prc_vw_par", "prc_ew", "prc_first", "prc_last", "prc_hi", "prc_lo",
    "prc_bid", "prc_ask", "bid_last",
    "trade_count", "qvolume", "dvolume", "bond_amt_outstanding",
    "ytm", "mod_dur", "mac_dur", "convexity", "credit_spread", "bond_maturity", "bond_age",
    "sp_rating", "mdy_rating", "spc_rating", "mdc_rating",
    "permno", "permco", "gvkey", "ff17num", "ff30num", "db_type",
)

# Stage 2 selects bond-days that carry at least one agency rating. The public DOWNLOAD of
# the Stage 1 panel has these columns blanked for licensing reasons, which would silently
# select zero rows -- so an all-null ratings column is a hard error, not an empty panel.
RATING_COLUMNS = ("sp_rating", "mdy_rating")


def validate_config(config: dict) -> None:
    """Validate the configuration, reporting every problem at once.

    Raises FileNotFoundError / ValueError with an actionable message.
    """
    problems: list[str] = []

    # --- Stage 0 / Stage 1 trees -------------------------------------------
    if not config["stage1_dir"].exists():
        problems.append(
            f"Stage 1 directory not found: {config['stage1_dir']}\n"
            f"    Stage 2 runs on your own machine against the stage0/ and stage1/\n"
            f"    folders produced on WRDS. Download them next to stage2/ first."
        )
    if not config["stage0_dir"].exists():
        problems.append(f"Stage 0 directory not found: {config['stage0_dir']}")

    # --- the daily panel ---------------------------------------------------
    daily = config["daily_input"]
    if not daily.exists():
        problems.append(
            f"Stage 1 daily panel not found: {daily}\n"
            f"    Expected stage1/data/stage1_YYYYMMDD.parquet, or set DAILY_INPUT\n"
            f"    in _stage2_settings.py (or the STAGE2_DAILY_INPUT environment variable)."
        )

    # --- auxiliary inputs from stage 0 / stage 1 ---------------------------
    if config["fisd_file"] is None:
        problems.append(
            f"FISD characteristics file not found in {config['stage0_dir'] / 'enhanced'}\n"
            f"    Expected trace_enhanced_fisd_YYYYMMDD.parquet (written by Stage 0)."
        )
    if config["call_dummy_file"] is None:
        problems.append(
            f"Callable-flag file not found in {config['stage1_data']}\n"
            f"    Expected call_dummy_YYYYMMDD.parquet (written by Stage 1)."
        )

    # --- factor source -----------------------------------------------------
    if config["factor_source"] not in ("public", "pinned"):
        problems.append(
            f"FACTOR_SOURCE={config['factor_source']!r} is not valid. "
            f"Use 'public' (default) or 'pinned'."
        )
    if config["factor_source"] == "pinned":
        pinned = config["factors_pinned_file"]
        if not pinned:
            problems.append(
                "FACTOR_SOURCE='pinned' requires FACTORS_PINNED_FILE to name a factors parquet."
            )
        elif not Path(pinned).exists():
            problems.append(f"FACTORS_PINNED_FILE not found: {pinned}")

    # --- WRDS credentials (needed only for the first-run caches) -----------
    cached = FACTOR_CACHE_DIR.exists() and any(FACTOR_CACHE_DIR.glob("*.parquet"))
    if not config["wrds_username"] and not cached:
        problems.append(
            "WRDS_USERNAME not set, and the factor caches are empty.\n"
            "    Stage 2's first run fetches Treasury returns, Fama-French factors and\n"
            "    VIX from WRDS, then caches them. Set WRDS_USERNAME in config.py or as\n"
            "    an environment variable. Later runs need no credentials."
        )

    if problems:
        raise FileNotFoundError(
            "Stage 2 configuration problems:\n\n  - " + "\n\n  - ".join(problems)
        )

    # --- schema + content checks on the daily panel ------------------------
    # Deferred until the paths are known good so the messages stay clear.
    _validate_daily_panel(daily)


def _validate_daily_panel(daily: Path) -> None:
    """Check the Stage 1 panel has the columns Stage 2 needs, with usable ratings."""
    try:
        import duckdb
    except ImportError:  # pragma: no cover - duckdb is a hard requirement at build time
        return

    con = duckdb.connect()
    try:
        src = str(daily.as_posix())
        present = {
            r[0] for r in con.execute(
                f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{src}'))"
            ).fetchall()
        }
        missing = [c for c in REQUIRED_DAILY_COLUMNS if c not in present]
        if missing:
            raise ValueError(
                f"The Stage 1 panel at {daily} is missing {len(missing)} column(s) "
                f"Stage 2 needs:\n    {', '.join(missing)}\n"
                f"    A panel produced by this repository's Stage 1 contains all of them."
            )

        # Ratings must actually carry values, not just exist.
        blank = [
            c for c in RATING_COLUMNS
            if con.execute(
                f"SELECT count(*) FROM (SELECT 1 FROM read_parquet('{src}') "
                f"WHERE {c} IS NOT NULL LIMIT 1)"
            ).fetchone()[0] == 0
        ]
        if len(blank) == len(RATING_COLUMNS):
            raise ValueError(
                f"Every agency rating column is empty in {daily.name} "
                f"({', '.join(RATING_COLUMNS)} are all NULL).\n"
                f"    Stage 2 keeps only bond-days carrying at least one agency rating, "
                f"so this input would produce an EMPTY panel.\n"
                f"    The Stage 1 file published for download has ratings removed for "
                f"licensing reasons and cannot drive Stage 2.\n"
                f"    Run Stage 0 and Stage 1 yourself on WRDS and use that output."
            )
    finally:
        con.close()


# ============================================================================
# ENGINE COMPATIBILITY SURFACE (DO NOT EDIT)
# ============================================================================
# The build engine under lib/ and steps/ is a verbatim port of a validated reference
# implementation, and it reads its configuration as `cfg.<NAME>`. Exposing that exact
# surface here means those modules port with ONLY their import line changed -- which is
# what makes the output provably identical to the reference.
#
# Everything below is an alias or a thin wrapper over the settings above. The names are
# the reference's, not ours; edit the user-facing settings at the top of this file.

REPO = ROOT_PATH                      # the reference called the tree root REPO
DATA_DIR = STAGE2_DATA                # cached external inputs
BBW_EXTENDED_CACHE = STAGE2_DATA / BBW_EXTENDED_ZIPKEY

# The reference carried several input "modes" (a frozen golden vintage vs a live build).
# The public pipeline has exactly one input: whatever Stage 1 produced.
INPUT_MODE = "stage1"

DATE_STAMP = date_stamp()

# Auxiliary merges. The reference pinned these to fixed files; here they are resolved
# from the Stage 0 / Stage 1 outputs, with environment overrides for testing.
AUX = {
    # One resolution path, shared with validate_config -- see `fisd_file`.
    "fisd": fisd_file(),
    "call": call_dummy_file(),
    # The reference merged a separate issuer-level linker file to attach permno/permco/
    # gvkey. Stage 1 now attaches those itself, at bond level and with dated windows, so
    # this merge is on its way out. It is kept here (default None = skip) so the port can
    # be proven identical to the reference before the behaviour is changed.
    "linker": Path(os.environ["STAGE2_LINKER_FILE"]) if os.environ.get("STAGE2_LINKER_FILE")
              else None,
}

# Validation targets exist only where a reference build is available to diff against.
# A public build has none, and the validators skip cleanly on an empty mapping.
#
# The one entry a public user can populate is "factors": FACTOR_SOURCE="pinned" reads it,
# which is how you reproduce a specific published vintage exactly.
GOLDEN_OUTPUTS: dict = {}
_pinned_factors = os.environ.get("STAGE2_FACTORS_PINNED") or FACTORS_PINNED_FILE
if _pinned_factors:
    GOLDEN_OUTPUTS["factors"] = Path(_pinned_factors)


def tret_max_date(mode: str | None = None) -> str | None:
    """Treasury-series vintage cutoff; None means use every month available.

    STAGE2_TRET_MAX_DATE overrides the setting, which is how a published vintage built
    against an older Treasury pull is reproduced exactly.
    """
    return os.environ.get("STAGE2_TRET_MAX_DATE") or TRET_MAX_DATE


def factor_source_for(mode: str | None = None) -> str:
    """The factor-panel source ('public' | 'pinned'). A CLI override wins over this."""
    return FACTOR_SOURCE


def print_config_summary(config: dict) -> None:
    """Print a readable summary of the resolved configuration."""
    line = "=" * 78
    print(line)
    print("STAGE 2 CONFIGURATION")
    print(line)
    print(f"  Root path            : {config['root_path']}")
    print(f"  Daily input          : {config['daily_input']}")
    print(f"  Output stamp         : {config['date_stamp']}")
    print(f"  FISD characteristics : {config['fisd_file'] or '(not found)'}")
    print(f"  Callable flags       : {config['call_dummy_file'] or '(not found)'}")
    print(f"  Factor source        : {config['factor_source']}")
    print(f"  Panel start          : {config['start_date']}")
    print(f"  Beta window          : {config['beta_window']} months "
          f"(min {config['beta_min_obs']} obs)")
    print(f"  DEF / TERM legs      : corporate tmat >= {config['def_corp_min_maturity']:g}y, "
          f"government {config['def_govt_tenor']:g}y")
    print(f"  Workers x threads    : {config['workers']} x {config['threads_per']}")
    print(f"  Output directory     : {config['output_dir']}")
    print(f"  Reports              : {config['report_dir']}")
    print(line)

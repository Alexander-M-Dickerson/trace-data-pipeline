# Stage 1 - TRACE Bond Analytics

This stage enriches the cleaned TRACE daily panels from Stage 0 with:
- **Bond characteristics** from FISD (maturity, coupon, offering amount, etc.)
- **Bond analytics** computed via QuantLib (duration, convexity, credit spreads, etc.)
- **Credit ratings** from S&P and Moody's
- **Equity identifiers** (CRSP PERMNO/PERMCO and GVKEY)
- **Ultra-distressed bond filters** to flag potentially erroneous prices
- **Fama-French industry classifications** for issuer analysis

The output is a comprehensive daily bond-level dataset ready for empirical research.

If you want to get started quickly, see **[QUICKSTART_stage1.md](QUICKSTART_stage1.md)**.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Repo Layout](#repo-layout-key-files)
- [Quick Start](#quick-start)
- [What Stage 1 Does](#what-stage-1-does)
- [Configuration](#configuration-choices-you-can-edit)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Understanding Accrued Interest Variables](#understanding-accrued-interest-variables)
- [Computing Returns](#computing-returns)
- [File Structure Requirements](#file-structure-requirements)
- [Platform Compatibility](#platform-compatibility)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [License & Citation](#license--citation)

---

## Overview

Stage 1 takes the cleaned daily TRACE panels from Stage 0 and computes:

1. **FISD bond characteristics** (coupon, maturity, issuer, etc.)
2. **Computed bond analytics**:
   - Modified duration, convexity
   - Yield to maturity (YTM)
   - Credit spreads (vs. treasury curve)
   - Accrued interest
3. **Credit ratings** (S&P and Moody's with numeric conversions)
4. **Equity identifiers** (CRSP PERMNO/PERMCO and GVKEY)
5. **Ultra-distressed filters** to flag suspicious prices
6. **Fama-French industry classifications**

The result is a research-ready dataset with ~50+ variables per bond-day observation.

---

## Prerequisites

**Required from Stage 0:**
- Completed Stage 0 processing (Enhanced, Standard, and/or 144A TRACE)
- Stage 0 outputs in `stage0/{enhanced,standard,144a}/trace_*_YYYYMMDD.parquet`

**Python version:** 3.10 or higher (tested with Python 3.12.11)

**Required packages:**

- `pandas == 2.2.3` (tested version)
- `numpy == 2.2.5` (tested version)
- `wrds >= 3.3.0` (for FISD, ratings data)
- `pyarrow >= 20.0.0`
- `tqdm`
- **`QuantLib == 1.37` (REQUIRED - must have this exact version)**
- **`joblib == 1.5.1` (REQUIRED - must have this exact version)**
- `openpyxl` (for reading Excel/Google Sheets)
- `requests` (for downloading external data)

**Optional (for report generation):**
- `matplotlib >= 3.8.0`

**IMPORTANT:** QuantLib 1.37 and joblib 1.5.1 are **required** for correct operation. The code was tested and validated with these specific versions on Python 3.12.11. The log files will print your Python version and package versions for reproducibility.

**WRDS Access Required:**
- TRACE (already used in Stage 0)
- FISD (Mergent Fixed Income Securities Database)
- S&P and Moody's ratings databases

---

## Repo layout (key files)

```
stage1/
  # Shell script for job submission
  run_stage1.sh                # Submits Stage 1 job to SGE

  # Configuration
  _stage1_settings.py          # Central configuration: paths, filters, parameters

  # Python runner (called by shell script)
  _run_stage1.py               # Runner (calls CreateDailyStage1)

  # Core processing module
  create_daily_stage1.py       # Main Stage 1 class (wraps all steps)

  # Helper functions (DO NOT EDIT)
  helper_functions.py          # All utility functions used by create_daily_stage1.py
  _debug_stage1_vFinal.py      # Original debug script (reference only)

  # Output directories (created automatically)
  logs/                        # Job logs (.out and .err files)
  data/                        # Final enriched dataset + reports
    stage1_YYYYMMDD.parquet    # Main output file
    reports/                   # LaTeX reports and figures (if generated)
```

**Important file relationships:**
- `_stage1_settings.py` → Contains ALL user-configurable parameters
- `run_stage1.sh` → Submits `_run_stage1.py` to SGE
- `_run_stage1.py` → Loads config and calls `create_daily_stage1.py`
- `create_daily_stage1.py` → Wraps `helper_functions.py` into a clean pipeline
- `helper_functions.py` → Contains all the actual processing logic (DO NOT EDIT)

---

## Quick Start

### 1. Install required Python packages

On WRDS Cloud or your local machine:

```bash
python -m pip install --user pandas==2.2.3 numpy==2.2.5 wrds pyarrow tqdm QuantLib==1.37 joblib==1.5.1 openpyxl requests matplotlib
```

Or use `requirements.txt` (recommended):

```bash
python -m pip install --user -r requirements.txt
```

Or if using a virtual environment (recommended):

```bash
# Create virtual environment in project root
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Navigate to stage1 directory and configure settings

```bash
cd ~/proj/stage1  # or wherever you cloned the repo
```

**CRITICAL:** Edit `_stage1_settings.py`:

```python
# Set your WRDS username
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username_here")

# Set ROOT_PATH (parent directory containing stage0/ and stage1/)
# Option 1: Leave blank for auto-detection (recommended)
ROOT_PATH = ""  # Auto-detects from current working directory

# Option 2: Manually specify (uncomment if needed)
# ROOT_PATH = Path("~/proj").expanduser()                          # Linux/Mac/WRDS
# ROOT_PATH = Path("C:\\Users\\YourName\\Documents\\trace_data")   # Windows

# Set the date stamp from your Stage 0 run
STAGE0_DATE_STAMP = "20251022"  # Match your stage0 output files

# Specify which TRACE datasets to include
TRACE_MEMBERS = ["enhanced", "standard", "144a"]
```

To edit via command line using `nano`:

```bash
nano _stage1_settings.py
```

Change the settings at the top. Save with Ctrl + O, then Enter. Exit with Ctrl + X.

Verify your changes:
```bash
grep "WRDS_USERNAME\|ROOT_PATH\|STAGE0_DATE_STAMP" _stage1_settings.py
```

### 3. Download Required Data Files (WRDS Only)

**IMPORTANT for WRDS users:** WRDS compute nodes (where SGE jobs run) don't have internet access. You must pre-download required data files on the login node.

```bash
# Create data directory
mkdir -p data

# Download Liu-Wu treasury yields from Google Sheets (run on WRDS login node)
wget -O data/liu_wu_yields.xlsx "https://docs.google.com/spreadsheets/d/11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t/export?format=xlsx&id=11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t"

# Download OSBAP Linker file (for equity identifiers)
wget -O data/linker_file_2025.zip "https://openbondassetpricing.com/wp-content/uploads/2025/11/linker_file_2025.zip"

# Unzip the linker file
unzip data/linker_file_2025.zip -d data/

# Verify downloads
ls -lh data/liu_wu_yields.xlsx
ls -lh data/OSBAP_Linker_October_2025.parquet

# Clean up zip file (optional)
rm data/linker_file_2025.zip
```

**Note:** If you're running locally (Mac/Windows) with internet access, the pipeline will automatically download these files, so you can skip this step.

### 4. Make script executable (run once)

```bash
chmod +x run_stage1.sh
```

### 5. Fix line endings (if editing on Windows)

```bash
sed -i 's/\r$//' run_stage1.sh
```

### 6. Submit the job

**On WRDS Cloud (SGE):**
```bash
qsub run_stage1.sh
```

**On local machine or Mac:**
```bash
bash run_stage1.sh
# or
./run_stage1.sh
```

**Monitor progress:**
```bash
# Check job status (WRDS only)
qstat

# Follow output logs
tail -f logs/stage1.out
tail -f logs/stage1.err  # Check for errors
```

**Expected runtime:** ~3 hours on WRDS Cloud (2 cores with 4 threads).

---

## What Stage 1 Does

The pipeline executes 10 steps in sequence:

### Step 1: Load Treasury Yields
- Fetches Liu-Wu zero-coupon treasury yields (recommended) or FRED yields
- Used for computing credit spreads in Step 5

### Step 2: Load TRACE Data from Stage 0
- Loads Enhanced, Standard, and 144A TRACE outputs from Stage 0
- Combines datasets with proper precedence (Enhanced > Standard > 144A)
- Handles date overlaps automatically
- Applies date cutoff filter
- Drops duplicates by (cusip_id, date)

### Step 3: Load FISD Data
- Connects to WRDS and fetches bond characteristics from FISD
- Includes: coupon, maturity, offering amount, issuer, security type, etc.
- Loads Fama-French industry mappings

### Step 4: Merge FISD with TRACE
- Left-joins FISD characteristics onto TRACE by cusip_id
- Separates data into columns for parallel processing vs. non-processed columns

### Step 5: Compute Bond Analytics
- **Uses QuantLib** to compute:
  - Modified duration, convexity
  - Yield to maturity (YTM)
  - Credit spreads (vs. Liu-Wu/FRED treasury curve)
  - Accrued interest
- Processes in parallel using `joblib` (configurable cores)
- Memory-efficient chunking for large datasets

### Step 6: Merge Credit Ratings
- Fetches S&P and Moody's ratings from WRDS
- Merges ratings by (cusip_id, date) with forward-fill logic
- Converts letter ratings to numeric scores
- Creates composite rating variables

### Step 7: Merge OSBAP Linker
- Downloads OSBAP (Open-Source Bond Asset Pricing) linker file
- Adds equity identifiers: PERMNO, PERMCO and GVKEY
- Enables cross-referencing with other datasets

### Step 8: Ultra-Distressed Bond Filters
- Applies sophisticated filters to flag potentially erroneous prices:
  - **Intraday inconsistency**: Large within-day price moves for low-priced bonds
  - **Anomaly detection**: Ultra-low prices inconsistent with recent history
  - **Plateau detection**: Suspicious flat pricing at very low levels
  - **Round number detection**: Prices at suspicious round numbers (0.01, 0.10, etc.)
  - **High spike detection**: Extreme high-low spreads that recover quickly

### Step 9: Final Filters
- Removes prices above threshold (default: 300% of par)
- Handles July 2002 TRACE pricing anomaly (first month of data)
- Tracks all filter statistics

### Step 10a: Build Filter Tables
- Creates LaTeX summary tables documenting all filters
- Saves final enriched dataset to `data/stage1_YYYYMMDD.parquet`

### Step 10: Generate Reports
- Generates comprehensive LaTeX data quality report
- Creates summary statistics tables
- Produces time-series figures (if enabled)
- Outputs saved to `data/reports/`

---

## Configuration choices you can edit

Open `_stage1_settings.py` and adjust the following:

### User Configuration

```python
# WRDS username
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_username_here")

# Root path (where stage0/ and stage1/ folders are located)
# Leave blank for auto-detection (recommended)
ROOT_PATH = ""  # Auto-detects from current working directory
# Or manually specify:
# ROOT_PATH = Path("~/proj").expanduser()                          # Linux/Mac/WRDS
# ROOT_PATH = Path("C:\\Users\\YourName\\Documents\\trace_data")   # Windows

# Stage 0 output date stamp
STAGE0_DATE_STAMP = "20251022"  # Must match your stage0 output files

# Which TRACE datasets to include
TRACE_MEMBERS = ["enhanced", "standard", "144a"]  # Or ["enhanced"] only

# Date filter
DATE_CUT_OFF = "2025-03-31"  # Only include data on/before this date

# Parallel processing
N_CORES = 10   # Number of CPU cores for bond analytics computation
N_CHUNKS = 2   # Number of chunks for parallel operations
```

### Output Settings

```python
OUTPUT_FORMAT = "parquet"      # Options: "parquet" (recommended), "csv"
GENERATE_REPORTS = True        # Generate LaTeX data quality reports
OUTPUT_FIGURES = True          # Generate time-series figures (can be slow)
```

### Yield Data Configuration

```python
YLD_TYPE = 'LIU_WU'  # Options: 'LIU_WU' (recommended), 'FRED'
```

**Liu-Wu** provides zero-coupon treasury yields at daily frequency with maturities from 1 month to 30 years. This is the recommended source for credit spread calculation.

**FRED** provides constant maturity treasury yields, but with fewer maturities and potential gaps.

### Ultra-Distressed Filter Configuration

Fine-tune the ultra-distressed bond filters in `ULTRA_DISTRESSED_CONFIG`:

```python
ULTRA_DISTRESSED_CONFIG = {
    'price_col': 'pr',  # Price column to analyze

    # Intraday inconsistency thresholds
    'intraday_range_threshold': 0.75,  # 75% within-day move triggers flag
    'intraday_price_threshold': 20,    # Only for prices below 20% of par

    # Anomaly detection
    'ultra_low_threshold': 0.10,       # 0.10% of par = $1
    'min_normal_price_ratio': 3.0,     # vs. recent median

    # Plateau detection
    'plateau_ultra_low_threshold': 0.15,  # 0.15% of par = $1.50
    'min_plateau_days': 2,                # Minimum days for plateau flag

    # Suspicious round numbers
    'suspicious_round_numbers': [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00],

    # High spike detection
    'high_spike_threshold': 5.0,       # High/Low > 5x
    'min_spike_ratio': 3.0,            # vs. recent median
    'recovery_ratio': 2.0,             # Quick recovery pattern
}
```

### Final Filters Configuration

```python
FINAL_FILTER_CONFIG = {
    'price_threshold': 300,    # Remove prices above 300% of par
    'dip_threshold': 35,       # Handle July 2002 pricing anomaly (below 35% of par)
}
```

---

## Running the Pipeline

### On WRDS Cloud (Recommended)

Submit to Sun Grid Engine:

```bash
cd ~/proj/stage1
qsub run_stage1.sh
```

Monitor job:
```bash
qstat                        # Check job status
tail -f logs/stage1.out      # Follow output log
tail -f logs/stage1.err      # Check for errors
```

Job states:
- `r` = running
- `qw` = queued, waiting
- `Eqw` = error

Cancel job if needed:
```bash
qdel <job_id>
```

### On Local Machine (Mac/Linux/Windows)

Run directly with Python:

```bash
cd /path/to/stage1
python3 _run_stage1.py
```

Or via shell script:
```bash
bash run_stage1.sh
# or
./run_stage1.sh
```

**Note:** On Windows, you may need to use:
```bash
python _run_stage1.py
```

---

## Outputs

### Output directory structure

```
stage1/
├── logs/                           # Execution logs
│   ├── stage1.out                  # Standard output
│   ├── stage1.err                  # Standard error
│   └── stage1_YYYYMMDD_HHMMSS.log  # Detailed processing log
│
└── data/                           # All output data files
    ├── stage1_YYYYMMDD.parquet     # Main enriched dataset
    └── reports/                    # Data quality reports (if generated)
        ├── stage1_data_report.tex
        ├── references.bib
        └── figures/
            ├── fig_*.pdf
            └── ...
```

### Main output file

**File:** `data/stage1_YYYYMMDD.parquet`

**Structure:** Panel data with one row per (cusip_id, trd_exctn_dt) combination

**Key variables (~45 columns):**

**Identifiers:**
- `cusip_id` - 9-character CUSIP identifier
- `issuer_cusip` - 6-character issuer CUSIP
- `permno` - CRSP PERMNO (equity identifier)
- `permco` - CRSP PERMCO (company identifier)
- `gvkey` - Compustat GVKEY (company identifier)
- `trd_exctn_dt` - Trade execution date

**Computed bond analytics (QuantLib):**
- `pr` - Volume-weighted price (clean price from TRACE)
- `prclean` - Clean price from QuantLib (should match `pr`)
- `prfull` - Dirty price (clean price plus accrued interest: `prclean + acclast`)
- `acclast` - Accrued interest since last coupon payment date
- `accpmt` - Cumulative sum of all coupon payments made on or before settlement date
- `accall` - Total accumulation (`acclast + accpmt`)
- `ytm` - Yield to maturity (%)
- `mod_dur` - Modified duration (years)
- `mac_dur` - Macaulay duration (years)
- `convexity` - Convexity
- `bond_maturity` - Time to maturity (years)
- `credit_spread` - Credit spread vs. treasury (%)

**TRACE pricing (from Stage 0):**
- `prc_ew` - Equal-weighted price
- `prc_vw_par` - Par volume-weighted price
- `prc_first` - First trade price of day
- `prc_last` - Last trade price of day
- `prc_hi` - Intraday high price
- `prc_lo` - Intraday low price
- `prc_bid` - Customer bid price
- `prc_ask` - Customer ask price
- `trade_count` - Number of trades
- `qvolume` - Par dollar volume ($ millions)
- `dvolume` - Dollar volume ($ millions)
- `bid_count` - Number of bid quotes
- `ask_count` - Number of ask quotes

**Bond characteristics (from FISD):**
- `coupon` - Coupon rate (%)
- `principal_amt` - Principal amount (typically 1000)
- `bond_age` - Age of bond in years
- `bond_amt_outstanding` - Amount outstanding ($ millions)
- `callable` - Callable indicator

**Industry classifications:**
- `ff17num` - Fama-French 17 industry code (1-17)
- `ff30num` - Fama-French 30 industry code (1-30)

**Credit ratings:**
- `sp_rating` - S&P letter rating
- `sp_naic` - NAIC rating category (from S&P)
- `mdy_rating` - Moody's letter rating
- `spc_rating` - S&P composite rating (S&P, else Moody's if S&P missing)
- `mdc_rating` - Moody's composite rating
- `comp_rating` - Composite rating: average of `spc_rating` and `mdc_rating`

**Database source:**
- `db_type` - Source database (1=Enhanced, 2=Standard, 3=144A)

### Log files

**`logs/stage1.out`** - Standard output from job execution

**`logs/stage1.err`** - Standard error messages (check here first if job fails)

**`logs/stage1_YYYYMMDD_HHMMSS.log`** - Detailed processing log with:
- Configuration summary
- System and package versions
- Memory usage tracking
- Row counts at each step
- Filter statistics
- Timing information

### Reports (if generated)

**Location:** `data/reports/`

**Files:**
- `stage1_data_report.tex` - LaTeX source for data quality report
- `references.bib` - Bibliography file
- `figures/*.pdf` - Time-series figures

**Report contents:**
- Summary statistics tables
- Filter application statistics
- Sample composition by year, rating, industry
- Time-series plots (if `OUTPUT_FIGURES = True`)

Compile the LaTeX report:
```bash
cd data/reports
pdflatex stage1_data_report.tex
bibtex stage1_data_report
pdflatex stage1_data_report.tex
pdflatex stage1_data_report.tex
```

---

## Understanding Accrued Interest Variables

The Stage 1 pipeline computes three accrued interest variables using QuantLib that are important for bond pricing and return calculations:

### Variable Definitions

#### `acclast` - Accrued Interest Since Last Coupon

Computed as `bond.accruedAmount(SettlementDate)` in QuantLib, this represents the accrued interest from the last coupon payment date up to the settlement date (transaction date + 2 business days).

**Interpretation**: This is the standard "AI" (accrued interest) in bond pricing. If you buy a bond, you pay the clean price plus this accrued interest to compensate the seller for the portion of the coupon earned but not yet received.

**Example**: A bond with a 5% annual coupon ($5 per $100 face) pays $2.50 semiannually. If 45 days have elapsed since the last coupon payment (out of 182 days between payments), then:
```
acclast = ($2.50) × (45/182) ≈ $0.62
```

#### `accpmt` - Cumulative Coupon Payments

Computed as `sum(cf.amount() for cf in bond.cashflows() if cf.date() <= SettlementDate)`, this represents the sum of all coupon payments that have been made on or before the settlement date since the bond was issued.

**Interpretation**: This is a cumulative measure that grows over the life of the bond as coupons are paid. For a 5% semiannual bond issued 5 years ago, this would be 10 coupon payments of $2.50 each = $25.00.

**Key Property**: The difference in `accpmt` between two dates captures any coupon payments received during that period. If `accpmt_t - accpmt_{t-1} = $2.50`, a coupon was paid between times t-1 and t.

#### `accall` - Total Accumulation

Computed as `acclast + accpmt`, this combines current accrued interest with all historical coupon payments.

**Interpretation**: This represents the total interest accumulation from issuance through the settlement date, including both realized coupons (`accpmt`) and accrued but unpaid interest (`acclast`).

**Critical Usage**: In return calculations, `accall` is used in the **NUMERATOR** (to capture total value including cash flows), while `acclast` is used in the **DENOMINATOR** (as part of the dirty price for standardization).

---

## Computing Returns

### Clean Returns (Price Appreciation Only)

Clean returns reflect only price changes, excluding accrued interest and coupon income:

$$
R_{\text{clean},t} = \frac{P_t}{P_{t-1}} - 1
$$

where $P_t$ can be any of the clean price measures: `pr`, `prc_ew`, `prc_vw`, `prc_first`, `prc_last`, etc.

**Use case**: Useful for analyzing pure price movements or when comparing bonds with different coupon structures.

### Total Returns (Including Accrued Interest and Coupons)

#### Method 1: Correct Formula Using `accall` and `prfull` (Recommended)

The **CORRECT** bond return formula uses `accall` in the numerator and dirty price (`prfull`) in the denominator:

$$
R_{\text{total},t} = \frac{(P_t + \text{accall}_t) - (P_{t-1} + \text{accall}_{t-1})}{P_{t-1} + \text{acclast}_{t-1}}
$$

**Key Distinction**:
- **Numerator**: `pr + accall` = price + accumulated payments (includes cash flows)
- **Denominator**: `prfull = pr + acclast` = dirty price (standardization base)

**Why this is correct**:
- The numerator captures total value change including cash flows (`accall`)
- The denominator uses the dirty price (`prfull = pr + acclast`) as the standardization base
- When a coupon is paid, `accall` increases by the coupon amount (captured in numerator)
- The dirty price provides the correct base for standardizing returns across bonds

**Python implementation**:
```python
import pandas as pd
import numpy as np

# Load Stage 1 output
df = pd.read_parquet('data/stage1_YYYYMMDD.parquet')

# Sort by bond and date (required for lagged calculations)
df = df.sort_values(['cusip_id', 'trd_exctn_dt']).reset_index(drop=True)

# Full price = clean price + accumulated payments (numerator)
df['fp'] = df['pr'] + df['accall']

# Compute lagged values
df['fp_lag'] = df.groupby('cusip_id', observed=True)['fp'].shift(1)
df['prfull_lag'] = df.groupby('cusip_id', observed=True)['prfull'].shift(1)
df['pr_lag'] = df.groupby('cusip_id', observed=True)['pr'].shift(1)

# Total return: (fp_t - fp_{t-1}) / prfull_{t-1}
df['ret_d'] = (df['fp'] - df['fp_lag']) / df['prfull_lag']

# Clean return: (pr_t - pr_{t-1}) / pr_{t-1}
df['ret_c'] = (df['pr'] - df['pr_lag']) / df['pr_lag']
```

**WARNING**: TRACE bond data is NOT contiguous - bonds may not trade every day. After computing returns, you must check the time gap between consecutive observations and apply appropriate filters. For example, compute the number of business days between `trd_exctn_dt` observations and exclude returns where the gap exceeds your chosen threshold (e.g., maximum 5 business days). Returns computed over long gaps may not reflect true holding period returns.

#### Method 2: Standard Formula with Explicit Coupons

The traditional bond return formula explicitly extracts coupon payments:

$$
R_{\text{total},t} = \frac{P_t + AI_t + C_t}{P_{t-1} + AI_{t-1}} - 1
$$

where:
- $P_t$ = clean price at time t
- $AI_t$ = accrued interest at time t (corresponds to `acclast_t`)
- $AI_{t-1}$ = accrued interest at time t-1 (corresponds to `acclast_{t-1}`)
- $C_t$ = coupon payment received between t-1 and t (extracted as `accpmt_t - accpmt_{t-1}`)

**Python implementation**:
```python
import pandas as pd
import numpy as np

# Load Stage 1 output
df = pd.read_parquet('data/stage1_YYYYMMDD.parquet')

# Sort by bond and date
df = df.sort_values(['cusip_id', 'trd_exctn_dt']).reset_index(drop=True)

# Extract coupon payments (0 if no coupon paid, coupon amount if paid)
df['coupon_received'] = df.groupby('cusip_id', observed=True)['accpmt'].diff().fillna(0)

# Calculate dirty prices
df['dirty_price'] = df['pr'] + df['acclast']
df['dirty_price_lag'] = df.groupby('cusip_id', observed=True)['dirty_price'].shift(1)

# Total return (standard formula)
df['ret_total_standard'] = (
    (df['dirty_price'] + df['coupon_received']) / df['dirty_price_lag']
) - 1
```

**Comparison of Methods**:
- **Method 1 (accall with prfull)**: Simpler, no need to extract coupon differences explicitly. Uses `accall` in numerator and `prfull = pr + acclast` in denominator.
- **Method 2 (standard with explicit coupons)**: Traditional formula, separates coupons explicitly, uses dirty price in denominator. Mathematically equivalent to Method 1.

**Both methods are correct and produce identical results**. Method 1 is preferred due to its simplicity - it avoids computing coupon differences while maintaining the correct standardization base (dirty price) in the denominator.

---

## File Structure Requirements

Stage 1 expects Stage 0 outputs to follow this structure:

```
ROOT_PATH/
├── stage0/
│   ├── enhanced/
│   │   └── trace_enhanced_YYYYMMDD.parquet    # Required if "enhanced" in TRACE_MEMBERS
│   ├── standard/
│   │   └── trace_standard_YYYYMMDD.parquet    # Required if "standard" in TRACE_MEMBERS
│   └── 144a/
│       └── trace_144a_YYYYMMDD.parquet        # Required if "144a" in TRACE_MEMBERS
│
└── stage1/
    ├── _stage1_settings.py
    ├── create_daily_stage1.py
    ├── _run_stage1.py
    ├── run_stage1.sh
    ├── helper_functions.py
    ├── logs/                                   # Created automatically
    └── data/                                   # Created automatically
```

**Important:**
- `STAGE0_DATE_STAMP` in `_stage1_settings.py` must match the date stamp in your Stage 0 output filenames
- `ROOT_PATH` can be left blank (auto-detects from current directory) or manually specified
- If you run the script from `~/proj/stage1`, ROOT_PATH is automatically set to `~/proj`

---

## Platform Compatibility

This code is designed to run on **WRDS Cloud**, **Mac**, and **Windows** with minimal configuration changes.

### WRDS Cloud (Linux)

```python
# In _stage1_settings.py:
ROOT_PATH = ""  # Auto-detect (recommended)
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_id")
```

Submit with SGE from the `stage1/` directory:
```bash
cd ~/proj/stage1
qsub run_stage1.sh
```

### Mac

```python
# In _stage1_settings.py:
ROOT_PATH = ""  # Auto-detect (recommended)
# Or manually: ROOT_PATH = Path("~/Documents/trace_data")
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_id")
```

Run locally from the `stage1/` directory:
```bash
cd ~/Documents/trace_data/stage1
python3 _run_stage1.py
```

### Windows

```python
# In _stage1_settings.py:
ROOT_PATH = ""  # Auto-detect (recommended)
# Or manually: ROOT_PATH = Path("C:\\Users\\YourName\\Documents\\trace_data")
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_id")
```

Run from Command Prompt or PowerShell from the `stage1\` directory:
```bash
cd C:\Users\YourName\Documents\trace_data\stage1
python _run_stage1.py
```

**Note:**
- Auto-detection works when you run the script from the `stage1/` directory
- Manual override available if running from a different location

---

## Troubleshooting

### Configuration Issues

**Error: "WRDS_USERNAME not set"**

Solution: Set in `_stage1_settings.py`:
```python
WRDS_USERNAME = "your_wrds_username_here"
```

Or set environment variable:
```bash
export WRDS_USERNAME="your_wrds_username_here"
```

**Error: "Stage0 directory not found"**

Solution:
1. Ensure you're running the script from the `stage1/` directory:
   ```bash
   cd ~/proj/stage1  # or wherever your stage1 folder is
   ```
2. If auto-detection doesn't work, manually specify `ROOT_PATH` in `_stage1_settings.py`:
   ```python
   ROOT_PATH = Path("~/proj").expanduser()  # Or your actual root path
   ```

**Error: "Stage0 output files not found"**

Solution: Verify `STAGE0_DATE_STAMP` matches your Stage 0 output files. Check:
```bash
ls stage0/enhanced/trace_enhanced_*.parquet
ls stage0/standard/trace_standard_*.parquet
ls stage0/144a/trace_144a_*.parquet
```

### Package Issues

**Error: "ModuleNotFoundError: No module named 'QuantLib'"**

Solution:
```bash
python -m pip install --user QuantLib==1.37
```

**IMPORTANT:** You must install QuantLib version 1.37 specifically for this code to work correctly.

**Error: "ModuleNotFoundError: No module named 'helper_functions'"**

Solution: Ensure `helper_functions.py` is in the `stage1/` directory and you're running the script from the `stage1/` directory.

### WRDS Connection Issues

**Error: "Unable to connect to WRDS"**

Solution: Verify `.pgpass` file is set up correctly:
```bash
chmod 600 ~/.pgpass
cat ~/.pgpass  # Should contain: wrds-pgdata.wharton.upenn.edu:9737:wrds:your_username:your_password
```

**Error: "No tables found in WRDS"**

Solution: Verify your WRDS account has access to:
- FISD (Mergent Fixed Income Securities Database)
- S&P ratings (COMPUSTAT or Capital IQ)
- Moody's ratings

### Memory Issues

**Error: "MemoryError" or "Killed"**

Solution: Reduce parallel processing:
```python
# In _stage1_settings.py:
N_CORES = 4    # Reduce from default 10
N_CHUNKS = 4   # Increase from default 2
```

Or request more memory on WRDS:
```bash
# In run_stage1.sh, add:
#$ -l m_mem_free=16G
```

### Performance Issues

**Pipeline is very slow**

Solutions:
1. Disable report generation:
   ```python
   GENERATE_REPORTS = False
   ```

2. Disable figure generation:
   ```python
   OUTPUT_FIGURES = False
   ```

3. Increase parallel cores (if you have memory):
   ```python
   N_CORES = 20
   ```

4. Process fewer TRACE datasets:
   ```python
   TRACE_MEMBERS = ["enhanced"]  # Only process Enhanced
   ```

---

## Performance Optimization

### Parallel Processing

Stage 1 uses `joblib` for parallel processing of bond analytics. Tune these settings:

```python
N_CORES = 10   # Number of CPU cores to use (adjust based on your machine)
N_CHUNKS = 2   # Number of chunks (increase if memory issues)
```

**Guidelines:**
- **WRDS Cloud**: Set `N_CORES = 10-20` (check `qstat -F` for available cores)
- **Local machine**: Set to number of physical cores - 2
- **Memory issues**: Increase `N_CHUNKS` to process smaller batches

### Output Format

```python
OUTPUT_FORMAT = "parquet"  # Recommended: 10x smaller and faster than CSV
# OUTPUT_FORMAT = "csv"    # Use only if you need human-readable output
```

### Report Generation

```python
GENERATE_REPORTS = True   # Full LaTeX reports with all tables
OUTPUT_FIGURES = True     # Time-series plots (can add 30+ minutes)
```

Disable reports for faster processing:
```python
GENERATE_REPORTS = False
```

Or generate reports without figures:
```python
GENERATE_REPORTS = True
OUTPUT_FIGURES = False  # Tables only, no plots
```

---

## License & Citation

### License

This code is provided under the MIT License. See LICENSE file for details.

### Citation

If you use this stage in your research, please cite:

**Primary Reference:**
```
Dickerson, A., Robotti, C., & Rossetti, G. (2025).
Common pitfalls in the evaluation of corporate bond strategies.
Working Paper.
```

**Secondary Reference:**
```
Dickerson, A., & Rossetti, G. (2025).
Constructing TRACE Corporate Bond Datasets.
Working Paper.
```

---

## Support

For questions, issues, or contributions:
- **Email**: alexander.dickerson1@unsw.edu.au
- **GitHub Issues**: [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)

---

## Version History

- **v1.0** (2025-11-17): Initial release
  - Bond characteristics from FISD
  - QuantLib-based analytics (duration, convexity, spreads)
  - Credit ratings (S&P, Moody's)
  - Equity identifiers (PERMNO, PERMCO, GVKEY)
  - Ultra-distressed filters
  - Comprehensive reporting

---

**Last updated:** November 2025

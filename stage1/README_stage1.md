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
- [Support](#support)
- [Version History](#version-history)

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

The result is a research-ready dataset of 44 columns per bond-day observation.

---

## Prerequisites

**Required from Stage 0:**
- Completed Stage 0 processing (Enhanced, Standard, and/or 144A TRACE)
- Stage 0 outputs in `stage0/{enhanced,standard,144a}/trace_*_YYYYMMDD.parquet`

**Python version:** 3.10 or higher

**Required packages:** everything in the repository's `requirements.txt`, which states minimum
versions (`QuantLib>=1.36`, `joblib>=1.4`, `pandas>=2.2.3`, `numpy>=2.0`, `pyarrow>=20.0.0`,
`wrds>=3.3.0`, plus `tqdm`, `openpyxl`, `requests` and `matplotlib` for the reports).

The 2026-09-21 run, which the 2026 data vintage was built from, used Python 3.14.5, pandas
2.2.3, NumPy 2.4.6, QuantLib 1.37 and joblib 1.5.1 on the WRDS Cloud. The log files print
your Python and package versions, so a run can always be matched to the environment that made it.

**WRDS Access Required:**
- TRACE (already used in Stage 0)
- FISD (Mergent Fixed Income Securities Database), including its ratings table
  `fisd.fisd_ratings`, which holds both the S&P and the Moody's ratings

---

## Repo layout (key files)

```
stage1/
  run_stage1.sh                # SGE job script; submit it from the repository root
  _stage1_settings.py          # Stage 1 settings: paths, filters, parameters
  _run_stage1.py               # Entry point, called by run_stage1.sh
  create_daily_stage1.py       # Wires the settings into the pipeline and runs it
  stage1_pipeline.py           # The steps, in order (run_all_steps)
  helper_functions.py          # The functions the steps call: bond analytics, filters
  _linker_join.py              # Attaches permno/permco/gvkey from the linker (step 7)
  _distressed_plot_helpers.py  # Figures and LaTeX for the distressed-bond report

  logs/                        # stage1.out, stage1.err, stage1_<timestamp>.log
  data/                        # the output, the downloaded inputs, the distressed report
  data_reports/                # the data-quality report
```

**How they call each other:** `run_stage1.sh` runs `_run_stage1.py`, which loads the settings
and calls `create_daily_stage1.py`, which runs `stage1_pipeline.run_all_steps()`. The steps call
`helper_functions.py` and `_linker_join.py`. Every setting you are meant to change is in
`_stage1_settings.py` or the root `config.py`. [CODE_MAP.md](../CODE_MAP.md) covers every file in
the repository.

---

## Quick Start

### 1. Install required Python packages

On WRDS Cloud or your local machine:

```bash
python -m pip install --user -r requirements.txt
```

❗Do **not** pin these to exact versions on the WRDS Cloud. `--user` installs shadow the
system packages, and WRDS already ships newer ones than any pin here (numpy 2.4.6,
pyarrow 24.0.0 on Python 3.14) — pinning downgrades a working environment. Use
`requirements.txt`, which states minimums.

Or, in a virtual environment:

```bash
# Create virtual environment in project root. --system-site-packages keeps the WRDS Cloud's own pandas 2.2 visible: pip has no build of it
# for the Cloud's Python 3.14.
python3 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Configure

Your WRDS username goes in the root `config.py`, or in the environment, which every stage
reads:

```bash
export WRDS_USERNAME="your_wrds_id"
```

A wrong username is not caught at startup: it fails when step 6 first connects to WRDS, about
1 hour 45 minutes into a full run. `TRACE_MEMBERS` (which TRACE databases to process) is also in
`config.py`, defaults to `["enhanced", "144a"]`, and can be set from the environment:
`TRACE_MEMBERS="enhanced standard 144a" ./run_pipeline.sh`.

In `stage1/_stage1_settings.py`, `ROOT_PATH` can stay blank: it is found from the folder you run
from (the repository root, or `stage1/`). Stage 0's date stamp is read from
`stage0/enhanced/trace_enhanced_<stamp>.parquet`; there is nothing to set.

### 3. Download the required data files (login node)

Stage 1 needs five files from the internet: the Liu-Wu Treasury yields, the bond-firm linker and
the three Fama-French industry files. It checks for them before it starts and stops if one is
missing. WRDS compute nodes have no internet, so fetch them on the login node, from the
repository root:

```bash
bash download_inputs.sh
```

`run_pipeline.sh` does this for you; run it yourself only when you submit stage 1 on its own.
On your own machine, run the same script (it needs `wget`).

### 4. Make script executable (older clones only)

The script ships executable, so a fresh clone needs nothing here. An older clone may need,
from the repository root:

```bash
chmod +x stage1/run_stage1.sh
```

### 5. Fix line endings (if editing on Windows)

```bash
sed -i 's/\r$//' stage1/run_stage1.sh
```

### 6. Submit the job

**On WRDS Cloud (SGE), from the repository root:**
```bash
qsub stage1/run_stage1.sh
```

**On your own machine**, from the repository root:
```bash
bash stage1/run_stage1.sh
```

**Monitor progress:**
```bash
# Check job status (WRDS only)
qstat

# Follow output logs (from the repo root)
tail -f stage1/logs/stage1.out
tail -f stage1/logs/stage1.err  # Check for errors
```

**Expected runtime:** about 2.5-2.7 hours on the WRDS Cloud, on the 4 slots and 40 GB `run_stage1.sh` requests (2.6 h on 2026-09-09, 2.4 h on 2026-09-10, 2.7 h on 2026-09-21).

---

## What Stage 1 Does

`run_all_steps()` runs the steps below in order. Two are easy to miss and both matter:
`variable_drop()`, which removes `issuer_cusip`, `prclean`, `coupon`, `principal_amt`,
`sp_naic`, `comp_rating` and `callable` and does the integer dtype casts; and
`step8b_build_distressed_report()`.

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
- Reads the FISD bond characteristics Stage 0 saved,
  `stage0/enhanced/trace_enhanced_fisd_<stamp>.parquet` (no WRDS connection yet)
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
- Connects to WRDS, for the first time in the run, and reads from FISD the S&P and Moody's
  ratings (`fisd.fisd_ratings`), the amount-outstanding history and the call data
- Merges ratings by (cusip_id, date) with forward-fill logic
- Converts letter ratings to numeric scores
- Creates composite rating variables

### Step 7: Merge the bond-firm linker
- Reads `bond_firm_linker_2026/fl_linker.parquet`, which `download_inputs.sh` fetched on the login node
- Adds equity identifiers: PERMNO, PERMCO and GVKEY, joined on the linker's dated identity window `[i0, i1]`, the same window Stage 2 joins (`LINKER_WINDOW` in `_stage1_settings.py`)
- Enables cross-referencing with other datasets

### Step 8: Ultra-Distressed Bond Filters
- Four filters flag prices that are probably errors (details in
  [README_distressed_filter.md](README_distressed_filter.md)):
  - **Anomaly**: ultra-low prices out of line with the bond's recent prices
  - **Upward spike**: a price above 5% of par (or a round price above 0.50), at least 3 times the median of the bond's
    recent lower prices, that falls back within 5 observations
  - **Plateau**: runs of flat pricing at very low levels
  - **Intraday inconsistency**: a large gap between the day's high and low for a low-priced bond
- Round prices (0.01, 0.10, 0.50 and so on) make a price more suspect inside these filters; they
  are not a separate filter

### Step 9: Final Filters
- Flags prices above 300% of par
- Flags each bond's first price change in July 2002 (TRACE's first month) when it is larger than
  35 points of par

### Step 10a: Apply the filters and save
- Removes, in order: rows without valid accrued-interest inputs (tested in step 4), rows with no
  S&P or Moody's rating, rows with less than one year to maturity, the rows step 8 flagged, and
  the two step 9 flags
- Winsorizes `ytm` and `credit_spread` at the 0.5th and 99.5th percentiles within each date
- Creates the LaTeX tables that count each filter; see
  [DATA_DICTIONARY.md](DATA_DICTIONARY.md#sample-defining-operations)
- Saves the dataset to `data/stage1_YYYYMMDD.parquet`

### Step 10: Generate Reports
- Generates comprehensive LaTeX data quality report
- Creates summary statistics tables
- Produces time-series figures (if enabled)
- Outputs saved to `stage1/data_reports/` (the distressed report goes to `stage1/data/data_reports/`)

---

## Configuration choices you can edit

Open `_stage1_settings.py` and adjust the following:

### User Configuration

In the root `config.py` -- shared by every stage:

```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")
TRACE_MEMBERS = os.getenv("TRACE_MEMBERS", "enhanced 144a").split()
```

In `stage1/_stage1_settings.py`:

```python
# Root path (where stage0/ and stage1/ live). Leave blank to auto-detect.
ROOT_PATH = ""

# There is no date-stamp setting: it is read from
# stage0/enhanced/trace_enhanced_<stamp>.parquet.

# Date filter. The DEFAULT IS ROLLING, not a fixed date:
#   "auto:complete"  the last month EVERY source covers through its final trading
#                    session -- the default
#   "auto:-3mo"      last day of the month 3 months before the least current
#                    source's last trade date
#   "2025-03-31"     a fixed date, used exactly as given
# An auto cutoff is also CLAMPED to the last date the treasury curve covers, so the
# published sample end can move between vintages without you changing anything.
DATE_CUT_OFF = "auto:complete"

# Parallel processing
N_CORES  = None   # resolves from $NSLOTS -- the slots actually granted (4 on WRDS)
N_CHUNKS = 10     # MORE chunks = smaller peak memory
```

### Output Settings

```python
OUTPUT_FORMAT = "parquet"      # imported from the shared config.py
```

❗**There are no `GENERATE_REPORTS` or `OUTPUT_FIGURES` knobs.** `_stage1_settings.py`
says so directly: *"Stage 1 always generates comprehensive reports and figures. These
outputs are essential for data quality assessment and cannot be disabled."* Setting
either name does nothing.

❗**`OUTPUT_FORMAT` must be `"parquet"`.** Stage 1's `save_outputs` always writes
Parquet regardless, and since v2.2.3 `stage0/_trace_settings.py` raises at import on any
other value -- Stage 0 would otherwise write `.csv.gzip` files that Stage 1 and the
report builder, which both read a hard-coded `*.parquet` name, cannot open.

### Yield Data Configuration

```python
YLD_TYPE = 'LIU_WU'  # Options: 'LIU_WU' (recommended), 'FRED'
```

**Liu-Wu** provides zero-coupon treasury yields at daily frequency with maturities from 1 month to 30 years. This is the recommended source for credit spread calculation.

**FRED** provides constant maturity treasury yields, but with fewer maturities and potential gaps. It is
downloaded while stage 1 runs, and WRDS compute nodes have no internet, so on the grid it fails:
use it only on a machine with internet access.

### Ultra-Distressed Filter Configuration

Fine-tune the ultra-distressed bond filters in `ULTRA_DISTRESSED_CONFIG`:

```python
ULTRA_DISTRESSED_CONFIG = {
    'price_col': 'pr',  # Price column to analyze

    # Intraday inconsistency thresholds
    'intraday_range_threshold': 0.75,  # 75% within-day move triggers flag
    'intraday_price_threshold': 20,    # Only for prices below 20% of par
    'price_cols': ['prc_hi', 'prc_lo'],  # the day's high and low

    # Anomaly detection
    'ultra_low_threshold': 0.10,       # 0.10% of par = $1
    'min_normal_price_ratio': 3.0,     # vs. recent median

    # Plateau detection
    'plateau_ultra_low_threshold': 0.15,  # 0.15% of par = $1.50
    'min_plateau_days': 2,                # minimum observations (days the bond traded)

    # Suspicious round numbers
    'suspicious_round_numbers': [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00],

    # Upward spike detection
    'high_spike_threshold': 5.0,       # spike candidate: a price above 5% of par (or round and above 0.50)
    'min_spike_ratio': 3.0,            # at least 3x the median of recent lower prices
    'recovery_ratio': 2.0,             # Quick recovery pattern

    'target_rows_per_chunk': 500000,   # rows per chunk while filtering
}
```

### Final Filters Configuration

```python
FINAL_FILTER_CONFIG = {
    'price_threshold': 300,    # Remove prices above 300% of par
    'dip_threshold': 35,       # flag a first July 2002 price change larger than 35 points
}
```

---

## Running the Pipeline

### On WRDS Cloud (Recommended)

Submit to Sun Grid Engine:

```bash
cd ~/trace-data-pipeline     # the repository root
qsub stage1/run_stage1.sh
```

Monitor job:
```bash
qstat                               # Check job status
tail -f stage1/logs/stage1.out      # Follow output log
tail -f stage1/logs/stage1.err      # Check for errors
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

Or via the shell script, from the repository root:
```bash
bash stage1/run_stage1.sh
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
├── data/
│   ├── stage1_YYYYMMDD.parquet     # the output
│   ├── call_dummy_YYYYMMDD.parquet # callable flag per bond (read by Stage 2)
│   ├── sp_ratings_YYYYMMDD.parquet, moodys_ratings_YYYYMMDD.parquet
│   ├── ultra_distressed_cusips_YYYYMMDD.csv   # bonds the distressed filter flagged
│   ├── liu_wu_yields.xlsx, Siccodes12/17/30.txt, bond_firm_linker_2026/  # downloaded inputs
│   └── data_reports/               # the distressed-bond report and its figures
│
└── data_reports/                   # the data-quality report
    ├── stage1_data_report_YYYYMMDD.tex (and .pdf)
    ├── references.bib
    ├── stage1_*.pdf                # figures
    └── time_series_data/
```

### Main output file

**File:** `data/stage1_YYYYMMDD.parquet`

**Structure:** Panel data with one row per (cusip_id, trd_exctn_dt) combination

**The output has 44 columns.** `stage1/DATA_DICTIONARY.md` is the authoritative
reference; this is a summary.

**Identifiers:**
- `cusip_id` - 9-character CUSIP identifier
- `permno` - CRSP PERMNO (equity identifier); NULL where no identity window of the linker covers the date
- `permco` - CRSP PERMCO (company identifier)
- `gvkey` - Compustat GVKEY (company identifier)
- `trd_exctn_dt` - Trade execution date

**Computed bond analytics (QuantLib):**
- `pr` - Volume-weighted price (clean price from TRACE)
- `prfull` - Dirty price (clean price plus accrued interest: `pr + acclast`)
- `acclast` - Accrued interest since last coupon payment date
- `accpmt` - Cumulative sum of all coupon payments made on or before settlement date
- `accall` - Total accumulation (`acclast + accpmt`)
- `ytm` - Yield to maturity, as a DECIMAL (0.045 = 4.5%), not a percentage.
  ❗Winsorised at the 0.5/99.5 quantiles **within each date** — see
  [DATA_DICTIONARY.md](DATA_DICTIONARY.md#sample-defining-operations).
- `mod_dur` - Modified duration (years)
- `mac_dur` - Macaulay duration (years)
- `convexity` - Convexity
- `bond_maturity` - Time to maturity (years)
- `credit_spread` - Credit spread over the MATURITY-matched treasury, as a DECIMAL
  (0.012 = 120bp), not a percentage. ❗Also winsorised per date — see
  [DATA_DICTIONARY.md](DATA_DICTIONARY.md#sample-defining-operations).

**TRACE pricing (from Stage 0):**
- `prc_ew` - Equal-weighted price
- `prc_vw_par` - Par volume-weighted price
- `prc_first` - First trade price of day
- `prc_last` - Last trade price of day
- `prc_hi` - Intraday high price
- `prc_lo` - Intraday low price
- `prc_bid` - Dealer bid: value-weighted price of trades where the DEALER bought
  from a customer (`rpt_side_cd=='B'`), i.e. the customer sold
- `prc_ask` - Dealer ask: the dealer SOLD to a customer (`rpt_side_cd=='S'`)
- `trade_count` - Number of trades
- `qvolume` - Par dollar volume ($ millions)
- `dvolume` - Dollar volume ($ millions)
- `bid_count` - Number of dealer BUY trades (TRACE has no quotes -- these are executions)
- `ask_count` - Number of dealer SELL trades

**Bond characteristics (from FISD):**
- `bond_age` - Age of bond in years
- `bond_amt_outstanding` - Amount outstanding, in **thousands of dollars**, exactly as
  FISD reports it. A bond with $250m outstanding carries `250000`. Market value is
  `bond_amt_outstanding * (pr + acclast) * 10`, in dollars.

`coupon`, `principal_amt` and `callable` are used during processing but are not in
the output; take them from FISD if you need them.

**Industry classifications** (from the issuer's SIC code in FISD; unmatched SIC codes
fall into each scheme's "Other" bucket, so these are never null):
- `ff12num` - Fama-French 12 industry code (1-12)
- `ff17num` - Fama-French 17 industry code (1-17)
- `ff30num` - Fama-French 30 industry code (1-30)

**Credit ratings:**
- `sp_rating` - S&P rating as a NUMERIC code, Int8 1-22 (1 = AAA, 22 = D)
- `mdy_rating` - Moody's rating as a NUMERIC code, Int8 1-21
- `spc_rating` - S&P composite rating (S&P, else Moody's if S&P missing)
- `mdc_rating` - Moody's composite rating

`sp_naic` and `comp_rating` are computed during processing but are not in the output.

**Database source:**
- `db_type` - Source database (1=Enhanced, 2=Standard, 3=144A)

### Log files

**`logs/stage1.out`** - Standard output from job execution, including the configuration summary

**`logs/stage1.err`** - Standard error messages (check here first if job fails)

**`logs/stage1_YYYYMMDD_HHMMSS.log`** - Detailed processing log with:
- System and package versions
- Memory usage tracking
- Row counts at each step
- Filter statistics
- Timing information

### Reports

**Location:** `stage1/data_reports/` -- figures and `time_series_data/*.csv` sit in that same directory, not a `figures/` subfolder

**Files:**
- `stage1_data_report_<STAMP>.tex` - LaTeX source for data quality report
- `references.bib` - Bibliography file
- `stage1_*.pdf` - figures

**Report contents:**
- Summary statistics tables
- Filter application statistics
- Sample composition by year, rating, industry
- Time-series figures

Compile the LaTeX report (the files carry the run stamp):
```bash
cd stage1/data_reports
pdflatex stage1_data_report_<STAMP>.tex
bibtex stage1_data_report_<STAMP>
pdflatex stage1_data_report_<STAMP>.tex
pdflatex stage1_data_report_<STAMP>.tex
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

where $P_t$ can be any of the clean price measures: `pr`, `prc_ew`, `prc_vw_par`, `prc_first`, `prc_last`, etc.

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
│   │   ├── trace_enhanced_YYYYMMDD.parquet    # Required if "enhanced" in TRACE_MEMBERS
│   │   └── trace_enhanced_fisd_YYYYMMDD.parquet  # Always required: step 3 reads it
│   ├── standard/
│   │   └── trace_standard_YYYYMMDD.parquet    # Required if "standard" in TRACE_MEMBERS
│   └── 144a/
│       └── trace_144a_YYYYMMDD.parquet        # Required if "144a" in TRACE_MEMBERS
│
└── stage1/                                    # the code, listed under Repo layout above
    ├── logs/                                   # Created automatically
    └── data/                                   # the downloaded inputs, then the output
```

**Important:**
- The date stamp is read from `stage0/enhanced/trace_enhanced_<stamp>.parquet`, and every
  member you process must carry the same stamp
- `ROOT_PATH` can be left blank (it is found from the folder you run from) or set by hand
- Run from `~/trace-data-pipeline/stage1` or from `~/trace-data-pipeline`: either way
  `ROOT_PATH` becomes `~/trace-data-pipeline`

---

## Platform Compatibility

This code is designed to run on **WRDS Cloud**, **Mac**, and **Windows** with minimal configuration changes.

### WRDS Cloud (Linux)

```python
# In the root config.py (or: export WRDS_USERNAME=your_wrds_id):
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_id")
# In stage1/_stage1_settings.py:
ROOT_PATH = ""  # Auto-detect (recommended)
```

Submit with SGE from the repository root:
```bash
cd ~/trace-data-pipeline     # the repository root
qsub stage1/run_stage1.sh
```

### Mac

```python
# In the root config.py (or: export WRDS_USERNAME=your_wrds_id):
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_id")
# In stage1/_stage1_settings.py:
ROOT_PATH = ""  # Auto-detect (recommended)
# Or manually: ROOT_PATH = Path("~/Documents/trace_data")
```

Run locally from the `stage1/` directory:
```bash
cd ~/Documents/trace_data/stage1
python3 _run_stage1.py
```

### Windows

```python
# In the root config.py (or set the WRDS_USERNAME environment variable):
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_id")
# In stage1/_stage1_settings.py:
ROOT_PATH = ""  # Auto-detect (recommended)
# Or manually: ROOT_PATH = Path("C:\\Users\\YourName\\Documents\\trace_data")
```

Run from Command Prompt or PowerShell from the `stage1\` directory:
```bash
cd C:\Users\YourName\Documents\trace_data\stage1
python _run_stage1.py
```

**Note:**
- Auto-detection works when you run from the `stage1/` directory or the repository root
- Manual override available if running from a different location

---

## Troubleshooting

### Configuration Issues

**Error: "WRDS_USERNAME not set"**

Solution: Set it in the root `config.py`:
```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username_here")
```

Or set environment variable:
```bash
export WRDS_USERNAME="your_wrds_username_here"
```

**Error: "Stage0 directory not found"**

Solution:
1. Run from the repository root or from the `stage1/` directory:
   ```bash
   cd ~/trace-data-pipeline  # or wherever you cloned the repository
   ```
2. If auto-detection doesn't work, manually specify `ROOT_PATH` in `_stage1_settings.py`:
   ```python
   ROOT_PATH = Path("~/trace-data-pipeline").expanduser()  # Or your actual root path
   ```

**Error: "Stage0 output files not found"**

Solution: The date stamp is read from `stage0/enhanced/trace_enhanced_<stamp>.parquet`, and every member in `TRACE_MEMBERS` needs a file with that same stamp. Check:
```bash
ls stage0/enhanced/trace_enhanced_*.parquet
ls stage0/standard/trace_standard_*.parquet
ls stage0/144a/trace_144a_*.parquet
```

### Package Issues

**Error: "ModuleNotFoundError: No module named 'QuantLib'"**

Solution:
```bash
python -m pip install --user QuantLib
```

`requirements.txt` asks for `QuantLib>=1.36`; do not pin an exact version.

**Error: "ModuleNotFoundError: No module named 'helper_functions'"**

Solution: Ensure `helper_functions.py` is in the `stage1/` directory and you're running the script from the `stage1/` directory.

### WRDS Connection Issues

**Error: "Unable to connect to WRDS"**

Solution:

**On the WRDS Cloud** no password file is needed: jobs there connect without one. A failed
connection is almost always the username: `WRDS_USERNAME` unset, or still `your_wrds_username`
in `config.py` (`python3 doctor.py --wrds` checks it). If the username is right, it is WRDS's
limit of 7 connections held at once (`stage0/README_stage0.md`).

**On your own computer** (stage 2's first run), the password must be saved where the `wrds`
package looks. Connect once by hand; it asks for the password and offers to save it
(`~/.pgpass`, or `%APPDATA%\postgresql\pgpass.conf` on Windows):
```bash
python -c "import wrds; wrds.Connection()"
```

**WRDS refuses access to a table**

Stage 1 reads only FISD from WRDS: `fisd.fisd_ratings` (both the S&P and the Moody's ratings),
`fisd.fisd_amt_out_hist`, `fisd.fisd_mergedissue` and `fisd.fisd_mergedredemption`. Check that
your subscription includes FISD.

### Memory Issues

**Error: "MemoryError" or "Killed"**

Solution: use more, smaller chunks, and leave the worker count alone:
```python
# In _stage1_settings.py:
N_CHUNKS = 20    # default 10; more chunks means a smaller peak
N_CORES = None   # default: follows the slots the job was granted
```

Or request more memory on WRDS, in `run_stage1.sh`. `m_mem_free` is charged per slot and WRDS
allows 48 GB per job, so with its 4 slots the most is `-l m_mem_free=12G` (48 GB). A request over
the limit waits in the queue forever, without an error.

### Performance Issues

**Pipeline is very slow**

Solutions:
1. Reports and figures cannot be disabled -- see Output Settings above. The knobs that
   do exist are the chunk count and the worker count.

2. Raise `N_CHUNKS` to lower peak memory (the default is 10; a SMALLER number makes each
   chunk bigger, which is the opposite of what you want under memory pressure).

3. Leave `N_CORES` alone unless you know better. It defaults to `None`, which resolves
   from `$NSLOTS` -- the slots Grid Engine actually granted, currently 4. Setting it
   explicitly skips that resolution entirely, so `N_CORES = 20` would start 20 joblib
   workers inside a 4-slot, 40 GB allocation:
   ```python
   N_CORES = None   # recommended: follow the grant
   ```

4. Process fewer TRACE datasets, in the root `config.py`:
   ```python
   TRACE_MEMBERS = ["enhanced"]  # Only process Enhanced
   ```

---

## Performance Optimization

### Parallel Processing

Stage 1 uses `joblib` for parallel processing of bond analytics. Tune these settings:

```python
N_CORES  = None   # default: resolve from $NSLOTS, the slots Grid Engine granted
N_CHUNKS = 10     # default
```

**Guidelines:**
- **WRDS Cloud**: leave `N_CORES = None`. `run_stage1.sh` requests `-pe onenode 4`, and
  the default resolution reads `$NSLOTS` and matches it. ❗Setting `N_CORES` explicitly
  SKIPS that resolution, so `N_CORES = 20` would start 20 joblib workers inside a
  4-slot, 40 GB allocation. `qstat -F` reports the node's cores, not your grant.
- **Local machine**: physical cores minus two is a reasonable choice.
- **Memory issues**: **raise** `N_CHUNKS` above 10 — more chunks means each one is
  smaller. Lowering it makes the problem worse.

### Output Format

```python
OUTPUT_FORMAT = "parquet"  # The only supported value: Stage 0 refuses any other; Stage 1 always writes Parquet
```
Convert after the fact if you need CSV -- see the [FAQ](../FAQ.md).

### Report Generation

Stage 1 always writes its LaTeX report, its tables and its time-series figures. This is
deliberate and there is no switch -- `_stage1_settings.py` calls them *"essential for
data quality assessment"*. (Stage 0's figures **can** be turned off, with
`STAGE0_OUTPUT_FIGURES` in the root `config.py`; that is a different stage.)

---

## License & Citation

### License

This code is provided under the MIT License. See LICENSE file for details.

### Citation

If you use this stage in your research, please cite:

**Primary Reference:**
```
Dickerson, A., Robotti, C., & Rossetti, G. (2026).
The Corporate Bond Factor Replication Crisis.
Working Paper. (Earlier versions circulated as "Common pitfalls in the
evaluation of corporate bond strategies.")
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

Every release, with what it changed, is in [CHANGELOG.md](../CHANGELOG.md). Stage 1 first shipped
in 2.0.0 (2025-12-11) with FISD characteristics, QuantLib analytics, S&P and Moody's ratings,
equity identifiers, the ultra-distressed filters and the data report.

---

**Last updated:** September 2026

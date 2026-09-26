# Quick Start (Stage 1) — Bond Analytics

## Prerequisites

- ✅ **Stage 0 completed** with outputs in `stage0/{enhanced,standard,144a}/`
- ✅ **SSH access to WRDS Cloud** (or local Python environment)
- ✅ **WRDS account** with FISD and ratings data access
- ✅ **Python ≥ 3.10** (the 2026-09-21 run, which the 2026 vintage was built from, used 3.14.5 on the WRDS Cloud)
- ✅ **`.pgpass`** configured for passwordless WRDS authentication

---

## What Stage 1 Does

Stage 1 enriches your cleaned TRACE data from Stage 0 with:

- 📊 **Bond characteristics** from FISD (coupon, maturity, issuer, etc.)
- 📈 **Bond analytics** via QuantLib (duration, convexity, YTM, credit spreads)
- ⭐ **Credit ratings** from S&P and Moody's
- 🔗 **Equity identifiers** (CRSP PERMNO/PERMCO and GVKEY)
- 🚨 **Ultra-distressed filters** to flag suspicious prices
- 🏭 **Fama-French industry classifications**

**Output:** A research-ready dataset of 44 columns per bond-day.

**Runtime:** about 2.5-2.7 hours on the WRDS Cloud with the 4 slots `run_stage1.sh` requests (2.4 h on the 2026-09-10 run, 2.7 h on 2026-09-21).

---

## Step-by-Step Instructions

### 1. Install Required Packages

**On WRDS Cloud:**

```bash
ssh <your_wrds_id>@wrds-cloud.wharton.upenn.edu
cd ~/trace-data-pipeline  # the repo root
```

**Install packages** (recommended method):

```bash
python -m pip install --user -r requirements.txt
```

**Alternative — Virtual environment (optional):**
```bash
# Create virtual environment in project root. --system-site-packages keeps the WRDS Cloud's own pandas 2.2 visible: pip has no build of it
# for the Cloud's Python 3.14.
python3 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

**Manual installation** (only if `requirements.txt` is unavailable):
```bash
python -m pip install --user pandas numpy wrds pyarrow tqdm QuantLib joblib openpyxl requests matplotlib
```

❗Do **not** pin these to exact versions on the WRDS Cloud. `--user` installs shadow the
system packages, and WRDS already ships newer ones than any pin here (numpy 2.4.6,
pyarrow 24.0.0 on Python 3.14) — pinning downgrades a working environment. Use
`requirements.txt`, which states minimums.

---

### 2. Configure Settings (Optional)

**Most settings are pre-configured and work automatically:**
- ✅ **WRDS_USERNAME** is set in `config.py` (root directory)
- ✅ **STAGE0_DATE_STAMP** is auto-detected from your Stage 0 output files
- ✅ **Data downloads** happen automatically via `run_pipeline.sh`

**Only edit settings if you need to customize:**

**Navigate to your root directory:**
```bash
cd ~/trace-data-pipeline  # the repo root directory
```

**Edit root config file (if needed):**
```bash
nano config.py
```

**Optional customizations in `config.py`:**
```python
# Your WRDS username
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")

# Which TRACE datasets to include
TRACE_MEMBERS = os.getenv("TRACE_MEMBERS", "enhanced 144a").split()  # add "standard" to opt in

# Output format
OUTPUT_FORMAT = "parquet"  # The only supported value
```

**Edit stage1 settings (if needed):**
```bash
nano stage1/_stage1_settings.py
```

**Optional customizations in `stage1/_stage1_settings.py`:**
```python
# Date cutoff. The default rolls with the data: "auto:complete" is the last month
# every source covers through its final trading session. A literal date overrides it.
DATE_CUT_OFF = "auto:complete"

# Parallel processing (adjust based on your machine)
N_CORES = None  # follows the slots the job was granted ($NSLOTS), else 4; or set STAGE1_N_CORES
```

**Notes:**
- `ROOT_PATH` is auto-detected from your current working directory
- `STAGE0_DATE_STAMP` is auto-detected from Stage 0 parquet files
- Data files are automatically downloaded by `run_pipeline.sh`

---

### 3. Run the Pipeline

**The easiest way is to use the automated pipeline wrapper:**

```bash
# From your root directory
cd ~/trace-data-pipeline  # the repo root

# Run the complete pipeline (downloads data + runs Stage 0 + Stage 1)
./run_pipeline.sh
```

**This automatically:**
1. Downloads required data files (Liu-Wu yields, bond-firm linker, FF industries)
2. Submits Stage 0 jobs (Enhanced and 144A by default; Standard is opt-in)
3. Submits Stage 1 job (waits for Stage 0 to complete)

**Manual Stage 1 execution (if you already ran Stage 0):**

```bash
# You should be in your root directory
pwd  # Verify your current location

# Fetch the five input files first (login node: compute nodes have no internet)
bash download_inputs.sh

# Submit the job
qsub stage1/run_stage1.sh
```

Stage 1 checks for those files before it starts and stops if one is missing.

❗**Submit from the repo ROOT, never from `stage1/`.** `run_stage1.sh` declares
`#$ -o stage1/logs/stage1.out` and then does `cd stage1`, all resolved against `-cwd`.
Submitted from inside `stage1/`, the log path becomes `stage1/stage1/logs/` — which does
not exist, so Grid Engine parks the job in `Eqw` — and the `cd` fails too.

**What happens:**
1. Loads treasury yields (Liu-Wu zero-coupon curve)
2. Loads TRACE data from Stage 0 outputs
3. Reads the FISD bond characteristics Stage 0 saved
4. Merges FISD with TRACE
5. Computes bond analytics (duration, convexity, YTM, credit spreads) using QuantLib
6. Connects to WRDS and merges S&P and Moody's credit ratings
7. Merges bond-firm linker (equity identifiers)
8. Flags probable price errors (the ultra-distressed filters)
9. Flags prices above 300% of par and large first price changes in July 2002
10. Removes the flagged rows, rows with no rating and rows with under a year to maturity,
    winsorizes `ytm` and `credit_spread` within each date, saves the file and builds the
    data-quality report

---

### 4. Monitor Progress

**On WRDS Cloud:**

```bash
# Check job status
qstat

# Follow output log (from root directory)
tail -f stage1/logs/stage1.out

# Or if you're in stage1 directory:
tail -f logs/stage1.out

# Check for errors
tail -f stage1/logs/stage1.err  # from root
# or
tail -f logs/stage1.err  # from stage1
```

**Job states:**
- `r` = running
- `qw` = queued, waiting
- `Eqw` = error (check logs/stage1.err)

**Stop tail:** Press `Ctrl + C`

**On local machine:**

Output will print to your terminal in real-time.

---

### 5. Check Output

**Output location (from root directory):**
```bash
ls -lh stage1/data/stage1_*.parquet

# Or from stage1 directory:
cd stage1
ls -lh data/stage1_*.parquet
```

**Expected output:**
```
stage1/
├── data/
│   ├── stage1_YYYYMMDD.parquet   # Main enriched dataset (~2.7 GB for the full sample)
│   └── data_reports/             # The ultra-distressed filter's figure pages
└── data_reports/                 # The Stage 1 data report
    ├── stage1_data_report_YYYYMMDD.tex
    ├── references.bib
    └── stage1_*.pdf              # its figures
```

**Verify data:**

```python
import pandas as pd

# Load the data (from stage1/)
df = pd.read_parquet('data/stage1_YYYYMMDD.parquet')  # Use your date

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check key variables
print(f"\nKey variables available:")
print(f"- Identifiers: cusip_id, permno, permco, gvkey")
print(f"- TRACE prices: pr, prc_ew, prc_vw_par, prc_hi, prc_lo")
print(f"- Bond analytics: ytm, mod_dur, convexity, credit_spread")
print(f"- Ratings: sp_rating, mdy_rating, spc_rating, mdc_rating")
```

---

### 6. Computing Returns

Once you have the Stage 1 output, you can compute bond returns for empirical analysis.

#### Setup and Sorting

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet('data/stage1_YYYYMMDD.parquet')

# Sort by bond and date (required for lagged calculations)
df = df.sort_values(['cusip_id', 'trd_exctn_dt']).reset_index(drop=True)
```

#### Clean Returns (Price Appreciation Only)

```python
# Compute lagged price
df['pr_lag'] = df.groupby('cusip_id', observed=True)['pr'].shift(1)

# Clean return: (pr_t - pr_{t-1}) / pr_{t-1}
df['ret_c'] = (df['pr'] - df['pr_lag']) / df['pr_lag']
```

#### Total Returns (Including Accumulated Payments)

```python
# Full price = clean price + accumulated payments (numerator)
df['fp'] = df['pr'] + df['accall']

# Compute lagged values
df['fp_lag'] = df.groupby('cusip_id', observed=True)['fp'].shift(1)
df['prfull_lag'] = df.groupby('cusip_id', observed=True)['prfull'].shift(1)

# Total return: (fp_t - fp_{t-1}) / prfull_{t-1}
df['ret_d'] = (df['fp'] - df['fp_lag']) / df['prfull_lag']
```

#### Filtering by Trading Gap

**Important**: TRACE data is NOT contiguous—bonds may not trade daily. You should filter out returns with large gaps between observations.

```python
# Compute lagged trade date
df['prev_trd_dt'] = df.groupby('cusip_id', observed=True)['trd_exctn_dt'].shift(1)

# Simple calendar day gap (for quick filtering)
df['day_gap'] = (df['trd_exctn_dt'] - df['prev_trd_dt']).dt.days

# Set returns to NaN where gap > threshold (e.g., 7 calendar days)
max_gap = 7
gap_mask = df['day_gap'] > max_gap
df.loc[gap_mask, ['ret_c', 'ret_d']] = np.nan
```

For **business-day gaps** (excluding weekends and holidays), count NYSE sessions between the two
dates, for example with `numpy.busday_count` and a holiday list from `pandas_market_calendars`.

#### Key Variables

| Variable | Formula | Description |
|----------|---------|-------------|
| `pr` | — | Clean price (% of par) |
| `prfull` | `pr + acclast` | Dirty price (standardization base) |
| `fp` | `pr + accall` | Full price with accumulated payments |
| `ret_c` | `(pr_t - pr_{t-1}) / pr_{t-1}` | Clean return (price appreciation only) |
| `ret_d` | `(fp_t - fp_{t-1}) / prfull_{t-1}` | Total return (includes cash flows) |

**Key Distinction:**
- `accall` = accumulated payments (includes cash flows) → **NUMERATOR**
- `acclast` = accrued interest (time-accrued component) → **DENOMINATOR**
- `prfull = pr + acclast` = dirty price (standardization base)

For detailed explanations of the accrued interest variables (`acclast`, `accpmt`, `accall`) and the return calculation methodology, see the **"Understanding Accrued Interest Variables"** section in [README_stage1.md](README_stage1.md).

---

### 7. Download Data (WRDS Cloud Users)

**Windows users (WinSCP):**
- Connect to WRDS Cloud via WinSCP
- Navigate to `~/trace-data-pipeline/stage1/data/`
- Download `stage1_YYYYMMDD.parquet` to your local machine

**Mac/Linux users (scp):**

```bash
# From your LOCAL machine, run:
scp -r <wrds_id>@wrds-cloud.wharton.upenn.edu:~/trace-data-pipeline/stage1/data ./local_destination/
```

---

## Configuration Quick Reference

### Essential Settings

Most settings are automatically configured. Only customize if needed.

#### Root Configuration (`config.py`)

| Setting | Description | Default | Auto? |
|---------|-------------|---------|-------|
| `WRDS_USERNAME` | Your WRDS username | `"your_wrds_username"` | ✅ Pre-set |
| `TRACE_MEMBERS` | TRACE datasets to include | `["enhanced", "144a"]` | Customizable; `"standard"` is opt-in |
| `OUTPUT_FORMAT` | Output file format | `"parquet"` | ✅ Pre-set |

#### Stage 1 Configuration (`stage1/_stage1_settings.py`)

| Setting | Description | Default | Auto? |
|---------|-------------|---------|-------|
| `ROOT_PATH` | Parent directory | `""` | ✅ Auto-detected |
| `STAGE0_DATE_STAMP` | Stage 0 output date | Auto-detected | ✅ Auto-detected from files |
| `DATE_CUT_OFF` | Latest date to include | `"auto:complete"` | ✅ Rolls with the data |
| `N_CORES` | Worker processes | `None`: the slots granted, else 4 | ✅ Follows the job's slots |
| ~~`GENERATE_REPORTS`~~ | does not exist -- reports always run | — | — |
| ~~`OUTPUT_FIGURES`~~ | does not exist -- figures always run | — | — |

**Notes:**
- **WRDS_USERNAME**: Set in `config.py` (root directory)
- **ROOT_PATH**: Auto-detected from current working directory
- **STAGE0_DATE_STAMP**: Auto-detected from Stage 0 parquet filenames
- **Data files**: Automatically downloaded by `run_pipeline.sh`

---

## Troubleshooting

### "Stage0 output files not found"

**Problem:** Can't find `stage0/enhanced/trace_enhanced_YYYYMMDD.parquet`

**Solution:**
1. Verify Stage 0 outputs exist (from root directory):
   ```bash
   ls stage0/enhanced/trace_enhanced_*.parquet
   ls stage0/standard/trace_standard_*.parquet
   ls stage0/144a/trace_144a_*.parquet
   ```
2. If files exist, the date stamp should auto-detect. If auto-detection fails, check the error message
3. Ensure you're running from the correct root directory (where `stage0/` and `stage1/` exist)

---

### "Unable to read script file"

**Problem:** `qsub stage1/run_stage1.sh` says "No such file or directory"

**Solution:** You're in the wrong directory. Either:

**Option 1 - Run from root directory:**
```bash
cd ~/trace-data-pipeline          # the repo ROOT
qsub stage1/run_stage1.sh
```

❗There is no "navigate into stage1" alternative. The script's `#$ -o stage1/logs/...`
and its `cd stage1` are both resolved from the submit directory, so it only works from
the root. `run_pipeline.sh` submits it exactly this way.

---

### "ModuleNotFoundError: No module named 'QuantLib'"

**Problem:** QuantLib not installed

**Solution:**
```bash
python -m pip install --user QuantLib
```

`requirements.txt` asks for `QuantLib>=1.36`; do not pin an exact version.

---

### "WRDS_USERNAME not set"

**Problem:** WRDS username not configured

**Solution:** Edit `config.py` (in the root directory):
```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")  # Change to your username
```

Or set environment variable:
```bash
export WRDS_USERNAME="your_wrds_username_here"
echo 'export WRDS_USERNAME="your_wrds_username_here"' >> ~/.bashrc
```

---

### "Unable to connect to WRDS"

**Problem:** WRDS connection failed

**Solution:** Check `.pgpass` file:
```bash
chmod 600 ~/.pgpass
cat ~/.pgpass
```

Should contain:
```
wrds-pgdata.wharton.upenn.edu:9737:wrds:your_username:your_password
```

---

### Job is very slow

**Problem:** Pipeline taking >8 hours

**Solutions:**

1. **Process fewer datasets** (in the root `config.py`, or `TRACE_MEMBERS="enhanced"` in the environment):
   ```python
   TRACE_MEMBERS = ["enhanced"]  # Only Enhanced, not Standard/144A
   ```

   There is no switch for Stage 1's reports; they always run.

2. **On your own machine only, use more workers** (`export STAGE1_N_CORES=20`). On WRDS leave
   `N_CORES = None`: it follows the 4 slots the job was granted, and a larger number starts more
   workers than the job has cores and memory for.

3. **Check you're not in the middle of a WRDS outage:**
   ```bash
   # Test WRDS connection
   python -c "import wrds; db = wrds.Connection(); print('Connected OK')"
   ```

---

## Quick Command Reference

```bash
# Configure settings (from root directory)
nano stage1/_stage1_settings.py

# Make executable (older clones only; the script ships executable)
chmod +x stage1/run_stage1.sh

# Submit job (WRDS)
qsub stage1/run_stage1.sh

# Or run locally
bash stage1/run_stage1.sh

# Monitor job (WRDS)
qstat
tail -f stage1/logs/stage1.out
tail -f stage1/logs/stage1.err

# Check output
ls -lh stage1/data/stage1_*.parquet

# Download from WRDS (Mac/Linux, run from LOCAL machine)
scp -r <wrds_id>@wrds-cloud.wharton.upenn.edu:~/trace-data-pipeline/stage1/data ./local_destination/
```

---

## Support

Having trouble? Check:
1. **Detailed README:** See [README_stage1.md](README_stage1.md)
2. **Log files:** Check `stage1/logs/stage1.err` for error messages
3. **Email:** alexander.dickerson1@unsw.edu.au
4. **GitHub Issues:** [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)

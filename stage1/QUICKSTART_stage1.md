# Quick Start (Stage 1) — Bond Analytics

## Prerequisites

- ✅ **Stage 0 completed** with outputs in `stage0/{enhanced,standard,144a}/`
- ✅ **SSH access to WRDS Cloud** (or local Python environment)
- ✅ **WRDS account** with FISD and ratings data access
- ✅ **Python ≥ 3.10** (tested with Python 3.12.11)
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

**Output:** A comprehensive research-ready dataset with ~50+ variables per bond-day.

**Runtime:** ~3-4 hours with 2 cores (4 cores with threading) on WRDS cloud. Potentially quicker on your home machine.

---

## Step-by-Step Instructions

### 1. Install Required Packages

**On WRDS Cloud:**

```bash
ssh <your_wrds_id>@wrds-cloud.wharton.upenn.edu
cd ~/proj  # Navigate to your project root
```

**Install packages** (recommended method):

```bash
python -m pip install --user -r requirements.txt
```

**Alternative — Virtual environment (optional):**
```bash
# Create virtual environment in project root
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

**Manual installation with specific versions:**
```bash
python -m pip install --user pandas==2.2.3 numpy==2.2.5 wrds pyarrow tqdm QuantLib==1.37 joblib==1.5.1 openpyxl requests matplotlib
```

**IMPORTANT:** QuantLib 1.37 and joblib 1.5.1 are **required**.

---

### 2. Configure Settings (Optional)

**Most settings are pre-configured and work automatically:**
- ✅ **WRDS_USERNAME** is set in `config.py` (root directory)
- ✅ **STAGE0_DATE_STAMP** is auto-detected from your Stage 0 output files
- ✅ **Data downloads** happen automatically via `run_pipeline.sh`

**Only edit settings if you need to customize:**

**Navigate to your root directory:**
```bash
cd ~/proj  # Navigate to your project root directory
```

**Edit root config file (if needed):**
```bash
nano config.py
```

**Optional customizations in `config.py`:**
```python
# Your WRDS username
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")

# Output format
OUTPUT_FORMAT = "parquet"  # Options: "parquet" (recommended), "csv"
```

**Edit stage1 settings (if needed):**
```bash
nano stage1/_stage1_settings.py
```

**Optional customizations in `stage1/_stage1_settings.py`:**
```python
# Which TRACE datasets to include
TRACE_MEMBERS = ["enhanced", "standard", "144a"]  # Or just ["enhanced"]

# Date cutoff
DATE_CUT_OFF = "2025-03-31"  # Only include data through this date

# Parallel processing (adjust based on your machine)
N_CORES = None  # Auto-detects available cores
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
cd ~/proj  # Navigate to your project root

# Run the complete pipeline (downloads data + runs Stage 0 + Stage 1)
./run_pipeline.sh
```

**This automatically:**
1. Downloads required data files (Liu-Wu yields, OSBAP linker, FF industries)
2. Submits Stage 0 jobs (Enhanced, Standard, 144A TRACE extraction)
3. Submits Stage 1 job (waits for Stage 0 to complete)

**Manual Stage 1 execution (if you already ran Stage 0):**

```bash
# You should be in your root directory
pwd  # Verify your current location

# Submit the job
qsub stage1/run_stage1.sh
```

**Option B - Run from stage1 directory:**

```bash
# Navigate to stage1 directory
cd stage1

# Submit the job
qsub run_stage1.sh
```

**What happens:**
1. Loads treasury yields (Liu-Wu zero-coupon curve)
2. Loads TRACE data from Stage 0 outputs
3. Fetches FISD bond characteristics from WRDS
4. Merges FISD with TRACE
5. Computes bond analytics (duration, convexity, YTM, credit spreads) using QuantLib
6. Merges S&P and Moody's credit ratings
7. Merges OSBAP linker (equity identifiers)
8. Applies ultra-distressed bond filters
9. Applies final filters (price > 300%, July 2002 anomaly)
10. Generates data quality reports

---

### 6. Monitor Progress

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

### 7. Check Output

**Output location (from root directory):**
```bash
ls -lh stage1/data/stage1_*.parquet

# Or from stage1 directory:
cd stage1
ls -lh data/stage1_*.parquet
```

**Expected output:**
```
data/
├── stage1_YYYYMMDD.parquet       # Main enriched dataset (~500MB-2GB)
└── reports/                      # Data quality reports (if enabled)
    ├── stage1_data_report.tex
    ├── references.bib
    └── figures/
```

**Verify data:**

```python
import pandas as pd

# Load the data
df = pd.read_parquet('data/stage1_20251117.parquet')  # Use your date

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check key variables
print(f"\nKey variables available:")
print(f"- Identifiers: cusip_id, permno, permco, gvkey")
print(f"- TRACE prices: prc_ew, prc_vw, prc_hi, prc_lo")
print(f"- Bond analytics: ytm, mod_dur, convexity, credit_spread")
print(f"- Ratings: sp_rating_num, moodys_rating_num")
print(f"- Filters: ultra_distressed_flag")
```

---

### 7. Computing Returns

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

For **business day gaps** (excluding weekends and holidays), see `stage2/illiq_helper_functions.py:business_days_between_vectorized()`.

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

### 8. Download Data (WRDS Cloud Users)

**Windows users (WinSCP):**
- Connect to WRDS Cloud via WinSCP
- Navigate to `~/proj/stage1/data/`
- Download `stage1_YYYYMMDD.parquet` to your local machine

**Mac/Linux users (scp):**

```bash
# From your LOCAL machine, run:
scp -r <wrds_id>@wrds-cloud.wharton.upenn.edu:~/proj/stage1/data ./local_destination/
```

---

## Configuration Quick Reference

### Essential Settings

Most settings are automatically configured. Only customize if needed.

#### Root Configuration (`config.py`)

| Setting | Description | Default | Auto? |
|---------|-------------|---------|-------|
| `WRDS_USERNAME` | Your WRDS username | `"your_wrds_username"` | ✅ Pre-set |
| `OUTPUT_FORMAT` | Output file format | `"parquet"` | ✅ Pre-set |

#### Stage 1 Configuration (`stage1/_stage1_settings.py`)

| Setting | Description | Default | Auto? |
|---------|-------------|---------|-------|
| `ROOT_PATH` | Parent directory | `""` | ✅ Auto-detected |
| `STAGE0_DATE_STAMP` | Stage 0 output date | Auto-detected | ✅ Auto-detected from files |
| `TRACE_MEMBERS` | TRACE datasets to include | `["enhanced", "standard", "144a"]` | Customizable |
| `DATE_CUT_OFF` | Latest date to include | `"2025-03-31"` | Customizable |
| `N_CORES` | CPU cores for parallel processing | Auto-detected | ✅ Auto-detected |
| `GENERATE_REPORTS` | Create LaTeX reports | `True` | Customizable |
| `OUTPUT_FIGURES` | Create time-series figures | `True` | Customizable |

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

**Problem:** `qsub run_stage1.sh` says "No such file or directory"

**Solution:** You're in the wrong directory. Either:

**Option 1 - Run from root directory:**
```bash
cd ~/proj  # Your root directory
qsub stage1/run_stage1.sh  # Specify full path
```

**Option 2 - Navigate to stage1:**
```bash
cd ~/proj/stage1
qsub run_stage1.sh
```

---

### "ModuleNotFoundError: No module named 'QuantLib'"

**Problem:** QuantLib not installed or wrong version

**Solution:**
```bash
python -m pip install --user QuantLib==1.37
```

**IMPORTANT:** You must install QuantLib version 1.37 specifically.

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

1. **Disable reports:**
   ```python
   # In _stage1_settings.py
   GENERATE_REPORTS = False
   ```

2. **Process fewer datasets:**
   ```python
   TRACE_MEMBERS = ["enhanced"]  # Only Enhanced, not Standard/144A
   ```

3. **Use more CPU cores:**
   ```python
   N_CORES = 20  # If available on your machine
   ```

4. **Check you're not in the middle of a WRDS outage:**
   ```bash
   # Test WRDS connection
   python -c "import wrds; db = wrds.Connection(); print('Connected OK')"
   ```

---

## Quick Command Reference

```bash
# Configure settings (from root directory)
nano stage1/_stage1_settings.py

# Make executable
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
scp -r <wrds_id>@wrds-cloud.wharton.upenn.edu:~/proj/stage1/data ./local_destination/
```

---

## Support

Having trouble? Check:
1. **Detailed README:** See [README_stage1.md](README_stage1.md)
2. **Log files:** Check `logs/stage1.err` for error messages
3. **Email:** alexander.dickerson1@unsw.edu.au
4. **GitHub Issues:** [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)

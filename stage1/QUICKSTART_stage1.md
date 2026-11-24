# Quick start (Stage 1): bond analytics

## Prerequisites

- Stage 0 has completed, with its outputs in `stage0/{enhanced,standard,144a}/`
- SSH access to the WRDS Cloud (or a local Python environment)
- A WRDS account with access to FISD and ratings data
- Python 3.10 or later (the 2026-09-21 run, which the 2026 vintage was built from, used 3.14.5 on the WRDS Cloud)
- No password file is needed on WRDS, because jobs on the WRDS Cloud connect without one

## What Stage 1 does

Stage 1 adds these to your cleaned TRACE data from Stage 0.

- Bond characteristics from FISD (coupon, maturity, issuer, etc.)
- Bond analytics from QuantLib (duration, convexity, YTM, credit spreads)
- Credit ratings from S&P and Moody's
- Equity identifiers (CRSP PERMNO/PERMCO and GVKEY)
- Ultra-distressed filters that flag suspicious prices
- Fama-French industry classifications

The output is a dataset of 44 columns per bond-day.

The run takes about 2.4 to 2.7 hours on the WRDS Cloud with the 4 slots `run_stage1.sh` requests (2.4 h on the 2026-09-10 run, 2.7 h on 2026-09-21).

## Step-by-step instructions

### 1. Install required packages

On the WRDS Cloud, log in and go to the repo root.

```bash
ssh <your_wrds_id>@wrds-cloud.wharton.upenn.edu
cd ~/trace-data-pipeline  # the repo root
```

Install the packages. This is the recommended method.

```bash
python -m pip install --user -r requirements.txt
```

A virtual environment is an optional alternative.

```bash
# Create virtual environment in project root. --system-site-packages keeps the WRDS Cloud's own pandas 2.2 visible: pip has no build of it
# for the Cloud's Python 3.14.
python3 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

**Do not pin these to exact versions** on the WRDS Cloud. `--user` installs shadow the
system packages, and WRDS already ships newer ones than any pin here (numpy 2.4.6,
pyarrow 24.0.0 on Python 3.14), so pinning downgrades a working environment. Use
`requirements.txt`, which states minimums.

### 2. Configure settings (optional)

Most settings are already configured and need no change.

- WRDS_USERNAME is set in `config.py` (root directory)
- STAGE0_DATE_STAMP is auto-detected from your Stage 0 output files
- Data downloads happen automatically via `run_pipeline.sh`

Edit the settings only if you need to customize them. First go to your root directory.

```bash
cd ~/trace-data-pipeline  # the repo root directory
```

Edit the root config file if you need to.

```bash
nano config.py
```

These are the optional customizations in `config.py`.

```python
# Your WRDS username
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")

# Which TRACE datasets to include
TRACE_MEMBERS = os.getenv("TRACE_MEMBERS", "enhanced 144a").split()  # add "standard" to opt in

# Output format
OUTPUT_FORMAT = "parquet"  # The only supported value
```

Edit the stage 1 settings if you need to.

```bash
nano stage1/_stage1_settings.py
```

These are the optional customizations in `stage1/_stage1_settings.py`.

```python
# Date cutoff. The default rolls with the data: "auto:complete" is the last month
# every source covers through its final trading session. A literal date overrides it.
DATE_CUT_OFF = "auto:complete"

# Parallel processing (adjust based on your machine)
N_CORES = None  # follows the slots the job was granted ($NSLOTS), else 4; or set STAGE1_N_CORES
```

You do not set any of these by hand.

- `ROOT_PATH` is auto-detected from your current working directory
- `STAGE0_DATE_STAMP` is auto-detected from Stage 0 parquet files
- Data files are automatically downloaded by `run_pipeline.sh`

### 3. Run the pipeline

The easiest way is the automated pipeline wrapper.

```bash
# From your root directory
cd ~/trace-data-pipeline  # the repo root

# Run the complete pipeline (downloads data + runs Stage 0 + Stage 1)
./run_pipeline.sh
```

The wrapper does these steps for you.

1. Downloads the required data files (Liu-Wu yields, bond-firm linker, FF industries)
2. Submits the Stage 0 jobs (Enhanced and 144A by default, Standard only if you opt in)
3. Submits the Stage 1 job, which waits for Stage 0 to complete

If you already ran Stage 0, you can run Stage 1 by hand.

```bash
# You should be in your root directory
pwd  # Verify your current location

# Fetch the five input files first (login node: compute nodes have no internet)
bash download_inputs.sh

# Submit the job
qsub stage1/run_stage1.sh
```

Stage 1 checks for those files before it starts and stops if one is missing.

Submit **from the repo root**, never from `stage1/`. `run_stage1.sh` declares
`#$ -o stage1/logs/stage1.out` and then does `cd stage1`, all resolved against `-cwd`.
Submitted from inside `stage1/`, the log path becomes `stage1/stage1/logs/`, which does
not exist, so Grid Engine parks the job in `Eqw`. The `cd` fails too.

The job then runs these steps.

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

### 4. Monitor progress

On the WRDS Cloud, watch the job with these commands.

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

A job is in one of these states.

- `r` = running
- `qw` = queued, waiting
- `Eqw` = error (check logs/stage1.err)

Press `Ctrl + C` to stop following a log.

On a local machine, the output prints to your terminal as the job runs.

### 5. Check output

From the root directory, list the output file.

```bash
ls -lh stage1/data/stage1_*.parquet

# Or from stage1 directory:
cd stage1
ls -lh data/stage1_*.parquet
```

Expect this layout.

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

To verify the data, run the lines below. They read only the file's metadata, so they are safe
on the WRDS login node, where heavy work is not allowed.

```python
import pyarrow.parquet as pq

# From stage1/
f = pq.ParquetFile('data/stage1_YYYYMMDD.parquet')  # your date
print(f"{f.metadata.num_rows:,} rows, {len(f.schema_arrow.names)} columns")
print(f.schema_arrow.names)

# Check key variables
print(f"\nKey variables available:")
print(f"- Identifiers: cusip_id, permno, permco, gvkey")
print(f"- TRACE prices: pr, prc_ew, prc_vw_par, prc_hi, prc_lo")
print(f"- Bond analytics: ytm, mod_dur, convexity, credit_spread")
print(f"- Ratings: sp_rating, mdy_rating, spc_rating, mdc_rating")
```

### 6. Computing returns

Once you have the Stage 1 output, you can compute bond returns for empirical analysis.

#### Setup and sorting

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet('data/stage1_YYYYMMDD.parquet')

# Sort by bond and date (required for lagged calculations)
df = df.sort_values(['cusip_id', 'trd_exctn_dt']).reset_index(drop=True)
```

#### Clean returns (price appreciation only)

```python
# Compute lagged price
df['pr_lag'] = df.groupby('cusip_id', observed=True)['pr'].shift(1)

# Clean return: (pr_t - pr_{t-1}) / pr_{t-1}
df['ret_c'] = (df['pr'] - df['pr_lag']) / df['pr_lag']
```

#### Total returns (including accumulated payments)

```python
# Full price = clean price + accumulated payments (numerator)
df['fp'] = df['pr'] + df['accall']

# Compute lagged values
df['fp_lag'] = df.groupby('cusip_id', observed=True)['fp'].shift(1)
df['prfull_lag'] = df.groupby('cusip_id', observed=True)['prfull'].shift(1)

# Total return: (fp_t - fp_{t-1}) / prfull_{t-1}
df['ret_d'] = (df['fp'] - df['fp_lag']) / df['prfull_lag']
```

#### Filtering by trading gap

TRACE data is not contiguous, because a bond may not trade every day. Filter out returns with large gaps between observations.

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

For business-day gaps (excluding weekends and holidays), count NYSE sessions between the two
dates, for example with `numpy.busday_count` and a holiday list from `pandas_market_calendars`.

#### Key variables

| Variable | Formula | Description |
|----------|---------|-------------|
| `pr` | | Clean price (% of par) |
| `prfull` | `pr + acclast` | Dirty price (standardization base) |
| `fp` | `pr + accall` | Full price with accumulated payments |
| `ret_c` | `(pr_t - pr_{t-1}) / pr_{t-1}` | Clean return (price appreciation only) |
| `ret_d` | `(fp_t - fp_{t-1}) / prfull_{t-1}` | Total return (includes cash flows) |

Keep the two accrual variables apart.

- `accall` is the accumulated payments (it includes cash flows) and goes in the numerator
- `acclast` is the accrued interest (the time-accrued component) and goes in the denominator
- `prfull = pr + acclast` is the dirty price (the standardization base)

The "Understanding Accrued Interest Variables" section in [README_stage1.md](README_stage1.md) explains the accrued interest variables (`acclast`, `accpmt`, `accall`) and how the returns are calculated.

### 7. Download data (WRDS Cloud users)

Zip the whole repository folder on WRDS and copy the zip down, as the
[Download Results](../QUICKSTART.md#download-results-to-your-local-machine) section of the
root QUICKSTART describes. Stage 2 reads `stage1/data/` and `stage0/enhanced/` from the same
folder, so bring both, in the layout they were built in. The stage 1 parquet alone is not
enough.

## Configuration quick reference

### Essential settings

Most settings are configured automatically. Customize only if you need to.

#### Root configuration (`config.py`)

| Setting | Description | Default | Auto? |
|---------|-------------|---------|-------|
| `WRDS_USERNAME` | Your WRDS username | `"your_wrds_username"` | Pre-set |
| `TRACE_MEMBERS` | TRACE datasets to include | `["enhanced", "144a"]` | Customizable, and `"standard"` is opt-in |
| `OUTPUT_FORMAT` | Output file format | `"parquet"` | Pre-set |

#### Stage 1 configuration (`stage1/_stage1_settings.py`)

| Setting | Description | Default | Auto? |
|---------|-------------|---------|-------|
| `ROOT_PATH` | Parent directory | `""` | Auto-detected |
| `STAGE0_DATE_STAMP` | Stage 0 output date | Auto-detected | Auto-detected from files |
| `DATE_CUT_OFF` | Latest date to include | `"auto:complete"` | Rolls with the data |
| `N_CORES` | Worker processes | `None`, which means the slots granted, else 4 | Follows the job's slots |
| ~~`GENERATE_REPORTS`~~ | does not exist. Reports always run | | |
| ~~`OUTPUT_FIGURES`~~ | does not exist. Figures always run | | |

- WRDS_USERNAME is set in `config.py` (root directory)
- ROOT_PATH is auto-detected from the current working directory
- STAGE0_DATE_STAMP is auto-detected from the Stage 0 parquet filenames
- The data files are downloaded automatically by `run_pipeline.sh`

## Troubleshooting

### "Stage0 output files not found"

The run cannot find `stage0/enhanced/trace_enhanced_YYYYMMDD.parquet`.

1. Check that the Stage 0 outputs exist (from the root directory).
   ```bash
   ls stage0/enhanced/trace_enhanced_*.parquet
   ls stage0/standard/trace_standard_*.parquet
   ls stage0/144a/trace_144a_*.parquet
   ```
2. If the files exist, the date stamp should auto-detect. If auto-detection fails, check the error message.
3. Check that you are running from the correct root directory (where `stage0/` and `stage1/` exist).

### "Unable to read script file"

`qsub stage1/run_stage1.sh` says "No such file or directory" when you are in the wrong directory.
Run it from the root directory.

```bash
cd ~/trace-data-pipeline          # the repo ROOT
qsub stage1/run_stage1.sh
```

There is no "navigate into stage1" alternative. The script's `#$ -o stage1/logs/...`
and its `cd stage1` are both resolved from the submit directory, so it only works from
the root. `run_pipeline.sh` submits it exactly this way.

### "ModuleNotFoundError: No module named 'QuantLib'"

QuantLib is not installed. Install it.

```bash
python -m pip install --user QuantLib
```

`requirements.txt` asks for `QuantLib>=1.36`. Do not pin an exact version.

### "WRDS_USERNAME not set"

Edit `config.py` (in the root directory).

```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")  # Change to your username
```

Or set the environment variable.

```bash
export WRDS_USERNAME="your_wrds_username_here"
echo 'export WRDS_USERNAME="your_wrds_username_here"' >> ~/.bashrc
```

### "Unable to connect to WRDS"

On the WRDS Cloud no password file is needed, because jobs there connect without one. A
failed connection is almost always the username. Either `WRDS_USERNAME` is unset, or it is
still `your_wrds_username` in `config.py` (`python3 doctor.py --wrds` checks it). If the
username is right, it is WRDS's limit of 7 connections held at once
(`stage0/README_stage0.md`).

On your own computer (stage 2's first run), the password must be saved where the `wrds`
package looks. Connect once by hand. It asks for the password and offers to save it
(`~/.pgpass`, or `%APPDATA%\postgresql\pgpass.conf` on Windows).

```bash
python -c "import wrds; wrds.Connection()"
```

### Job is very slow

The pipeline is taking more than 8 hours. Try these.

1. Process fewer datasets (in the root `config.py`, or `TRACE_MEMBERS="enhanced"` in the environment).
   ```python
   TRACE_MEMBERS = ["enhanced"]  # Only Enhanced, not Standard/144A
   ```

   There is no switch for Stage 1's reports. They always run.

2. On your own machine only, use more workers (`export STAGE1_N_CORES=20`). On WRDS leave
   `N_CORES = None`. It follows the 4 slots the job was granted, and a larger number starts more
   workers than the job has cores and memory for.

3. Check that you are not in the middle of a WRDS outage. Submit the smoke test
   (`qsub run_smoke_test.sh`, from the repository root). It connects inside a job and not on
   the login node, because every WRDS connection belongs inside a job.

## Quick command reference

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

# Download from WRDS: zip the whole folder, then scp (QUICKSTART.md, "Download Results")
```

## Support

If you are stuck, try these.

1. Read the detailed guide, [README_stage1.md](README_stage1.md)
2. Check `stage1/logs/stage1.err` for error messages
3. Email alexander.dickerson1@unsw.edu.au
4. Open a GitHub issue at [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)

# Stage 0 - TRACE Daily Processing (Enhanced, Standard, 144A)

This stage fetches and cleans TRACE data (Enhanced, Standard, and Rule 144A) on the WRDS Cloud and aggregates to daily panels. Jobs are submitted to Sun Grid Engine (SGE) from a PuTTY session (Windows users); files can be moved to/from WRDS with [WinSCP](https://winscp.net/eng/download.php) (Windows users). Mac users can simply use their terminal once connected to the WRDS Cloud. Mac users might find [ForkLift](https://binarynights.com/GUI-based) useful -- allows uploads, edits, and the ability to manage WRDS Cloud files through a UI. 

Besides generating daily bond pricing panels, the code also generates highly detailed TRACE data reports which document the effect of the filters at the transaction level. It produces (potentially) hundreds of time-series plots of every bond `cusip_id` that is impacted by the decimal shift and bounce-back correctors of Dickerson, Rossetti and Robotti (2025). 

If you want to get things going quickly see **Quick start**. 
Please also see **Generating the TRACE Data Reports** for instructions on how to generate the reports.

---

## Table of Contents

- [Repo Layout](#repo-layout-key-files)
- [Python on the WRDS Cloud](#python-on-the-wrds-cloud)
- [Getting the Code onto WRDS](#getting-the-code-onto-wrds)
- [Requirements](#requirements)
- [Quick Start](#quick-start-wrds-putty)
- [Running Jobs](#running-jobs-individually-alternative)
- [What the Runners Do](#what-the-runners-do)
- [Configuration](#configuration-choices-you-can-edit)
- [Outputs](#outputs)
- [Generating TRACE Data Reports](#generating-the-trace-data-reports)
- [Troubleshooting](#troubleshooting)
- [License & Citation](#license--citation)

---

## Repo layout (key files)

```
stage0/
  # Shell scripts for job submission
  run_all_trace.sh            # Submits all jobs (Enhanced, Standard, 144A)
  run_enhanced_trace.sh       # Submits Enhanced TRACE job
  run_standard_trace.sh       # Submits Standard TRACE job
  run_144a_trace.sh           # Submits Rule 144A TRACE job
  run_build_data_reports.sh   # Submits the data report generation job

  # Configuration
  _trace_settings.py          # Central configuration: filters, parameters, WRDS username

  # Python runners (called by shell scripts)
  _run_enhanced_trace.py      # Enhanced runner (calls CreateDailyEnhancedTRACE)
  _run_standard_trace.py      # Standard runner (calls CreateDailyStandardTRACE)
  _run_144a_trace.py          # 144A runner (calls CreateDailyStandardTRACE with data_type=144a)
  
  # Core processing modules
  create_daily_enhanced_trace.py # Main functions for Enhanced TRACE processing
  create_daily_standard_trace.py # Main functions for Standard/144A TRACE processing

  # Report generation
  _build_error_files.py       # Generates TRACE data quality reports
  _error_plot_helpers.py      # Helper functions for plotting and LaTeX report generation

  # Output directories (created automatically)
  logs/                       # Job logs (.out and .err files)
  data_reports/               # LaTeX reports and figures (if generated)
```

**Important:** `run_all_trace.sh` executes all three data processing jobs in parallel, then uses SGE's `-hold_jid` to automatically submit the report generation job once all three complete. This means everything runs in one go.

Each `run_*.sh` is a thin SGE wrapper that sets `-cwd` (current working directory), exports your environment (`-V`), and writes logs into `./logs/`.

---

## Python on the WRDS Cloud

Follow the WRDS guides to set yourself up on the cloud with Python.
Please read these in order:

1. [Python: On the WRDS Cloud](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-wrds-cloud/)
2. [Batch Python Jobs](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/submitting-python-programs/)
3. [Using SSH to Connect to the WRDS Cloud](https://wrds-www.wharton.upenn.edu/pages/support/the-wrds-cloud/using-ssh-connect-wrds-cloud/)
4. For Windows users who want a "point-and-click" UI for downloading data, please also see [Using SCP with Windows](https://wrds-www.wharton.upenn.edu/pages/support/the-wrds-cloud/managing-data/accessing-wrds-remotely-scp/).

This code package assumes you have:
- Access to the WRDS cloud
- Set up your `.pgpass` file for password-less authentication
- Basic familiarity with simple scripting commands in Windows/Mac
- Appropriate WRDS entitlements for TRACE Enhanced, Standard, and/or 144A data

---

## Getting the code onto WRDS

### Windows Users

From a PuTTY shell on your computer, choose one of the following:

#### Option A - Download a ZIP (no git required)
```bash
mkdir -p ~/proj && cd ~/proj
wget -O stage0.zip \
  https://github.com/Alexander-M-Dickerson/trace-data-pipeline/archive/refs/heads/main.zip
unzip stage0.zip
mv trace-data-pipeline-main/stage0 ./stage0
```

#### Option B - Using `curl` (also no git)
```bash
mkdir -p ~/proj && cd ~/proj
curl -L -o stage0.zip \
  https://github.com/Alexander-M-Dickerson/trace-data-pipeline/archive/refs/heads/main.zip
unzip stage0.zip
mv trace-data-pipeline-main/stage0 ./stage0
```

#### Option C - Clone (if `git` is available on your WRDS node)
```bash
mkdir -p ~/proj && cd ~/proj
git clone https://github.com/Alexander-M-Dickerson/trace-data-pipeline.git
cp -r trace-data-pipeline/stage0 ./stage0
```

You should now have `~/proj/stage0/` with the scripts listed above. 

**Note:** `proj` is the directory you have created on your WRDS file system - call it anything you like. Perhaps `trace` is an apt name.

---

### Linux/Mac Users

Open your terminal emulator application (*Terminal*/*iTerm2*) and connect to WRDS with SSH:

```bash
ssh wrds_username@wrds-cloud.wharton.upenn.edu
```

Once connected, you will see a WRDS prompt. From there, you can use **Option A, B, or C** as in the previous section.

#### Option D - Transferring files from your local computer

If you already have the `stage0` folder on your local computer and want to upload it after editing the scripts, you can use the secure copy protocol `scp`.

Log in to wrds-cloud with SSH and create a folder called proj:

```bash
mkdir proj
```

Transfer all the files in your local folder to the proj folder in WRDS cloud:
```bash
scp -r ~/path/to/stage0 wrds_username@wrds-cloud.wharton.upenn.edu:/home/university/wrds_username/proj/
```

Replace:
- `~/path/to/stage0` with the path to the folder on your local computer
- `/home/university/wrds_username/` with the full WRDS destination path you see when you run `pwd` after connecting to WRDS

---

## Requirements

**Python version:** 3.10 or higher (tested with Python 3.13.3)

**Required packages:**

- `pandas >= 2.2.3`
- `numpy >= 2.2.5`
- `wrds >= 3.3.0` (Python client for WRDS database access)
- `pandas_market_calendars >= 5.1.1`
- `pyarrow >= 20.0.0`
- `tqdm`

**Optional (for report generation):**
- `matplotlib >= 3.8.0`

Everything else used by the scripts is from the Python standard library (e.g., `logging`, `time`, `gc`, `functools`, `typing`, `pathlib`, `sys`).

This code was tested using the versions explicitly listed above. The log files print out your Python version and package versions for reproducibility.

### Installation

Install required packages on WRDS (if managing your own environment):
```bash
pip install --user pandas numpy wrds pandas-market-calendars pyarrow tqdm matplotlib
```

Or using the system's pip explicitly:
```bash
python -m pip install --user pandas numpy wrds pandas-market-calendars pyarrow tqdm matplotlib
```

---

## Quick start (WRDS, PuTTY)

### 1. Install required Python packages

SSH into the WRDS cloud and install required Python packages. The essential packages are `pyarrow`, `pandas_market_calendars`, `wrds`, and `tqdm`. The rest should come as standard with Python on WRDS. Extract the GitHub repo.

### 2. Navigate to the repository and configure settings

```bash
cd ~/proj/stage0
```

**CRITICAL:** Edit `_trace_settings.py` and change the `WRDS_USERNAME` variable to your WRDS username:

```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username_here")
```

To easily change this via command line by opening `nano`, follow these instructions:

```bash
nano _trace_settings.py
```
Change your username at the top, e.g., `wrds_user1` (set this to *your* username).
Save and exit `nano`, press Ctrl + O, then Enter to save. Press Ctrl + X to exit `nano`.
You can confirm the change was actioned by executing `grep WRDS_USERNAME _trace_settings.py`.

The WRDS password should be handled by the `.pgpass` file which you should have set up following the WRDS documentation.

Review the default filter settings in `_trace_settings.py`. All filters are enabled by default with recommended values from Dickerson, Rossetti and Robotti (2025). See the [Configuration](#configuration-choices-you-can-edit) section for more details.

### 3. Make scripts executable (run once)

```bash
chmod +x run_all_trace.sh run_enhanced_trace.sh run_standard_trace.sh run_144a_trace.sh run_build_data_reports.sh
```

### 4. Fix line endings (if editing on Windows)

If you edit the `.sh` scripts on Windows, your editor may save them with CRLF (Windows) endings.
SGE expects LF (Unix) endings - otherwise you'll get "bad interpreter" or `^M` errors.

To fix this once inside WRDS:
```bash
# Convert all .sh files in place
find . -name "*.sh" -exec sed -i 's/\r$//' {} \;
```

### 5. Submit all jobs (recommended)

Submit the complete automated pipeline:

```bash
./run_all_trace.sh
```

**What happens:**
1. Submits 3 jobs to run in parallel:
   - `trace_enhanced` (Enhanced TRACE)
   - `trace_standard` (Standard TRACE)
   - `trace_144a` (Rule 144A TRACE)
2. Submits a 4th job (`build_reports`) with `-hold_jid` dependency on the first three
3. The report job waits in the queue until ALL three data jobs complete
4. Once all three finish, the report generation job automatically starts
5. Reports are generated for all three datasets and saved in their respective folders

**Output from the script:**
```
[submit] Enhanced TRACE ...
[submit] Standard TRACE ...
[submit] 144A TRACE ...
[submit] Build data reports (after all TRACE jobs finish) ...
[ok] All jobs submitted with dependency. Reports will run after <job_ids> complete.
```

> **Tip:** Check status with `qstat`. The report job will show status `hqw` (hold) until the data jobs finish. Tail logs with `tail -f logs/01_enhanced.out` (or `.err`).

**Total runtime:** ~5 hours for the complete pipeline (data processing + report generation)

If you are having errors after attempting to debug, feel free to contact Alex Dickerson at `alexander.dickerson1@unsw.edu.au` for help.

---

## Running jobs individually (alternative)

If you prefer submitting by dataset, run:

```bash
qsub run_enhanced_trace.sh
qsub run_standard_trace.sh
qsub run_144a_trace.sh
```

Each wrapper uses SGE's `-cwd` so outputs/logs land under the current folder, and `-V` to pass your Python environment variables.

---

### Running report generation separately

If you run jobs individually and want to generate reports later, or if you want to regenerate reports with different settings:

```bash
qsub run_build_data_reports.sh
```

**Important:** If running report generation separately, edit `_build_error_files.py` to specify which datasets to process by modifying the `DATA_TYPES` variable:

```python
# In _build_error_files.py:
DATA_TYPES = ['enhanced', 'standard', '144a']  # Process all three
# Or:
DATA_TYPES = ['enhanced']  # Process only Enhanced TRACE
```

Change the filtering settings in `_trace_settings.py` if needed.

---

## What the runners do

### Enhanced TRACE (`_run_enhanced_trace.py`)

Calls `CreateDailyEnhancedTRACE` with the default cleaning/filters and audit logging:
- Processes the full Enhanced TRACE sample period (2002-07-01 to present)
- Applies Dick-Nielsen filters for cancellations, corrections, and agency duplicates
- Runs decimal-shift correction (detects and fixes 10x, 0.1x, 100x, 0.01x price errors)
- Applies bounce-back price-error filtering
- Computes daily price metrics: equal-weighted, volume-weighted, par-weighted, first, last, trade count
- Computes daily volume metrics: quantity volume and dollar volume (in millions)
- Computes customer-side bid/ask prices (value-weighted)
- Generates comprehensive audit logs for each filter stage
- **Saves all outputs to `enhanced/` subfolder**

### Standard TRACE (`_run_standard_trace.py`)

Calls `CreateDailyStandardTRACE` with the same controls for the Standard table:
- Default start date is set to `2024-10-01` (you can change this in `_trace_settings.py`)
- Applies pre-2012 and post-2012 cleaning rules
- Handles reversal trades specific to Standard TRACE
- Same decimal-shift and bounce-back filters as Enhanced
- Same daily aggregation metrics
- **Saves all outputs to `standard/` subfolder**

### Rule 144A TRACE (`_run_144a_trace.py`)

Calls `CreateDailyStandardTRACE` with `data_type='144a'`:
- Default start date is `2002-07-01` (Rule 144A has been available since TRACE inception, but most data only start around 2008)
- Uses the same cleaning pipeline as Standard TRACE
- Same parameter blocks for filters and aggregation
- **Saves all outputs to `144a/` subfolder**

### Core processing steps

Both `create_daily_enhanced_trace.py` and `create_daily_standard_trace.py`:

1. **Connect to WRDS** and establish database connection
2. **Filter FISD universe** based on configured parameters (USD only, fixed-rate, non-convertible, etc.)
3. **Chunk CUSIPs** into batches (default 250 CUSIPs per chunk)
4. **For each chunk:**
   - Fetch intraday TRACE rows from WRDS
   - Apply Dick-Nielsen filters (cancellations, corrections, reversals)
   - Apply agency de-duplication (if enabled)
   - Run decimal-shift corrector to fix multiplicative price errors
   - Apply bounce-back filter to flag erroneous price spikes
   - Filter by trading time (if enabled)
   - Filter by trading calendar (if enabled)
   - Apply price range filters (> 0, <= 1000)
   - Apply volume filters (dollar or par-based)
   - Filter yield != price trades
   - Filter trades where volume exceeds 50% of offering amount
   - Filter trades where execution date > maturity date
   - Aggregate to daily (cusip_id, trd_exctn_dt) panels
5. **Concatenate all chunks** and export to Parquet format
6. **Generate audit logs** with row counts for each filter stage
7. **Export CUSIP lists** of bonds affected by decimal-shift and bounce-back corrections

---

## Configuration choices you can edit

Open `_trace_settings.py` and adjust the following:

### WRDS Username
```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_username_here")
```

### FISD Universe Parameters (`FISD_PARAMS`)

Controls which bonds are included in the universe:
- `currency_usd_only`: Keep only USD-denominated bonds (default: `True`)
- `fixed_rate_only`: Exclude variable-rate bonds (default: `True`)
- `non_convertible_only`: Exclude convertible bonds (default: `True`)
- `non_asset_backed_only`: Exclude asset-backed securities (default: `True`)
- `exclude_bond_types`: Drop specific bond types like TXMU, MBS, FGOV, etc. (default: `True`)
- `valid_coupon_frequency_only`: Drop bonds with invalid interest frequency (default: `True`)
- `require_accrual_fields`: Require offering_date, dated_date to be non-null (default: `True`)
- `principal_amt_eq_1000_only`: Keep only bonds with $1000 par value (default: `True`)
- `exclude_equity_index_linked`: Exclude equity-linked and index-linked bonds (default: `True`)
- `enforce_tenor_min`: Require bonds to have minimum tenor (default: `True`)
- `tenor_min_years`: Minimum tenor in years (default: `1.0`)

### Filter Toggles (`FILTER_SWITCHES`)

All filters are boolean toggles:
- `dick_nielsen`: Apply Dick-Nielsen cleaning steps (default: `True`)
- `decimal_shift_corrector`: Fix decimal shift errors (default: `True`)
- `trading_time`: Filter by intraday time window (default: `False`)
- `trading_calendar`: Keep only valid trading days (default: `True`)
- `price_filters`: Remove negative prices and prices > 1000 (default: `True`)
- `volume_filter_toggle`: Apply volume threshold (default: `True`)
- `bounce_back_filter`: Flag price-change errors (default: `True`)
- `yld_price_filter`: Remove rows where yield = price (default: `True`)
- `amtout_volume_filter`: Remove trades > 50% of offering amount (default: `True`)
- `trd_exe_mat_filter`: Remove trades after maturity date (default: `True`)

### Decimal-Shift Corrector Parameters (`DS_PARAMS`)

Fine-tune the decimal shift correction algorithm:
- `factors`: Multiplicative factors to test - `(0.1, 0.01, 10.0, 100.0)`
- `tol_pct_good`: Relative error threshold for accepting a potential correction - `0.02` (2%)
- `tol_abs_good`: Absolute distance threshold in price points - `8.0`
- `tol_pct_bad`: Minimum raw relative error to trigger consideration - `0.05` (5%)
- `low_pr`, `high_pr`: Plausible price bounds - `5.0`, `300.0`
- `anchor`: Anchor type for comparison - `"rolling"`
- `window`: Rolling window half-width - `5`
- `improvement_frac`: Required improvement over raw error - `0.2` (20%)
- `par_snap`: Enable relaxed acceptance near par=100 - `True`
- `par_band`: Proximity band around par - `15.0`
- `output_type`: `"cleaned"` to apply corrections, `"uncleaned"` for audit only

### Bounce-Back Filter Parameters (`BB_PARAMS`)

Fine-tune the bounce-back price-error detection:
- `threshold_abs`: Minimum absolute price jump to flag a candidate error - `35.0`
- `lookahead`: Maximum rows ahead to search for bounce - `5`
- `max_span`: Maximum path length from start to resolution - `5`
- `window`: Backward window for trailing median anchor - `5`
- `back_to_anchor_tol`: Fraction of displacement to recover - `0.25`
- `candidate_slack_abs`: Slack around anchor when opening - `1.0`
- `reassignment_margin_abs`: Margin for tie-breaking in clusters - `5.0`
- `use_unique_trailing_median`: Use unique values in median - `True`
- `par_spike_heuristic`: Enable special handling at par - `True`
- `par_level`: Par value - `100.0`
- `par_equal_tol`: Tolerance for treating as par - `1e-8`
- `par_min_run`: Minimum par run length to flag - `3`
- `par_cooldown_after_flag`: Rows to skip after flagging - `2`

### Common Arguments (`COMMON_KWARGS`)

Settings applied to all runners:
- `output_format`: `"parquet"` (lightweight) or `"csv"` (larger `.csv.gzip`)
- `chunk_size`: Number of CUSIPs per batch - `250`
- `clean_agency`: Apply agency de-duplication - `True`
- `out_dir`: Output directory - `""` (current directory)
- `volume_filter`: Tuple of `(kind, threshold)`:
  - `("dollar", 10000)`: Dollar volume >= $10,000
  - `("par", 10000)`: Par volume >= $10,000
- `trade_times`: Intraday window - `["00:00:00", "23:59:59"]` (effectively disabled)
- `calendar_name`: Market calendar - `"NYSE"`

### Per-Dataset Overrides (`PER_DATASET`)

Specific settings for each dataset:
- **Enhanced**: No extra arguments (uses defaults)
- **Standard**: `start_date="2024-10-01"`, `data_type="standard"`
- **144A**: `start_date="2002-07-01"`, `data_type="144a"`

---

## Outputs

### Output directory structure

All outputs are organized into dataset-specific subfolders for data, with a single `data_reports/` folder containing subfolders for each dataset's reports:

```
stage0/
├── logs/                        # Job logs for all runs
│   ├── 01_enhanced.out
│   ├── 01_enhanced.err
│   ├── 02_standard.out
│   ├── 02_standard.err
│   ├── 03_144a.out
│   ├── 03_144a.err
│   ├── 04_reports.out
│   └── 04_reports.err
│
├── enhanced/                    # Enhanced TRACE data outputs
│   ├── enhanced_YYYYMMDD.parquet
│   ├── *_audit_*.parquet
│   └── *_cusips_*.parquet
│
├── standard/                    # Standard TRACE data outputs
│   ├── standard_YYYYMMDD.parquet
│   ├── *_audit_*.parquet
│   └── *_cusips_*.parquet
│
├── 144a/                        # Rule 144A data outputs
│   ├── 144a_YYYYMMDD.parquet
│   ├── *_audit_*.parquet
│   └── *_cusips_*.parquet
│
└── data_reports/                # Quality reports for ALL datasets
    ├── enhanced/
    │   ├── enhanced_data_report.tex
    │   ├── references.bib
    │   └── (figures)
    ├── standard/
    │   ├── standard_data_report.tex
    │   ├── references.bib
    │   └── (figures)
    └── 144a/
        ├── 144a_data_report.tex
        ├── references.bib
        └── (figures)
```

**For downstream stages:** When exporting to your home machine, maintain this structure:
```
data/
    stage0/
        enhanced/
        standard/
        144a/
        data_reports/
            enhanced/
            standard/
            144a/
    stage1/
    stage2/
```

### Files produced

**Logs** (under `./logs/`):
- `01_enhanced.out`, `01_enhanced.err`: Enhanced TRACE job logs
- `02_standard.out`, `02_standard.err`: Standard TRACE job logs
- `03_144a.out`, `03_144a.err`: 144A TRACE job logs
- `04_reports.out`, `04_reports.err`: Report generation logs (when using `run_all_trace.sh`)
- Logs contain timestamps, row counts, filter statistics, and any errors

**Daily panels** (Parquet format, in respective subfolders):
- `enhanced/enhanced_YYYYMMDD.parquet`: Enhanced TRACE daily panel (~30 million rows for full sample)
- `standard/standard_YYYYMMDD.parquet`: Standard TRACE daily panel
- `144a/144a_YYYYMMDD.parquet`: Rule 144A daily panel

All panels have identical column structure:
- **Keys**: `cusip_id`, `trd_exctn_dt`
- **Prices**: `prc_ew`, `prc_vw`, `prc_vw_par`, `prc_first`, `prc_last`, `trade_count`
- **Volumes** (millions): `qvolume`, `dvolume`
- **Bid/Ask**: `prc_bid`, `prc_ask`, `bid_count`, `ask_count`

**Audit files** (Parquet format, in respective subfolders):
- `dick_nielsen_filters_audit_{dtype}_{date}.parquet`: Dick-Nielsen step-by-step audit
- `drr_filters_audit_{dtype}_{date}.parquet`: Dickerson-Rossetti-Robotti filter audit
- `fisd_filters_{dtype}_{date}.parquet`: FISD universe construction audit

**CUSIP lists** (Parquet format, in respective subfolders):
- `bounce_back_cusips_{dtype}_{date}.parquet`: CUSIPs with bounce-back flags
- `decimal_shift_cusips_{dtype}_{date}.parquet`: CUSIPs with decimal-shift corrections

### Downloading outputs

**Windows users:** Use WinSCP to download the entire `enhanced/`, `standard/`, `144a/`, `data_reports/`, and `logs/` folders to your local machine.

**Mac/Linux users:** Use `scp` from your local machine:
```bash
# Download all outputs preserving folder structure
scp -r wrds_username@wrds-cloud.wharton.upenn.edu:~/proj/stage0/enhanced ./local_destination/stage0/
scp -r wrds_username@wrds-cloud.wharton.upenn.edu:~/proj/stage0/standard ./local_destination/stage0/
scp -r wrds_username@wrds-cloud.wharton.upenn.edu:~/proj/stage0/144a ./local_destination/stage0/
scp -r wrds_username@wrds-cloud.wharton.upenn.edu:~/proj/stage0/data_reports ./local_destination/stage0/
scp -r wrds_username@wrds-cloud.wharton.upenn.edu:~/proj/stage0/logs ./local_destination/stage0/
```

---

## Generating the TRACE Data Reports

When you run `./run_all_trace.sh`, reports are automatically generated for all three datasets after data processing completes using SGE's `-hold_jid` dependency feature. However, you can also generate or regenerate reports separately.

### Configuration

Edit the top of `_build_error_files.py`:

```python
DATE           = ""          # Leave this BLANK -- date is inherited [NB]
output_figures = True        # Set to False for tables only (faster)
DATA_TYPES     = ['enhanced', 'standard', '144a']  # Which datasets to process
IN_DIR         = ""          # Leave blank for current directory
OUT_DIR        = ""          # Leave blank for current directory
```

**Important notes:**
- `output_figures = True` creates hundreds of time-series plots and can take 30+ minutes for Enhanced TRACE
- `output_figures = False` only generates filter tables and runs in seconds
- For Enhanced TRACE with figures, the LaTeX document can exceed 500 pages
- `DATA_TYPES` controls which datasets are processed - modify this to process only specific datasets

### Running the report generator separately

If you want to regenerate reports with different settings or didn't run `./run_all_trace.sh`:

Make the script executable (first time only):
```bash
chmod +x run_build_data_reports.sh
```

Submit the job:
```bash
qsub run_build_data_reports.sh
```

### Output structure

Reports are automatically saved in a single `data_reports/` folder with subfolders for each dataset:

```
data_reports/
    ├── enhanced/
    │   ├── enhanced_data_report.tex
    │   ├── references.bib
    │   ├── enhanced_fig_page_001_ds.pdf
    │   ├── enhanced_fig_page_002_ds.pdf
    │   ├── ...
    │   ├── enhanced_fig_page_001_bb.pdf
    │   ├── enhanced_fig_page_002_bb.pdf
    │   └── ...
    ├── standard/
    │   ├── standard_data_report.tex
    │   ├── references.bib
    │   └── (figures if generated)
    └── 144a/
        ├── 144a_data_report.tex
        ├── references.bib
        └── (figures if generated)
```

Each `*_data_report.tex` file includes:
1. **Table 1**: Filter toggles and parameter settings
2. **Table 2**: FISD universe construction parameters
3. **Table 3**: Transaction-level filter records (Panel A: FISD, Panel B: DRR filters, Panel C: Dick-Nielsen)
4. **Figures** (if enabled): Time-series plots showing decimal-shift corrections and bounce-back eliminations

### Compiling the LaTeX report

Download the report folder to your local machine and compile:

```bash
cd data_reports/enhanced
pdflatex enhanced_data_report.tex
bibtex enhanced_data_report
pdflatex enhanced_data_report.tex
pdflatex enhanced_data_report.tex
```

Or use your favorite LaTeX editor (TeXShop, TeXstudio, Overleaf, etc.).

---

## Notes & tips

- **Environment setup**: Ensure your WRDS Python environment has all required packages. If you're using a module or conda environment, load/activate it before submitting jobs.

- **Automated workflow**: The `run_all_trace.sh` script uses SGE's `-hold_jid` feature to create job dependencies. The report generation job automatically waits in the queue (status `hqw`) until all three data jobs complete. This is the recommended workflow.

- **Job dependency**: When you run `./run_all_trace.sh`, you'll see four jobs in `qstat`:
  - `trace_enhanced` - running or queued
  - `trace_standard` - running or queued
  - `trace_144a` - running or queued
  - `build_reports` - status `hqw` (holding) until the above three finish

- **Memory considerations**: Each job processes 250 CUSIPs at a time by default. If you encounter memory issues, reduce `chunk_size` in `_trace_settings.py`.

- **Runtime expectations**:
  - Enhanced TRACE (full sample): 4-8 hours depending on WRDS load
  - Standard TRACE (from 2024): 30-60 minutes
  - Rule 144A (full sample): 2-4 hours
  - Data reports (with figures): 30-60 minutes per dataset

- **Disk space**: Enhanced TRACE generates ~30M rows. Parquet files are compressed and typically 500MB-1GB per dataset. CSV files are much larger.

---

## Troubleshooting

### Installation issues

- **pip: command not found**: 
  ```bash
  python -m pip install --user {package_name}
  ```

- **ImportError: wrds**: 
  Install `wrds` in your WRDS Python environment or activate the appropriate conda/module environment.
  ```bash
  python -m pip install --user wrds
  ```

- **ImportError: pandas_market_calendars**:
  ```bash
  python -m pip install --user pandas-market-calendars
  ```

### Script execution issues

- **Script not executable**: 
  ```bash
  chmod +x run_all_trace.sh run_enhanced_trace.sh run_standard_trace.sh run_144a_trace.sh
  ```

- **Permission denied** when trying to run `./run_all_trace.sh`:
  You probably missed the `chmod +x` step above.

- **Bad interpreter** or `^M` errors:
  Convert Windows line endings to Unix:
  ```bash
  sed -i 's/\r$//' run_all_trace.sh
  # Or fix all shell scripts at once:
  find . -name "*.sh" -exec sed -i 's/\r$//' {} \;
  ```

### SGE issues

- **SGE not submitting**: 
  - Confirm you are in the correct directory
  - Verify that `qsub` is available on your WRDS node
  - Check that shell scripts have Unix line endings

- **Job stays in queue**: 
  - Check `qstat` to see if resources are available
  - WRDS may have resource constraints during peak hours

### Data issues

- **No data returned**: 
  - Verify your CUSIP list/date range in the configuration
  - Confirm that your WRDS entitlements cover Enhanced/Standard/144A as applicable
  - Check logs for SQL errors or connection issues

- **Empty output files**:
  - Check that your date ranges are correct in `_trace_settings.py`
  - Verify that FISD filters aren't excluding all bonds
  - Review the audit files to see where rows were dropped

### Memory issues

- **Job killed due to memory**:
  - Reduce `chunk_size` in `_trace_settings.py` (try 100 or 150)
  - Request more memory in the shell scripts by adding:
    ```bash
    #$ -l m_mem_free=8G
    ```

### Report generation issues

- **Figures not generating**:
  - Verify that `matplotlib` is installed
  - Check that CUSIP lists exist in the expected location
  - Ensure `output_figures = True` in `_build_error_files.py`

- **LaTeX compilation errors**:
  - Ensure all figure files are present
  - Check that `references.bib` exists
  - Verify you have a complete LaTeX installation with required packages

---

## Monitoring

### Queue status
```bash
qstat                    # View all your jobs
qstat -u wrds_username   # View only your jobs
```

### Real-time log monitoring
```bash
tail -f logs/01_enhanced.out      # Follow Enhanced output log
tail -f logs/01_enhanced.err      # Follow Enhanced error log
tail -f logs/02_standard.out      # Follow Standard output log
```

### Checking job completion
```bash
ls -lh *.parquet         # List generated parquet files
wc -l logs/*.out         # Count lines in log files
```

### Resubmitting failed jobs

If a job fails:
1. Review the error log: `cat logs/01_enhanced.err`
2. Fix the issue in configuration or code
3. Resubmit: `qsub run_enhanced_trace.sh` (or `./run_all_trace.sh`)

---

## Advanced Usage

### Custom date ranges

To process a specific date range, modify the per-dataset overrides in `_trace_settings.py`:

```python
PER_DATASET = {
    "enhanced": dict(),
    "standard": dict(start_date="2020-01-01", data_type="standard"),
    "144a": dict(start_date="2020-01-01", data_type="144a"),
}
```

Note: Enhanced TRACE does not support custom start dates in the current implementation - it always processes the full sample.

### Custom output directories

To write outputs to a specific directory:

```python
COMMON_KWARGS = dict(
    ...
    out_dir = "/path/to/output/directory",
    ...
)
```

### Disabling specific filters

To disable any filter, set it to `False` in `_trace_settings.py`:

```python
FILTER_SWITCHES = dict(
    dick_nielsen            = True,
    decimal_shift_corrector = False,  # Disable decimal shift correction
    trading_time            = False,
    trading_calendar        = True,
    price_filters           = True,
    volume_filter_toggle    = True,
    bounce_back_filter      = False,  # Disable bounce-back filter
    yld_price_filter        = True,
    amtout_volume_filter    = True,
    trd_exe_mat_filter      = True,
)
```

### Custom volume filters

You can specify volume filters in two ways:

```python
# Dollar volume threshold (default)
COMMON_KWARGS = dict(
    ...
    volume_filter = ("dollar", 10000),  # $10,000
    ...
)

# Par value threshold (alternative)
COMMON_KWARGS = dict(
    ...
    volume_filter = ("par", 10000),  # $10,000
    ...
)
```

### Time-of-day filtering

To restrict to specific trading hours:

```python
FILTER_SWITCHES = dict(
    ...
    trading_time = True,  # Enable time filtering
    ...
)

COMMON_KWARGS = dict(
    ...
    trade_times = ["09:30:00", "16:00:00"],  # NYSE regular hours
    ...
)
```

---

## Performance optimization

### Chunking strategy

The default `chunk_size = 250` balances memory usage and processing speed. Adjust based on your needs:

- **Faster processing** (more memory): `chunk_size = 500`
- **Lower memory** (slower): `chunk_size = 100`

### Parallel processing

To maximize throughput, submit all three jobs simultaneously:
```bash
./run_all_trace.sh
```

This uses SGE's parallel job execution capability.

### Output format choice

- **Parquet** (default): Smaller files, faster read/write, better compression
- **CSV**: Larger files, human-readable, compatible with older tools

```python
COMMON_KWARGS = dict(
    ...
    output_format = "parquet",  # or "csv"
    ...
)
```

---

## License & Citation

### License

This code is provided under the MIT License. See LICENSE file for details.

### Citation

If you use or extend this stage, please cite:

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

### Acknowledgments

This pipeline implements cleaning procedures from:
- Dick-Nielsen, J. (2009). Liquidity biases in TRACE. *The Journal of Fixed Income*, 19(2), 43-55.
- Dick-Nielsen, J. (2014). How to clean enhanced TRACE data. Working Paper.
- van Binsbergen, J. H., Nozawa, Y., & Schwert, M. (2025). Duration-based valuation of corporate bonds. *The Review of Financial Studies*, 38(1), 158-191.

---

## Support

For questions, issues, or contributions:
- **Email**: alexander.dickerson1@unsw.edu.au
- **GitHub Issues**: [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)

---

## Version History

- **v1.0** (2025-01-20): Initial release
  - Enhanced, Standard, and 144A TRACE processing
  - Decimal-shift and bounce-back error correction
  - Comprehensive audit logging
  - LaTeX report generation

---

**Last updated:** January 2025





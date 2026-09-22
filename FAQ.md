# Frequently Asked Questions (FAQ)

## Table of Contents
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)
- [Academic Use](#academic-use)

- [Stage 1, 2 & 3](#stage-1-2--3)
- [Support](#support)
---

## Getting Started

### Do I need a WRDS subscription?
Yes, the pipeline requires WRDS access with TRACE Enhanced, Standard, or 144A entitlements, plus FISD and ratings data for Stage 1. You can check your entitlements by logging into WRDS and viewing your subscriptions.

### Which parts run where?
This is a **two-machine pipeline**, and the hand-off is a file you copy yourself.

| | Where | Why |
|---|---|---|
| Stages 0 and 1 | **WRDS Cloud** | they read the raw TRACE tape, which is a WRDS database |
| the hand-off | you | zip on WRDS, `scp` down (~5 GB) |
| Stage 2 | **your own computer** | it reads only Stage 1's output file; no WRDS connection needed |
| Stage 3 | **your own computer** | it reads only Stage 2's panel; optional, and no WRDS connection needed |

### Can I run Stages 0 and 1 on my own machine?
Technically yes, with a working WRDS connection -- but you would be pulling hundreds of
millions of trades across the internet, and `run_pipeline.sh` submits jobs to the WRDS grid,
so you would be invoking the stage scripts directly instead. Use WRDS Cloud.

### Can I skip Stages 0 and 1 and just download the Stage 1 file?
No, not for Stage 2. The published Stage 1 download has its rating columns removed, because
agency ratings are licensed. Stage 2 keeps only bond-months carrying a rating, so it would
produce an **empty panel with no error**. Stage 2 checks for exactly this and refuses to
start. The download is there for people who want the daily panel itself.

### How long does processing take?
Using `./run_pipeline.sh` (complete automated pipeline):
- **Pre-stage (data downloads)**: ~5 minutes
- **Stage 0 (Enhanced TRACE)**: **~2 hours**. It was ~4 hours serial; since v2.2.0 it
  pulls 5 CUSIP chunks at once over separate WRDS connections. The 2026-09-09 run did
  485 chunks in 2.01 h against a serial-equivalent 9.93 h -- a **4.94x** speedup.
- **Stage 0 (Rule 144A)**: **~45 minutes**, running at the same time as Enhanced
- **Stage 0 (Standard TRACE)**: ~30-60 minutes, and OPT-IN since v2.2.0. When requested
  it is scheduled after the other two rather than beside them.
- **Stage 0 (Report generation)**: **~15 minutes** since v2.2.2, and it no longer blocks
  Stage 1 -- the two run side by side. It was ~50 minutes before that release.
- **Stage 1 (Bond analytics)**: **~2.5-3 hours**

**End to end: about 4.5-5 hours.** The 2026-09-09 run took 4.63 h and the 2026-09-10 run
4.7 h.

Then, on your own machine: **Stage 2 about 8-13 minutes**, and **Stage 3 about 15 minutes**
on 24 cores with a PyBondLab build carrying the fast kernels. Without those kernels Stage
3 still runs, and the two uncertainty grids become the long part; see
[stage3/README_stage3.md](stage3/README_stage3.md).

### What if I only want Enhanced TRACE?
You can customize which datasets to process in `config.py` (applies to all stages):
```python
TRACE_MEMBERS = ["enhanced"]  # Process only Enhanced TRACE
```

Or run Stage 0 jobs individually:
```bash
qsub stage0/run_enhanced_trace.sh    # Enhanced only
qsub stage0/run_standard_trace.sh    # Standard only
qsub stage0/run_144a_trace.sh        # 144A only
```

### What Python version do I need?
Python 3.10 or higher is required. Check your version:
```bash
python --version
```

### Do I need to install anything besides Python packages?
For Stage 0, you need:
- WRDS subscription with TRACE access
- SSH access to WRDS Cloud
- `.pgpass` file configured for password-less authentication
- Required Python packages (installed via `requirements.txt`)

For Stage 3, one non-Python thing: **pdflatex** (TeX Live or MiKTeX), used by the last
step to compile every exhibit into one PDF. `stage3/tools/check_inputs.py` warns when it
is missing rather than failing -- without it you still get every table and figure as a
file, you just do not get `reports/exhibits.pdf`.

---

## Configuration

### Where do I configure settings?
Settings are organized hierarchically:

1. **Shared settings** (`config.py` in root):
   - `WRDS_USERNAME`: Your WRDS username
   - `OUTPUT_FORMAT`: Output file format. `"parquet"` only -- see the FAQ entry below.
   - `AUTHOR`: Your name
   - `TRACE_MEMBERS`: Which datasets to process (enhanced, standard, 144a) - **shared across all stages**
   - `STAGE0_OUTPUT_FIGURES`: Control Stage 0 error plot generation (can be slow)

2. **Stage 0 settings** (`stage0/_trace_settings.py`):
   - Filter switches, FISD parameters, chunk sizes
   - Decimal-shift and bounce-back parameters

3. **Stage 1 settings** (`stage1/_stage1_settings.py`):
   - Date cutoffs, performance tuning (cores, chunks)
   - Ultra-distressed filter configuration
   - **Note:** Stage 1 always generates reports and figures (no toggle)

### Do I need to configure anything to get started?
**Minimal configuration:** Just set your WRDS username in `config.py`:
```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_id")
```

Most other settings are **auto-detected**:
- ✅ `ROOT_PATH`: Auto-detected from working directory
- ✅ `STAGE0_DATE_STAMP`: Auto-detected from Stage 0 output files
- ✅ `N_CORES`: Auto-detected from available CPUs

### How do I change the date range?

**Stage 0:** Edit `stage0/_trace_settings.py` and modify `PER_DATASET`:
```python
# Keep each member's existing keys and add start_date -- do not replace the dict.
# ❗Dropping `n_workers` from "enhanced" silently returns Stage 0 to the serial,
# ~4-hour path, because the engine default is n_workers=1.
PER_DATASET = {
    "enhanced": dict(n_workers=WORKERS_OVERRIDE or CONCURRENCY["enhanced"]),
    "standard": dict(start_date="2024-10-01", data_type="standard",
                     n_workers=WORKERS_OVERRIDE or CONCURRENCY["standard"]),
    "144a":     dict(start_date="2002-07-01", data_type="144a",
                     n_workers=WORKERS_OVERRIDE or CONCURRENCY["144a"]),
}
```

**Stage 1:** Edit `stage1/_stage1_settings.py`:
```python
DATE_CUT_OFF = "2023-12-31"  # Only include data through this date
```

### How do I adjust memory usage?

Stage 0's memory is driven by two things: how big a chunk is, and how many workers hold
one at a time.

**Chunk size** — since v2.2.0 chunks are packed to a TRADE-ROW target, not a CUSIP count.
Lower it in `stage0/_trace_settings.py`:

```python
TARGET_ROWS_PER_CHUNK = 400_000    # default 750_000
```

or for a single run, without editing anything: `STAGE0_TARGET_ROWS=400000 ./run_pipeline.sh`.

(`chunk_size` still exists and still means CUSIPs-per-chunk, but it no longer sizes
Stage 0's work — it is read by the data-report job for its own chunking. Changing it will
not affect Stage 0 memory.)

**Worker count** — each worker holds one chunk, so this multiplies the above. Lower
`CONCURRENCY` in `stage0/_trace_settings.py`, or `STAGE0_WORKERS=3 ./run_pipeline.sh`
for one run.

**Requesting more memory** is not done in the job scripts. `run_pipeline.sh` computes the
request per member and passes it on the `qsub` command line, which overrides anything in
the `.sh`. Change `MEM_PER_SLOT_GB` in `stage0/_trace_settings.py`:

```python
MEM_PER_SLOT_GB = {"enhanced": 8, "standard": 8, "144a": 16}
```

❗`m_mem_free` is charged PER SLOT, so the total is `slots x mem` and must stay within the
WRDS caps of 8 cores and 48 GB per job. `qsub_resources()` refuses to emit anything over
them, because an over-request does not error — the job pends forever, silently.

### Can I disable certain filters?
Yes! Edit `FILTER_SWITCHES` in `stage0/_trace_settings.py`:

```python
FILTER_SWITCHES = dict(
    dick_nielsen            = True,
    decimal_shift_corrector = True,
    bounce_back_filter      = False,  # Disable this filter
    # ... etc
)
```

### How do I change which bonds are included?
Modify `FISD_PARAMS` in `stage0/_trace_settings.py`:

```python
FISD_PARAMS = {
    "currency_usd_only": True,      # Only USD bonds
    "fixed_rate_only": True,        # Only fixed-rate bonds
    "tenor_min_years": 1.0,         # Minimum tenor
    # ... etc
}
```

### Can I change the output format from Parquet to CSV?
No. `OUTPUT_FORMAT` in `config.py` accepts only `"parquet"`, and `_trace_settings.py`
raises at import if you set anything else.

Stage 0 can technically write `.csv.gzip`, but Stage 1 and the report builder both call
`pd.read_parquet` on a hard-coded `*.parquet` name, so a CSV run produces stage-0 files
that nothing downstream can open. Before v2.2.3 that failed several hours in, with a
misleading *"Expected: stage0/<member>/trace_<member>_<stamp>.parquet"*. It now fails
immediately instead.

To get a CSV copy of a finished dataset, convert it afterwards:
```python
import pandas as pd
pd.read_parquet("stage1/data/stage1_YYYYMMDD.parquet").to_csv("stage1.csv.gz", index=False)
```

### How do I control Stage 0 error plot generation?
Stage 0 error plots take about 5 minutes for Enhanced TRACE (measured 2026-09-09), and the report job runs beside Stage 1, so they no longer delay anything. Control them in `config.py`:
```python
STAGE0_OUTPUT_FIGURES = False  # Skip error plots (tables only - faster)
STAGE0_OUTPUT_FIGURES = True   # Generate error plots (slow but comprehensive)
```

**Note:** Stage 1 always generates reports and figures regardless of this setting (essential for data quality).

### How do I change which datasets to process across all stages?
Edit `TRACE_MEMBERS` in `config.py` once, and it applies to all stages:
```python
TRACE_MEMBERS = ["enhanced"]                       # Enhanced only
TRACE_MEMBERS = ["enhanced", "standard"]           # Two datasets
TRACE_MEMBERS = ["enhanced", "144a"]              # The DEFAULT
TRACE_MEMBERS = ["enhanced", "standard", "144a"]  # All three (Standard is opt-in)
```

---

## Output Files

### What format are the output files?
Output files are in **Parquet format** by default (compressed, efficient). Parquet is a columnar storage format that:
- Compresses well (~5-10x smaller than CSV)
- Loads faster
- Preserves data types
- Works with pandas, R, and many other tools

### How do I read the output files?
Using Python/pandas:
```python
import pandas as pd

# Read a single dataset (use your own date stamp)
df = pd.read_parquet('stage0/enhanced/trace_enhanced_20260910.parquet')

# Read every member's daily panel -- the pattern skips the FISD, audit and CUSIP-list files
import glob
files = glob.glob('stage0/*/trace_*_[0-9]*.parquet')
files = [f for f in files if 'fisd' not in f]
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
```

Using R:
```r
library(arrow)
df <- read_parquet('stage0/enhanced/trace_enhanced_20260910.parquet')
```

### What columns are in the output files?
All datasets (Enhanced, Standard, 144A) share the same 21-column structure:

**Identifiers**:
- `cusip_id`: 9-character CUSIP identifier
- `trd_exctn_dt`: Trade execution date

**Price metrics**:
- `prc_ew`: Equal-weighted price
- `prc_vw`: Volume-weighted price (dollar)
- `prc_vw_par`: Volume-weighted price (par)
- `prc_first`: First trade price of day
- `prc_last`: Last trade price of day
- `prc_lo`: Low price of the day
- `prc_hi`: High price of the day

**Trade-time metrics** (seconds since midnight):
- `time_ew`: Equal-weighted mean execution time
- `time_last`: Execution time of the last trade

**Volume metrics** (in millions):
- `qvolume`: Par volume
- `dvolume`: Dollar volume

**Bid/Ask metrics** (❗DEALER side -- see the note below):
- `prc_bid`: Dealer bid, value-weighted (the dealer BUYS here, so the customer sells)
- `prc_ask`: Dealer ask, value-weighted (the dealer SELLS here, so the customer buys)
- `bid_last`: Last dealer-bid price of the day
- `bid_time_ew`: Equal-weighted mean time of dealer-bid trades
- `bid_time_last`: Time of the last dealer-bid trade

**Count metrics**:
- `trade_count`: Number of trades
- `bid_count`: Number of dealer buys (= customer sells)
- `ask_count`: Number of dealer sells (= customer buys)

❗**TRACE reports the DEALER's side.** The bid/ask split is
`rpt_side_cd == 'B'` (dealer buying) and `'S'` (dealer selling), both filtered to
`cntra_mp_id == 'C'` (the counterparty is a customer). So a customer BUY appears as
`rpt_side_cd == 'S'`. Getting this backwards is the single commonest TRACE bug.

### What do the audit files contain?
Audit files track row counts at each filter stage:
- `dick_nielsen_filters_audit_*.parquet`: Dick-Nielsen filter effects
- `drr_filters_audit_*.parquet`: Decimal-shift and bounce-back filter effects
- `fisd_filters_*.parquet`: FISD universe construction audit

These help you understand how many transactions were removed at each cleaning step.

### What are the CUSIP list files?
These files identify bonds that were corrected:
- `decimal_shift_cusips_*.parquet`: CUSIPs with decimal-shift corrections
- `bounce_back_cusips_*.parquet`: CUSIPs with bounce-back flags
- `init_price_cusips_*.parquet`: CUSIPs with a removed initial-price error

Useful for understanding which bonds had price errors.

### What is the ultra_distressed_cusips CSV file? (Stage 1)
Stage 1 exports `stage1/data/ultra_distressed_cusips_{date}.csv`, one row for **every** CUSIP
in the panel, flagged or not. Filter on `flagged_observations > 0` for the flagged bonds. In
the 2026-09-10 run the file has 76,592 rows, of which 1,045 CUSIPs carry 10,185 flagged
bond-days (0.03% of all bond-days).

**Columns**:
- `cusip_id`: Bond identifier
- `total_observations`: Bond-days for this CUSIP
- `flagged_observations`: Bond-days the filter flagged
- `pct_flagged`: Percentage flagged (%)
- `first_trade_date`: Earliest trade date
- `last_trade_date`: Latest trade date

The file does not break the count down by filter. The per-filter flags are dropped before
export to save memory.

**Use cases**:
- Identify problematic bonds for manual review
- Cross-reference with other bond characteristics (ratings, maturity, etc.)
- Understand filtering patterns across bond universe
- Quality control and diagnostics

### Where are the reports saved?
Reports are saved in `data_reports/[enhanced|standard|144a]/` with:
- LaTeX source file (`.tex`)
- Bibliography file (`references.bib`)
- Figures (`.pdf` files)

You can compile the LaTeX to PDF or view the figures directly.

### How do I download files from WRDS Cloud?

The pipeline generates a large folder (~5 GB) with hundreds of files. **Zip the folder first** for faster, more reliable downloads.

**Step 1: Create zip file on WRDS (via SSH)**

```bash
# Connect to WRDS
ssh {wrds_username}@wrds-cloud.wharton.upenn.edu

# Zip to scratch space (your home directory has limited quota).
# Run from ~ on a RELATIVE path: zip stores the path you give it, so an absolute
# ~/trace-data-pipeline/ makes the archive extract four directories deep.
cd ~
zip -r "/scratch/$(basename "$(dirname "$HOME")")/trace-data-pipeline.zip" trace-data-pipeline/
```
`$(basename "$(dirname "$HOME")")` is your institution code, the folder between `/home` and your username. Type the line as written; nothing in it is a placeholder. For the `scp` line below, `{institution}` IS a placeholder: replace it, braces included, with that code (`echo $HOME` on WRDS shows it).

**Step 2: Download the zip file (from your LOCAL machine)**

**Windows (PowerShell/Terminal):**
```powershell
scp {wrds_username}@wrds-cloud.wharton.upenn.edu:/scratch/{institution}/trace-data-pipeline.zip "{local_destination}"
```

**Mac/Linux:**
```bash
scp {wrds_username}@wrds-cloud.wharton.upenn.edu:/scratch/{institution}/trace-data-pipeline.zip "{local_destination}"
```

**Windows (WinSCP - GUI alternative):**
1. Connect to `wrds-cloud.wharton.upenn.edu`
2. Navigate to `/scratch/{institution}/`
3. Download `trace-data-pipeline.zip`

**Step 3: Extract locally**
- **Windows**: Right-click → Extract All
- **Mac**: Double-click the zip file
- **Linux**: `unzip trace-data-pipeline.zip`

**Step 4: Clean up (optional)**
```bash
# On WRDS Cloud
rm /scratch/{institution}/trace-data-pipeline.zip
```

**Placeholders:**
- `{wrds_username}`: Your WRDS username
- `{institution}`: Your institution's scratch folder (e.g., `wharton`, `chicago`, `nyu`)
- `{local_destination}`: Path on your local machine (e.g., `~/Downloads` or `C:\Users\YourName\Downloads`)

For detailed instructions, see [QUICKSTART.md](QUICKSTART.md#download-results-to-your-local-machine).

---

## Troubleshooting

### My job keeps failing with memory errors
**Solutions** (see "How do I adjust memory usage?" above for detail):
1. Lower `TARGET_ROWS_PER_CHUNK` in `stage0/_trace_settings.py` — this, not `chunk_size`,
   is what sizes a Stage 0 chunk.
2. Lower `CONCURRENCY` for that member; each worker holds a chunk.
3. Raise `MEM_PER_SLOT_GB`, keeping `slots x mem <= 48 GB`. Do NOT add `#$ -l m_mem_free`
   to the job scripts — `run_pipeline.sh` passes the request on the command line, which
   wins.

### I'm getting "Permission denied" errors
**Solution**: Make scripts executable:
```bash
chmod +x run_pipeline.sh download_inputs.sh stage0/run_*.sh
```

### I'm getting "bad interpreter" or `^M` errors
**Solution**: Convert Windows line endings to Unix:
```bash
sed -i 's/\r$//' run_pipeline.sh
# Or fix all shell scripts:
find . -name "*.sh" -exec sed -i 's/\r$//' {} \;
```

### My job is stuck in queue (status 'qw')
**Possible causes**:
- WRDS resource constraints during peak hours
- Queue is full
- Requested resources not available

**Check status**:
```bash
qstat          # View all your jobs
qstat -j <job_id>  # View specific job details
```

### No data is being returned
**Check**:
1. Verify date ranges in `_trace_settings.py`
2. Confirm WRDS entitlements (Enhanced/Standard/144A)
3. Review logs for SQL errors: `cat stage0/logs/01_enhanced.err`
4. Ensure FISD filters aren't too restrictive

### Empty output files
**Check**:
1. Date ranges are correct
2. FISD filters aren't excluding all bonds
3. Review audit files to see where rows were dropped

### ImportError: No module named 'wrds' (or other package)
**Solution**: Install missing packages:
```bash
pip install --user wrds pandas numpy pandas-market-calendars
# Or install all requirements:
python -m pip install --user -r requirements.txt
```

### My `stage1.err` is full of ValueError tracebacks -- did the run fail?

Almost certainly not. On the WRDS Cloud (Python 3.14) a completed Stage 1 leaves a large
`stage1/logs/stage1.err` containing progress bars and roughly a hundred copies of this:

```
Traceback (most recent call last):
  File ".../python3.14/multiprocessing/resource_tracker.py", line 446, in main
    raise ValueError(
ValueError: Cannot register /dev/shm/joblib_memmapping_folder_... for automatic cleanup:
            unknown resource type folder
```

This comes from Python's `resource_tracker`, which does not recognise the `folder`
resource type that `joblib` registers when it memory-maps arrays between workers. It is
raised in the tracker process, not in the pipeline, and it appears once per joblib
worker pool -- so the count scales with how many parallel passes Stage 1 ran, not with
anything about your data. joblib cleans up its own temporary folders regardless.

**How to confirm your run was fine** -- check the output rather than the log:

```bash
python3 - <<'PY'
import pandas as pd, glob
f = sorted(glob.glob("stage1/data/stage1_*.parquet"))[-1]
df = pd.read_parquet(f, columns=["cusip_id", "trd_exctn_dt", "sp_rating"])
print(f, len(df), "rows,", df.cusip_id.nunique(), "cusips,",
      df.trd_exctn_dt.min(), "->", df.trd_exctn_dt.max())
PY
```

A healthy full run is tens of millions of rows spanning 2002-07 to your data frontier.
For reference, the 2026-09-10 production run wrote 31,412,833 rows from 2002-07-01 to
2025-12-31, and its `stage1.err` held 689 lines that were *all* this one harmless pattern
(683 on the 2026-09-09 run).

**What WOULD indicate a real failure:** a non-zero exit status from the job, a `stage1.out`
that stops mid-step, or a missing/short `stage1_YYYYMMDD.parquet`. Stage 0's `.err` files
should be empty; if one is not, read it.

### The report generation job (build_reports) never starts
**Explanation**: This is normal. `./run_pipeline.sh` holds the report job until every
stage-0 job it submitted completes. Stage 1 does NOT wait for the reports — since v2.2.2
it holds on the stage-0 data jobs and runs alongside the report job, because it reads
only the member panels and the FISD file, never anything the reports produce.

Not to be confused with `qw` **without** the `h`: that means the job is queued but not
held, and if it stays there the resource request is the thing to check. Stage 0 asks for
one slot per worker (`-pe onenode 5` for Enhanced by default) -- lower the member's
`CONCURRENCY` in `stage0/_trace_settings.py` and resubmit if your queue cannot place it.

**Check progress**:
```bash
qstat  # Look for 'hqw' status - this means it's waiting for dependencies
```

### LaTeX compilation errors in reports
**Possible issues**:
- Missing figure files
- Missing `references.bib`
- Incomplete LaTeX installation

**Solution**: Ensure all figure files generated before compiling, or set `STAGE0_OUTPUT_FIGURES = False` in `config.py`.

### The pipeline shows a disk space warning - what should I do?
Before it starts, `run_pipeline.sh` runs `check_disk_space.sh`. On WRDS the limit that matters is
your HOME QUOTA, 10 GB on most accounts. The check takes that limit from `quota` and then
MEASURES what your home directory holds, with `du`, at that moment.

**Enough room (4 GB or more):**
```
=== DISK SPACE CHECK ===
[info] Home quota limit: 10 GB
[info] Measuring what your home directory holds right now ...
[info] In use: 2.30 GB, measured now with du
[info] Available: 7.70 GB (measured now with du)
[ok] Enough room for a run (7.70 GB available, 4.0 GB needed)
```

**Not enough:** the check stops, shows the available space, and lists the largest things in
your home directory so you can see what to delete.
```
║  NOT ENOUGH DISK SPACE FOR A RUN                               ║
║  Available: 1.50 GB                                            ║
║  Needed:    4.0 GB                                             ║

The largest things in your home directory:
      1.43 GB  ~/old_run
      2.86 GB  ~/trace-data-pipeline
```
The usual causes are an earlier run's folder and a `trace-data-pipeline.zip` left in `~`. Zip to
`/scratch` instead, as the download steps in `QUICKSTART.md` show. Delete what you do not need
and run `./run_pipeline.sh` again. There is nothing to wait for, because the check measures the
disk each time.

To start anyway: `FORCE_RUN=1 ./run_pipeline.sh`. It reports the same numbers and does not stop.
A run that fills the disk part-way fails or writes a truncated file, so use this only when you
know the space is there.

### `quota` says my home directory is nearly full, and I just deleted the old run
`quota` is not live. WRDS refreshes the USED figure every 30 minutes, and its output says when it
last did ("Last updated: ..."). For up to half an hour after you delete a folder, `quota` still
counts it. The pipeline's own check does not read that figure. It measures with `du`, and when
the two disagree it prints both and says which it used. To see the real number yourself:
```bash
du -sh ~
```

### How do I check my WRDS disk quota?
```bash
quota  # Shows Home and Scratch directory usage/limits
```

Example output:
```
DIRECTORY  USED / LIMIT
    Home:  7.88GB / 10GB
 Scratch:  0B / 500GB
```

**Home directory** is where your project lives (10 GB limit on most WRDS accounts).
**Scratch directory** is shared storage (500 GB limit, shared with institution).

---

## Performance

### How can I make processing faster?
**Options**:

1. **Increase chunk size** (if you have enough memory) — fewer, larger chunks:
   ```python
   TARGET_ROWS_PER_CHUNK = 1_500_000   # default 750_000
   ```
   Or raise `CONCURRENCY` to pull more chunks at once, within the WRDS connection
   ceiling of 7 held simultaneously (`tests/probe_wrds_connections.py` measures it).

2. **Disable Stage 0 error plot generation**:
   - Set `STAGE0_OUTPUT_FIGURES = False` in `config.py`
   - Worth much less than it used to be. The whole report job now takes ~15 minutes and
     runs alongside Stage 1, so switching the figures off saves minutes and takes
     nothing off the critical path.

3. **Memory optimizations (automatic)**:
   - CUSIP columns use category dtype (~75% memory savings)
   - Optimized groupby operations for large datasets
   - Efficient parquet compression

### Can I run multiple datasets simultaneously?
Yes. `./run_pipeline.sh` submits Enhanced and 144A as parallel jobs by default. Standard is
opt-in (`TRACE_MEMBERS`), and when requested it is held until the other two finish so it can
use the whole WRDS connection budget.

---

## Contributing

### How can I contribute?
See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines. Key ways to contribute:
1. **Report bugs** using GitHub issues
2. **Suggest features** via feature requests
3. **Submit pull requests** for bug fixes or enhancements
4. **Improve documentation**
5. **Test on different WRDS environments**

### I found a bug. What should I do?
Create a GitHub issue with:
- Clear bug description
- Steps to reproduce
- Your environment (Python version, WRDS setup)
- Relevant log files
- Expected vs actual behavior

### I have an idea for a new feature
Great! Create a feature request issue explaining:
- What problem it solves
- How it would work
- Who would benefit
- Whether you can help implement it

### How do I test my changes?
1. Test locally if possible
2. Test on WRDS Cloud with sample data
3. Check log files for errors
4. Verify output format matches expectations
5. Run on small date range before full dataset

---

## Academic Use

### How do I cite this pipeline?
See the Citation section in [README.md](README.md). Use both references:

**Primary**:
```
Dickerson, A., Robotti, C., & Rossetti, G. (2026).
The Corporate Bond Factor Replication Crisis.
Working Paper. (Earlier versions circulated as "Common pitfalls in the
evaluation of corporate bond strategies.")
```

**Secondary**:
```
Dickerson, A., & Rossetti, G. (2025). 
Constructing TRACE Corporate Bond Datasets.
Working Paper.
```

### Can I use this for my research?
Absolutely! This project is MIT licensed and designed for academic research. Please cite appropriately.

### Will I be acknowledged for contributions?
Yes! Contributors are acknowledged in:
- The project README
- Release notes for significant contributions
- The academic paper underlying this work (for major contributors)

### What's the connection to PyBondLab?
This pipeline produces clean TRACE data. [PyBondLab](https://github.com/GiulioRossetti94/PyBondLab) is the companion repository for constructing corporate bond asset pricing factors from this data.

Workflow:
1. **trace-data-pipeline** (this repo): raw TRACE → daily panel (Stages 0-1) → monthly panel with 108 signals (Stage 2)
2. **PyBondLab**: monthly panel → sorted portfolios and factors (Stage 3 calls it for the paper's exhibits)

### Is this part of a larger project?
Yes! This is part of the [Open Bond Asset Pricing](https://openbondassetpricing.com/) project, which aims to provide open-source tools for corporate bond research.

---

## Support

### Where can I get help?
- **GitHub Issues**: [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)
- **Email**: alexander.dickerson1@unsw.edu.au
- **Documentation**: 
  - [Main README](README.md)
  - [Stage 0 README](stage0/README_stage0.md)
  - [Stage 1 README](stage1/README_stage1.md)
  - [Stage 2 README](stage2/README_stage2.md)
  - [Stage 3 README](stage3/README_stage3.md)
  - [Contributing Guide](CONTRIBUTING.md)

### How quickly will I get a response?
We aim to respond to issues within 1-3 business days. For urgent matters, email directly.

### Can I schedule a call to discuss my use case?
Yes! For collaboration or complex use cases, email alexander.dickerson1@unsw.edu.au to arrange a discussion.

---

## Stage 1, 2 & 3

### Is Stage 1 available?
**Yes!** Stage 1 is now in **public beta**. It enriches Stage 0 daily panels with:
- Bond characteristics from FISD (coupon, maturity, issuer, etc.)
- Bond analytics via QuantLib (duration, convexity, YTM, credit spreads)
- Credit ratings from S&P and Moody's
- Equity identifiers (PERMNO, PERMCO, GVKEY)
- Ultra-distressed filters
- Fama-French industry classifications

See [stage1/QUICKSTART_stage1.md](stage1/QUICKSTART_stage1.md) to get started.

### How do I run Stage 1?
The easiest way is to use `./run_pipeline.sh` which automatically runs both Stage 0 and Stage 1.

Alternatively, run Stage 1 manually (after Stage 0 completes):
```bash
qsub stage1/run_stage1.sh
```

### What about Stage 2?
Stage 2 builds the monthly asset-pricing panel from your Stage 1 output: **145 columns**
per bond-month, covering

- spreads, yields and size; value; momentum and reversal
- illiquidity measures; volatility and downside risk
- rolling 36-month betas on 37 factor models
- portfolio-ready returns, with and without duration adjustment

It runs on **your own machine**, not the WRDS grid, because it reads Stage 1's output
rather than the TRACE tape. Every column is defined in
[stage2/DATA_DICTIONARY.md](stage2/DATA_DICTIONARY.md).

**Status:** complete. The code is in this repository, and published vintages are available
at [openbondassetpricing.com](https://openbondassetpricing.com).

### What is Stage 3, and do I need it?
Stage 3 is what the data was built for, and it is **optional**: Stages 0-2 build the
panel, and the panel is useful on its own.

It turns the Stage-2 monthly panel into portfolio sorts, two uncertainty grids, and
**33 table files and 11 figures** reproducing *The Corporate Bond Factor Replication
Crisis* -- main text, appendix and Internet Appendix -- ending in a single compiled
`stage3/reports/exhibits.pdf`. Twenty-nine of the tables are the paper's; the other four
are Stage 3's own.

```bash
cd stage3
python tools/check_inputs.py     # are the five inputs there and the right shape?
bash run_stage3.sh               # everything, ending in reports/exhibits.pdf
```

Like Stage 2, it runs on your own machine and opens no WRDS connection. 906 s -- about
15 minutes -- on 24 cores, measured on a cold run on 2026-09-12. See [stage3/QUICKSTART_stage3.md](stage3/QUICKSTART_stage3.md).

### Are Stage 3's numbers the paper's printed numbers?
**No, and nothing in Stage 3 compares them to the paper's.** It produces exhibits from
whatever panel Stage 2 built for you; the title page of `exhibits.pdf` says so. A PDF of
tables under familiar captions is exactly the kind of artifact that gets mistaken for the
original, so the document states it rather than leaving you to work it out.

Two exhibits ship in **two variants**, because the published table and the paper's own
definitions differ -- Table B.1 and Table IA.IX. Both are produced; neither is silently
corrected.

### Why does Stage 3 refuse to run Section 5?
Section 5's two uncertainty grids need a PyBondLab build carrying `fast_sorts` and
`anomaly_assay_fast`. The 0.2.0 release the repository pins does not have them, and Stage
3 says so before fanning out rather than letting 108 workers each fail on an import.
Point `PYBONDLAB_DIR` at a build that has them.

Everything else runs on the pinned release: Stage 3 asks the installed engine once at
startup and takes the slow path automatically, with the same numbers. The pin is not
floated to fix this -- Stage 2's factor series depend on it exactly.

### What is redacted in the published panel?
`permco` and `gvkey` are set to null, and `spc_rat`/`mdc_rat` are collapsed to investment
grade (1) versus non-investment-grade and default (11). `permno` is kept. This applies to the
REDISTRIBUTED file only -- a panel you build yourself from your own WRDS subscription keeps
every identifier and the full 1-22 rating scale.

---

**Last updated**: September 2026

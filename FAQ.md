# Frequently Asked Questions (FAQ)

## Table of Contents
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)
- [Academic Use](#academic-use)

- [Stage 1 & 2](#stage-1--2)
- [Support](#support)
---

## Getting Started

### Do I need a WRDS subscription?
Yes, the pipeline requires WRDS access with TRACE Enhanced, Standard, or 144A entitlements, plus FISD and ratings data for Stage 1. You can check your entitlements by logging into WRDS and viewing your subscriptions.

### Can I run this on my local machine?
Both Stage 0 and Stage 1 are designed for WRDS Cloud due to database access requirements, though technically they can run locally with proper WRDS connectivity. WRDS Cloud is recommended for optimal performance.

### How long does processing take?
Using `./run_pipeline.sh` (complete automated pipeline):
- **Pre-stage (data downloads)**: ~5 minutes
- **Stage 0 (Enhanced TRACE)**: was ~4 hours serial. Since v2.2.0 it pulls 5 CUSIP
  chunks at once over separate WRDS connections -- measured 4.5x on the chunk loop
  itself, so expect materially less, though how much depends on how fetch and clean
  divide up on the day.
- **Stage 0 (Rule 144A)**: ~30-60 minutes, running at the same time as Enhanced
- **Stage 0 (Standard TRACE)**: ~30-60 minutes, and OPT-IN since v2.2.0. When requested
  it is scheduled after the other two rather than beside them.
- **Stage 0 (Report generation)**: ~30-60 minutes
- **Stage 1 (Bond analytics)**: ~2 hours

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
Stage 0 error plots are **very slow** to generate (30+ minutes for Enhanced TRACE). Control this in `config.py`:
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

# Read a single dataset
df = pd.read_parquet('enhanced/enhanced_20250120.parquet')

# Read multiple files
import glob
files = glob.glob('enhanced/*.parquet')
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
```

Using R:
```r
library(arrow)
df <- read_parquet('enhanced/enhanced_20250120.parquet')
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

Useful for understanding which bonds had price errors.

### What is the ultra_distressed_cusips CSV file? (Stage 1)
Stage 1 exports `stage1/data/ultra_distressed_cusips_{date}.csv` which contains all bonds flagged by the ultra-distressed filter. Each row shows:

**Columns**:
- `cusip_id`: Bond identifier
- `total_observations`: Total trades for this CUSIP
- `flagged_observations`: Number of flagged trades
- `pct_flagged`: Percentage flagged (%)
- `flag_anomalous_price`: Count of anomalous price flags
- `flag_upward_spike`: Count of upward spike flags
- `flag_plateau_sequence`: Count of plateau sequence flags
- `flag_intraday_inconsistent`: Count of intraday inconsistent flags
- `first_trade_date`: Earliest trade date
- `last_trade_date`: Latest trade date

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

The pipeline generates a large folder (~6 GB) with hundreds of files. **Zip the folder first** for faster, more reliable downloads.

**Step 1: Create zip file on WRDS (via SSH)**

```bash
# Connect to WRDS
ssh {wrds_username}@wrds-cloud.wharton.upenn.edu

# Zip to scratch space (your home directory has limited quota)
cd /scratch/{institution}/
zip -r trace-data-pipeline.zip ~/trace-data-pipeline/
```

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
3. Review logs for SQL errors: `cat logs/01_enhanced.err`
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
Before running the pipeline, it checks your WRDS quota. You'll see one of these:

**Sufficient space (>= 4 GB):**
```bash
=== DISK SPACE CHECK ===
[info] WRDS Quota - Home directory: 3.2 GB used / 10 GB limit
[info] Available space: 6.8 GB
[ok] Sufficient disk space available (6.8 GB >= 4.0 GB)
```
✅ **Proceed normally** - you have enough space.

**Insufficient space (< 4 GB):**
```bash
=== DISK SPACE CHECK ===
[info] WRDS Quota - Home directory: 7.88 GB used / 10 GB limit
[info] Available space: 2.12 GB
╔════════════════════════════════════════════════════════════════╗
║                         ⚠️  WARNING  ⚠️                       ║
║  INSUFFICIENT DISK SPACE DETECTED                              ║
║  Available: 2.12 GB                                            ║
║  Required:  At least 4.0 GB recommended                        ║
╚════════════════════════════════════════════════════════════════╝
[error] Exiting due to insufficient disk space.
```
⚠️ **Action required:**

**Option 1: Free up space (recommended)**
```bash
# Check what's using space
du -h ~/ | sort -h | tail -20

# Remove old files/logs
rm -rf ~/old_data/
rm -f ~/stage0/logs/*.log  # Old log files
rm -f ~/stage1/logs/*.log
```

**Option 2: Override warning (advanced users only)**
```bash
# Only if you're confident the pipeline won't exceed available space
FORCE_RUN=1 ./run_pipeline.sh
```

**Why this matters:** The pipeline generates large intermediate files. Running out of disk space mid-processing can corrupt output files or cause job failures.

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

2. **Disable Stage 0 error plot generation** (saves 30-60 minutes):
   - Set `STAGE0_OUTPUT_FIGURES = False` in `config.py`

3. **Memory optimizations (automatic)**:
   - CUSIP columns use category dtype (~75% memory savings)
   - Optimized groupby operations for large datasets
   - Efficient parquet compression

### Can I run multiple datasets simultaneously?
Yes! That's exactly what `./run_pipeline.sh` does. It submits Enhanced, Standard, and 144A as parallel jobs.

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

Use the bug report template when creating the issue.

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
Dickerson, A., Robotti, C., & Rossetti, G. (2025). 
Common pitfalls in the evaluation of corporate bond strategies.
Working Paper.
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
1. **trace-data-pipeline** (this repo): Clean raw TRACE → daily panels
2. **PyBondLab**: Daily panels → bond characteristics → factor portfolios

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
  - [Contributing Guide](CONTRIBUTING.md)

### How quickly will I get a response?
We aim to respond to issues within 1-3 business days. For urgent matters, email directly.

### Can I schedule a call to discuss my use case?
Yes! For collaboration or complex use cases, email alexander.dickerson1@unsw.edu.au to arrange a discussion.

---

## Stage 1 & 2

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
Stage 2 is **in development** and will produce monthly panels with:
- 50+ bond characteristic signals
- Credit risk factors
- Liquidity measures
- Momentum and reversal signals
- Portfolio-ready outputs

**Expected release:** Coming soon

### Can I beta test Stage 2?
Yes! Email alexander.dickerson1@unsw.edu.au to express interest in beta testing Stage 2.

---

**Last updated**: September 2026

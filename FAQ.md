# Frequently asked questions (FAQ)

## Table of Contents
- [Getting started](#getting-started)
- [Working with an AI assistant](#working-with-an-ai-assistant)
- [Configuration](#configuration)
- [Output files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)
- [Academic use](#academic-use)
- [Support](#support)
- [Stages 1 to 4](#stages-1-to-4)

## Getting started

### Do I need a WRDS subscription?
Yes. The pipeline needs WRDS access with a TRACE Enhanced, Standard or 144A entitlement, plus
FISD and ratings data for Stage 1. To check your entitlements, log in to WRDS and look at your
subscriptions.

### Which parts run where?
The pipeline runs on two machines, and the hand-off is a file you copy yourself.

| | Where | Why |
|---|---|---|
| Stages 0 and 1 | WRDS Cloud | They read the raw TRACE transactions, which are a WRDS database |
| The hand-off | You | Zip on WRDS, `scp` down (~5.6 GB) |
| Stage 2 | Your own computer | It reads Stage 1's output. Its first run connects to WRDS once to fetch and cache a few series |
| Stage 3 | Your own computer | It reads Stage 2's panel and Stage 1's daily file. It is optional and needs no WRDS connection |
| Stage 4 | Your own computer | It reads Stage 2's panel and blocks. It is optional and needs no WRDS connection |

### Can I run Stages 0 and 1 on my own machine?
You can, with a working WRDS connection, but you would pull hundreds of millions of trades
across the internet. `run_pipeline.sh` submits jobs to the WRDS grid, so you would have to
call the stage scripts directly instead. Use WRDS Cloud.

### Can I skip Stages 0 and 1 and just download the Stage 1 file?
No, not for Stage 2. The published Stage 1 download has its rating columns removed, because
agency ratings are licensed. Stage 2 keeps only bond-months that have a rating, so it cannot
use that file, and it refuses it at start-up with a message that says so. The download is
there for people who want the daily panel itself.

### How long does processing take?
With `./run_pipeline.sh`, the complete automated pipeline, the steps take these times.
- The pre-stage (data downloads) takes about 5 minutes.
- Stage 0 for Enhanced TRACE takes about 2 to 2.6 hours. It took about 4 hours when it ran
  serially. Since v2.2.0 it pulls 5 CUSIP chunks at once over separate WRDS connections, which
  is about twice as fast (2.0 h on 2026-09-09, 2.6 h on 2026-09-21).
- Stage 0 for Rule 144A takes about 40 minutes (38 on 2026-09-21) and runs at the same time as
  Enhanced.
- Stage 0 for Standard TRACE takes about 30 to 60 minutes and has been opt-in since v2.2.0.
  When requested, it is scheduled after the other two, not beside them.
- The Stage 0 report generation has taken about 20 minutes since v2.2.2 (22 on 2026-09-21). It
  no longer blocks Stage 1, and the two run side by side. It took about 50 minutes before that
  release.
- Stage 1 (bond analytics) takes about 2.4 to 2.7 hours (2.7 on 2026-09-21).

End to end, that is about 5 hours. The 2026-09-21 run took 5.3 h.

Then, on your own machine, Stage 2 takes about 8 to 18 minutes (the first run also downloads
its inputs), Stage 3 about 15 to 20 minutes on 24 cores, and Stage 4 about 8 minutes.

### What if I only want Enhanced TRACE?
Set which datasets to process in `config.py`, which is read by `run_pipeline.sh`, the report
job and Stage 1.
```python
TRACE_MEMBERS = ["enhanced"]  # Process only Enhanced TRACE
```

For one run, without editing anything, use `TRACE_MEMBERS=enhanced ./run_pipeline.sh`.
Submitting a stage 0 job script by hand (`qsub stage0/run_enhanced_trace.sh`, from the
repository root) skips the cores and memory that `run_pipeline.sh` requests for it, and on a
fresh clone it needs `mkdir -p stage0/logs` first. Prefer the line above.

### What Python version do I need?
Python 3.10 or higher on WRDS for Stages 0 and 1 (the WRDS Cloud runs 3.14), and 3.11 to 3.13
on your own computer for Stages 2 to 4, which install with two lines.
```bash
python -m pip install -r requirements-local.txt
python -m pip install --no-deps pybondlab==0.3.0
```
PyBondLab 0.3.0 declares `numpy<2` and this repository installs numpy 2, so it goes in without
its dependency list, and `requirements-local.txt` says why that is safe. Stages 2 to 4 stop at
start-up and print these two lines if PyBondLab is missing or is a different version.
This line prints your Python version.
```bash
python --version
```

### Do I need to install anything besides Python packages?
For Stage 0 you need a WRDS subscription with TRACE access, SSH access to WRDS Cloud, and the
required Python packages (installed from `requirements.txt`).

Stage 3 needs one thing that is not Python, pdflatex (TeX Live or MiKTeX). The last step uses
it to compile every exhibit into one PDF. When it is missing, `stage3/tools/check_inputs.py`
warns and does not fail. Without it you still get every table and figure as a file, but you
do not get `reports/exhibits.pdf`.

## Working with an AI assistant

### Can I run this with Claude Code, Codex or Copilot?
Yes. Open the assistant in the repository folder and it reads [AGENTS.md](AGENTS.md) by itself.
That file says what each stage does, what to run, what to check and which mistakes to avoid.
Skills walk each task step by step. They are `onboard`, `run-wrds`, `build-panel`,
`reproduce-exhibits`, `build-factors`, `explain-pipeline` and `add-a-column`. In Claude Code type
`/build-panel`, and in Codex type `$build-panel`. You can also ask in plain words, for example
"set this repository up", "build the monthly panel" or "what is `cs_sprd` and where is it
computed?".

### What does `doctor.py` do?
It says what is ready and what to run next. `python doctor.py` checks your own computer for
stages 2 to 4, and `python3 doctor.py --wrds` checks the WRDS login node for stages 0 and 1. It
asks each stage through that stage's own check. It downloads nothing, opens no WRDS connection
and writes nothing of its own, so it is always safe to run.

### Where is a column computed?
[TAGS.md](TAGS.md) lists, for every column of the daily and monthly panels, every cleaning filter
and every rule the code enforces, the file that holds it. Search that file for the tag, for
example `git grep -n "tag:col.cs_sprd"`. The column's definition is in the stage's
`DATA_DICTIONARY.md`.

## Configuration

### Where do I configure settings?
The shared settings come first, then each stage's own.

1. The shared settings are in `config.py`, in the root.
   - `WRDS_USERNAME` is your WRDS username.
   - `OUTPUT_FORMAT` is the output file format. It accepts `"parquet"` only (see the FAQ entry
     below).
   - `AUTHOR` is your name.
   - `TRACE_MEMBERS` says which datasets to process (enhanced, standard, 144a).
     `run_pipeline.sh`, the report job and Stage 1 read it.
   - `STAGE0_OUTPUT_FIGURES` controls whether Stage 0 draws its error plots, which can be slow.

2. The Stage 0 settings are in `stage0/_trace_settings.py`.
   - Filter switches, FISD parameters, chunk sizes
   - Decimal-shift and bounce-back parameters

3. The Stage 1 settings are in `stage1/_stage1_settings.py`.
   - Date cutoffs, performance tuning (cores, chunks)
   - Ultra-distressed filter configuration
   - Stage 1 always generates reports and figures. It has no toggle for them.

### Do I need to configure anything to get started?
The minimum is your WRDS username, set in `config.py`.
```python
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_id")
```

Most other settings are detected automatically.
- `ROOT_PATH` comes from the working directory.
- `STAGE0_DATE_STAMP` comes from the Stage 0 output files.
- `N_CORES` follows the slots the job was granted (`$NSLOTS`), else 4.

### How do I change the date range?

In Stage 0, Standard and 144A take a `start_date` in `PER_DATASET` in
`stage0/_trace_settings.py`. Enhanced does not. It always starts at 2002-07-01, and giving it
a `start_date` fails. To drop early Enhanced years, filter the Stage 1 output instead.
```python
# Keep each member's existing keys. Do not replace the dict.
# Dropping `n_workers` from "enhanced" returns Stage 0 to the serial,
# ~4-hour path without an error, because the engine default is n_workers=1.
PER_DATASET = {
    "enhanced": dict(n_workers=workers("enhanced")),
    "standard": dict(start_date="2024-10-01", data_type="standard",
                     n_workers=workers("standard")),
    "144a":     dict(start_date="2002-07-01", data_type="144a",
                     n_workers=workers("144a")),
}
```

For Stage 1, edit `stage1/_stage1_settings.py`. The default, `"auto:complete"`, ends the
sample at the last month every source covers in full. A literal date overrides it.
```python
DATE_CUT_OFF = "2023-12-31"  # Only include data through this date
```

### How do I adjust memory usage?

Two things set Stage 0's memory, the size of a chunk and the number of workers that hold one
at a time.

#### Chunk size

Since v2.2.0 chunks are packed to a trade-row target, not a CUSIP count. Lower the target in
`stage0/_trace_settings.py`.

```python
TARGET_ROWS_PER_CHUNK = 400_000    # default 750_000
```

For a single run, without editing anything, use `STAGE0_TARGET_ROWS=400000 ./run_pipeline.sh`.

`chunk_size` still exists and still means CUSIPs per chunk, but it no longer sizes Stage 0's
work. The data-report job reads it for its own chunking. Changing it will not affect Stage 0
memory.

#### Worker count

Each worker holds one chunk, so the worker count multiplies the chunk size. Lower
`CONCURRENCY` in `stage0/_trace_settings.py`. `STAGE0_WORKERS=N ./run_pipeline.sh` sets
the count for one run, for every member at once, 144A's included. The connection check and
the `qsub` request both use it.

#### Requesting more memory

More memory is not requested in the job scripts. `run_pipeline.sh` computes the request per
member and passes it on the `qsub` command line, which overrides anything in the `.sh`. Change
`MEM_PER_SLOT_GB` in `stage0/_trace_settings.py`.

```python
MEM_PER_SLOT_GB = {"enhanced": 8, "standard": 8, "144a": 16}
```

`m_mem_free` is charged per slot, so the total is `slots x mem` and must stay within the WRDS
caps of 8 cores and 48 GB per job. `qsub_resources()` refuses to emit anything over them,
because an over-request gives no error and the job waits in the queue forever without a
message.

### Can I disable certain filters?
Yes. Edit `FILTER_SWITCHES` in `stage0/_trace_settings.py`.

```python
FILTER_SWITCHES = dict(
    dick_nielsen            = True,
    decimal_shift_corrector = True,
    bounce_back_filter      = False,  # Disable this filter
    # ... etc
)
```

### How do I change which bonds are included?
Change `FISD_PARAMS` in `stage0/_trace_settings.py`.

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

Stage 0 itself can write `.csv.gzip`, but Stage 1 and the report builder both call
`pd.read_parquet` on a hard-coded `*.parquet` name, so a CSV run produces stage-0 files
that nothing downstream can open. Before v2.2.3 that failed several hours in, with a
misleading *"Expected: stage0/<member>/trace_<member>_<stamp>.parquet"*. It now fails
immediately.

To get a CSV copy of a finished dataset, convert it afterwards.
```python
import pandas as pd
pd.read_parquet("stage1/data/stage1_YYYYMMDD.parquet").to_csv("stage1.csv.gz", index=False)
```

### How do I control Stage 0 error plot generation?
With figures on, the report job re-cleans the flagged bonds and draws their plots, which took
about 20 minutes on the 2026-09-21 run. It runs beside Stage 1, so it delays nothing. Control
it in `config.py`.
```python
STAGE0_OUTPUT_FIGURES = False  # Tables only: skips the re-clean and the plots
STAGE0_OUTPUT_FIGURES = True   # Tables and error plots
```

Stage 1 always generates reports and figures regardless of this setting, because they are
essential for data quality.

### How do I change which datasets to process across all stages?
Edit `TRACE_MEMBERS` in `config.py` once. `run_pipeline.sh`, the report job and Stage 1 read
it. Stage 2 builds on whatever Stage 1 produced, and Stages 3 and 4 read Stage 2's panel.
```python
TRACE_MEMBERS = ["enhanced"]                       # Enhanced only
TRACE_MEMBERS = ["enhanced", "standard"]           # Two datasets
TRACE_MEMBERS = ["enhanced", "144a"]              # The DEFAULT
TRACE_MEMBERS = ["enhanced", "standard", "144a"]  # All three (Standard is opt-in)
```

## Output files

### What format are the output files?
Output files are in Parquet format, the only one `OUTPUT_FORMAT` accepts. Parquet is a
compressed columnar storage format. It compresses well (about 5 to 10 times smaller than CSV),
loads faster, preserves data types and works with pandas, R and many other tools.

### How do I read the output files?
With Python and pandas.
```python
import pandas as pd

# Read a single dataset (use your own date stamp)
df = pd.read_parquet('stage0/enhanced/trace_enhanced_20260910.parquet')

# Read every member's daily panel. The pattern skips the FISD, audit and CUSIP-list files
import glob
files = glob.glob('stage0/*/trace_*_[0-9]*.parquet')
files = [f for f in files if 'fisd' not in f]
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
```

With R.
```r
library(arrow)
df <- read_parquet('stage0/enhanced/trace_enhanced_20260910.parquet')
```

### What columns are in the output files?
The datasets (Enhanced, Standard, 144A) all have the same 21 columns, though their order
differs.

#### Identifiers

- `cusip_id` is the 9-character CUSIP identifier
- `trd_exctn_dt` is the trade execution date

#### Price metrics

- `prc_ew` is the equal-weighted price
- `prc_vw` is the volume-weighted price (dollar)
- `prc_vw_par` is the volume-weighted price (par)
- `prc_first` is the first trade price of the day
- `prc_last` is the last trade price of the day
- `prc_lo` is the low price of the day
- `prc_hi` is the high price of the day

#### Trade-time metrics

These are in seconds since midnight.

- `time_ew` is the equal-weighted mean execution time
- `time_last` is the execution time of the last trade

#### Volume metrics

These are in millions.

- `qvolume` is the par volume
- `dvolume` is the dollar volume

#### Bid/ask metrics

These are on the dealer side (see the note below).

- `prc_bid` is the dealer bid, value-weighted (the dealer buys here, so the customer sells)
- `prc_ask` is the dealer ask, value-weighted (the dealer sells here, so the customer buys)
- `bid_last` is the last dealer-bid price of the day
- `bid_time_ew` is the equal-weighted mean time of dealer-bid trades
- `bid_time_last` is the time of the last dealer-bid trade

#### Count metrics

- `trade_count` is the number of trades, customer and inter-dealer
- `bid_count` is the number of dealer buys (= customer sells)
- `ask_count` is the number of dealer sells (= customer buys)

TRACE reports the dealer's side. The bid/ask split is
`rpt_side_cd == 'B'` (dealer buying) and `'S'` (dealer selling), both restricted to
customer counterparties (`cntra_mp_id == 'C'` in Enhanced, `contra_party_type == 'C'` in
Standard and 144A). The prices, volumes and `trade_count` use all trades. So a customer buy
appears as `rpt_side_cd == 'S'`. Getting this backwards is the commonest TRACE bug.

### What do the audit files contain?
Audit files track row counts at each filter stage.
- `dick_nielsen_filters_audit_*.parquet` records the effects of the Dick-Nielsen filters
- `drr_filters_audit_*.parquet` has the row counts before and after each cleaning step (the
  start, Dick-Nielsen as one row, then decimal shift, trading time, calendar, price, volume,
  bounce-back, yield, amount outstanding, maturity, initial price)
- `fisd_filters_*.parquet` is the audit of the FISD universe construction

They show how many transactions each cleaning step removed.

### What are the CUSIP list files?
These files identify the bonds that were corrected.
- `decimal_shift_cusips_*.parquet` lists the CUSIPs with decimal-shift corrections
- `bounce_back_cusips_*.parquet` lists the CUSIPs with bounce-back flags
- `init_price_cusips_*.parquet` lists the CUSIPs with a removed initial-price error

Use them to see which bonds had price errors.

### What is the ultra_distressed_cusips CSV file? (Stage 1)
Stage 1 exports `stage1/data/ultra_distressed_cusips_{date}.csv`. Only the flagged bonds have
numbers, so filter on `flagged_observations > 0`. The file also holds a row for every other
CUSIP code the run saw, with zeros and no dates, so those rows say nothing about that bond. In
the 2026-09-21 run the file has 76,495 rows, of which 1,045 CUSIPs have 10,182 flagged
bond-days (0.03% of all bond-days).

The file has these columns.
- `cusip_id` is the bond identifier
- `total_observations` is the number of bond-days for this CUSIP (0 on the unflagged rows)
- `flagged_observations` is the number of bond-days the filter flagged
- `pct_flagged` is the percentage flagged (%)
- `first_trade_date` is the earliest trade date
- `last_trade_date` is the latest trade date

The file does not break the count down by filter. The per-filter flags are dropped before
export to save memory.

Use the file to pick out problematic bonds for manual review, to cross-reference with other
bond characteristics (ratings, maturity, etc.), to understand filtering patterns across the
bond universe, and for quality control and diagnostics.

### Where are the reports saved?
Reports are saved in `data_reports/[enhanced|standard|144a]/`. Each folder has a LaTeX source
file (`.tex`), a bibliography file (`references.bib`) and the figures (`.pdf` files). You can
compile the LaTeX to PDF or view the figures directly.

### `git pull` refuses after I unzipped the run on Windows. Why?

Unzipping on Windows drops the executable bit from every `.sh` file, so git counts each one as
changed and stops with "Please commit your changes or stash them before you merge". Nothing in
the files changed. Run `git config core.fileMode false` in the folder once, then pull again.

### How do I download files from WRDS Cloud?

The pipeline produces a large folder (~5.6 GB) with hundreds of files. Zip the folder first. The
download is then faster and more reliable.

#### Step 1. Create the zip file on WRDS (over SSH)

```bash
# Connect to WRDS
ssh {wrds_username}@wrds-cloud.wharton.upenn.edu

# Zip to scratch space (your home directory has limited quota).
# Run from ~ on a RELATIVE path: zip stores the path you give it, so an absolute
# ~/trace-data-pipeline/ makes the archive extract four directories deep.
cd ~
zip -r "/scratch/$(basename "$(dirname "$HOME")")/trace-data-pipeline.zip" trace-data-pipeline/
```
`$(basename "$(dirname "$HOME")")` is your institution code, the folder between `/home` and
your username. Type the line as written, because nothing in it is a placeholder. In the `scp`
line below, `{institution}` is a placeholder. Replace it, braces included, with that code
(`echo $HOME` on WRDS shows it).

#### Step 2. Download the zip file (from your local machine)

On Windows, run this in PowerShell or Terminal.
```powershell
scp {wrds_username}@wrds-cloud.wharton.upenn.edu:/scratch/{institution}/trace-data-pipeline.zip "{local_destination}"
```

On Mac or Linux, run this.
```bash
scp {wrds_username}@wrds-cloud.wharton.upenn.edu:/scratch/{institution}/trace-data-pipeline.zip "{local_destination}"
```

On Windows, WinSCP is a graphical alternative.
1. Connect to `wrds-cloud.wharton.upenn.edu`
2. Navigate to `/scratch/{institution}/`
3. Download `trace-data-pipeline.zip`

#### Step 3. Extract locally
- On Windows, right-click the file and choose Extract All
- On Mac, double-click the zip file
- On Linux, run `unzip trace-data-pipeline.zip`

#### Step 4. Clean up (optional)
```bash
# On WRDS Cloud
rm /scratch/{institution}/trace-data-pipeline.zip
```

Replace the placeholders as follows.
- `{wrds_username}` is your WRDS username
- `{institution}` is your institution's scratch folder (for example `wharton`, `chicago` or
  `nyu`)
- `{local_destination}` is a path on your local machine (for example `~/Downloads` or
  `C:\Users\YourName\Downloads`)

For detailed instructions, see [QUICKSTART.md](QUICKSTART.md#download-results-to-your-local-machine).

## Troubleshooting

### My job keeps failing with memory errors
Try these (see "How do I adjust memory usage?" above for the detail).
1. Lower `TARGET_ROWS_PER_CHUNK` in `stage0/_trace_settings.py`. That setting sizes a Stage 0
   chunk, and `chunk_size` does not.
2. Lower `CONCURRENCY` for that member, because each worker holds a chunk.
3. Raise `MEM_PER_SLOT_GB`, keeping `slots x mem <= 48 GB`. Do not add `#$ -l m_mem_free`
   to the job scripts. `run_pipeline.sh` passes the request on the command line, and the
   command line wins.

### I'm getting "Permission denied" errors
Make the scripts executable.
```bash
chmod +x run_pipeline.sh download_inputs.sh stage0/run_*.sh
```

### I'm getting "bad interpreter" or `^M` errors
Convert the Windows line endings to Unix ones.
```bash
sed -i 's/\r$//' run_pipeline.sh
# Or fix all shell scripts:
find . -name "*.sh" -exec sed -i 's/\r$//' {} \;
```

### My job is stuck in queue (status 'qw')
Possible causes are WRDS resource constraints during peak hours, a full queue, or requested
resources that are not available.

Check the status of your jobs.
```bash
qstat          # View all your jobs
qstat -j <job_id>  # View specific job details
```

### No data is being returned
1. Verify the date ranges in `_trace_settings.py`
2. Confirm your WRDS entitlements (Enhanced/Standard/144A)
3. Review the logs for SQL errors with `cat stage0/logs/01_enhanced.err`
4. Make sure the FISD filters are not too restrictive

### Empty output files
1. Check that the date ranges are correct
2. Check that the FISD filters are not excluding all bonds
3. Review the audit files to see where rows were dropped

### ImportError: No module named 'wrds' (or other package)
Install the missing packages.
```bash
python -m pip install --user -r requirements.txt
```

### My `stage1.err` is full of ValueError tracebacks. Did the run fail?

Almost certainly not. On the WRDS Cloud (Python 3.14) a completed Stage 1 leaves a large
`stage1/logs/stage1.err` containing progress bars and roughly a hundred copies of this
traceback.

```
Traceback (most recent call last):
  File ".../python3.14/multiprocessing/resource_tracker.py", line 446, in main
    raise ValueError(
ValueError: Cannot register /dev/shm/joblib_memmapping_folder_... for automatic cleanup:
            unknown resource type folder
```

It comes from Python's `resource_tracker`, which does not recognise the `folder`
resource type that `joblib` registers when it memory-maps arrays between workers. It is
raised in the tracker process, not in the pipeline, and it appears once per joblib
worker pool. So the count grows with the number of parallel passes Stage 1 ran and has
nothing to do with your data. joblib cleans up its own temporary folders regardless.

To confirm that your run was fine, check the output and not the log.

```bash
python3 - <<'PY'
import pandas as pd, glob
f = sorted(glob.glob("stage1/data/stage1_*.parquet"))[-1]
df = pd.read_parquet(f, columns=["cusip_id", "trd_exctn_dt", "sp_rating"])
print(f, len(df), "rows,", df.cusip_id.nunique(), "cusips,",
      df.trd_exctn_dt.min(), "->", df.trd_exctn_dt.max())
PY
```

A healthy full run is tens of millions of rows spanning 2002-07 to the last complete month
(with the default `DATE_CUT_OFF = "auto:complete"`).
For reference, the 2026-09-21 run wrote 31,344,732 rows from 2002-07-01 to 2025-11-28, and
all 689 lines of its `stage1.err` were this one harmless pattern.

A real failure would show up as a non-zero exit status from the job, a `stage1.out` that stops
mid-step, or a missing or short `stage1_YYYYMMDD.parquet`. Stage 0's `.err` files should be
empty. If one is not, read it.

### The report generation job (build_reports) never starts
This is normal. `./run_pipeline.sh` holds the report job until every stage-0 job it submitted
completes. Stage 1 does not wait for the reports. Since v2.2.2 it holds on the stage-0 data
jobs and runs alongside the report job, because it reads only the member panels and the FISD
file, and nothing the reports produce.

`qw` without the `h` is a different state. It means the job is queued but not held, and if it
stays there the resource request is the thing to check. Stage 0 asks for one slot per worker
(`-pe onenode 5` for Enhanced by default). If your queue cannot place it, lower the member's
`CONCURRENCY` in `stage0/_trace_settings.py` and resubmit.

Check progress with this line.
```bash
qstat  # Look for 'hqw' status - this means it's waiting for dependencies
```

### LaTeX compilation errors in reports
The cause may be missing figure files, a missing `references.bib` or an incomplete LaTeX
installation.

Make sure every figure file was generated before you compile, or set
`STAGE0_OUTPUT_FIGURES = False` in `config.py`.

### The pipeline shows a disk space warning. What should I do?
Before it starts, `run_pipeline.sh` runs `check_disk_space.sh`. On WRDS the limit that matters is
your home quota, 10 GB on most accounts. The check takes that limit from `quota` and then
measures what your home directory holds, with `du`, at that moment.

It asks for 6 GB, because a full run writes about 5.6 GB (the 2026-09-21 run's stage 0 and Stage 1
files).

With enough room (6 GB or more) it prints this.
```
=== DISK SPACE CHECK ===
[info] Home quota limit: 10 GB
[info] Measuring what your home directory holds right now ...
[info] In use: 2.30 GB, measured now with du
[info] Available: 7.70 GB (measured now with du)
[ok] Enough room for a run (7.70 GB available, 6.0 GB needed)
```

Without enough room, the check stops, shows the available space, and lists the largest things
in your home directory so you can see what to delete.
```
║  NOT ENOUGH DISK SPACE FOR A RUN                               ║
║  Available: 1.50 GB                                            ║
║  Needed:    6.0 GB                                             ║

The largest things in your home directory:
      1.43 GB  ~/old_run
      2.86 GB  ~/trace-data-pipeline
```
The usual causes are an earlier run's folder and a `trace-data-pipeline.zip` left in `~`. Zip to
`/scratch` instead, as the download steps in `QUICKSTART.md` show. Delete what you do not need
and run `./run_pipeline.sh` again. There is nothing to wait for, because the check measures the
disk each time.

To start anyway, run `FORCE_RUN=1 ./run_pipeline.sh`. It reports the same numbers and does not
stop. A run that fills the disk part-way fails or writes a truncated file, so use this only
when you know the space is there.

### `quota` says my home directory is nearly full, and I just deleted the old run
`quota` is not live. WRDS refreshes the USED figure every 30 minutes, and its output says when it
last did ("Last updated: ..."). For up to half an hour after you delete a folder, `quota` still
counts it. The pipeline's own check does not read that figure. It measures with `du`, and when
the two disagree it prints both and says which it used. To measure it yourself, run this.
```bash
du -sh ~
```

### How do I check my WRDS disk quota?
```bash
quota  # Shows Home and Scratch directory usage/limits
```

It prints something like this.
```
DIRECTORY  USED / LIMIT
    Home:  7.88GB / 10GB
 Scratch:  0B / 500GB
```

The home directory is where your project lives (10 GB limit on most WRDS accounts).
The scratch directory is shared storage (500 GB limit, shared with your institution).

## Performance

### How can I make processing faster?

1. Increase the chunk size, if you have enough memory. This gives fewer, larger chunks.
   ```python
   TARGET_ROWS_PER_CHUNK = 1_500_000   # default 750_000
   ```
   `CONCURRENCY` is already at the WRDS connection limit with the default members (Enhanced
   5 + 144A 1 = 6, with one of the 7 kept spare), and `run_pipeline.sh` refuses more. Raise
   Enhanced's only if you drop 144A from `TRACE_MEMBERS`
   (`tests/probe_wrds_connections.py` measures your account's limit).

2. Turn off the Stage 0 error plots.
   - Set `STAGE0_OUTPUT_FIGURES = False` in `config.py`
   - This is worth much less than it used to be. The whole report job now takes about 20
     minutes and runs alongside Stage 1, so switching the figures off takes nothing off the
     critical path.

3. The memory optimizations are automatic.
   - CUSIP columns use the category dtype (about 75% memory savings)
   - Optimized groupby operations for large datasets
   - Efficient parquet compression

### Can I run multiple datasets simultaneously?
Yes. `./run_pipeline.sh` submits Enhanced and 144A as parallel jobs by default. Standard is
opt-in (`TRACE_MEMBERS`), and when requested it is held until the other two finish so it can
use the whole WRDS connection budget.

## Contributing

### How can I contribute?
See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines. The main ways to contribute
are these.
1. Report bugs using GitHub issues
2. Suggest features via feature requests
3. Submit pull requests for bug fixes or improvements
4. Improve the documentation
5. Test on different WRDS environments

### I found a bug. What should I do?
Create a GitHub issue with a clear description of the bug, the steps to reproduce it, your
environment (Python version, WRDS setup), the relevant log files, and the expected and the
actual behavior.

### I have an idea for a new feature
Create a feature request issue that explains what problem it solves, how it would work, who
would benefit, and whether you can help implement it.

### How do I test my changes?
1. Test locally if possible
2. Test on WRDS Cloud with sample data
3. Check the log files for errors
4. Verify that the output format matches expectations
5. Run on a small date range before the full dataset

## Academic use

### How do I cite this pipeline?
See the Citation section in [README.md](README.md). Use both references.

The primary reference is this one.
```
Dickerson, A., Robotti, C., & Rossetti, G. (2026).
The Corporate Bond Factor Replication Crisis.
Working Paper. (Earlier versions circulated as "Common pitfalls in the
evaluation of corporate bond strategies.")
```

The secondary reference is this one.
```
Dickerson, A., & Rossetti, G. (2025). 
Constructing TRACE Corporate Bond Datasets.
Working Paper.
```

### Can I use this for my research?
Yes. This project is MIT licensed and designed for academic research. Please cite it.

### Will I be acknowledged for contributions?
Yes. Contributors are acknowledged in the project README, in the release notes for significant
contributions, and, for major contributors, in the academic paper underlying this work.

### What's the connection to PyBondLab?
This pipeline produces clean TRACE data.
[PyBondLab](https://github.com/GiulioRossetti94/PyBondLab) is the companion repository for
constructing corporate bond asset pricing factors from this data.

The work flows in this order.
1. trace-data-pipeline (this repo) turns raw TRACE into the daily panel (Stages 0 and 1), then
   into the monthly panel with 108 signals (Stage 2)
2. PyBondLab turns the monthly panel into sorted portfolios and factors (Stage 3 calls it for
   the paper's exhibits)

### Is this part of a larger project?
Yes. This is part of the [Open Source Bond Asset Pricing](https://openbondassetpricing.com/)
project, which aims to provide open-source tools for corporate bond research.

## Support

### Where can I get help?
- Open a GitHub issue at [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)
- Email alexander.dickerson1@unsw.edu.au
- Read the documentation
  - [Main README](README.md)
  - [Stage 0 README](stage0/README_stage0.md)
  - [Stage 1 README](stage1/README_stage1.md)
  - [Stage 2 README](stage2/README_stage2.md)
  - [Stage 3 README](stage3/README_stage3.md)
  - [Stage 4 README](stage4/README_stage4.md)
  - [Contributing Guide](CONTRIBUTING.md)

### How quickly will I get a response?
We aim to respond to issues within 1 to 3 business days. For urgent matters, email directly.

### Can I schedule a call to discuss my use case?
Yes. For collaboration or complex use cases, email alexander.dickerson1@unsw.edu.au to arrange
a discussion.

## Stages 1 to 4

### Is Stage 1 available?
Yes. Stage 1 is now in public beta. It adds these to the Stage 0 daily panels.
- Bond characteristics from FISD (coupon, maturity, issuer, etc.)
- Bond analytics from QuantLib (duration, convexity, YTM, credit spreads)
- Credit ratings from S&P and Moody's
- Equity identifiers (PERMNO, PERMCO, GVKEY)
- Ultra-distressed filters
- Fama-French industry classifications

See [stage1/QUICKSTART_stage1.md](stage1/QUICKSTART_stage1.md) to get started.

### How do I run Stage 1?
The easiest way is `./run_pipeline.sh`, which runs both Stage 0 and Stage 1 automatically.

To run Stage 1 by hand after Stage 0 completes, use these lines from the repository root.
```bash
bash download_inputs.sh      # login node: Stage 1 stops at once if its inputs are missing
qsub stage1/run_stage1.sh
```

### What about Stage 2?
Stage 2 builds the monthly asset-pricing panel from your Stage 1 output. The panel has 145
columns per bond-month. They cover the following.

- spreads, yields and size
- value
- momentum and reversal
- illiquidity measures
- volatility and downside risk
- rolling 36-month betas on 37 factor models
- portfolio-ready returns, with and without duration adjustment

It runs on your own machine, not the WRDS grid, because it reads Stage 1's output and not the
TRACE transactions. Every column is defined in
[stage2/DATA_DICTIONARY.md](stage2/DATA_DICTIONARY.md).

Stage 2 is complete. The code is in this repository, and published vintages are available
at [openbondassetpricing.com](https://openbondassetpricing.com).

### What is Stage 3, and do I need it?
Stage 3 is what the data was built for, and it is optional. Stages 0 to 2 build the panel, and
the panel is useful on its own.

It turns the Stage-2 monthly panel into portfolio sorts, two uncertainty grids, and 33 table
files and 11 figures that reproduce *The Corporate Bond Factor Replication Crisis* (main text,
appendix and Internet Appendix). It ends with a single compiled `stage3/reports/exhibits.pdf`.
Twenty-nine of the tables are the paper's, and the other four are Stage 3's own.

```bash
cd stage3
python tools/check_inputs.py     # are the five inputs there and the right shape?
bash run_stage3.sh               # everything, ending in reports/exhibits.pdf
```

It runs on your own machine and opens no WRDS connection. It takes about 16 minutes on 24
cores (15.7 minutes on a cold run on 2026-09-26). See
[stage3/QUICKSTART_stage3.md](stage3/QUICKSTART_stage3.md).

### Are Stage 3's numbers the paper's printed numbers?
No, and nothing in Stage 3 compares them to the paper's. It produces exhibits from whatever
panel Stage 2 built for you, and the title page of `exhibits.pdf` says so. A PDF of tables
under familiar captions is the kind of file that gets mistaken for the original, so the
document states where its numbers come from.

Two exhibits, Table B.1 and Table IA.IX, ship in two variants, because the published table and
the paper's own definitions differ. Both are produced, and neither is corrected without saying
so.

### What is Stage 4, and do I need it?
Stage 4 builds the TRACE-only bond factors that openbondassetpricing.com publishes, which are
108 signals as single sorts and within-firm sorts, for four return types and three rating
bands. It is optional, like Stage 3, and it reads only Stage 2's output.

```bash
cd stage4
bash run_stage4.sh      # build both sorts, then compare them with the published files
```

`compare_published.py` downloads the published files and compares them with yours, cell by
cell. On the 2026-09-21 run, every one of its 4,354,560 rows matches the OSBAP build of that
run. See [stage4/README_stage4.md](stage4/README_stage4.md).

### Why is a within-firm factor's `ls` not `l` minus `s`?
Because the factor is formed inside each firm. For within-firm sorts, `ls` is each firm's
high-signal bonds minus its low-signal bonds, averaged across firms and three rating groups.
`l` and `s` pool every firm's high and low bonds into one portfolio each, which mixes in the
differences between firms that the factor removes. For single sorts `ls` is exactly `l` minus
`s`. See [stage4/DATA_DICTIONARY.md](stage4/DATA_DICTIONARY.md).

### What is redacted in the published panel?
`permco` and `gvkey` are set to null, and `spc_rat`/`mdc_rat` are collapsed to investment
grade (1) versus non-investment-grade and default (11). `permno` is kept. This applies to the
redistributed file only. A panel you build yourself from your own WRDS subscription keeps
every identifier and the full 1 to 22 rating scale.

Last updated in November 2026.

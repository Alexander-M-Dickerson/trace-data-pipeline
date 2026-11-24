# Stage 0: TRACE daily processing (Enhanced, Standard, 144A)

This stage fetches and cleans TRACE data (Enhanced, Standard, and Rule 144A) on the WRDS Cloud
and aggregates it to daily panels. Jobs are submitted to Sun Grid Engine (SGE). Windows users
submit them from a PuTTY session and can move files to and from WRDS with
[WinSCP](https://winscp.net/eng/download.php). Mac users can use their terminal once connected
to the WRDS Cloud. They might also find [ForkLift](https://binarynights.com/) useful. It is a
GUI file manager that lets you upload, edit and manage WRDS Cloud files.

The code also writes detailed TRACE data reports, which document the effect of the filters at
the transaction level. The reports can include hundreds of time-series plots, covering every
bond `cusip_id` that the decimal-shift and bounce-back correctors of Dickerson, Robotti and
Rossetti (2026) affect.

To get going quickly, see "Quick start". "Generating the TRACE data reports" explains how to
build the reports.

## Table of Contents

- [Repo layout](#repo-layout-key-files)
- [How Stage 0 spends its time](#how-stage-0-spends-its-time-and-why-it-is-no-longer-4-hours)
- [Python on the WRDS Cloud](#python-on-the-wrds-cloud)
- [Getting the code onto WRDS](#getting-the-code-onto-wrds)
- [Requirements](#requirements)
- [Quick start](#quick-start-wrds-putty)
- [Running jobs](#running-jobs-individually-alternative)
- [What the runners do](#what-the-runners-do)
- [Configuration](#configuration-choices-you-can-edit)
- [Outputs](#outputs)
- [Generating TRACE data reports](#generating-the-trace-data-reports)
- [Notes & tips](#notes--tips)
- [Troubleshooting](#troubleshooting)
- [Monitoring](#monitoring)
- [Advanced usage](#advanced-usage)
- [Performance optimization](#performance-optimization)
- [License & citation](#license--citation)
- [Support](#support)
- [Version history](#version-history)

## Repo layout (key files)

```
stage0/
  # Shell scripts for job submission
  (submission lives in ../run_pipeline.sh, which reads TRACE_MEMBERS)
  run_enhanced_trace.sh       # Submits Enhanced TRACE job
  run_standard_trace.sh       # Submits Standard TRACE job
  run_144a_trace.sh           # Submits Rule 144A TRACE job
  run_build_data_reports.sh   # Submits the data report generation job

  # Configuration
  _trace_settings.py          # Central configuration: filters, parameters, CONCURRENCY,
                              # and the grid resource requests (the WRDS username is in
                              # the root config.py)

  # Python runners (called by shell scripts)
  _run_enhanced_trace.py      # Enhanced runner (calls CreateDailyEnhancedTRACE)
  _run_standard_trace.py      # Standard runner (calls CreateDailyStandardTRACE)
  _run_144a_trace.py          # 144A runner (calls CreateDailyStandardTRACE with data_type=144a)

  # Core processing modules
  create_daily_enhanced_trace.py # Main functions for Enhanced TRACE processing
  create_daily_standard_trace.py # Main functions for Standard/144A TRACE processing
  _chunk_runner.py            # How the CUSIP universe is split into work units, and
                              # the scheduler that runs several of them at once
  _wrds_pool.py               # One WRDS connection per worker process

  # Report generation
  _build_error_files.py       # Generates TRACE data quality reports
  _error_plot_helpers.py      # Helper functions for plotting and LaTeX report generation

  # Output directories (created automatically)
  logs/                       # Job logs (.out and .err files)
  data_reports/               # LaTeX reports and figures (if generated)
```

Submission lives in `../run_pipeline.sh`, which reads `TRACE_MEMBERS` from `config.py` and
submits exactly those members. Enhanced and 144A go in together. Standard, if requested, is held
behind them with `-hold_jid`. The report job is then held on every stage-0 job that was
submitted, and Stage 1 is held on the same stage-0 jobs. Stage 1 and the reports therefore run
alongside each other, and everything still goes in with one submission.

Each `run_*.sh` is a thin SGE wrapper. It sets `-cwd` (current working directory), exports your
environment (`-V`) and writes logs into `stage0/logs/`. The three data-job scripts set no cores
or memory. `run_pipeline.sh` computes them per member from `CONCURRENCY` and passes them on the
`qsub` command line, so the request cannot drift away from the worker count.
`run_build_data_reports.sh` has its own request (`-pe onenode 5 -l m_mem_free=8G`).

## How Stage 0 spends its time (and why it is no longer ~4 hours)

Stage 0 is a loop over chunks of CUSIPs. It fetches a chunk from WRDS, cleans it and aggregates
it to bond-days. Until v2.2.0 that loop was strictly serial on one WRDS connection, which is why
Enhanced took about four hours while holding a whole compute node.

Chunks are now packed by row count, not CUSIP count. Trading activity is very skewed, so
250-CUSIP chunks ranged from 7,806 rows to 3,392,802 over the Enhanced universe, and the memory
a job must reserve is set by the worst chunk, not the average. Packing to about 750,000 rows
brings the worst case to 749,992, a 4.5x reduction, for about the same number of chunks.
`target_rows_per_chunk` controls this. `chunk_size` still exists and still means CUSIPs per
chunk, because the report job uses it for its own independent chunking of flagged CUSIPs.

Re-chunking cannot change the cleaned data. Every per-chunk filter groups by `cusip_id` (the
decimal-shift anchor, the bounce-back scan, the initial-price-error scan and the Dick-Nielsen
reversal keys), and chunks are disjoint CUSIP sets, so the chunk a bond lands in cannot affect
its result. The audit tables do change, because their `chunk` column follows the new grouping.

Several chunks are fetched at once, and each worker process holds its own WRDS connection.
`CONCURRENCY` in `_trace_settings.py` sets how many. `STAGE0_WORKERS` overrides it for a single
run, for every member at once, 144A included. The connection check and the `qsub` request both
use it. `run_pipeline.sh` refuses a value above 3, because Enhanced and 144A run together and
hold twice that number.

Output is sorted canonically before export, by `(cusip_id, trd_exctn_dt)`. Row order no longer
depends on the work plan, so a scheduling change can be proved not to have altered the data. The
parquet files come out byte-identical whether one worker ran the chunks or six did. On 5 chunks
the run took 18 s of wall time against 81.3 s of serial work.

### The WRDS connection budget

| member | connections | when it runs |
|---|---|---|
| `enhanced` | 5 | alongside 144A |
| `144a` | 1 | alongside Enhanced |
| `standard` | 6 | after both, so it can use the lot |

The measured ceiling is 7 connections held simultaneously, and the 8th fails. (The "5" WRDS
publishes is the limit on concurrent jobs, a different limit.) Enhanced and 144A together
therefore hold 6, leaving one spare so a mid-run reconnect cannot be refused.
`validate_connection_budget` enforces this at submit time, not four hours in.

Measure the ceiling on your own account before raising anything, when no other job of yours is
running. The probe opens up to 10 connections, so it belongs in a job, not on the login node.
Put the line below in a job script that starts with the `#!/bin/bash`, `#$ -cwd` and `#$ -V`
lines the stage 0 job scripts start with, and `qsub` it from the repository root.

```bash
python3 tests/probe_wrds_connections.py --max 10
```

> A refused connection does not look like one. The `wrds` package answers every failed connect
> by asking for a username again. In a batch job stdin is closed, so the failure surfaces as
> `EOFError: EOF when reading a line`. That one error covers both a missing `WRDS_USERNAME` and
> the connection limit. `_wrds_pool.py` tells them apart and says which.

### Grid resources

`m_mem_free` is charged per slot, so `slots × m_mem_free ≤ 48 GB` (and ≤ 8 cores). A job that
asks for more does not fail. It waits in the queue forever without an error. `qsub_resources()`
derives the request from `CONCURRENCY` and refuses anything over the caps.

| member | request | total |
|---|---|---|
| `enhanced` | `-pe onenode 5 -l m_mem_free=8G` | 40 GB |
| `144a` | `-pe onenode 1 -l m_mem_free=16G` | 16 GB |
| `standard` | `-pe onenode 6 -l m_mem_free=8G` | 48 GB |

If a job sits at `qw` and will not start, check the slot request first. Lower the member's
`CONCURRENCY` and resubmit, and the resource request follows.

## Python on the WRDS Cloud

Follow the WRDS guides to set yourself up on the cloud with Python. Read them in this order.

1. [Python: On the WRDS Cloud](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-wrds-cloud/)
2. [Batch Python Jobs](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/submitting-python-programs/)
3. [Using SSH to Connect to the WRDS Cloud](https://wrds-www.wharton.upenn.edu/pages/support/the-wrds-cloud/using-ssh-connect-wrds-cloud/)
4. Windows users who want a point-and-click UI for downloading data can also read [Using SCP with Windows](https://wrds-www.wharton.upenn.edu/pages/support/the-wrds-cloud/managing-data/accessing-wrds-remotely-scp/).

This code package assumes you have the following.
- Access to the WRDS cloud
- Your WRDS username set (`WRDS_USERNAME`, in `config.py` or the environment). No password file
  is needed on the WRDS Cloud
- Basic familiarity with simple scripting commands in Windows or Mac
- WRDS entitlements for TRACE Enhanced, Standard, and/or 144A data

## Getting the code onto WRDS

### Windows users

From a PuTTY shell on your computer, choose one of the options below.

> **Take the whole repository**, not `stage0/` alone. Stage 0 cannot run by itself. The root
> `config.py` is where `WRDS_USERNAME` is set and stage 0 imports it, the job wrappers
> `cd stage0` from the repository root and log to `stage0/logs/`, and `run_pipeline.sh` lives
> at the root. A folder holding only the stage-0 files cannot be run.

Put it in your home directory, `~/trace-data-pipeline/`, because the zip and download lines in
[QUICKSTART.md](../QUICKSTART.md) assume that path.

#### Option A: download a ZIP (no git required)
```bash
cd ~
wget -O trace.zip \
  https://github.com/Alexander-M-Dickerson/trace-data-pipeline/archive/refs/heads/main.zip
unzip trace.zip
mv trace-data-pipeline-main trace-data-pipeline
cd trace-data-pipeline
```

#### Option B: using `curl` (also no git)
```bash
cd ~
curl -L -o trace.zip \
  https://github.com/Alexander-M-Dickerson/trace-data-pipeline/archive/refs/heads/main.zip
unzip trace.zip
mv trace-data-pipeline-main trace-data-pipeline
cd trace-data-pipeline
```

#### Option C: clone (if `git` is available on your WRDS node)
```bash
cd ~
git clone https://github.com/Alexander-M-Dickerson/trace-data-pipeline.git
cd trace-data-pipeline
```

You should now have `~/trace-data-pipeline/` holding `config.py`, `run_pipeline.sh`, `stage0/`
and `stage1/`. Run everything from that directory.

### Linux/Mac users

Open your terminal application (*Terminal* or *iTerm2*) and connect to WRDS with SSH.

```bash
ssh wrds_username@wrds-cloud.wharton.upenn.edu
```

Once connected, you will see a WRDS prompt. From there, use Option A, B or C from the previous
section.

#### Option D: transferring files from your local computer

If you already have the repository on your local computer and want to upload it after editing
the scripts, use the secure copy protocol `scp`. Upload the whole repository, because `stage0/`
alone cannot run (see the note above).

This line transfers the repository to your home directory on the WRDS Cloud.
```bash
scp -r ~/path/to/trace-data-pipeline wrds_username@wrds-cloud.wharton.upenn.edu:/home/university/wrds_username/
```

Replace these parts of the line.
- `~/path/to/trace-data-pipeline` with the path to the repository on your local computer
- `/home/university/wrds_username/` with the full WRDS destination path you see when you run
  `pwd` after connecting to WRDS

## Requirements

You need Python 3.10 or higher. The 2026-09-21 run used Python 3.14.5, the WRDS Cloud default.

The required packages are everything in the repository's `requirements.txt`, which states the
minimum versions. For stage 0 that is `pandas`, `numpy`, `wrds`, `SQLAlchemy`,
`pandas_market_calendars`, `pyarrow` and `matplotlib`. matplotlib is not optional, because the
report job, which `run_pipeline.sh` always submits, imports it. The log files print your Python
and package versions, so a run can be matched to the environment that made it.

### Installation

Run this from the repository root, on WRDS.
```bash
python -m pip install --user -r requirements.txt
```

[quickstart.md](quickstart.md) explains the choice between `--user` and a virtual environment.

## Quick start (WRDS, PuTTY)

### 1. Install required Python packages

SSH into the WRDS cloud, put the repository there (see
[Getting the code onto WRDS](#getting-the-code-onto-wrds)), and install the packages in
`requirements.txt` (see [Requirements](#requirements)).

### 2. Navigate to the repository and configure settings

```bash
cd ~/trace-data-pipeline   # wherever you put the repository
```

**Set your WRDS username.** It lives in the shared `config.py` at the repo root.
`_trace_settings.py` imports it from there, so editing `_trace_settings.py` does nothing.

The simplest way is an environment variable, which needs no file edited.

```bash
export WRDS_USERNAME="your_wrds_id"
echo 'export WRDS_USERNAME="your_wrds_id"' >> ~/.bashrc   # make it persistent
```

Or edit the fallback in `config.py`.

```bash
nano config.py
# WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")
```
Save and exit `nano` with Ctrl+O, Enter, then Ctrl+X. Confirm with
`python3 -c "from config import WRDS_USERNAME; print(WRDS_USERNAME)"`.

> If this is left as the placeholder `your_wrds_username`, the run dies with
> `EOFError: EOF when reading a line`, which is the `wrds` package prompting on a closed stdin.
> The connection limit gives the same error, but here the cause is the username.

No password file is needed on the WRDS Cloud, because jobs there connect without one. When you
see an `EOFError`, check the username first and the connection limit second.

Review the default filter settings in `_trace_settings.py`. All filters but two (`trading_time`
and `volume_filter_toggle`) are on by default, with the values Dickerson, Robotti and Rossetti
(2026) recommend. The [Configuration](#configuration-choices-you-can-edit) section has the
details.

### 3. Make scripts executable (older clones only)

The scripts ship executable, so a fresh clone needs nothing here. A clone made before the
executable bits were committed may need this line.

```bash
chmod +x run_pipeline.sh download_inputs.sh run_smoke_test.sh stage0/run_*.sh stage1/run_stage1.sh
```

### 4. Fix line endings (if editing on Windows)

If you edit the `.sh` scripts on Windows, your editor may save them with CRLF (Windows) endings.
SGE expects LF (Unix) endings, and with CRLF you get "bad interpreter" or `^M` errors.

Fix them once inside WRDS with this line.
```bash
# Convert all .sh files in place
find . -name "*.sh" -exec sed -i 's/\r$//' {} \;
```

### 5. Submit all jobs (recommended)

This submits the whole pipeline.

```bash
./run_pipeline.sh
```

The script does the following.
1. It reads `TRACE_MEMBERS` from `config.py` and submits exactly those members. The default is
   `["enhanced", "144a"]`, and Standard is opt-in.
2. Enhanced and 144A go in together, because their WRDS connection budgets are sized to fit
   side by side. Each asks for cores and memory matched to its worker count.
3. Standard, if requested, is held behind them with `-hold_jid`, so it runs alone and can use
   the whole connection budget.
4. `build_reports` is held on every stage-0 job that was submitted.
5. Stage 1 is held on the same stage-0 jobs, not on the report job. Since v2.2.2 the two run
   alongside each other, so the report job no longer holds up the rest of the run. That release
   also parallelised the report's re-clean, and the job went from ~51 minutes to about 20 (22
   min on the 2026-09-21 run).

The script prints output like this.
```
[info] members: enhanced 144a
[submit] enhanced TRACE  (-pe onenode 5 -l m_mem_free=8G) ...
         job 4821094
[submit] 144a TRACE  (-pe onenode 1 -l m_mem_free=16G) ...
         job 4821095
[submit] Build data reports (waits for 4821094,4821095) ...
```

> Check status with `qstat`. The report job shows status `hqw` (hold) until the data jobs
> finish. Tail logs with `tail -f stage0/logs/01_enhanced.out` (or `.err`).

The whole pipeline takes about 5 hours. The 2026-09-21 run, which the 2026 vintage was built
from, took 5.3 h end to end. Enhanced (2.58 h) and 144A (0.63 h) ran in parallel, then the
reports (0.37 h) and Stage 1 (2.69 h) ran side by side.

If errors remain after you have tried to debug them, contact Alex Dickerson at
`alexander.dickerson1@unsw.edu.au` for help.

## Running jobs individually (alternative)

You can also submit one dataset at a time.

Submit from the repository root. The wrappers `cd stage0` themselves and write their logs to
`stage0/logs/`, so a bare `qsub run_enhanced_trace.sh` from inside `stage0/` fails before it
runs a line.

> **Pass the resource request yourself.** The wrappers have no `-pe`/`-l` directives.
> `run_pipeline.sh` supplies them on the qsub command line from `qsub_resources()`. If you
> submit one by hand without them, SGE gives the job a single slot with default memory while
> the code still opens `CONCURRENCY[member]` connections, and they all contend for one core.

```bash
mkdir -p stage0/logs    # SGE opens the log before the job starts; without the folder the job sits in Eqw
qsub -pe onenode 5 -l m_mem_free=8G  stage0/run_enhanced_trace.sh    # enhanced
qsub -pe onenode 6 -l m_mem_free=8G  stage0/run_standard_trace.sh    # standard
qsub -pe onenode 1 -l m_mem_free=16G stage0/run_144a_trace.sh        # 144a
```

Those are the current values. If you have changed `CONCURRENCY` or `MEM_PER_SLOT_GB`, do not
copy them. Print the new ones with this line.

```bash
cd stage0 && python3 -c "from _trace_settings import qsub_resources; print(qsub_resources('enhanced'))"
```

Each wrapper uses SGE's `-cwd`, so outputs and logs land under the current folder, and `-V` to
pass your Python environment variables.

### Running report generation separately

Use this line if you run jobs individually and want the reports later, or if you want to
regenerate reports with different settings.

```bash
qsub stage0/run_build_data_reports.sh
```

The datasets the report job processes come from `TRACE_MEMBERS` in the root `config.py`, or
from the `--data-type` flag. `DATA_TYPES` was moved out of `_build_error_files.py` (its source
still has the note "DATA_TYPES moved to config.py as TRACE_MEMBERS"), so adding it back there
has no effect.

```bash
# One dataset, without touching config.py (from the repository root):
cd stage0 && python3 _build_error_files.py --data-type enhanced
```

```python
# In config.py, which run_pipeline.sh, the report job and Stage 1 read
TRACE_MEMBERS = ["enhanced"]
```

Change the filtering settings in `_trace_settings.py` if needed.

## What the runners do

### Enhanced TRACE (`_run_enhanced_trace.py`)

It calls `CreateDailyEnhancedTRACE` with the default cleaning filters and audit logging.
- Processes the full Enhanced TRACE sample period (2002-07-01 to present)
- Applies Dick-Nielsen filters for cancellations, corrections, and agency duplicates
- Runs decimal-shift correction (detects and fixes 10x, 0.1x, 100x, 0.01x price errors)
- Applies bounce-back price-error filtering
- Computes daily price metrics (equal-weighted, volume-weighted, par-weighted, first, last,
  trade count)
- Computes daily volume metrics (quantity volume and dollar volume, in millions)
- Computes bid/ask prices (value-weighted)
- Writes audit logs for each filter stage
- Saves all outputs to the `enhanced/` subfolder

### Standard TRACE (`_run_standard_trace.py`)

It calls `CreateDailyStandardTRACE` with the same controls for the Standard table.
- The default start date is `2024-10-01` (you can change it in `_trace_settings.py`)
- Removes cancellations, corrections and reversals using Standard TRACE's status codes (the
  pre/post-2012 rules and the agency de-duplication are Enhanced's)
- Same decimal-shift and bounce-back filters as Enhanced
- Same daily aggregation metrics
- Saves all outputs to the `standard/` subfolder

### Rule 144A TRACE (`_run_144a_trace.py`)

It calls `CreateDailyStandardTRACE` with `data_type='144a'`.
- The default start date is `2002-07-01`, though there is almost no data before 2014 (29
  bond-days before 2014 in the 2026-09-21 run, then 70,668 in 2014)
- Uses the same cleaning pipeline as Standard TRACE
- Same parameter blocks for filters and aggregation
- Saves all outputs to the `144a/` subfolder

### Core processing steps

Both `create_daily_enhanced_trace.py` and `create_daily_standard_trace.py` run these steps.

1. Connect to WRDS and open a database connection.
2. Filter the FISD universe with the configured parameters (USD only, fixed-rate,
   non-convertible, etc.).
3. Plan the chunks by packing CUSIPs to `target_rows_per_chunk` trade rows (default 750,000).
   `chunk_size` (250 CUSIPs) is only the fallback, used when the row-count query is
   unavailable.
4. Clean each chunk, several at once, with one WRDS connection per worker. The filters run in
   this order, and the numbering matches the `# Filter N:` comments in the source.

   | # | Filter | What it does |
   |---|---|---|
   | 0 | price-scale normalization | rescales unit-quoted bonds to percent of par (a no-op under the default FISD screen) |
   | 1 | Dick-Nielsen | cancellations, corrections and reversals, plus agency de-duplication in Enhanced only |
   | 2 | decimal-shift corrector | fixes multiplicative price errors (10x, 0.1x, 100x, 0.01x) |
   | 3 | trading time | intraday window, off by default |
   | 4 | trading calendar | drops non-session dates |
   | 5 | price range | `> 0` and `<= 1000` |
   | 6 | trading volume | dollar or par threshold, off by default |
   | 7 | bounce-back | flags price spikes that revert. It runs after the volume filter, not before |
   | 8 | yield != price | drops rows where the reported yield equals the price |
   | 9 | amount outstanding | drops volume above 50% of the offering amount |
   | 10 | execution date vs maturity | drops trades after maturity |
   | 11 | initial price error | flags implausible opening prints |

   Then aggregate to a daily `(cusip_id, trd_exctn_dt)` panel.
5. Reassemble the chunks in order, sort canonically by `(cusip_id, trd_exctn_dt)`, and export
   to Parquet.
6. Write audit logs with row counts for each filter stage.
7. Export CUSIP lists for the decimal-shift, bounce-back and initial-price-error filters (three
   files, one per filter).

## Configuration choices you can edit

Open `_trace_settings.py` and adjust the settings below. The WRDS username is the exception. It
lives in the shared `config.py` at the repo root, and `_trace_settings.py` only imports it, so
editing it there does nothing (see "Quick start" above).

### FISD universe parameters (`FISD_PARAMS`)

These control which bonds are in the universe.
- `currency_usd_only` keeps only USD-denominated bonds (default `True`)
- `fixed_rate_only` excludes variable-rate bonds (default `True`)
- `non_convertible_only` excludes convertible bonds (default `True`)
- `non_asset_backed_only` excludes asset-backed securities (default `True`)
- `exclude_bond_types` drops specific bond types like TXMU, MBS, FGOV, etc. (default `True`)
- `valid_coupon_frequency_only` drops bonds with invalid interest frequency (default `True`)
- `require_accrual_fields` requires offering_date and dated_date to be non-null (default `True`)
- `principal_amt_eq_1000_only` keeps only bonds with $1000 par value (default `True`). If you
  turn this off, leave `PRICE_NORM` on (see below).
- `exclude_equity_index_linked` excludes equity-linked and index-linked bonds (default `True`)
- `enforce_tenor_min` requires bonds to have a minimum tenor (default `True`)
- `tenor_min_years` is the minimum tenor in years (default `1.0`)

### Price-scale normalization (`PRICE_NORM`)

- `normalize_nonpar1000` rescales unit-quoted bonds to percent of par (default `True`)

TRACE's `rptd_pr` is a percent of par for the standard $1,000-principal bond, so at par it
prints `100`. Small-denomination issues (retail and structured notes with a principal of $10,
$25 or $100) are quoted in unit dollars instead, so a $10 note at par prints `10.00`, not `100`.

Everything downstream assumes percent of par. That covers the price bounds, the decimal-shift
gates, the bounce-back point threshold, Stage 1's ultra-distressed thresholds, dollar volume
(`entrd_vol_qt * rptd_pr / 100`) and QuantLib (face value 100). Left alone, a healthy $10 note
reads as a bond trading at 10% of par. It is flagged as distressed, its dollar volume is
understated tenfold and its yield is nonsense.

When enabled, each CUSIP whose `principal_amt` is not 1000 is rescaled by `100 / principal_amt`,
but only if that moves the bond's median price closer to par in log distance. The decision is
made once per bond from its median, so a run of distressed prints cannot flip the convention.
$1,000-principal bonds are never touched.

Under the default settings this does nothing. `principal_amt_eq_1000_only` is `True`, so no
non-$1,000 bond is in the universe and every factor resolves to `1.0`. The output is identical
whether the toggle is on or off. It matters only when you turn that screen off, and that is
when unit-quoted notes enter the data.

### Filter toggles (`FILTER_SWITCHES`)

Every filter is a boolean toggle.
- `dick_nielsen` applies the Dick-Nielsen cleaning steps (default `True`)
- `decimal_shift_corrector` fixes decimal shift errors (default `True`)
- `trading_time` filters by intraday time window (default `False`)
- `trading_calendar` keeps only valid trading days (default `True`)
- `price_filters` removes negative prices and prices > 1000 (default `True`)
- `volume_filter_toggle` applies a minimum-size threshold to each trade (default `False`)
  - It is off by design. The published OSBAP data keeps trades of all sizes. Turning it on
    removes ~26% of Enhanced trades per bond-day and changes every volume-weighted price, so a
    panel built with it on will not reconcile against the published data.
- `bounce_back_filter` flags price-change errors (default `True`)
- `yld_price_filter` removes rows where yield = price (default `True`)
- `amtout_volume_filter` removes trades > 50% of offering amount (default `True`)
- `trd_exe_mat_filter` removes trades after the maturity date (default `True`)
- `flag_initial_price_errors` flags implausible opening prints (default `True`)

> **Write all eleven keys** in any `FILTER_SWITCHES` you write. A key you leave out raises no
> error. It falls back to the engine's own default, which is not always the settings file's. If
> you leave out `volume_filter_toggle`, the $10,000 dollar floor switches on. `trading_time` and
> `volume_filter_toggle` are the two that are off by default.

### Decimal-shift corrector parameters (`DS_PARAMS`)

These tune the decimal shift correction algorithm.
- `factors` are the multiplicative factors to test, `(0.1, 0.01, 10.0, 100.0)`
- `tol_pct_good` is the relative error threshold for accepting a potential correction, `0.02`
  (2%)
- `tol_abs_good` is the absolute distance threshold in price points, `8.0`
- `tol_pct_bad` is the minimum raw relative error to trigger consideration, `0.05` (5%)
- `low_pr` and `high_pr` are the plausible price bounds, `5.0` and `300.0`
- `anchor` is the anchor type for comparison, `"rolling"`
- `window` is the rolling window half-width, `5`
- `improvement_frac` is the required improvement over raw error, `0.2` (20%)
- `par_snap` enables relaxed acceptance near par=100, `True`
- `par_band` is the proximity band around par, `15.0`
- `output_type` is `"cleaned"` to apply corrections or `"uncleaned"` for audit only

### Initial-price-error parameters (`INIT_ERROR`)

This is the last filter in the cascade. It looks at each bond's first few prints and flags them
when the price then moves sharply. That is the pattern of a bond whose opening marks were
placeholders, not trades.

- `abs_change` is the price move, in points of par, that marks the earlier prints as errors,
  `50.0`
- `n_transactions` is how many opening prints to examine, `3`

Flagged rows are dropped, and the affected CUSIPs are written to
`init_price_cusips_{dtype}_{stamp}.parquet` for the data report, which draws them as the `_ie`
figure pages.

### Bounce-back filter parameters (`BB_PARAMS`)

These tune the bounce-back price-error detection.
- `threshold_abs` is the minimum absolute price jump to flag a candidate error, `35.0`
- `lookahead` is the maximum number of rows ahead to search for a bounce, `5`
- `max_span` is the maximum path length from start to resolution, `5`
- `window` is the backward window for the trailing median anchor, `5`
- `back_to_anchor_tol` is how close the price must come back to the anchor, as a fraction of
  `threshold_abs` (0.25 × 35 = 8.75 points), `0.25`
- `candidate_slack_abs` is subtracted from `threshold_abs` when opening a candidate (a jump of
  34 points opens one), `1.0`
- `reassignment_margin_abs` is the margin for tie-breaking in clusters, `5.0`
- `use_unique_trailing_median` uses unique values in the median, `True`
- `par_spike_heuristic` enables special handling at par, `True`
- `par_level` is the par value, `100.0`
- `par_equal_tol` is the tolerance for treating a price as par, `1e-8`
- `par_min_run` is the minimum par run length to flag, `3`
- `par_cooldown_after_flag` is the number of rows to skip after flagging, `2`

### Common arguments (`COMMON_KWARGS`)

These settings apply to all runners.
- `output_format` is `"parquet"`, the only supported value. It comes from `OUTPUT_FORMAT` in
  the root `config.py`, and `_trace_settings.py` refuses anything else when it loads.
- `chunk_size` is the number of CUSIPs per batch, `250`. Since v2.2.0 this no longer decides
  Stage 0's own chunking (see `target_rows_per_chunk`), but the report job still reads it for
  its independent chunking of flagged CUSIPs, so it is kept.
- `target_rows_per_chunk` is the number of trade rows per chunk, `750_000`. This is the setting
  that sizes Stage 0's work units. Set it to `None` to fall back to fixed `chunk_size` chunks.
  Override it for one run with `STAGE0_TARGET_ROWS`.
- `limit_chunks` processes only the first N chunks, `None`. It exists for development and
  testing. Never set it for a production run. Override it with `STAGE0_LIMIT_CHUNKS`.
- `clean_agency` applies agency de-duplication, `True`
- `out_dir` is the output directory, `""` (current directory)
- `volume_filter` is a tuple of `(kind, threshold)`, used only when `volume_filter_toggle` is
  `True`.
  - `("dollar", 10000)` means dollar volume (`entrd_vol_qt * rptd_pr / 100`) >= $10,000
  - `("par", 10000)` means par volume >= $10,000
- `trade_times` is the intraday window, `["00:00:00", "23:59:59"]` (effectively disabled)
- `calendar_name` is the market calendar, `"NYSE"`

### Per-dataset overrides (`PER_DATASET`)

Each dataset has its own settings.
- Enhanced has `n_workers` only. It takes no `start_date` and always starts at 2002-07-01
- Standard has `start_date="2024-10-01"`, `data_type="standard"` and `n_workers`
- 144A has `start_date="2002-07-01"`, `data_type="144a"` and `n_workers`

`n_workers` is how many chunks are fetched at once, one WRDS connection each. It comes from
`CONCURRENCY` for each member, and `STAGE0_WORKERS` overrides it for one run (see the warning under
[How Stage 0 spends its time](#how-stage-0-spends-its-time-and-why-it-is-no-longer-4-hours)).

## Outputs

### Output directory structure

Data outputs go into one subfolder per dataset. A single `data_reports/` folder holds a
subfolder for each dataset's reports.

```
stage0/
├── logs/                        # Job logs for all runs
│   ├── 01_enhanced.out
│   ├── 01_enhanced.err
│   ├── 02_standard.out
│   ├── 02_standard.err
│   ├── 03_144a.out
│   ├── 03_144a.err
│   ├── _data_reports.out
│   └── _data_reports.err
│
├── enhanced/                    # Enhanced TRACE data outputs (nine files)
│   ├── trace_enhanced_YYYYMMDD.parquet
│   ├── trace_enhanced_fisd_YYYYMMDD.parquet
│   ├── fisd_filters_enhanced_YYYYMMDD.parquet
│   ├── dick_nielsen_filters_audit_enhanced_YYYYMMDD.parquet
│   ├── drr_filters_audit_enhanced_YYYYMMDD.parquet
│   ├── bounce_back_cusips_enhanced_YYYYMMDD.parquet
│   ├── decimal_shift_cusips_enhanced_YYYYMMDD.parquet
│   ├── init_price_cusips_enhanced_YYYYMMDD.parquet
│   └── cusip_row_counts_YYYYMMDD.parquet
│
├── standard/                    # Standard TRACE data outputs (nine files)
│   ├── trace_standard_YYYYMMDD.parquet
│   ├── trace_fisd_standard_YYYYMMDD.parquet
│   ├── fisd_filters_standard_YYYYMMDD.parquet
│   ├── dick_nielsen_filters_audit_standard_YYYYMMDD.parquet
│   ├── drr_filters_audit_standard_YYYYMMDD.parquet
│   ├── bounce_back_cusips_standard_YYYYMMDD.parquet
│   ├── decimal_shift_cusips_standard_YYYYMMDD.parquet
│   ├── init_price_cusips_standard_YYYYMMDD.parquet
│   └── cusip_row_counts_YYYYMMDD.parquet
│
├── 144a/                        # Rule 144A data outputs (nine files)
│   ├── trace_144a_YYYYMMDD.parquet
│   ├── trace_fisd_144a_YYYYMMDD.parquet
│   ├── fisd_filters_144a_YYYYMMDD.parquet
│   ├── dick_nielsen_filters_audit_144a_YYYYMMDD.parquet
│   ├── drr_filters_audit_144a_YYYYMMDD.parquet
│   ├── bounce_back_cusips_144a_YYYYMMDD.parquet
│   ├── decimal_shift_cusips_144a_YYYYMMDD.parquet
│   ├── init_price_cusips_144a_YYYYMMDD.parquet
│   └── cusip_row_counts_YYYYMMDD.parquet
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

For the later stages, bring the whole repository folder home as it is. Stage 2 reads
`stage0/enhanced/` and `stage1/data/` from beside `stage2/`, so the folder must keep its
layout.

### Files produced

The logs are under `stage0/logs/`.
- `01_enhanced.out` and `01_enhanced.err` are the Enhanced TRACE job logs
- `02_standard.out` and `02_standard.err` are the Standard TRACE job logs
- `03_144a.out` and `03_144a.err` are the 144A TRACE job logs
- `_data_reports.out` and `_data_reports.err` are the report generation logs
- Logs contain timestamps, row counts, filter statistics, and any errors

The daily panels are Parquet files, each in its dataset's subfolder.
- `enhanced/trace_enhanced_YYYYMMDD.parquet` is the Enhanced TRACE daily panel (~31 million rows
  for the full sample)
- `standard/trace_standard_YYYYMMDD.parquet` is the Standard TRACE daily panel
- `144a/trace_144a_YYYYMMDD.parquet` is the Rule 144A daily panel

All panels have the same 21 column names, each defined in
[DATA_DICTIONARY.md](DATA_DICTIONARY.md). Column order is not part of the contract and does
differ between members.
- The keys are `cusip_id` and `trd_exctn_dt`
- The prices are `prc_ew`, `prc_vw`, `prc_vw_par`, `prc_first`, `prc_last`, `prc_hi` and `prc_lo`
- The trade timing columns are `time_ew` and `time_last`
- The volumes (millions) are `qvolume` and `dvolume`
- The counts are `trade_count`, `bid_count` and `ask_count`
- The bid/ask columns are `prc_bid`, `prc_ask`, `bid_last`, `bid_time_ew` and `bid_time_last`

The audit files are Parquet files in the same subfolders.
- `dick_nielsen_filters_audit_{dtype}_{date}.parquet` is the step-by-step Dick-Nielsen audit
- `drr_filters_audit_{dtype}_{date}.parquet` is the Dickerson-Robotti-Rossetti filter audit
- `fisd_filters_{dtype}_{date}.parquet` is the FISD universe construction audit

The CUSIP lists are Parquet files in the same subfolders. There is one per corrector, and the
data report draws a figure family from each.
- `bounce_back_cusips_{dtype}_{date}.parquet` lists the CUSIPs with bounce-back flags (`_bb`
  pages)
- `decimal_shift_cusips_{dtype}_{date}.parquet` lists the CUSIPs corrected by the decimal-shift
  corrector (`_ds` pages)
- `init_price_cusips_{dtype}_{date}.parquet` lists the CUSIPs with flagged opening prints (`_ie`
  pages)

These files are also written. They match neither wildcard above, so they can be overlooked.
- `trace_enhanced_fisd_{date}.parquet` (Enhanced) and `trace_fisd_{dtype}_{date}.parquet` hold
  the screened FISD universe. **Stage 1 reads the Enhanced one**, so it is an input to the next
  stage.
- `cusip_row_counts_{date}.parquet` holds the per-CUSIP row counts behind the chunk packing
  (31 s to count on the 2026-09-21 run). Only a re-run on the same day reuses it.

### Downloading outputs

Zip the whole repository folder on WRDS and copy the zip down, as the
[Download Results section of QUICKSTART](../QUICKSTART.md#download-results-to-your-local-machine)
describes. Stage 2 needs `stage0/` and `stage1/` together, in the folder layout they were built
in.

## Generating the TRACE data reports

When you run `./run_pipeline.sh`, the reports for the members in `TRACE_MEMBERS` (by default
Enhanced and 144A) are built once their data jobs complete, using SGE's `-hold_jid`. Since
v2.2.2 the report job runs alongside Stage 1, not before it. You can also build or rebuild the
reports separately.

### Configuration

Two settings in the root `config.py` control the report job.

```python
STAGE0_OUTPUT_FIGURES = True   # False: tables only (also settable from the environment)
TRACE_MEMBERS = ["enhanced", "144a"]   # which datasets get a report
```

`DATE`, `IN_DIR` and `OUT_DIR` at the top of `_build_error_files.py` can stay blank.

- With figures, the job re-cleans the flagged bonds and draws hundreds of time-series plots. The
  whole job took 22 minutes on the 2026-09-21 run
- `STAGE0_OUTPUT_FIGURES = False` builds only the filter tables and runs in seconds
- For Enhanced TRACE with figures, the LaTeX document can exceed 500 pages

### Running the report generator separately

If you want to regenerate reports with different settings, or did not run `./run_pipeline.sh`,
submit the job yourself.
```bash
qsub stage0/run_build_data_reports.sh
```

### Output structure

Reports are saved in a single `data_reports/` folder, with a subfolder for each dataset.

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
    │   ├── ...
    │   ├── enhanced_ie_fig_page_001_ie.pdf
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

Each `*_data_report.tex` file includes the following.
1. Table 1, the filter toggles and parameter settings
2. Table 2, the FISD universe construction parameters
3. Table 3, the transaction-level filter records (Panel A for FISD, Panel B for the DRR filters,
   Panel C for Dick-Nielsen)
4. Figures (if enabled), which are time-series plots showing decimal-shift corrections,
   bounce-back eliminations and initial-price errors

### Compiling the LaTeX report

Download the report folder to your local machine and compile it.

```bash
cd data_reports/enhanced
pdflatex enhanced_data_report.tex
bibtex enhanced_data_report
pdflatex enhanced_data_report.tex
pdflatex enhanced_data_report.tex
```

Or use your favorite LaTeX editor (TeXShop, TeXstudio, Overleaf, etc.).

## Notes & tips

- Check that your WRDS Python environment has all the required packages. If you use a module or
  conda environment, load or activate it before submitting jobs.

- `run_pipeline.sh` uses SGE's `-hold_jid` to create job dependencies, built from the jobs that
  were submitted. The report job waits at `hqw` until every stage-0 job finishes. This is the
  recommended workflow.

- With the default `TRACE_MEMBERS = ["enhanced", "144a"]` you will see four jobs in `qstat`.
  - `trace_enhanced` is running or queued, with 5 slots
  - `trace_144a` is running or queued, with 1 slot
  - `build_reports` stays at `hqw` until both finish
  - `stage1_pipeline` stays at `hqw` until the data jobs finish, then runs alongside
    `build_reports` (it does not read anything the reports produce)

  Add `standard` to `TRACE_MEMBERS` and a fifth appears. It is held behind the other two and
  does not run beside them.

- Chunks are packed to ~750,000 trade rows, and each worker holds one chunk at a time. If you
  hit memory trouble, lower `target_rows_per_chunk` first and `CONCURRENCY` second, since the
  per-worker peak is set by chunk size.

- Expect these runtimes.
  - Enhanced TRACE (full sample) took about 4 hours serial before v2.2.0. Since then it pulls 5
    chunks at once and takes 2 to 2.6 hours (2.0 h on 2026-09-09, 2.6 h on 2026-09-21), about
    twice as fast.
  - Standard TRACE (from 2024) takes 30 to 60 minutes, and is opt-in
  - Rule 144A (full sample) takes 30 to 60 minutes (38 minutes on the 2026-09-21 run)
  - Data reports (with figures) used to take ~51 minutes for Enhanced. Since v2.2.2 the
    re-clean pulls 5 chunks at once (22 minutes for the whole job on 2026-09-21), and the job
    no longer blocks stage 1

- Enhanced TRACE generates ~31M rows. On the 2026-09-21 run the Enhanced panel was about 2.2 GB
  and the 144A panel about 250 MB, and the whole `stage0/` folder about 2.7 GB.

## Troubleshooting

### Installation issues

- For "pip: command not found", install with this line.
  ```bash
  python -m pip install --user {package_name}
  ```

- For "ImportError: wrds", install `wrds` in your WRDS Python environment or activate the
  appropriate conda/module environment.
  ```bash
  python -m pip install --user wrds
  ```

- For "ImportError: pandas_market_calendars", install the package.
  ```bash
  python -m pip install --user pandas-market-calendars
  ```

### Script execution issues

- If a script is not executable, run this line.
  ```bash
  chmod +x run_pipeline.sh download_inputs.sh run_smoke_test.sh stage0/run_*.sh stage1/run_stage1.sh
  ```

- If you get "Permission denied" when trying to run `./run_pipeline.sh`, you probably missed
  the `chmod +x` step above.

- For "Bad interpreter" or `^M` errors, convert Windows line endings to Unix.
  ```bash
  sed -i 's/\r$//' run_pipeline.sh
  # Or fix all shell scripts at once:
  find . -name "*.sh" -exec sed -i 's/\r$//' {} \;
  ```

### SGE issues

- SGE is not submitting
  - Confirm you are in the correct directory
  - Verify that `qsub` is available on your WRDS node
  - Check that shell scripts have Unix line endings

- A job stays in the queue
  - Check `qstat` to see if resources are available
  - WRDS may have resource constraints during peak hours

### Data issues

- No data is returned
  - Verify your CUSIP list/date range in the configuration
  - Confirm that your WRDS entitlements cover Enhanced/Standard/144A as applicable
  - Check logs for SQL errors or connection issues

- The output files are empty
  - Check that your date ranges are correct in `_trace_settings.py`
  - Verify that FISD filters aren't excluding all bonds
  - Review the audit files to see where rows were dropped

### Memory issues

- A job is killed for memory
  - Lower `TARGET_ROWS_PER_CHUNK` in `_trace_settings.py` (try 400_000). That setting, not
    `chunk_size`, sizes a Stage 0 chunk. Or lower `CONCURRENCY`, since each worker holds one
    chunk.
  - Or give each worker more memory by raising `MEM_PER_SLOT_GB` in `_trace_settings.py`.
    `run_pipeline.sh` passes it to `qsub`, so editing the job scripts changes nothing.
    Slots × memory must stay within the WRDS limit of 48 GB per job, and `qsub_resources()`
    refuses a request over it.

### Report generation issues

- Figures are not generated
  - Verify that `matplotlib` is installed
  - Check that CUSIP lists exist in the expected location
  - Make sure `STAGE0_OUTPUT_FIGURES = True` in the root `config.py`

- LaTeX compilation errors
  - Check that all figure files are present
  - Check that `references.bib` exists
  - Verify you have a complete LaTeX installation with required packages

## Monitoring

### Queue status
```bash
qstat                    # View all your jobs
qstat -u wrds_username   # View only your jobs
```

### Real-time log monitoring
```bash
tail -f stage0/logs/01_enhanced.out      # Follow Enhanced output log
tail -f stage0/logs/01_enhanced.err      # Follow Enhanced error log
tail -f stage0/logs/02_standard.out      # Follow Standard output log
```

### Checking job completion
```bash
ls -lh stage0/*/*.parquet       # List generated parquet files
wc -l stage0/logs/*.out        # Count lines in log files
```

### Resubmitting failed jobs

If a job fails, take these steps.
1. Review the error log with `cat stage0/logs/01_enhanced.err`
2. Fix the issue in configuration or code
3. Resubmit from the repo root with `mkdir -p stage0/logs && qsub -pe onenode 5 -l m_mem_free=8G stage0/run_enhanced_trace.sh` (or just `./run_pipeline.sh`)

## Advanced usage

### Custom date ranges

To process a specific date range, modify the per-dataset overrides in `_trace_settings.py`.

```python
# Keep `n_workers` on every member. Dropping it returns that member to the serial
# path (the engine default is n_workers=1) while run_pipeline.sh still requests 5 slots.
PER_DATASET = {
    "enhanced": dict(n_workers=workers("enhanced")),
    "standard": dict(start_date="2020-01-01", data_type="standard",
                     n_workers=workers("standard")),
    "144a":     dict(start_date="2020-01-01", data_type="144a",
                     n_workers=workers("144a")),
}
```

Enhanced TRACE does not support custom start dates in the current implementation. It always
processes the full sample.

### Custom output directories

This writes outputs to a specific directory.

```python
COMMON_KWARGS = dict(
    ...
    out_dir = "/path/to/output/directory",
    ...
)
```

> Stage 1 reads `stage0/` beside it and the report job reads its current folder, so neither
> finds output written anywhere else. Use this only when you run stage 0 on its own.

### Disabling specific filters

To disable any filter, set it to `False` in `_trace_settings.py`.

```python
# Write ALL ELEVEN keys. An omitted key falls back to the engine's own default,
# not to this file's: leave out volume_filter_toggle and the $10,000 dollar
# floor switches ON.
FILTER_SWITCHES = dict(
    dick_nielsen              = True,
    decimal_shift_corrector   = False,  # Disable decimal shift correction
    trading_time              = False,  # OFF by default
    trading_calendar          = True,
    price_filters             = True,
    volume_filter_toggle      = True,   # OFF by default; ON here as an example
    bounce_back_filter        = False,  # Disable bounce-back filter
    yld_price_filter          = True,
    amtout_volume_filter      = True,
    trd_exe_mat_filter        = True,
    flag_initial_price_errors = True,
)
```

### Custom volume filters

You can specify the volume filter in two ways.

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

This restricts the data to specific trading hours.

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

## Performance optimization

### Chunking strategy

Chunks are packed to `target_rows_per_chunk` trade rows (default `750_000`), not to a fixed
CUSIP count. Size it against the memory available per worker. The peak inside
`decimal_shift_corrector` is roughly 2.5x the raw chunk, because it copies the frame and adds
columns.

- For lower memory per worker, use `target_rows_per_chunk = 400_000`
- For fewer, larger chunks, use `target_rows_per_chunk = 1_500_000`
- To restore the old fixed-CUSIP behaviour, use `target_rows_per_chunk = None`

The row counts behind the packing are measured once per run and cached beside the output as
`cusip_row_counts_<stamp>.parquet` (31 s for the Enhanced universe on the 2026-09-21 run). If
that query fails, the planner does not abort. It falls back to fixed `chunk_size` chunks.

### Parallel processing

There are two levels, and they work together.

Within a job, `CONCURRENCY` decides how many chunks are fetched at once, each on its own WRDS
connection.

Across jobs, `run_pipeline.sh` submits Enhanced and 144A together, and holds Standard behind
them.
```bash
./run_pipeline.sh
```

### Output format

Parquet only. `OUTPUT_FORMAT` in the root `config.py` must be `"parquet"`, because Stage 1 and
the report job read Parquet files, and `_trace_settings.py` refuses any other value when it
loads. To get CSV, convert the Parquet files afterwards (see the [FAQ](../FAQ.md)).

## License & citation

### License

This code is provided under the MIT License. See the LICENSE file for details.

### Citation

If you use or extend this stage, please cite this paper as the primary reference.
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

### Acknowledgments

This pipeline implements cleaning procedures from these papers.
- Dick-Nielsen, J. (2009). Liquidity biases in TRACE. *The Journal of Fixed Income*, 19(2), 43-55.
- Dick-Nielsen, J. (2014). How to clean enhanced TRACE data. Working Paper.
- van Binsbergen, J. H., Nozawa, Y., & Schwert, M. (2025). Duration-based valuation of corporate bonds. *The Review of Financial Studies*, 38(1), 158-191.

## Support

For questions, issues, or contributions, use either of these.
- Email alexander.dickerson1@unsw.edu.au
- GitHub issues at [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)

## Version history

Every release, with what it changed, is in [CHANGELOG.md](../CHANGELOG.md). Stage 0 first shipped
in 1.0.0 (2025-11-01) with Enhanced, Standard and 144A processing, the decimal-shift and
bounce-back correctors, audit logging and the LaTeX reports.

Last updated September 2026.

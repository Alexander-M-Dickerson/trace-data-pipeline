# Stage 0 - TRACE Daily Processing (Enhanced, Standard, 144A)

This stage fetches and cleans TRACE data (Enhanced, Standard, and Rule 144A) on the WRDS Cloud and aggregates to daily panels. Jobs are submitted to Sun Grid Engine (SGE) from a PuTTY session (Windows users); files can be moved to/from WRDS with [WinSCP](https://winscp.net/eng/download.php) (Windows users). Mac users can simply use their terminal once connected to the WRDS Cloud. Mac users might find [ForkLift](https://binarynights.com/) useful -- a GUI file manager that allows uploads, edits, and the ability to manage WRDS Cloud files through a UI. 

Besides generating daily bond pricing panels, the code also generates highly detailed TRACE data reports which document the effect of the filters at the transaction level. It produces (potentially) hundreds of time-series plots of every bond `cusip_id` that is impacted by the decimal shift and bounce-back correctors of Dickerson, Robotti and Rossetti (2026). 

If you want to get things going quickly see **Quick start**. 
Please also see **Generating the TRACE Data Reports** for instructions on how to generate the reports.

---

## Table of Contents

- [Repo Layout](#repo-layout-key-files)
- [How Stage 0 Spends Its Time](#how-stage-0-spends-its-time-and-why-it-is-no-longer-4-hours)
- [Python on the WRDS Cloud](#python-on-the-wrds-cloud)
- [Getting the Code onto WRDS](#getting-the-code-onto-wrds)
- [Requirements](#requirements)
- [Quick Start](#quick-start-wrds-putty)
- [Running Jobs](#running-jobs-individually-alternative)
- [What the Runners Do](#what-the-runners-do)
- [Configuration](#configuration-choices-you-can-edit)
- [Outputs](#outputs)
- [Generating TRACE Data Reports](#generating-the-trace-data-reports)

- [Notes & tips](#notes--tips)
- [Troubleshooting](#troubleshooting)

- [Performance optimization](#performance-optimization)

- [Advanced Usage](#advanced-usage)

- [Monitoring](#monitoring)
- [License & Citation](#license--citation)


- [Version History](#version-history)

- [Support](#support)
---

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

**Important:** submission lives in `../run_pipeline.sh`, which reads `TRACE_MEMBERS` from
`config.py` and submits exactly those members. Enhanced and 144A go in together;
Standard, if requested, is held behind them with `-hold_jid`. The report job is then held
on every stage-0 job actually submitted, and Stage 1 on those same stage-0 jobs — so
Stage 1 and the reports run alongside each other and the whole thing still goes in one
submission.

Each `run_*.sh` is a thin SGE wrapper that sets `-cwd` (current working directory),
exports your environment (`-V`), and writes logs into `stage0/logs/`. The three data-job
scripts set no cores or memory: `run_pipeline.sh` computes them per member from
`CONCURRENCY` and passes them on the `qsub` command line, so the request cannot drift
away from the worker count. `run_build_data_reports.sh` carries its own request
(`-pe onenode 5 -l m_mem_free=8G`).

---

## How Stage 0 spends its time (and why it is no longer ~4 hours)

Stage 0 is a loop over chunks of CUSIPs: fetch a chunk from WRDS, clean it, aggregate it
to bond-days. Until v2.2.0 that loop was strictly serial on one WRDS connection, which is
why Enhanced took about four hours while holding a whole compute node.

**Chunks are now packed by ROW COUNT, not CUSIP count.** Trading activity is enormously
skewed, so 250-CUSIP chunks ranged from 7,806 rows to 3,392,802 over the Enhanced
universe — and the memory a job must reserve is set by the worst chunk, not the average.
Packing to ~750,000 rows brings the worst case to 749,992, a 4.5x reduction, for about
the same number of chunks. `target_rows_per_chunk` controls this. `chunk_size` still
exists and still means CUSIPs-per-chunk, because the report job uses it for its own
independent chunking of flagged CUSIPs.

Re-chunking cannot change the cleaned data. Every per-chunk filter groups by `cusip_id` —
the decimal-shift anchor, the bounce-back scan, the initial-price-error scan, the
Dick-Nielsen reversal keys — and chunks are disjoint CUSIP sets, so which chunk a bond
lands in cannot affect its result. What *does* change is the audit tables, whose `chunk`
column follows the new grouping.

**Several chunks are fetched at once**, each worker process holding its own WRDS
connection. `CONCURRENCY` in `_trace_settings.py` sets how many. `STAGE0_WORKERS` overrides
it for a single run, but for every member at once, 144A included, and neither the connection
check nor the `qsub` request sees the override: keep twice its value at 6 or below.

**Output is sorted canonically before export**, by `(cusip_id, trd_exctn_dt)`. Row order
no longer depends on the work plan, which is what makes it possible to *prove* a
scheduling change did not alter the data — the parquet files come out byte-identical
whether one worker ran the chunks or six did. Measured on 5 chunks: 18 s wall against
81.3 s of serial work.

### The WRDS connection budget

| member | connections | when it runs |
|---|---|---|
| `enhanced` | 5 | alongside 144A |
| `144a` | 1 | alongside Enhanced |
| `standard` | 6 | after both, so it can use the lot |

**The measured ceiling is 7 connections held simultaneously** — the 8th fails. (The "5"
WRDS publishes is the concurrent-*job* limit, a different thing.) Enhanced + 144A is
therefore 6, leaving one spare so a mid-run reconnect cannot be refused.
`validate_connection_budget` enforces this at submit time rather than four hours in.

Measure it on your own account before raising anything:

```bash
python3 tests/probe_wrds_connections.py --max 10
```

❗**A refused connection does not look like one.** The `wrds` package answers *every*
failed connect by re-prompting for a username, so in a batch job — where stdin is closed
— it surfaces as `EOFError: EOF when reading a line`. That single error covers both a
missing `WRDS_USERNAME` and the connection limit; `_wrds_pool.py` distinguishes them and
says which.

### Grid resources

`m_mem_free` is charged **per slot**, so `slots × m_mem_free ≤ 48 GB` (and ≤ 8 cores).
Get this wrong and the job does not error — it pends forever, silently.
`qsub_resources()` derives the request from `CONCURRENCY` and refuses anything over the
caps:

| member | request | total |
|---|---|---|
| `enhanced` | `-pe onenode 5 -l m_mem_free=8G` | 40 GB |
| `144a` | `-pe onenode 1 -l m_mem_free=16G` | 16 GB |
| `standard` | `-pe onenode 6 -l m_mem_free=8G` | 48 GB |

If a job sits at `qw` and will not start, the slot request is the first thing to check:
lower the member's `CONCURRENCY` and resubmit — the resource request follows.

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

❗**Take the whole repository, not just `stage0/`.** Stage 0 does not stand alone:
the root `config.py` is where `WRDS_USERNAME` is set and stage 0 imports it, the job wrappers
`cd stage0` from the repository root and log to `stage0/logs/`, and `run_pipeline.sh`
lives at the root. A `~/proj/stage0/` holding only the stage-0 files cannot be run.

#### Option A - Download a ZIP (no git required)
```bash
mkdir -p ~/proj && cd ~/proj
wget -O trace.zip \
  https://github.com/Alexander-M-Dickerson/trace-data-pipeline/archive/refs/heads/main.zip
unzip trace.zip
mv trace-data-pipeline-main trace-data-pipeline
cd trace-data-pipeline
```

#### Option B - Using `curl` (also no git)
```bash
mkdir -p ~/proj && cd ~/proj
curl -L -o trace.zip \
  https://github.com/Alexander-M-Dickerson/trace-data-pipeline/archive/refs/heads/main.zip
unzip trace.zip
mv trace-data-pipeline-main trace-data-pipeline
cd trace-data-pipeline
```

#### Option C - Clone (if `git` is available on your WRDS node)
```bash
mkdir -p ~/proj && cd ~/proj
git clone https://github.com/Alexander-M-Dickerson/trace-data-pipeline.git
cd trace-data-pipeline
```

You should now have `~/proj/trace-data-pipeline/` holding `config.py`,
`run_pipeline.sh`, `stage0/` and `stage1/`. Run everything from that directory. 

**Note:** `proj` is the directory you have created on your WRDS file system - call it anything you like. Perhaps `trace` is an apt name.

---

### Linux/Mac Users

Open your terminal emulator application (*Terminal*/*iTerm2*) and connect to WRDS with SSH:

```bash
ssh wrds_username@wrds-cloud.wharton.upenn.edu
```

Once connected, you will see a WRDS prompt. From there, you can use **Option A, B, or C** as in the previous section.

#### Option D - Transferring files from your local computer

If you already have the repository on your local computer and want to upload it after editing the scripts, you can use the secure copy protocol `scp`. Upload the whole repository: as the note above says, `stage0/` alone cannot run.

Log in to wrds-cloud with SSH and create a folder called proj:

```bash
mkdir proj
```

Transfer the repository to the proj folder in WRDS cloud:
```bash
scp -r ~/path/to/trace-data-pipeline wrds_username@wrds-cloud.wharton.upenn.edu:/home/university/wrds_username/proj/
```

Replace:
- `~/path/to/trace-data-pipeline` with the path to the repository on your local computer
- `/home/university/wrds_username/` with the full WRDS destination path you see when you run `pwd` after connecting to WRDS

---

## Requirements

**Python version:** 3.10 or higher (the 2026-09-10 production run used Python 3.14.5, the WRDS Cloud default)

**Required packages:** everything in the repository's `requirements.txt`, which states the
minimum versions. For stage 0 that is `pandas`, `numpy`, `wrds`, `SQLAlchemy`,
`pandas_market_calendars`, `pyarrow` and `matplotlib`. matplotlib is not optional: the report
job, which `run_pipeline.sh` always submits, imports it. The log files print your Python and
package versions, so a run can be matched to the environment that made it.

### Installation

From the repository root, on WRDS:
```bash
python -m pip install --user -r requirements.txt
```

[quickstart.md](quickstart.md) explains the choice between `--user` and a virtual environment.

---

## Quick start (WRDS, PuTTY)

### 1. Install required Python packages

SSH into the WRDS cloud, put the repository there (see [Getting the code onto WRDS](#getting-the-code-onto-wrds)), and install the packages in `requirements.txt` (see [Requirements](#requirements)).

### 2. Navigate to the repository and configure settings

```bash
cd ~/proj/trace-data-pipeline   # wherever you put the repository
```

**CRITICAL:** set your WRDS username. It lives in the shared `config.py` at the repo
ROOT -- `_trace_settings.py` imports it from there, so editing `_trace_settings.py` will
not do anything.

The simplest way is an environment variable, which needs no file edited at all:

```bash
export WRDS_USERNAME="your_wrds_id"
echo 'export WRDS_USERNAME="your_wrds_id"' >> ~/.bashrc   # make it persistent
```

Or edit the fallback in `config.py`:

```bash
nano config.py
# WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")
```
Save and exit `nano` with Ctrl+O, Enter, then Ctrl+X. Confirm with
`python3 -c "from config import WRDS_USERNAME; print(WRDS_USERNAME)"`.

❗If this is left as the placeholder `your_wrds_username`, the run dies with
`EOFError: EOF when reading a line` -- the `wrds` package prompting on a closed stdin.
That looks exactly like the connection limit and is not.

The WRDS password should be handled by the `.pgpass` file which you should have set up following the WRDS documentation.

Review the default filter settings in `_trace_settings.py`. All filters but two (`trading_time` and `volume_filter_toggle`) are on by default, with recommended values from Dickerson, Robotti and Rossetti (2026). See the [Configuration](#configuration-choices-you-can-edit) section for more details.

### 3. Make scripts executable (older clones only)

The scripts ship executable, so a fresh clone needs nothing here. A clone made before the
executable bits were committed may need:

```bash
chmod +x run_pipeline.sh download_inputs.sh run_smoke_test.sh stage0/run_*.sh stage1/run_stage1.sh
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
./run_pipeline.sh
```

**What happens:**
1. Reads `TRACE_MEMBERS` from `config.py` and submits exactly those members. The
   default is `["enhanced", "144a"]`; Standard is opt-in.
2. Enhanced and 144A go in together -- their WRDS connection budgets are sized to
   co-exist. Each asks for cores and memory matched to its worker count.
3. Standard, if requested, is held behind them with `-hold_jid`, so it runs alone and
   can use the whole connection budget rather than a slice of it.
4. `build_reports` is held on every stage-0 job actually submitted.
5. Stage 1 is held on **the same stage-0 jobs**, not on the report job. Since v2.2.2
   the two run alongside each other, so the report job is off the critical path
   entirely. That release also parallelised the report's re-clean: the job went from ~51
   minutes to about 20 (22 min on the 2026-09-21 run).

**Output from the script:**
```
[info] members: enhanced 144a
[submit] enhanced TRACE  (-pe onenode 5 -l m_mem_free=8G) ...
         job 4821094
[submit] 144a TRACE  (-pe onenode 1 -l m_mem_free=16G) ...
         job 4821095
[submit] Build data reports (waits for 4821094,4821095) ...
```

> **Tip:** Check status with `qstat`. The report job will show status `hqw` (hold) until the data jobs finish. Tail logs with `tail -f stage0/logs/01_enhanced.out` (or `.err`).

**Total runtime:** about 5 hours for the complete pipeline. The 2026-09-21 run, which the 2026
vintage was built from, took 5.3 h end to end: Enhanced 2.58 h and 144A 0.63 h in parallel,
then the reports (0.37 h) and Stage 1 (2.69 h) side by side.

If you are having errors after attempting to debug, feel free to contact Alex Dickerson at `alexander.dickerson1@unsw.edu.au` for help.

---

## Running jobs individually (alternative)

If you prefer submitting by dataset, run:

Submit **from the repository root** -- the wrappers `cd stage0` themselves and write
their logs to `stage0/logs/`, so a bare `qsub run_enhanced_trace.sh` from inside
`stage0/` fails before it runs a line.

❗**Pass the resource request yourself.** The wrappers carry no `-pe`/`-l` directives;
`run_pipeline.sh` supplies them on the qsub command line from `qsub_resources()`. Submit
one by hand without them and SGE gives the job a single slot with default memory, while
the code still opens `CONCURRENCY[member]` connections -- they contend for one core.

```bash
qsub -pe onenode 5 -l m_mem_free=8G  stage0/run_enhanced_trace.sh    # enhanced
qsub -pe onenode 6 -l m_mem_free=8G  stage0/run_standard_trace.sh    # standard
qsub -pe onenode 1 -l m_mem_free=16G stage0/run_144a_trace.sh        # 144a
```

Those are the current values; print them rather than copying if you have changed
`CONCURRENCY` or `MEM_PER_SLOT_GB`:

```bash
cd stage0 && python3 -c "from _trace_settings import qsub_resources; print(qsub_resources('enhanced'))"
```

Each wrapper uses SGE's `-cwd` so outputs/logs land under the current folder, and `-V` to pass your Python environment variables.

---

### Running report generation separately

If you run jobs individually and want to generate reports later, or if you want to regenerate reports with different settings:

```bash
qsub stage0/run_build_data_reports.sh
```

**Important:** which datasets the report job processes comes from `TRACE_MEMBERS` in
the root `config.py`, or from the `--data-type` flag. `DATA_TYPES` was moved out of
`_build_error_files.py` (its source still carries the note "DATA_TYPES moved to config.py
as TRACE_MEMBERS"), so adding it back there has no effect:

```bash
# One dataset, without touching config.py (from the repository root):
cd stage0 && python3 _build_error_files.py --data-type enhanced
```

```python
# In config.py -- read by run_pipeline.sh, the report job and Stage 1:
TRACE_MEMBERS = ["enhanced"]
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
- Computes bid/ask prices (value-weighted)
- Generates comprehensive audit logs for each filter stage
- **Saves all outputs to `enhanced/` subfolder**

### Standard TRACE (`_run_standard_trace.py`)

Calls `CreateDailyStandardTRACE` with the same controls for the Standard table:
- Default start date is set to `2024-10-01` (you can change this in `_trace_settings.py`)
- Removes cancellations, corrections and reversals using Standard TRACE's status codes (the
  pre/post-2012 rules and the agency de-duplication are Enhanced's)
- Same decimal-shift and bounce-back filters as Enhanced
- Same daily aggregation metrics
- **Saves all outputs to `standard/` subfolder**

### Rule 144A TRACE (`_run_144a_trace.py`)

Calls `CreateDailyStandardTRACE` with `data_type='144a'`:
- Default start date is `2002-07-01`, though the data are negligible before 2014 (29 bond-days before 2014 in the 2026-09-21 run, then 70,668 in 2014)
- Uses the same cleaning pipeline as Standard TRACE
- Same parameter blocks for filters and aggregation
- **Saves all outputs to `144a/` subfolder**

### Core processing steps

Both `create_daily_enhanced_trace.py` and `create_daily_standard_trace.py`:

1. **Connect to WRDS** and establish database connection
2. **Filter FISD universe** based on configured parameters (USD only, fixed-rate, non-convertible, etc.)
3. **Plan the chunks** by packing CUSIPs to `target_rows_per_chunk` trade rows
   (default 750,000). `chunk_size` (250 CUSIPs) is only the fallback, used when the
   row-count query is unavailable.
4. **For each chunk** — several at once, one WRDS connection per worker. The filters run
   in this order, and the numbering matches the `# Filter N:` comments in the source:

   | # | Filter | What it does |
   |---|---|---|
   | 0 | price-scale normalization | rescales unit-quoted bonds to percent of par (a no-op under the default FISD screen) |
   | 1 | Dick-Nielsen | cancellations, corrections, reversals; agency de-duplication in Enhanced only |
   | 2 | decimal-shift corrector | fixes multiplicative price errors (10x, 0.1x, 100x, 0.01x) |
   | 3 | trading time | intraday window — **off by default** |
   | 4 | trading calendar | drops non-session dates |
   | 5 | price range | `> 0` and `<= 1000` |
   | 6 | trading volume | dollar or par threshold |
   | 7 | **bounce-back** | flags price spikes that revert — runs AFTER the volume filter, not before |
   | 8 | yield != price | drops rows where the reported yield equals the price |
   | 9 | amount outstanding | drops volume above 50% of the offering amount |
   | 10 | execution date vs maturity | drops trades after maturity |
   | 11 | **initial price error** | flags implausible opening prints |

   then aggregate to a daily `(cusip_id, trd_exctn_dt)` panel.
5. **Reassemble the chunks in order**, sort canonically by `(cusip_id, trd_exctn_dt)`,
   and export to Parquet.
6. **Generate audit logs** with row counts for each filter stage.
7. **Export CUSIP lists** for the decimal-shift, bounce-back and initial-price-error
   filters (three files, one per filter).

---

## Configuration choices you can edit

Open `_trace_settings.py` and adjust the following. (❗Not the WRDS username --
that lives in the shared `config.py` at the repo ROOT; `_trace_settings.py` imports it,
so editing it there does nothing. See "Quick start" above.)

### FISD Universe Parameters (`FISD_PARAMS`)

Controls which bonds are included in the universe:
- `currency_usd_only`: Keep only USD-denominated bonds (default: `True`)
- `fixed_rate_only`: Exclude variable-rate bonds (default: `True`)
- `non_convertible_only`: Exclude convertible bonds (default: `True`)
- `non_asset_backed_only`: Exclude asset-backed securities (default: `True`)
- `exclude_bond_types`: Drop specific bond types like TXMU, MBS, FGOV, etc. (default: `True`)
- `valid_coupon_frequency_only`: Drop bonds with invalid interest frequency (default: `True`)
- `require_accrual_fields`: Require offering_date, dated_date to be non-null (default: `True`)
- `principal_amt_eq_1000_only`: Keep only bonds with $1000 par value (default: `True`).
  **If you turn this off, leave `PRICE_NORM` on** -- see below.
- `exclude_equity_index_linked`: Exclude equity-linked and index-linked bonds (default: `True`)
- `enforce_tenor_min`: Require bonds to have minimum tenor (default: `True`)
- `tenor_min_years`: Minimum tenor in years (default: `1.0`)

### Price-Scale Normalization (`PRICE_NORM`)

- `normalize_nonpar1000`: Rescale unit-quoted bonds to percent of par (default: `True`)

TRACE's `rptd_pr` is a **percent of par** for the standard $1,000-principal bond: at
par it prints `100`. Small-denomination issues -- retail and structured notes with a
principal of $10, $25 or $100 -- are quoted in **unit dollars** instead, so a $10 note
at par prints `10.00`, not `100`.

Everything downstream assumes percent of par: the price bounds, the decimal-shift
gates, the bounce-back point threshold, Stage 1's ultra-distressed thresholds, dollar
volume (`entrd_vol_qt * rptd_pr / 100`) and QuantLib (face value 100). Left alone, a
perfectly healthy $10 note reads as a bond trading at 10% of par -- flagged as
distressed, with dollar volume understated tenfold and a nonsense yield.

When enabled, each CUSIP whose `principal_amt` is not 1000 is rescaled by
`100 / principal_amt`, but only if that moves the bond's **median** price closer to par
in log distance. The decision is made once per bond from its median, so a run of
distressed prints cannot flip the convention. $1,000-principal bonds are never touched.

**Under the default settings this does nothing.** `principal_amt_eq_1000_only` is
`True`, so no non-$1,000 bond is in the universe and every factor resolves to `1.0` --
the output is identical whether the toggle is on or off. It matters only when you turn
that screen off, which is precisely when the tape fills with unit-quoted notes.

### Filter Toggles (`FILTER_SWITCHES`)

All filters are boolean toggles:
- `dick_nielsen`: Apply Dick-Nielsen cleaning steps (default: `True`)
- `decimal_shift_corrector`: Fix decimal shift errors (default: `True`)
- `trading_time`: Filter by intraday time window (default: `False`)
- `trading_calendar`: Keep only valid trading days (default: `True`)
- `price_filters`: Remove negative prices and prices > 1000 (default: `True`)
- `volume_filter_toggle`: Apply a minimum-size threshold to each trade (default: **`False`**)
  - ❗Off by design. The published OSBAP data keeps trades of **all** sizes. Turning it
    on removes ~26% of Enhanced trades per bond-day and changes every volume-weighted
    price, so a panel built with it on will not reconcile against the published data.
- `bounce_back_filter`: Flag price-change errors (default: `True`)
- `yld_price_filter`: Remove rows where yield = price (default: `True`)
- `amtout_volume_filter`: Remove trades > 50% of offering amount (default: `True`)
- `trd_exe_mat_filter`: Remove trades after maturity date (default: `True`)
- `flag_initial_price_errors`: Flag implausible opening prints (default: `True`)

❗Write all ELEVEN keys in any `FILTER_SWITCHES` you write. A key you leave out raises no
error: it falls back to the engine's own default, which is not always the settings file's.
Leave out `volume_filter_toggle` and the $10,000 dollar floor switches ON.
`trading_time` and `volume_filter_toggle` are the two off by default.

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

### Initial-Price-Error Parameters (`INIT_ERROR`)

The last filter in the cascade. It looks at each bond's FIRST few prints and flags them
when the price then moves sharply — the pattern of a bond whose opening marks were
placeholders rather than trades.

- `abs_change`: the price move, in points of par, that marks the earlier prints as
  errors — `50.0`
- `n_transactions`: how many opening prints to examine — `3`

Flagged rows are dropped, and the affected CUSIPs are written to
`init_price_cusips_{dtype}_{stamp}.parquet` for the data report, which draws them as the
`_ie` figure pages.

### Bounce-Back Filter Parameters (`BB_PARAMS`)

Fine-tune the bounce-back price-error detection:
- `threshold_abs`: Minimum absolute price jump to flag a candidate error - `35.0`
- `lookahead`: Maximum rows ahead to search for bounce - `5`
- `max_span`: Maximum path length from start to resolution - `5`
- `window`: Backward window for trailing median anchor - `5`
- `back_to_anchor_tol`: how close the price must come back to the anchor, as a fraction of `threshold_abs` (0.25 × 35 = 8.75 points) - `0.25`
- `candidate_slack_abs`: subtracted from `threshold_abs` when opening a candidate (a jump of 34 points opens one) - `1.0`
- `reassignment_margin_abs`: Margin for tie-breaking in clusters - `5.0`
- `use_unique_trailing_median`: Use unique values in median - `True`
- `par_spike_heuristic`: Enable special handling at par - `True`
- `par_level`: Par value - `100.0`
- `par_equal_tol`: Tolerance for treating as par - `1e-8`
- `par_min_run`: Minimum par run length to flag - `3`
- `par_cooldown_after_flag`: Rows to skip after flagging - `2`

### Common Arguments (`COMMON_KWARGS`)

Settings applied to all runners:
- `output_format`: `"parquet"`, the only supported value. It comes from `OUTPUT_FORMAT` in
  the root `config.py`, and `_trace_settings.py` refuses anything else when it loads
- `chunk_size`: CUSIPs per batch - `250`. Since v2.2.0 this no longer decides Stage 0's
  own chunking (see `target_rows_per_chunk`), but it is still read by the report job for
  its independent chunking of flagged CUSIPs, so it is kept.
- `target_rows_per_chunk`: trade rows per chunk - `750_000`. THIS is what sizes Stage 0's
  work units. Set to `None` to fall back to fixed `chunk_size` chunks.
  Override for one run with `STAGE0_TARGET_ROWS`.
- `limit_chunks`: process only the first N chunks - `None`. A dev/test escape hatch;
  never set it for a production run. Override with `STAGE0_LIMIT_CHUNKS`.
- `clean_agency`: Apply agency de-duplication - `True`
- `out_dir`: Output directory - `""` (current directory)
- `volume_filter`: Tuple of `(kind, threshold)`, used **only** when
  `volume_filter_toggle` is `True`:
  - `("dollar", 10000)`: dollar volume (`entrd_vol_qt * rptd_pr / 100`) >= $10,000
  - `("par", 10000)`: par volume >= $10,000
- `trade_times`: Intraday window - `["00:00:00", "23:59:59"]` (effectively disabled)
- `calendar_name`: Market calendar - `"NYSE"`

### Per-Dataset Overrides (`PER_DATASET`)

Specific settings for each dataset:
- **Enhanced**: `n_workers` only. Enhanced takes no `start_date`: it always starts at 2002-07-01
- **Standard**: `start_date="2024-10-01"`, `data_type="standard"`, `n_workers`
- **144A**: `start_date="2002-07-01"`, `data_type="144a"`, `n_workers`

`n_workers` is how many chunks are fetched at once, one WRDS connection each. It comes from
`CONCURRENCY` for each member; `STAGE0_WORKERS` overrides it for one run (see the warning under
[How Stage 0 spends its time](#how-stage-0-spends-its-time-and-why-it-is-no-longer-4-hours)).

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

**Logs** (under `stage0/logs/`):
- `01_enhanced.out`, `01_enhanced.err`: Enhanced TRACE job logs
- `02_standard.out`, `02_standard.err`: Standard TRACE job logs
- `03_144a.out`, `03_144a.err`: 144A TRACE job logs
- `_data_reports.out`, `_data_reports.err`: Report generation logs
- Logs contain timestamps, row counts, filter statistics, and any errors

**Daily panels** (Parquet format, in respective subfolders):
- `enhanced/trace_enhanced_YYYYMMDD.parquet`: Enhanced TRACE daily panel (~31 million rows for the full sample)
- `standard/trace_standard_YYYYMMDD.parquet`: Standard TRACE daily panel
- `144a/trace_144a_YYYYMMDD.parquet`: Rule 144A daily panel

All panels have identical column NAMES -- 21 of them, every one defined in
[DATA_DICTIONARY.md](DATA_DICTIONARY.md). Column ORDER is not part of the contract and does
differ between members.
- **Keys**: `cusip_id`, `trd_exctn_dt`
- **Prices**: `prc_ew`, `prc_vw`, `prc_vw_par`, `prc_first`, `prc_last`, `prc_hi`, `prc_lo`
- **Trade timing**: `time_ew`, `time_last`
- **Volumes** (millions): `qvolume`, `dvolume`
- **Counts**: `trade_count`, `bid_count`, `ask_count`
- **Bid/Ask**: `prc_bid`, `prc_ask`, `bid_last`, `bid_time_ew`, `bid_time_last`

**Audit files** (Parquet format, in respective subfolders):
- `dick_nielsen_filters_audit_{dtype}_{date}.parquet`: Dick-Nielsen step-by-step audit
- `drr_filters_audit_{dtype}_{date}.parquet`: Dickerson-Rossetti-Robotti filter audit
- `fisd_filters_{dtype}_{date}.parquet`: FISD universe construction audit

**CUSIP lists** (Parquet format, in respective subfolders) -- one per corrector, and
the data report draws a figure family from each:
- `bounce_back_cusips_{dtype}_{date}.parquet`: CUSIPs with bounce-back flags (`_bb` pages)
- `decimal_shift_cusips_{dtype}_{date}.parquet`: CUSIPs corrected by the decimal-shift
  corrector (`_ds` pages)
- `init_price_cusips_{dtype}_{date}.parquet`: CUSIPs with flagged opening prints
  (`_ie` pages)

**Also written, and easy to miss because they match neither wildcard above:**
- `trace_enhanced_fisd_{date}.parquet` (Enhanced) / `trace_fisd_{dtype}_{date}.parquet`
  -- the screened FISD universe. ❗**Stage 1 reads the Enhanced one**, so it is an input
  to the next stage, not just an artifact.
- `cusip_row_counts_{date}.parquet` -- the per-CUSIP row counts behind the chunk
  packing (31 s to count on the 2026-09-21 run). Reused only by a re-run on the same day.

### Downloading outputs

**Windows users:** Use WinSCP to download the entire `enhanced/`, `standard/`, `144a/`, `data_reports/`, and `logs/` folders to your local machine.

**Mac/Linux users:** Use `scp` from your local machine:
```bash
# Download all outputs preserving folder structure
scp -r wrds_username@wrds-cloud.wharton.upenn.edu:~/proj/trace-data-pipeline/stage0/enhanced ./local_destination/stage0/
scp -r wrds_username@wrds-cloud.wharton.upenn.edu:~/proj/trace-data-pipeline/stage0/standard ./local_destination/stage0/
scp -r wrds_username@wrds-cloud.wharton.upenn.edu:~/proj/trace-data-pipeline/stage0/144a ./local_destination/stage0/
scp -r wrds_username@wrds-cloud.wharton.upenn.edu:~/proj/trace-data-pipeline/stage0/data_reports ./local_destination/stage0/
scp -r wrds_username@wrds-cloud.wharton.upenn.edu:~/proj/trace-data-pipeline/stage0/logs ./local_destination/stage0/
```

---

## Generating the TRACE Data Reports

When you run `./run_pipeline.sh`, reports are generated automatically for the members in `TRACE_MEMBERS` (by default Enhanced and 144A) once their data jobs complete, using SGE's `-hold_jid`. Since v2.2.2 the report job runs alongside Stage 1 rather than before it. You can also generate or regenerate reports separately.

### Configuration

Two settings in the root `config.py` control the report job:

```python
STAGE0_OUTPUT_FIGURES = True   # False: tables only (also settable from the environment)
TRACE_MEMBERS = ["enhanced", "144a"]   # which datasets get a report
```

`DATE`, `IN_DIR` and `OUT_DIR` at the top of `_build_error_files.py` can stay blank.

**Important notes:**
- With figures, the job re-cleans the flagged bonds and draws hundreds of time-series
  plots: 22 minutes for the whole job on the 2026-09-21 run
- `STAGE0_OUTPUT_FIGURES = False` builds only the filter tables and runs in seconds
- For Enhanced TRACE with figures, the LaTeX document can exceed 500 pages

### Running the report generator separately

If you want to regenerate reports with different settings or didn't run `./run_pipeline.sh`:

Submit the job:
```bash
qsub stage0/run_build_data_reports.sh
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

Each `*_data_report.tex` file includes:
1. **Table 1**: Filter toggles and parameter settings
2. **Table 2**: FISD universe construction parameters
3. **Table 3**: Transaction-level filter records (Panel A: FISD, Panel B: DRR filters, Panel C: Dick-Nielsen)
4. **Figures** (if enabled): Time-series plots showing decimal-shift corrections, bounce-back eliminations and initial-price errors

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

- **Automated workflow**: `run_pipeline.sh` uses SGE's `-hold_jid` to create job
  dependencies, built from the jobs actually submitted. The report job waits at `hqw`
  until every stage-0 job finishes. This is the recommended workflow.

- **Job dependency**: with the default `TRACE_MEMBERS = ["enhanced", "144a"]` you will
  see four jobs in `qstat`:
  - `trace_enhanced` - running or queued, 5 slots
  - `trace_144a` - running or queued, 1 slot
  - `build_reports` - `hqw` until both finish
  - `stage1_pipeline` - `hqw` until the DATA jobs finish, then runs alongside
    `build_reports` (it does not read anything the reports produce)

  Add `standard` to `TRACE_MEMBERS` and a fifth appears, itself held behind the other
  two rather than running beside them.

- **Memory considerations**: chunks are packed to ~750,000 trade rows, and each worker
  holds one chunk at a time. If you hit memory trouble, lower `target_rows_per_chunk` or
  `CONCURRENCY` — in that order, since the per-worker peak is set by chunk size.

- **Runtime expectations**:
  - Enhanced TRACE (full sample): about 4 hours serial before v2.2.0; since then it pulls 5
    chunks at once and takes 2-2.6 hours (2.0 h on 2026-09-09, 2.6 h on 2026-09-21), about
    twice as fast.
  - Standard TRACE (from 2024): 30-60 minutes, and opt-in
  - Rule 144A (full sample): 30-60 minutes (39 minutes on the 2026-09-10 run)
  - Data reports (with figures): was ~50 minutes for Enhanced; since v2.2.2 its
    re-clean pulls 5 chunks at once (22 minutes for the whole job on 2026-09-21), and the job
    no longer blocks stage 1

- **Disk space**: Enhanced TRACE generates ~31M rows. On the 2026-09-10 run the Enhanced panel was about 2.2 GB and the 144A panel about 250 MB, and the whole `stage0/` folder about 2.7 GB.

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
  chmod +x run_pipeline.sh download_inputs.sh stage0/run_*.sh
  ```

- **Permission denied** when trying to run `./run_pipeline.sh`:
  You probably missed the `chmod +x` step above.

- **Bad interpreter** or `^M` errors:
  Convert Windows line endings to Unix:
  ```bash
  sed -i 's/\r$//' ../run_pipeline.sh
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
  - Lower `TARGET_ROWS_PER_CHUNK` in `_trace_settings.py` (try 400_000) -- this, not
    `chunk_size`, sizes a Stage 0 chunk. Or lower `CONCURRENCY`; each worker holds one.
  - Or give each worker more memory: raise `MEM_PER_SLOT_GB` in `_trace_settings.py`.
    `run_pipeline.sh` passes it to `qsub`, so editing the job scripts changes nothing.
    Slots × memory must stay within the WRDS limit of 48 GB per job, and `qsub_resources()`
    refuses a request over it.

### Report generation issues

- **Figures not generating**:
  - Verify that `matplotlib` is installed
  - Check that CUSIP lists exist in the expected location
  - Ensure `STAGE0_OUTPUT_FIGURES = True` in the root `config.py`

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

If a job fails:
1. Review the error log: `cat stage0/logs/01_enhanced.err`
2. Fix the issue in configuration or code
3. Resubmit from the repo root: `qsub -pe onenode 5 -l m_mem_free=8G stage0/run_enhanced_trace.sh` (or just `./run_pipeline.sh`)

---

## Advanced Usage

### Custom date ranges

To process a specific date range, modify the per-dataset overrides in `_trace_settings.py`:

```python
# ❗Keep `n_workers` on every member. Dropping it returns that member to the serial
# path (the engine default is n_workers=1) while run_pipeline.sh still requests 5 slots.
PER_DATASET = {
    "enhanced": dict(n_workers=WORKERS_OVERRIDE or CONCURRENCY["enhanced"]),
    "standard": dict(start_date="2020-01-01", data_type="standard",
                     n_workers=WORKERS_OVERRIDE or CONCURRENCY["standard"]),
    "144a":     dict(start_date="2020-01-01", data_type="144a",
                     n_workers=WORKERS_OVERRIDE or CONCURRENCY["144a"]),
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

❗Stage 1 reads `stage0/` beside it and the report job reads its current folder, so neither
finds output written anywhere else. Use this only when you run stage 0 on its own.

### Disabling specific filters

To disable any filter, set it to `False` in `_trace_settings.py`:

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

Chunks are packed to `target_rows_per_chunk` trade rows (default `750_000`), not to a
fixed CUSIP count. Size it against the memory available per worker: the peak inside
`decimal_shift_corrector` is roughly 2.5x the raw chunk, because it copies the frame and
adds columns.

- **Lower memory per worker**: `target_rows_per_chunk = 400_000`
- **Fewer, larger chunks**: `target_rows_per_chunk = 1_500_000`
- **Restore the old fixed-CUSIP behaviour**: `target_rows_per_chunk = None`

The row counts behind the packing are measured once per run and cached beside the output
as `cusip_row_counts_<stamp>.parquet` (31 s for the Enhanced universe on the 2026-09-21 run). If that
query fails, the planner falls back to fixed `chunk_size` chunks rather than aborting.

### Parallel processing

Two levels, and they compose:

**Within a job**, `CONCURRENCY` decides how many chunks are fetched at once, each on its
own WRDS connection.

**Across jobs**, `run_pipeline.sh` submits Enhanced and 144A together, and holds Standard
behind them:
```bash
./run_pipeline.sh
```

### Output format

Parquet only. `OUTPUT_FORMAT` in the root `config.py` must be `"parquet"`: Stage 1 and the
report job read Parquet files, and `_trace_settings.py` refuses any other value when it loads.
To get CSV, convert the Parquet files afterwards (see the [FAQ](../FAQ.md)).

---

## License & Citation

### License

This code is provided under the MIT License. See LICENSE file for details.

### Citation

If you use or extend this stage, please cite:

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

Every release, with what it changed, is in [CHANGELOG.md](../CHANGELOG.md). Stage 0 first shipped
in 1.0.0 (2025-11-01) with Enhanced, Standard and 144A processing, the decimal-shift and
bounce-back correctors, audit logging and the LaTeX reports.

---

**Last updated:** September 2026





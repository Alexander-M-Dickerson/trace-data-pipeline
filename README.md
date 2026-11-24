# TRACE Data Pipeline

This pipeline turns raw Enhanced, Standard and 144A TRACE (Trade Reporting and Compliance Engine) corporate bond transactions into cleaned daily and monthly bond panels.
It is part of the [Open Source Bond Asset Pricing project](https://openbondassetpricing.com/).
The companion library [PyBondLab](https://pypi.org/project/pybondlab/) forms corporate bond asset pricing factors, and Stages 2 to 4 run on it.

[![Website](https://img.shields.io/badge/Website-Visit-blue?logo=google-chrome&logoColor=white)](https://openbondassetpricing.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![tests](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/actions/workflows/tests.yml)
[![Stage 0](https://img.shields.io/badge/Stage%200-Public%20Beta-green)](stage0/)
[![Stage 1](https://img.shields.io/badge/Stage%201-Public%20Beta-green)](stage1/)
[![Stage 2](https://img.shields.io/badge/Stage%202-Public%20Beta-green)](stage2/)
[![Stage 3](https://img.shields.io/badge/Stage%203-Public%20Beta-green)](stage3/)
[![Stage 4](https://img.shields.io/badge/Stage%204-Public%20Beta-green)](stage4/)

[Link to paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6088966)

## Which machine am I on?

The pipeline runs in two places, WRDS and your own computer, and you copy one zip file from the
first to the second.

| | Where | What happens | How long |
|---|---|---|---|
| 1 | WRDS Cloud | Stages 0 and 1 build the daily bond panel from the raw TRACE transactions | ~5 hours, mostly waiting |
| 2 | In between | You zip the output on WRDS and copy it down to your own computer | ~15 min for ~5.6 GB |
| 3 | Your own computer | Stage 2 turns that daily panel into the monthly asset-pricing panel | ~8-18 minutes |
| 4 | Your own computer | Stage 3 turns the monthly panel into sorted portfolios and the paper's exhibits | ~15-20 minutes |
| 5 | Your own computer | Stage 4 builds the TRACE-only bond factors openbondassetpricing.com publishes, and checks yours against them | ~8 minutes |

Stages 0 and 1 read the raw TRACE transaction data. The TRACE transactions are a WRDS database, so both stages
run on WRDS, as jobs on its grid. Stage 2 works from the files Stage 1 produced and runs faster
on your own machine. Its first run still connects to WRDS once, to fetch and cache a few
series (CRSP Treasury returns, Fama-French factors, VIX and FISD coupon terms), so it needs
your WRDS login. Later runs read the cache.

### Step by step

#### On WRDS

You are logged in to `wrds-cloud.wharton.upenn.edu` over SSH.

1. Clone this repository into your WRDS home directory (`cd ~` first).
2. Install the Python packages (`python -m pip install --user -r requirements.txt`).
3. Set your WRDS username with `export WRDS_USERNAME=your_wrds_id`, or edit `config.py`.
4. Run `bash download_inputs.sh` on the login node. Compute nodes have no internet.
5. Submit the pipeline with `./run_pipeline.sh`. Stage 0 runs in parallel per TRACE member, then
   Stage 1 starts automatically when they finish. Check on it with `qstat`.

#### Moving the results

The output is roughly 5.6 GB across hundreds of files.

6. On WRDS, zip it with `cd ~ && rm -f "/scratch/$(basename "$(dirname "$HOME")")/trace-data-pipeline.zip" && zip -r "/scratch/$(basename "$(dirname "$HOME")")/trace-data-pipeline.zip" trace-data-pipeline/`
7. From your own computer, copy it down with `scp`, then unzip.

> Zip from `~` using a relative path, exactly as written. `zip -r out.zip ~/trace-data-pipeline/`
> stores the absolute path and you get an archive nested four directories deep.
> `$(basename "$(dirname "$HOME")")` is your institution code, the folder between `/home` and your username. Type the line as written, because nothing in it is a placeholder. In the `scp` line, `{institution}` is a placeholder. Replace it, braces included, with that code (`echo $HOME` on WRDS shows it).

#### On your own computer

8. In a Python 3.11 to 3.13 environment, install the requirements for stages 2 to 4 with two
   lines, `python -m pip install -r requirements-local.txt` and then
   `python -m pip install --no-deps pybondlab==0.3.0`. Python 3.14 cannot install them yet, and
   `requirements-local.txt` says why. PyBondLab 0.3.0 declares `numpy<2` and
   this repository installs numpy 2, so it goes in without its dependency list
   (`requirements-local.txt` says why that is safe). Stage 2's first run fetches a few series
   from WRDS, so set your username in `config.py` or `export WRDS_USERNAME=...`.
9. `cd stage2 && python _run_stage2.py`. Add `--factor-source pinned` to use the factors a
   published panel was built with (see [stage2/QUICKSTART_stage2.md](stage2/QUICKSTART_stage2.md)).
10. You now have `stage2/output/panel/main_panel_<mode>.parquet`, with 145 columns per bond-month.

#### The research output, also on your own computer

11. `cd ../stage3 && python tools/check_inputs.py` confirms Stage 3 can see what Stage 2 made.
12. `bash run_stage3.sh` writes the portfolio sorts, the uncertainty grids, and 33 tables and
    11 figures into `stage3/reports/`.
13. `cd ../stage2 && python make_excess_blocks.py --mode stage1 --verify`, then `--benchmark all`, then
    `cd ../stage4 && bash run_stage4.sh`. This writes the TRACE-only bond factors that
    openbondassetpricing.com publishes into `stage4/output/`, then compares them cell by cell
    with the published files.

On Windows, run the `bash` lines in Git Bash. Each wrapper only calls Python scripts, which
work from any terminal. `run_stage3.sh` is `python tools/check_inputs.py` then
`python _run_stage3.py`, and `run_stage4.sh` is `python build_factors.py` then
`python compare_published.py`.

Stages 0 to 2 build the data. Stage 3 reproduces every
exhibit of *The Corporate Bond Factor Replication Crisis* from the panel you just made.
Stage 4 builds the TRACE-only bond factors from the same panel. Stages 3 and 4 are optional,
and the panel is useful without them.

The full detail is in [QUICKSTART.md](QUICKSTART.md) for stages 0 and 1,
[stage2/QUICKSTART_stage2.md](stage2/QUICKSTART_stage2.md) for stage 2,
[stage3/QUICKSTART_stage3.md](stage3/QUICKSTART_stage3.md) for stage 3 and
[stage4/README_stage4.md](stage4/README_stage4.md) for stage 4.

An AI assistant (Claude Code, Codex, Copilot and others) reads
[AGENTS.md](AGENTS.md) by itself, and a skill walks each stage step by step (see
[Documentation](#documentation)). `python doctor.py` says what is ready and what to run next.

> You cannot skip stages 0 and 1 by downloading the published Stage 1 file. The public
> download has its rating columns removed, because agency ratings are licensed. Stage 2 keeps
> only bond-days that have a rating, so it cannot use that file, and it refuses it at
> start-up with a message that says so. Run stages 0 and 1 yourself.

## Overview

The pipeline builds TRACE corporate bond datasets in five stages. Stages 0 and 1 run on WRDS, and
Stages 2, 3 and 4 run on your own computer.

### Stage 0: Intraday to daily processing  (public beta)

Stage 0 turns raw intraday TRACE transactions into clean daily panels. It handles three types of
TRACE data.
- Enhanced TRACE
- Standard TRACE
- Rule 144A bonds

From the project root, `./run_pipeline.sh` runs Stages 0 and 1 on WRDS and sets the job
dependencies automatically. The Stage 0 jobs run in parallel, and Stage 1 starts
automatically when they finish.

Stage 0 is a public beta and is ready for testing. It runs on the WRDS Cloud grid, because it
reads the raw TRACE transactions, which are on WRDS. Its documentation is
[stage0/README_stage0.md](stage0/README_stage0.md), [stage0/quickstart.md](stage0/quickstart.md)
and [stage0/DATA_DICTIONARY.md](stage0/DATA_DICTIONARY.md).

### Stage 1: Daily bond analytics  (public beta)

Stage 1 adds bond analytics and characteristics to the Stage 0 daily panels.
- Bond analytics via QuantLib (duration, convexity, YTM, credit spreads)
- Credit ratings from S&P and Moody's with numeric conversions
- Equity identifiers (CRSP PERMNO/PERMCO and Compustat GVKEY) from the bond-firm linker
- FISD bond characteristics (coupon, maturity, issuer, amount outstanding, etc.)
- Fama-French industry classifications (12, 17 and 30 industries)
- Ultra-distressed filters to flag potentially erroneous prices

Stage 1 is a public beta and is ready for testing. It runs on the WRDS Cloud grid and starts
automatically after Stage 0. Its documentation is
[stage1/README_stage1.md](stage1/README_stage1.md),
[stage1/QUICKSTART_stage1.md](stage1/QUICKSTART_stage1.md) and
[stage1/DATA_DICTIONARY.md](stage1/DATA_DICTIONARY.md).

### Stage 2: Monthly panel with factor signals

Stage 2 produces a clean, error-corrected monthly panel with dozens of corporate bond signals for
asset pricing research.
- 108 bond characteristic signals across 145 panel columns
- Credit risk factors
- Liquidity measures
- Momentum and reversal signals
- Spread, yield and value signals
- A panel you can use directly for monthly portfolio construction (see
  [PyBondLab](https://pypi.org/project/pybondlab/))

Stage 2 is complete. It builds, it is tested and gated, and it is released. Published vintages
are at [openbondassetpricing.com](https://openbondassetpricing.com). It runs on your own machine,
not on the WRDS grid, and a WRDS subscription is still required.

### Stage 3: Sorts and the paper's exhibits

Stage 3 turns the Stage 2 panel into portfolio sorts, two uncertainty grids, and 33 tables and 11
figures, compiled into one PDF. These are 29 tables and the figures of *The Corporate Bond Factor
Replication Crisis*, and four tables of Stage 3's own. Stage 3 is optional.

It runs on your own machine and opens no WRDS connection. Its documentation is
[stage3/README_stage3.md](stage3/README_stage3.md) and [stage3/INDEX.md](stage3/INDEX.md).

### Stage 4: The TRACE-only bond factors

Stage 4 turns the Stage 2 panel into the TRACE-only bond factors published on
[openbondassetpricing.com](https://openbondassetpricing.com). The factors are 108 signals as
single sorts and within-firm sorts, for four return types (excess, and duration-adjusted against
three Treasury benchmarks) and three rating bands. `compare_published.py` then checks your
factors against the published files, cell by cell. Stage 4 is optional.

It runs on your own machine and opens no WRDS connection. Its documentation is
[stage4/README_stage4.md](stage4/README_stage4.md).

## Project status & timeline

- Stage 0 is available now as a public beta, ready for testing.
- Stage 1 is available now as a public beta, ready for testing.
- Stage 2 is released. It builds the monthly panel from your Stage 1 output, and published
  vintages are on [openbondassetpricing.com](https://openbondassetpricing.com).
- The Stage 3 code is available. It turns that monthly panel into portfolio sorts, uncertainty
  grids and 33 tables and 11 figures (29 tables are the paper's).
- The Stage 4 code is available. It builds the TRACE-only bond factors published on
  openbondassetpricing.com from that monthly panel, and checks them against the published files.

This project is under active development and any feedback is greatly appreciated.
Please reach out to `alexander.dickerson1@unsw.edu.au` if you would like to collaborate.

## Key features

### Stage 0: Robust error correction
- The decimal-shift corrector automatically detects and fixes multiplicative price errors (10x,
  0.1x, 100x, 0.01x)
- The bounce-back filter identifies and removes erroneous price spikes that revert quickly
- Algorithms designed by Dickerson, Robotti & Rossetti (2026) account for TRACE idiosyncrasies
- The full documentation is in
  [README_decimal_shift_corrector.md](stage0/README_decimal_shift_corrector.md) and
  [README_bounce_back_filter.md](stage0/README_bounce_back_filter.md)

### Stage 0: Comprehensive data cleaning
- Dick-Nielsen (2009, 2014) cancellation, correction, and reversal filters
- van Binsbergen, Nozawa and Schwert (2025) filters
- Agency trade de-duplication
- Pre-2012 and post-2012 cleaning rules
- Price range filters and volume screens
- Trading calendar and time-of-day filters

### Stage 0: Quality assurance & reporting
- Transaction-level audit logs for every filter stage
- CUSIP-level lists of corrected bonds
- LaTeX reports with detailed filtering statistics
- Optional time-series plots for visual inspection (can generate 500+ page reports)
- Row count reconciliation at each processing stage

### Stage 0: Daily aggregation metrics
- Price metrics (equal-weighted, volume-weighted, par-weighted, first, last, trade count)
- Volume metrics (par volume and dollar volume, in millions)
- Bid/ask metrics (value-weighted bid and ask prices)

### Stage 1: Bond analytics
- Bond characteristics from FISD (maturity, coupon, offering amount, issuer, security features)
- Computed bond analytics via QuantLib (duration, convexity, yields, credit spreads, accrued
  interest)
- Credit ratings from S&P and Moody's with numeric conversions
- Equity identifiers (PERMNO, PERMCO, GVKEY) from the bond-firm linker
- Ultra-distressed bond filters to flag potentially erroneous prices
- Fama-French industry classifications (12, 17 and 30 industry groups)
- A daily bond-level dataset of 44 columns (Stage 2 turns it into the 108-signal monthly panel)
- The ultra-distressed filter catches suspicious "rounded" price numbers at very low prices,
  often associated with issues trading under default. See
  [README_distressed_filter.md](stage1/README_distressed_filter.md)

## Quick start

### Prerequisites
- A WRDS subscription with access to TRACE, FISD, and ratings data
- Python 3.10 or higher on WRDS (the 2026-09-21 production run used Python 3.14.5 there), and
  3.11 to 3.13 on your own computer for Stages 2 to 4
- SSH access to WRDS Cloud (or a local Python environment)
- No password file on WRDS, because jobs on the WRDS Cloud connect without one

### Setup

1. Clone the repository.
```bash
# On WRDS Cloud
ssh <your_wrds_id>@wrds-cloud.wharton.upenn.edu
cd ~
git clone https://github.com/Alexander-M-Dickerson/trace-data-pipeline.git
cd trace-data-pipeline
```

Set your WRDS username and the author name in one of two ways.

Option A is an environment variable (recommended).
```bash
export WRDS_USERNAME="your_wrds_id"
echo 'export WRDS_USERNAME="your_wrds_id"' >> ~/.bashrc  # Make persistent
```

Option B is to edit `config.py`.
```bash
nano config.py
# Change: WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_id")
# Change: AUTHOR = "Your Name"  # Default is "Open Source Bond Asset Pricing"
```

Your password is never in code. On WRDS none is needed. On your own computer (stage 2's first
run), connect once by hand with `python -c "import wrds; wrds.Connection()"`. It asks for the
password and offers to save it where the `wrds` package looks (`~/.pgpass`, or
`%APPDATA%\postgresql\pgpass.conf` on Windows).

2. Install the dependencies. Stages 0 and 1 both need them.
```bash
python -m pip install --user -r requirements.txt
```

3. Check that the chain works before spending hours on it (recommended).
```bash
bash download_inputs.sh     # on the login node only, because compute nodes have no internet
qsub run_smoke_test.sh      # ~10 min; output lands in smoke_test.out
```
This runs the same Stage 0 and Stage 1 code as a full run, on a handful of CUSIP chunks, and
asserts the cross-stage invariants in `tests/smoke_assertions.py`. It writes to `smoke/` and
never touches production output.

4. Run Stages 0 and 1.
```bash
./run_pipeline.sh
```

#### What happens

1. Before Stage 0, `download_inputs.sh` fetches the Liu-Wu yields, the bond-firm linker
   and the FF industry files on the login node (compute nodes have no internet).
2. Stage 0 submits exactly the members in `TRACE_MEMBERS`. By default these are Enhanced and
   144A, which run at the same time. Standard is opt-in and, when requested, is held
   until the other two finish so it can use the whole WRDS connection budget.
3. The Stage 0 reports are generated automatically when every submitted TRACE job completes.
4. Stage 1 starts automatically as soon as the Stage 0 data is ready, and runs
   alongside the report job. It reads the member panels and the FISD file, and nothing the
   reports produce.

#### What is automated

- Data downloads (no manual wget required)
- Job dependencies built from the jobs actually submitted
- Cores and memory requested per member, and checked against the WRDS caps before
  submission (`m_mem_free` is charged per slot, and an over-request stays pending forever
  without a message)
- Centralized settings (`config.py` is shared, and `stage0/_trace_settings.py` is for Stage 0)

The detailed instructions are in [QUICKSTART.md](QUICKSTART.md) (the complete walkthrough), in
[stage0/README_stage0.md](stage0/README_stage0.md) or [stage0/quickstart.md](stage0/quickstart.md)
for Stage 0, and in [stage1/README_stage1.md](stage1/README_stage1.md) or
[stage1/QUICKSTART_stage1.md](stage1/QUICKSTART_stage1.md) for Stage 1.

## Documentation

Start with [INDEX.md](INDEX.md), which says which doc answers which question.
[CODE_MAP.md](CODE_MAP.md) says what each stage reads and writes and what every code file does.
[TAGS.md](TAGS.md) names the line where each panel column is computed, each filter applied and
each rule enforced. `python doctor.py` says what is ready and what to run next.

Claude Code, Codex, Cursor and Copilot read [AGENTS.md](AGENTS.md) by themselves. It tells an AI
assistant what to run, what to check, and the traps. Seven skills walk a task step by step. They
are `/onboard`, `/run-wrds`, `/build-panel`, `/reproduce-exhibits`, `/build-factors`,
`/explain-pipeline` and `/add-a-column` in Claude Code (`$onboard` and so on in Codex). Ask the
assistant to set the repository up, to build a stage, what a column is and where it is computed,
or to add a column to the panel.

### Docs for Stage 0 (TRACE data processing)

- [README](stage0/README_stage0.md), the complete guide to intraday to daily TRACE processing
- [QUICKSTART](stage0/quickstart.md), a short guide for getting started quickly
- [Data Dictionary](stage0/DATA_DICTIONARY.md), the 21-column bond-day panel and the 23-column
  FISD attribute file
- [Bounce-back filter](stage0/README_bounce_back_filter.md), how reversal artefacts are detected
  and removed
- [Decimal-shift corrector](stage0/README_decimal_shift_corrector.md), how mis-scaled prices are
  found and repaired
- [Configuration Guide](stage0/README_stage0.md#configuration-choices-you-can-edit), all
  configurable parameters
- [Troubleshooting](stage0/README_stage0.md#troubleshooting), common issues and solutions

### Docs for Stage 1 (bond analytics)

- [README](stage1/README_stage1.md), the complete guide to bond analytics and enrichment
- [QUICKSTART](stage1/QUICKSTART_stage1.md), a short guide for getting started quickly
- [Data Dictionary](stage1/DATA_DICTIONARY.md), all 44 columns, and what the public download
  leaves out
- [Distressed-bond filter](stage1/README_distressed_filter.md), the ultra-distressed screen and
  why it exists
- [Configuration Guide](stage1/README_stage1.md#configuration-choices-you-can-edit), all
  configurable parameters
- [Troubleshooting](stage1/README_stage1.md#troubleshooting), common issues and solutions

### Docs for Stage 2 (monthly panel)

- [Stage 2 README](stage2/README_stage2.md), what it builds and how to run it
- [Quick Start](stage2/QUICKSTART_stage2.md), the shortest path from Stage 1 output to a panel
- [Data Dictionary](stage2/DATA_DICTIONARY.md), all 145 columns, the factor models, and the
  `_mmn` sidecar
- [Defaulted-bond returns](stage2/README_Default.md), how returns are computed when a bond enters
  or trades in default
- [Value signals](stage2/README_Value.md), the value-signal family and its
  duration-times-spread variants

### Docs for Stage 3 (sorts and exhibits)

- [Stage 3 README](stage3/README_stage3.md), what it produces, the five inputs, and the
  conventions that decide what a number means
- [Stage 3 Quickstart](stage3/QUICKSTART_stage3.md), a minimal run, and what to do when a step
  complains
- [Stage 3 Data Dictionary](stage3/DATA_DICTIONARY.md), every artifact and column
- [Stage 3 Exhibit Index](stage3/INDEX.md), every table and figure the report prints,
  mapped to the file that produces it, its LaTeX label and the sample it states.
  It is generated, and `tools/build_index.py --check` fails when the exhibits or their files
  change

### Docs for Stage 4 (TRACE-only factors)

- [Stage 4 README](stage4/README_stage4.md), what it builds, how long it takes, and the check
  against the published files
- [Stage 4 Data Dictionary](stage4/DATA_DICTIONARY.md), every column of the factor files

## Downloading results to your local machine

The pipeline generates a large folder (~5.6 GB) with hundreds of files. Zip the folder first, then
download a single file, which is faster and more reliable.

### Quick overview

1. SSH into WRDS and zip to scratch space, which avoids the home directory quota.
   ```bash
   ssh {wrds_username}@wrds-cloud.wharton.upenn.edu
   cd ~
   zip -r "/scratch/$(basename "$(dirname "$HOME")")/trace-data-pipeline.zip" trace-data-pipeline/
   ```
   `$(basename "$(dirname "$HOME")")` is your institution code, the folder between `/home` and your username. Type the line as written, because nothing in it is a placeholder. In the `scp` line below, `{institution}` is a placeholder. Replace it, braces included, with that code (`echo $HOME` on WRDS shows it).

   Zip from `~` on a relative path. `zip` stores the path you hand it, so an absolute
   `~/trace-data-pipeline/` strips the leading `/` and makes the archive extract as
   `home/{institution}/{wrds_username}/trace-data-pipeline/...`, four directories deep.

2. Download the zip from your local machine.
   ```bash
   scp {wrds_username}@wrds-cloud.wharton.upenn.edu:/scratch/{institution}/trace-data-pipeline.zip "{local_destination}"
   ```

3. Extract it locally.
   - On Windows, right-click the zip file and choose Extract All
   - On a Mac, double-click the zip file
   - On Linux, run `unzip trace-data-pipeline.zip`

For detailed instructions, including Windows GUI options, see
[QUICKSTART.md](QUICKSTART.md#download-results-to-your-local-machine).

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `{wrds_username}` | Your WRDS username | `jsmith` |
| `{institution}` | Your institution's scratch folder | `wharton`, `chicago`, `nyu` |
| `{local_destination}` | Local path | `~/Downloads` or `C:\Users\YourName\Downloads` |

## Repository structure

```
trace-data-pipeline/
├── README.md  QUICKSTART.md  FAQ.md  CONTRIBUTING.md  CHANGELOG.md
├── INDEX.md                # where to look for each task
├── CODE_MAP.md             # what each stage reads and writes, and what every file does
├── TAGS.md                 # the line each panel column, filter and rule lives on
├── AGENTS.md  CLAUDE.md    # instructions for AI assistants; skills in .claude/ and .agents/
├── doctor.py               # what is ready, and what to run next
├── tools/                  # tags.py: checks the code tags and writes TAGS.md
├── docs/                   # the questions an assistant must answer, and what has been run
├── config.py               # settings shared by stages 0-2
├── run_pipeline.sh         # runs stages 0 and 1 on WRDS
├── download_inputs.sh  check_disk_space.sh  run_smoke_test.sh
├── requirements.txt        # stages 0 and 1, on WRDS
├── requirements-local.txt  # stages 2-4, on your computer
├── pybondlab_pin.py        # the PyBondLab release stages 2-4 run on
├── tests/                  # repo-wide tests, including the docs against the code
├── stage0/                 # WRDS: raw TRACE to daily panels
├── stage1/                 # WRDS: daily bond analytics
├── stage2/                 # your computer: the monthly panel
├── stage3/                 # your computer: the paper's tables and figures
└── stage4/                 # your computer: the TRACE-only bond factors
```

[CODE_MAP.md](CODE_MAP.md) lists every file, what it does, and what each stage reads and writes.

## Output data structure

### Stage 0 output: Daily TRACE panels

Stage 0 produces daily panels in dataset-specific subfolders, at these paths.
- `enhanced/trace_enhanced_YYYYMMDD.parquet`
- `standard/trace_standard_YYYYMMDD.parquet`
- `144a/trace_144a_YYYYMMDD.parquet`

The quality reports are in these folders.
- `data_reports/enhanced/` for the Enhanced TRACE reports
- `data_reports/standard/` for the Standard TRACE reports
- `data_reports/144a/` for the Rule 144A reports

Each panel has 21 columns, and every one is defined in
[stage0/DATA_DICTIONARY.md](stage0/DATA_DICTIONARY.md).

| Column | Description |
|--------|-------------|
| `cusip_id` | 9-character CUSIP identifier |
| `trd_exctn_dt` | Trade execution date |
| `prc_ew` | Equal-weighted price |
| `prc_vw` | Volume-weighted price (dollar) |
| `prc_vw_par` | Volume-weighted price (par) |
| `prc_first` | First trade price of day |
| `prc_last` | Last trade price of day |
| `prc_hi` | High price of the day |
| `prc_lo` | Low price of the day |
| `trade_count` | Number of trades |
| `time_ew` | Equal-weighted mean trade time (seconds after midnight) |
| `time_last` | Time of the last trade (seconds after midnight) |
| `qvolume` | Par volume (millions) |
| `dvolume` | Dollar volume (millions) |
| `prc_bid` | Dealer bid (value-weighted) |
| `bid_last` | Last dealer bid price of the day |
| `bid_time_ew` | Equal-weighted mean time of dealer bids |
| `bid_time_last` | Time of the last dealer bid |
| `prc_ask` | Dealer ask (value-weighted) |
| `bid_count` | Number of dealer buys (= customer sells) |
| `ask_count` | Number of dealer sells (= customer buys) |

Expect output of these sizes, which are from the 2026-09-21 run.
- Enhanced TRACE (2002-07 to the data frontier) has ~31 million rows, about 2.2 GB
- Standard TRACE (from 2024-10) is opt-in and not part of the default run
- Rule 144A (first trade 2003-10) has ~3.8 million rows, about 250 MB

Stage 0 also writes these outputs.
- Audit files documenting filter effects (in dataset subfolders)
- CUSIP lists of bonds with corrections (in dataset subfolders)
- Data quality reports with LaTeX + figures (in `data_reports/` subfolder)

### Stage 1 output

Stage 1 writes `stage1/data/stage1_YYYYMMDD.parquet`. It is panel data with one row per
(cusip_id, trd_exctn_dt) combination, and it is about 2.7 GB for the full sample with Enhanced
and 144A (2.68 GB on the 2026-09-21 run). The data is available for download in zipped parquet
format on [Open Source Bond Asset Pricing](https://openbondassetpricing.com/data).

The file has 44 columns.

#### Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `cusip_id` | category | 9-character CUSIP identifier |
| `issuer_cusip`* | category | 6-character issuer CUSIP |
| `permno` | Int32 | CRSP PERMNO equity identifier |
| `permco`† | Int32 | CRSP PERMCO company identifier |
| `gvkey`† | Int32 | Compustat GVKEY identifier |
| `trd_exctn_dt` | datetime | Trade execution date |

#### Computed bond analytics (QuantLib)

All prices are in percentage of par (e.g., 99 = 99% of par = $990 for a $1,000 principal bond).

| Column | Type | Description |
|--------|------|-------------|
| `pr` | float32 | Volume-weighted clean price (% of par) |
| `prfull` | float32 | Dirty price = pr + acclast (% of par) |
| `acclast` | float32 | Accrued interest, the pure time-accrued interest component |
| `accpmt` | float32 | Accumulated coupon payments since issue |
| `accall` | float32 | Accumulated payments, which include cash flows + accrued interest. Used for return calculations |
| `ytm` | float64 | Yield to maturity (annualized) |
| `mod_dur` | float32 | Modified duration (years) |
| `mac_dur` | float32 | Macaulay duration (years) |
| `convexity` | float32 | Bond convexity |
| `bond_maturity` | float32 | Time to maturity (years) |
| `credit_spread` | float64 | Credit spread over the Treasury yield at the bond's maturity (Liu-Wu curve) |

#### TRACE pricing (from Stage 0)

All prices are in percentage of par.

| Column | Type | Description |
|--------|------|-------------|
| `prc_ew` | float32 | Equal-weighted price |
| `prc_vw_par` | float32 | Par volume-weighted price |
| `prc_first` | float32 | First trade price of day |
| `prc_last` | float32 | Last trade price of day |
| `prc_hi` | float32 | High price of the day |
| `prc_lo` | float32 | Low price of the day |
| `trade_count` | Int16 | Number of trades |
| `time_ew`‡ | float32 | Average trade time (seconds after midnight) |
| `time_last`‡ | Int32 | Last trade time (seconds after midnight) |
| `qvolume` | float32 | Par volume (millions USD) |
| `dvolume` | float32 | Dollar volume (millions USD) |

#### Dealer bid/ask metrics

| Column | Type | Description |
|--------|------|-------------|
| `prc_bid` | float32 | Dealer bid price, value-weighted (% of par) |
| `bid_last` | float32 | Last dealer bid price of day (% of par) |
| `bid_time_ew`‡ | float32 | Average dealer bid time (seconds after midnight) |
| `bid_time_last`‡ | Int32 | Last dealer bid time (seconds after midnight) |
| `prc_ask` | float32 | Dealer ask price, value-weighted (% of par) |
| `bid_count`‡ | Int16 | Number of dealer buys (can be NaN) |
| `ask_count`‡ | Int16 | Number of dealer sells (can be NaN) |

#### Database source

| Column | Type | Description |
|--------|------|-------------|
| `db_type` | Int8 | Source database (1=Enhanced, 2=Standard, 3=144A) |

#### Bond characteristics (from FISD)

| Column | Type | Description |
|--------|------|-------------|
| `coupon`* | float32 | Annual coupon rate (%) |
| `principal_amt`* | Int16 | Principal amount per bond (typically $1,000) |
| `bond_age` | float32 | Bond age since issuance (years) |
| `bond_amt_outstanding` | Int64 | Amount outstanding, in thousands of dollars (as FISD reports it) |
| `callable`* | Int8 | Callable flag (1=callable, 0=not callable) |

#### Industry classifications

| Column | Type | Description |
|--------|------|-------------|
| `ff12num` | int8 | Fama-French 12 industry classification |
| `ff17num` | int8 | Fama-French 17 industry classification |
| `ff30num` | int8 | Fama-French 30 industry classification |

#### Credit ratings

| Column | Type | Description |
|--------|------|-------------|
| `sp_rating`† | Int8 | S&P credit rating (1-22, where 22=default) |
| `sp_naic`* | Int8 | S&P NAIC category (1-6) |
| `mdy_rating`† | Int8 | Moody's credit rating (1-21, where 21=default) |
| `spc_rating`† | Int8 | S&P composite rating (1-22), with missing values filled with mdy_rating (scaled to 22 for default) |
| `mdc_rating`† | Int8 | Moody's composite rating (1-22). It is mdy_rating with its default code 21 moved to 22, and missing values are filled with sp_rating |
| `comp_rating`* | float64 | Average of spc_rating and mdc_rating |

#### Notes

- \*Columns marked with an asterisk are not in the main Stage 1 file. Each one is available
  elsewhere.
  - For `coupon` and `principal_amt`, merge
    `stage0/enhanced/trace_enhanced_fisd_YYYYMMDD.parquet` (keyed by `complete_cusip`)
  - `callable` is in `stage1/data/call_dummy_YYYYMMDD.parquet`, keyed by `issue_id`. Map it to
    CUSIPs through the FISD file's `issue_id`
  - `sp_naic` is `sp_naic_numeric` in `stage1/data/sp_ratings_YYYYMMDD.parquet`
  - `issuer_cusip` is `cusip_id[:6]`, so no merge is needed
  - `comp_rating` is not written anywhere. It is computed and then dropped before
    export. Reconstruct it as the mean of `spc_rating` and `mdc_rating`.
- †Columns marked with a dagger are excluded from the public download because of proprietary
  data restrictions
- ‡Columns marked with a double dagger are excluded from the public download to reduce file size
- All `prc_*` prices are in percentage of par (99 = 99% of $1,000 = $990)

### Stage 2 output

Stage 2 writes a monthly panel of 145 columns that holds 108 corporate bond signals for asset
pricing research, plus the unadjusted `_mmn` twins and the factor time series. Every column is
defined in [stage2/DATA_DICTIONARY.md](stage2/DATA_DICTIONARY.md).

## Performance

### Expected runtime (WRDS Cloud)

These are the times with `./run_pipeline.sh`, which runs Stages 0 and 1 from the project
root.
- Stage 0 data processing (Enhanced and 144A in parallel)
  - Enhanced TRACE takes ~2-2.6 hours. It was ~4 hours before v2.2.0. It now pulls 5 CUSIP
    chunks at a time over separate WRDS connections, about twice as fast (2.0 h on
    2026-09-09, 2.6 h on 2026-09-21).
  - 144A TRACE takes ~40 minutes (38 on 2026-09-21).
  - Standard TRACE takes ~30-60 minutes, and has been opt-in since v2.2.0 (`TRACE_MEMBERS`).
    When requested it is scheduled after the other two, not beside them.
- Stage 0 report generation takes ~20 minutes since v2.2.2 (it was ~50 before, and it took 22
  on 2026-09-21). It waits for every member submitted, and runs alongside Stage 1, not before it.
- Stage 1 bond analytics takes ~2.4-2.7 hours (2.7 on 2026-09-21). It waits for the Stage 0
  data jobs.

End to end, a run takes about 5 hours. The 2026-09-21 run, which the 2026 vintage was built
from, took 5.3 h wall clock (04:17 to 09:34) and wrote a Stage 1 panel of 31,344,732 rows over
70,684 CUSIPs, 2002-07-01 to 2025-11-28. Its `DATE_CUT_OFF = "auto:complete"` ends the sample at
the last month every source covers through its final trading session. (The earlier 2026-09-10
run predates that rule. Cut the same way afterwards, it gave the same 31,344,732 rows.)

### How it works

The script uses SGE's `-hold_jid` to create automatic dependency chains.
1. Stage 0 submits the members in `TRACE_MEMBERS`. By default these are Enhanced and 144A,
   which run at the same time. Standard, if requested, is held behind them.
2. The Stage 0 report job waits until every submitted extraction job completes.
3. Stage 1 also waits on the extraction jobs, then runs in parallel with the reports.
   It reads only the member panels and the FISD file, and nothing the reports produce.
4. All jobs are submitted with a single command from the project root.

### Resource usage

`m_mem_free` is charged per slot, so the total is `slots x mem` and must stay within the WRDS
caps of 8 cores and 48 GB per job. The stage 0 member jobs are sized by `run_pipeline.sh` from
`stage0/_trace_settings.py`. The report job and Stage 1 have their requests in their own job
scripts.

| job | request | total |
|---|---|---|
| Enhanced | `-pe onenode 5 -l m_mem_free=8G` | 40 GB |
| 144A | `-pe onenode 1 -l m_mem_free=16G` | 16 GB |
| Standard (opt-in) | `-pe onenode 6 -l m_mem_free=8G` | 48 GB |
| Data reports | `-pe onenode 5 -l m_mem_free=8G` | 40 GB |
| Stage 1 | `-pe onenode 4 -l m_mem_free=10G` | 40 GB |

- A full run needs about 5.6 GB of disk (on 2026-09-21 the Enhanced panel was 2.2 GB, 144A
  0.25 GB, the Stage 1 panel 2.7 GB, and the data reports and smaller files 0.4 GB).
- Tune stage 0 with `CONCURRENCY`, `MEM_PER_SLOT_GB` and `TARGET_ROWS_PER_CHUNK` in
  `stage0/_trace_settings.py`, and not by editing its job scripts, because the command-line
  request overrides their directives. The report job's and Stage 1's requests are in
  `stage0/run_build_data_reports.sh` and `stage1/run_stage1.sh`.

## Citation

If you use this pipeline in your research, please cite the papers below.

```bibtex
@unpublished{dickerson2026replication,
  author = {Dickerson, Alexander and Robotti, Cesare and Rossetti, Giulio},
  title = {The Corporate Bond Factor Replication Crisis},
  year = {2026},
  note = {Working Paper. Earlier versions circulated as "Common pitfalls in the evaluation of corporate bond strategies"}
}

@unpublished{dickerson2025constructing,
  author = {Dickerson, Alexander and Rossetti, Giulio},
  title = {Constructing TRACE Corporate Bond Datasets},
  year = {2025},
  note = {Working Paper}
}
```

## References

This pipeline builds on methods from these papers.

- Dick-Nielsen, J. (2009). Liquidity biases in TRACE. *The Journal of Fixed Income*, 19(2), 43-55.
- Dick-Nielsen, J. (2014). How to clean enhanced TRACE data. Working Paper.
- van Binsbergen, J. H., Nozawa, Y., & Schwert, M. (2025). Duration-based valuation of corporate bonds. *The Review of Financial Studies*, 38(1), 158-191.

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
Contributions would help most in these areas.
- Testing Stage 0 on different WRDS environments
- Additional filter implementations
- Performance optimizations
- Extended documentation
- Bug fixes and error reporting

## Support

- Email alexander.dickerson1@unsw.edu.au
- Report issues on [GitHub Issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)
- We welcome collaborators, so please reach out

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Last updated in September 2026. The version is 4.2.1, and [CHANGELOG.md](CHANGELOG.md) says what
each release changed.

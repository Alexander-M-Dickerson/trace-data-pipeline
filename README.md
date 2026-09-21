# TRACE Data Pipeline

A comprehensive pipeline for processing Enhanced, Standard and 144A TRACE (Trade Reporting and Compliance Engine) corporate bond transaction data. 
It is a part of the [Open Bond Asset Pricing project](https://openbondassetpricing.com/).
This pipeline implements cleaning procedures and error-correction algorithms to produce *high-quality, reproducible* daily and monthly corporate bond panels from raw TRACE transaction data.
The companion repository is [PyBondLab](https://github.com/GiulioRossetti94/PyBondLab/tree/main/examples) which can be used to form corporate bond asset pricing factors.

[![Website](https://img.shields.io/badge/Website-Visit-blue?logo=google-chrome&logoColor=white)](https://openbondassetpricing.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Stage 0](https://img.shields.io/badge/Stage%200-Public%20Beta-green)](stage0/)
[![Stage 1](https://img.shields.io/badge/Stage%201-Public%20Beta-green)](stage1/)
[![Stage 2](https://img.shields.io/badge/Stage%202-Public%20Beta-green)](stage2/)
[![Stage 3](https://img.shields.io/badge/Stage%203-Public%20Beta-green)](stage3/)

[📄 Link to paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4575879)
---

## Which machine am I on?

**This pipeline runs in two places, and the hand-off between them is a file you copy yourself.**
Getting this straight first will save you an afternoon.

| | Where | What happens | How long |
|---|---|---|---|
| **1** | **WRDS Cloud** | Stages 0 and 1 build the daily bond panel from the raw TRACE tape | ~5 hours, mostly waiting |
| **2** | **In between** | You zip the output on WRDS and copy it down to your own computer | ~15 min for ~5 GB |
| **3** | **Your own computer** | Stage 2 turns that daily panel into the monthly asset-pricing panel | ~8-13 minutes |
| **4** | **Your own computer** | Stage 3 turns the monthly panel into sorted portfolios and the paper's exhibits | ~15 minutes |

**Why the split?** Stages 0 and 1 read the raw TRACE transaction tape, which is a WRDS
database — so they have to run where the data is, submitted to the WRDS job grid. Stage 2
reads nothing but the file Stage 1 produced, so there is no reason to queue for it; it runs
faster on your own machine and needs no WRDS connection at all.

### Step by step

**On WRDS** — you will be SSH'd into `wrds-cloud.wharton.upenn.edu`:

1. Clone this repository into your WRDS home directory.
2. Install the Python packages (`pip install --user -r requirements.txt`).
3. Run `bash download_inputs.sh` **on the login node** — compute nodes have no internet.
4. Submit the pipeline: `./run_pipeline.sh`. Stage 0 runs in parallel per TRACE member, then
   Stage 1 starts automatically when they finish.
5. Wait. Check on it with `qstat`.

**Moving the results** — you end up with roughly 5 GB across hundreds of files:

6. On WRDS, zip it: `cd ~ && zip -r /scratch/{institution}/trace-data-pipeline.zip trace-data-pipeline/`
7. From your own computer, copy it down with `scp`, then unzip.

> ❗Zip from `~` using a **relative** path, exactly as written. `zip -r out.zip ~/trace-data-pipeline/`
> stores the absolute path and you get an archive nested four directories deep.

**On your own computer:**

8. Install the same requirements (`pip install -r requirements.txt`) in a Python 3.10+
   environment. You do **not** need a WRDS connection from here.
9. `cd stage2 && python _run_stage2.py`
10. You now have `stage2/output/panel/main_panel_<mode>.parquet` — 145 columns per bond-month.

**Still on your own computer, if you want the research output too:**

11. `cd stage3 && python tools/check_inputs.py` — confirms Stage 3 can see what Stage 2 made.
12. `bash run_stage3.sh` — portfolio sorts, the uncertainty grids, and 33 tables and
    11 figures into `stage3/reports/`.

Stages 0-2 build the DATA. Stage 3 is what the data was built for: it reproduces every
exhibit of *The Corporate Bond Factor Replication Crisis* from the panel you just made.
It is optional — the panel is useful on its own.

Full detail: [QUICKSTART.md](QUICKSTART.md) for stages 0-1,
[stage2/QUICKSTART_stage2.md](stage2/QUICKSTART_stage2.md) for stage 2,
[stage3/QUICKSTART_stage3.md](stage3/QUICKSTART_stage3.md) for stage 3.

> ❗**You cannot skip stages 0 and 1 by downloading the published Stage 1 file.** The public
> download has its rating columns removed, because agency ratings are licensed. Stage 2 keeps
> only bond-months that carry a rating, so it would silently produce an empty panel. Stage 2
> detects this and refuses to start. Run stages 0 and 1 yourself — that is what the WRDS
> subscription is for.

---

## Overview

This is a **four-stage pipeline** for building *clean, reliable and reproducible* TRACE corporate bond datasets. Stages 0 and 1 run on WRDS, Stages 2 and 3 on your own computer.

### Stage 0: Intraday to Daily Processing  **PUBLIC BETA**
Processes raw intraday TRACE transaction data to clean daily panels. Handles three types of TRACE data:
- **Enhanced TRACE**
- **Standard TRACE**
- **Rule 144A bonds**

**Automated workflow:** Run `./run_pipeline.sh` from the project ROOT to execute the complete multi-stage pipeline with automatic job dependencies. Stage 0 jobs run in parallel, then automatically chain to Stage 1 processing when complete.

**Status:** Public beta - fully functional and ready for testing
**Execution:** **On the WRDS Cloud grid.** Stage 0 reads the raw TRACE tape, which lives on WRDS.
**Documentation:** See [stage0/README_stage0.md](stage0/README_stage0.md), [stage0/quickstart.md](stage0/quickstart.md) and [stage0/DATA_DICTIONARY.md](stage0/DATA_DICTIONARY.md)

### Stage 1: Daily Bond Analytics  **PUBLIC BETA**
Enriches Stage 0 daily panels with comprehensive bond analytics and characteristics:
- **Bond analytics** via QuantLib (duration, convexity, YTM, credit spreads)
- **Credit ratings** from S&P and Moody's with numeric conversions
- **Equity identifiers** equity linkers
- **FISD bond characteristics** (coupon, maturity, issuer, amount outstanding, etc.)
- **Fama-French industry classifications** (12, 17 and 30 industries)
- **Ultra-distressed filters** to flag potentially erroneous prices

**Status:** Public beta - fully functional and ready for testing
**Execution:** **On the WRDS Cloud grid**, chained automatically after Stage 0.
**Documentation:** See [stage1/README_stage1.md](stage1/README_stage1.md), [stage1/QUICKSTART_stage1.md](stage1/QUICKSTART_stage1.md) and [stage1/DATA_DICTIONARY.md](stage1/DATA_DICTIONARY.md)

### Stage 2: Monthly Panel with Factor Signals
Produces a clean, error-corrected monthly panel with dozens of corporate bond signals for asset pricing research:
- 108 bond characteristic signals across 145 panel columns
- Credit risk factors
- Liquidity measures
- Momentum and reversal signals
- Carry and value signals
- Ready-to-use for monthly portfolio construction -- see [PyBondLab](https://github.com/GiulioRossetti94/PyBondLab/tree/main/examples)

**Status:** Complete -- builds, tested, gated, and released
**Release:** Published vintages at [openbondassetpricing.com](https://openbondassetpricing.com)
**Execution:** Your own machine, NOT the WRDS grid (a WRDS subscription is still required)

### Stage 3: Sorts and the Paper's Exhibits
Turns the Stage 2 panel into portfolio sorts, two uncertainty grids, and the 33 tables and
11 figures of *The Corporate Bond Factor Replication Crisis*, compiled into one PDF. Optional.

**Execution:** Your own machine, and it opens no WRDS connection
**Documentation:** See [stage3/README_stage3.md](stage3/README_stage3.md) and [stage3/INDEX.md](stage3/INDEX.md)

---

## Project Status & Timeline

- **Stage 0**: ✅ **Now available** - Public beta, ready for testing
- **Stage 1**: ✅ **Now available** - Public beta, ready for testing
- **Stage 2**: ✅ **Released** - Builds the monthly panel from your Stage 1 output; published vintages are on [openbondassetpricing.com](https://openbondassetpricing.com)
- **Stage 3**: 🔨 **Code available** - Turns that monthly panel into portfolio sorts, uncertainty grids and the paper's 33 tables and 11 figures

**This project is under active development and any feedback is greatly appreciated.**
Please reach out to `alexander.dickerson1@unsw.edu.au` if you would like to collaborate.

---

## Key Features

### Stage 0: Robust Error Correction
- **Decimal-shift corrector**: Automatically detects and fixes multiplicative price errors (10x, 0.1x, 100x, 0.01x)
- **Bounce-back filter**: Identifies and removes erroneous price spikes that revert quickly
- Algorithms designed by Dickerson, Robotti & Rossetti (2026) account for TRACE idiosyncrasies
- **Full documentation**: See [README_decimal_shift_corrector.md](stage0/README_decimal_shift_corrector.md) and [README_bounce_back_filter.md](stage0/README_bounce_back_filter.md)


### Stage 0: Comprehensive Data Cleaning
- Dick-Nielsen (2009, 2014) cancellation, correction, and reversal filters
- van Binsbergen, Nozawa and Schwert (2025) filters
- Agency trade de-duplication
- Pre-2012 and post-2012 cleaning rules
- Price range filters and volume screens
- Trading calendar and time-of-day filters

### Stage 0: Quality Assurance & Reporting
- Transaction-level audit logs for every filter stage
- CUSIP-level lists of corrected bonds
- LaTeX reports with detailed filtering statistics
- Optional time-series plots for visual inspection (can generate 500+ page reports)
- Row count reconciliation at each processing stage

### Stage 0: Daily Aggregation Metrics
- **Price metrics**: Equal-weighted, volume-weighted, par-weighted, first, last, trade count
- **Volume metrics**: Par volume and dollar volume (in millions)
- **Bid/Ask metrics**: Value-weighted bid and ask prices

### Stage 1: Bond Analytics
- **Bond characteristics** from FISD (maturity, coupon, offering amount, issuer, security features)
- **Computed bond analytics** via QuantLib (duration, convexity, yields, credit spreads, accrued interest)
- **Credit ratings** from S&P and Moody's with numeric conversions
- **External identifiers** 
- **Ultra-distressed bond filters** to flag potentially erroneous prices
- **Fama-French industry classifications** (12, 17 and 30 industry groups)
- Produces a daily bond-level dataset of 44 columns (Stage 2 turns it into the 108-signal monthly panel)
- Ultra-distressed filter catches suspicious "rounded" price numbers at very low prices often associated with issues trading under default. See [README_distressed_filter.md](stage1/README_distressed_filter.md)

---

## Quick Start

### Prerequisites
- WRDS subscription with access to TRACE, FISD, and ratings data
- Python 3.10 or higher (the 2026-09-10 production run used Python 3.14.5 on the WRDS Cloud)
- SSH access to WRDS Cloud (or local Python environment)
- `.pgpass` configured for passwordless WRDS authentication

### Setup

1. **Clone the repository:**
```bash
# On WRDS Cloud
ssh <your_wrds_id>@wrds-cloud.wharton.upenn.edu
cd ~
git clone https://github.com/Alexander-M-Dickerson/trace-data-pipeline.git
cd trace-data-pipeline
```

**Configure WRDS username and author** (choose one method):

**Option A — Environment variable (recommended):**
```bash
export WRDS_USERNAME="your_wrds_id"
echo 'export WRDS_USERNAME="your_wrds_id"' >> ~/.bashrc  # Make persistent
```

**Option B — Edit `config.py`:**
```bash
nano config.py
# Change: WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_id")
# Change: AUTHOR = "Your Name"  # Default is "Open Source Bond Asset Pricing"
```

*Note: Password comes from `.pgpass`, not code.*

2. **Install Stage 1 dependencies:**
```bash
# Stage 0 uses system Python (no installation needed)
# Stage 1 requires additional packages
python -m pip install --user -r requirements.txt
```

3. **Check the chain works, before spending hours on it (recommended):**
```bash
bash download_inputs.sh     # LOGIN NODE ONLY -- compute nodes have no internet
qsub run_smoke_test.sh      # ~10 min; output lands in smoke_test.out
```
This runs the real Stage 0 → Stage 1 code on a handful of CUSIP chunks and asserts
28 cross-stage invariants. It writes to `smoke/` and never touches production output.

4. **Run the complete pipeline:**
```bash
./run_pipeline.sh
```

**What happens:**
1. **Pre-stage**: `download_inputs.sh` fetches the Liu-Wu yields, the bond-firm linker
   and the FF industry files on the login node (compute nodes have no internet).
2. **Stage 0**: submits exactly the members in `TRACE_MEMBERS` — by default Enhanced and
   144A, which run at the same time. Standard is opt-in and, when requested, is held
   until the other two finish so it can use the whole WRDS connection budget.
3. **Stage 0 Reports**: auto-generates when every submitted TRACE job completes.
4. **Stage 1**: auto-starts as soon as the Stage 0 *data* is ready, and runs
   ALONGSIDE the report job rather than behind it — it reads the member panels and
   the FISD file, nothing the reports produce.

**Automated features:**
- ✅ Data downloads (no manual wget required)
- ✅ Job dependencies built from the jobs actually submitted
- ✅ Cores and memory requested per member, validated against the WRDS caps before
  submission (`m_mem_free` is charged per slot; an over-request pends forever in silence)
- ✅ Centralized settings (`config.py` shared; `stage0/_trace_settings.py` for Stage 0)

**For detailed instructions:**
- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) for complete walkthrough
- **Stage 0**: See [stage0/README_stage0.md](stage0/README_stage0.md) or [stage0/quickstart.md](stage0/quickstart.md)
- **Stage 1**: See [stage1/README_stage1.md](stage1/README_stage1.md) or [stage1/QUICKSTART_stage1.md](stage1/QUICKSTART_stage1.md)

---

## Documentation

**Stage 0 - TRACE Data Processing:**
- **[README](stage0/README_stage0.md)**: Complete guide for intraday to daily TRACE processing
- **[QUICKSTART](stage0/quickstart.md)**: Fast-track guide to get started quickly
- **[Data Dictionary](stage0/DATA_DICTIONARY.md)**: The 21-column bond-day panel and the 23-column FISD attribute file
- **[Bounce-back filter](stage0/README_bounce_back_filter.md)**: How reversal artefacts are detected and removed
- **[Decimal-shift corrector](stage0/README_decimal_shift_corrector.md)**: How mis-scaled prices are found and repaired
- **[Configuration Guide](stage0/README_stage0.md#configuration-choices-you-can-edit)**: All configurable parameters
- **[Troubleshooting](stage0/README_stage0.md#troubleshooting)**: Common issues and solutions

**Stage 1 - Bond Analytics:**
- **[README](stage1/README_stage1.md)**: Complete guide for bond analytics and enrichment
- **[QUICKSTART](stage1/QUICKSTART_stage1.md)**: Fast-track guide to get started quickly
- **[Data Dictionary](stage1/DATA_DICTIONARY.md)**: All 44 columns, and what the public download leaves out
- **[Distressed-bond filter](stage1/README_distressed_filter.md)**: The ultra-distressed screen and why it exists
- **[Configuration Guide](stage1/README_stage1.md#configuration-choices-you-can-edit)**: All configurable parameters
- **[Troubleshooting](stage1/README_stage1.md#troubleshooting)**: Common issues and solutions

**Stage 2 - Monthly Panel:**
- **[Stage 2 README](stage2/README_stage2.md)**: What it builds and how to run it
- **[Quick Start](stage2/QUICKSTART_stage2.md)**: The shortest path from Stage 1 output to a panel
- **[Data Dictionary](stage2/DATA_DICTIONARY.md)**: All 145 columns, the factor models, and the `_mmn` sidecar
- **[Defaulted-bond returns](stage2/README_Default.md)**: How returns are computed when a bond enters or trades in default
- **[Value signals](stage2/README_Value.md)**: The value-signal family and its duration-times-spread variants

**Stage 3 - Sorts and Exhibits:**
- **[Stage 3 README](stage3/README_stage3.md)**: What it produces, the five inputs, and the conventions that decide what a number means
- **[Stage 3 Quickstart](stage3/QUICKSTART_stage3.md)**: A minimal run, and what to do when a step complains
- **[Stage 3 Data Dictionary](stage3/DATA_DICTIONARY.md)**: Every artifact and column
- **[Stage 3 Exhibit Index](stage3/INDEX.md)**: Every table and figure the report prints,
  mapped to the file that produces it, its LaTeX label and the sample it states.
  Generated, and gated so it cannot go stale

---

## Downloading Results to Your Local Machine

The pipeline generates a large folder (~5 GB) with hundreds of files. **Zip the folder first**, then download a single file for reliability and speed.

### Quick Overview

1. **SSH into WRDS** and zip to scratch space (avoids home directory quota):
   ```bash
   ssh {wrds_username}@wrds-cloud.wharton.upenn.edu
   cd ~
   zip -r /scratch/{institution}/trace-data-pipeline.zip trace-data-pipeline/
   ```
   ❗Zip from `~` on a **relative** path. `zip` stores the path you hand it, so an absolute
   `~/trace-data-pipeline/` strips the leading `/` and makes the archive extract as
   `home/{institution}/{wrds_username}/trace-data-pipeline/...`, four directories deep.

2. **Download the zip** (from your LOCAL machine):
   ```bash
   scp {wrds_username}@wrds-cloud.wharton.upenn.edu:/scratch/{institution}/trace-data-pipeline.zip "{local_destination}"
   ```

3. **Extract locally**:
   - **Windows**: Right-click → Extract All
   - **Mac**: Double-click the zip file
   - **Linux**: `unzip trace-data-pipeline.zip`

**For detailed instructions** (including Windows GUI options): See [QUICKSTART.md](QUICKSTART.md#download-results-to-your-local-machine)

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `{wrds_username}` | Your WRDS username | `jsmith` |
| `{institution}` | Your institution's scratch folder | `wharton`, `chicago`, `nyu` |
| `{local_destination}` | Local path | `~/Downloads` or `C:\Users\YourName\Downloads` |

---

## Repository Structure

```
trace-data-pipeline/
├── LICENSE                           # MIT License
├── README.md                         # This file
├── QUICKSTART.md                     # Fast-track guide (all stages)
├── FAQ.md                            # Common questions
├── CONTRIBUTING.md                   # Contribution guidelines
├── CHANGELOG.md                      # Version history
├── requirements.txt                  # Python dependencies (all stages)
├── config.py                         # Shared settings (TRACE_MEMBERS, username, ...)
├── run_pipeline.sh                   # ✨ One-push button orchestrator (ROOT)
├── check_disk_space.sh               # Is there room for a run? Measures the disk, called by run_pipeline.sh
├── download_inputs.sh                # Fetches stage 1's external inputs (LOGIN NODE)
├── run_smoke_test.sh                 # Whole-chain validation in minutes
├── .gitignore
│
├── tests/                            # Run before committing; only the probe needs WRDS
│   ├── smoke_assertions.py           # The 28 cross-stage invariants
│   ├── test_chunk_plan.py            # Chunk-partition properties
│   ├── test_chunk_scheduler.py       # Ordering + failure handling
│   ├── test_merge_keys.py            # Lookups must be one row per key
│   ├── test_linker_window.py         # Stage 1 joins the linker's identity window, as Stage 2 does
│   ├── test_cut_off_basis.py         # The auto:complete sample-end rule
│   ├── test_public_boundary.py       # No private path or name in committed files
│   ├── test_docs.py                  # Docs vs the code they describe
│   └── probe_wrds_connections.py     # Measures your account's connection ceiling
│
├── stage0/                           # ✅ PUBLIC BETA - Intraday to daily processing
│   ├── README_stage0.md              # Detailed documentation
│   ├── quickstart.md                 # Fast-track guide
│   ├── README_bounce_back_filter.md
│   ├── README_decimal_shift_corrector.md
│   ├── _trace_settings.py            # Configuration (CONCURRENCY, filters, FISD, ...)
│   ├── create_daily_enhanced_trace.py
│   ├── create_daily_standard_trace.py
│   ├── _chunk_runner.py              # Chunk planning + the concurrent scheduler
│   ├── _wrds_pool.py                 # One WRDS connection per worker process
│   ├── _run_enhanced_trace.py        # Runner scripts
│   ├── _run_standard_trace.py
│   ├── _run_144a_trace.py
│   ├── _build_error_files.py         # Report generation
│   ├── _error_plot_helpers.py        # Plotting utilities
│   ├── run_enhanced_trace.sh         # Individual job scripts
│   ├── run_standard_trace.sh
│   ├── run_144a_trace.sh
│   ├── run_build_data_reports.sh
│   │
│   ├── enhanced/                     # Enhanced TRACE output (auto-created)
│   ├── standard/                     # Standard TRACE output (auto-created, opt-in)
│   ├── 144a/                         # Rule 144A output (auto-created)
│   │
│   └── data_reports/                 # Quality reports (auto-created)
│       ├── enhanced/
│       ├── standard/
│       └── 144a/
│
├── stage1/                           # ✅ PUBLIC BETA - Daily bond analytics
│   ├── README_stage1.md              # Detailed documentation
│   ├── QUICKSTART_stage1.md          # Fast-track guide
│   ├── DATA_DICTIONARY.md            # Every output column
│   ├── README_distressed_filter.md
│   ├── _stage1_settings.py           # Configuration file
│   ├── create_daily_stage1.py        # Driver
│   ├── stage1_pipeline.py            # Main processing module
│   ├── _linker_join.py               # The bond-firm linker join (identity window)
│   ├── helper_functions.py           # Utility functions
│   ├── _distressed_plot_helpers.py
│   ├── _run_stage1.py                # Runner script
│   ├── run_stage1.sh                 # Job submission script
│   │
│   ├── data/                         # Stage 1 output + downloaded inputs
│   │   ├── stage1_YYYYMMDD.parquet   # Enriched dataset
│   │   ├── liu_wu_yields.xlsx        # Downloaded treasury yields
│   │   ├── bond_firm_linker_2026/    # Downloaded bond->firm linker
│   │   ├── Siccodes12.txt            # FF12 industry file
│   │   ├── Siccodes17.txt            # FF17 industry file
│   │   ├── Siccodes30.txt            # FF30 industry file
│   │   └── data_reports/             # Ultra-distressed filter report
│   │
│   ├── data_reports/                 # Stage 1 data-quality report (LaTeX/PDF)
│   └── logs/                         # Execution logs (auto-created)
│
├── smoke/                            # Scratch root for run_smoke_test.sh (auto-created)
│
├── stage2/                           # Monthly asset-pricing panel (runs on YOUR machine)
│   ├── _run_stage2.py                # Entry point
│   ├── _stage2_settings.py           # Settings + fail-loud input contract
│   ├── _build_data_report.py         # LaTeX/PDF data report
│   ├── make_release.py               # Packages a vintage for publication
│   ├── make_excess_blocks.py         # Optional: betas and momentum on another Treasury benchmark
│   ├── lib/                          # Engine: returns, illiquidity, betas, value, ...
│   │   └── contract.py               # The panel's frozen 145-column contract
│   ├── steps/                        # step1..step7 + the factor build
│   ├── reference/                    # The original BBW factors, shipped in the BBW bundle
│   ├── tests/                        # Contract, parity and boundary gates
│   ├── DATA_DICTIONARY.md            # Every column, every factor model
│   ├── data/  output/  data_reports/ # Gitignored build artifacts
│   └── logs/
│
└── stage3/                           # Sorts and the paper's exhibits (runs on YOUR machine)
    ├── _run_stage3.py                # Entry point: 41 steps, producers, exhibits, then the PDF
    ├── run_stage3.sh                 # Input contract, then the above
    ├── _stage3_settings.py           # Every path and constant
    ├── paths.py                      # Paths derived from the settings
    ├── pblenv.py                     # Which PyBondLab, asserted and fingerprinted
    ├── drrlib.py                     # Newey-West, CAPM_B alpha, paired difference, manifests
    ├── fastrun.py                    # Process-parallel fan-out for the grids
    ├── make_report.py                # Assembles every exhibit into one compiled PDF
    ├── captions.py  bench.py  latex_format.py  helper_functions.py
    ├── s0_data/                      # The data appendix
    ├── s1_lib/                       # The paper's Section 3 - latent implementation bias
    ├── s2_lab/                       # The paper's Section 4 - look-ahead bias
    ├── s3_nse/                       # The paper's Section 5 - non-standard errors
    ├── s4_zoo/                       # The factor zoo
    ├── spec/inputs.json              # The five-file input contract
    ├── spec/signal_definitions.json  # What each of the 145 panel columns MEANS
    ├── tools/check_inputs.py         # Enforces the input contract
    ├── tools/build_index.py          # Generates INDEX.md; --check gates it
    ├── tests/                        # Contract, purge, provenance and index gates
    ├── INDEX.md                      # Every exhibit -> its driver, label and sample
    ├── README_stage3.md  QUICKSTART_stage3.md  DATA_DICTIONARY.md
    ├── RECONCILIATION_ia08.md        # Table IA.VIII vs Stage 2 vs the code
    └── data/  reports/               # Gitignored build artifacts
```

> ⚠ **`s1_lib/` is the paper's Section 3 and `s3_nse/` its Section 5.** The folder
> numbers are the order the sections were built, not the section numbers.

---

## Output Data Structure

### Stage 0 Output: Daily TRACE Panels

Stage 0 produces daily panels in dataset-specific subfolders with the following structure:

**File locations:**
- `enhanced/trace_enhanced_YYYYMMDD.parquet`
- `standard/trace_standard_YYYYMMDD.parquet`
- `144a/trace_144a_YYYYMMDD.parquet`

**Quality reports location:**
- `data_reports/enhanced/` - Enhanced TRACE reports
- `data_reports/standard/` - Standard TRACE reports
- `data_reports/144a/` - Rule 144A reports

**Column structure** (21 columns; every one defined in [stage0/DATA_DICTIONARY.md](stage0/DATA_DICTIONARY.md)):

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

**Expected output size** (the 2026-09-10 run):
- Enhanced TRACE (2002-07 to the data frontier): ~31 million rows, about 2.2 GB
- Standard TRACE (from 2024-10): opt-in, not part of the default run
- Rule 144A (first trade 2003-10): ~3.8 million rows, about 250 MB

**Additional outputs:**
- Audit files documenting filter effects (in dataset subfolders)
- CUSIP lists of bonds with corrections (in dataset subfolders)
- Data quality reports with LaTeX + figures (in `data_reports/` subfolder)

### Stage 1 Output

**File location:** `stage1/data/stage1_YYYYMMDD.parquet`

**Structure:** Panel data with one row per (cusip_id, trd_exctn_dt) combination

**Output size:** about 2.7 GB for the full sample with Enhanced and 144A (the 2026-09-10 run)

**Data download:** Available in zipped parquet format on [Open Bond Asset Pricing](https://openbondassetpricing.com/data)

**Column structure (44 columns):**

#### Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `cusip_id` | category | 9-character CUSIP identifier |
| `issuer_cusip`* | category | 6-character issuer CUSIP |
| `permno` | Int32 | CRSP PERMNO equity identifier |
| `permco`† | Int32 | CRSP PERMCO company identifier |
| `gvkey`† | Int32 | Compustat GVKEY identifier |
| `trd_exctn_dt` | datetime | Trade execution date |

#### Computed Bond Analytics (QuantLib)

All prices are in **percentage of par** (e.g., 99 = 99% of par = $990 for a $1,000 principal bond).

| Column | Type | Description |
|--------|------|-------------|
| `pr` | float32 | Volume-weighted clean price (% of par) |
| `prfull` | float32 | Dirty price = pr + acclast (% of par) |
| `acclast` | float32 | Accrued interest — pure time-accrued interest component |
| `accpmt` | float32 | Accumulated coupon payments since issue |
| `accall` | float32 | Accumulated payments — includes cash flows + accrued interest; used for return calculations |
| `ytm` | float64 | Yield to maturity (annualized) |
| `mod_dur` | float32 | Modified duration (years) |
| `mac_dur` | float32 | Macaulay duration (years) |
| `convexity` | float32 | Bond convexity |
| `bond_maturity` | float32 | Time to maturity (years) |
| `credit_spread` | float64 | Credit spread over the Treasury yield at the bond's maturity (Liu-Wu curve) |

#### TRACE Pricing (from Stage 0)

All prices are in **percentage of par**.

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

#### Dealer Bid/Ask Metrics

| Column | Type | Description |
|--------|------|-------------|
| `prc_bid` | float32 | Dealer bid price, value-weighted (% of par) |
| `bid_last` | float32 | Last dealer bid price of day (% of par) |
| `bid_time_ew`‡ | float32 | Average dealer bid time (seconds after midnight) |
| `bid_time_last`‡ | Int32 | Last dealer bid time (seconds after midnight) |
| `prc_ask` | float32 | Dealer ask price, value-weighted (% of par) |
| `bid_count`‡ | Int16 | Number of dealer buys (can be NaN) |
| `ask_count`‡ | Int16 | Number of dealer sells (can be NaN) |

#### Database Source

| Column | Type | Description |
|--------|------|-------------|
| `db_type` | Int8 | Source database: 1=Enhanced, 2=Standard, 3=144A |

#### Bond Characteristics (from FISD)

| Column | Type | Description |
|--------|------|-------------|
| `coupon`* | float32 | Annual coupon rate (%) |
| `principal_amt`* | Int16 | Principal amount per bond (typically $1,000) |
| `bond_age` | float32 | Bond age since issuance (years) |
| `bond_amt_outstanding` | Int64 | Units of the bond outstanding |
| `callable`* | Int8 | Callable flag: 1=callable, 0=not callable |

#### Industry Classifications

| Column | Type | Description |
|--------|------|-------------|
| `ff12num` | int8 | Fama-French 12 industry classification |
| `ff17num` | int8 | Fama-French 17 industry classification |
| `ff30num` | int8 | Fama-French 30 industry classification |

#### Credit Ratings

| Column | Type | Description |
|--------|------|-------------|
| `sp_rating`† | Int8 | S&P credit rating (1-22, where 22=default) |
| `sp_naic`* | Int8 | S&P NAIC category (1-6) |
| `mdy_rating`† | Int8 | Moody's credit rating (1-21, where 21=default) |
| `spc_rating`† | Int8 | S&P composite rating (1-22); missing values filled with mdy_rating (scaled to 22 for default) |
| `mdc_rating`† | Int8 | Moody's composite rating (1-22); missing values filled with sp_rating (scaled to 21 for default) |
| `comp_rating`* | float64 | Average of spc_rating and mdc_rating |

**Notes:**
- \*Columns marked with an asterisk are not in the main Stage 1 file. Where to get each:
  - `coupon`, `principal_amt` -- merge `stage0/enhanced/trace_enhanced_fisd_YYYYMMDD.parquet`
  - `callable` -- `stage1/data/call_dummy_YYYYMMDD.parquet`
  - `sp_naic` -- as `sp_naic_numeric` in `stage1/data/sp_ratings_YYYYMMDD.parquet`
  - `issuer_cusip` -- it is simply `cusip_id[:6]`; no merge needed
  - `comp_rating` -- **not written anywhere**. It is computed and then dropped before
    export; reconstruct it as the mean of `spc_rating` and `mdc_rating`.
- †Columns marked with dagger are excluded from the public download due to proprietary data restrictions
- ‡Columns marked with double dagger are excluded from the public download to reduce file size
- All `prc_*` prices are in percentage of par (99 = 99% of $1,000 = $990)

### Stage 2 Output
Monthly panel of 145 columns carrying 108 corporate bond signals, ready for asset pricing
research, plus the unadjusted `_mmn` twins and the factor time series. Every column is
defined in [stage2/DATA_DICTIONARY.md](stage2/DATA_DICTIONARY.md).

---

## Performance

**Expected Runtime (WRDS Cloud):**

Using `./run_pipeline.sh` (complete automated pipeline from ROOT):
- **Stage 0 - Data processing** (Enhanced and 144A in parallel):
  - Enhanced TRACE: **~2 hours**. It was ~4 hours before v2.2.0; it now pulls 5 CUSIP
    chunks at a time over separate WRDS connections. Measured 2026-09-09: 485 chunks in
    2.01 h against a serial-equivalent 9.93 h, a **4.94x** speedup.
  - 144A TRACE: **~45 minutes**
  - Standard TRACE: ~30-60 minutes, and OPT-IN since v2.2.0 (`TRACE_MEMBERS`). When
    requested it is scheduled after the other two, not beside them.
- **Stage 0 - Report generation**: ~10-15 minutes since v2.2.2 (was ~50); waits for every
  member submitted, and runs ALONGSIDE Stage 1 rather than before it
- **Stage 1 - Bond analytics**: **~2.5-3 hours** (waits for the Stage 0 DATA jobs)

**End to end: about 4.5-5 hours.** The 2026-09-10 production run took 4.7 h wall clock
(07:33 to 12:16) and wrote a Stage 1 panel of 31,412,833 rows ending 2025-12-31. That run
predates `DATE_CUT_OFF = "auto:complete"`, which ends the sample at the last month every
source covers through its final trading session. Cut there, the panel is 31,344,732 rows over
70,684 CUSIPs, 2002-07-01 to 2025-11-28, and that is the file the 2026 vintage was built from.
A run with the current code writes that cut directly.

**How it works:**
The script uses SGE's `-hold_jid` to create automatic dependency chains:
1. Stage 0: the members in `TRACE_MEMBERS` are submitted — by default Enhanced and 144A,
   which run at the same time. Standard, if requested, is held behind them.
2. Stage 0: the report job waits until every submitted extraction job completes.
3. Stage 1: also waits on the extraction jobs, then runs **in parallel with the reports**
   — it reads only the member panels and the FISD file, nothing the reports produce.
4. All jobs submitted with a single command from ROOT.

**Resource Usage** (requested per member by `run_pipeline.sh`; `m_mem_free` is charged
PER SLOT, so the total is `slots x mem` and must stay within the WRDS caps of 8 cores and
48 GB per job):

| job | request | total |
|---|---|---|
| Enhanced | `-pe onenode 5 -l m_mem_free=8G` | 40 GB |
| 144A | `-pe onenode 1 -l m_mem_free=16G` | 16 GB |
| Standard (opt-in) | `-pe onenode 6 -l m_mem_free=8G` | 48 GB |
| Data reports | `-pe onenode 5 -l m_mem_free=8G` | 40 GB |
| Stage 1 | `-pe onenode 4 -l m_mem_free=10G` | 40 GB |

- **Disk**: ~1-2 GB per dataset (Parquet)
- Tune with `CONCURRENCY`, `MEM_PER_SLOT_GB` and `TARGET_ROWS_PER_CHUNK` in
  `stage0/_trace_settings.py` — not by editing the job scripts, whose directives the
  command-line request overrides.

---

## Citation

If you use this pipeline in your research, please cite:

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

---

## References

This pipeline builds on methods from:

- **Dick-Nielsen, J.** (2009). Liquidity biases in TRACE. *The Journal of Fixed Income*, 19(2), 43-55.
- **Dick-Nielsen, J.** (2014). How to clean enhanced TRACE data. Working Paper.
- **van Binsbergen, J. H., Nozawa, Y., & Schwert, M.** (2025). Duration-based valuation of corporate bonds. *The Review of Financial Studies*, 38(1), 158-191.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas where contributions would be valuable:**
- Testing Stage 0 on different WRDS environments
- Additional filter implementations
- Performance optimizations
- Extended documentation
- Bug fixes and error reporting

---

## Support

- **Email**: alexander.dickerson1@unsw.edu.au
- **Issues**: [GitHub Issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)
- **Collaboration**: We welcome collaborators - please reach out!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated:** September 2026
**Version:** 3.2.0 — see [CHANGELOG.md](CHANGELOG.md) for what each release changed.

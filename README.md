# TRACE Data Pipeline

A comprehensive, pipeline for processing Enhanced, Standard and 144A TRACE (Trade Reporting and Compliance Engine) corporate bond transaction data. 
It is apart of the [Open Bond Asset Pricing project](https://openbondassetpricing.com/).
This pipeline implements cleaning procedures and error-correction algorithms to produce *high-quality, reproducible* daily and monthly corporate bond panels from raw TRACE transaction data.
The companion repository is [PyBondLab](https://github.com/GiulioRossetti94/PyBondLab/tree/main/examples) which can be used to form corporate bond asset pricing factors.

[![Website](https://img.shields.io/badge/Website-Visit-blue?logo=google-chrome&logoColor=white)](https://openbondassetpricing.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Stage 0](https://img.shields.io/badge/Stage%200-Public%20Beta-green)](stage0/)
[![Stage 1](https://img.shields.io/badge/Stage%201-Public%20Beta-green)](stage1/)
[![Stage 2](https://img.shields.io/badge/Stage%202-December%202025-orange)](stage2/)

[📄 Link to paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4575879)
---

## Overview

This is a **three-stage pipeline** for building *clean, reliable and reproducible* TRACE corporate bond datasets. 

### Stage 0: Intraday to Daily Processing  **PUBLIC BETA**
Processes raw intraday TRACE transaction data to clean daily panels. Handles three types of TRACE data:
- **Enhanced TRACE**
- **Standard TRACE**
- **Rule 144A bonds**

**Automated workflow:** Run `./run_pipeline.sh` from the project ROOT to execute the complete multi-stage pipeline with automatic job dependencies. Stage 0 jobs run in parallel, then automatically chain to Stage 1 processing when complete.

**Status:** Public beta - fully functional and ready for testing
**Execution:** WRDS Cloud or your home machine (WRDS subscription required)
**Documentation:** See [stage0/README_stage0.md](stage0/README_stage0.md) and [stage0/QUICKSTART_stage0.md](stage0/QUICKSTART_stage0.md)

### Stage 1: Daily Bond Analytics  **PUBLIC BETA**
Enriches Stage 0 daily panels with comprehensive bond analytics and characteristics:
- **Bond analytics** via QuantLib (duration, convexity, YTM, credit spreads)
- **Credit ratings** from S&P and Moody's with numeric conversions
- **Equity identifiers** equity linkers
- **FISD bond characteristics** (coupon, maturity, issuer, amount outstanding, etc.)
- **Fama-French industry classifications** (17 and 30 industries)
- **Ultra-distressed filters** to flag potentially erroneous prices

**Status:** Public beta - fully functional and ready for testing
**Execution:** WRDS Cloud or your home machine (WRDS subscription required)
**Documentation:** See [stage1/README_stage1.md](stage1/README_stage1.md) and [stage1/QUICKSTART_stage1.md](stage1/QUICKSTART_stage1.md)

### Stage 2: Monthly Panel with Factor Signals  **IN DEVELOPMENT**
Produces a clean, error-corrected monthly panel with dozens of corporate bond signals for asset pricing research:
- 50+ bond characteristic signals
- Credit risk factors
- Liquidity measures
- Momentum and reversal signals
- Carry and value signals
- Ready-to-use for monthly portfolio construction -- see [PyBondLab](https://github.com/GiulioRossetti94/PyBondLab/tree/main/examples)

**Status:** In development
**Release:** Coming soon
**Execution:** WRDS Cloud or your home machine (WRDS subscription required)

---

## Project Status & Timeline

- **Stage 0**: ✅ **Now available** - Public beta, ready for testing
- **Stage 1**: ✅ **Now available** - Public beta, ready for testing
- **Stage 2**: 🚧 **Coming soon** - In development

**This project is under active development and any feedback is greatly appreciated.**
Please reach out to `alexander.dickerson1@unsw.edu.au` if you would like to collaborate or beta test.

---

## Key Features

### Stage 0: Robust Error Correction
- **Decimal-shift corrector**: Automatically detects and fixes multiplicative price errors (10x, 0.1x, 100x, 0.01x)
- **Bounce-back filter**: Identifies and removes erroneous price spikes that revert quickly
- Algorithms designed by Dickerson, Robotti & Rossetti (2025) account for TRACE idiosyncrasies
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
- **Fama-French industry classifications** (17 and 30 industry groups)
- Produces comprehensive daily bond-level dataset with 50+ variables
- Ultra-distressed filter catches suspiciupus "rounded" price numbers at very low prices often associated with issues trading under default. See [README_distressed_filter.md](stage1/README_distressed_filter.md)

---

## Quick Start

### Prerequisites
- WRDS subscription with access to TRACE, FISD, and ratings data
- Python 3.10 or higher (tested on Python 3.12.11)
- SSH access to WRDS Cloud (or local Python environment)
- `.pgpass` configured for passwordless WRDS authentication

### 3-Step Setup

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

3. **Run the complete pipeline:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

**What happens:**
1. **Pre-stage**: Auto-downloads required data files (Liu-Wu yields, OSBAP linker, FF industries)
2. **Stage 0**: Submits 3 parallel jobs (Enhanced, Standard, 144A TRACE)
3. **Stage 0 Reports**: Auto-generates when all TRACE jobs complete
4. **Stage 1**: Auto-starts after Stage 0 reports finish
5. **Total runtime**: ~7 hours on WRDS Cloud

**Automated features:**
- ✅ Data downloads (no manual wget required)
- ✅ Job dependencies (stages run in correct order)
- ✅ Configuration auto-detection (STAGE0_DATE_STAMP, ROOT_PATH, N_CORES)
- ✅ Centralized settings (`config.py` for shared settings)

**For detailed instructions:**
- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) for complete walkthrough
- **Stage 0**: See [stage0/README_stage0.md](stage0/README_stage0.md) or [stage0/QUICKSTART_stage0.md](stage0/QUICKSTART_stage0.md)
- **Stage 1**: See [stage1/README_stage1.md](stage1/README_stage1.md) or [stage1/QUICKSTART_stage1.md](stage1/QUICKSTART_stage1.md)

---

## Documentation

**Stage 0 - TRACE Data Processing:**
- **[README](stage0/README_stage0.md)**: Complete guide for intraday to daily TRACE processing
- **[QUICKSTART](stage0/QUICKSTART_stage0.md)**: Fast-track guide to get started quickly
- **[Configuration Guide](stage0/README_stage0.md#configuration-choices-you-can-edit)**: All configurable parameters
- **[Troubleshooting](stage0/README_stage0.md#troubleshooting)**: Common issues and solutions

**Stage 1 - Bond Analytics:**
- **[README](stage1/README_stage1.md)**: Complete guide for bond analytics and enrichment
- **[QUICKSTART](stage1/QUICKSTART_stage1.md)**: Fast-track guide to get started quickly
- **[Configuration Guide](stage1/README_stage1.md#configuration-choices-you-can-edit)**: All configurable parameters
- **[Troubleshooting](stage1/README_stage1.md#troubleshooting)**: Common issues and solutions

**Stage 2 - Monthly Panel:**
- Coming soon

---

## Downloading Results to Your Local Machine

The pipeline generates a large folder (~6 GB) with hundreds of files. **Zip the folder first**, then download a single file for reliability and speed.

### Quick Overview

1. **SSH into WRDS** and zip to scratch space (avoids home directory quota):
   ```bash
   ssh {wrds_username}@wrds-cloud.wharton.upenn.edu
   cd /scratch/{institution}/
   zip -r trace-data-pipeline.zip ~/trace-data-pipeline/
   ```

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
├── CONTRIBUTING.md                   # Contribution guidelines
├── CHANGELOG.md                      # Version history
├── requirements.txt                  # Python dependencies (all stages)
├── run_pipeline.sh                   # ✨ One-push button orchestrator (ROOT)
├── .gitignore                        # Git ignore rules
│
├── stage0/                           # ✅ PUBLIC BETA - Intraday to daily processing
│   ├── README_stage0.md              # Detailed documentation
│   ├── QUICKSTART_stage0.md          # Fast-track guide
│   ├── _trace_settings.py            # Configuration file
│   ├── create_daily_enhanced_trace.py
│   ├── create_daily_standard_trace.py
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
│   ├── standard/                     # Standard TRACE output (auto-created)
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
│   ├── _stage1_settings.py           # Configuration file
│   ├── create_daily_stage1.py        # Main processing module
│   ├── helper_functions.py           # Utility functions
│   ├── _run_stage1.py                # Runner script
│   ├── run_stage1.sh                 # Job submission script
│   ├── requirements.txt              # Stage 1 specific dependencies
│   │
│   ├── data/                         # Stage 1 output (auto-created)
│   │   ├── stage1_YYYYMMDD.parquet   # Enriched dataset
│   │   ├── liu_wu_yields.xlsx        # Downloaded treasury yields
│   │   ├── OSBAP_Linker_*.parquet    # Downloaded linker file
│   │   ├── Siccodes17.txt            # FF17 industry file
│   │   ├── Siccodes30.txt            # FF30 industry file
│   │   └── reports/                  # Data quality reports
│   │
│   └── logs/                         # Execution logs (auto-created)
│
└── stage2/                           # 🚧 COMING SOON - Monthly panel with signals
    └── (In development)
```

---

## Output Data Structure

### Stage 0 Output: Daily TRACE Panels

Stage 0 produces daily panels in dataset-specific subfolders with the following structure:

**File locations:**
- `enhanced/enhanced_YYYYMMDD.parquet`
- `standard/standard_YYYYMMDD.parquet`
- `144a/144a_YYYYMMDD.parquet`

**Quality reports location:**
- `data_reports/enhanced/` - Enhanced TRACE reports
- `data_reports/standard/` - Standard TRACE reports
- `data_reports/144a/` - Rule 144A reports

**Column structure:**

| Column | Description |
|--------|-------------|
| `cusip_id` | 9-character CUSIP identifier |
| `trd_exctn_dt` | Trade execution date |
| `prc_ew` | Equal-weighted price |
| `prc_vw` | Volume-weighted price (dollar) |
| `prc_vw_par` | Volume-weighted price (par) |
| `prc_first` | First trade price of day |
| `prc_last` | Last trade price of day |
| `trade_count` | Number of trades |
| `qvolume` | Par volume (millions) |
| `dvolume` | Dollar volume (millions) |
| `prc_bid` | Dealer bid (value-weighted) |
| `prc_ask` | Dealer ask (value-weighted) |
| `prc_lo` | Low price of the day |
| `prc_hi` | High price of the day |
| `bid_count` | Number of buys |
| `ask_count` | Number of sells |

**Expected output size:**
- Enhanced TRACE (2002-present): ~30 million rows
- Standard TRACE (2024-present): ~2-3 million rows
- Rule 144A (2002-present): ~5-8 million rows

**Additional outputs:**
- Audit files documenting filter effects (in dataset subfolders)
- CUSIP lists of bonds with corrections (in dataset subfolders)
- Data quality reports with LaTeX + figures (in `data_reports/` subfolder)

### Stage 1 Output

**File location:** `stage1/data/stage1_YYYYMMDD.parquet`

**Structure:** Panel data with one row per (cusip_id, trd_exctn_dt) combination

**Output size:** ~500MB-2GB (depending on time period and datasets included)

**Data download:** Available in zipped parquet format on [Open Bond Asset Pricing](https://openbondassetpricing.com/data)

**Column structure (43 columns):**

#### Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `cusip_id` | category | 9-character CUSIP identifier |
| `issuer_cusip`* | category | 6-character issuer CUSIP |
| `permno` | Int32 | CRSP PERMNO equity identifier |
| `permco` | Int32 | CRSP PERMCO company identifier |
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
| `credit_spread` | float64 | Credit spread over duration-matched Treasury yield |

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
- \*Columns marked with asterisk are not included in the output file but can be obtained by merging with FISD data in `stage0/enhanced/trace_enhanced_fisd_YYYYMMDD.parquet`
- †Columns marked with dagger are excluded from the public download due to proprietary data restrictions
- ‡Columns marked with double dagger are excluded from the public download to reduce file size
- All `prc_*` prices are in percentage of par (99 = 99% of $1,000 = $990)

### Stage 2 Output (Coming Soon)
Monthly panel with 50+ corporate bond signals ready for asset pricing research.

---

## Performance

**Expected Runtime (WRDS Cloud):**

Using `./run_pipeline.sh` (complete automated pipeline from ROOT):
- **Stage 0 - Data processing** (parallel): ~4-8 hours
  - Enhanced TRACE: ~4 hours
  - Standard TRACE: ~30-60 minutes
  - 144A TRACE: ~30-60 minutes
- **Stage 0 - Report generation**: ~30-60 minutes (waits for all three datasets)
- **Stage 1 - Bond analytics**: ~2 hours (waits for Stage 0 reports)
- **Total**: ~7 hours for complete pipeline (Stage 0 + Stage 1)

**How it works:**
The script uses SGE's `-hold_jid` feature to create automatic dependency chains:
1. Stage 0: Three data extraction jobs run in parallel
2. Stage 0: Report job waits until all three extraction jobs complete
3. Stage 1: Analytics job waits until Stage 0 reports complete
4. All jobs submitted with a single command from ROOT

**Resource Usage:**
- **Stage 0:**
  - Memory: ~4-8GB per job (with default chunk_size=250)
  - Disk: ~1-2GB per dataset (Parquet format)
  - Parallel execution: All three datasets can run simultaneously
- **Stage 1:**
  - Memory: 24GB RAM required in WRDS (specified in run_stage1.sh as `#$ -l m_mem_free=24G`)
  - Processing time: 2-6 hours depending on dataset size and parallel cores

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@unpublished{dickerson2025pitfalls,
  author = {Dickerson, Alexander and Robotti, Cesare and Rossetti, Giulio},
  title = {Common pitfalls in the evaluation of corporate bond strategies},
  year = {2025},
  note = {Working Paper}
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

**Last Updated:** November 2025
**Stage 0 Version:** 1.0.0 (Public Beta)
**Stage 1 Version:** 1.0.0 (Public Beta)

# TRACE Data Pipeline

A comprehensive, pipeline for processing Enhanced, Standard and 144A TRACE (Trade Reporting and Compliance Engine) corporate bond transaction data. 
It is apart of the [Open Bond Asset Pricing project](https://openbondassetpricing.com/).
This pipeline implements cleaning procedures and error-correction algorithms to produce *high-quality, reproducible* daily and monthly corporate bond panels from raw TRACE transaction data.
The companion repository is [PyBondLab](https://github.com/GiulioRossetti94/PyBondLab/tree/main/examples) which can be used to form corporate bond asset pricing factors.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Stage 0](https://img.shields.io/badge/Stage%200-Public%20Beta-green)](stage0/)
[![Stage 1](https://img.shields.io/badge/Stage%201-November%202025-orange)](stage1/)
[![Stage 2](https://img.shields.io/badge/Stage%202-November%202025-orange)](stage2/)

---

## Overview

This is a **three-stage pipeline** for building *clean, reliable and reproducible* TRACE corporate bond datasets. 

### Stage 0: Intraday to Daily Processing (WRDS Cloud)  **PUBLIC - READY FOR TESTING**
Processes raw intraday TRACE transaction data to clean daily panels on the WRDS Cloud. Handles three types of TRACE data:
- **Enhanced TRACE**
- **Standard TRACE**
- **Rule 144A bonds**

**Automated workflow:** Run `./run_all_trace.sh` to execute all three data processing jobs in parallel, then automatically generate comprehensive quality reports when complete using SGE job dependencies. All code runs with a "one-push button".

**Status:** Public beta - fully functional and ready to be tested   
**Execution:** WRDS Cloud (WRDS subscription required)  
**Documentation:** See [stage0/README_stage0.md](stage0/README_stage0.md)

### Stage 1: Daily Bond Metrics (Your Home Machine)  **IN DEVELOPMENT**
Calculates comprehensive daily bond metrics from Stage 0 output, including:
- Accrued interest 
- Daily credit spreads
- Duration and convexity measures
- Yield-to-maturity 
- Additional bond characteristics (rating, amount outstanding etc.)

**Status:** In development  
**Release:** End of November 2025  
**Execution:** Your home machine (with WRDS connection)

### Stage 2: Monthly Panel with Factor Signals (Your Home Machine)  **IN DEVELOPMENT**
Produces a clean, error-corrected monthly panel with dozens of corporate bond signals for asset pricing research:
- 50+ bond characteristic signals
- Credit risk factors
- Liquidity measures
- Momentum and reversal signals
- Carry and value signals
- Ready-to-use for monthly portfolio construction -- see [PyBondLab](https://github.com/GiulioRossetti94/PyBondLab/tree/main/examples)

**Status:** In development  
**Release:** End of November 2025  
**Execution:** Your home machine (with WRDS connection)

---

## Project Status & Timeline

- **Stage 0**:  **Now available** - Public beta, ready for testing
- **Stages 1 & 2**:  **Coming November 2025** - In development, close to completion

**This project is under active development and any feedback is greatly appreciated.**  
Please reach out to `alexander.dickerson1@unsw.edu.au` if you would like to collaborate or beta test.

---

## Key Features

### Stage 0: Robust Error Correction
- **Decimal-shift corrector**: Automatically detects and fixes multiplicative price errors (10x, 0.1x, 100x, 0.01x)
- **Bounce-back filter**: Identifies and removes erroneous price spikes that revert quickly
- Algorithms designed by Dickerson, Robotti & Rossetti (2025) account for TRACE idiosyncrasies
- **Full documentation Coming November 2025**


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
- **Bid/Ask metrics**: Customer-side value-weighted bid and ask prices

---

## Quick Start (Stage 0)

### Prerequisites
- WRDS subscription with access to TRACE data
- Python 3.10 or higher
- SSH access to WRDS Cloud

### Installation

1. **Clone the repository on WRDS:**
```bash
ssh username@wrds-cloud.wharton.upenn.edu
cd ~
git clone https://github.com/Alexander-M-Dickerson/trace-data-pipeline.git
cd trace-data-pipeline/stage0
```

2. **Install dependencies:**
```bash
pip install --user -r requirements.txt
```

3. **Configure settings:**
Edit `_trace_settings.py` and set your WRDS username:
```python
WRDS_USERNAME = "your_wrds_username"
```

4. **Run the pipeline:**
```bash
chmod +x run_all_trace.sh
./run_all_trace.sh
```
This executes all three data processing jobs in parallel, then uses SGE's `-hold_jid` to automatically submit the report generation job once all three complete.

**What happens:**
1. Submits 3 parallel jobs: Enhanced, Standard, and 144A
2. Submits a 4th job with `-hold_jid` dependency that waits for all three to finish
3. Data outputs save to dataset-specific folders: `enhanced/`, `standard/`, `144a/`
4. Reports automatically generate and save to `data_reports/` with subfolders for each dataset
5. Total runtime: ~5 hours for the complete pipeline

For detailed instructions, see [stage0/README_stage0.md](stage0/README_stage0.md).

---

## Documentation

- **[Stage 0 README](stage0/README_stage0.md)**: Complete guide for intraday to daily TRACE processing
- **[Configuration Guide](stage0/README_stage0.md#configuration-choices-you-can-edit)**: All configurable parameters explained
- **[Troubleshooting](stage0/README_stage0.md#troubleshooting)**: Common issues and solutions
- **Stage 1 README**: Coming November 2025
- **Stage 2 README**: Coming November 2025

---

## Repository Structure

```
trace-data-pipeline/
├── LICENSE                           # MIT License
├── README.md                         # This file
├── CONTRIBUTING.md                   # Contribution guidelines
├── CHANGELOG.md                      # Version history
├── requirements.txt                  # Python dependencies (Stage 0)
├── .gitignore                        # Git ignore rules
│
├── stage0/                           #  PUBLIC - Intraday to daily processing (WRDS Cloud)
│   ├── README_stage0.md             # Detailed documentation
│   ├── _trace_settings.py           # Configuration file
│   ├── create_daily_enhanced_trace.py
│   ├── create_daily_standard_trace.py
│   ├── _run_enhanced_trace.py
│   ├── _run_standard_trace.py
│   ├── _run_144a_trace.py
│   ├── _build_error_files.py        # Report generation
│   ├── _error_plot_helpers.py       # Plotting utilities
│   ├── run_all_trace.sh             # Submit all jobs with auto-report generation
│   ├── run_enhanced_trace.sh
│   ├── run_standard_trace.sh
│   ├── run_144a_trace.sh
│   ├── run_build_data_reports.sh
│   │
│   ├── enhanced/                    # Enhanced TRACE data (auto-created)
│   ├── standard/                    # Standard TRACE data (auto-created)
│   ├── 144a/                        # Rule 144A data (auto-created)
│   │
│   └── data_reports/                # Quality reports for ALL datasets (auto-created)
│       ├── enhanced/
│       ├── standard/
│       └── 144a/
│
├── stage1/                           #  November 2025 - Daily bond metrics
│   └── (Coming soon)
│
└── stage2/                           #  November 2025 - Monthly panel with signals
    └── (Coming soon)
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
| `prc_bid` | Customer-side bid (value-weighted) |
| `prc_ask` | Customer-side ask (value-weighted) |
| `prc_lo` | Low price of the day |
| `prc_hi` | High price of the day |
| `bid_count` | Number of customer buys |
| `ask_count` | Number of customer sells |

**Expected output size:**
- Enhanced TRACE (2002-present): ~30 million rows
- Standard TRACE (2024-present): ~2-3 million rows
- Rule 144A (2002-present): ~5-8 million rows

**Additional outputs:**
- Audit files documenting filter effects (in dataset subfolders)
- CUSIP lists of bonds with corrections (in dataset subfolders)
- Data quality reports with LaTeX + figures (in `data_reports/` subfolder)

### Stage 1 Output (Coming November 2025)
Daily bond metrics including accrued interest, credit spreads, duration, convexity, yields, and rating-adjusted measures.

### Stage 2 Output (Coming November 2025)
Monthly panel with 50+ corporate bond signals ready for asset pricing research.

---

## Performance

**Expected Runtime (WRDS Cloud - Stage 0):**

Using `./run_all_trace.sh` (complete automated pipeline):
- **Data processing** (parallel): ~4-8 hours
  - Enhanced: 4 hours
  - Standard: 30-60 minutes
  - 144A: 30-60 minutes
- **Report generation** (after all data jobs complete): ~30-60 minutes
- **Total**: ~5 hours for everything

**How it works:**
The script uses SGE's `-hold_jid` feature to create a dependency chain:
1. Three data jobs run in parallel
2. Report job waits in queue (status `hqw`) until all three complete
3. Report job automatically starts when dependencies are satisfied

**Resource Usage (Stage 0):**
- Memory: ~4-8GB per job (with default chunk_size=250)
- Disk: ~1-2GB per dataset (Parquet format)
- Parallel execution: All three datasets can run simultaneously

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

**Last Updated:** January 2025  
**Stage 0 Version:** 1.0.0 (Public Beta)

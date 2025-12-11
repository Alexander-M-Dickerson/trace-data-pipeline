# Changelog

All notable changes to the TRACE Data Pipeline project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Coming Soon
- **Stage 2**: Monthly panel with factor signals (Coming soon)
  - 50+ bond characteristic signals
  - Credit risk factors
  - Liquidity measures
  - Momentum and reversal signals
  - Carry and value signals
  - Portfolio construction ready outputs

---

## [2.0.0] - 2025-12-11

### Added - Stage 1 Release (Bond Analytics & Enrichment)

#### Core Features
- **Complete bond analytics pipeline** enriching Stage 0 daily data with comprehensive metrics
- **Automated orchestration** via `run_pipeline.sh` (handles both Stage 0 and Stage 1)
- **Research-ready output** with ~50+ variables per bond-day observation

#### Bond Characteristics Integration (FISD)
- **Comprehensive bond attributes** from Mergent FISD
  - Coupon rates and payment frequencies
  - Maturity dates and bond tenors
  - Offering amounts and amounts outstanding
  - Callable and puttable features
  - Security types and bond classifications
  - Issuer identifiers and names
- **Automated FISD filtering**
  - USD-only bonds
  - Fixed-rate securities
  - Tenor constraints (minimum/maximum)
  - Valid maturity dates
  - Currency and interest type validation

#### Bond Analytics via QuantLib
- **Yield-to-maturity (YTM)** calculations using QuantLib bond pricing engine
- **Macaulay duration** and **modified duration** (interest rate sensitivity)
- **Convexity** (second-order price sensitivity)
- **Option-adjusted spreads (OAS)** relative to treasury curve
- **Credit spreads** computed against Liu-Wu zero-coupon treasury yields
- **Robust error handling** for bonds with missing or invalid parameters
- **Efficient multi-core processing** with joblib parallelization

#### Credit Ratings Integration
- **S&P ratings** from WRDS CompuStat Ratings Monthly
  - Numeric ratings (1-22 scale)
  - NAIC designations
  - Historical rating changes tracking
- **Moody's ratings** from WRDS Mergent FISD
  - Numeric ratings (1-22 scale)
  - Composite rating methodology
- **Automatic rating alignment** with bond-month observations

#### Equity Identifiers (OSBAP Linker)
- **CRSP identifiers**: PERMNO and PERMCO
- **Compustat identifier**: GVKEY
- **Enables cross-asset research** linking bonds to equities

#### Industry Classifications
- **Fama-French 17 industry classification**
- **Fama-French 30 industry classification**
- **SIC code mapping** from CRSP via PERMNO linkage
- **Automatic monthly assignment** based on equity identifiers

#### Ultra-Distressed Bond Filters
- **Advanced price anomaly detection** to identify suspicious observations
- **Five-stage filtering methodology**:
  1. **Anomalous price detection**: Ultra-low prices with normal price context
  2. **Upward spike detection**: High prices inconsistent with recent trading
  3. **Plateau sequence detection**: Sustained ultra-low price sequences
  4. **Intraday inconsistency**: Wide intraday ranges at distressed prices
  5. **Round number detection**: Suspicious exact prices (0.01, 0.10, etc.)
- **Refined composite flag** (`flag_refined_any`) combining all detection methods
- **CUSIP-level export** tracking all flagged bonds with detailed statistics
  - Export file: `stage1/data/ultra_distressed_cusips_{date}.csv`
  - Includes flag counts, percentages, and date ranges
  - Enables quality control and manual review workflows

#### Treasury Yield Integration
- **Liu-Wu zero-coupon treasury yields** (1961-present)
  - Downloaded automatically from public source
  - Monthly interpolated yields (1-30 years maturity)
  - Used for credit spread calculations
- **FRED treasury yields** (alternative source, configurable)

#### Configuration & Settings
- **Harmonized configuration system** with single source of truth
  - `config.py`: Shared settings across all stages
  - `TRACE_MEMBERS`: Dataset selection (enhanced, standard, 144a)
  - `STAGE0_OUTPUT_FIGURES`: Control Stage 0 error plots (slow)
  - Stage 1 always generates comprehensive reports (no toggle)
- **Auto-detection features**:
  - Stage 0 date stamp from parquet files
  - CPU core count optimization
  - Root path detection
- **Minimal user configuration** required (just WRDS username)

#### Performance Optimizations
- **Memory efficiency**:
  - CUSIP columns use category dtype (~75% memory savings)
  - Optimized groupby operations for 30M+ row datasets
  - Efficient parquet compression
  - Strategic garbage collection
- **Processing speed**:
  - Multi-core parallelization for bond analytics
  - Chunked processing for large datasets
  - Vectorized operations throughout pipeline
- **WRDS quota monitoring**:
  - Pre-flight disk space check before pipeline execution
  - Parses WRDS quota (not filesystem) for accurate warnings
  - Warns if < 4 GB available (prevents job failures)
  - `FORCE_RUN=1` override for advanced users

#### Output Files & Reports
- **Comprehensive daily bond dataset** (`stage1_YYYYMMDD.parquet`)
  - All Stage 0 price/volume metrics
  - FISD bond characteristics
  - QuantLib analytics (duration, convexity, YTM, OAS, spreads)
  - Credit ratings (S&P and Moody's)
  - Equity identifiers (PERMNO, PERMCO, GVKEY)
  - Industry classifications (FF17, FF30)
  - Ultra-distressed filter flags
- **LaTeX data quality reports** (always generated)
  - 8 comprehensive tables analyzing data quality
  - Time-series visualization plots
  - Filter effect summaries
  - Organized output structure
- **Flagged CUSIP export** for quality control
  - CSV file with all ultra-distressed flagged bonds
  - Statistics per CUSIP (total obs, flagged obs, percentages)
  - Breakdown by flag type
  - Date range for each flagged bond

#### Documentation
- **Comprehensive Stage 1 documentation**:
  - `stage1/README_stage1.md`: Full technical documentation
  - `stage1/QUICKSTART_stage1.md`: Quick start guide
  - `stage1/README_distressed_filter.md`: Ultra-distressed filter methodology
- **Updated FAQ** with Stage 1-specific sections
  - Configuration guidance
  - Output file descriptions
  - Troubleshooting disk space warnings
  - Performance optimization tips
- **Updated main README** reflecting Stage 1 availability

#### Infrastructure & Automation
- **Unified pipeline orchestrator** (`run_pipeline.sh`)
  - Pre-stage: Download required data files (Liu-Wu yields, OSBAP linker, FF classifications)
  - Stage 0: Parallel TRACE extraction (Enhanced, Standard, 144A)
  - Stage 0: Report generation after extraction
  - Stage 1: Bond analytics after Stage 0 completion
  - Automatic job dependency management with SGE `-hold_jid`
- **Disk space validation**:
  - Checks WRDS user quota before execution
  - Prevents pipeline failures from insufficient space
  - Clear warnings with remediation steps
- **Automatic data downloads** on login node (WRDS compute nodes have no internet)
  - Liu-Wu treasury yields
  - OSBAP linker file (ISIN/FIGI/Bloomberg identifiers)
  - Fama-French industry classifications (FF17, FF30)

#### Runtime Performance
- **Stage 1 processing time**: ~2 hours (WRDS Cloud, 2-4 cores)
- **Complete pipeline (Stage 0 + Stage 1)**: ~7 hours total
  - Stage 0 (Enhanced): ~4 hours
  - Stage 0 (Standard): ~30-60 minutes
  - Stage 0 (144A): ~30-60 minutes
  - Stage 0 (Reports): ~30-60 minutes
  - Stage 1: ~2 hours

### Changed
- **Configuration structure** now harmonized across all stages
  - `config.py` is single source of truth for shared settings
  - Removed redundant `TRACE_MEMBERS` from `stage1/_stage1_settings.py`
  - Removed unused `GENERATE_REPORTS` and `OUTPUT_FIGURES` from Stage 1
  - Stage 0 figure generation controlled via `STAGE0_OUTPUT_FIGURES` in `config.py`

### Fixed
- **Disk space check** now uses WRDS quota instead of filesystem space
  - Previous version showed 8TB+ available (filesystem) when user had <2GB (quota)
  - Now correctly parses `quota` command output
  - Accurate warnings prevent job failures from disk space exhaustion
- **CUSIP export performance** optimized for 30M+ row datasets
  - Replaced O(n*m) loop-based approach with O(n) vectorized groupby
  - ~1000-5000x faster for typical datasets
  - Completes in seconds instead of hours

---

## [1.0.0] - 2025-11-01

### Added - Initial Public Beta Release

#### Core Processing Pipeline (Stage 0)
- **Enhanced TRACE processing** (2002-07-01 to present)
  - Full intraday to daily conversion pipeline
  - Configurable parameters via `_trace_settings.py`
  - Automated parallel job submission with `run_all_trace.sh`
  - Output to dedicated `enhanced/` subfolder
  
- **Standard TRACE processing** (configurable start date, default 2024-10-01)
  - Pre-2012 and post-2012 cleaning rules
  - Reversal trade handling specific to Standard TRACE
  - Output to dedicated `standard/` subfolder
  
- **Rule 144A TRACE processing** (2002-07-01 to present)
  - Same cleaning pipeline as Standard TRACE
  - Dedicated processing for private placement bonds
  - Output to dedicated `144a/` subfolder

#### Data Cleaning & Error Correction
- **Decimal-shift correction algorithm**
  - Automatic detection and correction of multiplicative price errors
  - Handles 10x, 0.1x, 100x, and 0.01x errors
  - Novel algorithms by Dickerson, Robotti & Rossetti (2025)
  
- **Bounce-back filter**
  - Identifies and removes erroneous price spikes
  - Detects prices that revert quickly to previous levels
  - Configurable threshold parameters
  
- **Dick-Nielsen filters** (2009, 2014)
  - Cancellation filtering
  - Correction filtering
  - Agency trade de-duplication
  - Reversal handling
  
- **van Binsbergen, Nozawa & Schwert filters** (2025)
  - Advanced trade filtering
  - Duration-based validation

#### Data Quality & Validation
- **Price range filters**
  - Minimum price validation (> 0)
  - Maximum price validation (<= 1000)
  
- **Volume filters**
  - Dollar volume thresholds
  - Par volume thresholds
  - Configurable limits per dataset
  
- **Additional filters**
  - Trading calendar validation (NYSE calendar)
  - Time-of-day filtering (configurable windows)
  - Yield != price trade filtering
  - Volume > 50% offering amount filtering
  - Execution date > maturity date filtering

#### Output & Reporting
- **Daily aggregated metrics**
  - Equal-weighted price (`prc_ew`)
  - Volume-weighted price - dollar (`prc_vw`)
  - Volume-weighted price - par (`prc_vw_par`)
  - First trade price (`prc_first`)
  - Last trade price (`prc_last`)
  - Trade count (`trade_count`)
  - Par volume in millions (`qvolume`)
  - Dollar volume in millions (`dvolume`)
  - Customer-side bid price - value-weighted (`prc_bid`)
  - Customer-side ask price - value-weighted (`prc_ask`)
  - Daily high price (`prc_hi`)
  - Daily low price (`prc_lo`)
  - Bid trade count (`bid_count`)
  - Ask trade count (`ask_count`)
  
- **Audit logging system**
  - Transaction-level audit trails
  - Row count reconciliation at each filter stage
  - CUSIP-level correction lists
  - Comprehensive filter effect documentation
  
- **LaTeX report generation**
  - Automated quality reports for each dataset
  - Detailed filtering statistics
  - Optional time-series visualization plots
  - Organized in `data_reports/` with dataset subfolders
  - Bibliography and citation support

#### Automation & Infrastructure
- **Parallel job execution**
  - `run_all_trace.sh` master script
  - SGE job dependency management with `-hold_jid`
  - Automatic report generation after data processing
  - Individual dataset runners: `run_enhanced_trace.sh`, `run_standard_trace.sh`, `run_144a_trace.sh`
  
- **Output organization**
  - Dataset-specific subfolders (`enhanced/`, `standard/`, `144a/`)
  - Centralized reports folder (`data_reports/`)
  - Parquet format for efficient storage
  - Comprehensive log files in `logs/` directory
  
- **WRDS Cloud integration**
  - Password-less authentication via `.pgpass`
  - Efficient chunked processing (default 250 CUSIPs per chunk)
  - Memory-optimized design (~4-8GB per job)
  - Fast execution (~5 hours complete pipeline)

#### Documentation
- **Comprehensive README files**
  - Main project README with overview
  - Stage 0 detailed documentation (`stage0/README_stage0.md`)
  - Contributing guidelines (`CONTRIBUTING.md`)
  - Clear installation and setup instructions
  
- **Configuration documentation**
  - All parameters explained in `_trace_settings.py`
  - Per-dataset override examples
  - Filter parameter descriptions
  - Aggregation metric specifications
  
- **Troubleshooting guide**
  - Common issues and solutions
  - WRDS setup guidance
  - Performance optimization tips

#### Project Infrastructure
- **MIT License**
  - Open source availability
  - Permissive licensing for research use
  
- **Version control**
  - GitHub repository structure
  - Issue tracking setup
  - Pull request templates
  
- **Dependencies**
  - Python 3.10+ requirement
  - Clear `requirements.txt` for Stage 0
  - WRDS subscription requirements documented

#### Academic Integration
- **Citations and references**
  - Primary citation: Dickerson, Robotti & Rossetti (2025)
  - Secondary citation: Dickerson & Rossetti (2025)
  - Acknowledgment of foundational methods
  
- **Open Bond Asset Pricing integration**
  - Part of broader [Open Bond Asset Pricing project](https://openbondassetpricing.com/)
  - Companion [PyBondLab repository](https://github.com/GiulioRossetti94/PyBondLab) for factor construction
  - Reproducible research framework

### Performance Characteristics
- **Runtime (WRDS Cloud)**
  - Enhanced TRACE: ~4 hours
  - Standard TRACE: ~30-60 minutes
  - Rule 144A: ~30-60 minutes
  - Report generation: ~30-60 minutes
  - Total pipeline: ~5 hours
  
- **Resource usage**
  - Memory: ~4-8GB per job
  - Disk: ~1-2GB per dataset (Parquet format)
  - Parallel execution supported
  
- **Data scale**
  - Enhanced TRACE: ~30 million rows (2002-present)
  - Standard TRACE: ~2-3 million rows (2024-present)
  - Rule 144A: ~5-8 million rows (2002-present)

---

## Project Roadmap

### Version 1.x - Stage 0 Enhancements (Ongoing)
- Bug fixes and performance improvements
- Additional filter options
- Enhanced documentation
- Community contributions integration

### Version 2.0 - Stage 1 Release (November 2025)
- Daily bond metrics calculation module
- Duration and convexity measures
- Credit spread computation
- Yield calculations

### Version 3.0 - Stage 2 Release (November 2025)
- Monthly panel construction
- 50+ bond characteristic signals
- Factor construction tools
- Portfolio-ready outputs
- Integration with PyBondLab

---

## Support & Contribution

For questions, issues, or to contribute:
- **Email**: alexander.dickerson1@unsw.edu.au
- **GitHub Issues**: [trace-data-pipeline/issues](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/issues)
- **Collaboration**: Beta testers and collaborators welcome!

---

## Acknowledgments

This pipeline implements and extends methods from:
- Dick-Nielsen, J. (2009). Liquidity biases in TRACE. *The Journal of Fixed Income*, 19(2), 43-55.
- Dick-Nielsen, J. (2014). How to clean enhanced TRACE data. Working Paper.
- van Binsbergen, J. H., Nozawa, Y., & Schwert, M. (2025). Duration-based valuation of corporate bonds. *The Review of Financial Studies*, 38(1), 158-191.

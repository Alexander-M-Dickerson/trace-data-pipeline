# Changelog

All notable changes to the TRACE Data Pipeline project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Coming Soon
- **Stage 1**: Daily bond metrics module (November 2025)
  - Accrued interest calculations
  - Daily credit spreads
  - Duration and convexity measures
  - Yield-to-maturity calculations
  - Additional bond characteristics (rating, amount outstanding)
- **Stage 2**: Monthly panel with factor signals (November 2025)
  - 50+ bond characteristic signals
  - Credit risk factors
  - Liquidity measures
  - Momentum and reversal signals
  - Carry and value signals
  - Portfolio construction ready outputs

---

## [1.0.0] - 2025-01-20

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

---

[Unreleased]: https://github.com/Alexander-M-Dickerson/trace-data-pipeline/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Alexander-M-Dickerson/trace-data-pipeline/releases/tag/v1.0.0

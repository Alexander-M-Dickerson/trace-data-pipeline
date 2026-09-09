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

## [2.2.0] - 2026-09-09

Stage 0 was the pipeline's long pole: ~4 hours for Enhanced, spent in a chunk loop
that ran strictly one CUSIP chunk at a time on a single WRDS connection while the job
held a whole compute node. This release runs those chunks concurrently.

### Added
- **Concurrent chunk fetching.** Enhanced pulls 5 CUSIP chunks at once, each worker
  process holding its own WRDS connection; 144A runs alongside on one; Standard, when
  requested, runs afterwards and may use the whole budget. Set by `CONCURRENCY` in
  `stage0/_trace_settings.py`, overridable per run with `STAGE0_WORKERS`.
- **`stage0/_wrds_pool.py`** -- one connection per worker, opened inside the child so
  no socket is inherited across a fork, with a serialised staggered handshake.
- **`stage0/_chunk_runner.py`** -- row-balanced chunk planning plus the scheduler,
  which returns results in chunk order however they completed and aborts the run if
  any chunk is missing.
- **Row-balanced chunks.** Chunks are packed to ~750,000 trade rows instead of a fixed
  250 CUSIPs. Measured over the Enhanced universe (111,727 CUSIPs / 345,874,974
  trades), the worst chunk falls from 3,392,802 rows to 749,992 -- a 4.5x cut in the
  memory a job must reserve, since that is set by the worst chunk and not the average.
  `chunk_size` keeps its old meaning for the report job's own chunking.
- **Grid resource requests** derived from each member's worker count and validated
  against the WRDS caps (8 cores, 48 GB per job) before submission. `m_mem_free` is
  charged PER SLOT, so an over-request does not error -- it pends forever, silently.
- **A test suite**, the repo's first: `run_smoke_test.sh` runs stage0 -> reports ->
  stage1 on a few chunks in minutes and asserts 28 cross-stage invariants;
  `tests/test_chunk_plan.py` and `tests/test_chunk_scheduler.py` cover the partition
  and scheduling properties without needing WRDS; `tests/probe_wrds_connections.py`
  measures your own account's connection ceiling.
- **`download_inputs.sh`** -- stage 1's external inputs (Liu-Wu yields, the bond-firm
  linker, the FF industry files) split out of `run_pipeline.sh` into their own script.
  They must be fetched on the LOGIN NODE, because compute nodes have no internet, so a
  submitted smoke test could never fetch them itself. It now exits non-zero when a file
  is missing instead of warning and continuing: every one is required for stage 1's 44
  columns, so a miss is a run that dies hours later having burned the grid time.
- **Canonical output ordering.** Stage 0 sorts by `(cusip_id, trd_exctn_dt)` before
  export. Row order no longer depends on the work plan, which is what makes a
  before/after comparison meaningful at all. The row SET is unchanged.

### Changed
- **`TRACE_MEMBERS` now drives submission**, not just what later stages read.
  `run_pipeline.sh` used to submit all three members regardless.
- **The default is `["enhanced", "144a"]`. Standard is opt-in.** Stage 1 keeps
  Standard rows only after the last Enhanced date, so nearly all of a Standard run was
  being discarded. Ask for it with
  `TRACE_MEMBERS="enhanced standard 144a" ./run_pipeline.sh`.
- Per-filter log lines are collected per chunk and emitted by the parent in chunk
  order, so the `.out` reads the same at any worker count.
- The NYSE trading calendar is built once per run rather than rebuilt inside every
  chunk (185 ms -> 12 ms per chunk).

### Fixed
- **`db_type` was assigned by POSITION in `TRACE_MEMBERS`** (`db_type = i` over
  `enumerate`), so dropping a member silently renumbered the rest. With the new
  default, 144A would have become `db_type = 2`, and the overlap clip immediately
  below keeps `db_type == 2` rows only after the last Enhanced date -- deleting
  almost every 144A row. The job would have exited 0 with a normal-looking file.
  Reproduced deliberately before fixing: 1,362 of 1,527 144A rows destroyed, 11%
  retention, exit code 0. Now an explicit `DB_TYPE_BY_MEMBER` map, with a startup
  assertion that every configured member has a code.
- **`-hold_jid ${J1},${J2},${J3}` broke when a member was not submitted**, leaving an
  unset variable and a malformed `-hold_jid 123,,125`. The hold list is now built from
  the jobs actually submitted, verified across four member sets.
- **`DATE_CUT_OFF` could auto-roll past the treasury curve** and abort any run
  including 144A -- a regression from 2.1.0's own auto-roll, caught by 2.1.0's own
  guard. Now clamped to the last date the curve covers.
- **The report job could not find stage-0 files if the run crossed a date boundary.**
  It resolves the stamp from its own start date, probing +/-1 day; a longer gap failed
  outright. It now falls back to the newest complete set on disk, as stage 1 already
  did.
- **`create_daily_stage1.py` located `stage1_pipeline.py` via `ROOT_PATH`** instead of
  its own directory, so it broke under a redirected root.
- **A refused WRDS connection was fatal where a dropped one was retried.** Both
  engines' `_raw_sql_with_retry` now recognise it. Note that the wrds package reports
  EVERY connect failure as `EOFError: EOF when reading a line`, because its failure
  path calls `input()` -- that covers both a missing username and the connection
  limit, and the pool now distinguishes them.
- **`_run_*_trace.py` had no `__main__` guard**, which on a spawn platform would have
  had every pool worker restart the whole pipeline recursively.
- The `ChainedAssignmentError` FutureWarning is suppressed by message, so stage-0
  `.err` logs stay readable without hiding every other warning.

Three more surfaced the first time the smoke test was SUBMITTED to a real grid rather
than run locally, which is the only way it is meant to be used on WRDS:

- **`run_smoke_test.sh` resolved the repo to SGE's spool directory.** `qsub` does not run
  the script where it sits -- it copies it into the spool tree and runs the copy -- so
  `dirname "$BASH_SOURCE"` gave `/gridware/sge/default/spool/<node>/job_scripts` and
  every path built from it pointed nowhere. The visible symptom was the run reporting
  five external inputs as missing while they sat in the repo. It now tries
  `SGE_O_WORKDIR`, then `PWD`, then the script's own directory, taking the first that
  actually contains `stage0/` and `stage1/`.
- **`#$ -o smoke/logs/smoke.out` on a fresh clone.** SGE opens the output file before
  running a line, and that directory does not exist until the script creates it, so a
  fresh clone went straight to `Eqw` having executed nothing. Output now goes to
  `smoke_test.out` in the repo root.
- **Every shell script was committed as `100644`**, so a fresh clone on Linux gets
  "Permission denied" from `./download_inputs.sh` or `./run_pipeline.sh`. Long-standing,
  and caused by `core.filemode=false` on the Windows clone these are authored from, where
  `chmod +x` never reaches a commit. Set explicitly with `git update-index --chmod=+x`;
  `run_pipeline.sh` also now invokes its sibling through `bash` so it does not care.

### Documentation
- Repository structure listings in `README.md` and `QUICKSTART.md` rebuilt against the
  actual file list. They had drifted: both showed `stage0/QUICKSTART_stage0.md` and
  `stage1/requirements.txt`, neither of which exists, and neither listed `config.py`,
  `FAQ.md`, `stage1/stage1_pipeline.py` or `stage1/DATA_DICTIONARY.md`.
- **`run_all_trace.sh` no longer exists but was still referenced 23 times** across the
  FAQ and both stage-0 guides. Replaced with `run_pipeline.sh` throughout.
- **The stage-0 quick start told you to set `WRDS_USERNAME` in `_trace_settings.py`.**
  That has not worked since the shared `config.py` was introduced -- `_trace_settings.py`
  imports the value from there, so editing it does nothing. Corrected, with the
  environment-variable route given first.
- `OSBAP_Linker_*.parquet` renamed to `bond_firm_linker_2026/` wherever it appeared;
  the linker changed in 2.1.0 and the docs had not followed.
- `ff12num` shipped in 2.1.0 but the headline feature lists still said "17 and 30".
- `CONTRIBUTING.md` documents the test suite and states the bar for any stage-0
  scheduling change: byte-identical parquet output against a banked reference.

### Verification
Concurrent output is byte-identical to serial, on both engines, with chunks completing
out of order and with several chunks per worker: all stage-0 parquet files match,
audit tables included, and the replayed filter logs match character for character.
The measured WRDS connection ceiling on the development account is 7 held
simultaneously (the 8th fails); the budget leaves one spare for a mid-run reconnect.

Confirmed live on the WRDS grid: five workers opened five connections inside one second
and ran five chunks in 18 s of wall clock against 81.3 s of serial work (4.5x), with
completions arriving out of order and all 28 smoke assertions still passing. The
`-pe onenode 5 -l m_mem_free=8G` request placed immediately.

---

## [2.1.0] - 2026-09-09

### Added
- **`ff12num`**: the Fama-French 12 industry classification, alongside the existing
  FF17 and FF30. Verified against an independent build across all 69,091 traded
  CUSIPs with zero differences -- as were `ff17num` and `ff30num`, proving they were
  not disturbed.
- **`PRICE_NORM`** (Stage 0, on by default): rescales unit-quoted bonds to percent of
  par. Small-denomination issues ($10, $25, $100 notes) are quoted in unit dollars,
  so a $10 note at par prints `10.00` rather than `100` -- and every filter
  downstream assumes percent of par, so those bonds read as deeply distressed with
  tenfold-understated volume. **This is a no-op under the default settings**, because
  `principal_amt_eq_1000_only` keeps only $1,000-principal bonds; it matters when you
  turn that screen off.
- **Auto-rolling `DATE_CUT_OFF`**: accepts `"auto:-Nmo"` (default `"auto:-3mo"`) and
  resolves against the data's last trade date, so the sample end tracks the data
  instead of needing an edit each vintage. A fixed `"YYYY-MM-DD"` still works.
- **`limit_chunks`** (Stage 0): process only the first N CUSIP chunks, so a config
  change can be checked in minutes rather than a ~4-hour run. Default `None`.
- **Fail-fast input validation**: `validate_config()` now checks for the files
  `run_pipeline.sh` downloads on the login node, and Stage 1 asserts that the
  treasury curve covers the panel before spending an hour on analytics.
- **`.gitattributes`**: pins the shell runners to LF endings. On a Windows clone they
  previously came down as CRLF, which `bash` on the WRDS cloud rejects.

### Changed
- **BREAKING for `permno` / `permco` / `gvkey` consumers.** The bond-firm linker is
  now bond-level and dated: one row per (9-character CUSIP, ownership window), rather
  than issuer-CUSIP-6 matched to a calendar month and forward-filled. Bonds are
  attributed to the firm that owned them *at the time* rather than to whichever firm
  owned them last. Measured against the previous linker on a 30.4M-row panel:
  3,424 bonds gain a link, 4,808 are relabelled, 1,424 are absent from the new
  linker, and 4,177 lose their identifiers outside the ownership window -- mostly
  bonds still trading after the firm's equity stopped being listed. Row-level
  coverage falls from 89.94% to 87.90%. That is the intended direction: a missing
  link is an answer, a stale one is a silent error.
  The release also ships `fl_verdicts.parquet` (every refusal and its reason) and
  `firm_names.parquet` (permno to a dated firm name).
- `gvkey` remains `Int32`; the source ships it zero-padded, so re-pad to 6 characters
  before joining to Compustat.
- `issuer_cusip` is no longer a join key. It was already absent from the output.

### Fixed
- **Step 5 I/O amplification.** Each chunk read the whole accumulated parquet back,
  concatenated and rewrote it -- roughly 25 GB of I/O to produce a 2.5 GB file, while
  holding three copies of the data inside a 24 GB job. Chunks now write part files
  that are concatenated once. Output is unchanged.
- **Worker over-subscription.** `N_CORES` defaulted to the host's core count while
  Grid Engine grants this job a single slot, and joblib copies data per worker. It
  now defaults to 4, honours `STAGE1_N_CORES` or `NSLOTS`, and is capped by the real
  core count. `calculate_credit_spreads` no longer falls back to a hard-coded 10.
  *No scheduler directive changed*: `m_mem_free` is a per-slot request here, so
  adding `-pe onenode N` would multiply the memory request N-fold.
- **Artifact stamp mismatch.** One run could emit `stage1_20251206.parquet` beside
  `sp_ratings_20251118.parquet`, and a run crossing midnight could stamp its own
  outputs with two different dates. Every artifact of a run now shares one stamp.
- Removed dtype casts for columns already dropped; aligned the ultra-distressed
  filter's defaults with the config the pipeline actually passes (documentation only
  -- the filter's behaviour is unchanged); escaped the last invalid escape sequence,
  so the repo compiles clean under `-W error::SyntaxWarning`.
- Documentation corrections: the data dictionary listed 7 columns that are not in the
  output and is now checked against the real schema; `bond_amt_outstanding` is in
  **thousands of dollars** (the README said millions, the dictionary said "bond
  units"); Stage 0's output files are `trace_<member>_YYYYMMDD.parquet`, not
  `<member>_YYYYMMDD.parquet`; and three claims in the 2.0.0 notes above did not
  match the code (no OAS is computed; the ratings come from Mergent FISD rather than
  an unnamed WRDS source; SIC codes come from FISD's issuer table, not from CRSP via
  PERMNO).

### Investigated -- no change
- **`dated_date` filtering** is retained. Of 17,845 FISD CUSIPs with no `dated_date`,
  only **5** actually trade in TRACE, so relaxing the screen would gain nothing.
- **`bond_amt_outstanding` scaling** is correct as-is: raw FISD `amount_outstanding`
  in $ thousands, with no rescaling anywhere in the pipeline.
- **Standard TRACE (`db_type=2`) never survives Stage 1.** Standard rows are kept only
  for dates after the last Enhanced date, and any trailing cutoff falls before that,
  so the two conditions cannot both hold. This was already true under the previous
  fixed cutoff -- which is why shipped Stage 1 files contain only db_type 1 and 3.

### Considered, not included
Order-flow measures (`qbuy`, `qsell`, `order_imbalance` and their 28-day trailing
sums), O'Hara-Zhou realized half-spreads, a quoted `bid_ask_bps`, ask-side symmetry
(`ask_last`, `ask_time_ew`, `ask_time_last`), and any change to Stage 0's serial
chunk loop.

---

## [2.0.0] - 2025-12-11

### Added - Stage 1 Release (Bond Analytics)

#### Core Features
- **Complete bond analytics pipeline** enriching Stage 0 daily data with comprehensive metrics
- **Automated orchestration** via `run_pipeline.sh` (handles both Stage 0 and Stage 1)
- **Research-ready output** with ~50+ variables per bond-day observation

#### Bond Analytics via QuantLib
- **Yield-to-maturity (YTM)** calculations using QuantLib bond pricing engine
- **Macaulay duration** and **modified duration** (interest rate sensitivity)
- **Convexity** (second-order price sensitivity)
- **Credit spreads** computed against Liu-Wu zero-coupon treasury yields
- **Robust error handling** for bonds with missing or invalid parameters
- **Efficient multi-core processing** with joblib parallelization

#### Credit Ratings Integration
- **S&P ratings** from Mergent FISD (`fisd.fisd_ratings`, `rating_type='SPR'`)
  - Numeric ratings (1-22 scale)
  - NAIC designations
- **Moody's ratings** from Mergent FISD (`fisd.fisd_ratings`, `rating_type='MR'`)
  - Numeric ratings (1-21 scale)
- **Automatic rating alignment** with bond-month observations

#### Equity Identifiers 
- **CRSP identifiers**: PERMNO and PERMCO
- **Compustat identifier**: GVKEY

#### Industry Classifications
- **Fama-French 17 industry classification**
- **Fama-French 30 industry classification**
- **SIC code mapping** from the issuer's SIC code in Mergent FISD

#### Ultra-Distressed Bond Filters
- **Price anomaly detection** to identify suspicious observations
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
- **LaTeX data quality reports** 
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
- **Stage 1 processing time**: ~3 hours (WRDS Cloud, 2-4 cores)
- **Complete pipeline (Stage 0 + Stage 1)**: ~7-10 hours total
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

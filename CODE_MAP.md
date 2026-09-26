# Code map

What each stage reads and writes, and what every code file does. For "where do I look to do X",
see [INDEX.md](INDEX.md). For which file makes each table and figure of the paper, see
[stage3/INDEX.md](stage3/INDEX.md).

`tests/test_docs.py` fails if a tracked `.py` or `.sh` file is missing from this page, so the
map cannot fall behind the code. Exempt: files inside `tests/` folders (described per folder
below) and package `__init__` files; stage 3's drivers may be listed in `stage3/INDEX.md` instead. The check
needs git, and is skipped without it.

## How the data moves

| stage | runs on | reads | writes | settings |
|---|---|---|---|---|
| 0 | the WRDS grid | WRDS: `trace.trace_enhanced`, `trace.trace_btds144a`, `trace.trace` (Standard, opt-in), `fisd.fisd_mergedissue`, `fisd.fisd_mergedissuer` | `stage0/<member>/trace_<member>_<stamp>.parquet` (the daily panel), the member's FISD file (`trace_enhanced_fisd_<stamp>.parquet`, `trace_fisd_144a_<stamp>.parquet`), the filter audit files, `stage0/data_reports/` | `config.py`, `stage0/_trace_settings.py` |
| 1 | the WRDS grid, after stage 0 | the stage 0 panels and Enhanced FISD file; WRDS: `fisd.fisd_ratings`, `fisd.fisd_mergedissue`, `fisd.fisd_mergedredemption`, `fisd.fisd_amt_out_hist`; the files `download_inputs.sh` fetched | `stage1/data/stage1_<stamp>.parquet` (44 columns), `call_dummy_<stamp>.parquet`, `sp_ratings_<stamp>.parquet`, `moodys_ratings_<stamp>.parquet`, `ultra_distressed_cusips_<stamp>.csv`, `stage1/data_reports/` | `config.py`, `stage1/_stage1_settings.py` |
| 2 | your own computer | `stage1_<stamp>.parquet`, `trace_enhanced_fisd_<stamp>.parquet`, `call_dummy_<stamp>.parquet`; on the first run, WRDS (`crsp.tfz_idx`, `crsp.tfz_mth_ft`, `ff.factors_monthly`, `ff.fivefactors_monthly`, `cboe.cboe`, `fisd.fisd_mergedissue`), read with `WRDS_USERNAME` (`config.py` or the environment), and public downloads, all cached in `stage2/data/` | `stage2/output/panel/main_panel_<mode>.parquet` (145 columns), `stage2/output/blocks/<mode>/`, `stage2/manifests/`, and with `make_release.py` the shareable `stage2/release/` | `stage2/_stage2_settings.py` |
| 3 | your own computer | the stage 2 panel, its blocks and factor file, and the stage 1 daily panel | `stage3/data/` (sort results, grids, statistics), `stage3/reports/tables/`, `stage3/reports/figures/`, `stage3/reports/exhibits.pdf`, `stage3/reports/timings.jsonl` | `stage3/_stage3_settings.py` |
| 4 | your own computer | the stage 2 panel; its blocks `betas_x`, `mom_retx`, `betas_bns`, `mom_retx_bns`, `betas_cls`, `mom_retx_cls`; the risk-free rate in its factor file; for the check, the published TRACE-only factors (downloaded) | `stage4/output/<sort>_sort_panel_trace_<vintage>/` (the long factor panel, `flip_set.json`, `MANIFEST.json`) and `stage4/output/<sort>_sort_trace_<vintage>_csv/` (24 CSV files) | `stage4/_stage4_settings.py`, `stage4/spec/factors.json` |

`<stamp>` is the run date, `YYYYMMDD`. `<member>` is `enhanced`, `144a` or `standard`. `<mode>`
is `stage1` unless you set another. `<sort>` is `single` or `within_firm`; `<vintage>` is the
release year, taken from the stage 1 file's stamp.

**Downloads.** Stage 1's come from `download_inputs.sh`, run on the WRDS login node: the Liu-Wu
Treasury yields, the bond-firm linker, and the Fama-French industry files. Stage 2 fetches its
own the first time it needs them. The URLs are in `stage2/_stage2_settings.py` and
`stage2/lib/duration_adjusted.py`, and `stage2/check_external_data.py` lists every source with
the last month each cache holds.

## Top level

| file | what it does |
|---|---|
| `config.py` | Settings shared by stages 0-2: `WRDS_USERNAME`, `AUTHOR`, `TRACE_MEMBERS`, `OUTPUT_FORMAT`, `STAGE0_OUTPUT_FIGURES` |
| `requirements.txt` | The Python packages for stages 0 and 1, installed on WRDS |
| `requirements-local.txt` | The packages for stages 2-4, on your own computer: `requirements.txt` plus numba. PyBondLab goes in on a second line, as the file explains |
| `constraints-2026.txt` | Optional: the exact package versions the published 2026 vintage was built with, for a build that matches it to the last digit |
| `pybondlab_pin.py` | The PyBondLab release stages 2-4 run on, the start-up check that it is the one installed, and its fingerprint for manifests |
| `numeric_setup.py` | Makes pandas use `numexpr` (required: it changes float32 results, and the published panels were built with it) and not `bottleneck`, so stages 2-4 compute the same numbers on any machine |
| `run_pipeline.sh` | Runs stages 0 and 1 on WRDS: downloads the inputs, submits the stage 0 jobs, then the report job and stage 1, each waiting on the jobs it needs |
| `download_inputs.sh` | Fetches the files stage 1 needs from the internet. Login node only: WRDS compute nodes have no internet |
| `check_disk_space.sh` | Checks there is room in your WRDS home quota before a run. Called by `run_pipeline.sh` |
| `run_smoke_test.sh` | Runs the real stage 0 and stage 1 code on a few CUSIP chunks and checks the results, in about 10 minutes. Its data goes under `smoke/`, its log to `smoke_test.out` and `smoke_test.err` |

## Stage 0: raw TRACE to daily panels (WRDS)

| file | what it does |
|---|---|
| `stage0/_trace_settings.py` | Stage 0 settings: the filters, the FISD screens, the date range, `CONCURRENCY` (WRDS connections per member) and the job sizes |
| `stage0/run_enhanced_trace.sh` | SGE job script for Enhanced TRACE |
| `stage0/run_144a_trace.sh` | SGE job script for Rule 144A |
| `stage0/run_standard_trace.sh` | SGE job script for Standard TRACE (opt-in) |
| `stage0/run_build_data_reports.sh` | SGE job script for the stage 0 data-quality reports |
| `stage0/_run_enhanced_trace.py` | Entry point for Enhanced: reads its settings and runs the cleaner |
| `stage0/_run_144a_trace.py` | Entry point for Rule 144A |
| `stage0/_run_standard_trace.py` | Entry point for Standard TRACE |
| `stage0/create_daily_enhanced_trace.py` | The Enhanced cleaner: pulls trades from WRDS, applies every filter and correction, and writes the daily panel and its audit files |
| `stage0/create_daily_standard_trace.py` | The same for Standard TRACE and Rule 144A |
| `stage0/_chunk_runner.py` | Splits the CUSIP universe into chunks and runs several at once |
| `stage0/_wrds_pool.py` | Opens one WRDS connection per worker process, safely |
| `stage0/_build_error_files.py` | Builds the stage 0 data-quality reports (the LaTeX and its figure PDFs; you compile the report) |
| `stage0/_error_plot_helpers.py` | Plotting and filter helpers for those reports |

## Stage 1: daily bond analytics (WRDS)

| file | what it does |
|---|---|
| `stage1/_stage1_settings.py` | Stage 1 settings: inputs, the sample end (`DATE_CUT_OFF`), the distressed-bond filter, the linker |
| `stage1/run_stage1.sh` | SGE job script for stage 1 |
| `stage1/_run_stage1.py` | Entry point, called by `run_stage1.sh` |
| `stage1/create_daily_stage1.py` | Wires the settings into the pipeline and runs it |
| `stage1/stage1_pipeline.py` | The pipeline: loads the stage 0 panels, adds FISD characteristics, QuantLib analytics, ratings, firm ids and industries, applies the distressed-bond filter, writes `stage1_<stamp>.parquet` |
| `stage1/helper_functions.py` | The functions the pipeline calls, including the bond analytics |
| `stage1/_linker_join.py` | Attaches `permno`, `permco` and `gvkey` from the bond-firm linker, on its identity window |
| `stage1/_distressed_plot_helpers.py` | Figures and the LaTeX report for the distressed-bond filter |

## Stage 2: the monthly panel (your computer)

| file | what it does |
|---|---|
| `stage2/_run_stage2.py` | Entry point: checks the settings, prints them, then runs `build_panel.py` |
| `stage2/run_stage2.sh` | Shell wrapper for `_run_stage2.py`. Not a WRDS job |
| `stage2/_stage2_settings.py` | Stage 2 settings and the input checks: inputs, the factor source, workers, the published-factor URLs |
| `stage2/build_panel.py` | The build: the factor series, then the seven steps in fresh processes (steps 1 and 2 in turn, steps 3-4 and 5-6 side by side, then step 7) |
| `stage2/db.py` | DuckDB access: every query reads parquet files; there is no database file |
| `stage2/validate_coverage.py` | Checks that every column reaches the panel's last month, and names the ones whose source stops early |
| `stage2/validate_stage2.py` | Compares a build's step outputs with a reference build's, column by column, within tolerances |
| `stage2/check_external_data.py` | Lists every external series stage 2 uses, where it comes from, and the last month its cache holds |
| `stage2/make_excess_blocks.py` | Optional: re-estimates the 68 beta and momentum columns on the two other Treasury benchmarks |
| `stage2/make_release.py` | Packages a build for sharing: removes the licensed values and refuses to write a file that still has them |
| `stage2/_build_data_report.py` | Builds the stage 2 data report (LaTeX, and the PDF when TeX is installed) |
| `stage2/_report_helpers.py` | Tables, figures and LaTeX for that report |
| `stage2/run_build_data_reports.sh` | Shell wrapper for the data report |
| `stage2/conftest.py` | Lets the tests import the stage 2 modules |
| `stage2/steps/compute_factors.py` | The factor time series, from the public sources or the published file (`--factor-source`) |
| `stage2/steps/step1_returns.py` | Step 1: monthly returns, month-end signals, default handling, duration-matched Treasury returns |
| `stage2/steps/step2_illiquidity.py` | Step 2: illiquidity and risk signals per bond, and the market illiquidity factors |
| `stage2/steps/step3_bbw.py` | Step 3: the Bai-Bali-Wen bond factors, from double sorts through PyBondLab |
| `stage2/steps/step4_betas.py` | Step 4: rolling 36-month betas on every factor model, and systematic momentum |
| `stage2/steps/step5_value.py` | Step 5: the value signals and the spread-momentum signals |
| `stage2/steps/step6_momentum.py` | Step 6: momentum, long-term reversal, industry momentum, VaR and expected shortfall |
| `stage2/steps/step7_final.py` | Step 7: joins everything into the 145-column panel and the `_mmn` sidecar |
| `stage2/lib/contract.py` | The 145 column names and their order. A build that changes either fails |
| `stage2/lib/betas.py` | The rolling-beta models (`BETA_MODELS`) and the code that runs them |
| `stage2/lib/rolling_kernels.py` | The numba kernels behind the betas, momentum and skewness |
| `stage2/lib/momentum.py` | Momentum and long-term reversal |
| `stage2/lib/value.py` | The value signals |
| `stage2/lib/var_es.py` | Rolling value-at-risk and expected shortfall |
| `stage2/lib/illiq_pandas.py` | The four illiquidity measures that must be computed in pandas to match the reference numbers |
| `stage2/lib/wrangle.py` | The final merge and column order |
| `stage2/lib/pin.py` | The daily panel projection every step reads, built once |
| `stage2/lib/treasury.py` | Duration-matched Treasury returns (`tret`), from CRSP via WRDS, fetched once and cached |
| `stage2/lib/duration_adjusted.py` | The five Treasury benchmarks built from each bond's own cash flows (`tret_bns`, `tret_cls`, ...) |
| `stage2/lib/ff5.py` | Fama-French five factors from WRDS, fetched once and cached |
| `stage2/lib/vix.py` | Daily VIX from WRDS, fetched once and cached |
| `stage2/lib/factor_fetch.py` | The sources for `--factor-source public` (Ken French, FRED, He-Kelly-Manela, Ludvigson, EPU, and monthly VIX from WRDS), fetched once and cached |
| `stage2/lib/extended_factors.py` | The BBW factors before TRACE starts (1973 to 2002-07), downloaded and cached |
| `stage2/lib/quote.py` | The 1997-2002 quote-return panel, downloaded and cached |
| `stage2/lib/linker.py` | The bond-firm linker and the window the panel joins it on |
| `stage2/lib/frontier.py` | Refuses to publish a last month that holds too few bonds to be a real cross-section |
| `stage2/lib/nyse_calendar.py` | The NYSE trading calendar |
| `stage2/lib/month_boundaries.py` | Each month's first and last trading days, from that calendar |
| `stage2/lib/manifest.py` | Writes the run manifest: inputs and their hashes, settings, timings, checks |
| `stage2/lib/validate_core.py` | The column-by-column comparison `validate_stage2.py` uses |
| `stage2/lib/phase_timer.py` | Times each phase of a step |
| `stage2/reference/bbw_factors_original_2004_2021.csv` | The BBW authors' original factor series, shipped in the BBW download |

## Stage 3: the paper's exhibits (your computer)

Stage 3's drivers, one per table or figure, are listed with the exhibit they make in
[stage3/INDEX.md](stage3/INDEX.md). The folders are numbered in the order they were built, not by
paper section: `s1_lib` is the paper's Section 3, `s2_lab` Section 4, `s3_nse` Section 5.

| file | what it does |
|---|---|
| `stage3/run_stage3.sh` | Runs everything: the input check, then `_run_stage3.py` |
| `stage3/_run_stage3.py` | Entry point: section by section, the producers (sorts and grids) then their exhibits, then the PDF |
| `stage3/_stage3_settings.py` | Every path and constant. Inputs can be moved with environment variables |
| `stage3/paths.py` | The paths, derived from the settings |
| `stage3/pblenv.py` | Checks the installed PyBondLab against `pybondlab_pin.py` before any sort, and records its version and content hash with every result |
| `stage3/drrlib.py` | Shared loading, statistics (Newey-West, CAPM_B alpha, paired differences) and result manifests |
| `stage3/fastrun.py` | Runs the grids one signal per fresh process |
| `stage3/bench.py` | Times each run and records its own checks in `reports/timings.jsonl` |
| `stage3/captions.py` | Every table caption |
| `stage3/latex_format.py` | Number formatting for the LaTeX tables |
| `stage3/helper_functions.py` | Small shared conventions |
| `stage3/make_report.py` | Puts every table and figure into `reports/exhibits.pdf` |
| `stage3/s0_data/data_engine.py` | Statistics for the descriptive tables of the daily and monthly panels (Tables IA.I to IA.VII) |
| `stage3/s1_lib/lib_engine.py` | Statistics for every Section 3 exhibit |
| `stage3/s1_lib/two_row.py` | The table layout shared by the Section 3 tables |
| `stage3/s2_lab/lab_engine.py` | Statistics for every Section 4 exhibit |
| `stage3/s2_lab/lab_exhibit.py` | The table layout shared by the Section 4 tables |
| `stage3/s3_nse/nse_engine.py` | Statistics for every Section 5 exhibit |
| `stage3/s3_nse/mua_engines.py` | The method-uncertainty grid: its 216 specifications, through PyBondLab's `assay_anomaly_fast` |
| `stage3/s3_nse/clusters.py` | The 108 signals' nine clusters, and the grid definitions |
| `stage3/s3_nse/cluster_table.py` | The table layout shared by Tables 5 and 6 |
| `stage3/s4_zoo/zoo_engine.py` | Statistics for the factor-zoo exhibits |
| `stage3/s4_zoo/zoo_frames.py` | The four factor-zoo statistics tables, computed once |
| `stage3/tools/check_inputs.py` | Checks the five inputs before a long run |
| `stage3/tools/build_index.py` | Writes `stage3/INDEX.md` from the code; `--check` fails if it is out of date |
| `stage3/spec/inputs.json` | The five inputs and their expected shape |
| `stage3/spec/signal_definitions.json` | What each of the 145 panel columns means, with its citation |

## Stage 4: the TRACE-only bond factors (your computer)

| file | what it does |
|---|---|
| `stage4/run_stage4.sh` | Runs everything: `build_factors.py`, then `compare_published.py` |
| `stage4/build_factors.py` | Entry point: loads each return type's panel once, sorts it both ways over the three rating bands, and writes the factors, the flip set and a manifest |
| `stage4/compare_published.py` | Downloads the TRACE-only factors openbondassetpricing.com serves and compares them with yours, cell by cell |
| `stage4/_stage4_settings.py` | The paths (movable with environment variables) and the release year |
| `stage4/spec/factors.json` | The grid: the 108 signals, the four return types, the rating bands, the portfolio counts and the 68 columns a duration-adjusted return type swaps |
| `stage4/factorlib/inputs.py` | The panel for each return type: excess of the T-bill, or duration-adjusted with its own betas and momentum |
| `stage4/factorlib/sorts.py` | One return type through PyBondLab for every band, unflipped, and the flip set |
| `stage4/factorlib/release.py` | Writes the files in the published archives' layout, and the wide CSVs |
| `stage4/conftest.py` | Lets the tests import the stage 4 modules |

## Tests

| folder | what it covers |
|---|---|
| `tests/` | The whole repo: stage 0 chunking and scheduling, the disk-space check, one row per key on every lookup, the stage 1 linker window and sample-end rule, no private paths in tracked files, the docs against the code (`test_docs.py`), the smoke test's checks (`smoke_assertions.py`), and a probe of your WRDS connection limit (`probe_wrds_connections.py`, needs WRDS) |
| `stage2/tests/` | Stage 2: the column contract and order, the `_mmn` twin rule, the release redaction, the factor sources, the calendar and month boundaries, the data dictionary against the code, and parity with a reference build (skipped when there is none) |
| `stage3/tests/` | Stage 3: the input contract, the runner and its steps, the exhibit index, no absolute or private paths, the Section 5 counting rules, the signal definitions, and the skip-and-rebuild rule |
| `stage4/tests/` | Stage 4, on small synthetic panels: the grid in the spec, the duration swap, the flip set, the CSV pivot, and one run through PyBondLab |

Run `python -m pytest stage2/tests tests stage3/tests stage4/tests -q`. Nothing in it needs WRDS or the network.

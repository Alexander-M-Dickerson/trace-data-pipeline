# Stage 1 with an AI assistant

Stage 1 runs on the WRDS grid after stage 0. It combines stage 0's daily panels into one,
prices every bond-day with QuantLib, adds ratings, amounts outstanding, industries and firm
identifiers, filters the sample, and writes the 44-column daily bond panel that stage 2 reads.
Read the repository's [AGENTS.md](../AGENTS.md) first; the full guide is
[README_stage1.md](README_stage1.md), and every column is defined in
[DATA_DICTIONARY.md](DATA_DICTIONARY.md).

The `run-wrds` skill (`/run-wrds` in Claude Code, `$run-wrds` in Codex) walks stages 0 and 1
step by step, and `python3 doctor.py --wrds`, from the repository root, says whether the login
node is ready.

## Before it runs

- **Its external inputs come from `download_inputs.sh`, run on the login node**, because
  compute nodes have no internet [ref:trap.no_internet]: the Liu-Wu Treasury yields, the
  bond-firm linker and the Fama-French industry files. `bash download_inputs.sh --check` says
  whether they are there without downloading anything. `run_pipeline.sh` fetches them itself.
- Stage 1 opens its first WRDS connection at step 6 (ratings), well into the run, so a missing
  `WRDS_USERNAME` surfaces late. `python doctor.py --wrds` checks it first.

## What it does, in order

The job script `run_stage1.sh` runs `_run_stage1.py`, which hands the settings to
`create_daily_stage1.py`, which runs these steps in `stage1_pipeline.py`.

1. **Treasury yields**, for the credit spreads.
2. **Stage 0's panels**, combined, one row per bond and day [ref:daily.db_type], cut at the
   sample end `DATE_CUT_OFF` [ref:rule.complete_months].
3. and 4. **FISD terms**, merged by CUSIP; bond age and maturity [ref:daily.bond_maturity].
   Rows QuantLib cannot price are set aside here [ref:filter.accrued_inputs].
5. **Analytics with QuantLib**: yield, durations, convexity, accrued interest
   [ref:daily.ytm], then the credit spread over the Treasury curve [ref:daily.credit_spread].
6. **Ratings, amount outstanding and the call flag**, from WRDS, each as of the trade date
   [ref:daily.sp_rating] [ref:daily.spc_rating] [ref:daily.bond_amt_outstanding].
7. **Firm identifiers** from the bond-firm linker, on its identity window
   [ref:daily.permno] [ref:rule.linker_window]. Stage 2 joins the same window.
8. **The ultra-distressed filter** flags probable price errors among very low-priced bonds
   [ref:filter.ultra_distressed_flags]; [README_distressed_filter.md](README_distressed_filter.md)
   explains it.
9. and 10a. **The sample filters**, applied in this order: rows QuantLib cannot price, rows with
   no rating [ref:filter.rating_present], bond-days within a year of maturity
   [ref:filter.maturity_one_year], the distressed flags [ref:filter.ultra_distressed], a first
   price change in July 2002 above 35 points [ref:filter.july_2002] and prices above 300
   [ref:filter.price_above_300]. Then `ytm` and `credit_spread` are clipped within each date
   [ref:filter.winsorize]. "Sample-defining operations" in the data dictionary gives the share
   each removed in the last run.
10. **The data report**, in `stage1/data_reports/`.

## Settings

All in `_stage1_settings.py`: `DATE_CUT_OFF` (default `auto:complete`, the last month every
source covers in full), `LINKER_WINDOW`, `FINAL_FILTER_CONFIG` (the two price thresholds) and
the ultra-distressed filter's parameters. `TRACE_MEMBERS` and `WRDS_USERNAME` come from the root
`config.py`.

## Traps

- **A finished stage 1 leaves about a hundred `ValueError` tracebacks in `stage1.err`**, from
  Python's `resource_tracker` at shutdown. They are harmless ([FAQ.md](../FAQ.md) says why).
  The run succeeded when the last lines of `stage1/logs/stage1.out` say "Stage 1 processing
  completed successfully".
- **Standard TRACE is not run by default**, and when it is, it can move the sample's end.
  Stage 1 keeps Standard only for dates after the last Enhanced date, and the automatic
  cut-off counts Standard with Enhanced (`CUT_OFF_POPULATIONS`), so adding it to
  `TRACE_MEMBERS` can carry `auto:complete` past the last Enhanced month.
- **The two linker windows are not interchangeable.** Stages 1 and 2 label each bond's issuer
  with the identity window; the evidence window is for joining equity data. Change
  `LINKER_WINDOW` in both settings files or in neither [ref:rule.linker_window].
- **The winsorization is per date**, so a small test sample does not reproduce a full run's
  `ytm` and `credit_spread` at the tails.

## What it writes

`stage1/data/stage1_<YYYYMMDD>.parquet` (the daily panel), `call_dummy_<YYYYMMDD>.parquet`,
`sp_ratings_<YYYYMMDD>.parquet`, `moodys_ratings_<YYYYMMDD>.parquet`,
`ultra_distressed_cusips_<YYYYMMDD>.csv` and `stage1/data_reports/`. Stage 2 needs the first two,
with stage 0's Enhanced FISD file.

## When something fails

Read the job's log in `stage1/logs/`, then "Troubleshooting" in
[README_stage1.md](README_stage1.md#troubleshooting) and the [FAQ](../FAQ.md#troubleshooting).

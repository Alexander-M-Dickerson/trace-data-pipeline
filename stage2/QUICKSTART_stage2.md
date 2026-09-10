# Quick Start (Stage 2) — Monthly Asset-Pricing Panel

The shortest path from a Stage 1 daily panel to a 140-column monthly bond panel.

Stage 2 runs on **your own machine**, not the WRDS grid. It reads Stage 1's output rather
than the TRACE tape, so there is no job to submit and no queue to wait in. Measured on 24
cores and the full 2002-2025 sample, the seven steps take 125 s, 111 s, 31 s, 68 s, 53 s,
13 s and 61 s — under eight minutes in total, and less than that in practice because the
orchestrator overlaps two of the chains.

---

## Prerequisites

**1. Stage 0 and Stage 1 output, produced by you.**

| file | from | why |
|---|---|---|
| `stage1/data/stage1_<YYYYMMDD>.parquet` | Stage 1 | the daily bond panel Stage 2 aggregates |
| `stage0/enhanced/trace_enhanced_fisd_<YYYYMMDD>.parquet` | Stage 0 | 144A flag, country, SIC, offering amount |
| `stage1/data/call_dummy_<YYYYMMDD>.parquet` | Stage 1 | callable flag |

All three are found automatically — newest date stamp wins.

> ❗**The OSBAP download will not work here.** The public Stage 1 file has its rating
> columns removed (they are proprietary), and Stage 2 selects only bond-months with at
> least one rating. Fed the download, it would produce an **empty panel with no error**.
> Stage 2 checks for this and refuses to start. Run Stage 0 and Stage 1 yourself.

**2. A WRDS account.** Stage 2 pulls Treasury returns, the Fama-French factors and VIX.
These are fetched once and cached under `stage2/data/`.

**3. Python packages.**

```bash
python3 -m pip install -r requirements.txt
```

DuckDB is the engine and `PyBondLab==0.2.0` builds the bond factors — that version is
pinned deliberately, see the comment in `requirements.txt`.

---

## Step-by-step

### 1. Check the configuration

```bash
cd stage2
python3 _run_stage2.py --dry-run
```

This resolves every input, prints what it found, and exits. If a file is missing it says
which one and where it looked. Nothing is built.

Set your WRDS username first if it is not already in the environment:

```bash
export WRDS_USERNAME=your_username        # macOS/Linux
setx WRDS_USERNAME your_username          # Windows, then reopen the shell
```

### 2. Build

```bash
python3 _run_stage2.py
```

or, equivalently, `bash run_stage2.sh`.

Seven steps run in sequence, each in a fresh process:

| step | what it does |
|---|---|
| 1 | month-end returns, prices, ratings, firm identifiers |
| 2 | illiquidity measures and the illiquidity factors |
| 3 | the BBW bond factors (5×5 double sorts) |
| 4 | the factor matrix and 37 models of rolling betas |
| 5 | value signals and spread momentum |
| 6 | momentum, long-term reversal, VaR and expected shortfall |
| 7 | the final merge into `main_panel_<mode>.parquet` |

Useful flags:

```bash
python3 _run_stage2.py --limit-cusips 200      # a fast smoke build
python3 _run_stage2.py --from-step 4           # resume after a failure
python3 _run_stage2.py --factor-source pinned  # reproduce a published vintage exactly
python3 _run_stage2.py --validate              # run the validation sweep afterwards
```

### 3. Check what you built

```bash
python3 validate_coverage.py
```

Every column should reach within a month of the panel's last date. Three are expected to
lag, and the gate says so by name: `b_cptlt` (He-Kelly-Manela have not published past
2025-05) and `b_dcpi` / `b_cpi_vol6` (FRED's CPIAUCSL is missing an observation). Those are
upstream limits, not build failures.

### 4. Build the data report (optional)

```bash
bash run_build_data_reports.sh
bash run_build_data_reports.sh --no-external    # skip the comparisons, no network needed
```

14 tables and 11 figures describing coverage, distributions, extreme returns and how the
panel compares with the DFPS and WRDS bond databases. Roughly 2½ minutes with the
comparison suites, 1 minute without. The PDF is built if `pdflatex` is installed; without
it you still get the `.tex`.

---

## What you end up with

```
stage2/output/panel/main_panel_<mode>.parquet     the 140-column panel
stage2/output/blocks/<mode>/                      intermediate blocks, incl.
                                                    betas_x, mom_retx, returns_alt,
                                                    factors, and the _mmn sidecar
stage2/data_reports/                              the report, its figures and PDF
```

Every column is defined in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md). The column names
**and their order** are frozen in `lib/contract.py` and checked at the end of every build,
so a panel that builds is a panel you can read positionally.

---

## If something goes wrong

| symptom | cause |
|---|---|
| "the OSBAP download cannot drive Stage 2" | you pointed it at the public Stage 1 file; ratings are stripped from it |
| a missing-input box at start-up | Stage 0/1 output is absent or has a different date stamp; `--dry-run` shows where it looked |
| `FileNotFoundError` on a factor | first run with no cache and no internet; the fetchers need one online run |
| WRDS asks for a password on every step | no `.pgpass`; create one, or run `wrds.Connection()` once interactively |
| the build asserts on the column contract | something changed the panel's columns; the message names them — see `lib/contract.py` |
| step 2 is much slower than 25 s | you are running steps in one process; use `_run_stage2.py`, which forks per step |

More detail in [`README_stage2.md`](README_stage2.md).

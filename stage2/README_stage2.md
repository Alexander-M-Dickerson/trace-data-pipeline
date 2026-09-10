# Stage 2 — Monthly Asset-Pricing Panel

Stage 2 turns Stage 1's daily bond-day panel into the **monthly asset-pricing panel**:
returns, bond characteristics, ~100 signals, factor time series, and rolling factor betas.

> **Status:** under construction. The configuration layer (`_stage2_settings.py`,
> `_run_stage2.py`, `run_stage2.sh`) is in place and `--dry-run` works today. The build
> engine lands next.

---

## Contents

- [Where Stage 2 runs](#where-stage-2-runs)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [What Stage 2 produces](#what-stage-2-produces)
- [External data](#external-data)
- [Troubleshooting](#troubleshooting)
- [Support](#support)

---

## Where Stage 2 runs

❗**On your own machine, not on WRDS.** Stages 0 and 1 are grid jobs; Stage 2 is not, and
must **never** be submitted with `qsub`. Two reasons:

1. It needs more memory than a WRDS grid slot allows — the panel is worked on whole.
2. It opens no TRACE database connection at all. Everything it reads is a parquet file
   Stage 0 and Stage 1 already wrote.

`run_pipeline.sh` therefore does not submit it. When your WRDS run finishes, download
`stage0/` and `stage1/` to your own machine and run Stage 2 there.

A full build is roughly **7 minutes on 24 cores / 128 GB**. It is comfortable on a
modern desktop and unhappy on a laptop with 16 GB.

---

## Prerequisites

Your local tree must look like this:

```
trace-data-pipeline/
├── config.py
├── stage0/
│   └── enhanced/
│       └── trace_enhanced_fisd_YYYYMMDD.parquet     # bond characteristics
├── stage1/
│   └── data/
│       ├── stage1_YYYYMMDD.parquet                  # THE daily panel
│       └── call_dummy_YYYYMMDD.parquet              # callable flags
└── stage2/
```

❗**The Stage 1 file published for download will not work.** It has the agency ratings
blanked for licensing reasons, and Stage 2 keeps only bond-days carrying at least one
agency rating — so that file yields an empty panel. Stage 2 detects this and refuses to
start. Run Stages 0 and 1 yourself on WRDS and use their output.

Python packages beyond Stage 1's: `duckdb`, `numba`, `scipy`, `PyBondLab`,
`pandas_market_calendars`, `openpyxl`.

---

## Quick start

```bash
cd stage2
./run_stage2.sh --dry-run     # resolve and check everything, build nothing
./run_stage2.sh               # the full build
```

`--dry-run` prints the resolved configuration and validates every input, reporting all
problems at once. Run it first — it is fast and it catches a wrong tree before you spend
seven minutes discovering it.

Useful flags (passed straight through to `_run_stage2.py`):

| Flag | Effect |
|---|---|
| `--dry-run` | Validate the configuration and exit |
| `--limit-cusips 200` | Build on the first 200 CUSIPs — a fast development loop |
| `--from-step N --to-step M` | Run part of the pipeline (steps 1-7) |
| `--factor-source pinned` | Use a pre-built factor file instead of fetching public sources |
| `--validate` | Run the validation sweep after building |

---

## Configuration

Everything lives in `_stage2_settings.py`. The values you are most likely to touch:

| Setting | Default | Meaning |
|---|---|---|
| `ROOT_PATH` | auto-detect | The folder holding `stage0/`, `stage1/`, `stage2/` |
| `DAILY_INPUT` | `None` | Pin a specific Stage 1 panel; otherwise the newest is used |
| `FACTOR_SOURCE` | `"public"` | Build the factor matrix from public sources |
| `WORKERS` × `THREADS_PER` | 6 × 2 | Keep the product at or below your core count |
| `START_DATE` | `2002-07-31` | First month-end in the panel |
| `BETA_WINDOW` / `BETA_MIN_OBS` | 36 / 12 | Rolling beta window |
| `DEF_CORP_MIN_MATURITY` / `DEF_GOVT_TENOR` | 10y / 20y | The default-premium factor's two legs |

The processing parameters below those are carried over verbatim from the reference
implementation and define the published panel. Changing one changes your results — which
is fine, as long as it is deliberate.

`STAGE2_DAILY_INPUT` overrides the input path from the environment, which is handy for
testing without editing the file.

---

## What Stage 2 produces

| Output | Contents |
|---|---|
| `output/panel/main_panel_<mode>.parquet` | The main panel — 140 columns, MMN-adjusted signals, used with `ret_vw` |
| `output/blocks/<mode>/mmn_price_based_signals_<stamp>.parquet` | The 38 unadjusted twins (`*_mmn`), used with `ret_vw_bgn` |
| `output/blocks/<mode>/` | Per-step intermediates: `betas_std`, `betas_x`, `mom_ret`, `mom_retx`, `returns_alt`, `factors`, `factors_merged`, `bbw_factors`, `illiq_factors` |
| `data_reports/` | The LaTeX data report, its figures and the PDF |
| `manifests/` | A JSON run manifest per build: inputs, hashes, config, timings |
| `release/` | What `make_release.py` packages for publication |

`<mode>` is the build label (`stage1` by default); released files are renamed to the
vintage year, e.g. `main_panel_2026.parquet`. `<stamp>` is Stage 1's date stamp.

**The column contract.** The panel's 140 names **and their order** are frozen in
[`lib/contract.py`](lib/contract.py) and asserted at the end of step 7. Adding, removing or
moving a column is a public API change and a CHANGELOG entry — the build fails rather than
shipping a silently permuted file. This caught a real reordering: the DEF/TERM fix swapped
`b_defb` and `b_termb` before anyone noticed.

**Your panel is complete.** `permco`, `gvkey` and the raw 1-22 agency ratings are all
present in what you build. The copies published on openbondassetpricing.com have those
redacted, because they can be downloaded by people who hold no licence for them — but that
redaction happens in `make_release.py`, which you only run if you are publishing a vintage
for others. Nothing in the build touches those columns.

**The data report.**

```bash
bash run_build_data_reports.sh                 # 14 tables, 11 figures, PDF
bash run_build_data_reports.sh --no-external   # skip the DFPS/WRDS comparisons
```

Roughly 2½ minutes with both comparison suites, 1 minute without.

Every column is defined in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md). Two companion
notes cover the trickier methodology: [`README_Default.md`](README_Default.md) for
defaulted-bond returns, and the market-microstructure section of the dictionary for the
adjusted / unadjusted signal split.

❗**The two-panel rule.** Use the main panel's signals with `ret_vw`, or the `_mmn`
signals with `ret_vw_bgn` — never mix them. Mixing reintroduces the microstructure bias
the split exists to remove.

---

## External data

Stage 2 downloads and caches a few published inputs into `stage2/data/` on first use:

| Input | Why |
|---|---|
| Pre-2002 quote returns | Lets rolling signals reach a common 2002-08 start |
| Extended BBW factor series | Backfills factor history before 2002-08 |
| Fama-French, FRED, He-Kelly-Manela, Ludvigson, Policy Uncertainty | The monthly factor matrix |

It also fetches Treasury returns, Fama-French factors and VIX from WRDS **once** and
caches them, so only the first run needs your credentials.

---

## Troubleshooting

**"Stage 1 directory not found"** — you are running from the wrong place, or you have not
downloaded `stage0/` and `stage1/` yet. Run `./run_stage2.sh --dry-run` to see every path
Stage 2 resolved.

**"Every agency rating column is empty"** — you are pointing at the published Stage 1
download rather than your own run. See [Prerequisites](#prerequisites).

**"is missing N column(s) Stage 2 needs"** — the Stage 1 panel is from an older release.
Re-run Stage 1 with the current code.

**Out of memory** — lower `WORKERS`, or build in pieces with `--from-step` / `--to-step`.
Each step writes its blocks to disk, so a build resumes cheaply.

**A public factor URL has moved** — vendors rotate filenames. The fetcher recovers from
this for some sources; for others, update the URL in `_stage2_settings.py`.

---

## Support

Questions and problems: Alex Dickerson, `alexander.dickerson1@unsw.edu.au`.

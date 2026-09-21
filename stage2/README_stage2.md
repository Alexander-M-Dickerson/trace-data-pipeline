# Stage 2 — Monthly Asset-Pricing Panel

Stage 2 turns Stage 1's daily bond-day panel into the **monthly asset-pricing panel**:
returns, bond characteristics, 108 signals, factor time series, and rolling factor betas.

> **Status:** complete. Seven steps, a frozen 145-column contract asserted at the end of
> every build, and a release packager that refuses to publish an unredacted panel.
> `python _run_stage2.py --dry-run` resolves and validates the configuration without
> building anything.

---

## Contents

- [Where Stage 2 runs](#where-stage-2-runs)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [What Stage 2 produces](#what-stage-2-produces)
- [Alternative Treasury benchmarks: rebuilding the betas](#alternative-treasury-benchmarks-rebuilding-the-betas)
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

A full build takes **8 to 13 minutes on 24 cores / 128 GB** (the last two full builds, 2026-09-14
and 2026-09-16). It is comfortable on a modern desktop and unhappy on a laptop with 16 GB.

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
ten minutes discovering it.

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
| `output/panel/main_panel_<mode>.parquet` | The main panel — 145 columns, MMN-adjusted signals, used with `ret_vw` |
| `output/blocks/<mode>/mmn_price_based_signals_<stamp>.parquet` | The 38 unadjusted twins (`*_mmn`), used with `ret_vw_bgn` |
| `output/blocks/<mode>/` | Per-step intermediates: `betas_std`, `betas_x`, `mom_ret`, `mom_retx`, `returns_alt`, `factors`, `factors_merged`, `bbw_factors`, `illiq_factors` |
| `output/blocks/<mode>/betas_<bm>`, `mom_retx_<bm>` | **Optional.** The 68 beta/momentum columns re-estimated on an alternative Treasury benchmark — only if you run `make_excess_blocks.py`. See below |
| `data_reports/` | The LaTeX data report, its figures and the PDF |
| `manifests/` | A JSON run manifest per build: inputs, hashes, config, timings |
| `release/` | What `make_release.py` packages for publication: the redacted panel, the factor panel, the Stage 1 daily panel in its 32-column public layout (`--what daily`, which withholds the agency ratings, `permco` and `gvkey` and refuses any column nobody has classified), and `osbap_bbw_factors_<vintage>.zip`, the corrected Bai-Bali-Wen four factors for every return definition on the TRACE and extended samples with the authors' original series beside them (`--what bbw`, built from `blocks/<mode>/bbw_factors*.parquet` and `factors_merged.parquet`; the original lives in `reference/`) |

`<mode>` is the build label (`stage1` by default); released files are renamed to the
vintage year, e.g. `main_panel_2026.parquet`. `<stamp>` is Stage 1's date stamp.

**The column contract.** The panel's 145 names **and their order** are frozen in
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

## Alternative Treasury benchmarks: rebuilding the betas

The panel ships five Treasury benchmarks beside `tret`, but every duration-adjusted quantity in
it -- the 68 beta and momentum columns, plus `ret_vwx` and `str` -- is built from
`ret_vw - tret`. So out of the box the panel offers alternative *benchmarks* and not alternative
*systems*: nothing can be sorted on a `tret_bns`-adjusted beta.

`make_excess_blocks.py` closes that. It is **optional** and runs **after** a normal Stage 2
build, reading the blocks that build already wrote. It changes no panel and overwrites nothing.

```bash
cd stage2
python make_excess_blocks.py --mode stage1 --verify              # gate first (see below)
python make_excess_blocks.py --mode stage1 --benchmark bns       # one benchmark
python make_excess_blocks.py --mode stage1 --benchmark all       # bns + cls
```

It writes `output/blocks/<mode>/betas_<bm>.parquet` and `mom_retx_<bm>.parquet` beside the
existing `betas_x` / `mom_retx`. The column names **inside** are canonical -- `b_amd`, not
`b_amd_bns` -- so a block drops into the same recipe the duration-adjusted panel already uses:
drop the 68 columns from the standard panel, merge the two blocks back on `(cusip, date)`, and
set `ret_vwx = ret_vw - tret_<bm>` and `str = str - tret_<bm>`.

**Run `--verify` first.** It puts the incumbent `tret` through the same generalised code and
requires the result to reproduce the shipped `betas_x` and `mom_retx` bit-for-bit. Same
arithmetic, same order, so exact equality is achievable, and anything less means the code moved
something it should not have.

❗**A benchmark's betas need that benchmark's own factors.** A duration-adjusted regression does
not simply change the left-hand side: the bond-market factors are replaced by twins estimated on
the same excess return, and `term` moves with them. `make_excess_blocks.py` therefore rebuilds
the BBW double sorts per benchmark before it rebuilds any beta, and `compute_all_betas` **raises**
if a benchmark's factor twins are missing rather than falling back to the `tret` ones -- a
fallback would produce `b_*` columns that look entirely normal and mean nothing.

**Cost.** One benchmark is roughly half a full beta run (~33 s of beta work on 2.4M bond-months),
plus its factor sorts; the output is comparable in size to `betas_x`.

**A sanity check worth running.** `corr(b_mktb, b_mktb)` between a benchmark block and `betas_x`
should be high but never exactly 1.0. Exactly 1.0 means the factor swap did not take. Expect
about 0.98 on the TRACE era, and materially lower before it -- pre-1986 the long end of the
Treasury curve is extrapolated flat, so the benchmarks diverge most where the curve is least
anchored.

---

## External data

Stage 2 downloads and caches a few published inputs into `stage2/data/` on first use:

| Input | Why |
|---|---|
| Pre-2002 quote returns | Lets rolling signals reach a common 2002-08 start. Carries the five `tret_*` benchmarks as well as `tret`, so the duration-adjusted blocks have the same pre-history whichever benchmark they use |
| Extended BBW factor series | Backfills factor history before 2002-08 |
| Bond-firm linker (`bond_firm_linker_2026`) | `permno` / `permco` / `gvkey`, joined on the linker's identity window |
| Federal Reserve GSW yield curve | The zero-coupon Treasury returns behind `tret_bns`, `tret_cfm`, `tret_gprs` and `tret_cls` |
| Fama-French, FRED, He-Kelly-Manela, Ludvigson, Policy Uncertainty | The monthly factor matrix |

It also fetches Treasury returns, Fama-French factors, VIX and the FISD coupon terms the
`tret_*` benchmarks need (`fisd_cashflow_terms.parquet`) from WRDS **once** and caches them,
so only the first run needs your credentials.

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

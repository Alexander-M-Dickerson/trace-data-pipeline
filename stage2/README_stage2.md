# Stage 2: monthly asset-pricing panel

Stage 2 turns Stage 1's daily bond-day panel into the monthly asset-pricing panel. The panel
has returns, bond characteristics, 108 signals, factor time series and rolling factor betas.

Stage 2 is complete. It runs in seven steps. Every build ends with a check of the panel
against a frozen 145-column contract, and the release packager refuses to publish an
unredacted panel. `python _run_stage2.py --dry-run` resolves and validates the configuration
without building anything.

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

## Where Stage 2 runs

Stage 2 runs on your own machine, not on WRDS. Stages 0 and 1 are grid jobs. Stage 2 is not
one, so **never submit it with `qsub`**. There are two reasons.

1. It needs more memory than a WRDS grid slot allows, because it works on the whole panel at
   once.
2. It reads no TRACE data. Its inputs are the files Stage 0 and Stage 1 wrote, plus a few
   series it fetches once, from WRDS and the web, and caches (see [External data](#external-data)).

For that reason `run_pipeline.sh` does not submit it. When your WRDS run finishes, download
`stage0/` and `stage1/` to your own machine and run Stage 2 there.

A full build takes 8 to 18 minutes on 24 cores and 128 GB. That range is from four full builds
between 2026-09-14 and 2026-09-23. The slowest was the first in a fresh checkout, which also
downloads the external series. A build runs comfortably on a modern desktop and struggles on a
laptop with 16 GB.

## Prerequisites

Your local tree must look like this.

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

The Stage 1 file published for download **will not work**. It has the agency ratings,
`permco` and `gvkey` removed for licensing reasons, and Stage 2 keeps only bond-days that have
at least one agency rating. Stage 2 recognises that file and refuses it at start-up. Run
Stages 0 and 1 yourself on WRDS and use their output.

Beyond Stage 1's Python packages, Stage 2 needs `numba` (Stage 1 takes it only below Python
3.14), `numexpr`, `pytest` and PyBondLab 0.3.0, in Python 3.11 to 3.13. Python 3.14 cannot
install them yet, and `requirements-local.txt` says why. Two lines install them all.

```bash
python -m pip install -r requirements-local.txt
python -m pip install --no-deps pybondlab==0.3.0
```

PyBondLab declares `numpy<2` and this repository installs numpy 2, so it goes in without its
dependency list (`requirements-local.txt` says why that is safe). The dry run says if anything
is missing.

## Quick start

```bash
cd stage2
./run_stage2.sh --dry-run     # resolve and check everything, build nothing
./run_stage2.sh               # the full build
```

`--dry-run` prints the resolved configuration and validates every input, and it reports all
the problems at once. Run it first. It is fast, and it catches a wrong tree before you spend
ten minutes finding out.

These flags are useful. They are passed straight through to `_run_stage2.py`.

| Flag | Effect |
|---|---|
| `--dry-run` | Validate the configuration and exit |
| `--limit-cusips 200` | Build on the first 200 CUSIPs, for a fast development loop. It writes to the same `output/` as a full build and **replaces that build's panel and blocks** |
| `--from-step N --to-step M` | Run part of the pipeline (steps 1 to 7) |
| `--factor-source pinned` | Use the factor file published with your vintage (downloaded once) instead of rebuilding it from public sources. You need it to reproduce a published panel, which also takes the same WRDS run of stages 0 and 1 |
| `--validate` | After the build, run the column-coverage check (`validate_coverage.py`) and record it in the run manifest |

## Configuration

Every setting is in `_stage2_settings.py`. These are the ones you are most likely to change.

| Setting | Default | Meaning |
|---|---|---|
| `ROOT_PATH` | auto-detect | The folder holding `stage0/`, `stage1/`, `stage2/` |
| `DAILY_INPUT` | `None` | Pin a specific Stage 1 panel. Otherwise the newest is used |
| `FACTOR_SOURCE` | `"public"` | Build the factor matrix from public sources |
| `WORKERS` × `THREADS_PER` | 6 × 2 | Keep the product at or below your core count |
| `START_DATE` | `2002-07-31` | First month-end priced. A return needs the month before, so the panel starts a month later, 2002-08-31 |
| `BETA_WINDOW` / `BETA_MIN_OBS` | 36 / 12 | Rolling beta window |
| `DEF_CORP_MIN_MATURITY` / `DEF_GOVT_TENOR` | 10y / 20y | The default-premium factor's two legs |

The processing parameters below those are copied unchanged from the reference implementation,
and they define the published panel. Changing one changes your results. That is fine if you
do it on purpose.

Environment variables can override the inputs without an edit to the file. They are
`STAGE2_DAILY_INPUT` (the Stage 1 panel), `STAGE2_FISD_FILE` and `STAGE2_CALL_FILE` (Stage 0's
FISD file and Stage 1's call flags), and `STAGE2_ROOT` (the folder holding `stage0/`,
`stage1/` and `stage2/`).

## What Stage 2 produces

| Output | Contents |
|---|---|
| `output/panel/main_panel_<mode>.parquet` | The main panel, with 145 columns and MMN-adjusted signals, used with `ret_vw` |
| `output/blocks/<mode>/mmn_price_based_signals_<stamp>.parquet` | The 38 unadjusted twins (`*_mmn`), used with `ret_vw_bgn` |
| `output/blocks/<mode>/` | Per-step intermediates (`betas_std`, `betas_x`, `mom_ret`, `mom_retx`, `returns_alt`, `factors`, `factors_merged`, `bbw_factors`, `illiq_factors`) |
| `output/blocks/<mode>/betas_<bm>`, `mom_retx_<bm>` | Optional. The 68 beta/momentum columns re-estimated on an alternative Treasury benchmark, written only if you run `make_excess_blocks.py`. See below |
| `data_reports/` | The LaTeX data report, its figures and the PDF |
| `manifests/` | JSON run manifests with the inputs, hashes, config and timings. A full build writes one for the whole run and one per group of steps |
| `release/` | What `make_release.py` packages for publication. That is the redacted panel, the factor panel, the Stage 1 daily panel in its 32-column public layout (`--what daily`, which withholds the agency ratings, `permco` and `gvkey` and refuses any column nobody has classified), and `osbap_bbw_factors_<vintage>.zip`. The zip has the corrected Bai-Bali-Wen four factors for every return definition on the TRACE and extended samples, with the authors' original series beside them (`--what bbw`, built from `blocks/<mode>/bbw_factors*.parquet` and `factors_merged.parquet`). The original is in `reference/`. And `osbap_monthly_factors_<vintage>.zip` (`--what monthly_factors`), every monthly factor the betas are estimated on in one file: step 4's `factors_merged.parquet`, with the bond factors spliced from the Lehman-ICE series before 2002-08 and TRACE after, checked value for value against the blocks it came from |

`<mode>` is the build label (`stage1` by default). Released files are renamed to the vintage
year, for example `main_panel_2026.parquet`. `<stamp>` is Stage 1's date stamp.

The panel's 145 column names and their order are frozen in
[`lib/contract.py`](lib/contract.py) and checked at the end of step 7. This is the column
contract. Adding, removing or moving a column is a public API change and needs a CHANGELOG
entry. If the columns differ from the contract the build fails, so a file with its columns in
another order cannot be written without an error. The check has caught a real reordering, when
the DEF/TERM fix swapped `b_defb` and `b_termb` before anyone noticed.

The panel you build is complete. `permco`, `gvkey` and the raw 1 to 22 agency ratings are all
in it. The copies published on openbondassetpricing.com have those redacted, because people
who hold no licence for them can download the copies. The redaction happens in
`make_release.py`, which you run only if you are publishing a vintage for others. Nothing in
the build touches those columns.

To build the data report, run one of these.

```bash
bash run_build_data_reports.sh                 # 14 tables, 11 figures, PDF
bash run_build_data_reports.sh --no-external   # skip the DFPS/WRDS comparisons
```

It takes roughly 2½ minutes with both comparison suites and 1 minute without.

Every column is defined in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md). Two more notes cover
the harder parts of the method. [`README_Default.md`](README_Default.md) covers
defaulted-bond returns, and the market-microstructure section of the dictionary covers the
split between adjusted and unadjusted signals.

The two-panel rule is to use the main panel's signals with `ret_vw`, or the `_mmn` signals
with `ret_vw_bgn`. **Never mix them.** Mixing brings back the microstructure bias the split
exists to remove.

## Alternative Treasury benchmarks: rebuilding the betas

The panel has five Treasury benchmarks beside `tret`, but its 68 beta and momentum columns are
estimated on `ret_vw`, and their duration-adjusted versions (the `betas_x` and `mom_retx`
blocks) on `ret_vw - tret` only. A plain build therefore gives you the alternative benchmarks
and no betas or momentum columns estimated on them, so nothing can be sorted on a
`tret_bns`-adjusted beta.

`make_excess_blocks.py` fills that gap. It is optional. Run it after a normal Stage 2 build,
because it reads the blocks that build wrote. It changes no panel and overwrites nothing.

```bash
cd stage2
python make_excess_blocks.py --mode stage1 --verify              # gate first (see below)
python make_excess_blocks.py --mode stage1 --benchmark bns       # one benchmark
python make_excess_blocks.py --mode stage1 --benchmark all       # bns + cls
```

It writes `output/blocks/<mode>/betas_<bm>.parquet` and `mom_retx_<bm>.parquet` beside the
existing `betas_x` / `mom_retx`. The columns inside keep the standard names (`b_amd`, not
`b_amd_bns`), so a block fits the recipe the duration-adjusted panel already uses. Drop the 68
columns from the standard panel, merge the two blocks back on `(cusip, date)`, and set
`ret_vwx = ret_vw - tret_<bm>` and `str = str - tret_<bm>`.

Run `--verify` first. It puts the existing benchmark, `tret`, through the same general code and
requires the result to reproduce the shipped `betas_x` and `mom_retx` bit for bit. The
arithmetic and its order are the same, so the result can be exactly equal. Any difference
means the code changed something it should not have.

A benchmark's betas need that benchmark's own factors. A duration-adjusted regression changes
more than the left-hand side. The bond-market factors are replaced by twins estimated on the
same excess return, and `term` moves with them. So `make_excess_blocks.py` rebuilds the BBW
double sorts per benchmark before it rebuilds any beta. If a benchmark's factor twins are
missing, `compute_all_betas` raises an error. It does not fall back to the `tret` ones, because
a fallback would produce `b_*` columns that look normal and mean nothing.

One benchmark costs roughly half a full beta run (~33 s of beta work on 2.4M bond-months),
plus its factor sorts. The output is about the size of `betas_x`.

As a sanity check, `corr(b_mktb, b_mktb)` between a benchmark block and `betas_x` should be
high but never exactly 1.0. Exactly 1.0 means the factor swap did not happen. Expect about
0.98 on the TRACE era, and lower before 2002-08 (about 0.92 for 1997 to 2002), where the
rolling window runs on pre-TRACE quote returns.

## External data

On first use, Stage 2 downloads a few published inputs and caches them in `stage2/data/`.

| Input | Why |
|---|---|
| Pre-2002 quote returns | Lets rolling signals reach a common 2002-08 start. It has the five `tret_*` benchmarks as well as `tret`, so the duration-adjusted blocks have the same pre-history whichever benchmark they use |
| Extended BBW factor series | Backfills factor history before 2002-08 |
| Bond-firm linker (`bond_firm_linker_2026`) | `permno` / `permco` / `gvkey`, joined on the linker's identity window |
| Federal Reserve GSW yield curve | The zero-coupon Treasury returns behind `tret_bns`, `tret_cfm`, `tret_gprs` and `tret_cls` |
| Fama-French, FRED, He-Kelly-Manela, Ludvigson, Policy Uncertainty | The monthly factor matrix |

It also fetches Treasury returns, Fama-French factors, VIX and the FISD coupon terms the
`tret_*` benchmarks need (`fisd_cashflow_terms.parquet`) from WRDS once and caches them,
so only the first run needs your credentials.

## Troubleshooting

"Stage 1 directory not found" means you are running from the wrong place, or you have not
downloaded `stage0/` and `stage1/` yet. Run `./run_stage2.sh --dry-run` to see every path
Stage 2 resolved.

"It looks like the Stage 1 file published for download" means you are pointing at the public
download and not at your own run. See [Prerequisites](#prerequisites).

"Every agency rating column is empty" means the Stage 1 panel has the rating columns but no
ratings. Re-run Stage 1 with the current code.

"is missing N column(s) Stage 2 needs" means the Stage 1 panel is from an older release.
Re-run Stage 1 with the current code.

If the build runs out of memory, lower `WORKERS`, or build in pieces with `--from-step` /
`--to-step`. Each step writes its blocks to disk, so a build resumes cheaply.

A public factor URL can move, because vendors rotate filenames. The fetcher recovers from
this for some sources. For the others, update the URL in `_stage2_settings.py`.

## Support

Send questions and problems to Alex Dickerson, `alexander.dickerson1@unsw.edu.au`.

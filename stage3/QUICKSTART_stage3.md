# Quickstart — Stage 3

Stage 3 turns the Stage-2 monthly panel into the paper's 33 tables and 11 figures.
It runs **on your own computer**, like Stage 2, and needs no WRDS connection.

If you have just finished Stage 2, everything below should work with no configuration.

**You need two things beyond Stage 2's requirements:**

- **PyBondLab** — the portfolio-sorting library every sort runs through. The
  repository's own `requirements.txt` installs it
  (`python -m pip install -r ../requirements.txt`); the source is at
  [github.com/GiulioRossetti94/PyBondLab](https://github.com/GiulioRossetti94/PyBondLab).
- **pdflatex** — only for the last step, which compiles the exhibits into one PDF. TeX
  Live or MiKTeX. Without it, run with `--no-compile` and you still get every table and
  figure as a file; you just do not get `exhibits.pdf`.

`python tools/check_inputs.py` warns if pdflatex is missing rather than failing, because
`--no-compile` is a legitimate way to run.

---

## 1. Check the inputs

```bash
cd stage3
python tools/check_inputs.py
```

You want `5/5 inputs satisfy the contract`. It checks rows, columns and date spans, and
reports every problem at once.

If something is missing, it is almost always one of two things:

- **Stage 2 has not run**, or ran under a different mode name. Stage 2 writes its panel
  as `main_panel_<mode>.parquet`; Stage 3 looks for `stage1`, which is what the public
  pipeline uses. Other name? `export STAGE3_MODE=<yours>`.
- **The pipeline lives somewhere else.** Then set the directories:

```bash
export STAGE0_DIR=/path/to/trace-data-pipeline/stage0
export STAGE1_DIR=/path/to/trace-data-pipeline/stage1
export STAGE2_DIR=/path/to/trace-data-pipeline/stage2
```

Any single input can also be pointed at directly — `STAGE2_PANEL`, `STAGE2_MMN`,
`STAGE2_BBW`, `STAGE2_FACTORS`, `STAGE1_DAILY`.

### Every environment variable Stage 3 reads

Nothing here is required: each has a working default, and `_stage3_settings.py` is the
alternative to exporting any of them.

| variable | default | what it does |
|---|---|---|
| `STAGE0_DIR`, `STAGE1_DIR`, `STAGE2_DIR` | the sibling folders | where the earlier stages live |
| `STAGE2_PANEL`, `STAGE2_MMN`, `STAGE2_BBW`, `STAGE2_FACTORS`, `STAGE1_DAILY` | derived from the above | one input file each, when the layout is not standard |
| `STAGE3_MODE` | `stage1` | which panel to read: Stage 2 writes `main_panel_<mode>.parquet` |
| `STAGE3_DATA` | `stage3/data` | where intermediate results are written. Point it at a fast disk, or at a scratch area to leave the repo clean |
| `STAGE3_REPORTS` | `stage3/reports` | where tables, figures, `timings.jsonl` and the PDF go |
| `PYBONDLAB_DIR` | the installed package | a PyBondLab checkout to use instead. Section 5 needs one with the fast kernels; everything else falls back on its own |
| `STAGE3_WORKERS` | cores, clamped | processes for the two grids. `--workers` overrides per run |
| `STAGE3_MEMORY_LIMIT` | `4GB` | DuckDB's memory cap in the data appendix |
| `STAGE3_WORKER_THREADS` | set by `fastrun.pmap` | internal: how many threads one worker may use. Set by the parent, read by the child; you do not set this |
| `NUMBA_NUM_THREADS` | set by the grid runners | internal, same idea, for numba inside a worker |
| `STAGE3_ALLOW_STALE_LEDGER` | unset | ❗a correctness bypass — see below |

**`STAGE3_ALLOW_STALE_LEDGER=1`** switches off one guard: the check that Section 5's
status ledger is not older than the MUA grid it describes. Re-running the grid does not
invalidate the summarizer's completion marker, so without the guard the orchestrator
skips the summarizer and every Section-5 exhibit is built on a statistics layer derived
from a *different* grid. That is not hypothetical — on 2026-09-11 it produced six cells
holding a full 268-month series in the grid and `n_obs = 0` in the summary, seven minutes
apart. The fix is almost always `python s3_nse/mua_summarize.py`. Set the variable only
when you know the exhibits will describe a grid that is no longer on disk and you want
them anyway.

## 2. Point at a PyBondLab build

```bash
export PYBONDLAB_DIR=/path/to/PyBondLab
```

Leave it unset to use whatever `import PyBondLab` finds. Either way Stage 3 prints which
build it resolved, and records it in every manifest:

```
[pblenv] PyBondLab v0.2.0 tree=2f2dfb0d7cdd443e  fast kernels: yes  <- /path/to/PyBondLab
```

> ❗`fast kernels: NO` means **only Section 5 is blocked** — the two uncertainty grids
> need `PyBondLab.fast_sorts` and `PyBondLab.anomaly_assay_fast`, and the 0.2.0 release
> does not carry them. Everything else runs on the slow path automatically, with the
> same numbers: `_run_stage3.py` asks once at startup and prints which path it took.
> Expect roughly fourteen times the sort time there (43.8 s against 3.2 s, measured on
> one sort).
>
> There is no minimum version to give you: at the time of writing the build with the
> kernels and the release without them both report `0.2.0`, so `pblenv` looks for the
> modules rather than comparing version strings.

## 3. Smoke run

```bash
python _run_stage3.py --dry-run     # resolve and print the configuration, compute nothing
python _run_stage3.py --list        # the 40 steps, and which outputs already exist
```

Then one cheap section end to end, which exercises the whole shape — a producer, a
statistics layer, tables and figures:

```bash
python _run_stage3.py --section lib
```

About four and a half minutes on 24 cores with the kernels — 242 s of benched work in
the cold run of 2026-09-12, most of it the four 108-signal sorts. You should end up with
`reports/tables/table01.tex` and `reports/figures/fig03_cumret.pdf`.

## 4. Everything

```bash
bash run_stage3.sh
```

**906 s — about 15 minutes** on 24 cores with the fast kernels, measured on a cold
run (`data/` and `reports/` wiped first) on 2026-09-12, all 40 steps. The two
uncertainty grids are 55% of it. See **What it costs** in
[README_stage3.md](README_stage3.md) for the per-section split, the disk and memory
figures, and what to lower first on a smaller machine.

### Which sample

By default the exhibits run to **whatever month your Stage-2 panel reaches**. To
reproduce the published window instead:

```bash
python _run_stage3.py --sample paper      # 2002-09 to 2024-12, T = 268
```

Every caption states which one produced it, so a PDF is never ambiguous about its own
sample. One switch fans out to each section's own flag, so running a driver by hand is
unchanged.

### Resuming, forcing, and what a failure does

A producer whose output already exists is **skipped**, so if a run stops you can simply
run it again and it resumes. `--force` recomputes from scratch.

A failed **exhibit** does not abandon the run: the remaining steps still run and the PDF
is still produced, and the run exits non-zero so the failure is not lost. A failed
**producer** stops the chain, because everything downstream of it would read missing or
stale data -- `--keep-going` overrides that too, when you want to see everything that is
broken at once.

## 5. Read the results

```
reports/tables/     33 .tex files, one per exhibit
reports/figures/    11 .pdf files
data/<section>/     the statistics frames behind them, as CSV, plus a manifest per result
reports/timings.jsonl   one line per run: phases, wall clock, and its own check
```

Each run prints a `PASS` or `FAIL` line saying what it checked — every printed cell
populated, a grid complete, a figure agreeing with its own table. Those lines are the
run's own audit and they are collected in `reports/timings.jsonl`; `make_report.py`
prints any that failed on the PDF's title page, so a red check cannot leave the document
silently.

❗One check is **red by design** on this build: `s3_nse/t06_mua_nse.py`'s twin-invariance check
fails whenever the sort engine's unstable empty cell bites (see *What is not
reproducible* in README_stage3.md). The run continues, the PDF is produced, and the exit
code is non-zero. That is the intended behaviour, not a broken install.

---

## Expected cost

A single sort over the full monthly panel takes about a second. Asking for **turnover**
costs roughly twelve times that, and it is required for the net-of-cost and
portfolio-size exhibits — so plan around turnover, not around worker count.

On worker count: on Windows the panel is re-pickled into every worker, so twelve workers
return about three times, not twelve. The grids avoid this by making a unit of work one
signal and having each worker read its own column slice, rather than inheriting the
panel. That is why they are minutes rather than hours.

`--workers` and `--threads` are exposed on both grid runners, and both now clamp their
own defaults so `workers * threads` stays at or below your core count. Measured peak on
this build is **0.7 GB per DUA worker** (`max_worker_rss_gb` in the grid's own manifest),
so on a normal machine the grids are bounded by cores, not by memory.

---

## Check it

```bash
cd stage3
python -m pytest tests/ -q          # the contract suite: 68 checks, about 1 s
python tools/check_inputs.py        # the five inputs, before a long run
python _run_stage3.py --dry-run     # what would run, what would be skipped
```

The contract suite reads what is on disk and asserts what the code must be; it does not
recompute any statistic, so it is fast enough to run on every edit. The run itself checks
its own numbers as it goes — each driver records a PASS/FAIL line in
`reports/timings.jsonl`, and `make_report.py` prints those that failed on the title page.

---

## If something goes wrong

| symptom | what it means |
|---|---|
| `the DUA statistics layer is not at ...` | run `s3_nse/run_dua_grid.py`, then the same file with `--stats` |
| `the MUA summary is not at ...` | run `s3_nse/run_mua_grid.py`, then `s3_nse/mua_summarize.py` |
| `this exhibit needs sort CSVs under ...` | the message names the missing files and the command that makes them |
| `needs PyBondLab's fast kernels` | set `PYBONDLAB_DIR` to a build that has them |
| `--stats with a --signals subset` | refused on purpose: it would overwrite the full statistics with subset-only frames, and no exhibit downstream could tell |
| `the MUA status ledger is not at ...` | run `s3_nse/mua_summarize.py`; it writes the ledger beside the summary |
| `the MUA status ledger is OLDER than the grid it describes` | the grid was rebuilt and the summary was not. Re-run `s3_nse/mua_summarize.py`. `STAGE3_ALLOW_STALE_LEDGER=1` proceeds anyway |
| `the ledger has N rows, expected 23,328` | the summarizer did not see the whole grid (108 signals x 216 specs). Re-run it |
| `the series on disk were built for ...` | not an error: Section 4's window is a producer argument, so a different `--sample` rebuilds its cells |
| `expected T=268, got NNN` | the sample window moved. Every t-statistic depends on T through the lag count, so this stops rather than printing quietly wrong numbers |

# Quickstart — Stage 3

Stage 3 turns the Stage-2 monthly panel into the paper's 32 tables and 11 figures.
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

About four minutes. You should end up with `reports/tables/table01.tex` and
`reports/figures/fig03_cumret.pdf`.

## 4. Everything

```bash
bash run_stage3.sh
```

Roughly 25 minutes on 24 cores with the fast kernels. The two uncertainty grids are
most of it.

A producer whose output already exists is **skipped**, so if a run stops you can simply
run it again and it resumes. `--force` recomputes from scratch. `--keep-going` runs the
rest of the steps after a failure instead of stopping, which is what you want when you
are trying to see everything that is broken at once.

## 5. Read the results

```
reports/tables/     32 .tex files, one per exhibit
reports/figures/    11 .pdf files
data/<section>/     the statistics frames behind them, as CSV, plus a manifest per result
reports/timings.jsonl   one line per run: phases, wall clock, and its own check
```

Each run prints a `PASS` or `FAIL` line saying what it checked — every printed cell
populated, a grid complete, a figure agreeing with its own table. A `FAIL` sets the exit
code, so `run_stage3.sh` stops on it.

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

## If something goes wrong

| symptom | what it means |
|---|---|
| `the DUA statistics layer is not at ...` | run `s3_nse/run_dua_grid.py`, then the same file with `--stats` |
| `the MUA summary is not at ...` | run `s3_nse/run_mua_grid.py`, then `s3_nse/mua_summarize.py` |
| `this exhibit needs sort CSVs under ...` | the message names the missing files and the command that makes them |
| `needs PyBondLab's fast kernels` | set `PYBONDLAB_DIR` to a build that has them |
| `--stats with a --signals subset` | refused on purpose: it would overwrite the full statistics with subset-only frames, and no exhibit downstream could tell |
| `expected T=268, got NNN` | the sample window moved. Every t-statistic depends on T through the lag count, so this stops rather than printing quietly wrong numbers |

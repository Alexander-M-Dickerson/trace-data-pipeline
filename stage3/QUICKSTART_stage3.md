# Quickstart — Stage 3

Stage 3 turns the Stage-2 monthly panel into the paper's 32 tables and 11 figures.
It runs **on your own computer**, like Stage 2, and needs no WRDS connection.

If you have just finished Stage 2, everything below should work with no configuration.

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

> ❗`fast kernels: no` means **Section 5 will refuse to run** — the two uncertainty
> grids need `PyBondLab.fast_sorts` and `PyBondLab.anomaly_assay_fast`, which the 0.2.0
> release does not carry. Everything else runs fine without them.

## 3. Smoke run

```bash
python _run_stage3.py --dry-run     # resolve and print the configuration, compute nothing
python _run_stage3.py --list        # the 39 steps, and which outputs already exist
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

`--workers` and `--threads` are exposed on both grid runners. Keep `workers * threads`
at or below your core count. The DUA grid holds 10–20 GB per worker, so size that one
against memory rather than cores.

---

## If something goes wrong

| symptom | what it means |
|---|---|
| `the DUA statistics layer is not at ...` | run `s3_nse/run_dua_grid.py`, then the same file with `--stats` |
| `the MUA summary is not at ...` | run `s3_nse/run_mua_grid.py`, then `s3_nse/mua_summarize.py` |
| `this exhibit needs sort CSVs under ...` | the message names the missing files and the command that makes them |
| `needs PyBondLab's fast kernels` | set `PYBONDLAB_DIR` to a build that has them |
| `--stats with a --signals subset` | refused on purpose: it would overwrite the full statistics with subset-only frames, and no exhibit downstream could tell |
| `T=NNN, expected 268` | the sample window moved. Every t-statistic depends on T through the lag count, so this stops rather than printing quietly wrong numbers |

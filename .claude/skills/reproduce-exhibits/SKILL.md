---
name: reproduce-exhibits
description: Build the exhibits of The Corporate Bond Factor Replication Crisis (stage 3 of trace-data-pipeline) - portfolio sorts, tex tables and figures - from the panel stage 2 built, on the user's own computer. Use when the user wants the tables and figures produced, one section rerun, or a failed stage 3 run diagnosed.
---

# Build the exhibits (stage 3)

Read first: `AGENTS.md`, then `stage3/AGENTS.md`, which this follows, and
`stage3/QUICKSTART_stage3.md` for the detail. `stage3/INDEX.md` maps each exhibit to the
file that makes it.

## Use when

- stage 2 has finished and the user wants the exhibits
- one section needs rerunning, or a stage 3 run failed

## Do not use when

- stage 2 has not finished: use `build-panel`
- the user wants to know what an exhibit computes: use `explain`

## Steps (from `stage3/`)

1. **Ask which sample**: the whole panel (the default, `--sample frontier`) or the paper's
   window (`--sample paper`, 2002-09 to 2024-12).
2. **Inputs**: `python tools/check_inputs.py`. Each input must exist and have the expected
   shape.
3. **Dry run**: `python _run_stage3.py --dry-run` shows what would run and what exists already.
4. **Run** in the background with a log: `bash run_stage3.sh`, adding `--sample paper` for the
   paper's window, about 15 to 20 minutes. Where several Pythons exist, set
   `PY=/path/to/python` so it uses the one the requirements went into.
5. **Tests**: `python -m pytest tests -q`.
6. **Report** what was written: `stage3/reports/tables/`, `stage3/reports/figures/`,
   `stage3/reports/exhibits.pdf`, and `stage3/reports/timings.jsonl` (one line per step, with
   its own checks).

## Things to know

- A second run skips the producers whose output exists AND whose recorded inputs are
  unchanged; after a new stage 2 build they rerun by themselves. `--force` recomputes all.
- A failed producer stops the run, because everything after it would read missing data. A
  failed exhibit does not: the others still run, and the exit code is non-zero.
- `python _run_stage3.py --section <name>` reruns one section.
- The PDF needs `pdflatex`. Without it every table and figure is still written as a file.

## Stop and report when

- `tools/check_inputs.py` fails: stage 2 must be (re)run first
- a producer fails: show its log, and do not rerun with `--keep-going` to hide it

## Never

- compare the exhibits with the numbers printed in the paper and call a difference a bug: on
  newer data they are expected to differ
- quote or summarise the paper: it is being rewritten and is not part of this repository

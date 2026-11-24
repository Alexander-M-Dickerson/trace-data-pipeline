---
name: reproduce-exhibits
description: Build the exhibits of The Corporate Bond Factor Replication Crisis (stage 3 of trace-data-pipeline), which are portfolio sorts, tex tables and figures, from the panel stage 2 built, on the user's own computer. Use when the user wants the tables and figures produced, one section rerun, or a failed stage 3 run diagnosed.
---

# Build the exhibits (stage 3)

First read `AGENTS.md`, then `stage3/AGENTS.md`, which this follows, and
`stage3/QUICKSTART_stage3.md` for the detail. `stage3/INDEX.md` maps each exhibit to the
file that makes it.

## Use when

- stage 2 has finished and the user wants the exhibits
- one section needs rerunning, or a stage 3 run failed

## Do not use when

- stage 2 has not finished. Use `build-panel`.
- the user wants to know what an exhibit computes. Use `explain-pipeline`.

## Steps (from `stage3/`)

1. Ask which sample the user wants, the whole panel (the default, `--sample frontier`) or the
   paper's window (`--sample paper`, 2002-09 to 2024-12). The return is excess of the bill
   unless the user asks for duration-adjusted returns. Then use `--returns dbns` (or `dur`,
   `dcls`), which swaps the beta and momentum signals too, needs stage 2's blocks for that
   return, and writes to `stage3/variants/<type>/` instead of `stage3/data/` and
   `stage3/reports/`.
2. Check the inputs with `python tools/check_inputs.py`. Each input must exist and have the
   expected shape.
3. Do a dry run. `python _run_stage3.py --dry-run` shows what would run and what exists
   already.
4. Run `bash run_stage3.sh` in the background with a log, adding `--sample paper` for the
   paper's window. The run takes about 15 to 20 minutes. Where several Pythons exist, set
   `PY=/path/to/python` so it uses the one the requirements went into.
5. Run the tests with `python -m pytest tests -q`.
6. Report what was written. That is `stage3/reports/tables/`, `stage3/reports/figures/`,
   `stage3/reports/exhibits.pdf`, and `stage3/reports/timings.jsonl` (one line per step, with
   its own checks). After `--returns <type>`, the same output is under
   `stage3/variants/<type>/reports/`, and
   `python tools/compare_runs.py data variants/<type>/data` says what moved.

## Things to know

- A second run skips the producers whose output exists and whose recorded inputs are also
  unchanged. After a new stage 2 build they rerun by themselves. `--force` recomputes all.
- A failed producer stops the run, because everything after it would read missing data. A
  failed exhibit does not stop it. The others still run, and the exit code is non-zero.
- `python _run_stage3.py --section <name>` reruns one section.
- The PDF needs `pdflatex`. Without it every table and figure is still written as a file.

## Stop and report when

- `tools/check_inputs.py` fails. Stage 2 must be (re)run first.
- a producer fails. Show its log, and do not rerun with `--keep-going` to hide it.

## Never

- compare the exhibits with the numbers printed in the paper and call a difference a bug. On
  newer data they are expected to differ.
- quote or summarise the paper. It is not part of this repository.

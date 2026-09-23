# Stage 3 with an AI assistant

Stage 3 reproduces the exhibits of *The Corporate Bond Factor Replication Crisis* (tex tables and
figures) from the panel stage 2 built. Read the repository's [AGENTS.md](../AGENTS.md) first. The full
human guide is [QUICKSTART_stage3.md](QUICKSTART_stage3.md).

## Before running

- Stage 2 must have finished: stage 3 reads `stage2/output/panel/main_panel_stage1.parquet` and the
  blocks beside it. Another panel name: `export STAGE3_MODE=<mode>`.
- Stage 3 sorts portfolios with PyBondLab. Section 5 needs a PyBondLab build with the fast kernels;
  point at one with `export PYBONDLAB_DIR=/path/to/PyBondLab`. Stage 3 prints which build it used.
- Ask the user which sample they want: the full panel (default) or the paper's sample
  (`--sample paper`, 2002-09 to 2024-12).

## The commands, in order

Run from `stage3/`.

```bash
python tools/check_inputs.py        # 1. the five inputs exist and have the expected shape
python _run_stage3.py --dry-run     # 2. what would run, and what already exists
bash run_stage3.sh                  # 3. everything, about 15-20 min on 24 cores
python -m pytest tests -q           # 4. the stage's own tests
```

Run step 3 in the background with a log. It takes about 15-20 minutes cold. `run_stage3.sh` calls
the `python` on PATH; where several interpreters exist, set `PY=/path/to/python` so it uses the one
the requirements were installed into.

A second run skips the producers whose output already exists AND whose recorded inputs are
unchanged. After a new Stage 2 build, the producers that read it run again by themselves
(the runner prints `[rebuild] ... built from different Stage 2 inputs`); `--force` is only
needed to recompute everything regardless.

## What the user ends up with

```
stage3/reports/tables/      one .tex file per exhibit
stage3/reports/figures/     the figures, as PDF
stage3/reports/timings.jsonl   one line per run: phases, wall clock, the run's own checks
```

Stage 3 produces the exhibits on the user's data. It does not compare them with the numbers
printed in the paper: on newer data they are expected to differ.

## When something fails

`run_stage3.sh` stops at the first failing step and says which. See "If something goes wrong" in
[QUICKSTART_stage3.md](QUICKSTART_stage3.md). `python _run_stage3.py --section <name>` reruns one
section.

# Stage 3 with an AI assistant

Stage 3 reproduces the exhibits of *The Corporate Bond Factor Replication Crisis* (tex tables and
figures) from the panel stage 2 built. Read the repository's [AGENTS.md](../AGENTS.md) first. The full
human guide is [QUICKSTART_stage3.md](QUICKSTART_stage3.md).

The `reproduce-exhibits` skill (`/reproduce-exhibits` in Claude Code, `$reproduce-exhibits` in
Codex) walks this file step by step, and `python doctor.py`, from the repository root, says
whether this stage's inputs are ready.

## Before running

- Stage 2 must have finished: stage 3 reads `stage2/output/panel/main_panel_stage1.parquet` and the
  blocks beside it. Another panel name: `export STAGE3_MODE=<mode>`.
- Stage 3 sorts portfolios with PyBondLab 0.3.0, installed with the two lines in
  `requirements-local.txt`. It stops and prints them if PyBondLab is missing or a different
  version, and prints the version it used.
- Ask the user which sample they want: the whole panel (the default, `--sample frontier`) or the
  paper's sample (`--sample paper`, 2002-09 to 2024-12).
- The return is excess of the bill unless they ask otherwise. For duration-adjusted returns,
  `bash run_stage3.sh --returns dbns` (or `dur`, `dcls`): it swaps the beta and momentum
  signals for ones estimated on that return, writes its own tree `variants/<type>/`, and needs
  stage 2's blocks for it (`python make_excess_blocks.py --benchmark bns` in `stage2/`).
  `python tools/compare_runs.py data variants/dbns/data` compares it with the standard run.
  [ref:rule.return_types]

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

Which columns are signals is decided in one place, `signal_set.py`: the 108 Cluster rows of the
Table IA.VIII spec [ref:rule.signal_set]. Every section sorts those and nothing else, and a panel
column the spec does not classify stops the run. A Section 3 census built before 4.1.1 holds five
Treasury benchmark returns as if they were signals; Table B.1 refuses it, and
`python _run_stage3.py --section lib --force` rebuilds it.

## What the user ends up with

```
stage3/reports/tables/      one .tex file per exhibit
stage3/reports/figures/     the figures, as PDF
stage3/reports/timings.jsonl   one line per step: phases, wall clock, the step's own checks
```

Stage 3 produces the exhibits on the user's data. It does not compare them with the numbers
printed in the paper: on newer data they are expected to differ.

## When something fails

A failed producer (a sort or a grid) stops `run_stage3.sh`, because everything after it would read
missing data; `--keep-going` overrides that. A failed exhibit does not stop it: the other steps
run, the PDF is built, and the exit code is non-zero. See "If something goes wrong" in
[QUICKSTART_stage3.md](QUICKSTART_stage3.md). `python _run_stage3.py --section <name>` reruns one
section.

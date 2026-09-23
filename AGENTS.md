# Working in this repository with an AI assistant

This file is for AI coding assistants (Claude Code, Codex and similar) helping someone run the
pipeline. `CLAUDE.md` in the same folder imports it, so both tools read the same instructions.
People should start at [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).

## What this repository does

It turns the WRDS TRACE corporate bond tape into a monthly bond asset pricing panel, then
reproduces the exhibits of *The Corporate Bond Factor Replication Crisis*. It runs in four stages
on two machines:

| stage | where it runs | what it does | time |
|---|---|---|---|
| 0 and 1 | **WRDS Cloud** (SGE grid) | clean the raw tape, build the daily bond panel | about 5 h |
| hand-off | the user | zip the folder on WRDS, copy it to their own computer | -- |
| 2 | **the user's computer** | build the monthly panel (145 columns per bond-month) | about 10-20 min (first run includes downloads) |
| 3 | **the user's computer** | portfolio sorts and the paper's exhibits (tex tables, figures) | about 15-20 min |

Stages 2 and 3 are what most users run locally with an assistant. Each has its own instructions:
[stage2/AGENTS.md](stage2/AGENTS.md) and [stage3/AGENTS.md](stage3/AGENTS.md).

## Rules for the assistant

1. **Read the stage's QUICKSTART before running anything**: `stage2/QUICKSTART_stage2.md`,
   `stage3/QUICKSTART_stage3.md`. They hold the exact commands and the expected run times.
2. **Always dry-run first.** `python _run_stage2.py --dry-run` and `python _run_stage3.py --dry-run`
   print every input they resolved. Show the user that list before building.
3. **Settings live in one file per stage**: `stage2/_stage2_settings.py`,
   `stage3/_stage3_settings.py`. Change settings there, or with the documented command-line
   flags and environment variables. Do not edit any other code to get past a check.
4. **A failing check is the answer, not an obstacle.** The stages check their own inputs and
   outputs (column contracts, coverage, redaction). If one fails, stop, show the message, and
   explain its cause. Never weaken or skip a check to make a run finish.
5. **Run long builds in the background with a log file**, and report progress from the log. Use
   the runners (`_run_stage2.py`, `run_stage3.sh`): they start each step in a fresh process,
   which the pipeline relies on for speed.
6. **Ask the user, don't guess**, when a choice changes the numbers: the factor source for
   stage 2 (see stage2/AGENTS.md) and the sample for stage 3.
7. **Outputs are not committed.** `stage2/data/`, `stage2/output/`, `stage3/data/` and
   `stage3/reports/` are gitignored. Do not add them to git.

## What the user must provide

- The folder produced on WRDS by stages 0 and 1, unzipped on their computer. Stage 2 finds its
  three inputs by date stamp: `stage1/data/stage1_<YYYYMMDD>.parquet`,
  `stage0/enhanced/trace_enhanced_fisd_<YYYYMMDD>.parquet`,
  `stage1/data/call_dummy_<YYYYMMDD>.parquet`.
- A WRDS account (`WRDS_USERNAME`) for stage 2's first run, which fetches and caches Treasury
  returns, Fama-French factors and VIX. Later runs use the cache.
- Python 3.10+ with `python -m pip install -r requirements.txt`.

## Traps that have caught people

- **The daily file downloaded from openbondassetpricing.com cannot drive stage 2.** Its rating
  columns are removed (they are licensed), and stage 2 keeps only rated bond-months, so it would
  produce an empty panel. Stage 2 refuses to start on it. The user must run stages 0 and 1.
- **A fresh default build does not reproduce a published panel.** Stage 2's default builds its
  factors from live public sources, which revise their history. To reproduce a published vintage
  exactly, use `--factor-source pinned` (stage2/AGENTS.md).
- **Do not run the stage 2 steps yourself in one Python process.** A long-lived process makes
  DuckDB lose its parallelism and steps take several times longer. The runner starts a fresh
  process per step.

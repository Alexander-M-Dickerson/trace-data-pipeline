# Stage 4 with an AI assistant

Stage 4 builds the TRACE-only bond factors published on openbondassetpricing.com from the panel
stage 2 built, and compares them with the published files. Read the repository's
[AGENTS.md](../AGENTS.md) first. The full human guide is [README_stage4.md](README_stage4.md).

## Before running

- Stage 2 must have finished: stage 4 reads `stage2/output/panel/main_panel_stage1.parquet` and
  the blocks beside it, never `stage2/release/` (the released copy is redacted).
- The `dbns` and `dcls` return types need Stage 2's benchmark blocks:
  `python make_excess_blocks.py --mode stage1 --benchmark all`, run in `stage2/`.
- PyBondLab 0.3.0 must be installed with the two lines in `requirements-local.txt`. Stage 4 stops
  and prints them if it is missing or a different version.

## The commands, in order

Run from `stage4/`.

```bash
python build_factors.py --dry-run   # 1. the inputs, and the PyBondLab release in use
bash run_stage4.sh                  # 2. build both sorts, then compare with the published files
python -m pytest tests -q           # 3. the stage's own tests
```

Step 2 takes about 8 minutes; run it in the background with a log. `compare_published.py`
downloads about 37 MB per sort the first time.

## What the user ends up with

Four folders in `stage4/output/`, one panel folder and one CSV folder per sort, laid out like the
published archives. `DATA_DICTIONARY.md` describes every column.

## Reading the comparison

On the same Stage 2 panel as the published files, every cell matches. A panel from a newer WRDS
run is expected to differ; report the differences, do not treat them as a failure to fix. Never
loosen `--tol` to turn a difference into a pass without saying so.

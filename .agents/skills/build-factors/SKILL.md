---
name: build-factors
description: Build the TRACE-only bond factors published on openbondassetpricing.com (stage 4 of trace-data-pipeline) from the panel stage 2 built, and compare them with the published files. Use when the user wants the factors built, a subset built quickly, or the comparison with the published factors run and explained.
---

# Build the TRACE-only factors (stage 4)

Read first: `AGENTS.md`, then `stage4/AGENTS.md`, which this follows, and
`stage4/README_stage4.md` for the detail. Every column of the output is in
`stage4/DATA_DICTIONARY.md`.

## Use when

- stage 2 has finished and the user wants the factors, or the comparison with the published ones

## Do not use when

- stage 2 has not finished: use `build-panel`

## Steps (from `stage4/`)

1. **The benchmark blocks.** The `dbns` and `dcls` return types need stage 2's benchmark
   blocks. If the dry run says they are missing: in `stage2/`,
   `python make_excess_blocks.py --mode stage1 --benchmark all`.
2. **Dry run**: `python build_factors.py --dry-run` lists the inputs and the PyBondLab
   release in use.
3. **Run** in the background with a log: `bash run_stage4.sh`, about 8 minutes. It builds both
   sorts, then runs `compare_published.py`, which downloads about 37 MB per sort the first time.
   A quick run on two signals: `python build_factors.py --signals cs mom6_1`, written to
   `output/_subset` and never over a full build.
4. **Tests**: `python -m pytest tests -q`.

## Reading the comparison

On the same stage 2 panel as the published files, every cell matches. On a panel from a newer
WRDS run, differences are expected: report them, with how many cells and which factors, and do
not treat them as something to fix.

## Stop and report when

- the dry run refuses to start: show why

## Never

- loosen `--tol` to turn a difference into a pass without saying so
- read `stage2/release/`: stage 4 reads the unredacted panel in `stage2/output/`

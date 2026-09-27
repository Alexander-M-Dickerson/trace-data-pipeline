---
name: build-panel
description: Build the monthly bond panel (stage 2 of trace-data-pipeline) on the user's own computer from the stage 0 and 1 output of a WRDS run - the factor-source choice, the dry run, the build, the coverage check and the tests, and optionally the alternative Treasury benchmarks and a shareable release. Use when the user wants the 145-column monthly panel built, rebuilt or checked.
---

# Build the monthly panel (stage 2)

Read first: `AGENTS.md`, then `stage2/AGENTS.md`, which this follows, and
`stage2/QUICKSTART_stage2.md` for the detail.

## Use when

- the stage 0 and 1 output of a WRDS run is on the user's computer, and they want the panel
- a stage 2 build failed, or its output needs checking

## Do not use when

- `python doctor.py` does not show stage 2's inputs as ready: use `onboard`
- the user wants the exhibits or the factors, and the panel is already built: use
  `reproduce-exhibits` or `build-factors`

## Steps (from `stage2/`)

1. **Ask which factors**, because the two sources give different numbers:
   - `public` (the default): assembled from the live public sources, the newest data;
   - `pinned`: the factor panel published for the vintage, to reproduce a published panel.

   A `public` build does not reproduce a published panel, and that is expected.
2. **Dry run**, and show the user every input it resolved:
   `python _run_stage2.py --dry-run [--factor-source pinned]`.
3. **Build** in the background with a log: `python _run_stage2.py [--factor-source pinned]`,
   about 8 to 18 minutes. Report each step as the log shows it start and end. The build checks
   its own output at the end: the 145 column names and their order are fixed.
4. **Coverage**: `python validate_coverage.py`. A column that ends early WITH a named upstream
   reason (listed as UPSTREAM-LIMITED) is expected; one that ends early without a reason is a
   problem to report.
5. **Tests**: `python -m pytest tests -q`. Some skip when no reference build is present; say
   which.
6. Optional, when the user asks, and always before stage 4's `dbns` and `dcls` return types:
   `python make_excess_blocks.py --mode stage1 --verify`, then
   `python make_excess_blocks.py --mode stage1 --benchmark all`.
7. Optional, to share the panel: `python make_release.py --what panel`. It blanks `permco` and
   `gvkey`, reduces the ratings to investment grade and high yield, and refuses to write a file
   that still carries licensed values.

## Resuming and testing

`python _run_stage2.py --from-step N` restarts at step N. A quick test on a small sample:
`python _run_stage2.py --limit-cusips 200`. It writes to the same `output/` folder and replaces
a full build's panel, so run it before the full build, not after.

## Stop and report when

- the dry run fails: show its CONFIGURATION ERROR box and explain it
- the column contract, the coverage check or the redaction fails: these are the answer, not
  obstacles

## Never

- run the steps yourself in one Python process: the runner starts them in fresh processes,
  and a long-lived process makes DuckDB lose its parallelism
- edit any file other than `_stage2_settings.py`, or use anything other than the documented
  flags and environment variables, to get a build through
- claim a `public` build reproduces a published panel

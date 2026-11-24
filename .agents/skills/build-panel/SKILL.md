---
name: build-panel
description: Build the monthly bond panel (stage 2 of trace-data-pipeline) on the user's own computer from the stage 0 and 1 output of a WRDS run. Covers the factor-source choice, the dry run, the build, the coverage check and the tests, and optionally the alternative Treasury benchmarks and a shareable release. Use when the user wants the 145-column monthly panel built, rebuilt or checked.
---

# Build the monthly panel (stage 2)

First read `AGENTS.md`, then `stage2/AGENTS.md`, which this follows, and
`stage2/QUICKSTART_stage2.md` for the detail.

## Use when

- the stage 0 and 1 output of a WRDS run is on the user's computer, and they want the panel
- a stage 2 build failed, or its output needs checking

## Do not use when

- `python doctor.py` does not show stage 2's inputs as ready. Use `onboard`.
- the user wants the exhibits or the factors, and the panel is already built. Use
  `reproduce-exhibits` or `build-factors`.

## Steps (from `stage2/`)

1. Ask which factors, because the two sources give different numbers.
   - `public` (the default) is assembled from the live public sources, the newest data.
   - `pinned` is the factor panel published for the vintage. It reproduces a published panel
     only from the same WRDS run of stages 0 and 1. A user's own run differs whichever factors
     it uses.

   A `public` build does not reproduce a published panel, and that is expected.
2. Do a dry run with `python _run_stage2.py --dry-run [--factor-source pinned]`, and show the
   user every input it resolved.
3. Build in the background with a log, using `python _run_stage2.py [--factor-source pinned]`.
   The build takes about 8 to 18 minutes. Report each step as the log shows it start and end.
   The build checks its own output at the end. The 145 column names and their order are fixed.
4. Check the coverage with `python validate_coverage.py`. A column that ends early with a named
   upstream reason (listed as UPSTREAM-LIMITED) is expected. One that ends early without a
   reason is a problem to report.
5. Run the tests with `python -m pytest tests -q`. Some skip when no reference build is
   present. Say which.
6. This step is optional. Do it when the user asks, and always before stage 4's `dbns` and
   `dcls` return types. Run `python make_excess_blocks.py --mode stage1 --verify`, then
   `python make_excess_blocks.py --mode stage1 --benchmark all`.
7. This step is optional too, and is for sharing the panel. Run
   `python make_release.py --what panel`. It blanks `permco` and `gvkey`, reduces the ratings
   to investment grade and high yield, and refuses to write a file that still holds licensed
   values.

## Resuming and testing

`python _run_stage2.py --from-step N` restarts at step N. For a quick test on a small sample,
run `python _run_stage2.py --limit-cusips 200`. It writes to the same `output/` folder and
replaces a full build's panel, so run it before the full build, not after.

## Stop and report when

- the dry run fails. Show its CONFIGURATION ERROR box and explain it.
- the column contract, the coverage check or the redaction fails. Each of these failures is
  the answer, and none is an obstacle to get past.

## Never

- run the steps yourself in one Python process. The runner starts them in fresh processes,
  and a long-lived process makes DuckDB lose its parallelism.
- edit any file other than `_stage2_settings.py`, or use anything other than the documented
  flags and environment variables, to get a build through.
- claim a `public` build reproduces a published panel.

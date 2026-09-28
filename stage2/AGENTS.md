# Stage 2 with an AI assistant

Stage 2 builds the monthly bond asset pricing panel from stage 1's daily panel, on the user's own
computer. Read the repository's [AGENTS.md](../AGENTS.md) first; this file adds what is specific to
stage 2. The full human guide is [QUICKSTART_stage2.md](QUICKSTART_stage2.md).

The `build-panel` skill (`/build-panel` in Claude Code, `$build-panel` in Codex) walks this file
step by step, and `python doctor.py`, from the repository root, says whether this stage's inputs
are ready.

## Ask the user one question before building: which factors?

Stage 2 needs a monthly factor panel (Fama-French, VIX, uncertainty indices, the BBW bond
factors). There are two sources, and they give different numbers:

| `--factor-source` | what it does | use it when |
|---|---|---|
| `public` (default) | assembles the factors from the live public sources | the user wants the newest data |
| `pinned` | downloads the factor panel published for this vintage (from the osbap-site GitHub release, the file openbondassetpricing.com links to) | the user wants the factors a published panel used |

The public sources revise their history (Ken French restates factors, FRED re-adjusts CPI,
Ludvigson re-estimates uncertainty), so a `public` build will not match a published release
bit for bit. That is expected, not a bug. For a `pinned` build to match as closely as it can, the
environment also needs the package versions in `constraints-2026.txt`
(`pip install -r requirements-local.txt -c constraints-2026.txt`); six columns still differ in
the 14th digit, as two builds on one machine do.

## The commands, in order

Run from `stage2/`. If the cache under `stage2/data/` is empty, the first run needs the WRDS
username, in the root `config.py` or `export WRDS_USERNAME=...`; the dry run says if it is
missing.

```bash
python _run_stage2.py --dry-run [--factor-source pinned]   # 1. resolve and print every input
python _run_stage2.py [--factor-source pinned]             # 2. the build, about 8-18 min
python validate_coverage.py                                # 3. every column reaches the panel's end
python -m pytest tests -q                                  # 4. the stage's own tests
python make_excess_blocks.py --mode stage1 --verify       # 5. optional: other Treasury benchmarks
python make_release.py --what panel                        # 6. optional: the redacted public panel
```

1. **Dry run.** Show the user the resolved paths: the daily panel, the FISD file, the callable
   flags, and the factor source (with `pinned`, the file or URL it will use). The three files come
   from the user's own stage 0/1 folder, from one WRDS run. They can carry two stamps: stage 1
   names its files by the day it ran, the FISD file carries the day stage 0 ran (the 2026-09-27
   run: `stage1_20260927`, `trace_enhanced_fisd_20260926`). Stage 2 takes the newest of each on
   its own and does not compare them, so check the list shows no file from an older run.
2. **Build.** Run it in the background with a log. It runs the 7 steps in fresh processes: step
   1, step 2, then steps 3-4 and 5-6 as two processes side by side, then step 7. It prints a
   line as each starts and ends. It checks its own output at the end: the 145
   column names and their order are fixed in `lib/contract.py`, and a build that changes them
   fails.
3. **Coverage.** Every column should reach the panel's last month. A column whose source stops
   publishing early is listed under "UPSTREAM-LIMITED" with the reason (for the 2026 vintage,
   `b_cptlt`: He-Kelly-Manela end in 2025-05). That is not a failure. Anything that ends early
   WITHOUT a named upstream reason is a problem to report.
4. **Tests.** They should all pass; some are skipped when a reference build is not present.
5. **Optional: other Treasury benchmarks.** `python make_excess_blocks.py --mode stage1 --verify`,
   then `--benchmark all`, re-estimates the 68 beta and momentum columns on the two alternative
   Treasury benchmarks (see "Alternative Treasury benchmarks" in [README_stage2.md](README_stage2.md)). It builds
   the `_bns`/`_cls` factor twins itself (step 3 again, per benchmark) and takes their history
   before 2002-08 from the extended BBW series, so the factor source makes no difference here.
6. **Release.** Writes the version that may be shared: `permco` and `gvkey` blanked, the composite
   ratings reduced to investment grade / high yield. It refuses to write a file that still carries
   licensed values. The user's own unredacted panel stays in `output/panel/`. Plain
   `make_release.py` means `--what all`, which also builds the BBW bundle when step 5 has been
   run, and otherwise skips it, says so and exits 1 after writing the other bundles.

## Reproducing a published panel: what "identical" means

This needs the same WRDS run of stages 0 and 1 as the published panel, which only its builder
has. A user's own run ends later and carries WRDS's revisions, so `pinned` then removes the
factor difference and nothing else. Say so before a user compares their panel with a published
one.

With `--factor-source pinned`, a build from the same WRDS run reproduces the published panel
once the release redaction is applied (`permco`, `gvkey`, `spc_rat`, `mdc_rat`; step 6):
same rows, same columns, and identical values in all but six liquidity columns (`cs_sprd`,
`spd_rel`, `spd_abs`, `ar_sprd`, `p_fht`, `vov`). Those six can differ by less than 1e-13 on a
few hundred rows, because DuckDB adds numbers across threads in whatever order the threads
finish. That is floating-point rounding, not a different result.

## What the user ends up with

```
stage2/output/panel/main_panel_<mode>.parquet   the 145-column panel (mode is "stage1" by default)
stage2/output/blocks/<mode>/                    betas, momentum, returns, factors, the _mmn sidecar
```

Every column is defined in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## When something fails

See the table under "If something goes wrong" in [QUICKSTART_stage2.md](QUICKSTART_stage2.md). To
resume after a failure, `python _run_stage2.py --from-step N` restarts at step N. A quick test on a
small sample: `python _run_stage2.py --limit-cusips 200`, which writes to the same `output/`
folder and replaces a full build's panel, so run it before the full build, not after.

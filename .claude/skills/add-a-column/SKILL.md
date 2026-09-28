---
name: add-a-column
description: Add a column (a signal or bond characteristic) to the 145-column monthly panel of trace-data-pipeline, or rename, move or remove one - the step that computes it, the frozen column contract, the unadjusted _mmn twin of a price-based signal, the definition in the data dictionary and the Table IA.VIII spec, the code tag, the redaction, and the release record. Use when the user wants to change which columns the monthly panel carries.
---

# Add a column to the monthly panel

Read first: `AGENTS.md` ("Before changing code"), the docstring of `stage2/lib/contract.py`,
and `stage2/tests/test_propagation_drill.py`, which skips each step below in turn and shows the
check that then fails.

The panel's column names and their order are published. Adding, removing or moving a column is
a public change: a CHANGELOG entry and a new vintage, never a silent edit.

## Use when

- the user wants a new signal or characteristic in `main_panel_<mode>.parquet`
- or wants a panel column renamed, moved or removed: the same places, changed the other way

## Do not use when

- the user asks what an existing column is: use `explain-pipeline`
- the variable is for the user's own analysis, not the published panel: compute it from the
  built panel in their own code, and change nothing here

## Steps

1. **Say what it is computed from, before writing code.**
   - The step that owns its family: returns and month-end signals (step 1), liquidity and
     within-month risk (step 2), betas (step 4), value and spread momentum (step 5), momentum
     and reversal (step 6). `CODE_MAP.md` lists the files; each step writes a block that step 7
     joins.
   - Whether it is **price-based**, that is, read from a month-end price. If so, it is built
     twice (step 3 below).
   - Whether it needs more than a month-end price and the bond's terms: trade prints, daily
     returns, trade dates, or factors built from the tape. If so, add it to the matching
     `REQUIRES_*` group in `stage2/lib/contract.py`.
2. **Compute it** in that step, and put a `col.<name>` tag, with a one-line definition, in a
   comment on the line that computes it, written as the tags beside it are. `tests/test_tags.py`
   fails on a panel column with no tag. Steps 1 and 2 pass on only the columns they name: step
   2 writes those in `SIGNAL_ORDER` (`stage2/steps/step2_illiquidity.py`), and step 7 keeps
   step 1's columns only from the lists in `stage2/lib/wrangle.py` (`end_cols`,
   `signal_char_cols`, `sig_keep_cols`). Add the name there too, or it never reaches the panel
   and the contract check reports it MISSING.
3. **A price-based signal ships twice.** Compute the gap-adjusted form too, as `<name>_adj`
   in the adjusted frame. Step 7 (`swap_adj_signals` in `stage2/lib/wrangle.py`) puts the
   adjusted form in the panel and writes the unadjusted one to the sidecar as `<name>_mmn`.
   Add the name to `MMN_TWINNED`. Step 7 fails if a declared twin is missing, if a twin ships
   undeclared, or if the panel carries the unadjusted form.
4. **The contract.** Add the name to `PANEL_COLUMNS` in `stage2/lib/contract.py`, where the
   build puts it. The columns after `sig_gap` come in the order step 7 joins the blocks, and a
   build whose order differs from the contract stops and prints both positions. Moving any
   other column to make room is itself a change to the published order.
5. **The definition, in two places that must agree.**
   - A row in `stage3/spec/signal_definitions.json`: `mnemonic`, `group` (one of the groups
     already there), `name`, `description`, and the citation fields. A citation key must be
     defined in the report's bibliography, `get_references_bib()` in `stage2/_report_helpers.py`.
   - The same name and description in the Signal Definitions section of
     `stage2/DATA_DICTIONARY.md`, under the same group.

   `stage2/tests/test_signal_definitions.py` compares the two word by word. The data report's
   Table 7 and stage 3's Table IA.VIII are printed from the spec, so neither needs an edit.
6. **Licensed data.** A column that carries, or is derived from, agency ratings, `permco` or
   `gvkey` goes into `REDACT_NULL` or `REDACT_RATINGS` in `stage2/make_release.py`, with a line
   in the dictionary's Redaction section. `permno` may be published.
7. **Coverage.** If the column's source stops publishing before the panel's last month, add it
   to `UPSTREAM_LIMITED` in `stage2/validate_coverage.py` with the reason. Otherwise leave that
   list alone: a column that ends early for no named reason must fail.
8. **The counts this column moves.** The panel's column count, 145, is in two tests
   (`stage3/tests/test_stage3_contract.py`, `tests/test_tags.py`), which change with the column,
   and in the docs: `git grep -n "145" -- "*.md"` and change each line that counts the panel's
   columns. A new beta or momentum column is also estimated on the duration-adjusted return and
   on the other Treasury benchmarks, so it lands in the `betas_x` or `mom_retx` block and in the
   `_bns`/`_cls` blocks `stage2/make_excess_blocks.py` writes; check that it does. Stage 4 lists
   those 68 columns (51 betas, then 17) by name, under `swap_columns` in
   `stage4/spec/factors.json`, and refuses a block column it does not know
   (`stage4/factorlib/inputs.py`): add the name there, and change the 68 (and the split at 51,
   for a beta) in `stage4/tests/test_stage4.py` and in the docs.
9. **Build and check**, from `stage2/`:
   - copy `output/panel/main_panel_stage1.parquet` aside first, if there is one;
   - `python _run_stage2.py --limit-cusips 200`, a quick build that reaches the contract check
     in step 7. It replaces a full build's panel, so the full build comes after it;
   - `python _run_stage2.py`, then `python validate_coverage.py`;
   - every column the old panel had must be unchanged in the new one: compare them on `cusip`
     and `date`;
   - from the repository root, `python -m pytest stage2/tests tests stage3/tests stage4/tests -q`.
10. **Record it.** `python tools/tags.py` rewrites `TAGS.md`. Write the `CHANGELOG.md` entry:
    the column, what it is, why, and that the release is a new vintage.

## Stop and report when

- the column needs a data source the pipeline does not already download: name it, and ask
  before adding a download
- a check still fails after you have made the change it asks for: show its message

## Never

- add the column in one place only: each place was once skipped, which is why the drill exists
- ship a price-based signal without its `_mmn` twin, or put the unadjusted form in the panel
- write `PANEL_COLUMNS` from a built panel: a list read off the panel cannot catch a reordering
- weaken a check to get a build through; the only counts to change are the ones step 8 names
- change the values of an existing column as a side effect

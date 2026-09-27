---
name: explain
description: Explain what trace-data-pipeline computes and where - a column of the daily or monthly panel, a cleaning filter, a factor, a setting, a stage 3 exhibit, a check that failed, or a known trap - from the code and the data dictionaries, citing file and line. Use for any "what is", "how is it computed", "where does it come from" or "why does it look like this" question about the data or the code.
---

# Explain the data and the code

Answer from this repository, never from memory. Every answer names the file and line it rests
on. The paper is being rewritten and is not part of this repository: explain what the code
computes, never what the paper concludes.

## Use when

- the user asks what a column, filter, factor, setting, exhibit or check is, how it is built,
  or why a value looks the way it does

## Do not use when

- the user wants something run: use `onboard`, `run-wrds`, `build-panel`,
  `reproduce-exhibits` or `build-factors`

## Where to start

| the question is about | start at | then read |
|---|---|---|
| a monthly panel column | its tag: `grep -rn "tag:col.<name>" .` | its row in `stage2/DATA_DICTIONARY.md` |
| a daily panel column | its tag: `grep -rn "daily.<name>" .` | its row in `stage1/DATA_DICTIONARY.md` |
| a cleaning filter | its tag, in the `filter` section of `TAGS.md` | the order in `stage0/AGENTS.md` or `stage1/AGENTS.md` |
| a TRACE-only factor | `stage4/DATA_DICTIONARY.md` | `stage4/spec/factors.json`, `stage4/README_stage4.md` |
| a table or figure of stage 3 | `stage3/INDEX.md`: exhibit to the file that makes it | that file's docstring, `stage3/DATA_DICTIONARY.md` |
| a setting | the stage's settings file (`INDEX.md`, "Changing settings") | the settings section of the stage's README |
| a check that failed | its tag, in the `rule` section of `TAGS.md` | the check's own message |
| a trap | the `trap` section of `TAGS.md` | "Traps" in `AGENTS.md` |
| what a code file does | `CODE_MAP.md` | the file's docstring |
| what changed, and when | `CHANGELOG.md` | `git log -p` on the file |

`TAGS.md` lists every tag with its file and a one-line description. A group, rather than a
tag, is the same thing in several places (the two stage 0 cleaners): read every member.

## How to answer

1. Open the code at the tag and read enough around it to follow the computation and its inputs.
2. Read the data dictionary's definition. If the code and the dictionary disagree, say so and
   quote both: the code is what the data is.
3. If the user has the data, check a claim on it (a count, a few rows) before stating it.
4. Say: what it is, how it is computed (the formula in words), its inputs, the stage and
   `file:line`, and any caveat below that applies.

## Easy to get wrong

- **The 38 price-based signals in the main panel are the gap-adjusted form** (`MMN_TWINNED`
  in `stage2/lib/contract.py`). Yields, spreads, duration, size and value are read at the
  signal trade, at least one session before the month-end price (`sig_dt`, `sig_gap`); the
  liquidity and within-month risk measures leave out the bond's last day of trading in the
  month. The unadjusted twin of each is in the `_mmn` sidecar. Use the main panel's form with
  `ret_vw` and the twin with `ret_vw_bgn`: "The Three Approaches" in `stage2/DATA_DICTIONARY.md`.
- **The betas are 36-month rolling regressions** of the bond's total return `ret_vw`, with at
  least 12 months (`BETA_WINDOW`, `BETA_MIN_OBS` in `stage2/_stage2_settings.py`). Each model
  is one row of `BETA_MODELS` in `stage2/lib/betas.py`, and names its other regressors.
- **The published panel is redacted**: `permco` and `gvkey` are blank and the two ratings are
  reduced to 1 (investment grade) and 11 (high yield). A panel the user builds has them all.
- **Stage 0 has two cleaners**, Enhanced in one file and Standard and 144A in the other, with
  the same filters in each.
- **Stage 3's folder numbers are not the paper's section numbers**: `s1_lib` is Section 3,
  `s2_lab` Section 4 and `s3_nse` Section 5 (`stage3/INDEX.md`).

## Never

- say what the paper finds, or quote it
- give a formula you have not read in the code: say what you read and what is still unclear

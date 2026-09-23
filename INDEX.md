# Where to look

A task-by-task guide to the docs, for people and for AI assistants. What each code file does is in
[CODE_MAP.md](CODE_MAP.md). Which file makes each table and figure of the paper is in
[stage3/INDEX.md](stage3/INDEX.md), the exhibit index; this page is the index of the docs.

## Running the pipeline

| to... | read | run |
|---|---|---|
| see which stage runs where, and why | [README: Which machine am I on?](README.md#which-machine-am-i-on) | |
| run stages 0 and 1 on WRDS | [QUICKSTART.md](QUICKSTART.md) | `./run_pipeline.sh` |
| check the chain works before a 5-hour run | [CONTRIBUTING: Testing](CONTRIBUTING.md#testing) | `qsub run_smoke_test.sh` |
| copy the results to your own computer | [QUICKSTART: Download Results](QUICKSTART.md#download-results-to-your-local-machine) | `zip` on WRDS, then `scp` |
| build the monthly panel | [stage2/QUICKSTART_stage2.md](stage2/QUICKSTART_stage2.md) | `python stage2/_run_stage2.py` |
| reproduce a published panel exactly | [stage2/AGENTS.md](stage2/AGENTS.md) | `_run_stage2.py --factor-source pinned` |
| share a panel you built | [stage2/README_stage2.md](stage2/README_stage2.md) | `python stage2/make_release.py --what panel` |
| rebuild the betas on another Treasury benchmark | [README_stage2: Alternative Treasury benchmarks](stage2/README_stage2.md#alternative-treasury-benchmarks-rebuilding-the-betas) | `python stage2/make_excess_blocks.py` |
| reproduce the paper's tables and figures | [stage3/QUICKSTART_stage3.md](stage3/QUICKSTART_stage3.md) | `bash stage3/run_stage3.sh` |
| run a stage with Claude Code or Codex | [AGENTS.md](AGENTS.md) | |

## Understanding the data

| to... | read |
|---|---|
| look up a stage 0 column (the daily TRACE panel) | [stage0/DATA_DICTIONARY.md](stage0/DATA_DICTIONARY.md) |
| look up a stage 1 column (the daily bond panel) | [stage1/DATA_DICTIONARY.md](stage1/DATA_DICTIONARY.md) |
| look up a monthly panel column or a factor model | [stage2/DATA_DICTIONARY.md](stage2/DATA_DICTIONARY.md) |
| look up a stage 3 output file | [stage3/DATA_DICTIONARY.md](stage3/DATA_DICTIONARY.md) |
| find the code behind a table or figure of the paper | [stage3/INDEX.md](stage3/INDEX.md) |
| understand the decimal-shift corrector | [stage0/README_decimal_shift_corrector.md](stage0/README_decimal_shift_corrector.md) |
| understand the bounce-back filter | [stage0/README_bounce_back_filter.md](stage0/README_bounce_back_filter.md) |
| understand the distressed-bond filter | [stage1/README_distressed_filter.md](stage1/README_distressed_filter.md) |
| understand returns for defaulted bonds | [stage2/README_Default.md](stage2/README_Default.md) |
| understand the value signals | [stage2/README_Value.md](stage2/README_Value.md) |
| know what the public download leaves out, and why | [FAQ: What is redacted in the published panel?](FAQ.md#what-is-redacted-in-the-published-panel) |

## Changing settings

| to change... | edit | read |
|---|---|---|
| your WRDS username, or which TRACE databases to process | `config.py` | [FAQ: Configuration](FAQ.md#configuration) |
| stage 0's filters, connections and job sizes | `stage0/_trace_settings.py` | [README_stage0: Configuration](stage0/README_stage0.md#configuration-choices-you-can-edit) |
| stage 1's settings | `stage1/_stage1_settings.py` | [README_stage1: Configuration](stage1/README_stage1.md#configuration-choices-you-can-edit) |
| stage 2's settings | `stage2/_stage2_settings.py` | [README_stage2: Configuration](stage2/README_stage2.md#configuration) |
| stage 3's inputs | `stage3/_stage3_settings.py` or environment variables | [QUICKSTART_stage3: environment variables](stage3/QUICKSTART_stage3.md#every-environment-variable-stage-3-reads) |
| stage 3's sample (full panel or the paper's window) | `--sample` on `_run_stage3.py` | [QUICKSTART_stage3: Which sample](stage3/QUICKSTART_stage3.md#which-sample) |

## When something fails

| stage | read |
|---|---|
| 0 | [README_stage0: Troubleshooting](stage0/README_stage0.md#troubleshooting), [FAQ: Troubleshooting](FAQ.md#troubleshooting) |
| 1 | [README_stage1: Troubleshooting](stage1/README_stage1.md#troubleshooting), [FAQ: Troubleshooting](FAQ.md#troubleshooting) |
| 2 | [QUICKSTART_stage2: If something goes wrong](stage2/QUICKSTART_stage2.md#if-something-goes-wrong) |
| 3 | [QUICKSTART_stage3: If something goes wrong](stage3/QUICKSTART_stage3.md#if-something-goes-wrong) |

## Working on the code

| to... | read |
|---|---|
| find what a code file does | [CODE_MAP.md](CODE_MAP.md) |
| run the tests before a pull request | [CONTRIBUTING: Testing](CONTRIBUTING.md#testing) |
| see what changed in each release | [CHANGELOG.md](CHANGELOG.md) |

# Where to look

A task-by-task guide to the docs, for people and for AI assistants. What each code file does is in
[CODE_MAP.md](CODE_MAP.md). Which file makes each table and figure of the paper is in
[stage3/INDEX.md](stage3/INDEX.md), the exhibit index; this page is the index of the docs.

## Running the pipeline

| to... | read | run |
|---|---|---|
| see what is ready and what to run next | [AGENTS.md: Start here](AGENTS.md#start-here) | `python doctor.py` (`--wrds` on WRDS) |
| see which stage runs where, and why | [README: Which machine am I on?](README.md#which-machine-am-i-on) | |
| run stages 0 and 1 on WRDS | [QUICKSTART.md](QUICKSTART.md) | `./run_pipeline.sh` |
| check the chain works before a 5-hour run | [CONTRIBUTING: Testing](CONTRIBUTING.md#testing) | `qsub run_smoke_test.sh` |
| copy the results to your own computer | [QUICKSTART: Download Results](QUICKSTART.md#download-results-to-your-local-machine) | `zip` on WRDS, then `scp` |
| build the monthly panel | [stage2/QUICKSTART_stage2.md](stage2/QUICKSTART_stage2.md) | `python stage2/_run_stage2.py` |
| use the factors a published panel was built with | [stage2/AGENTS.md](stage2/AGENTS.md), [constraints-2026.txt](constraints-2026.txt) | `_run_stage2.py --factor-source pinned`, in an environment installed with `-c constraints-2026.txt` |
| share a panel you built | [stage2/README_stage2.md](stage2/README_stage2.md) | `python stage2/make_release.py --what panel` |
| rebuild the betas on another Treasury benchmark | [README_stage2: Alternative Treasury benchmarks](stage2/README_stage2.md#alternative-treasury-benchmarks-rebuilding-the-betas) | `python stage2/make_excess_blocks.py` |
| reproduce the paper's tables and figures | [stage3/QUICKSTART_stage3.md](stage3/QUICKSTART_stage3.md) | `bash stage3/run_stage3.sh` |
| build the TRACE-only bond factors, and check them against the published ones | [stage4/README_stage4.md](stage4/README_stage4.md) | `bash stage4/run_stage4.sh` |
| install what stages 2-4 need | [requirements-local.txt](requirements-local.txt) | `pip install -r requirements-local.txt`, then `pip install --no-deps pybondlab==0.3.0` |
| run a stage with Claude Code, Codex or another assistant | [AGENTS.md](AGENTS.md), and the skills it lists | `/build-panel` in Claude Code, `$build-panel` in Codex |
| see what has been run end to end, and when | [docs/validation/validated_runs.csv](docs/validation/validated_runs.csv) | |

## Understanding the data

| to... | read |
|---|---|
| look up a stage 0 column (the daily TRACE panel) | [stage0/DATA_DICTIONARY.md](stage0/DATA_DICTIONARY.md) |
| look up a stage 1 column (the daily bond panel) | [stage1/DATA_DICTIONARY.md](stage1/DATA_DICTIONARY.md) |
| look up a monthly panel column or a factor model | [stage2/DATA_DICTIONARY.md](stage2/DATA_DICTIONARY.md) |
| look up a stage 3 output file | [stage3/DATA_DICTIONARY.md](stage3/DATA_DICTIONARY.md) |
| look up a column of the TRACE-only factor files | [stage4/DATA_DICTIONARY.md](stage4/DATA_DICTIONARY.md) |
| find the code behind a table or figure of the paper | [stage3/INDEX.md](stage3/INDEX.md) |
| find the line that computes a column, applies a filter or enforces a rule | [TAGS.md](TAGS.md), or `git grep -n "tag:col.<name>"` |
| understand the decimal-shift corrector | [stage0/README_decimal_shift_corrector.md](stage0/README_decimal_shift_corrector.md) |
| understand the bounce-back filter | [stage0/README_bounce_back_filter.md](stage0/README_bounce_back_filter.md) |
| understand the distressed-bond filter | [stage1/README_distressed_filter.md](stage1/README_distressed_filter.md) |
| understand returns for defaulted bonds | [stage2/README_Default.md](stage2/README_Default.md) |
| understand the value signals | [stage2/README_Value.md](stage2/README_Value.md) |
| know what the public downloads leave out, and why | [FAQ: What is redacted in the published panel?](FAQ.md#what-is-redacted-in-the-published-panel) for the monthly panel; the note at the top of [stage1/DATA_DICTIONARY.md](stage1/DATA_DICTIONARY.md) for the 32-column daily file |

## Changing settings

| to change... | edit | read |
|---|---|---|
| your WRDS username, or which TRACE databases to process | `config.py` | [FAQ: Configuration](FAQ.md#configuration) |
| stage 0's filters, connections and job sizes | `stage0/_trace_settings.py` | [README_stage0: Configuration](stage0/README_stage0.md#configuration-choices-you-can-edit) |
| stage 1's settings | `stage1/_stage1_settings.py` | [README_stage1: Configuration](stage1/README_stage1.md#configuration-choices-you-can-edit) |
| stage 2's settings | `stage2/_stage2_settings.py` | [README_stage2: Configuration](stage2/README_stage2.md#configuration) |
| stage 3's inputs | `stage3/_stage3_settings.py` or environment variables | [QUICKSTART_stage3: environment variables](stage3/QUICKSTART_stage3.md#every-environment-variable-stage-3-reads) |
| stage 3's sample (the whole panel or the paper's window) | `--sample frontier` or `--sample paper` on `_run_stage3.py` | [QUICKSTART_stage3: Which sample](stage3/QUICKSTART_stage3.md#which-sample) |
| stage 4's inputs and output folder | environment variables | [README_stage4: Settings](stage4/README_stage4.md#settings) |

## When something fails

| stage | read |
|---|---|
| 0 | [README_stage0: Troubleshooting](stage0/README_stage0.md#troubleshooting), [FAQ: Troubleshooting](FAQ.md#troubleshooting) |
| 1 | [README_stage1: Troubleshooting](stage1/README_stage1.md#troubleshooting), [FAQ: Troubleshooting](FAQ.md#troubleshooting) |
| 2 | [QUICKSTART_stage2: If something goes wrong](stage2/QUICKSTART_stage2.md#if-something-goes-wrong) |
| 3 | [QUICKSTART_stage3: If something goes wrong](stage3/QUICKSTART_stage3.md#if-something-goes-wrong) |
| 4 | [README_stage4: If something goes wrong](stage4/README_stage4.md#if-something-goes-wrong) |

## Working on the code

| to... | read |
|---|---|
| find what a code file does | [CODE_MAP.md](CODE_MAP.md) |
| add, rename or remove a monthly panel column | [AGENTS.md: Before changing code](AGENTS.md#before-changing-code), and the `add-a-column` skill |
| tag a new column or filter, and check the tags | [AGENTS.md: The code tags](AGENTS.md#the-code-tags) |
| run the tests before a pull request | [CONTRIBUTING: Testing](CONTRIBUTING.md#testing) |
| see what changed in each release | [CHANGELOG.md](CHANGELOG.md) |

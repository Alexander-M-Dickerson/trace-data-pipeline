# Working in this repository with an AI assistant

This file is for AI coding assistants (Claude Code, Codex, Cursor, Copilot and others) working
beside someone who runs the pipeline. `CLAUDE.md` imports it, so every tool reads the same
instructions. People should start at [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).

## What this repository does

It turns the WRDS TRACE corporate bond tape into a monthly bond asset pricing panel, then
reproduces the exhibits of *The Corporate Bond Factor Replication Crisis* and builds the
TRACE-only bond factors published on openbondassetpricing.com. It runs in five stages on two
machines:

| stage | where it runs | what it does | time |
|---|---|---|---|
| 0 and 1 | **WRDS Cloud** (SGE grid) | clean the raw tape, build the daily bond panel | about 5 h |
| hand-off | the user | zip the folder on WRDS, copy it to their own computer | -- |
| 2 | **the user's computer** | build the monthly panel (145 columns per bond-month) | about 8-18 min (the first run also downloads its inputs) |
| 3 | **the user's computer** | portfolio sorts and the paper's exhibits (tex tables, figures) | about 15-20 min |
| 4 | **the user's computer** | the TRACE-only bond factors, checked against the published files | about 8 min |

## Your job here

You are the data scientist running this pipeline with the user. That means three things:

1. **Run it.** Take the user from wherever they are to the next finished stage, one checked
   step at a time.
2. **Check it.** Every stage checks its own inputs and outputs. Read what each check prints
   and report it; a run that exits 0 has only finished, it has not been verified.
3. **Explain it.** Answer what a column, filter, factor or exhibit is from the code and the
   data dictionaries, citing `file:line`. Never answer from memory or from the paper: the paper
   is being rewritten and is not part of this repository.

## Start here

```bash
python doctor.py            # on the user's computer: what is ready, what is missing, the next step
python3 doctor.py --wrds    # the same on the WRDS login node, for stages 0 and 1
```

`doctor.py` only looks: it downloads nothing, opens no WRDS connection and writes nothing of its own. It
asks each stage through that stage's own check, so its answer is the stage's answer.

## Skills

Each is a step-by-step procedure in `.claude/skills/<name>/SKILL.md`, with the same file in
`.agents/skills/<name>/SKILL.md` for other tools. In Claude Code type `/<name>`; in Codex, `$<name>`.

| skill | use it to |
|---|---|
| `onboard` | take a fresh clone to a ready machine (local or WRDS) |
| `run-wrds` | run stages 0 and 1 on the WRDS grid, and bring the results home |
| `build-panel` | build the monthly panel (stage 2) |
| `reproduce-exhibits` | build the exhibits (stage 3) |
| `build-factors` | build the TRACE-only factors and compare them with the published ones (stage 4) |
| `explain-pipeline` | explain a column, filter, factor, setting, exhibit or trap, from the code |
| `add-a-column` | add, rename, move or remove a monthly panel column, in every place it must change |

## Where to look

| to find | read |
|---|---|
| which doc answers a question, and which command does a job | [INDEX.md](INDEX.md) |
| what each stage reads and writes, and what every code file does | [CODE_MAP.md](CODE_MAP.md) |
| where a column, filter, rule or trap lives: the file, and the tag to search it for | [TAGS.md](TAGS.md) |
| the code behind each table and figure of the paper | [stage3/INDEX.md](stage3/INDEX.md) |
| a column's definition | the stage's `DATA_DICTIONARY.md` (stages 0 to 4) |
| the questions users actually ask | [FAQ.md](FAQ.md) |
| what changed in each release | [CHANGELOG.md](CHANGELOG.md) |
| what has been run end to end, and when | [docs/validation/validated_runs.csv](docs/validation/validated_runs.csv) |

## The code tags

Key lines carry a named tag in a comment, in [tagref's](https://github.com/stepchowfun/tagref)
syntax: a tag in square brackets marks a place, and a ref points to it. Every column of the
monthly and daily panels has one where it is computed, as does each cleaning filter, each rule
a check enforces, each known trap and each entry point. To go from a name to the code:

```bash
git grep -n "tag:col.cs_sprd"         # where the Corwin-Schultz spread is computed
git grep -n "filter.bounce_back"      # the filter in both stage 0 cleaners, and every pointer to it
```

The namespaces are `col.` (monthly panel), `daily.` (daily panel), `filter.`, `rule.`, `trap.`
and `entry.`. A group, rather than a tag, marks the same thing done in several places (the two
stage 0 cleaners), which must be changed together. `python tools/tags.py` checks every tag and
rewrites [TAGS.md](TAGS.md), which lists every tag's file for a copy without git; the test suite fails when a ref points nowhere, a tag is
duplicated, or a panel column has no tag.

## Rules for the assistant

1. **Read the stage's `AGENTS.md` and its guide before running anything**: the `AGENTS.md` in
   that stage's folder (some tools read only this root file), then `stage2/QUICKSTART_stage2.md`,
   `stage3/QUICKSTART_stage3.md` or `stage4/README_stage4.md`. They hold the exact commands and
   the expected run times.
2. **Always dry-run first.** `python _run_stage2.py --dry-run`, `python _run_stage3.py --dry-run`
   and `python build_factors.py --dry-run` print every input they resolved. Show the user that
   list before building.
3. **Settings live in one file per stage**: `stage2/_stage2_settings.py`,
   `stage3/_stage3_settings.py`, `stage4/_stage4_settings.py`. Change settings there, or with the
   documented command-line flags and environment variables. Do not edit any other code to get
   past a check.
4. **A failing check is the answer, not an obstacle.** The stages check their own inputs and
   outputs: the column contract [ref:rule.column_contract], coverage, the redaction
   [ref:rule.redaction]. If one fails, stop, show the message, and explain its cause. Never
   weaken or skip a check to make a run finish.
5. **Run long builds in the background with a log file**, and report progress from the log. Use
   the runners (`_run_stage2.py`, `run_stage3.sh`, `run_stage4.sh`). Stages 2 and 3 start each
   step in a fresh process, which the pipeline relies on for speed [ref:rule.fresh_process].
6. **Ask the user, don't guess**, when a choice changes the numbers: the factor source for
   stage 2 [ref:trap.factor_source] and the sample for stage 3.
7. **Outputs are not committed.** `stage2/data/`, `stage2/output/`, `stage2/release/`,
   `stage3/data/`, `stage3/reports/` and `stage4/output/` are gitignored. Do not add them to git.

## Stages 0 and 1 on WRDS

If the user runs you on the WRDS Cloud, these are the rules that cost people a day when missed.
[QUICKSTART.md](QUICKSTART.md) has the full walk-through, and [stage0/AGENTS.md](stage0/AGENTS.md)
and [stage1/AGENTS.md](stage1/AGENTS.md) what each stage does.

- **Run `bash download_inputs.sh` on the login node.** Compute nodes have no internet, so stage 1
  cannot fetch its inputs itself [ref:trap.no_internet].
- **Before a full run, suggest `qsub run_smoke_test.sh`.** It runs the real code on a few CUSIP
  chunks in about 10 minutes and catches most setup problems.
- **Submit from the repository root**: `./run_pipeline.sh`. It submits the jobs in order, each
  waiting on the ones it needs.
- **Set `WRDS_USERNAME` and `TRACE_MEMBERS` in the root `config.py`** (or the environment). Every
  stage that needs them reads them from there.
- **Change stage 0's job sizes in `stage0/_trace_settings.py`** (`CONCURRENCY`, `MEM_PER_SLOT_GB`).
  `run_pipeline.sh` passes them to `qsub`, which overrides the settings inside the stage 0 job
  scripts, so editing those scripts changes nothing. Memory (`m_mem_free`) is charged per slot,
  and WRDS allows 8 slots and 48 GB per job; a request over that waits in the queue forever
  without an error, so `qsub_resources()` refuses to make one [ref:rule.memory_per_slot].
- **WRDS limits how many database connections one account holds at once**, measured at 7
  (`MAX_WRDS_CONNECTIONS`). The Enhanced and 144A jobs run at the same time, so their
  `CONCURRENCY` values together must stay below it [ref:rule.connection_cap]. A failed
  connection shows up as `EOFError: EOF when reading a line`, which looks like a keyboard
  problem. It has two causes: first check the username (`WRDS_USERNAME` unset or still the
  placeholder), which is the commoner one [ref:trap.eof_error]; only then this limit. Jobs on
  the WRDS Cloud need no password file, so do not send the user to make one.
- **To move the results off WRDS**, zip from `~` with a relative path, exactly as QUICKSTART writes
  it. An absolute path makes an archive nested four folders deep.

## What the user must provide

- The folder produced on WRDS by stages 0 and 1, unzipped on their computer. Stage 2 finds its
  three inputs by date stamp: `stage1/data/stage1_<YYYYMMDD>.parquet`,
  `stage0/enhanced/trace_enhanced_fisd_<YYYYMMDD>.parquet`,
  `stage1/data/call_dummy_<YYYYMMDD>.parquet`.
- A WRDS account (`WRDS_USERNAME`, in `config.py` or the environment) for stage 2's first run,
  which fetches and caches Treasury returns, Fama-French factors, VIX and FISD coupon terms.
  Later runs use the cache.
- Python 3.11 to 3.13 on their computer (3.14 cannot install yet), in an environment made for this
  repository (the `onboard` skill makes one, `.venv`, and calls its Python by path), and two
  install lines, in this order:
  ```
  python -m pip install -r requirements-local.txt
  python -m pip install --no-deps pybondlab==0.3.0
  ```
  PyBondLab 0.3.0 declares `numpy<2` and this repository installs numpy 2, so it goes in
  without its dependency list; `requirements-local.txt` says why that is safe. Stages 2-4 stop
  at start-up and print these two lines if PyBondLab is missing or a different version
  [ref:rule.pybondlab_pin].

## Traps that have caught people

- **The daily file downloaded from openbondassetpricing.com cannot drive stage 2.** Its rating
  columns are removed (they are licensed), and stage 2 keeps only rated bond-months, so it would
  produce an empty panel. Stage 2 refuses to start on it [ref:trap.public_daily_file]. The user
  must run stages 0 and 1.
- **A fresh default build does not reproduce a published panel.** Stage 2's default builds its
  factors from live public sources, which revise their history. To reproduce a published vintage
  as closely as it can be, use `--factor-source pinned`; stage2/AGENTS.md says how close.
- **Do not run the stage 2 steps yourself in one Python process.** A long-lived process makes
  DuckDB lose its parallelism and steps take several times longer. The runner starts the steps in
  fresh processes.
- **A cached download is reused whenever it exists** [ref:trap.stale_cache]. Pointing a setting
  at a new file does not refetch by itself; each loader checks the cache before trusting it.

## Before changing code

1. Run `python -m pytest stage2/tests tests stage3/tests stage4/tests -q`. It needs no WRDS or
   network, takes under a minute, and includes `tests/test_docs.py`, which fails when a doc stops
   matching the code. [tests/AGENTS.md](tests/AGENTS.md) says what each gate guards.
2. **A new or renamed `.py` or `.sh` file** goes into [CODE_MAP.md](CODE_MAP.md).
3. **A new panel column** (the `add-a-column` skill walks it) needs its name in `stage2/lib/contract.py` at the right position, the
   same definition in `stage2/DATA_DICTIONARY.md` and `stage3/spec/signal_definitions.json`,
   and a `col.` tag on the line that computes it. A price-based signal also needs its
   unadjusted `_mmn` twin and a place in `MMN_TWINNED`. The column list is published, so a
   change to it is a CHANGELOG entry and a new vintage. **A new filter** needs a `filter.`
   tag. Then run `python tools/tags.py` to rewrite TAGS.md.
4. Nothing in this repository names a private project, a person's path or a machine:
   `tests/test_public_boundary.py` fails on it. Write paths relative to the repository.

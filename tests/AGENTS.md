# The tests, with an AI assistant

Read the repository's [AGENTS.md](../AGENTS.md) first. This folder holds the checks that run
without WRDS, data or a network; each stage keeps its own beside it (`stage2/tests`,
`stage3/tests`, `stage4/tests`). All of them run in under a minute:

```bash
python -m pytest stage2/tests tests stage3/tests stage4/tests -q
python tests/test_chunk_plan.py; python tests/test_chunk_scheduler.py; python tests/test_merge_keys.py
```

The last three are scripts, not pytest modules: pytest collects nothing from them, so run them
directly.

Some skip themselves when what they need is absent (git, bash, a reference build). A skip is
not a pass: say which ones skipped and why.

## Rules

1. **A failing test is a finding.** Read its message, find the cause, and fix the cause. Never
   loosen an assertion, add a skip, or widen a tolerance to make it pass without saying so and
   why.
2. **Many of these tests are gates**: they fail on a whole class of mistake, not one bug. Each
   has been seen to fail on a planted example. When you add one, plant the mistake it is meant
   to catch, watch it fail, then remove the plant.
3. **Tests that plant text spell it in pieces** (`"[" + "tag:"`) so the repository's own checks
   do not mistake the plant for the real thing.

## What each file in this folder guards

| file | guards |
|---|---|
| `test_docs.py` | the docs against the code: every code file in CODE_MAP.md, output trees, links, anchors, quoted counts |
| `test_public_boundary.py` | nothing private in any tracked file: no private project names, personal paths or other drives, and no pointer to a file the repository does not have |
| `test_tags.py` | the code tags: refs resolve, no duplicates, every panel column tagged, TAGS.md current |
| `test_skills.py` | the assistant skills: the two copies identical, the frontmatter valid, every file they name present, and AGENTS.md and CLAUDE.md listing exactly these skills |
| `test_doctor.py` | `doctor.py`: the Python range, that it never connects to WRDS, and the next step in each state |
| `test_environment.py` | `numeric_setup.py` and `pybondlab_pin.py`, which make a run independent of the machine |
| `test_download_inputs.py` | `download_inputs.sh`: a failed download keeps the earlier file, and `--check` downloads nothing |
| `test_disk_check.py` | `check_disk_space.sh`, which decides whether a WRDS run starts |
| `test_chunk_plan.py`, `test_chunk_scheduler.py` (scripts) | stage 0's chunks: a true partition of the CUSIPs, returned in order |
| `test_worker_override.py` | `STAGE0_WORKERS` reaches the connection check and the job request, and one over the cap is refused |
| `test_cut_off_basis.py` | `DATE_CUT_OFF` is measured from the least current source |
| `test_linker_window.py` | stage 1 joins firm ids on the identity window |
| `test_merge_keys.py` (a script) | every lookup stage 1 joins has one row per key |

`smoke_assertions.py` is not a pytest file: `run_smoke_test.sh` runs it on WRDS after a small
real run. `probe_wrds_connections.py` measures an account's connection limit, on WRDS, by hand.

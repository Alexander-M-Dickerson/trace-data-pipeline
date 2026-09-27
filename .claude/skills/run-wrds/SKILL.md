---
name: run-wrds
description: Run stages 0 and 1 of trace-data-pipeline on the WRDS Cloud grid - the smoke test, the full run, watching the jobs, reading their logs, and bringing the results home. Use when the user is on the WRDS login node and wants the daily bond panel built, or a WRDS job has failed or will not start.
---

# Run stages 0 and 1 on WRDS

Read first: `AGENTS.md` (the section "Stages 0 and 1 on WRDS"), `stage0/AGENTS.md`,
`stage1/AGENTS.md`. The full walk-through is `QUICKSTART.md`.

## Use when

- the user is on the WRDS login node, set up (see `onboard`), and wants stages 0 and 1 run
- a stage 0 or stage 1 job failed, or sits in the queue and never starts

## Do not use when

- the machine is not set up yet: use `onboard` first
- the user is on their own computer: stages 2 to 4 are `build-panel`, `reproduce-exhibits`
  and `build-factors`

## Steps

1. `python3 doctor.py --wrds`. Every line should be `[ok]`; if not, go back to `onboard`.
2. **Offer the smoke test first**, about 10 minutes, on a few CUSIP chunks with the real code:
   ```bash
   bash download_inputs.sh        # login node only: compute nodes have no internet
   qsub run_smoke_test.sh         # its verdict is at the end of smoke_test.out
   ```
   Read `smoke_test.out` and report the verdict and anything it flags.
3. **The full run**, from the repository root: `./run_pipeline.sh`. It checks disk space,
   fetches the inputs, and submits every job in order, each waiting on the ones it needs.
   About 5 hours.
4. **Watch it** with `qstat` and the logs: `stage0/logs/01_enhanced.out`,
   `stage0/logs/03_144a.out`, `stage1/logs/stage1.out`, and the `.err` files beside them.
   Report progress from the logs, not from guesses.
5. **Check the result**: `ls -lh stage1/data/stage1_*.parquet`, and the data reports under
   `stage0/data_reports/` and `stage1/data_reports/`.
6. **Bring it home**, exactly as "Download Results to Your Local Machine" in `QUICKSTART.md`
   writes it: `cd ~` first, then zip the folder by its RELATIVE name into scratch, then `scp`.

## When a job fails or will not start

- **Stuck at `qw`**: the memory request. `m_mem_free` is charged per slot, and a job asking
  for more than 8 slots or 48 GB waits forever with no error. Lower `CONCURRENCY` in
  `stage0/_trace_settings.py` and resubmit; `qsub_resources()` derives the request from it.
- **`EOFError: EOF when reading a line`**: first the login (the username unset or still the
  `config.py` placeholder, or no `~/.pgpass` entry), then the connection cap (7 held at once
  per account).
- **Any other failure**: the job's `.err` log, then "Troubleshooting" in
  `stage0/README_stage0.md` or `stage1/README_stage1.md`, and `FAQ.md`.

## Stop and report when

- the smoke test fails: do not start the full run
- a filter or setting would have to change to get past an error: that changes the data, so
  it is the user's decision

## Never

- open a WRDS connection on the login node: every connection belongs inside a job
- edit the `#$` lines of the stage 0 job scripts to change their size: `run_pipeline.sh`
  overrides them from `stage0/_trace_settings.py`
- turn on the volume filter (`volume_filter_toggle`) unless the user asks: the published data
  keeps trades of every size

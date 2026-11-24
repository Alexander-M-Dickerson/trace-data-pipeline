---
name: run-wrds
description: Run stages 0 and 1 of trace-data-pipeline on the WRDS Cloud grid. Covers the smoke test, the full run, watching the jobs, reading their logs, and bringing the results home. Use when the user is on the WRDS login node and wants the daily bond panel built, or a WRDS job has failed or will not start.
---

# Run stages 0 and 1 on WRDS

First read `AGENTS.md` (the section "Stages 0 and 1 on WRDS"), `stage0/AGENTS.md` and
`stage1/AGENTS.md`. The full walk-through is `QUICKSTART.md`.

## Use when

- the user is on the WRDS login node, set up (see `onboard`), and wants stages 0 and 1 run
- a stage 0 or stage 1 job failed, or sits in the queue and never starts

## Do not use when

- the machine is not set up yet. Use `onboard` first.
- the user is on their own computer. Stages 2 to 4 are `build-panel`, `reproduce-exhibits`
  and `build-factors`.

## Steps

1. Run `python3 doctor.py --wrds`. Every line should be `[ok]`. If not, go back to `onboard`.
2. Offer the smoke test first. It takes about 10 minutes and runs the pipeline's own code on a
   few CUSIP chunks.
   ```bash
   bash download_inputs.sh        # login node only: compute nodes have no internet
   qsub run_smoke_test.sh         # its verdict is at the end of smoke_test.out
   ```
   Read `smoke_test.out` and report the verdict and anything it flags. Wait until `qstat` no
   longer lists it before the full run, because the two together hold more WRDS connections
   than an account may.
3. Start the full run from the repository root with `./run_pipeline.sh`. It checks disk space,
   fetches the inputs, and submits every job in order, each waiting on the ones it needs. It
   takes about 5 hours.
4. Watch it with `qstat` and the logs, which are `stage0/logs/01_enhanced.out`,
   `stage0/logs/03_144a.out`, `stage1/logs/stage1.out`, and the `.err` files beside them.
   Report progress from the logs, not from guesses. A finished stage 1 leaves about a hundred
   harmless `resource_tracker` tracebacks in `stage1.err`. Success is "Stage 1 processing
   completed successfully" at the end of `stage1.out`.
5. Check the result with `ls -lh stage1/data/stage1_*.parquet` and with the data reports under
   `stage0/data_reports/` and `stage1/data_reports/`.
6. Bring it home exactly as "Download Results to Your Local Machine" in `QUICKSTART.md` writes
   it. Run `cd ~` first, then zip the folder by its **relative** name into scratch, then
   `scp`.

## When a job fails or will not start

- For a job waiting in the queue, `hqw` means it is held behind another, which is normal. A
  long `qw` is usually the grid. WRDS runs at most 5 of your jobs at once, and
  `qstat -j <id>` says why a job waits. A request over 8 slots or 48 GB waits forever with no
  error, but `qsub_resources()` refuses to make one, so that happens only to a job submitted
  by hand with its own `-pe`/`-l`.
- For `EOFError: EOF when reading a line`, check the username first (unset, or still the
  `config.py` placeholder), then the connection cap (7 held at once per account). Jobs on the
  WRDS Cloud need no password file.
- If stage 1 failed and stage 0 finished, fix the cause and re-run stage 1 alone, from the
  repository root, with `mkdir -p stage1/logs && qsub stage1/run_stage1.sh`.
  `./run_pipeline.sh` would repeat stage 0 too, about 3 more hours.
- For any other failure, read the job's `.err` log, then "Troubleshooting" in
  `stage0/README_stage0.md` or `stage1/README_stage1.md`, and `FAQ.md`.

## Stop and report when

- the smoke test fails. Do not start the full run.
- a filter or setting would have to change to get past an error. That changes the data, so
  it is the user's decision.

## Never

- open a WRDS connection on the login node. Every connection belongs inside a job.
- edit the `#$` lines of the stage 0 job scripts to change their size. `run_pipeline.sh`
  overrides them from `stage0/_trace_settings.py`.
- turn on the volume filter (`volume_filter_toggle`) unless the user asks. The published data
  keeps trades of every size.

#!/bin/bash
#$ -S /bin/bash                 # Ensure the job uses Bash shell
#$ -V                           # Export environment variables (important for Python envs)
#$ -cwd                         # Run from current working directory
# Cores and memory.
#
# 5 slots because the Enhanced re-clean now pulls 5 chunks at once, one WRDS connection
# each (REPORTS_CONCURRENCY in _trace_settings.py). This job runs alongside stage 1
# since 2.2.2, so 5 here + stage 1's 1 = 6 against the measured ceiling of 7.
#
# ❗m_mem_free is charged PER SLOT: 5 x 8G = 40 GB, against the WRDS caps of 8 cores and
# 48 GB per job. An over-request does not error -- the job pends forever, in silence.
# Measured peak RSS on the serial 2026-09-09 run was 4.78 GB; the parent still holds the
# accumulated figure frames while workers hold a chunk each.
#$ -pe onenode 5
#$ -l m_mem_free=8G
#$ -o stage0/logs/_data_reports.out    # Log standard output
#$ -e stage0/logs/_data_reports.err    # Log standard error

cd stage0
python3 _build_error_files.py
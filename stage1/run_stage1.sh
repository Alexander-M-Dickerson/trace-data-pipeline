#!/bin/bash
#$ -S /bin/bash             # Ensure the job uses Bash shell
#$ -V                       # Export environment variables (important for Python envs)
#$ -cwd                     # Run from current working directory
# Cores and memory.
#
# ❗m_mem_free is charged PER SLOT, so this is 4 x 10G = 40 GB against the WRDS
# caps of 8 cores and 48 GB per job. Do not raise one without dividing the other:
# an over-request does not error, the job pends forever in silence.
#
# Slots are requested because stage 1 ALREADY uses them -- _stage1_settings.py
# resolves N_CORES from $NSLOTS and joblib runs that many workers. Without a -pe
# request it ran 4 workers inside a 1-slot allocation.
#
# 40 GB rather than the previous 24 GB because the panel is ~31M x 47 at step 6
# (16.35 GB measured on the 2026-09-09 run) and steps 5, 8 and 10 each build a
# list of chunk frames and then concat, peaking near 2x the panel.
#$ -pe onenode 4
#$ -l m_mem_free=10G
#$ -o stage1/logs/stage1.out       # Log standard output
#$ -e stage1/logs/stage1.err       # Log standard error

cd stage1
python3 _run_stage1.py

#!/bin/bash
#$ -S /bin/bash             # Ensure the job uses Bash shell
#$ -V                       # Export environment variables (important for Python envs)
#$ -cwd                     # Run from current working directory
#$ -o stage0/logs/01_enhanced.out    # Log standard output
#$ -e stage0/logs/01_enhanced.err    # Log standard error

cd stage0
python3 _run_enhanced_trace.py
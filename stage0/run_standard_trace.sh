#!/bin/bash
#$ -S /bin/bash             # Ensure the job uses Bash shell
#$ -V                       # Export environment variables (important for Python envs)
#$ -cwd                     # Run from current working directory
#$ -o logs/02_standard.out    # Log standard output
#$ -e logs/02_standard.err    # Log standard error

python3 _run_standard_trace.py
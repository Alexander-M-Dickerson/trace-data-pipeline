# -*- coding: utf-8 -*-
"""
_run_stage1.py
==============
Runner script for Stage 1 TRACE processing.
This script is called by run_stage1.sh for SGE job submission.

Author: Alex Dickerson
Created: 2025-11-17
"""

from create_daily_stage1 import run_stage1
from _stage1_settings import get_config, validate_config, print_config_summary
import logging
import sys
import gc

# Force garbage collection before starting
gc.collect()

# Get configuration
config = get_config()

# Print configuration summary
print_config_summary(config)

# Validate configuration
try:
    validate_config(config)
except (ValueError, FileNotFoundError) as e:
    print(f"\n{'=' * 80}")
    print("CONFIGURATION ERROR")
    print(f"{'=' * 80}")
    print(str(e))
    print(f"{'=' * 80}\n")
    sys.exit(1)

# Run Stage 1 pipeline
try:
    processor = run_stage1(config)
    print("\nStage 1 processing completed successfully!")
    sys.exit(0)
except Exception as e:
    print(f"\n{'=' * 80}")
    print("PIPELINE ERROR")
    print(f"{'=' * 80}")
    print(f"Error: {str(e)}")
    print(f"{'=' * 80}\n")
    logging.exception("Full traceback:")
    sys.exit(1)

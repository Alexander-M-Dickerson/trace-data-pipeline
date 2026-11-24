# -*- coding: utf-8 -*-
"""
Shared Configuration Across All Pipeline Stages
================================================
This file contains settings shared by stage0, stage1, and stage2.
Edit this file once, and all stages inherit the settings.

Author: Alex Dickerson
Created: 2025-11-17
"""

import os

# ============================================================================
# CREDENTIALS
# ============================================================================
# WRDS username - preferably set as environment variable
# Set in your shell: export WRDS_USERNAME="your_id"
#
# If not set as environment variable, fallback to hardcoded default:
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "your_wrds_username")

# ============================================================================
# AUTHOR INFORMATION. Please change it to your own.
# ============================================================================
AUTHOR = "Open Source Bond Asset Pricing"

# ============================================================================
# SHARED OUTPUT SETTINGS
# ============================================================================
# Stage 0 can write either format, but only "parquet" works end to end. Stage 1 and the
# report builder both call pd.read_parquet on a hard-coded "*.parquet" name. Setting
# "csv" here makes Stage 0 write .csv.gzip files that nothing downstream can read, so
# _trace_settings.py rejects it at import, before the run can die hours later.
OUTPUT_FORMAT = "parquet"  # Options: "parquet". "csv" is stage-0 only and not supported.

# ============================================================================
# TRACE DATABASE SELECTION
# ============================================================================
# Which TRACE datasets to process across all stages.
# Options: "enhanced", "standard", "144a"
#
# This list decides what run_pipeline.sh submits as well as what the later stages
# read. The script submits exactly these members. It used to submit all three, and
# this setting only decided what stage 1 picked up afterwards.
#
# Standard is not in the default. It is WRDS Standard TRACE, the delayed feed with
# less detail, and Enhanced plus 144A already cover the universe the pipeline is built
# around. Standard costs a whole extra job of several hours for coverage that is then
# clipped. Stage 1 keeps Standard rows only after the last Enhanced date
# (stage1_pipeline.py's overlap clip), so nearly all of it is discarded. Add it back
# when you want that trailing window.
#
#     TRACE_MEMBERS="enhanced standard 144a" ./run_pipeline.sh
#
# When Standard is requested it is scheduled after the other two, not beside them, so
# it can use the whole WRDS connection budget.
TRACE_MEMBERS = os.getenv("TRACE_MEMBERS", "enhanced 144a").split()

# ============================================================================
# STAGE-SPECIFIC OUTPUT SETTINGS
# ============================================================================
# Stage 0: Error plot generation (WARNING: Can take VERY long to run)
# Overridable from the environment so a smoke run can skip the slow plot pass:
#     STAGE0_OUTPUT_FIGURES=0 ./run_smoke_test.sh --with-reports
STAGE0_OUTPUT_FIGURES = os.getenv("STAGE0_OUTPUT_FIGURES", "1") not in ("0", "false", "False")

# Stage 1: Always generates reports and figures (no config needed)
# Stage 1 outputs are essential for data quality assessment and always produced

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
# Each stage imports this file as follows.
#
#   import sys
#   from pathlib import Path
#   sys.path.insert(0, str(Path(__file__).parent.parent))
#   from config import WRDS_USERNAME, AUTHOR, OUTPUT_FORMAT, TRACE_MEMBERS, STAGE0_OUTPUT_FIGURES
#
# The shared settings then have one source for every stage.

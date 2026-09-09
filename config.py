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
# AUTHOR INFORMATION -- Change accordingly please
# ============================================================================
AUTHOR = "Open Source Bond Asset Pricing"

# ============================================================================
# SHARED OUTPUT SETTINGS
# ============================================================================
OUTPUT_FORMAT = "parquet"  # Options: "parquet" (recommended), "csv"

# ============================================================================
# TRACE DATABASE SELECTION
# ============================================================================
# Which TRACE datasets to process across all stages.
# Options: "enhanced", "standard", "144a"
#
# This now drives SUBMISSION as well as what the later stages read: run_pipeline.sh
# submits exactly these members. It used to submit all three regardless, and this knob
# only decided what stage 1 picked up afterwards.
#
# STANDARD IS NO LONGER A DEFAULT. It is the WRDS Standard tape -- the delayed,
# lower-detail feed -- and Enhanced plus 144A already cover the universe the pipeline
# is built around. Standard costs a whole extra multi-hour job for coverage that is
# then clipped anyway: stage1 keeps Standard rows only AFTER the last Enhanced date
# (stage1_pipeline.py's overlap clip), so nearly all of it is discarded. Add it back
# when you specifically want that trailing window:
#
#     TRACE_MEMBERS="enhanced standard 144a" ./run_pipeline.sh
#
# When Standard IS requested it is scheduled after the other two rather than beside
# them, so it can use the whole WRDS connection budget instead of a slice of it.
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
# Each stage imports this file:
#
#   import sys
#   from pathlib import Path
#   sys.path.insert(0, str(Path(__file__).parent.parent))
#   from config import WRDS_USERNAME, AUTHOR, OUTPUT_FORMAT, TRACE_MEMBERS, STAGE0_OUTPUT_FIGURES
#
# This ensures single source of truth for shared settings across all stages.

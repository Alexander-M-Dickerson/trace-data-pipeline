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
# Which TRACE datasets to process across all stages
# Options: "enhanced", "standard", "144a"
# Overridable from the environment (space-separated) so a test run can select members
# without editing this file:  TRACE_MEMBERS="enhanced 144a" ./run_smoke_test.sh
TRACE_MEMBERS = os.getenv("TRACE_MEMBERS", "enhanced standard 144a").split()

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

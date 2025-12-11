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
TRACE_MEMBERS = ["enhanced", "standard", "144a"]

# ============================================================================
# STAGE-SPECIFIC OUTPUT SETTINGS
# ============================================================================
# Stage 0: Error plot generation (WARNING: Can take VERY long to run)
STAGE0_OUTPUT_FIGURES = True  # Set to False to skip error plots for faster processing

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

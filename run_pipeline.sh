#!/bin/bash
# TRACE Data Pipeline Orchestrator
#
# This script orchestrates the entire multi-stage TRACE data pipeline:
#   - Pre-Stage: Download required data files (Liu-Wu yields, OSBAP linker, FF industries)
#   - Stage 0: Data extraction (Enhanced, Standard, 144A TRACE) + Report building
#   - Stage 1: Daily aggregation and analytics
#   - Stage 2: (Future) Additional processing stages
#
# IMPORTANT: This script MUST be executed from the project ROOT directory.
#            All paths are relative to ROOT and jobs are submitted with
#            working directories set appropriately for each stage.
#
# NOTE: Pre-stage downloads happen on the login node (internet access required).
#       This is necessary because WRDS compute nodes have no internet access.

set -euo pipefail

# Verify we're in the project root
if [[ ! -d "stage0" ]] || [[ ! -d "stage1" ]]; then
    echo "ERROR: This script must be run from the project ROOT directory."
    echo "Current directory: $(pwd)"
    echo "Expected structure: stage0/, stage1/ subdirectories"
    exit 1
fi

# Check available disk space (use quota on WRDS, fallback to df elsewhere)
echo ""
echo "=== DISK SPACE CHECK ==="

# Try WRDS quota command first (more accurate for WRDS users)
if command -v quota &> /dev/null; then
    # Parse quota output for Home directory
    # Expected format: "Home:  7.88GB / 10GB"
    QUOTA_LINE=$(quota 2>/dev/null | grep -i "Home:" | head -1)

    if [[ -n "$QUOTA_LINE" ]]; then
        # Extract used and limit from "Home:  7.88GB / 10GB"
        USED=$(echo "$QUOTA_LINE" | awk '{print $2}' | sed 's/GB//g')
        LIMIT=$(echo "$QUOTA_LINE" | awk '{print $4}' | sed 's/GB//g')

        if [[ -n "$USED" ]] && [[ -n "$LIMIT" ]]; then
            # Calculate available space
            AVAIL_GB=$(awk "BEGIN {printf \"%.2f\", $LIMIT - $USED}")
            echo "[info] WRDS Quota - Home directory: ${USED} GB used / ${LIMIT} GB limit"
            echo "[info] Available space: ${AVAIL_GB} GB"
        else
            # Quota parsing failed, fall back to df
            echo "[warn] Could not parse quota output, using df instead"
            AVAIL_KB=$(df -k . | awk 'NR==2 {print $4}')
            AVAIL_GB=$(awk "BEGIN {printf \"%.2f\", $AVAIL_KB/1024/1024}")
            echo "[info] Available disk space (filesystem): ${AVAIL_GB} GB"
        fi
    else
        # quota command exists but no Home line found, fall back to df
        echo "[info] No quota detected, checking filesystem space"
        AVAIL_KB=$(df -k . | awk 'NR==2 {print $4}')
        AVAIL_GB=$(awk "BEGIN {printf \"%.2f\", $AVAIL_KB/1024/1024}")
        echo "[info] Available disk space: ${AVAIL_GB} GB"
    fi
else
    # quota command not available (non-WRDS system), use df
    echo "[info] Checking filesystem space (quota not available)"
    AVAIL_KB=$(df -k . | awk 'NR==2 {print $4}')
    AVAIL_GB=$(awk "BEGIN {printf \"%.2f\", $AVAIL_KB/1024/1024}")
    echo "[info] Available disk space: ${AVAIL_GB} GB"
fi

# Check if less than 4 GB available
if (( $(awk "BEGIN {print ($AVAIL_GB < 4.0)}") )); then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                         !   WARNING  !                         ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo "║  INSUFFICIENT DISK SPACE DETECTED                              ║"
    echo "║                                                                ║"
    echo "║  Available: ${AVAIL_GB} GB                                     ║"
    echo "║  Required:  At least 4.0 GB recommended                        ║"
    echo "║                                                                ║"
    echo "║  The pipeline generates large intermediate files and may fail  ║"
    echo "║  or corrupt data if disk space runs out during processing.     ║"
    echo "║                                                                ║"
    echo "║  RECOMMENDATION: Stop execution and free up disk space         ║"
    echo "║                                                                ║"
    echo "║  To continue anyway: Re-run with FORCE_RUN=1                   ║"
    echo "║  Example: FORCE_RUN=1 ./run_pipeline.sh                        ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # Allow override with FORCE_RUN environment variable
    if [[ "${FORCE_RUN:-0}" != "1" ]]; then
        echo "[error] Exiting due to insufficient disk space."
        echo "[info] Free up space or set FORCE_RUN=1 to override this check."
        exit 1
    else
        echo "[warn] FORCE_RUN=1 detected - continuing despite low disk space"
        echo "[warn] Proceed at your own risk!"
    fi
else
    echo "[ok] Sufficient disk space available (${AVAIL_GB} GB >= 4.0 GB)"
fi
echo ""

# Create log directories for all stages
echo "[setup] Creating log directories..."
mkdir -p stage0/logs
mkdir -p stage1/logs
mkdir -p stage1/data

# Download required data files for Stage 1 (WRDS compute nodes have no internet)
# This must be done on the login node before submitting jobs
echo ""
echo "=== PRE-STAGE: Downloading Required Data Files ==="
echo "[download] Liu-Wu treasury yields..."
wget -q -O stage1/data/liu_wu_yields.xlsx \
    "https://docs.google.com/spreadsheets/d/11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t/export?format=xlsx&id=11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t" \
    && echo "[ok] Liu-Wu yields downloaded" \
    || echo "[warn] Failed to download Liu-Wu yields (may already exist)"

echo "[download] OSBAP Linker file..."
wget -q -O stage1/data/linker_file_2025.zip \
    "https://openbondassetpricing.com/wp-content/uploads/2025/11/linker_file_2025.zip" \
    && echo "[ok] OSBAP Linker downloaded" \
    || echo "[warn] Failed to download OSBAP Linker (may already exist)"

if [[ -f "stage1/data/linker_file_2025.zip" ]]; then
    echo "[extract] Unzipping OSBAP Linker..."
    unzip -q -o stage1/data/linker_file_2025.zip -d stage1/data/ \
        && echo "[ok] OSBAP Linker extracted" \
        || echo "[warn] Failed to extract OSBAP Linker"
    rm -f stage1/data/linker_file_2025.zip
fi

echo "[download] Fama-French 17 Industry Classification..."
wget -q -O stage1/data/Siccodes17.zip \
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes17.zip" \
    && echo "[ok] FF17 downloaded" \
    || echo "[warn] Failed to download FF17 (may already exist)"

if [[ -f "stage1/data/Siccodes17.zip" ]]; then
    echo "[extract] Unzipping FF17..."
    unzip -q -o stage1/data/Siccodes17.zip -d stage1/data/ \
        && echo "[ok] FF17 extracted" \
        || echo "[warn] Failed to extract FF17"
    rm -f stage1/data/Siccodes17.zip
fi

echo "[download] Fama-French 30 Industry Classification..."
wget -q -O stage1/data/Siccodes30.zip \
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes30.zip" \
    && echo "[ok] FF30 downloaded" \
    || echo "[warn] Failed to download FF30 (may already exist)"

if [[ -f "stage1/data/Siccodes30.zip" ]]; then
    echo "[extract] Unzipping FF30..."
    unzip -q -o stage1/data/Siccodes30.zip -d stage1/data/ \
        && echo "[ok] FF30 extracted" \
        || echo "[warn] Failed to extract FF30"
    rm -f stage1/data/Siccodes30.zip
fi

echo "[verify] Checking downloaded files..."
MISSING_FILES=0
for file in "liu_wu_yields.xlsx" "OSBAP_Linker_October_2025.parquet" "Siccodes17.txt" "Siccodes30.txt"; do
    if [[ -f "stage1/data/$file" ]]; then
        echo "[ok] $file"
    else
        echo "[error] Missing: $file"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [[ $MISSING_FILES -gt 0 ]]; then
    echo "[warn] Some files are missing. Stage 1 may fail."
    echo "[warn] You can manually download them following stage1/QUICKSTART_stage1.md"
else
    echo "[ok] All required data files present"
fi

# Stage 0: Submit Enhanced, Standard, and 144A TRACE data extraction jobs
# These run in parallel and use the stage0 directory as working directory
echo ""
echo "=== STAGE 0: TRACE Data Extraction ==="
echo "[submit] Enhanced TRACE ..."
J1=$(qsub -terse -N trace_enhanced stage0/run_enhanced_trace.sh)

echo "[submit] Standard TRACE ..."
J2=$(qsub -terse -N trace_standard stage0/run_standard_trace.sh)

echo "[submit] 144A TRACE ..."
J3=$(qsub -terse -N trace_144a stage0/run_144a_trace.sh)

# Stage 0: Build data reports after all extraction jobs complete
echo "[submit] Build data reports (waits for all TRACE jobs) ..."
J4=$(qsub -terse -N build_reports -hold_jid ${J1},${J2},${J3} stage0/run_build_data_reports.sh)

# Stage 1: Process daily aggregation after stage0 reports are ready
echo ""
echo "=== STAGE 1: Daily Aggregation & Analytics ==="
echo "[submit] Stage 1 pipeline (waits for stage0 reports) ..."
J5=$(qsub -terse -N stage1_pipeline -hold_jid ${J4} stage1/run_stage1.sh)

# Stage 2: (Future placeholder)
# echo ""
# echo "=== STAGE 2: Advanced Analytics ==="
# echo "[submit] Stage 2 pipeline (waits for stage1) ..."
# mkdir -p stage2/logs
# J6=$(qsub -terse -wd "$PWD/stage2" -N stage2_pipeline -hold_jid ${J5} stage2/run_stage2.sh)

# Summary
echo ""
echo "=== SUBMISSION COMPLETE ==="
echo "[ok] Pre-stage data downloads completed"
echo "[ok] All jobs submitted with dependencies:"
echo ""
echo "  Stage 0 - Data Extraction (parallel):"
echo "    Enhanced TRACE: ${J1}"
echo "    Standard TRACE: ${J2}"
echo "    144A TRACE:     ${J3}"
echo ""
echo "  Stage 0 - Reports (waits for data):"
echo "    Build Reports:  ${J4}"
echo ""
echo "  Stage 1 - Analytics (waits for reports):"
echo "    Daily Pipeline: ${J5}"
echo ""
# echo "  Stage 2 - Advanced (waits for stage1):"
# echo "    Stage2 Pipeline: ${J6}"
# echo ""
echo "Monitor jobs with: qstat"
echo "Check logs in: stage0/logs/, stage1/logs/"
echo "Downloaded data in: stage1/data/"
echo ""

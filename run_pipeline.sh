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

# Stage 1's external inputs. Must happen on the LOGIN NODE -- compute nodes have no
# internet. Kept in its own script so the smoke test can point at the same command.
echo ""
# Invoked through bash rather than as ./download_inputs.sh so this does not depend
# on the executable bit surviving the clone -- it has not, historically, because
# these scripts are authored on Windows where core.filemode is false.
bash ./download_inputs.sh || { echo "[error] input download failed"; exit 1; }


# Stage 0: submit exactly the members named in TRACE_MEMBERS.
#
# This block used to hard-code all three qsub calls and then hold the report job on
# "-hold_jid ${J1},${J2},${J3}". Dropping a member left its variable unset and the
# flag became "-hold_jid 123,,125", so the ids are built up here from the jobs
# ACTUALLY submitted.
echo ""
echo "=== STAGE 0: TRACE Data Extraction ==="

# python3 on WRDS; overridable so the submission graph can be dry-run elsewhere.
PY="${PYTHON:-python3}"

MEMBERS=$("${PY}" -c "import sys; sys.path.insert(0,'.'); from config import TRACE_MEMBERS; print(' '.join(TRACE_MEMBERS))") || {
    echo "[error] could not read TRACE_MEMBERS from config.py"; exit 1; }
echo "[info] members: ${MEMBERS}"

# Fail HERE if the members ask for more WRDS connections than the account can hold.
# Four hours in, the same problem arrives as "EOFError: EOF when reading a line".
"${PY}" -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'stage0')
from _trace_settings import validate_connection_budget
validate_connection_budget('''${MEMBERS}'''.split())
" || { echo "[error] WRDS connection budget exceeded -- see stage0/_trace_settings.py"; exit 1; }

declare -A RUNNER=( [enhanced]=stage0/run_enhanced_trace.sh \
                    [standard]=stage0/run_standard_trace.sh \
                    [144a]=stage0/run_144a_trace.sh )
declare -A JOBNAME=( [enhanced]=trace_enhanced [standard]=trace_standard [144a]=trace_144a )

# Cores and memory, DERIVED from each member's worker count rather than written into
# the job scripts, so the request cannot drift away from CONCURRENCY. The scripts' own
# resource directives stay untouched. qsub_resources refuses to emit anything above
# the WRDS caps, because an unsatisfiable request does not fail -- it pends forever,
# in silence -- and m_mem_free is charged PER SLOT, so asking for 96 GB by accident
# is a two-character mistake.
res_for() {
    "${PY}" -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'stage0')
from _trace_settings import qsub_resources
print(qsub_resources('$1'))
"
}

STAGE0_IDS=()          # every stage0 job, for the report job's hold list
CONCURRENT_IDS=()      # enhanced + 144a, which Standard waits on

# Enhanced and 144A first: they run side by side and their connection budgets are
# sized to co-exist.
for member in enhanced 144a; do
    [[ " ${MEMBERS} " == *" ${member} "* ]] || continue
    res=$(res_for "${member}") || { echo "[error] bad resource request for ${member}"; exit 1; }
    echo "[submit] ${member} TRACE  (${res}) ..."
    jid=$(qsub -terse -N "${JOBNAME[$member]}" ${res} "${RUNNER[$member]}")
    STAGE0_IDS+=("${jid}"); CONCURRENT_IDS+=("${jid}")
    echo "         job ${jid}"
done

# Standard, if asked for, runs AFTER them -- so it gets the whole connection budget
# rather than a slice, and the peak memory of two big jobs never overlaps.
if [[ " ${MEMBERS} " == *" standard "* ]]; then
    res=$(res_for standard) || { echo "[error] bad resource request for standard"; exit 1; }
    echo "[submit] standard TRACE  (${res}, held until enhanced/144a finish) ..."
    if [[ ${#CONCURRENT_IDS[@]} -gt 0 ]]; then
        hold=$(IFS=,; echo "${CONCURRENT_IDS[*]}")
        jid=$(qsub -terse -N "${JOBNAME[standard]}" ${res} -hold_jid "${hold}" "${RUNNER[standard]}")
    else
        jid=$(qsub -terse -N "${JOBNAME[standard]}" ${res} "${RUNNER[standard]}")
    fi
    STAGE0_IDS+=("${jid}")
    echo "         job ${jid}"
fi

if [[ ${#STAGE0_IDS[@]} -eq 0 ]]; then
    echo "[error] TRACE_MEMBERS selected no known member: '${MEMBERS}'"
    exit 1
fi

# Stage 0: build the data reports once every extraction job is done.
HOLD_STAGE0=$(IFS=,; echo "${STAGE0_IDS[*]}")
echo "[submit] Build data reports (waits for ${HOLD_STAGE0}) ..."
J4=$(qsub -terse -N build_reports -hold_jid "${HOLD_STAGE0}" stage0/run_build_data_reports.sh)

# Stage 1: daily aggregation, as soon as the stage-0 DATA is ready.
#
# It holds on the stage-0 jobs, NOT on the report job, so the reports and stage 1 run
# side by side. Stage 1 does not read anything the reports produce -- it reads exactly
# two paths, stage0/<member>/trace_<member>_<stamp>.parquet and
# stage0/enhanced/trace_enhanced_fisd_<stamp>.parquet (stage1_pipeline.py:267 and :424).
# The data_reports directory stage 1 refers to is its OWN. Chaining it behind the
# reports put ~50 minutes of report generation on the critical path for nothing.
#
# The two overlap safely: the stage-0 jobs have released their WRDS connections by then,
# so the reports (up to 5) plus stage 1 (1) sit at 6 against the measured ceiling of 7 --
# the same one spare the stage-0 phase leaves. They write to separate trees and both read
# stage 0's parquet read-only.
echo ""
echo "=== STAGE 1: Daily Aggregation & Analytics ==="
echo "[submit] Stage 1 pipeline (waits for stage0 data; runs alongside the reports) ..."
J5=$(qsub -terse -N stage1_pipeline -hold_jid "${HOLD_STAGE0}" stage1/run_stage1.sh)

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
echo "  Stage 0 - Data Extraction (${MEMBERS}):"
echo "    job ids: ${STAGE0_IDS[*]}"
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

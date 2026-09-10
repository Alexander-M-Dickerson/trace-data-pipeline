#!/bin/bash
# =============================================================================
# run_stage2.sh -- build the monthly asset-pricing panel
# =============================================================================
#
# Stage 2 runs on YOUR OWN MACHINE, not on the WRDS cluster. It is NOT an SGE job
# and must NOT be submitted with qsub: it needs more memory than a grid slot
# allows and no TRACE database access at all.
#
# Before running, download the stage0/ and stage1/ folders from WRDS so the tree
# looks like:
#
#     trace-data-pipeline/
#     |-- stage0/enhanced/trace_enhanced_fisd_YYYYMMDD.parquet
#     |-- stage1/data/stage1_YYYYMMDD.parquet
#     |               call_dummy_YYYYMMDD.parquet
#     `-- stage2/
#
# Usage:
#     ./run_stage2.sh                 # full build
#     ./run_stage2.sh --dry-run       # check the configuration, build nothing
#     ./run_stage2.sh --limit-cusips 200
#
# Any arguments are passed straight through to _run_stage2.py.
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs

PYTHON="${PYTHON:-python3}"

echo "==============================================================================="
echo "STAGE 2 - Monthly asset-pricing panel"
echo "==============================================================================="
echo "  python : $($PYTHON --version 2>&1)"
echo "  started: $(date)"
echo

"$PYTHON" -u _run_stage2.py "$@" 2>&1 | tee "logs/stage2_$(date +%Y%m%d_%H%M%S).log"

status=${PIPESTATUS[0]}
echo
echo "  finished: $(date)  (exit ${status})"
exit "${status}"

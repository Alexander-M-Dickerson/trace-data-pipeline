#!/bin/bash
# Submit Enhanced, Standard, and 144A TRACE jobs to SGE, then the report job once all finish.

set -euo pipefail
mkdir -p logs

echo "[submit] Enhanced TRACE ..."
J1=$(qsub -terse -N trace_enhanced run_enhanced_trace.sh)

echo "[submit] Standard TRACE ..."
J2=$(qsub -terse -N trace_standard run_standard_trace.sh)

echo "[submit] 144A TRACE ..."
J3=$(qsub -terse -N trace_144a run_144a_trace.sh)

echo "[submit] Build data reports (after all TRACE jobs finish) ..."
qsub -terse -N build_reports -hold_jid ${J1},${J2},${J3} run_build_data_reports.sh

echo "[ok] All jobs submitted with dependency. Reports will run after ${J1},${J2},${J3} complete."


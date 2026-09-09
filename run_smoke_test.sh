#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -pe onenode 1
#$ -l m_mem_free=16G
#$ -o smoke/logs/smoke.out
#$ -e smoke/logs/smoke.err
#
# run_smoke_test.sh -- whole-chain validation on a handful of CUSIP chunks.
#
# Runs stage0 -> (reports) -> stage1 end to end in minutes rather than hours, then
# asserts the cross-stage invariants. Use it before committing, and before spending
# four hours on a real run.
#
#   ./run_smoke_test.sh                      # default: 5 chunks per member
#   ./run_smoke_test.sh --chunks 2           # faster
#   ./run_smoke_test.sh --members "enhanced 144a"
#   ./run_smoke_test.sh --with-reports       # also exercise the data-report job
#
# On the WRDS Cloud this must be SUBMITTED, not run on the login node -- CPU- and
# memory-intensive work is not permitted on the head nodes:
#
#   qsub run_smoke_test.sh
#
# It runs the REAL stage0/stage1 code straight out of this repo. Nothing is copied,
# so the run can never be validating a stale duplicate. Outputs are redirected purely
# by the working directory: both settings modules derive their root from the cwd, so
# running from smoke/stage0 and smoke/stage1 puts every artifact under smoke/ and
# leaves production outputs untouched.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${REPO}/smoke"
CHUNKS=5
MEMBERS=""
WITH_REPORTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunks)       CHUNKS="$2"; shift 2 ;;
        --root)         ROOT="$2";   shift 2 ;;
        --members)      MEMBERS="$2"; shift 2 ;;
        --with-reports) WITH_REPORTS=1; shift ;;
        -h|--help)      sed -n '10,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[error] unknown option: $1"; exit 2 ;;
    esac
done

PY="${PYTHON:-python3}"
export STAGE0_LIMIT_CHUNKS="${CHUNKS}"
[[ -n "${MEMBERS}" ]] && export TRACE_MEMBERS="${MEMBERS}"

# Figures are the slowest part of stage0 and prove nothing about the cross-stage seams.
export STAGE0_OUTPUT_FIGURES=0

MEMBER_LIST=$("${PY}" -c "import sys; sys.path.insert(0, '${REPO}'); from config import TRACE_MEMBERS; print(' '.join(TRACE_MEMBERS))") || {
    echo "[error] could not read TRACE_MEMBERS from config.py"; exit 1; }

echo "================================================================"
echo " SMOKE TEST"
echo "================================================================"
echo " repo          : ${REPO}"
echo " scratch root  : ${ROOT}"
echo " members       : ${MEMBER_LIST}"
echo " chunks/member : ${CHUNKS}"
echo " data reports  : $([[ ${WITH_REPORTS} -eq 1 ]] && echo yes || echo 'skipped (--with-reports to include)')"
echo "================================================================"

# ---------------------------------------------------------------- scratch root
rm -rf "${ROOT}"
mkdir -p "${ROOT}/stage0/logs" "${ROOT}/stage1/data" "${ROOT}/stage1/logs"

# Stage 1 needs the files run_pipeline.sh fetches on the login node. Reuse the
# repo's copies if they are already there; otherwise say so plainly rather than
# failing thirty minutes in.
MISSING=0
for f in liu_wu_yields.xlsx Siccodes12.txt Siccodes17.txt Siccodes30.txt; do
    if [[ -f "${REPO}/stage1/data/${f}" ]]; then
        cp "${REPO}/stage1/data/${f}" "${ROOT}/stage1/data/${f}"
    else
        echo "[error] missing external input: stage1/data/${f}"; MISSING=1
    fi
done
LINKER_DIR="${REPO}/stage1/data/bond_firm_linker_2026"
if [[ -d "${LINKER_DIR}" ]]; then
    cp -r "${LINKER_DIR}" "${ROOT}/stage1/data/"
else
    echo "[error] missing external input: stage1/data/bond_firm_linker_2026/"; MISSING=1
fi
if [[ ${MISSING} -ne 0 ]]; then
    echo
    echo "[error] Run the PRE-STAGE download block of run_pipeline.sh first (login node)."
    exit 1
fi

# ------------------------------------------------------------------- stage 0
for member in ${MEMBER_LIST}; do
    case "${member}" in
        enhanced) runner="_run_enhanced_trace.py" ;;
        standard) runner="_run_standard_trace.py" ;;
        144a)     runner="_run_144a_trace.py" ;;
        *) echo "[error] unknown TRACE member: ${member}"; exit 2 ;;
    esac
    echo
    echo "---------------- stage0: ${member} (${CHUNKS} chunks) ----------------"
    ( cd "${ROOT}/stage0" && "${PY}" -u "${REPO}/stage0/${runner}" ) \
        > "${ROOT}/stage0/logs/${member}.log" 2>&1
    rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[FAIL] stage0/${member} exited ${rc}. Last 25 lines:"
        tail -25 "${ROOT}/stage0/logs/${member}.log"
        exit ${rc}
    fi
    echo "[ok] stage0/${member} complete"
done

# ------------------------------------------------------------------- reports
if [[ ${WITH_REPORTS} -eq 1 ]]; then
    echo
    echo "---------------- stage0: data reports ----------------"
    ( cd "${ROOT}/stage0" && "${PY}" -u "${REPO}/stage0/_build_error_files.py" ) \
        > "${ROOT}/stage0/logs/reports.log" 2>&1
    rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[FAIL] data reports exited ${rc}. Last 25 lines:"
        tail -25 "${ROOT}/stage0/logs/reports.log"
        exit ${rc}
    fi
    echo "[ok] data reports complete"
fi

# ------------------------------------------------------------------- stage 1
echo
echo "---------------- stage1 ----------------"
( cd "${ROOT}/stage1" && "${PY}" -u "${REPO}/stage1/_run_stage1.py" ) \
    > "${ROOT}/stage1/logs/stage1.log" 2>&1
rc=$?
if [[ ${rc} -ne 0 ]]; then
    echo "[FAIL] stage1 exited ${rc}. Last 25 lines:"
    tail -25 "${ROOT}/stage1/logs/stage1.log"
    exit ${rc}
fi
echo "[ok] stage1 complete"

# ---------------------------------------------------------------- assertions
echo
"${PY}" "${REPO}/tests/smoke_assertions.py" --root "${ROOT}" --members ${MEMBER_LIST}
rc=$?

echo
if [[ ${rc} -eq 0 ]]; then
    echo "SMOKE TEST PASSED. Logs under ${ROOT}"
else
    echo "SMOKE TEST FAILED. Logs under ${ROOT}"
fi
exit ${rc}

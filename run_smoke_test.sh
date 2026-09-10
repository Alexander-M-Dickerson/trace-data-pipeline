#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -pe onenode 1
#$ -l m_mem_free=16G
# Written to the REPO ROOT, not smoke/logs/, on purpose: SGE opens these before the
# script runs, and smoke/logs/ does not exist in a fresh clone -- the job would go
# straight to Eqw without executing a line. The root always exists.
#$ -o smoke_test.out
#$ -e smoke_test.err
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
#   ./run_smoke_test.sh --with-figures       # ...INCLUDING its re-clean and figures
#
# On the WRDS Cloud this must be SUBMITTED, not run on the login node -- CPU- and
# memory-intensive work is not permitted on the head nodes:
#
#   ./download_inputs.sh          # once, on the login node -- needs internet
#   qsub run_smoke_test.sh        # from the REPO ROOT
#
# It runs the REAL stage0/stage1 code straight out of this repo. Nothing is copied,
# so the run can never be validating a stale duplicate. Outputs are redirected purely
# by the working directory: both settings modules derive their root from the cwd, so
# running from smoke/stage0 and smoke/stage1 puts every artifact under smoke/ and
# leaves production outputs untouched.

set -uo pipefail

# Find the repo. This is not as simple as dirname "$0", because under qsub SGE COPIES
# the job script into its spool directory and runs the copy -- so BASH_SOURCE points at
# something like /gridware/sge/default/spool/<node>/job_scripts, and every path built
# from it lands nowhere. That made the submitted run, which is the documented way to
# run this on WRDS, report the repo's own inputs as missing.
#
# So: try the candidates in order and take the first that actually LOOKS like the repo,
# rather than trusting any single one of them.
#   SGE_O_WORKDIR  the directory qsub was invoked from -- correct under a job
#   PWD            "#$ -cwd" starts the job there too, and it is right for a local run
#   BASH_SOURCE    correct locally; the spool copy under qsub
_looks_like_repo() { [[ -d "$1/stage0" && -d "$1/stage1" ]]; }

REPO=""
for _cand in "${SGE_O_WORKDIR:-}" "${PWD}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
    [[ -n "${_cand}" ]] || continue
    if _looks_like_repo "${_cand}"; then REPO="${_cand}"; break; fi
done
if [[ -z "${REPO}" ]]; then
    echo "[error] Cannot locate the repo (no candidate has both stage0/ and stage1/)."
    echo "        Tried: SGE_O_WORKDIR='${SGE_O_WORKDIR:-}' PWD='${PWD}'"
    echo "        Submit from the repo root:  cd ~/trace-data-pipeline && qsub run_smoke_test.sh"
    exit 1
fi
ROOT="${REPO}/smoke"
CHUNKS=5
MEMBERS=""
WITH_REPORTS=0
WITH_FIGURES=0
# Production packs chunks to ~750k trade rows. That is the right size for a real run
# and far too big for a smoke run, where the point is to exercise every code path
# quickly. Pack small instead: the packing logic is still exercised, the volume is not.
TARGET_ROWS=40000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunks)       CHUNKS="$2"; shift 2 ;;
        --target-rows)  TARGET_ROWS="$2"; shift 2 ;;
        --root)         ROOT="$2";   shift 2 ;;
        --members)      MEMBERS="$2"; shift 2 ;;
        --with-reports) WITH_REPORTS=1; shift ;;
        --with-figures) WITH_REPORTS=1; WITH_FIGURES=1; shift ;;
        -h|--help)      sed -n '13,34p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[error] unknown option: $1"; exit 2 ;;
    esac
done

PY="${PYTHON:-python3}"

# Check the credentials BEFORE spending anything on them. Without WRDS_USERNAME,
# config.py falls back to the literal "your_wrds_username", the wrds package prompts on
# stdin, and the run dies minutes in with "EOFError: EOF when reading a line" -- which
# reads exactly like the connection limit and sends you hunting the wrong thing
# entirely. Two seconds here saves that.
if [[ -z "${WRDS_USERNAME:-}" || "${WRDS_USERNAME}" == "your_wrds_username" ]]; then
    echo "[error] WRDS_USERNAME is not set (got: ${WRDS_USERNAME:-<unset>})"
    echo "        export WRDS_USERNAME=\"your_id\"    # then re-run"
    echo "        You also need a ~/.pgpass entry for wrds-pgdata.wharton.upenn.edu:9737."
    exit 1
fi

export STAGE0_LIMIT_CHUNKS="${CHUNKS}"
export STAGE0_TARGET_ROWS="${TARGET_ROWS}"
[[ -n "${MEMBERS}" ]] && export TRACE_MEMBERS="${MEMBERS}"

# Figures are the slowest part of stage0 and prove nothing about the cross-stage seams,
# so they are off by default.
#
# ❗But the report job's error_checks -- the re-clean that this harness is otherwise
# blind to -- sits INSIDE `if STAGE0_OUTPUT_FIGURES:`. With figures off, --with-reports
# goes green having executed none of it. --with-figures turns it on AND forces the
# flagged universe into several small chunks, so the concurrent path actually runs.
if [[ ${WITH_FIGURES} -eq 1 ]]; then
    export STAGE0_OUTPUT_FIGURES=1
    export STAGE0_REPORT_CHUNK_SIZE="${STAGE0_REPORT_CHUNK_SIZE:-3}"
    export STAGE0_REPORT_WORKERS="${STAGE0_REPORT_WORKERS:-3}"
else
    export STAGE0_OUTPUT_FIGURES=0
fi

MEMBER_LIST=$("${PY}" -c "import sys; sys.path.insert(0, '${REPO}'); from config import TRACE_MEMBERS; print(' '.join(TRACE_MEMBERS))") || {
    echo "[error] could not read TRACE_MEMBERS from config.py"; exit 1; }

echo "================================================================"
echo " SMOKE TEST"
echo "================================================================"
echo " repo          : ${REPO}"
echo " scratch root  : ${ROOT}"
echo " members       : ${MEMBER_LIST}"
echo " chunks/member : ${CHUNKS}"
echo " rows/chunk    : ${TARGET_ROWS} (production packs to 750000)"
echo " data reports  : $([[ ${WITH_REPORTS} -eq 1 ]] && echo yes || echo 'skipped (--with-reports to include)')"
echo " report figures: $([[ ${WITH_FIGURES} -eq 1 ]] && echo "yes (chunk_size=${STAGE0_REPORT_CHUNK_SIZE}, workers=${STAGE0_REPORT_WORKERS})" || echo 'skipped (--with-figures exercises error_checks)')"
echo "================================================================"

# ---------------------------------------------------------------- scratch root
# Keep the CUSIP row counts across runs. The aggregate that produces them scans the
# whole source table -- 93 s for Enhanced -- which is nothing against a real run but
# dominates a smoke run, and re-measuring it on every iteration makes this tool
# annoying enough that people stop using it.
CACHE_KEEP="$(mktemp -d)"
if compgen -G "${ROOT}/stage0/*/cusip_row_counts_*.parquet" > /dev/null 2>&1; then
    for f in "${ROOT}"/stage0/*/cusip_row_counts_*.parquet; do
        member="$(basename "$(dirname "${f}")")"
        mkdir -p "${CACHE_KEEP}/${member}"
        cp "${f}" "${CACHE_KEEP}/${member}/"
    done
    echo "[info] preserving cached CUSIP row counts across the rebuild"
fi

rm -rf "${ROOT}"
mkdir -p "${ROOT}/stage0/logs" "${ROOT}/stage1/data" "${ROOT}/stage1/logs"

if compgen -G "${CACHE_KEEP}/*/cusip_row_counts_*.parquet" > /dev/null 2>&1; then
    for f in "${CACHE_KEEP}"/*/cusip_row_counts_*.parquet; do
        member="$(basename "$(dirname "${f}")")"
        mkdir -p "${ROOT}/stage0/${member}"
        cp "${f}" "${ROOT}/stage0/${member}/"
    done
fi
rm -rf "${CACHE_KEEP}"

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
    echo "[error] Stage 1's external inputs are not in the repo yet."
    echo "        Fetch them ON THE LOGIN NODE -- compute nodes have no internet -- then"
    echo "        submit this again:"
    echo
    echo "            cd ${REPO} && ./download_inputs.sh"
    echo "            qsub run_smoke_test.sh"
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
    echo "SMOKE TEST PASSED. Per-stage logs under ${ROOT}; job output in smoke_test.out"
else
    echo "SMOKE TEST FAILED. Per-stage logs under ${ROOT}; job output in smoke_test.out"
fi
exit ${rc}

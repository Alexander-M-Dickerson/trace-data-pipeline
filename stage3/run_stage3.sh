#!/usr/bin/env bash
# Run every Stage-3 step in order: the input contract first, then produce, then render.
#
#   bash run_stage3.sh                 # everything
#   bash run_stage3.sh --section lib   # one section (any _run_stage3.py flag passes through)
#
# PY overrides the interpreter. PYBONDLAB_DIR points at the build to use -- the
# uncertainty grids need one carrying the fast kernels; see README_stage3.md.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

echo "== checking the input contract =="
"$PY" tools/check_inputs.py

echo
echo "== running Stage 3 =="
"$PY" _run_stage3.py "$@"

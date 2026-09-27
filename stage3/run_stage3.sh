#!/usr/bin/env bash
# [tag:entry.stage3] stage 3, the exhibits: bash run_stage3.sh
# Run every Stage-3 step in order: the input contract first, then produce, then render.
#
#   bash run_stage3.sh                 # everything
#   bash run_stage3.sh --section lib   # one section (any _run_stage3.py flag passes through)
#
# PY overrides the interpreter: the Python where requirements-local.txt and PyBondLab are
# installed.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

# The input contract, for the inputs THIS run reads: one section's with --section, all five
# otherwise. --help, --list and --dry-run build nothing, so they go straight to the runner,
# which still says what is missing. No arrays: bash 3.2 (macOS) and `set -u` disagree on them.
check=1
section=""
prev=""
for a in "$@"; do
  case "$a" in
    -h|--help|--list|--dry-run) check=0 ;;
    --section=*) section="${a#--section=}" ;;
  esac
  if [[ "$prev" == "--section" ]]; then section="$a"; fi
  prev="$a"
done
if [[ $check -eq 1 ]]; then
  echo "== checking the input contract =="
  if [[ -n "$section" ]]; then
    "$PY" tools/check_inputs.py --section "$section"
  else
    "$PY" tools/check_inputs.py
  fi
fi

echo
echo "== running Stage 3 =="
"$PY" _run_stage3.py "$@"

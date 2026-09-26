#!/usr/bin/env bash
# Stage 4: the TRACE-only bond factors from Stage 2's panel, then the check against the
# files openbondassetpricing.com publishes.
#
#   bash run_stage4.sh                 # both sorts, then the comparison
#   bash run_stage4.sh --sort single   # any build_factors.py flag passes through
#
# PY overrides the interpreter: the Python where requirements-local.txt and PyBondLab are
# installed.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

echo "== building the factors =="
"$PY" build_factors.py "$@"

# A dry run computes nothing and a --signals run writes to output/_subset: nothing to compare.
case " $* " in *" --dry-run "*|*" --signals "*) exit 0 ;; esac
# Plain "$@", not an array: bash before 4.4 (macOS ships 3.2) treats an empty array as
# unbound under `set -u`.
sort=all
prev=""
for a in "$@"; do
  if [[ "$prev" == "--sort" ]]; then sort="$a"; fi
  prev="$a"
done

echo
echo "== comparing with the published files =="
"$PY" compare_published.py --sort "$sort"

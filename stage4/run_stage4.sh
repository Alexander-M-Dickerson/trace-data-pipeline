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

# Help and a dry run build nothing; a --signals or --return-types run writes a partial grid to
# output/_subset. None of them is compared with the published files.
# Plain "$@", not an array: bash before 4.4 (macOS ships 3.2) treats an empty array as
# unbound under `set -u`.
sort=all
prev=""
for a in "$@"; do
  case "$a" in
    -h|--help|--dry-run|--signals|--signals=*|--return-types|--return-types=*) exit 0 ;;
    --sort=*) sort="${a#--sort=}" ;;
  esac
  if [[ "$prev" == "--sort" ]]; then sort="$a"; fi
  prev="$a"
done

echo
echo "== comparing with the published files =="
"$PY" compare_published.py --sort "$sort"

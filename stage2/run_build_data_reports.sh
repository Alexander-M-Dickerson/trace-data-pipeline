#!/bin/bash
#
# Build the Stage 2 data report.
#
# Stage 2 runs on your own machine, not on the WRDS grid, so this is a plain script
# and not a qsub wrapper. There are no #$ directives here on purpose. Run it from
# the repository root:
#
#     bash stage2/run_build_data_reports.sh
#     bash stage2/run_build_data_reports.sh --no-external      # no network needed
#     bash stage2/run_build_data_reports.sh --mode stage1      # a specific build
#
# Any arguments are passed straight through to _build_data_report.py.
#
# Output lands in stage2/data_reports/ (gitignored): the .tex, its figures, and the
# PDF if pdflatex is installed.
#
# The report is built from a panel that already exists. It never triggers a build.
# Run the pipeline first if there is nothing under stage2/output/panel/.
#
# The two comparison suites (DFPS, WRDS Bond Returns) are on by default. They need
# the internet, and the WRDS one needs credentials. Pass --no-external to skip both.

set -euo pipefail

cd "$(dirname "$0")"

# PY overrides the interpreter, as in run_stage2.sh, and PYTHON is read too. The default is
# `python`: a Windows venv has no `python3`, and the one on PATH is a Store shortcut.
PY="${PY:-${PYTHON:-python}}"

exec "$PY" _build_data_report.py "$@"

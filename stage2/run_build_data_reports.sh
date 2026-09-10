#!/bin/bash
#
# Build the Stage 2 data report.
#
# Stage 2 runs on your OWN machine, not on the WRDS grid, so this is a plain script
# and not a qsub wrapper -- there are no #$ directives here on purpose. Run it from
# the repository root:
#
#     bash stage2/run_build_data_reports.sh
#     bash stage2/run_build_data_reports.sh --no-external      # no network needed
#     bash stage2/run_build_data_reports.sh --mode prod_final  # a specific build
#
# Any arguments are passed straight through to _build_data_report.py.
#
# Output lands in stage2/data_reports/ (gitignored): the .tex, its figures, and the
# PDF if pdflatex is installed.
#
# The report is built from a panel that already exists -- it never triggers a build.
# Run the pipeline first if there is nothing under stage2/output/panel/.
#
# The two comparison suites (DFPS, WRDS Bond Returns) are ON by default. They need
# the internet, and the WRDS one needs credentials; pass --no-external to skip both.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

exec "$PYTHON" _build_data_report.py "$@"

#!/bin/bash
# check_disk_space.sh -- is there room for a run? Called by run_pipeline.sh.
#
#   bash check_disk_space.sh              exit 0 when there is room, 1 when there is not
#   FORCE_RUN=1 bash check_disk_space.sh  report, and never refuse
#
# On WRDS the limit that matters is your HOME QUOTA (10 GB on most accounts), not the free
# space on the filesystem, which is terabytes and tells you nothing.
#
# The quota's LIMIT is taken from `quota`. The space USED is MEASURED, with `du`, at the moment
# you run this. It is not taken from `quota`, because WRDS refreshes that figure only every 30
# minutes. The usual way to start a run is to delete the last one first, and for the next half
# hour `quota` still counts the folder you just removed. This check used to trust it, and
# refused a run with 7.7 GB free by reporting 2.6.

set -uo pipefail

NEED_GB="${DISK_CHECK_NEED_GB:-4.0}"
MEASURE_TIMEOUT="${DISK_CHECK_TIMEOUT:-120}"      # seconds allowed for `du` over $HOME

echo ""
echo "=== DISK SPACE CHECK ==="

gb_from_kb() { awk -v kb="$1" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }'; }
is_number()  { [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; }

# Run a command with a time limit where `timeout` exists, and without one where it does not.
limited() {
    if command -v timeout >/dev/null 2>&1; then timeout "$MEASURE_TIMEOUT" "$@"; else "$@"; fi
}

filesystem_free() {
    local kb
    kb=$(df -Pk . 2>/dev/null | awk 'NR==2 {print $4}')
    if is_number "$kb"; then gb_from_kb "$kb"; else echo ""; fi
}

HERE=$(pwd -P)
HOME_REAL=$(cd "${HOME:-/nonexistent}" 2>/dev/null && pwd -P || echo "")
AVAIL_GB=""
BASIS=""
UNDER_QUOTA=0

# ---- the home quota, when there is one and the run lives under it ------------------------
if command -v quota >/dev/null 2>&1 && [[ -n "$HOME_REAL" ]] && [[ "$HERE/" == "$HOME_REAL/"* ]]; then
    # "    Home:  7.36GB / 10GB"
    QUOTA_LINE=$(quota 2>/dev/null | grep -i "Home:" | head -1)
    QUOTA_USED=$(echo "$QUOTA_LINE" | awk '{print $2}' | sed 's/GB//I')
    LIMIT=$(echo "$QUOTA_LINE" | awk '{print $4}' | sed 's/GB//I')

    if is_number "$LIMIT"; then
        UNDER_QUOTA=1
        echo "[info] Home quota limit: ${LIMIT} GB"
        echo "[info] Measuring what your home directory holds right now ..."
        USED_KB=$(limited du -sk "$HOME_REAL" 2>/dev/null | awk 'END {print $1}')

        if is_number "$USED_KB"; then
            USED=$(gb_from_kb "$USED_KB")
            BASIS="measured now with du"
            echo "[info] In use: ${USED} GB, ${BASIS}"
            if is_number "$QUOTA_USED"; then
                DRIFT=$(awk -v a="$QUOTA_USED" -v b="$USED" 'BEGIN { d = a - b; if (d < 0) d = -d; print (d > 0.5) }')
                if [[ "$DRIFT" == "1" ]]; then
                    echo "[info] \`quota\` reports ${QUOTA_USED} GB. WRDS refreshes that figure every 30 minutes,"
                    echo "       so it lags anything you deleted or wrote since. The measured figure is used."
                fi
            fi
        elif is_number "$QUOTA_USED"; then
            USED="$QUOTA_USED"
            BASIS="from \`quota\`, which can be 30 minutes old"
            echo "[warn] Could not measure your home directory within ${MEASURE_TIMEOUT} s."
            echo "[info] In use: ${USED} GB, ${BASIS}"
        else
            USED=""
        fi

        if is_number "${USED:-}"; then
            AVAIL_GB=$(awk -v l="$LIMIT" -v u="$USED" 'BEGIN { printf "%.2f", l - u }')
        fi
    fi
fi

# ---- anywhere else: the filesystem -------------------------------------------------------
if [[ -z "$AVAIL_GB" ]]; then
    UNDER_QUOTA=0
    AVAIL_GB=$(filesystem_free)
    BASIS="free space on this filesystem"
    if ! is_number "$AVAIL_GB"; then
        echo "[warn] Could not measure disk space here. Continuing without the check."
        echo ""
        exit 0
    fi
fi
echo "[info] Available: ${AVAIL_GB} GB (${BASIS})"

ENOUGH=$(awk -v a="$AVAIL_GB" -v n="$NEED_GB" 'BEGIN { print (a >= n) }')
if [[ "$ENOUGH" == "1" ]]; then
    echo "[ok] Enough room for a run (${AVAIL_GB} GB available, ${NEED_GB} GB needed)"
    echo ""
    exit 0
fi

# ---- not enough: say so, say why, and say what to do -------------------------------------
row() { printf '║  %-62s║\n' "$1"; }
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
row "NOT ENOUGH DISK SPACE FOR A RUN"
echo "╠════════════════════════════════════════════════════════════════╣"
row "Available: ${AVAIL_GB} GB"
row "Needed:    ${NEED_GB} GB"
row ""
row "A run that fills the disk part-way fails or writes a"
row "truncated file. Free some space, then run this again."
echo "╚════════════════════════════════════════════════════════════════╝"

if [[ "$UNDER_QUOTA" == "1" ]]; then
    echo ""
    echo "The largest things in your home directory:"
    (
        shopt -s nullglob dotglob               # hidden folders too, and no literal "*" when empty
        items=("$HOME_REAL"/*)
        if (( ${#items[@]} )); then
            limited du -sk "${items[@]}" 2>/dev/null | sort -n | tail -6 \
                | awk -v home="$HOME_REAL" '{ kb = $1; $1 = ""; sub(/^ +/, ""); sub("^" home, "~");
                                              printf "  %8.2f GB  %s\n", kb / 1024 / 1024, $0 }'
        fi
    )
    echo ""
    echo "Usual causes: an earlier run's folder, or a trace-data-pipeline.zip left in ~"
    echo "(zip to /scratch instead, see QUICKSTART.md)."
fi

echo ""
if [[ "${FORCE_RUN:-0}" == "1" ]]; then
    echo "[warn] FORCE_RUN=1 is set. Continuing with ${AVAIL_GB} GB available."
    echo ""
    exit 0
fi
echo "[error] Stopping. To run anyway: FORCE_RUN=1 ./run_pipeline.sh"
echo ""
exit 1

#!/bin/bash
# [tag:entry.download_inputs] fetches stage 1's inputs: bash download_inputs.sh, on the login node
# download_inputs.sh -- fetch the external files stage 1 needs.
#
# Stage 1 needs three things this repo does not ship: the Liu-Wu zero-coupon treasury
# curve, the bond-firm linker, and the Fama-French industry classifications (three files). They come
# off the internet, and WRDS COMPUTE NODES HAVE NO INTERNET -- so this has to run on
# the login node, before anything is submitted.
#
#   ./download_inputs.sh
#   ./download_inputs.sh --check    say which of the files are already here; download nothing
#
# run_pipeline.sh calls this for you. Run it yourself before ./run_smoke_test.sh,
# which is submitted straight to the grid and so cannot fetch anything.
#
# Safe to re-run: each file is downloaded beside its destination and moved into place only
# when it arrived whole, so a failed download leaves the previous copy exactly as it was.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1
[[ $CHECK_ONLY -eq 1 ]] || mkdir -p stage1/data

# fetch URL DEST [zip]: download to DEST.part, check it is non-empty (and, for a zip -- an
# .xlsx is one -- that it opens), then move it over DEST. `wget -O DEST` would empty DEST before the download starts,
# so a failed re-run would replace a good file with an empty one.
fetch() {
    local url="$1" dest="$2" kind="${3:-}" part="$2.part"
    rm -f "$part"
    if wget -q -O "$part" "$url" && [[ -s "$part" ]]; then
        if [[ "$kind" == "zip" ]] && ! unzip -tq "$part" >/dev/null 2>&1; then
            rm -f "$part"
            return 1
        fi
        mv -f "$part" "$dest"
        return 0
    fi
    rm -f "$part"
    return 1
}

# fetch_zip URL NAME LABEL: fetch a zip and extract it into stage1/data/. A failed download
# keeps whatever an earlier run extracted.
fetch_zip() {
    local url="$1" zip="stage1/data/$2" label="$3"
    echo "[download] ${label}..."
    if fetch "$url" "$zip" zip; then
        echo "[ok] ${label} downloaded"
        unzip -q -o "$zip" -d stage1/data/ \
            && echo "[ok] ${label} extracted" \
            || echo "[warn] Failed to extract ${label}"
        rm -f "$zip"
    else
        echo "[warn] Failed to download ${label}; any copy from an earlier run is kept"
    fi
}

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "[check] Downloading nothing; checking the files already here."
else
    # [tag:trap.no_internet] WRDS compute nodes have no internet, so this runs on the login node first
    # Download required data files for Stage 1 (WRDS compute nodes have no internet)
    # This must be done on the login node before submitting jobs
    echo ""
    echo "=== PRE-STAGE: Downloading Required Data Files ==="
    echo "[download] Liu-Wu treasury yields..."
    fetch "https://docs.google.com/spreadsheets/d/11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t/export?format=xlsx&id=11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t" \
          stage1/data/liu_wu_yields.xlsx zip \
        && echo "[ok] Liu-Wu yields downloaded" \
        || echo "[warn] Failed to download Liu-Wu yields; any copy from an earlier run is kept"

    fetch_zip "https://openbondassetpricing.com/wp-content/uploads/2026/09/bond_firm_linker_2026.zip" \
              bond_firm_linker_2026.zip "Bond-firm linker"

    # The release ships its own checker: it re-derives every count in its docs from the parquets
    # and exits non-zero if anything drifted. One second here beats discovering a damaged linker
    # seven hours into the pipeline, so a failure stops the run.
    if [[ -f "stage1/data/bond_firm_linker_2026/verify_release.py" ]]; then
        echo "[verify] Checking bond-firm linker release..."
        if out=$(cd stage1/data/bond_firm_linker_2026 && python3 verify_release.py 2>&1); then
            echo "[ok] Bond-firm linker release verified"
        else
            echo "$out" | tail -20
            echo "[error] The bond-firm linker failed its own check (above). If it names a"
            echo "[error] missing package, install requirements.txt; otherwise delete"
            echo "[error] stage1/data/bond_firm_linker_2026/ and run this script again."
            exit 1
        fi
    fi

    for n in 12 17 30; do
        fetch_zip "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes${n}.zip" \
                  "Siccodes${n}.zip" "Fama-French ${n} Industry Classification"
    done
fi

echo "[verify] Checking downloaded files..."
MISSING_FILES=0
for file in "liu_wu_yields.xlsx" "bond_firm_linker_2026/fl_linker.parquet" "Siccodes12.txt" "Siccodes17.txt" "Siccodes30.txt"; do
    # -s, not -f: an empty file is as missing as no file.
    if [[ -s "stage1/data/$file" ]]; then
        echo "[ok] $file"
    else
        echo "[error] Missing or empty: $file"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [[ $MISSING_FILES -gt 0 ]]; then
    # Exit non-zero rather than warn-and-continue, which is what this used to do inside
    # run_pipeline.sh. Every one of these files is required for stage 1's 44-column
    # output -- the yields for credit_spread, the linker for permno, the FF files for
    # the industry codes -- so a missing one is not a warning, it is a run that fails
    # several hours later having burned the grid time.
    echo "[error] ${MISSING_FILES} required file(s) missing. Not continuing."
    echo "[error] Check your internet access on this node, or fetch them by hand"
    echo "[error] following stage1/QUICKSTART_stage1.md."
    exit 1
fi
echo "[ok] All required data files present"

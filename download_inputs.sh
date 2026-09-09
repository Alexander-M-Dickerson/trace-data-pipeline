#!/bin/bash
# download_inputs.sh -- fetch the external files stage 1 needs.
#
# Stage 1 needs four things this repo does not ship: the Liu-Wu zero-coupon treasury
# curve, the bond-firm linker, and the Fama-French industry classifications. They come
# off the internet, and WRDS COMPUTE NODES HAVE NO INTERNET -- so this has to run on
# the login node, before anything is submitted.
#
#   ./download_inputs.sh
#
# run_pipeline.sh calls this for you. Run it yourself before ./run_smoke_test.sh,
# which is submitted straight to the grid and so cannot fetch anything.
#
# Safe to re-run: every download overwrites, and a failure is reported rather than
# silently leaving a truncated file in place.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p stage1/data

# Download required data files for Stage 1 (WRDS compute nodes have no internet)
# This must be done on the login node before submitting jobs
echo ""
echo "=== PRE-STAGE: Downloading Required Data Files ==="
echo "[download] Liu-Wu treasury yields..."
wget -q -O stage1/data/liu_wu_yields.xlsx \
    "https://docs.google.com/spreadsheets/d/11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t/export?format=xlsx&id=11HsxLl_u2tBNt3FyN5iXGsIKLwxvVz7t" \
    && echo "[ok] Liu-Wu yields downloaded" \
    || echo "[warn] Failed to download Liu-Wu yields (may already exist)"

echo "[download] Bond-firm linker..."
wget -q -O stage1/data/bond_firm_linker_2026.zip \
    "https://openbondassetpricing.com/wp-content/uploads/2026/09/bond_firm_linker_2026.zip" \
    && echo "[ok] Bond-firm linker downloaded" \
    || echo "[warn] Failed to download bond-firm linker (may already exist)"

if [[ -f "stage1/data/bond_firm_linker_2026.zip" ]]; then
    echo "[extract] Unzipping bond-firm linker..."
    unzip -q -o stage1/data/bond_firm_linker_2026.zip -d stage1/data/ \
        && echo "[ok] Bond-firm linker extracted" \
        || echo "[warn] Failed to extract bond-firm linker"
    rm -f stage1/data/bond_firm_linker_2026.zip

    # The release ships its own checker: it re-derives every count in its docs from
    # the parquets and exits non-zero if anything drifted. One second here beats
    # discovering a truncated download seven hours into the pipeline.
    if [[ -f "stage1/data/bond_firm_linker_2026/verify_release.py" ]]; then
        echo "[verify] Checking bond-firm linker release..."
        ( cd stage1/data/bond_firm_linker_2026 && python3 verify_release.py >/dev/null 2>&1 ) \
            && echo "[ok] Bond-firm linker release verified" \
            || echo "[warn] verify_release.py reported a problem -- check the linker download"
    fi
fi

echo "[download] Fama-French 12 Industry Classification..."
wget -q -O stage1/data/Siccodes12.zip \
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes12.zip" \
    && echo "[ok] FF12 downloaded" \
    || echo "[warn] Failed to download FF12 (may already exist)"

if [[ -f "stage1/data/Siccodes12.zip" ]]; then
    echo "[extract] Unzipping FF12..."
    unzip -q -o stage1/data/Siccodes12.zip -d stage1/data/ \
        && echo "[ok] FF12 extracted" \
        || echo "[warn] Failed to extract FF12"
    rm -f stage1/data/Siccodes12.zip
fi

echo "[download] Fama-French 17 Industry Classification..."
wget -q -O stage1/data/Siccodes17.zip \
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes17.zip" \
    && echo "[ok] FF17 downloaded" \
    || echo "[warn] Failed to download FF17 (may already exist)"

if [[ -f "stage1/data/Siccodes17.zip" ]]; then
    echo "[extract] Unzipping FF17..."
    unzip -q -o stage1/data/Siccodes17.zip -d stage1/data/ \
        && echo "[ok] FF17 extracted" \
        || echo "[warn] Failed to extract FF17"
    rm -f stage1/data/Siccodes17.zip
fi

echo "[download] Fama-French 30 Industry Classification..."
wget -q -O stage1/data/Siccodes30.zip \
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes30.zip" \
    && echo "[ok] FF30 downloaded" \
    || echo "[warn] Failed to download FF30 (may already exist)"

if [[ -f "stage1/data/Siccodes30.zip" ]]; then
    echo "[extract] Unzipping FF30..."
    unzip -q -o stage1/data/Siccodes30.zip -d stage1/data/ \
        && echo "[ok] FF30 extracted" \
        || echo "[warn] Failed to extract FF30"
    rm -f stage1/data/Siccodes30.zip
fi

echo "[verify] Checking downloaded files..."
MISSING_FILES=0
for file in "liu_wu_yields.xlsx" "bond_firm_linker_2026/fl_linker.parquet" "Siccodes12.txt" "Siccodes17.txt" "Siccodes30.txt"; do
    if [[ -f "stage1/data/$file" ]]; then
        echo "[ok] $file"
    else
        echo "[error] Missing: $file"
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

#!/bin/bash
# Run preprocess + XGBoost for a 3x3 matrix:
#   rows    : O0-only | O3-only | Combined (O0+O3)
#   columns : 80 | 160 | 320 (or custom sizes)
#
# Usage  : bash combo.sh <folder_name> [size1 size2 size3]
# Example: bash combo.sh CVE-2025-11187-static-combine-10events_mix 80 160 320

set -euo pipefail

FOLDER="${1:?Usage: $0 <folder_name> [sizes...]}"
shift
SIZES=("${@}")
if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=(80 160 320)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER_PATH="$SCRIPT_DIR/$FOLDER"

if [ ! -d "$FOLDER_PATH" ]; then
    echo "ERROR: Folder not found: $FOLDER_PATH"
    exit 1
fi

# Auto-detect unique pattern prefixes (the part before _O<digit>_<number>.json)
mapfile -t PATTERNS < <(
    ls "$FOLDER_PATH/" \
    | sed -n 's/pmc_features_\(.*\)_O[0-9]*_[0-9]*\.json$/\1/p' \
    | sort -u
)

if [ ${#PATTERNS[@]} -eq 0 ]; then
    echo "ERROR: No pmc_features_*_O<n>_<n>.json files found in $FOLDER"
    exit 1
fi

echo "Folder   : $FOLDER"
echo "Patterns : ${PATTERNS[*]}"
echo "Sizes    : ${SIZES[*]}"
echo ""

# run_combo <sample_per_pattern_spec> -> prints test accuracy string
run_combo() {
    local spec="$1"
    python3 preprocess_features.py \
        --features "${FOLDER}/pmc_features_*.json" \
        --sample-per-pattern "$spec" \
        --output features_16 > /dev/null 2>&1

    python3 train_xgboost_gpu.py --cache 2>&1 \
        | grep "Test Accuracy:" | tail -1 \
        | sed 's/.*Test Accuracy: *//'
}

cd "$SCRIPT_DIR"

# Collect results into a 2-D associative array: results[ROW,COL]
declare -A results

ROWS=("O0" "O3" "Combined")

for SIZE in "${SIZES[@]}"; do
    # --- O0 only ---
    PARTS=()
    for PATTERN in "${PATTERNS[@]}"; do
        PARTS+=("${PATTERN}_O0:${SIZE}")
    done
    SPEC=$(IFS=','; echo "${PARTS[*]}")
    results["O0,$SIZE"]=$(run_combo "$SPEC")

    # --- O3 only ---
    PARTS=()
    for PATTERN in "${PATTERNS[@]}"; do
        PARTS+=("${PATTERN}_O3:${SIZE}")
    done
    SPEC=$(IFS=','; echo "${PARTS[*]}")
    results["O3,$SIZE"]=$(run_combo "$SPEC")

    # --- Combined (O0 + O3, same size each) ---
    PARTS=()
    for PATTERN in "${PATTERNS[@]}"; do
        PARTS+=("${PATTERN}_O0:${SIZE}")
        PARTS+=("${PATTERN}_O3:${SIZE}")
    done
    SPEC=$(IFS=','; echo "${PARTS[*]}")
    results["Combined,$SIZE"]=$(run_combo "$SPEC")
done

# Print results as a matrix
COL_W=14
printf "%-12s" ""
for SIZE in "${SIZES[@]}"; do
    printf "%-${COL_W}s" "$SIZE"
done
echo ""

printf "%-12s" "$(printf '%0.s-' {1..12})"
for SIZE in "${SIZES[@]}"; do
    printf "%-${COL_W}s" "$(printf '%0.s-' $(seq 1 $((COL_W-1))))"
done
echo ""

for ROW in "${ROWS[@]}"; do
    printf "%-12s" "$ROW"
    for SIZE in "${SIZES[@]}"; do
        printf "%-${COL_W}s" "${results[$ROW,$SIZE]}"
    done
    echo ""
done

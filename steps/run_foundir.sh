#!/usr/bin/env bash
# Usage: steps/run_foundir.sh <input_dir> <output_dir> <errors_log>
set -euo pipefail

INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath "$2")"
ERRORS_LOG="$(realpath "$3")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../config.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_FOUNDIR" || { echo "[foundir] failed to activate conda env '$CONDA_ENV_FOUNDIR'" >> "$ERRORS_LOG"; exit 1; }

mkdir -p "$OUTPUT_DIR"

# FoundIR outputs to ./results/ relative to its own directory.
# Weights must be at $FOUNDIR_DIR/premodel/model-2000.pt
cd "$FOUNDIR_DIR"
rm -rf ./results

if ! python test.py --dataroot "$INPUT_DIR" --meta None 2>>"$ERRORS_LOG"; then
    echo "[foundir] test.py failed — check errors.log for details" >> "$ERRORS_LOG"
    exit 1
fi

# Copy results to pipeline output dir, logging failures per file
PASS=0; FAIL=0
for src in ./results/*.png; do
    [ -f "$src" ] || { echo "[foundir] test.py produced no PNG output in ./results/" >> "$ERRORS_LOG"; FAIL=$((FAIL+1)); continue; }
    fname="$(basename "$src")"
    if cp "$src" "$OUTPUT_DIR/$fname" 2>>"$ERRORS_LOG"; then
        PASS=$((PASS+1))
    else
        echo "[foundir] failed to copy $fname" >> "$ERRORS_LOG"
        FAIL=$((FAIL+1))
    fi
done

echo "foundir: $PASS passed, $FAIL failed"

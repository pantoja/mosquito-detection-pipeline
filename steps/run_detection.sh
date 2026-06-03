#!/usr/bin/env bash
# Usage: steps/run_detection.sh <input_dir> <output_dir> <detector> <errors_log>
set -euo pipefail

INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath "$2")"
DETECTOR="$3"
ERRORS_LOG="$(realpath "$4")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../config.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_DETECT" || { echo "[detection] failed to activate conda env '$CONDA_ENV_DETECT'" >> "$ERRORS_LOG"; exit 1; }

if [ "$DETECTOR" = "yolo" ]; then
    WEIGHTS="$YOLO_WEIGHTS"
else
    WEIGHTS="$RTDETR_WEIGHTS"
fi

python "$SCRIPT_DIR/detect.py" \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --detector   "$DETECTOR" \
    --weights    "$WEIGHTS" \
    --errors-log "$ERRORS_LOG"

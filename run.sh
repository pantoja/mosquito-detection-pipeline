#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 --input <image_or_folder> --detector <yolo|rtdetr>"
    exit 1
}

INPUT=""
DETECTOR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)    [[ $# -ge 2 ]] || usage; INPUT="$2";    shift 2 ;;
        --detector) [[ $# -ge 2 ]] || usage; DETECTOR="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$INPUT" || -z "$DETECTOR" ]] && usage
[[ "$DETECTOR" != "yolo" && "$DETECTOR" != "rtdetr" ]] && {
    echo "Error: --detector must be 'yolo' or 'rtdetr'"; exit 1; }
[[ ! -e "$INPUT" ]] && { echo "Error: input '$INPUT' does not exist"; exit 1; }

# ── Output directory setup ────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="$SCRIPT_DIR/output/$TIMESTAMP"
DIR_01="$OUT_BASE/01_foundir"
DIR_02="$OUT_BASE/02_foundir_compressed"
DIR_03="$OUT_BASE/03_stablesr"
DIR_04="$OUT_BASE/04_results"
ERRORS_LOG="$OUT_BASE/errors.log"

mkdir -p "$DIR_01" "$DIR_02" "$DIR_03" "$DIR_04"
touch "$ERRORS_LOG"

echo "Output: $OUT_BASE"
echo "Detector: $DETECTOR"
echo "----------------------------------------"

# ── Normalize single-file input to a directory ───────────────────────────────
TMP_INPUT=""
if [[ -f "$INPUT" ]]; then
    TMP_INPUT="$(mktemp -d)"
    cp "$INPUT" "$TMP_INPUT/"
    INPUT_DIR="$TMP_INPUT"
else
    INPUT_DIR="$(realpath "$INPUT")"
fi

cleanup() { [[ -n "$TMP_INPUT" ]] && rm -rf "$TMP_INPUT"; }
trap cleanup EXIT

# ── Run steps ────────────────────────────────────────────────────────────────
echo "[1/4] FoundIR restoration..."
"$SCRIPT_DIR/steps/run_foundir.sh"   "$INPUT_DIR" "$DIR_01" "$ERRORS_LOG"

echo "[2/4] Compression PNG→JPG..."
"$SCRIPT_DIR/steps/compress.sh"      "$DIR_01"    "$DIR_02" "$ERRORS_LOG"

echo "[3/4] StableSR super-resolution..."
"$SCRIPT_DIR/steps/run_stablesr.sh"  "$DIR_02"    "$DIR_03" "$ERRORS_LOG"

echo "[4/4] Object detection ($DETECTOR)..."
"$SCRIPT_DIR/steps/run_detection.sh" "$DIR_03"    "$DIR_04" "$DETECTOR" "$ERRORS_LOG"

# ── Summary ───────────────────────────────────────────────────────────────────
echo "----------------------------------------"
echo "Done. Results in: $OUT_BASE"
if [[ -s "$ERRORS_LOG" ]]; then
    echo "Errors logged:"
    cat "$ERRORS_LOG"
fi

#!/usr/bin/env bash
# Usage: steps/compress.sh <input_dir> <output_dir> <errors_log>
set -euo pipefail

INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath "$2")"
ERRORS_LOG="$(realpath "$3")"

mkdir -p "$OUTPUT_DIR"

PASS=0; FAIL=0

for src in "$INPUT_DIR"/*.png; do
    [ -f "$src" ] || continue
    fname="$(basename "${src%.png}.jpg")"
    dest="$OUTPUT_DIR/$fname"

    if command -v ffmpeg &>/dev/null; then
        if ffmpeg -y -i "$src" -q:v 2 "$dest" 2>>"$ERRORS_LOG"; then
            PASS=$((PASS+1))
        else
            echo "[compress] ffmpeg failed on $src" >> "$ERRORS_LOG"
            FAIL=$((FAIL+1))
        fi
    else
        if SRC="$src" DEST="$dest" python3 -c "
import os
from PIL import Image
img = Image.open(os.environ['SRC']).convert('RGB')
img.save(os.environ['DEST'], 'JPEG', quality=95)
" 2>>"$ERRORS_LOG"; then
            PASS=$((PASS+1))
        else
            echo "[compress] pillow failed on $src" >> "$ERRORS_LOG"
            FAIL=$((FAIL+1))
        fi
    fi
done

echo "compress: $PASS passed, $FAIL failed"

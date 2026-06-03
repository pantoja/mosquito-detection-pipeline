#!/usr/bin/env bash
# Usage: steps/run_stablesr.sh <input_dir> <output_dir> <errors_log>
set -euo pipefail

INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath "$2")"
ERRORS_LOG="$(realpath "$3")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../config.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_STABLESR" || { echo "[stablesr] failed to activate conda env '$CONDA_ENV_STABLESR'" >> "$ERRORS_LOG"; exit 1; }

mkdir -p "$OUTPUT_DIR"

cd "$STABLESR_DIR"

PASS=0; FAIL=0

for src in "$INPUT_DIR"/*.jpg; do
    [ -f "$src" ] || continue
    fname="$(basename "$src")"

    if python scripts/sr_val_ddpm_text_T_vqganfin_old.py \
        --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
        --ckpt "$STABLESR_CKPT" \
        --vqgan_ckpt "$STABLESR_VQGAN_CKPT" \
        --init-img "$src" \
        --outdir "$OUTPUT_DIR" \
        --ddpm_steps 200 \
        --dec_w 0.5 \
        --colorfix_type adain \
        --n_samples 1 2>>"$ERRORS_LOG"; then
        PASS=$((PASS+1))
    else
        echo "[stablesr] failed on $fname" >> "$ERRORS_LOG"
        FAIL=$((FAIL+1))
    fi
done

echo "stablesr: $PASS passed, $FAIL failed"

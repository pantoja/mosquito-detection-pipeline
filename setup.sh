#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Image Pipeline Setup ==="
echo ""

# ── Clone repos ───────────────────────────────────────────────────────────────
clone_if_missing() {
    local url="$1" dest="$2" name="$3"
    if [[ -d "$dest/.git" ]]; then
        echo "[$name] already cloned at $dest"
    else
        echo "[$name] cloning..."
        git clone "$url" "$dest"
        echo "[$name] done"
    fi
}

clone_if_missing "https://github.com/House-Leo/FoundIR"          "$FOUNDIR_DIR"  "FoundIR"
clone_if_missing "https://github.com/IceClear/StableSR"          "$STABLESR_DIR" "StableSR"

echo ""

# ── Create conda environments ─────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"

create_env() {
    local name="$1" req_file="$2"
    if conda info --envs | grep -qE "^$name\s"; then
        echo "[conda] environment '$name' already exists"
    else
        echo "[conda] creating '$name'..."
        conda env create -n "$name" -f "$req_file"
        echo "[conda] '$name' created"
    fi
}

create_env "$CONDA_ENV_FOUNDIR"  "$FOUNDIR_DIR/environment.yaml"
create_env "$CONDA_ENV_STABLESR" "$STABLESR_DIR/environment.yaml"

# Detection env: create from scratch with ultralytics (covers YOLO + RT-DETR)
if conda info --envs | grep -qE "^$CONDA_ENV_DETECT\s"; then
    echo "[conda] environment '$CONDA_ENV_DETECT' already exists"
else
    echo "[conda] creating '$CONDA_ENV_DETECT' with ultralytics..."
    conda create -n "$CONDA_ENV_DETECT" python=3.10 -y
    conda run -n "$CONDA_ENV_DETECT" pip install ultralytics opencv-python-headless
    echo "[conda] '$CONDA_ENV_DETECT' created"
fi

echo ""

# ── Weight download ───────────────────────────────────────────────────────────
echo "=== Weights ==="
HF_REPO="vicpantoja2/aedes-egg-weights"

if command -v huggingface-cli &>/dev/null; then
    if [[ ! -t 0 ]]; then
        echo "Non-interactive shell detected; skipping weight download prompt."
        echo "Run setup.sh interactively to download weights, or see README.md."
    else
    read -rp "Download weights via huggingface-cli? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "Downloading FoundIR weights..."
        huggingface-cli download "$HF_REPO" --include "foundir/*" --local-dir /tmp/pipeline_weights
        echo ""
        echo "Place the downloaded weight file as:"
        echo "  $FOUNDIR_DIR/premodel/model-2000.pt"
        echo ""
        echo "Downloading StableSR weights..."
        huggingface-cli download "$HF_REPO" --include "stablesr/*" --local-dir /tmp/pipeline_weights
        echo "Set STABLESR_CKPT and STABLESR_VQGAN_CKPT in config.sh"
        echo ""
        echo "Downloading detection weights..."
        huggingface-cli download "$HF_REPO" --include "yolov26/*" "rtdetr/*" --local-dir /tmp/pipeline_weights
        echo "Set YOLO_WEIGHTS and RTDETR_WEIGHTS in config.sh"
    fi
    fi
else
    echo "huggingface-cli not found. Download weights manually from README.md"
fi

echo ""
echo "=== Setup complete ==="
echo "Next: edit config.sh with your model paths, then run:"
echo "  ./run.sh --input /path/to/images --detector yolo"

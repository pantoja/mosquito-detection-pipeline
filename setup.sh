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
echo "=== Pesos ==="
HF_REPO="vicpantoja2/aedes-egg-weights"
WEIGHTS_DIR="$SCRIPT_DIR/weights"

if command -v huggingface-cli &>/dev/null; then
    if [[ ! -t 0 ]]; then
        echo "Shell não-interativo detectado; pulando download de pesos."
        echo "Execute setup.sh interativamente para baixar os pesos, ou consulte o README.md."
    else
        read -rp "Baixar pesos via huggingface-cli? [s/N] " answer
        if [[ "$answer" =~ ^[SsYy]$ ]]; then
            echo "Baixando pesos FoundIR..."
            huggingface-cli download "$HF_REPO" --include "foundir/*" --local-dir "$WEIGHTS_DIR"

            echo "Baixando pesos StableSR..."
            huggingface-cli download "$HF_REPO" --include "stablesr/*" --local-dir "$WEIGHTS_DIR"

            echo "Baixando pesos de detecção..."
            huggingface-cli download "$HF_REPO" --include "yolov26/*" "rtdetr/*" --local-dir "$WEIGHTS_DIR"

            echo ""
            echo "Configurando caminhos dos pesos automaticamente..."

            # FoundIR: copy to hardcoded path required by test.py
            FOUNDIR_PT=$(ls "$WEIGHTS_DIR/foundir/"*.pt 2>/dev/null | head -1 || true)
            if [[ -n "$FOUNDIR_PT" ]]; then
                mkdir -p "$FOUNDIR_DIR/premodel"
                cp "$FOUNDIR_PT" "$FOUNDIR_DIR/premodel/model-2000.pt"
                echo "[FoundIR] peso copiado para $FOUNDIR_DIR/premodel/model-2000.pt"
            else
                echo "[FoundIR] AVISO: nenhum .pt encontrado em $WEIGHTS_DIR/foundir/"
            fi

            # StableSR: detect VQGAN (matches *vqgan*) and main checkpoint (the rest)
            STABLESR_VQGAN=$(ls "$WEIGHTS_DIR/stablesr/"*vqgan* 2>/dev/null | head -1 || true)
            STABLESR_MAIN=$(ls "$WEIGHTS_DIR/stablesr/"*.ckpt "$WEIGHTS_DIR/stablesr/"*.pth 2>/dev/null \
                | grep -iv vqgan | head -1 || true)

            # Detection weights
            YOLO_PT=$(ls "$WEIGHTS_DIR/yolov26/"*.pt 2>/dev/null | head -1 || true)
            RTDETR_PT=$(ls "$WEIGHTS_DIR/rtdetr/"*.pt 2>/dev/null | head -1 || true)

            # Write weights.env.sh — sourced automatically by config.sh
            cat > "$SCRIPT_DIR/weights.env.sh" << EOF
# Gerado automaticamente por setup.sh — não editar manualmente
STABLESR_CKPT="$STABLESR_MAIN"
STABLESR_VQGAN_CKPT="$STABLESR_VQGAN"
YOLO_WEIGHTS="$YOLO_PT"
RTDETR_WEIGHTS="$RTDETR_PT"
EOF
            echo "[pesos] caminhos salvos em weights.env.sh"
        fi
    fi
else
    echo "huggingface-cli não encontrado. Baixe os pesos manualmente conforme o README.md"
fi

echo ""
echo "=== Setup concluído ==="
echo "Próximo passo: execute o pipeline com:"
echo "  ./run.sh --input /caminho/para/imagens --detector yolo"

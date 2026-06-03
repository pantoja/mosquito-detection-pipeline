#!/usr/bin/env bash
# Edit all paths below before running the pipeline.

# FoundIR
# Clone: https://github.com/House-Leo/FoundIR
# Weights must be placed at: $FOUNDIR_DIR/premodel/model-2000.pt
FOUNDIR_DIR="/path/to/FoundIR"

# StableSR
# Clone: https://github.com/IceClear/StableSR
# Needs two checkpoint files: main model and VQGAN
STABLESR_DIR="/path/to/StableSR"
STABLESR_CKPT="/path/to/stablesr_model.ckpt"
STABLESR_VQGAN_CKPT="/path/to/vqgan_cfw_00011.ckpt"

# Detection weights (ultralytics .pt format)
YOLO_WEIGHTS="/path/to/yolov26.pt"
RTDETR_WEIGHTS="/path/to/rtdetr.pt"

# Conda environment names (created by setup.sh)
CONDA_ENV_FOUNDIR="foundir"
CONDA_ENV_STABLESR="stablesr"
CONDA_ENV_DETECT="detect"

# Caminhos dos pesos gerados automaticamente pelo setup.sh — não editar
_PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$_PIPELINE_DIR/weights.env.sh" ]] && source "$_PIPELINE_DIR/weights.env.sh"
unset _PIPELINE_DIR

# Image Processing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bash pipeline that runs PNG images through FoundIR → compression → StableSR → object detection (YOLOv26 or RT-DETR), with staged outputs per step.

**Architecture:** An orchestrator (`run.sh`) sources a user-editable config and calls four step scripts sequentially. Each step receives an input directory and writes to its own output subdirectory. Failures are logged per image and the pipeline continues. A separate `setup.sh` handles repo cloning and environment creation.

**Tech Stack:** Bash, Conda, Python 3, ffmpeg/Pillow (compression), ultralytics (YOLO + RT-DETR detection), FoundIR (residual diffusion restoration), StableSR (stable diffusion SR), huggingface-cli (optional weight download).

---

## File Map

| File | Responsibility |
|------|---------------|
| `config.sh` | All user-editable paths and conda env names |
| `run.sh` | Orchestrator: parse args, create output dirs, call steps, print summary |
| `steps/run_foundir.sh` | Activate foundir env, run FoundIR test.py, copy results |
| `steps/compress.sh` | Convert PNG→JPG at quality 95 using ffmpeg or Pillow |
| `steps/run_stablesr.sh` | Activate stablesr env, run StableSR inference |
| `steps/run_detection.sh` | Activate detect env, call detect.py |
| `steps/detect.py` | Python: run YOLO or RT-DETR via ultralytics, save annotated images + JSON |
| `setup.sh` | Clone repos, create conda envs, offer weight download |
| `README.md` | Weight URLs, setup instructions, usage examples |

---

## Task 1: Create `config.sh`

**Files:**
- Create: `config.sh`

- [ ] **Step 1: Write config.sh**

```bash
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
```

- [ ] **Step 2: Commit**

```bash
git add config.sh
git commit -m "feat: add config.sh with user-editable paths"
```

---

## Task 2: Create `steps/run_foundir.sh`

**Files:**
- Create: `steps/run_foundir.sh`

FoundIR's `test.py` hardcodes its weight path to `premodel/model-2000.pt` relative to the FoundIR directory, and outputs to `./results/`. This script runs `test.py` from inside the FoundIR directory and copies results to the pipeline output dir.

- [ ] **Step 1: Create steps/ directory**

```bash
mkdir -p steps
```

- [ ] **Step 2: Write run_foundir.sh**

```bash
#!/usr/bin/env bash
# Usage: steps/run_foundir.sh <input_dir> <output_dir> <errors_log>
set -euo pipefail

INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath "$2")"
ERRORS_LOG="$(realpath "$3")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../config.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_FOUNDIR"

mkdir -p "$OUTPUT_DIR"

# FoundIR outputs to ./results/ relative to its own directory.
# Weights must be at $FOUNDIR_DIR/premodel/model-2000.pt
cd "$FOUNDIR_DIR"
rm -rf ./results

python test.py --dataroot "$INPUT_DIR" --meta None

# Copy results to pipeline output dir, logging failures per file
PASS=0; FAIL=0
for src in ./results/*.png; do
    [ -f "$src" ] || { echo "[foundir] no output files found" >> "$ERRORS_LOG"; FAIL=$((FAIL+1)); continue; }
    fname="$(basename "$src")"
    if cp "$src" "$OUTPUT_DIR/$fname" 2>>"$ERRORS_LOG"; then
        PASS=$((PASS+1))
    else
        echo "[foundir] failed to copy $fname" >> "$ERRORS_LOG"
        FAIL=$((FAIL+1))
    fi
done

echo "foundir: $PASS passed, $FAIL failed"
```

- [ ] **Step 3: Make executable**

```bash
chmod +x steps/run_foundir.sh
```

- [ ] **Step 4: Commit**

```bash
git add steps/run_foundir.sh
git commit -m "feat: add run_foundir.sh step script"
```

---

## Task 3: Create `steps/compress.sh`

**Files:**
- Create: `steps/compress.sh`

Converts every PNG in the input dir to JPG at quality 95. Uses ffmpeg if available, falls back to Python Pillow.

- [ ] **Step 1: Write compress.sh**

```bash
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
        if python3 -c "
from PIL import Image
img = Image.open('$src').convert('RGB')
img.save('$dest', 'JPEG', quality=95)
" 2>>"$ERRORS_LOG"; then
            PASS=$((PASS+1))
        else
            echo "[compress] pillow failed on $src" >> "$ERRORS_LOG"
            FAIL=$((FAIL+1))
        fi
    fi
done

echo "compress: $PASS passed, $FAIL failed"
```

Note on `ffmpeg -q:v 2`: ffmpeg's JPEG quality scale runs 2 (best) to 31 (worst), so `-q:v 2` is maximum quality, equivalent to ~95 in Pillow terms.

- [ ] **Step 2: Make executable**

```bash
chmod +x steps/compress.sh
```

- [ ] **Step 3: Commit**

```bash
git add steps/compress.sh
git commit -m "feat: add compress.sh step script"
```

---

## Task 4: Create `steps/run_stablesr.sh`

**Files:**
- Create: `steps/run_stablesr.sh`

StableSR requires two checkpoints: the main diffusion model and the VQGAN. It is run from inside the StableSR directory so its config references resolve correctly.

- [ ] **Step 1: Write run_stablesr.sh**

```bash
#!/usr/bin/env bash
# Usage: steps/run_stablesr.sh <input_dir> <output_dir> <errors_log>
set -euo pipefail

INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath "$2")"
ERRORS_LOG="$(realpath "$3")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../config.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_STABLESR"

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
```

- [ ] **Step 2: Make executable**

```bash
chmod +x steps/run_stablesr.sh
```

- [ ] **Step 3: Commit**

```bash
git add steps/run_stablesr.sh
git commit -m "feat: add run_stablesr.sh step script"
```

---

## Task 5: Create `steps/detect.py` and `steps/run_detection.sh`

**Files:**
- Create: `steps/detect.py`
- Create: `steps/run_detection.sh`

`detect.py` uses the ultralytics Python API which natively supports both YOLO and RT-DETR. It processes all images in the input dir, saves annotated images to `<output_dir>/annotated/` and per-image JSON files to `<output_dir>/data/`.

- [ ] **Step 1: Write detect.py**

```python
#!/usr/bin/env python3
"""Run YOLO or RT-DETR detection on a directory of images."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
from ultralytics import RTDETR, YOLO


def run(input_dir: str, output_dir: str, detector: str, weights: str, errors_log: str) -> None:
    input_path = Path(input_dir)
    out_annotated = Path(output_dir) / "annotated"
    out_data = Path(output_dir) / "data"
    out_annotated.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    if detector == "yolo":
        model = YOLO(weights)
    else:
        model = RTDETR(weights)

    image_files = sorted(
        p for p in input_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    passed = 0
    failed = 0

    for img_path in image_files:
        try:
            results = model(str(img_path), verbose=False)
            result = results[0]

            # Save annotated image
            annotated = result.plot()
            out_img = out_annotated / img_path.name
            cv2.imwrite(str(out_img), annotated)

            # Save JSON detections
            detections = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    detections.append({
                        "class_id": cls_id,
                        "class_name": result.names[cls_id],
                        "confidence": round(float(boxes.conf[i].item()), 4),
                        "bbox_xyxy": [round(float(v), 2) for v in boxes.xyxy[i].tolist()],
                    })

            json_out = out_data / (img_path.stem + ".json")
            with open(json_out, "w") as f:
                json.dump({"image": img_path.name, "detections": detections}, f, indent=2)

            passed += 1
        except Exception as exc:  # noqa: BLE001
            with open(errors_log, "a") as f:
                f.write(f"[detection] failed on {img_path.name}: {exc}\n")
            failed += 1

    print(f"detection: {passed} passed, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detector", required=True, choices=["yolo", "rtdetr"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--errors-log", required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.detector, args.weights, args.errors_log)
```

- [ ] **Step 2: Write run_detection.sh**

```bash
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
conda activate "$CONDA_ENV_DETECT"

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
```

- [ ] **Step 3: Make scripts executable**

```bash
chmod +x steps/run_detection.sh steps/detect.py
```

- [ ] **Step 4: Commit**

```bash
git add steps/detect.py steps/run_detection.sh
git commit -m "feat: add detection step (YOLO + RT-DETR via ultralytics)"
```

---

## Task 6: Create `run.sh`

**Files:**
- Create: `run.sh`

The orchestrator validates inputs, creates the timestamped output tree, calls each step, and prints a summary. If `--input` is a single file, it is placed into a temp directory so all step scripts always receive a directory.

- [ ] **Step 1: Write run.sh**

```bash
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
        --input)    INPUT="$2";    shift 2 ;;
        --detector) DETECTOR="$2"; shift 2 ;;
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
if [[ -f "$INPUT" ]]; then
    TMP_INPUT="$(mktemp -d)"
    cp "$INPUT" "$TMP_INPUT/"
    INPUT_DIR="$TMP_INPUT"
else
    INPUT_DIR="$(realpath "$INPUT")"
    TMP_INPUT=""
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
```

- [ ] **Step 2: Make executable**

```bash
chmod +x run.sh
```

- [ ] **Step 3: Commit**

```bash
git add run.sh
git commit -m "feat: add run.sh orchestrator"
```

---

## Task 7: Create `setup.sh`

**Files:**
- Create: `setup.sh`

Clones the three model repos, creates three conda environments, and optionally downloads weights via `huggingface-cli`.

- [ ] **Step 1: Write setup.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "=== Image Pipeline Setup ==="
echo ""

# ── Clone repos ───────────────────────────────────────────────────────────────
clone_if_missing() {
    local url="$1" dest="$2" name="$3"
    if [[ -d "$dest" ]]; then
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
else
    echo "huggingface-cli not found. Download weights manually from README.md"
fi

echo ""
echo "=== Setup complete ==="
echo "Next: edit config.sh with your model paths, then run:"
echo "  ./run.sh --input /path/to/images --detector yolo"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x setup.sh
```

- [ ] **Step 3: Commit**

```bash
git add setup.sh
git commit -m "feat: add setup.sh for repo cloning and env creation"
```

---

## Task 8: Create `README.md`

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Image Processing Pipeline

Four-step pipeline: FoundIR restoration → PNG compression → StableSR super-resolution → object detection.

## Requirements

- Linux (tested on HPC with SLURM; runs sequentially on a single machine)
- Conda
- NVIDIA GPU with CUDA
- ffmpeg (or Python with Pillow) for compression

## Setup

### 1. Clone this repo

```bash
git clone <this-repo> pipeline && cd pipeline
```

### 2. Edit `config.sh`

Open `config.sh` and set:
- `FOUNDIR_DIR` — where to clone FoundIR
- `STABLESR_DIR` — where to clone StableSR
- `STABLESR_CKPT` / `STABLESR_VQGAN_CKPT` — paths to StableSR checkpoint files
- `YOLO_WEIGHTS` / `RTDETR_WEIGHTS` — paths to detection weights
- Conda environment names (defaults are fine)

### 3. Run setup

```bash
./setup.sh
```

This clones FoundIR and StableSR, creates three conda environments, and optionally downloads weights.

### 4. Place FoundIR weights

FoundIR requires its weights at a specific path:

```
$FOUNDIR_DIR/premodel/model-2000.pt
```

Download from HuggingFace and rename/move accordingly:

```bash
mkdir -p $FOUNDIR_DIR/premodel
mv <downloaded_file>.pt $FOUNDIR_DIR/premodel/model-2000.pt
```

## Weights

All weights are hosted at: https://huggingface.co/vicpantoja2/aedes-egg-weights

| Model | URL |
|-------|-----|
| FoundIR | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/foundir |
| StableSR | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/stablesr |
| RT-DETR | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/rtdetr |
| YOLOv26 | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/yolov26 |

Download with huggingface-cli:

```bash
huggingface-cli download vicpantoja2/aedes-egg-weights --local-dir ./weights
```

## Usage

```bash
# Run with YOLO detection
./run.sh --input /path/to/images/ --detector yolo

# Run with RT-DETR detection
./run.sh --input /path/to/single_image.png --detector rtdetr
```

Input must be PNG. Can be a single file or a directory of PNG files.

## Output

Each run creates a timestamped directory under `output/`:

```
output/YYYYMMDD_HHMMSS/
├── 01_foundir/               # restored images (PNG)
├── 02_foundir_compressed/    # compressed images (JPG, quality 95)
├── 03_stablesr/              # super-resolved images
├── 04_results/
│   ├── annotated/            # images with bounding boxes
│   └── data/                 # JSON detection files
└── errors.log                # per-image failures
```

Detection JSON format:

```json
{
  "image": "example.jpg",
  "detections": [
    {
      "class_id": 0,
      "class_name": "aedes",
      "confidence": 0.95,
      "bbox_xyxy": [120.5, 80.2, 340.1, 210.8]
    }
  ]
}
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```

---

## Self-Review Notes

- FoundIR's weight path constraint (`premodel/model-2000.pt`) is called out in config.sh comments, README, and setup.sh output — three places, hard to miss.
- StableSR needs two checkpoint files; config.sh has two separate vars (`STABLESR_CKPT`, `STABLESR_VQGAN_CKPT`), and run_stablesr.sh uses both.
- All step scripts use `realpath` to resolve paths before `cd`-ing into model repos.
- The detection conda env is created fresh (no repo env file) with just `ultralytics` and `opencv-python-headless`, which covers both YOLO and RT-DETR.
- The `cleanup` trap in `run.sh` removes the temp dir on exit (both success and failure).
- Per-image error handling: each step script catches failures with `||`, logs them, and prints counts — the orchestrator prints the log at the end.

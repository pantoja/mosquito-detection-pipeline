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

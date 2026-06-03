# Image Processing Pipeline Design

**Date:** 2026-06-03
**Status:** Approved

## Overview

A bash-based image processing pipeline that accepts a single image or folder of images and passes them through four sequential steps: restoration (FoundIR), compression (PNG→JPG), super-resolution (StableSR), and object detection (YOLOv26 or RT-DETR). Targets remote HPC/server environments with GPU access.

---

## Directory Structure

```
pipeline/
├── run.sh               # orchestrator: entry point
├── setup.sh             # one-time setup: clone repos, create envs, download weights
├── steps/
│   ├── run_foundir.sh   # step 1: FoundIR restoration
│   ├── compress.sh      # step 2: PNG → JPG at quality 95
│   ├── run_stablesr.sh  # step 3: StableSR super-resolution
│   └── run_detection.sh # step 4: YOLOv26 or RT-DETR object detection
├── config.sh            # user-editable paths and environment names
└── README.md            # weight download links and setup instructions
```

---

## Output Layout

Each run creates a timestamped output directory:

```
output/YYYYMMDD_HHMMSS/
├── 01_foundir/               # restored images (PNG)
├── 02_foundir_compressed/    # compressed images (JPG, quality 95)
├── 03_stablesr/              # super-resolved images
├── 04_results/
│   ├── annotated/            # images with bounding boxes drawn
│   └── data/                 # JSON files with detections per image
└── errors.log                # per-image failures across all steps
```

---

## Invocation

```bash
./run.sh --input /path/to/image_or_folder --detector yolo
./run.sh --input /path/to/image_or_folder --detector rtdetr
```

Flags:
- `--input` (required): path to a single image or a directory of images
- `--detector` (required): `yolo` or `rtdetr`

---

## Orchestrator (`run.sh`)

1. Source `config.sh`
2. Parse and validate `--input` and `--detector` flags
3. Validate that input path exists; `--detector` must be `yolo` or `rtdetr`
4. Create timestamped output directory
5. Call each step script sequentially, passing previous output dir as next input dir
6. On per-image failure: log to `errors.log`, skip image, continue
7. Print end-of-run summary: images succeeded/failed per step

If `--input` is a single file, `run.sh` copies it into a temporary directory so all step scripts always receive a directory.

Step call convention:
```bash
steps/run_foundir.sh   <input_dir>              <output/YYYYMMDD_HHMMSS/01_foundir>
steps/compress.sh      <01_foundir>             <output/YYYYMMDD_HHMMSS/02_foundir_compressed>
steps/run_stablesr.sh  <02_foundir_compressed>  <output/YYYYMMDD_HHMMSS/03_stablesr>
steps/run_detection.sh <03_stablesr>            <output/YYYYMMDD_HHMMSS/04_results> <detector>
```

---

## Configuration (`config.sh`)

User edits this file once after cloning the pipeline repo:

```bash
FOUNDIR_DIR="/path/to/FoundIR"
FOUNDIR_WEIGHTS="/path/to/foundir.pth"

STABLESR_DIR="/path/to/StableSR"
STABLESR_WEIGHTS="/path/to/stablesr.pth"

YOLO_WEIGHTS="/path/to/yolov26.pt"
RTDETR_WEIGHTS="/path/to/rtdetr.pt"

CONDA_ENV_FOUNDIR="foundir"
CONDA_ENV_STABLESR="stablesr"
CONDA_ENV_DETECT="detect"
```

Each model gets its own conda environment to avoid dependency conflicts common in HPC setups.

---

## Step Scripts

### `steps/run_foundir.sh <input_dir> <output_dir>`
- Activates `$CONDA_ENV_FOUNDIR`
- Iterates over images in `<input_dir>`
- Calls FoundIR's inference script per image
- Writes restored PNGs to `<output_dir>`
- On failure: logs to `errors.log`, skips image

### `steps/compress.sh <input_dir> <output_dir>`
- No conda env required
- Uses `ffmpeg` (preferred) or Python Pillow as fallback
- Converts each PNG to JPG at quality 95
- Writes to `<output_dir>`
- On failure: logs to `errors.log`, skips image

### `steps/run_stablesr.sh <input_dir> <output_dir>`
- Activates `$CONDA_ENV_STABLESR`
- Runs StableSR inference on compressed JPGs
- Writes super-resolved images to `<output_dir>`
- On failure: logs to `errors.log`, skips image

### `steps/run_detection.sh <input_dir> <output_dir> <detector>`
- Activates `$CONDA_ENV_DETECT`
- Runs YOLOv26 or RT-DETR depending on `<detector>` argument
- For each image writes:
  - `<output_dir>/annotated/<image>` — image with bounding boxes
  - `<output_dir>/data/<image>.json` — detections: class, confidence, bounding box

---

## Setup (`setup.sh`)

1. Clone FoundIR, StableSR, and RT-DETR repos
2. Create three conda environments from each repo's requirements
3. Install `ultralytics` package in the detection env
4. Offer to download weights via `huggingface-cli` if available; otherwise print manual URLs

---

## Weights

All weights are hosted at: `https://huggingface.co/vicpantoja2/aedes-egg-weights`

| Model    | Path                                                                                      |
|----------|-------------------------------------------------------------------------------------------|
| FoundIR  | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/foundir                   |
| StableSR | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/stablesr                  |
| RT-DETR  | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/rtdetr                    |
| YOLOv26  | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/yolov26                   |

---

## Error Handling

- Failure scope: per-image, per-step
- On failure: log image path + step + error message to `errors.log`, skip, continue
- Pipeline does not abort on individual image failures
- End-of-run summary reports total counts per step

---

## Assumptions & Constraints

- Each model repo must be cloned and its path set in `config.sh` before running
- Conda must be available on the server
- GPU is expected; scripts do not include CPU fallback logic
- Input images must be PNG; the pipeline does not convert input formats
- `ffmpeg` or Python with Pillow must be available for the compression step

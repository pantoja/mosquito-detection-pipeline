#!/usr/bin/env python3
"""Run YOLO or RT-DETR detection on a directory of images."""

import argparse
import json
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

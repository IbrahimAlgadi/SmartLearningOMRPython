#!/usr/bin/env python3
"""
train_bubble_detector.py
────────────────────────
Train a YOLOv8 detection model on the synthetic OMR bubble dataset.

Usage
-----
  python train_bubble_detector.py                      # defaults
  python train_bubble_detector.py --data dataset_v2/data.yaml --epochs 150
  python train_bubble_detector.py --model yolov8s.pt   # larger model

Classes
-------
  0  bubble_empty
  1  bubble_filled
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    default="dataset_v2/data.yaml",
                    help="path to data.yaml")
    ap.add_argument("--model",   default="yolov8n.pt",
                    help="base YOLO model (default: yolov8n.pt)")
    ap.add_argument("--epochs",  type=int, default=100)
    ap.add_argument("--imgsz",   type=int, default=640,
                    help="training image size (default: 640)")
    ap.add_argument("--batch",   type=int, default=8,
                    help="batch size (-1 = auto, default: 8)")
    ap.add_argument("--device",  default="0",
                    help="device: 0=GPU0, cpu (default: 0)")
    ap.add_argument("--project", default="runs/bubble_detect")
    ap.add_argument("--name",    default="v2")
    ap.add_argument("--resume",  action="store_true",
                    help="resume the last interrupted run")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Run: python generate_omr_dataset_v2.py first."
        )

    model = YOLO(args.model)

    print(f"\n{'='*60}")
    print(f"  Model   : {args.model}")
    print(f"  Data    : {data_path.resolve()}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : {args.device}")
    print(f"{'='*60}\n")

    results = model.train(
        data    = str(data_path),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        project = args.project,
        name    = args.name,
        resume  = args.resume,

        # detection-tuning
        max_det   = 600,      # up to 500 bubbles on a 100Q sheet
        conf      = 0.25,
        iou       = 0.45,

        # augmentation (keep modest — we already distort in dataset gen)
        degrees   = 5.0,
        translate = 0.05,
        scale     = 0.10,
        fliplr    = 0.0,      # OMR sheets are not horizontally symmetric
        flipud    = 0.0,
        mosaic    = 0.5,
        mixup     = 0.0,

        # training stability
        patience  = 30,
        save_period = 10,
        workers   = 4,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights : {best}")
    print(f"\nTo test on an image:")
    print(f"  yolo detect predict model={best} source=<image.jpg> imgsz={args.imgsz}")


if __name__ == "__main__":
    main()

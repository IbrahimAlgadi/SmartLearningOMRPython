#!/usr/bin/env python3
"""
OMR YOLO Model Trainer
======================
Trains 3 YOLOv8n models sequentially:

  1. doc_model.pt     — detects paper boundary in a raw photo
  2. layout_model.pt  — detects answer_grid, qr_area, student_id_block
  3. bubble_model.pt  — detects bubble_filled and bubble_empty

Prerequisites:
  pip install ultralytics

Usage:
  # Train all 3 models (dataset must exist first):
  python train_models.py --data dataset

  # Train only one model:
  python train_models.py --data dataset --model bubbles

  # Quick smoke-test (5 epochs, small imgsz):
  python train_models.py --data dataset --epochs 5 --imgsz 416

After training, weights land in:
  models/doc_model.pt
  models/layout_model.pt
  models/bubble_model.pt
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

MODEL_SPECS = {
    "doc": {
        "name":     "doc_model",
        "out_file": "doc_model.pt",
        "desc":     "Paper boundary detector (1 class: paper)",
        "epochs":   80,
        "imgsz":    640,
        "batch":    16,
        # Larger objects (paper fills most of the image) → no mosaic needed
        "extra_args": {
            "mosaic": 0.3,
            "degrees": 5.0,
            "translate": 0.1,
            "scale": 0.3,
            "fliplr": 0.5,
        },
    },
    "layout": {
        "name":     "layout_model",
        "out_file": "layout_model.pt",
        "desc":     "Layout detector (3 classes: answer_grid, qr_area, student_id_block)",
        "epochs":   100,
        "imgsz":    640,
        "batch":    16,
        "extra_args": {
            "mosaic": 0.5,
            "degrees": 3.0,
            "translate": 0.05,
            "scale": 0.2,
            "fliplr": 0.3,
        },
    },
    "bubbles": {
        "name":     "bubble_model",
        "data_subdir": "bubbles",
        "out_file": "bubble_model.pt",
        "desc":     "Bubble detector (2 classes: bubble_filled, bubble_empty)",
        "epochs":   120,
        "imgsz":    1280,
        "batch":    8,   # reduced for 4GB VRAM at imgsz=1280
        # Small, dense objects — keep mosaic & close-mosaic high
        "extra_args": {
            "mosaic":       1.0,
            "close_mosaic": 15,
            "degrees":      2.0,
            "translate":    0.05,
            "scale":        0.15,
            "fliplr":       0.5,
            "workers":      0,    # avoids Windows shared-memory error at large imgsz
            "max_det":      500,  # each sheet has 400 bubbles; default 300 caps recall
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def train_model(spec: dict, data_root: pathlib.Path,
                models_dir: pathlib.Path,
                epochs: int = None, imgsz: int = None,
                batch: int = None, device: str = "cpu",
                resume: bool = False) -> pathlib.Path:
    """
    Train one YOLOv8n model and copy the best weights to models_dir.
    Returns the path to the saved .pt file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.\n"
              "        Run: pip install ultralytics")
        sys.exit(1)

    yaml_path = data_root / spec.get("data_subdir", spec["name"].replace("_model", "")) / "dataset.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] dataset.yaml not found: {yaml_path}")
        print("        Generate the dataset first:  python generate_omr_dataset.py")
        sys.exit(1)

    # Resolve effective hyper-parameters
    eff_epochs = epochs if epochs is not None else spec["epochs"]
    eff_imgsz  = imgsz  if imgsz  is not None else spec["imgsz"]
    eff_batch  = batch  if batch  is not None else spec["batch"]

    print(f"\n{'='*60}")
    print(f"  Training: {spec['name']}")
    print(f"  Desc:     {spec['desc']}")
    print(f"  Data:     {yaml_path}")
    print(f"  Epochs:   {eff_epochs}  imgsz={eff_imgsz}  batch={eff_batch}  device={device}")
    print(f"{'='*60}")

    run_name = f"{spec['name']}_run"
    model    = YOLO("yolov8n.pt")     # download once, reused for all 3

    train_kwargs = dict(
        data    = str(yaml_path),
        epochs  = eff_epochs,
        imgsz   = eff_imgsz,
        batch   = eff_batch,
        device  = device,
        name    = run_name,
        project = str(models_dir / "runs"),
        exist_ok= True,
        resume  = resume,
        verbose = True,
        # Validation on every epoch — lets you see when the model plateaus
        val     = True,
        patience= 30,   # early-stop if no improvement for 30 epochs
        # Confidence & IoU thresholds for validation mAP calculation
        conf    = 0.001,
        iou     = 0.6,
        **spec.get("extra_args", {}),
    )

    t0 = time.time()
    results = model.train(**train_kwargs)
    elapsed = int(time.time() - t0)

    # Copy best weights to models_dir/<out_file>
    best_pt = pathlib.Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = pathlib.Path(results.save_dir) / "weights" / "last.pt"

    dest = models_dir / spec["out_file"]
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, dest)

    print(f"\n[OK] {spec['name']} done in {elapsed // 60}m {elapsed % 60}s")
    print(f"     Best weights → {dest}")

    # Print quick validation summary
    try:
        metrics = results.results_dict
        map50   = metrics.get("metrics/mAP50(B)", 0)
        map5095 = metrics.get("metrics/mAP50-95(B)", 0)
        print(f"     mAP@50={map50:.3f}  mAP@50:95={map5095:.3f}")
    except Exception:
        pass

    return dest


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def validate_model(model_path: pathlib.Path, yaml_path: pathlib.Path,
                   device: str = "cpu"):
    """Run validation on an already-trained model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        return

    if not model_path.exists():
        print(f"[SKIP] {model_path} not found.")
        return

    model   = YOLO(str(model_path))
    metrics = model.val(data=str(yaml_path), device=device, verbose=False)
    print(f"  {model_path.name:25s}  "
          f"mAP@50={metrics.box.map50:.3f}  "
          f"mAP@50:95={metrics.box.map:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Train 3 YOLOv8n OMR models sequentially"
    )
    ap.add_argument("--data",    type=str, default="dataset",
                    help="Root folder produced by generate_omr_dataset.py")
    ap.add_argument("--models",  type=str, default="models",
                    help="Output folder for trained weights (default: models)")
    ap.add_argument("--model",   type=str, default="all",
                    choices=["all", "doc", "layout", "bubbles"],
                    help="Which model to train (default: all)")
    ap.add_argument("--epochs",  type=int, default=None,
                    help="Override epoch count for all models")
    ap.add_argument("--imgsz",   type=int, default=None,
                    help="Override image size (default per-model)")
    ap.add_argument("--batch",   type=int, default=None,
                    help="Override batch size")
    ap.add_argument("--device",  type=str, default="cpu",
                    help="Training device: cpu / 0 / 0,1 (default: cpu)")
    ap.add_argument("--resume",  action="store_true",
                    help="Resume interrupted training run")
    ap.add_argument("--val-only",action="store_true",
                    help="Skip training; only run validation on existing weights")
    args = ap.parse_args()

    data_dir   = pathlib.Path(args.data)
    models_dir = pathlib.Path(args.models)

    if not data_dir.exists():
        print(f"[ERROR] Dataset folder not found: {data_dir}")
        print("        Generate it first:  python generate_omr_dataset.py")
        sys.exit(1)

    to_train = (["doc", "layout", "bubbles"]
                if args.model == "all" else [args.model])

    if args.val_only:
        print("\n--- Validation results ---")
        for key in to_train:
            spec   = MODEL_SPECS[key]
            mpt    = models_dir / spec["out_file"]
            sub    = key.replace("_model", "")
            yaml_p = data_dir / sub / "dataset.yaml"
            validate_model(mpt, yaml_p, args.device)
        return

    trained: list[pathlib.Path] = []
    for key in to_train:
        spec = MODEL_SPECS[key]
        pt   = train_model(
            spec, data_dir, models_dir,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            resume=args.resume,
        )
        trained.append(pt)

    print(f"\n{'='*60}")
    print("  Training complete. Models saved:")
    for pt in trained:
        size_mb = pt.stat().st_size / 1024 / 1024
        print(f"    {pt}  ({size_mb:.1f} MB)")
    print(f"{'='*60}")
    print("\nNext step: update omr_8_states_detector.py with:")
    for pt in trained:
        print(f"  YOLO('{pt}')")


if __name__ == "__main__":
    main()

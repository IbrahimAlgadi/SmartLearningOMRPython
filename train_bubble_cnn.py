#!/usr/bin/env python3
"""
train_bubble_cnn.py
────────────────────
Trains a tiny CNN classifier for OMR bubble crops produced by
generate_omr_dataset_v3.py.

Architecture: 3 × (Conv → BN → ReLU → MaxPool) + FC head
Input  : 32×32 greyscale float32 normalised to [0,1]
Classes: 0=empty  1=filled  2=ambiguous
Output : bubble_classifier_v3.pt  (TorchScript, loaded via torch.jit.load)
         bubble_classifier_v3.onnx (for onnxruntime)

Usage:
  python train_bubble_cnn.py
  python train_bubble_cnn.py --data dataset_v3 --epochs 30 --batch 256
  python train_bubble_cnn.py --data dataset_v3 --epochs 50 --lr 3e-4
"""

from __future__ import annotations

import argparse
import pathlib
import random
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2

# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

class BubbleCNN(nn.Module):
    """
    ~45 KB model, ~90k params.
    Input: (N, 1, 32, 32)
    Output: (N, 3) raw logits
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # block 1: 32→16
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16×16

            # block 2: 16→8
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 8×8

            # block 3: 8→4
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 4×4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["empty", "filled", "ambiguous"]

class BubbleCropDataset(Dataset):
    """
    Loads from a compressed npz archive produced by generate_omr_dataset_v3.py.

    npz layout:
      images : uint8  (N, 32, 32)
      labels : int8   (N,)   0=empty  1=filled  2=ambiguous
    """

    def __init__(self, npz_path: pathlib.Path, augment: bool = False) -> None:
        self.augment = augment
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {npz_path}")
        data = np.load(str(npz_path))
        self.images = data["images"]   # (N, 32, 32) uint8
        self.labels = data["labels"].astype(np.int64)
        counts = np.bincount(self.labels, minlength=3)
        print(f"  {npz_path.name}: {len(self.images)} crops  "
              f"empty={counts[0]}  filled={counts[1]}  ambiguous={counts[2]}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img   = self.images[idx].copy()
        label = int(self.labels[idx])

        if self.augment:
            img = self._augment(img)

        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return tensor, label

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        # small rotation
        if random.random() < 0.5:
            angle = random.uniform(-12, 12)
            M = cv2.getRotationMatrix2D((16, 16), angle, 1.0)
            img = cv2.warpAffine(img, M, (32, 32), borderMode=cv2.BORDER_REFLECT)

        # tiny translation
        if random.random() < 0.4:
            tx, ty = random.randint(-2, 2), random.randint(-2, 2)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (32, 32), borderMode=cv2.BORDER_REFLECT)

        # brightness / contrast jitter
        alpha = random.uniform(0.75, 1.25)
        beta  = random.randint(-20, 20)
        img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 8, img.shape).astype(np.float32)
            img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img


# ─────────────────────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(dataset: BubbleCropDataset) -> torch.Tensor:
    counts = np.bincount(dataset.labels.astype(np.int64), minlength=3).astype(np.float64)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (3 * counts)
    print(f"  Class weights: empty={weights[0]:.3f}  "
          f"filled={weights[1]:.3f}  ambiguous={weights[2]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model: nn.Module, loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    n_batches = len(loader)
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(imgs)
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches:
            print(f"    batch {batch_idx+1}/{n_batches}  "
                  f"loss={total_loss/n:.4f}  acc={correct/n:.2%}",
                  flush=True)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader,
               criterion: nn.Module,
               device: torch.device) -> Tuple[float, float, np.ndarray]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    conf_matrix = np.zeros((3, 3), dtype=int)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        preds = logits.argmax(1)
        total_loss += loss.item() * len(imgs)
        correct += (preds == labels).sum().item()
        n += len(imgs)
        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            conf_matrix[t, p] += 1
    return total_loss / n, correct / n, conf_matrix


def export_models(model: nn.Module, out_dir: pathlib.Path) -> None:
    model.eval().cpu()
    dummy = torch.zeros(1, 1, 32, 32)

    # TorchScript (traced)
    pt_path = out_dir / "bubble_classifier_v3.pt"
    traced = torch.jit.trace(model, dummy)
    traced.save(str(pt_path))
    print(f"  Saved TorchScript: {pt_path}  ({pt_path.stat().st_size // 1024} KB)")

    # ONNX
    onnx_path = out_dir / "bubble_classifier_v3.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"  Saved ONNX:        {onnx_path}  ({onnx_path.stat().st_size // 1024} KB)")

    # Quick sanity check with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        out  = sess.run(None, {"input": dummy.numpy()})[0]
        assert out.shape == (1, 3), f"unexpected shape {out.shape}"
        print(f"  ONNX sanity check: OK  (logits={out[0]})")
    except Exception as e:
        print(f"  ONNX sanity check: FAILED — {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train OMR bubble CNN classifier (v3)"
    )
    ap.add_argument("--data",   default="dataset_v3",
                    help="Dataset root directory (default: dataset_v3)")
    ap.add_argument("--out",    default=".",
                    help="Where to save .pt and .onnx models (default: .)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch",  type=int, default=256)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = pathlib.Path(args.data)
    out_dir   = pathlib.Path(args.out)

    train_npz = data_root / "crops_train.npz"
    val_npz   = data_root / "crops_val.npz"

    if not train_npz.exists():
        print(f"ERROR: {train_npz} does not exist. Run generate_omr_dataset_v3.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = BubbleCropDataset(train_npz, augment=True)
    val_ds   = BubbleCropDataset(val_npz,   augment=False) if val_npz.exists() \
               else BubbleCropDataset(train_npz, augment=False)

    if len(train_ds) == 0:
        print("ERROR: Training set is empty!")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BubbleCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters  "
          f"(~{total_params * 4 // 1024} KB float32)")

    weights  = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_state   = None
    print(f"\nTraining for {args.epochs} epochs  "
          f"(batch={args.batch}, lr={args.lr})...\n")
    print(f"  {'Ep':>3}  {'TrainLoss':>9}  {'TrainAcc':>8}  "
          f"{'ValLoss':>8}  {'ValAcc':>7}  {'LR':>8}")
    print("  " + "-" * 58)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                      optimizer, device)
        vl_loss, vl_acc, conf = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        marker = " *" if vl_acc > best_val_acc else ""
        print(f"  {epoch:>3}  {tr_loss:>9.4f}  {tr_acc:>7.2%}  "
              f"{vl_loss:>8.4f}  {vl_acc:>6.2%}  {lr_now:>8.6f}{marker}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Restore best ──────────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    print(f"\nBest val accuracy: {best_val_acc:.2%}")

    # Final confusion matrix
    _, final_acc, conf = eval_epoch(model, val_loader, criterion, device)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"            {'empty':>8} {'filled':>8} {'ambig':>8}")
    for i, name in enumerate(CLASS_NAMES):
        row = conf[i]
        total = row.sum()
        pct   = row[i] / max(total, 1) * 100
        print(f"  {name:>10}  {row[0]:>8} {row[1]:>8} {row[2]:>8}  "
              f"({pct:.0f}% correct, n={total})")

    # ── Export ────────────────────────────────────────────────────────────────
    print("\nExporting models...")
    export_models(model, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()

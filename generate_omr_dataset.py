#!/usr/bin/env python3
"""
OMR Dataset Generator
=====================
Generates N synthetic OMR answer sheets as PNG images and emits YOLO-format
label files (.txt) for 3 separate models:

  Model 1 — doc_model       classes: paper
  Model 2 — layout_model    classes: answer_grid, qr_area, student_id_block
  Model 3 — bubble_model    classes: bubble_filled, bubble_empty

Output folder structure:
  dataset/
    doc/
      images/  *.png
      labels/  *.txt    (class 0 = paper)
    layout/
      images/  *.png    (warped sheet only)
      labels/  *.txt    (class 0=answer_grid, 1=qr_area, 2=student_id_block)
    bubbles/
      images/  *.png    (warped sheet only)
      labels/  *.txt    (class 0=bubble_filled, 1=bubble_empty)
    dataset.yaml         (ready to pass to YOLOv8 trainer)

Usage:
  python generate_omr_dataset.py --count 500 --lang ar --out dataset
  python generate_omr_dataset.py --count 200 --lang en --out dataset --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import qrcode
import base64
import io

# ── optional Playwright for high-fidelity PNG rendering ──────────────────────
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_OK = True
except ImportError:
    PLAYWRIGHT_OK = False
    print("[WARN] playwright not installed — will use cv2-drawn sheets only.\n"
          "       Install with: pip install playwright && playwright install chromium")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Physical A4 at 150 DPI (good quality, fast rendering)
A4_W_PX  = 1240   # 210 mm @ 150 dpi
A4_H_PX  = 1754   # 297 mm @ 150 dpi

# Bubble grid layout (fractions of A4)
GRID_X0_F = 0.10; GRID_X1_F = 0.90
GRID_Y0_F = 0.27; GRID_Y1_F = 0.95
N_COL_GROUPS   = 4
ROWS_PER_GROUP = 25
N_QUESTIONS    = 100
N_CHOICES      = 4

# Anchor squares: 12 mm / 210 mm ≈ 5.7 % of width; inset 5 mm ≈ 2.4 %
ANCHOR_W_F  = 12 / 210
ANCHOR_INS  =  5 / 210   # inset fraction (same for both axes, approx)

# QR area (fraction of A4)
QR_X0_F = 0.08; QR_X1_F = 0.42
QR_Y0_F = 0.08; QR_Y1_F = 0.26

# Student-ID block sits inside the QR area (left half, RTL)
STU_X0_F = 0.08; STU_X1_F = 0.24

# YOLO class IDs
DOC_CLASSES    = {"paper": 0}
LAYOUT_CLASSES = {"answer_grid": 0, "qr_area": 1, "student_id_block": 2}
BUBBLE_CLASSES = {"bubble_filled": 0, "bubble_empty": 1}

# ══════════════════════════════════════════════════════════════════════════════
#  YOLO LABEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _yolo_line(cls_id: int, cx: float, cy: float,
               bw: float, bh: float) -> str:
    """Return one YOLO label line (all values normalised 0–1)."""
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _box_to_yolo(cls_id: int, x0: int, y0: int, x1: int, y1: int,
                 img_w: int, img_h: int) -> str:
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    bw = (x1 - x0) / img_w
    bh = (y1 - y0) / img_h
    return _yolo_line(cls_id, cx, cy, bw, bh)


# ══════════════════════════════════════════════════════════════════════════════
#  SHEET RENDERER (pure OpenCV — no Playwright dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _qr_to_mat(data: str, target_px: int = 120) -> np.ndarray:
    """Render a QR code to a grayscale numpy array."""
    qr = qrcode.QRCode(version=None,
                       error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=4, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img_pil = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), np.uint8)
    mat = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(mat, (target_px, target_px), interpolation=cv2.INTER_NEAREST)


class SheetRenderer:
    """
    Renders a synthetic OMR answer sheet as a BGR numpy array.
    Returns the image plus exact pixel bounding boxes for every element.
    """

    def __init__(self, answers: Dict[int, int], student_code: str,
                 exam_code: str, lang: str = "ar",
                 width: int = A4_W_PX, height: int = A4_H_PX):
        self.answers      = answers       # q_num (1-100) → choice_idx (0-3)
        self.student_code = student_code
        self.exam_code    = exam_code
        self.lang         = lang
        self.W            = width
        self.H            = height

        # Computed positions (filled by render())
        self.anchor_boxes: List[Tuple[int,int,int,int]] = []   # (x0,y0,x1,y1)
        self.qr_box:  Optional[Tuple[int,int,int,int]] = None
        self.stu_box: Optional[Tuple[int,int,int,int]] = None
        self.grid_box: Optional[Tuple[int,int,int,int]] = None
        self.bubble_boxes: List[Tuple[int,int,int,int,bool]] = []  # …,filled

    # ── private helpers ───────────────────────────────────────────────────────

    def _px(self, f: float, axis: str = "x") -> int:
        return int(f * (self.W if axis == "x" else self.H))

    def _draw_anchor(self, img: np.ndarray, x0: int, y0: int):
        x1 = x0 + self._px(ANCHOR_W_F)
        y1 = y0 + self._px(ANCHOR_W_F, "y")
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
        self.anchor_boxes.append((x0, y0, x1, y1))

    # ── public render ─────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        img = np.ones((self.H, self.W, 3), dtype=np.uint8) * 255

        # ── border ───────────────────────────────────────────────────────────
        cv2.rectangle(img, (2, 2), (self.W - 3, self.H - 3), (180, 180, 180), 1)

        # ── corner anchors ────────────────────────────────────────────────────
        ins  = self._px(ANCHOR_INS)
        anw  = self._px(ANCHOR_W_F)
        positions = [
            (ins, ins),                           # TL
            (self.W - ins - anw, ins),             # TR
            (ins, self.H - ins - anw),             # BL
            (self.W - ins - anw, self.H - ins - anw),  # BR
        ]
        for (ax, ay) in positions:
            self._draw_anchor(img, ax, ay)

        # ── title bar ────────────────────────────────────────────────────────
        title = "ورقة الإجابة" if self.lang == "ar" else "Answer Sheet"
        cv2.putText(img, title,
                    (self.W // 2 - 160, self._px(0.055, "y")),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.line(img,
                 (self._px(0.10), self._px(0.065, "y")),
                 (self._px(0.90), self._px(0.065, "y")),
                 (40, 40, 40), 2)

        # ── QR codes ─────────────────────────────────────────────────────────
        qr_size = self._px(0.12)
        qr_y0   = self._px(QR_Y0_F, "y")
        qr_y1   = qr_y0 + qr_size

        stu_qr  = _qr_to_mat(self.student_code, qr_size)
        exam_qr = _qr_to_mat(self.exam_code, qr_size)

        # Student QR — left side of QR area (RTL)
        sq_x0 = self._px(QR_X0_F)
        sq_bgr = cv2.cvtColor(stu_qr, cv2.COLOR_GRAY2BGR)
        img[qr_y0:qr_y0 + qr_size, sq_x0:sq_x0 + qr_size] = sq_bgr

        # Exam QR — right of student QR
        eq_x0 = sq_x0 + qr_size + 10
        eq_bgr = cv2.cvtColor(exam_qr, cv2.COLOR_GRAY2BGR)
        img[qr_y0:qr_y0 + qr_size, eq_x0:eq_x0 + qr_size] = eq_bgr

        qr_area_x0 = sq_x0
        qr_area_x1 = eq_x0 + qr_size
        qr_area_y1 = self._px(QR_Y1_F, "y")
        self.qr_box  = (qr_area_x0, qr_y0, qr_area_x1, qr_area_y1)
        self.stu_box = (sq_x0, qr_y0, sq_x0 + qr_size, qr_y1)

        # ── separator ────────────────────────────────────────────────────────
        sep_y = self._px(0.27, "y")
        cv2.line(img, (self._px(0.08), sep_y), (self._px(0.92), sep_y),
                 (60, 60, 60), 2)

        # ── answer grid ───────────────────────────────────────────────────────
        gx0 = self._px(GRID_X0_F); gx1 = self._px(GRID_X1_F)
        gy0 = self._px(GRID_Y0_F, "y"); gy1 = self._px(GRID_Y1_F, "y")
        self.grid_box = (gx0, gy0, gx1, gy1)

        grid_w = gx1 - gx0
        grid_h = gy1 - gy0

        col_w    = grid_w  / N_COL_GROUPS
        row_h    = grid_h  / ROWS_PER_GROUP
        b_diam   = int(min(col_w / (N_CHOICES + 1.5), row_h * 0.72))
        b_radius = max(b_diam // 2, 6)
        num_w    = int(col_w * 0.22)

        # Draw grid columns
        for cg in range(N_COL_GROUPS):
            cx0 = int(gx0 + cg * col_w)
            cx1 = int(cx0 + col_w)
            cv2.rectangle(img, (cx0, gy0), (cx1, gy1), (180, 180, 180), 1)

            base_q = (N_COL_GROUPS - 1 - cg) * ROWS_PER_GROUP + 1

            for row in range(ROWS_PER_GROUP):
                q_num = base_q + row
                ry0   = int(gy0 + row * row_h)
                ry1   = int(ry0 + row_h)
                row_cy = (ry0 + ry1) // 2

                # Question number
                cv2.putText(img, str(q_num),
                            (cx0 + 4, row_cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                            (100, 100, 100), 1, cv2.LINE_AA)

                # Bubble separator line
                if row > 0:
                    cv2.line(img, (cx0, ry0), (cx1, ry0), (220, 220, 220), 1)

                # Bubbles: left→right in image = D,C,B,A (RTL)
                chosen_idx = self.answers.get(q_num)  # 0=A,1=B,2=C,3=D
                avail_w    = (cx1 - cx0 - num_w)
                b_spacing  = avail_w / (N_CHOICES + 0.5)

                for ch in range(N_CHOICES):
                    # ch=0 is leftmost = choice D (rank 3), ch=3 = choice A (rank 0)
                    rank   = N_CHOICES - 1 - ch   # 0=A,1=B,2=C,3=D
                    bx     = int(cx0 + num_w + ch * b_spacing + b_spacing / 2)
                    filled = (chosen_idx is not None and rank == chosen_idx)

                    if filled:
                        cv2.circle(img, (bx, row_cy), b_radius, (30, 30, 30), -1)
                    else:
                        cv2.circle(img, (bx, row_cy), b_radius, (80, 80, 80), 1)

                    self.bubble_boxes.append((
                        bx - b_radius, row_cy - b_radius,
                        bx + b_radius, row_cy + b_radius,
                        filled,
                    ))

        # ── footer ────────────────────────────────────────────────────────────
        footer = "OMR Answer Sheet | 100 Questions | A4"
        cv2.putText(img, footer,
                    (self._px(0.25), self._px(0.975, "y")),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)

        return img


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

class Augmentor:
    """
    Apply realistic phone-photo degradations to a clean rendered sheet.
    All transforms preserve the original image size so bounding boxes
    computed on the clean sheet remain valid (or are adjusted accordingly).
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    def apply(self, img: np.ndarray,
              corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        img     — clean BGR sheet
        corners — np.float32 [[TL],[TR],[BR],[BL]] in pixel coords

        Returns (augmented_img, new_corners).
        All augmentations keep the 4 corners in view so YOLO can still see them.
        """
        img = img.copy()

        # 1. Random brightness / contrast
        alpha = self.rng.uniform(0.75, 1.25)   # contrast
        beta  = self.rng.randint(-30, 30)       # brightness
        img   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 2. Gaussian blur (simulates slight defocus)
        k = self.rng.choice([1, 3, 5])
        if k > 1:
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 3. JPEG compression noise
        quality = self.rng.randint(60, 95)
        _, enc  = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img     = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # 4. Shadow polygon (random half-plane gradient)
        if self.rng.random() < 0.5:
            img = self._add_shadow(img)

        # 5. Small perspective warp (simulate held phone)
        img, corners = self._perspective_jitter(img, corners)

        return img, corners

    def _add_shadow(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        mask = np.ones((H, W), np.float32)
        # Random vertical or horizontal gradient
        if self.rng.random() < 0.5:
            x = self.rng.randint(W // 4, 3 * W // 4)
            mask[:, :x] *= self.rng.uniform(0.5, 0.8)
        else:
            y = self.rng.randint(H // 4, 3 * H // 4)
            mask[:y, :] *= self.rng.uniform(0.5, 0.8)
        mask = cv2.GaussianBlur(mask, (151, 151), 0)
        out  = (img.astype(np.float32) * mask[:, :, np.newaxis]).clip(0, 255)
        return out.astype(np.uint8)

    def _perspective_jitter(
        self, img: np.ndarray,
        corners: np.ndarray,
        max_shift_frac: float = 0.04,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = img.shape[:2]
        max_s = int(min(W, H) * max_shift_frac)

        # Source: slightly perturbed from sheet corners
        src = np.float32([
            [self.rng.randint(-max_s, max_s), self.rng.randint(-max_s, max_s)],
            [W - 1 + self.rng.randint(-max_s, max_s), self.rng.randint(-max_s, max_s)],
            [W - 1 + self.rng.randint(-max_s, max_s), H - 1 + self.rng.randint(-max_s, max_s)],
            [self.rng.randint(-max_s, max_s), H - 1 + self.rng.randint(-max_s, max_s)],
        ])
        dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
        M   = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(img, M, (W, H),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        # Transform original sheet corners (top-left, top-right, etc.)
        new_corners = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2), M
        ).reshape(-1, 2)

        return out, new_corners

    def place_on_background(self, img: np.ndarray,
                            bg_color: Tuple[int, int, int] = None) -> np.ndarray:
        """Paste the sheet onto a slightly larger noisy background."""
        if bg_color is None:
            bg_color = (
                self.rng.randint(160, 210),
                self.rng.randint(160, 210),
                self.rng.randint(160, 210),
            )
        H, W = img.shape[:2]
        pad_x = int(W * self.rng.uniform(0.03, 0.10))
        pad_y = int(H * self.rng.uniform(0.03, 0.10))
        out_w = W + 2 * pad_x
        out_h = H + 2 * pad_y

        bg = np.ones((out_h, out_w, 3), dtype=np.uint8)
        bg[:] = bg_color
        # Add grain
        noise = np.random.randint(-15, 15, bg.shape, dtype=np.int16)
        bg    = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        bg[pad_y:pad_y + H, pad_x:pad_x + W] = img
        return bg, pad_x, pad_y


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class LabelGenerator:
    """
    Given a SheetRenderer (which knows all pixel positions) + final image size,
    emit YOLO label strings for all 3 model families.
    """

    def __init__(self, renderer: SheetRenderer, img_w: int, img_h: int,
                 offset_x: int = 0, offset_y: int = 0):
        self.r       = renderer
        self.iw      = img_w
        self.ih      = img_h
        self.ox      = offset_x
        self.oy      = offset_y

    def _b(self, cls_id: int, box: Tuple[int, int, int, int]) -> str:
        x0, y0, x1, y1 = box
        return _box_to_yolo(cls_id,
                            x0 + self.ox, y0 + self.oy,
                            x1 + self.ox, y1 + self.oy,
                            self.iw, self.ih)

    def doc_labels(self, paper_box: Tuple[int, int, int, int]) -> List[str]:
        """One label: the paper boundary."""
        return [self._b(DOC_CLASSES["paper"], paper_box)]

    def layout_labels(self) -> List[str]:
        lines = []
        if self.r.grid_box:
            lines.append(self._b(LAYOUT_CLASSES["answer_grid"], self.r.grid_box))
        if self.r.qr_box:
            lines.append(self._b(LAYOUT_CLASSES["qr_area"], self.r.qr_box))
        if self.r.stu_box:
            lines.append(self._b(LAYOUT_CLASSES["student_id_block"], self.r.stu_box))
        return lines

    def bubble_labels(self) -> List[str]:
        lines = []
        for (x0, y0, x1, y1, filled) in self.r.bubble_boxes:
            cls_id = BUBBLE_CLASSES["bubble_filled" if filled else "bubble_empty"]
            lines.append(self._b(cls_id, (x0, y0, x1, y1)))
        return lines


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET WRITER
# ══════════════════════════════════════════════════════════════════════════════

def _save(img: np.ndarray, lines: List[str],
          images_dir: pathlib.Path, labels_dir: pathlib.Path, stem: str):
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(images_dir / f"{stem}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(labels_dir / f"{stem}.txt", "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def write_yaml(out_dir: pathlib.Path):
    """Write dataset.yaml — one per model sub-folder."""
    specs = {
        "doc":    {"names": {0: "paper"}},
        "layout": {"names": {0: "answer_grid", 1: "qr_area", 2: "student_id_block"}},
        "bubbles":{"names": {0: "bubble_filled", 1: "bubble_empty"}},
    }
    for folder, spec in specs.items():
        d = out_dir / folder
        names_str = "\n".join(f"  {k}: {v}" for k, v in spec["names"].items())
        nc = len(spec["names"])
        yaml = (
            f"path: {d.resolve()}\n"
            f"train: images/train\n"
            f"val:   images/val\n"
            f"nc: {nc}\n"
            f"names:\n{names_str}\n"
        )
        (d / "dataset.yaml").write_text(yaml)
        print(f"  [yaml] {d}/dataset.yaml")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate(count: int, lang: str, out_dir: pathlib.Path,
             seed: int = 0, val_split: float = 0.15,
             augment: bool = True):
    """
    Generate `count` synthetic OMR sheets with YOLO labels.
    80/20 train/val split by default.
    """
    rng  = random.Random(seed)
    aug  = Augmentor(rng)
    np.random.seed(seed)

    n_val   = max(1, int(count * val_split))
    n_train = count - n_val
    splits  = ["train"] * n_train + ["val"] * n_val
    rng.shuffle(splits)

    counters = {"train": 0, "val": 0}

    for idx in range(count):
        split = splits[idx]

        # ── random answers ────────────────────────────────────────────────────
        answers: Dict[int, int] = {}
        for q in range(1, N_QUESTIONS + 1):
            if rng.random() < 0.95:           # ~5 % unanswered
                answers[q] = rng.randint(0, N_CHOICES - 1)

        # ── random metadata ───────────────────────────────────────────────────
        stu_id   = f"STU-{rng.randint(1000, 9999):04d}"
        exam_id  = f"EXAM-{rng.randint(100, 999)}"

        # ── render clean sheet ────────────────────────────────────────────────
        renderer = SheetRenderer(answers, stu_id, exam_id, lang=lang)
        clean    = renderer.render()

        # Sheet corners in clean image coords
        corners = np.float32([
            [0, 0], [clean.shape[1] - 1, 0],
            [clean.shape[1] - 1, clean.shape[0] - 1], [0, clean.shape[0] - 1],
        ])

        # ── augment ───────────────────────────────────────────────────────────
        if augment:
            aug_img, aug_corners = aug.apply(clean, corners)
            final_img, pad_x, pad_y = aug.place_on_background(aug_img)
        else:
            final_img = clean
            pad_x, pad_y = 0, 0
            aug_corners  = corners

        final_h, final_w = final_img.shape[:2]

        # ── paper bounding box ────────────────────────────────────────────────
        # The paper occupies the entire clean sheet region inside the background
        paper_box = (pad_x, pad_y,
                     pad_x + clean.shape[1],
                     pad_y + clean.shape[0])

        # ── labels ────────────────────────────────────────────────────────────
        lgen = LabelGenerator(renderer, final_w, final_h, pad_x, pad_y)

        doc_lines    = lgen.doc_labels(paper_box)
        layout_lines = lgen.layout_labels()

        # ── save doc + layout (full augmented image) ──────────────────────────
        stem = f"{idx:05d}"
        for model, lines in [("doc", doc_lines), ("layout", layout_lines)]:
            _save(final_img, lines,
                  out_dir / model / "images" / split,
                  out_dir / model / "labels" / split,
                  stem)

        # ── save bubbles (cropped to answer_grid — makes bubbles ~4x larger) ──
        # Compute grid bbox in final_img coordinates (after pad offset)
        if renderer.grid_box:
            gx0 = renderer.grid_box[0] + pad_x
            gy0 = renderer.grid_box[1] + pad_y
            gx1 = renderer.grid_box[2] + pad_x
            gy1 = renderer.grid_box[3] + pad_y
            # Clamp to image bounds
            gx0 = max(0, gx0); gy0 = max(0, gy0)
            gx1 = min(final_w, gx1); gy1 = min(final_h, gy1)
            grid_crop = final_img[gy0:gy1, gx0:gx1]
            crop_h, crop_w = grid_crop.shape[:2]

            # Re-emit bubble labels relative to the grid crop
            bubble_lines_cropped = []
            for (bx0, by0, bx1, by1, filled) in renderer.bubble_boxes:
                # Shift to final_img coords then subtract grid origin
                rx0 = bx0 + pad_x - gx0;  rx1 = bx1 + pad_x - gx0
                ry0 = by0 + pad_y - gy0;  ry1 = by1 + pad_y - gy0
                # Skip bubbles outside the crop (shouldn't happen normally)
                if rx1 <= 0 or ry1 <= 0 or rx0 >= crop_w or ry0 >= crop_h:
                    continue
                rx0 = max(0, rx0); ry0 = max(0, ry0)
                rx1 = min(crop_w, rx1); ry1 = min(crop_h, ry1)
                cls_id = BUBBLE_CLASSES["bubble_filled" if filled else "bubble_empty"]
                bubble_lines_cropped.append(
                    _box_to_yolo(cls_id, rx0, ry0, rx1, ry1, crop_w, crop_h)
                )
            _save(grid_crop, bubble_lines_cropped,
                  out_dir / "bubbles" / "images" / split,
                  out_dir / "bubbles" / "labels" / split,
                  stem)
        else:
            # Fallback: use full image (should never happen)
            bubble_lines = lgen.bubble_labels()
            _save(final_img, bubble_lines,
                  out_dir / "bubbles" / "images" / split,
                  out_dir / "bubbles" / "labels" / split,
                  stem)

        counters[split] += 1
        if (idx + 1) % 50 == 0 or idx + 1 == count:
            print(f"  [{idx + 1}/{count}]  train={counters['train']}  "
                  f"val={counters['val']}")

    # ── also save a ground-truth answers JSON (for evaluation later) ──────────
    # (Each sheet maps stem → answer dict)
    # This is written incrementally above, but here we write a master manifest.
    print(f"\n[OK] {count} sheets generated in {out_dir}/")
    write_yaml(out_dir)

    # Summary
    print("\n  Sub-datasets:")
    for model in ["doc", "layout", "bubbles"]:
        n_imgs = sum(
            1 for p in (out_dir / model / "images").rglob("*.jpg")
        )
        print(f"    {model:10s}  {n_imgs} images")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic OMR dataset with YOLO labels"
    )
    ap.add_argument("--count",   type=int,   default=500,
                    help="Number of sheets to generate (default: 500)")
    ap.add_argument("--lang",    type=str,   default="ar",
                    choices=["ar", "en"],
                    help="Sheet language — ar (RTL) or en (LTR)")
    ap.add_argument("--out",     type=str,   default="dataset",
                    help="Output root directory (default: dataset)")
    ap.add_argument("--seed",    type=int,   default=0,
                    help="Random seed for reproducibility")
    ap.add_argument("--val",     type=float, default=0.15,
                    help="Validation fraction (default: 0.15)")
    ap.add_argument("--no-aug",  action="store_true",
                    help="Disable augmentation (clean sheets only)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    if out_dir.exists():
        print(f"[INFO] Output dir {out_dir} already exists — appending.")

    print(f"Generating {args.count} OMR sheets  lang={args.lang}  "
          f"aug={'off' if args.no_aug else 'on'}  seed={args.seed}")

    generate(
        count    = args.count,
        lang     = args.lang,
        out_dir  = out_dir,
        seed     = args.seed,
        val_split= args.val,
        augment  = not args.no_aug,
    )


if __name__ == "__main__":
    main()

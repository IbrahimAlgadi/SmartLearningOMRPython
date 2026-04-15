#!/usr/bin/env python3
"""
generate_omr_dataset_v3.py
──────────────────────────
Synthetic OMR dataset generator — v3 (template-driven).

Sheet geometry is owned entirely by TemplateSpec from omr_templates.py.
The three production families (Q20/Q50/Q100 × 5-choices) match
generate_sheet_v3.py's HTML layout; additional 2/3/4-choice variants
add training diversity.

Per sample, three artefacts are written:
  1. Full distorted sheet  →  dataset_v3/sheets/{train|val}/
  2. Bubble crops (32×32)  →  dataset_v3/crops/{train|val}/{empty|filled|ambiguous}/
  3. QA preview            →  dataset_v3/qa/
       blue outline   = empty
       green outline  = filled
       orange outline = ambiguous

Metadata is written to:
  dataset_v3/metadata/metadata.jsonl   (one JSON line per sheet)
  dataset_v3/data_crops.yaml           (ready for a CNN trainer)

Usage:
  python generate_omr_dataset_v3.py
  python generate_omr_dataset_v3.py --out my_data --samples 3000 --dpi 150
  python generate_omr_dataset_v3.py --templates Q100_5ch Q50_5ch Q20_5ch
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from omr_templates import (
    A4_W_MM, A4_H_MM,
    ANCHOR_CORNER_MM, ANCHOR_CORNER_INSET_MM,
    ANCHOR_MID_MM,
    COL_BAR_H_MM,
    GRID_X0_MM, GRID_X1_MM, GRID_Y0_MM, GRID_Y1_MM,
    WARP_GW,
    BZONE_LEFT_FRAC, BZONE_RIGHT_FRAC, BZONE_RIGHT_BASE_W,
    REGISTRY,
    TemplateSpec,
    get_template,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Render constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DPI  = 150
CROP_SIZE    = 32      # CNN input size
CROP_MARGIN  = 4       # extra pixels beyond bubble radius in crop

MARK_STYLES = [
    "pen_solid", "pen_partial",
    "pencil_dark",
    "pencil_light",   # → "ambiguous" class
    "marker", "x_mark", "check", "sloppy_fill",
]
DISTORTION_LEVELS = ["clean", "light", "medium", "heavy"]

# Bubble label colours for QA preview (BGR)
QA_COLOR = {
    "empty":     (210,  80,   0),   # blue
    "filled":    (  0, 200,   0),   # green
    "ambiguous": (  0, 165, 255),   # orange
}


# ─────────────────────────────────────────────────────────────────────────────
#  Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Bubble:
    cx:       int         # centre x (PIL pixels)
    cy:       int         # centre y (PIL pixels)
    r:        int         # radius (PIL pixels)
    question: int         # 1-based
    rank:     int         # choice rank (0 = first label)
    label:    str         # choice label, e.g. "A" / "أ"
    status:   str = "empty"   # "empty" | "filled" | "ambiguous"


# ─────────────────────────────────────────────────────────────────────────────
#  Font helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
#  Sheet renderer — PIL-based, geometry derived from TemplateSpec + DPI
# ─────────────────────────────────────────────────────────────────────────────

class SheetRendererV3:
    """
    Renders a blank A4 OMR sheet in PIL that is geometrically consistent
    with generate_sheet_v3.py's HTML layout.

    The rendered sheet has:
      • 4 corner anchors (12 mm black squares at 4 mm inset)
      • 2 mid-edge anchors (7 mm squares, flush, vertically centred)
      • Column black bars above and below the question-row grid
      • Choice-letter header rows
      • Question rows with bubble circles and question numbers

    Returns (bgr_ndarray, list[Bubble]) with exact PIL pixel positions.
    """

    def __init__(self, template: TemplateSpec, dpi: int = DEFAULT_DPI):
        self.t   = template
        self.dpi = dpi
        self.ppm = dpi / 25.4   # pixels per mm

        # Scale CSS px (96dpi) → physical pixels at this DPI
        self.css  = dpi / 96.0

        # Physical image dimensions
        self.W = round(A4_W_MM * self.ppm)
        self.H = round(A4_H_MM * self.ppm)

        # Grid area (PIL pixels)
        self.gx0 = round(GRID_X0_MM * self.ppm)
        self.gx1 = round(GRID_X1_MM * self.ppm)
        self.gy0 = round(GRID_Y0_MM * self.ppm)
        self.gy1 = round(GRID_Y1_MM * self.ppm)
        self.gw  = self.gx1 - self.gx0
        self.gh  = self.gy1 - self.gy0

        # Column geometry in PIL pixels
        self.col_w   = self.gw // template.n_cols
        self.row_h   = self.gh / template.visual_rows

        # Bubble sizing in PIL pixels
        bub_mm       = template.bubble_css_px * (25.4 / 96.0)
        self.bub_r   = max(5, round(bub_mm * self.ppm / 2))

        # Fonts
        _sz = max(8, round(template.qn_css_px  * self.css))
        _ch = max(7, round(template.ch_css_px  * self.css))
        _ti = max(14, round(22 * self.css))
        self._f_qn   = _load_font(_sz, bold=True)
        self._f_ch   = _load_font(_ch, bold=True)
        self._f_hdr  = _load_font(max(8, round(9 * self.css)))
        self._f_title= _load_font(_ti, bold=True)

    # ── public ────────────────────────────────────────────────────────────────

    def render(self) -> Tuple[np.ndarray, List[Bubble]]:
        img  = Image.new("RGB", (self.W, self.H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        self._draw_anchors(draw)
        self._draw_header(draw)
        bubbles = self._draw_grid(draw)
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return bgr, bubbles

    # ── anchors ───────────────────────────────────────────────────────────────

    def _draw_anchors(self, draw: ImageDraw.Draw) -> None:
        s  = round(ANCHOR_CORNER_MM  * self.ppm)
        ins= round(ANCHOR_CORNER_INSET_MM * self.ppm)
        for ax, ay in [
            (ins,          ins),
            (self.W - ins - s, ins),
            (ins,          self.H - ins - s),
            (self.W - ins - s, self.H - ins - s),
        ]:
            draw.rectangle([ax, ay, ax + s, ay + s], fill=(0, 0, 0))

        # Mid-edge anchors
        m  = round(ANCHOR_MID_MM * self.ppm)
        my = self.H // 2
        for ax in (0, self.W - m):
            draw.rectangle([ax, my - m // 2, ax + m, my + m // 2],
                           fill=(0, 0, 0))

    # ── header (title + meta + legend) ───────────────────────────────────────

    def _draw_header(self, draw: ImageDraw.Draw) -> None:
        t = self.t
        # Title
        title   = "ورقة الإجابة" if t.rtl else "Answer Sheet"
        tb      = draw.textbbox((0, 0), title, font=self._f_title)
        tw, th  = tb[2] - tb[0], tb[3] - tb[1]
        ty      = round(CONTENT_TOP_MM_PIL(self.ppm)) + 4
        draw.text(((self.W - tw) // 2, ty), title,
                  fill=(0, 0, 0), font=self._f_title)

        # Horizontal rule
        rule_y = ty + th + 6
        draw.line([(self.gx0, rule_y), (self.gx1, rule_y)], fill=(0, 0, 0), width=2)

        # Minimal meta row
        meta_y = rule_y + 8
        meta   = [("STU-20240001", "رقم الطالب" if t.rtl else "Student Code"),
                  ("EXAM-2024-01",  "رمز الامتحان" if t.rtl else "Exam Code")]
        cw     = (self.gx1 - self.gx0) // len(meta)
        for i, (val, lbl) in enumerate(meta):
            mx = self.gx0 + i * cw + 4
            draw.text((mx, meta_y),      lbl, fill=(120, 120, 120), font=self._f_hdr)
            draw.text((mx, meta_y + 14), val, fill=(0, 0, 0),       font=self._f_hdr)

        # Legend bar (simplified)
        leg_y = self.gy0 - round(9 * self.ppm)
        lh    = round(6 * self.ppm)
        draw.rectangle([self.gx0, leg_y, self.gx1, leg_y + lh],
                       fill=(245, 245, 245), outline=(180, 180, 180), width=1)
        ftr = (f"{'ورقة إجابة OMR' if t.rtl else 'OMR Answer Sheet'} | "
               f"{t.n_questions} {'سؤال' if t.rtl else 'Q'} | "
               f"{t.n_choices} {'خيارات' if t.rtl else 'choices'} | A4")
        draw.text((self.gx0 + 6, leg_y + 4), ftr,
                  fill=(80, 80, 80), font=self._f_hdr)

        # Footer
        foot_y = self.gy1 + round(COL_BAR_H_MM * self.ppm) + 4
        draw.text(((self.W) // 2, foot_y), ftr,
                  fill=(160, 160, 160), font=self._f_hdr, anchor="mt")

    # ── question grid ─────────────────────────────────────────────────────────

    def _draw_grid(self, draw: ImageDraw.Draw) -> List[Bubble]:
        t       = self.t
        bubbles: List[Bubble] = []
        bar_h   = round(COL_BAR_H_MM * self.ppm)

        for col_idx in range(t.n_cols):
            cx0 = self.gx0 + col_idx * self.col_w
            cx1 = cx0 + self.col_w

            # Bubble zone x bounds (same formula as v3 detector)
            span       = float(self.col_w)
            bz_x0      = cx0 + span * BZONE_LEFT_FRAC
            right_base = BZONE_RIGHT_BASE_W / (WARP_GW / self.gw)  # convert to PIL
            bz_x1      = cx1 - (right_base + span * BZONE_RIGHT_FRAC)

            # Top and bottom bars (above / below the grid area)
            bar_top_y0 = self.gy0 - bar_h
            bar_bot_y0 = self.gy1
            for by in (bar_top_y0, bar_bot_y0):
                draw.rectangle([round(bz_x0), by,
                                round(bz_x1), by + bar_h],
                               fill=(0, 0, 0))

            # Column separator
            if col_idx < t.n_cols - 1:
                draw.line([(cx1, self.gy0), (cx1, self.gy1)],
                          fill=(200, 200, 200), width=1)

            # Bubble x positions (linspace from bz_x0 to bz_x1, RTL reversed)
            if t.n_choices == 1:
                xs_ltr = [round((bz_x0 + bz_x1) / 2)]
            else:
                step   = (bz_x1 - bz_x0) / (t.n_choices - 1)
                xs_ltr = [round(bz_x0 + i * step) for i in range(t.n_choices)]

            xs = list(reversed(xs_ltr)) if t.rtl else xs_ltr

            # Rows
            for row_idx in range(t.rows_per_col):
                q_num = t.question_for(col_idx, row_idx)
                if q_num > t.n_questions:
                    break

                group  = row_idx // t.header_every
                vis_row = row_idx + group + 1   # +1 per group for header row

                # Insert choice-label header before each new group
                if row_idx % t.header_every == 0:
                    hdr_vis = group * (t.header_every + 1)
                    hy      = round(self.gy0 + (hdr_vis + 0.5) * self.row_h)
                    hdr_h   = round(self.row_h)
                    draw.rectangle([cx0, hy - hdr_h // 2,
                                   cx1 - 1, hy + hdr_h // 2],
                                  fill=(238, 238, 248))
                    for rank, lbl in enumerate(t.choice_labels):
                        bx = xs[rank]
                        draw.text((bx, hy), lbl,
                                  fill=(60, 60, 180), font=self._f_ch, anchor="mm")

                # Question row
                qy = round(self.gy0 + (vis_row + 0.5) * self.row_h)

                # Question number (right side for RTL)
                qn_x = cx1 - round(2 * self.ppm)
                draw.text((qn_x, qy), str(q_num),
                          fill=(0, 0, 0), font=self._f_qn, anchor="rm")

                # Bubble circles
                r = self.bub_r
                for rank, bx in enumerate(xs):
                    draw.ellipse([bx - r, qy - r, bx + r, qy + r],
                                 outline=(0, 0, 0), width=2)
                    bubbles.append(Bubble(
                        cx=bx, cy=qy, r=r,
                        question=q_num,
                        rank=rank,
                        label=t.choice_labels[rank],
                    ))

        return bubbles


def CONTENT_TOP_MM_PIL(ppm: float) -> float:
    return (ANCHOR_CORNER_INSET_MM + ANCHOR_CORNER_MM + 2) * ppm


# ─────────────────────────────────────────────────────────────────────────────
#  Answer simulator (adapted from v2)
# ─────────────────────────────────────────────────────────────────────────────

class AnswerSimulator:
    """Mark bubbles with realistic student styles; returns updated image."""

    @staticmethod
    def random_answers(n_q: int, n_ch: int,
                       blank_prob:  float = 0.05,
                       double_prob: float = 0.08,
                       triple_prob: float = 0.03) -> Dict[int, List[int]]:
        answers: Dict[int, List[int]] = {}
        for q in range(1, n_q + 1):
            p = random.random()
            if p < blank_prob:
                answers[q] = []
            elif p < blank_prob + triple_prob:
                answers[q] = random.sample(range(n_ch), min(3, n_ch))
            elif p < blank_prob + triple_prob + double_prob:
                answers[q] = random.sample(range(n_ch), min(2, n_ch))
            else:
                answers[q] = [random.randint(0, n_ch - 1)]
        return answers

    def simulate(self, img: np.ndarray,
                 bubbles: List[Bubble],
                 answers: Dict[int, List[int]],
                 style: str) -> np.ndarray:
        out = img.copy()
        for b in bubbles:
            if b.question in answers and b.rank in answers[b.question]:
                b.status = "ambiguous" if style == "pencil_light" else "filled"
                self._mark(out, b, style)
        return out

    def _mark(self, img: np.ndarray, b: Bubble, style: str) -> None:
        cx, cy, r = b.cx, b.cy, b.r

        if style == "pen_solid":
            cv2.circle(img, (cx, cy), r - 2, (15, 15, 15), -1)

        elif style == "pen_partial":
            ox, oy = random.randint(-2, 2), random.randint(-2, 2)
            cv2.circle(img, (cx + ox, cy + oy), r - 3, (25, 25, 25), -1)

        elif style == "pencil_light":
            gray = random.randint(155, 200)
            cv2.circle(img, (cx, cy), r - 2, (gray, gray, gray), -1)
            self._pencil_texture(img, cx, cy, r, lo=170, hi=230)

        elif style == "pencil_dark":
            gray = random.randint(55, 105)
            cv2.circle(img, (cx, cy), r - 2, (gray, gray, gray), -1)
            self._pencil_texture(img, cx, cy, r, lo=90, hi=150)

        elif style == "marker":
            cv2.circle(img, (cx, cy), r - 1, (8, 8, 25), -1)
            for _ in range(6):
                cv2.circle(img, (cx + random.randint(-2, 2),
                                cy + random.randint(-2, 2)), r, (30, 30, 55), 1)

        elif style == "x_mark":
            d     = r - 2
            thick = max(1, r // 5)
            cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), (10, 10, 10), thick)
            cv2.line(img, (cx + d, cy - d), (cx - d, cy + d), (10, 10, 10), thick)

        elif style == "check":
            d    = r - 2
            pts  = np.array([[cx - d, cy],
                             [cx - d // 3, cy + d - 1],
                             [cx + d, cy - d // 2]], np.int32)
            thick = max(1, r // 5)
            cv2.polylines(img, [pts], False, (10, 10, 10), thick)

        elif style == "sloppy_fill":
            mask = np.zeros(img.shape[:2], np.uint8)
            for _ in range(6):
                ox = random.randint(-r // 3, r // 3)
                oy = random.randint(-r // 3, r // 3)
                ax = r - 1 + random.randint(-2, 2)
                ay = r - 1 + random.randint(-2, 2)
                cv2.ellipse(mask, (cx + ox, cy + oy),
                            (max(ax, 3), max(ay, 3)),
                            random.randint(0, 360), 0, 360, 255, -1)
            img[mask > 0] = [20, 20, 20]

    @staticmethod
    def _pencil_texture(img, cx, cy, r, lo=150, hi=220):
        for _ in range(18):
            px = cx + random.randint(-(r - 2), r - 2)
            py = cy + random.randint(-(r - 2), r - 2)
            if (px - cx) ** 2 + (py - cy) ** 2 < (r - 2) ** 2:
                v = random.randint(lo, hi)
                cv2.circle(img, (px, py), 1, (v, v, v), -1)


# ─────────────────────────────────────────────────────────────────────────────
#  Distortion engine (from v2, adapted)
# ─────────────────────────────────────────────────────────────────────────────

class DistortionEngine:
    """
    Apply geometric / photometric distortions to simulate real scans.
    Bubble cx/cy are updated to remain accurate in the distorted image.
    """

    @staticmethod
    def apply(img: np.ndarray,
              bubbles: List[Bubble],
              level: str) -> Tuple[np.ndarray, List[Bubble]]:
        pts = np.float32([[b.cx, b.cy] for b in bubbles])
        out = img.copy()

        if level == "clean":
            return out, bubbles

        # Rotation
        max_angle = {"light": 1.5, "medium": 3.5, "heavy": 7.0}[level]
        angle = random.uniform(-max_angle, max_angle)
        out, pts = DistortionEngine._rotate(out, pts, angle)

        # Perspective
        if level in ("medium", "heavy"):
            strength = {"medium": 0.008, "heavy": 0.020}[level]
            out, pts = DistortionEngine._perspective(out, pts, strength)

        # Brightness / contrast
        b_range = {"light": (0.82, 1.18),
                   "medium": (0.60, 1.40),
                   "heavy":  (0.38, 1.65)}[level]
        out = DistortionEngine._brightness(out, b_range)

        # Gaussian noise
        noise_std = {"light": 6, "medium": 16, "heavy": 32}[level]
        out = DistortionEngine._noise(out, noise_std)

        # Blur / JPEG
        if level == "heavy" and random.random() < 0.55:
            out = DistortionEngine._motion_blur(out)
        if level == "heavy":
            out = DistortionEngine._jpeg(out, random.randint(38, 68))
        elif level == "medium" and random.random() < 0.35:
            out = DistortionEngine._jpeg(out, random.randint(58, 82))

        new_bubbles = [
            Bubble(cx=int(round(nx)), cy=int(round(ny)),
                   r=b.r, question=b.question,
                   rank=b.rank, label=b.label, status=b.status)
            for b, (nx, ny) in zip(bubbles, pts.tolist())
        ]
        return out, new_bubbles

    @staticmethod
    def _rotate(img, pts, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        out = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(240, 240, 240))
        if len(pts):
            ones  = np.ones((len(pts), 1), dtype=np.float32)
            pts   = (M @ np.hstack([pts, ones]).T).T
        return out, pts

    @staticmethod
    def _perspective(img, pts, strength):
        h, w = img.shape[:2]
        d    = int(min(h, w) * strength)
        src  = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst  = src + np.random.randint(-d, d + 1, src.shape).astype(np.float32)
        M    = cv2.getPerspectiveTransform(src, dst)
        out  = cv2.warpPerspective(img, M, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(240, 240, 240))
        if len(pts):
            pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        return out, pts

    @staticmethod
    def _brightness(img, rng):
        alpha = random.uniform(*rng)
        beta  = random.randint(-25, 25)
        return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    @staticmethod
    def _noise(img, std):
        n = np.random.normal(0, std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    @staticmethod
    def _motion_blur(img):
        sz  = random.choice([5, 7, 9])
        ang = random.uniform(0, 180)
        k   = np.zeros((sz, sz), np.float32)
        k[sz // 2, :] = 1.0 / sz
        M   = cv2.getRotationMatrix2D((sz // 2, sz // 2), ang, 1)
        k   = cv2.warpAffine(k, M, (sz, sz))
        k  /= k.sum() + 1e-8
        return cv2.filter2D(img, -1, k)

    @staticmethod
    def _jpeg(img, quality):
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# ─────────────────────────────────────────────────────────────────────────────
#  QA preview renderer
# ─────────────────────────────────────────────────────────────────────────────

def make_qa_preview(img: np.ndarray, bubbles: List[Bubble],
                    template_id: str, style: str, level: str) -> np.ndarray:
    """
    Annotate the sheet image with per-bubble label outlines:
      blue   = empty
      green  = filled
      orange = ambiguous
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    for b in bubbles:
        color = QA_COLOR[b.status]
        r_draw = b.r + 3
        cx, cy = b.cx, b.cy
        # Clamp to image bounds
        if 0 < cx < w and 0 < cy < h:
            cv2.circle(vis, (cx, cy), r_draw, color, 2)

    # Info bar
    bar_h = max(28, round(h * 0.018))
    cv2.rectangle(vis, (0, 0), (w, bar_h), (30, 30, 30), -1)
    n_filled = sum(1 for b in bubbles if b.status == "filled")
    n_ambig  = sum(1 for b in bubbles if b.status == "ambiguous")
    info = (f"{template_id} | {style} | {level} | "
            f"filled={n_filled} ambig={n_ambig} empty={len(bubbles)-n_filled-n_ambig}")
    cv2.putText(vis, info, (6, bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
#  Crop extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_crop(gray: np.ndarray, cx: int, cy: int, r: int,
                 size: int = CROP_SIZE, margin: int = CROP_MARGIN) -> np.ndarray:
    """Extract a square grayscale crop centred at (cx,cy), resized to size×size."""
    H, W = gray.shape
    pad  = r + margin
    x0   = max(0, cx - pad);  x1 = min(W, cx + pad)
    y0   = max(0, cy - pad);  y1 = min(H, cy + pad)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return np.full((size, size), 128, dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset generator
# ─────────────────────────────────────────────────────────────────────────────

class DatasetGeneratorV3:

    def __init__(self, out_dir: str,
                 n_total: int = 2400,
                 dpi: int = DEFAULT_DPI,
                 seed: int = 42,
                 template_ids: Optional[List[str]] = None):
        self.out     = pathlib.Path(out_dir)
        self.n_total = n_total
        self.dpi     = dpi
        random.seed(seed)
        np.random.seed(seed)

        self.templates = (
            [REGISTRY[tid] for tid in template_ids]
            if template_ids
            else list(REGISTRY.values())
        )

        # Create directory tree
        for split in ("train", "val"):
            (self.out / "sheets" / split).mkdir(parents=True, exist_ok=True)
        (self.out / "qa"      ).mkdir(exist_ok=True)
        (self.out / "metadata").mkdir(exist_ok=True)

    # ── main entry ────────────────────────────────────────────────────────────

    def generate(self) -> None:
        n_tpl    = len(self.templates)
        n_per    = max(1, self.n_total // n_tpl)
        meta_fh  = open(self.out / "metadata" / "metadata.jsonl", "w",
                        encoding="utf-8")

        total = crop_total = 0
        print(f"  {n_tpl} templates x {n_per} samples ~{n_tpl * n_per} sheets\n")

        # Accumulate ALL crops in RAM; flush once at the very end.
        # Avoids the O(n²) cost of repeated npz reload-append cycles.
        all_crops: dict = {
            "train": {"images": [], "labels": []},
            "val":   {"images": [], "labels": []},
        }
        CLASS_TO_IDX = {"empty": 0, "filled": 1, "ambiguous": 2}
        QA_PER_TPL   = 5   # only save QA preview for first N sheets per template

        for tpl in self.templates:
            print(f"  [{tpl.template_id}]  rendering base sheet ...",
                  end="", flush=True)
            renderer  = SheetRendererV3(tpl, self.dpi)
            base_img, base_bubbles = renderer.render()
            print(f" ({len(base_bubbles)} bubbles)")
            sim = AnswerSimulator()

            for i in range(n_per):
                style   = random.choice(MARK_STYLES)
                level   = random.choice(DISTORTION_LEVELS)
                answers = AnswerSimulator.random_answers(tpl.n_questions,
                                                         tpl.n_choices)

                # Reset fill flags on the canonical bubble list
                for b in base_bubbles:
                    b.status = "empty"

                marked = sim.simulate(base_img, list(base_bubbles), answers, style)

                # Distortion (returns new bubble list with updated coords)
                distorted, moved = DistortionEngine.apply(
                    marked, list(base_bubbles), level
                )

                split = "val" if total % 5 == 0 else "train"
                stem  = f"omr_{tpl.template_id}_{total:06d}"

                # ── 1. Full sheet (first 3 per template for inspection) ───────
                if i < 3:
                    sheet_rel = f"sheets/{split}/{stem}.jpg"
                    cv2.imwrite(str(self.out / sheet_rel), distorted,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                else:
                    sheet_rel = None

                # ── 2. Bubble crops → in-memory accumulation ──────────────────
                gray = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)
                # Apply same background normalization as the v3 detector so
                # training crops match exactly what the CNN receives at inference.
                bg_blur  = cv2.medianBlur(gray, 71)
                bg_blur  = np.maximum(bg_blur, 1)
                gray     = np.clip(
                    gray.astype(np.float32) / bg_blur.astype(np.float32) * 255,
                    0, 255,
                ).astype(np.uint8)
                bubble_meta: list = []
                for b in moved:
                    crop = extract_crop(gray, b.cx, b.cy, b.r)
                    lbl  = CLASS_TO_IDX[b.status]
                    all_crops[split]["images"].append(crop)
                    all_crops[split]["labels"].append(lbl)
                    crop_total += 1
                    bubble_meta.append({
                        "question":  b.question,
                        "rank":      b.rank,
                        "label":     b.label,
                        "status":    b.status,
                        "cx_pil":    b.cx,
                        "cy_pil":    b.cy,
                        "r_pil":     b.r,
                    })

                # ── 3. QA preview (first QA_PER_TPL per template only) ───────
                if i < QA_PER_TPL:
                    qa_img = make_qa_preview(distorted, moved,
                                             tpl.template_id, style, level)
                    qa_rel = f"qa/{stem}_qa.jpg"
                    cv2.imwrite(str(self.out / qa_rel), qa_img,
                                [cv2.IMWRITE_JPEG_QUALITY, 88])
                else:
                    qa_rel = None

                # ── 4. Metadata ───────────────────────────────────────────────
                record = {
                    "stem":        stem,
                    "template_id": tpl.template_id,
                    "n_questions": tpl.n_questions,
                    "n_choices":   tpl.n_choices,
                    "mark_style":  style,
                    "distortion":  level,
                    "split":       split,
                    "sheet_path":  sheet_rel,
                    "qa_path":     qa_rel,
                    "answers":     {str(q): answers.get(q, [])
                                    for q in range(1, tpl.n_questions + 1)},
                    "bubbles":     bubble_meta,
                }
                meta_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                total += 1

        meta_fh.close()

        # ── Save all crops as two npz archives ────────────────────────────────
        print("\n  Saving npz archives...")
        for split in ("train", "val"):
            self._save_npz(all_crops[split], split)
        del all_crops  # free RAM

        self._write_yaml()
        print(f"\n  Done - {total} sheets, {crop_total} crops -> {self.out.resolve()}")

    # ── NPZ save ──────────────────────────────────────────────────────────────

    def _save_npz(self, buf: dict, split: str) -> None:
        if not buf["images"]:
            return
        npz_path = self.out / f"crops_{split}.npz"
        imgs = np.array(buf["images"], dtype=np.uint8)
        lbls = np.array(buf["labels"], dtype=np.int8)
        np.savez_compressed(str(npz_path), images=imgs, labels=lbls)
        counts = np.bincount(lbls.astype(np.int64), minlength=3)
        print(f"    {split}: {len(imgs)} crops  "
              f"empty={counts[0]}  filled={counts[1]}  ambiguous={counts[2]}  "
              f"-> {npz_path.name}  ({npz_path.stat().st_size // (1024*1024)} MB)")

    # ── YAML for CNN trainer ──────────────────────────────────────────────────

    def _write_yaml(self) -> None:
        yaml = textwrap.dedent(f"""\
            # OMR v3 bubble-crop CNN dataset
            path:  {self.out.resolve().as_posix()}
            train: crops_train.npz
            val:   crops_val.npz

            nc: 3
            names:
              0: empty
              1: filled
              2: ambiguous

            # Crop size: {CROP_SIZE}x{CROP_SIZE} px greyscale
            # Rendered at DPI={self.dpi}
        """)
        (self.out / "data_crops.yaml").write_text(yaml)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="OMR Dataset Generator v3 — template-driven synthetic sheets"
    )
    ap.add_argument("--out",       default="dataset_v3",
                    help="output directory (default: dataset_v3)")
    ap.add_argument("--samples",   type=int, default=2400,
                    help="total sheets to generate (default: 2400)")
    ap.add_argument("--dpi",       type=int, default=DEFAULT_DPI,
                    help=f"render DPI (default: {DEFAULT_DPI})")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--templates", nargs="+", default=None,
                    metavar="ID",
                    help=("Template IDs to include, e.g. Q100_5ch Q50_5ch Q20_5ch. "
                          f"Available: {sorted(REGISTRY)}"))
    args = ap.parse_args()

    if args.templates:
        for tid in args.templates:
            if tid not in REGISTRY:
                ap.error(f"Unknown template '{tid}'. "
                         f"Available: {sorted(REGISTRY)}")

    print(f"DPI={args.dpi}  samples={args.samples}  out={args.out}")
    DatasetGeneratorV3(
        out_dir      = args.out,
        n_total      = args.samples,
        dpi          = args.dpi,
        seed         = args.seed,
        template_ids = args.templates,
    ).generate()


if __name__ == "__main__":
    main()

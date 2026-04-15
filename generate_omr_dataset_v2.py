#!/usr/bin/env python3
"""
generate_omr_dataset_v2.py
──────────────────────────
Synthetic OMR answer-sheet dataset generator.

Sheets    : 20 / 50 / 100 questions
Choices   : 2 / 3 / 4 / 5 per row
Marks     : pen_solid, pen_partial, pencil_light, pencil_dark,
            marker, x_mark, check, sloppy_fill
Answers   : blank, single, double (2 choices), triple (3 choices)
Distortion: clean, light, medium, heavy
            (rotation · perspective · brightness · noise · blur · JPEG)

Outputs
───────
  <out>/images/{train,val}/   – distorted JPEGs
  <out>/labels/{train,val}/   – YOLO labels  (class cx cy w h  – normalised)
  <out>/samples/              – annotated previews for visual QA
  <out>/data.yaml             – YOLOv8 dataset config

YOLO classes
────────────
  0  bubble_empty
  1  bubble_filled
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random, math, textwrap, argparse, itertools
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ─────────────────────────────────────────────────────────────────────────────
# Page geometry — all values in mm, resolved to pixels at runtime via init_geometry()
# ─────────────────────────────────────────────────────────────────────────────
DPI    = 100        # default; overridden by --dpi
PX_MM  = DPI / 25.4

# These will be set by init_geometry() after DPI is known
A4_W = A4_H = 0
MAR_X = MAR_TOP = MAR_BOT = HDR_H = 0
GRID_X0 = GRID_Y0 = GRID_X1 = GRID_Y1 = 0
ANCHOR_BIG = ANCHOR_MID = COL_DOT = 0


def init_geometry(dpi: int):
    """Recalculate all pixel constants for the chosen DPI."""
    global DPI, PX_MM
    global A4_W, A4_H
    global MAR_X, MAR_TOP, MAR_BOT, HDR_H
    global GRID_X0, GRID_Y0, GRID_X1, GRID_Y1
    global ANCHOR_BIG, ANCHOR_MID, COL_DOT

    DPI   = dpi
    PX_MM = dpi / 25.4

    A4_W    = round(210 * PX_MM)
    A4_H    = round(297 * PX_MM)
    MAR_X   = round(14  * PX_MM)
    MAR_TOP = round(11  * PX_MM)
    MAR_BOT = round(9   * PX_MM)
    HDR_H   = round(48  * PX_MM)
    GRID_X0 = MAR_X
    GRID_Y0 = MAR_TOP + HDR_H
    GRID_X1 = A4_W - MAR_X
    GRID_Y1 = A4_H - MAR_BOT
    ANCHOR_BIG = round(10 * PX_MM)
    ANCHOR_MID = round(6  * PX_MM)
    COL_DOT    = round(4  * PX_MM)

CHOICES_EN = list("ABCDE")
CHOICES_AR = ["أ", "ب", "ج", "د", "هـ"]

MARK_STYLES  = [
    "pen_solid", "pen_partial",
    "pencil_light", "pencil_dark",
    "marker", "x_mark", "check", "sloppy_fill",
]
DISTORTION_LEVELS = ["clean", "light", "medium", "heavy"]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Bubble:
    cx: int; cy: int; r: int
    question: int   # 1-based
    choice:   int   # 0-based
    filled:   bool  = False


@dataclass
class SheetSpec:
    n_q:   int
    n_ch:  int
    n_cols: int
    rpc:   int    # rows per column
    row_h: int
    col_w: int
    bub_r: int
    qn_w:  int    # question-number zone width (px)


def spec_for(n_q: int, n_ch: int) -> SheetSpec:
    n_cols = 2 if n_q <= 20 else 4
    rpc    = math.ceil(n_q / n_cols)
    n_hdrs = math.ceil(rpc / 5)          # header rows every 5 questions
    col_w  = (GRID_X1 - GRID_X0) // n_cols
    row_h  = (GRID_Y1 - GRID_Y0) // (rpc + n_hdrs)
    bub_r  = min(int(row_h * 0.30), int(col_w * 0.075 / max(n_ch - 2, 1)))
    bub_r  = max(bub_r, 7)
    qn_w   = max(round(3.5 * PX_MM), bub_r * 2)
    return SheetSpec(n_q=n_q, n_ch=n_ch, n_cols=n_cols, rpc=rpc,
                     row_h=row_h, col_w=col_w, bub_r=bub_r, qn_w=qn_w)


# ─────────────────────────────────────────────────────────────────────────────
# Font loader
# ─────────────────────────────────────────────────────────────────────────────
def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# Sheet renderer
# ─────────────────────────────────────────────────────────────────────────────
class SheetRenderer:
    """Renders a blank OMR sheet; returns (BGR ndarray, list[Bubble])."""

    HEADER_EVERY = 5

    def __init__(self, spec: SheetSpec):
        self.sp = spec
        sz_map = {20: 14, 50: 11, 100: 9}
        sz = next((v for k, v in sorted(sz_map.items()) if spec.n_q <= k), 9)
        self._font      = _load_font(sz)
        self._font_bold = _load_font(sz, bold=True)
        self._font_hdr  = _load_font(sz - 1, bold=True)
        self._font_title = _load_font(22, bold=True)

    # ── public ────────────────────────────────────────────────────────────────
    def render(self) -> Tuple[np.ndarray, List[Bubble]]:
        img  = Image.new("RGB", (A4_W, A4_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        self._draw_anchors(draw)
        self._draw_header(draw)
        bubbles = self._draw_grid(draw)
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return bgr, bubbles

    # ── anchors ───────────────────────────────────────────────────────────────
    def _draw_anchors(self, draw: ImageDraw.Draw):
        s = ANCHOR_BIG
        # four corners
        for ax, ay in [
            (MAR_X,           MAR_TOP),
            (A4_W - MAR_X - s, MAR_TOP),
            (MAR_X,           A4_H - MAR_BOT - s),
            (A4_W - MAR_X - s, A4_H - MAR_BOT - s),
        ]:
            draw.rectangle([ax, ay, ax + s, ay + s], fill=(0, 0, 0))

        # mid-edge
        m  = ANCHOR_MID
        my = A4_H // 2
        for ax in [MAR_X, A4_W - MAR_X - m]:
            draw.rectangle([ax, my - m // 2, ax + m, my + m // 2], fill=(0, 0, 0))

        # col-boundary dots (top + bottom of each column)
        sp = self.sp
        d  = COL_DOT
        for col in range(sp.n_cols):
            dcx = GRID_X0 + col * sp.col_w + sp.col_w // 2
            for dcy in [GRID_Y0 - round(4 * PX_MM), GRID_Y1 + round(2 * PX_MM)]:
                draw.rectangle([dcx - d // 2, dcy - d // 2,
                                dcx + d // 2, dcy + d // 2], fill=(0, 0, 0))

    # ── header ────────────────────────────────────────────────────────────────
    def _draw_header(self, draw: ImageDraw.Draw):
        sp = self.sp
        title = "Answer Sheet"
        bbox  = draw.textbbox((0, 0), title, font=self._font_title)
        tw    = bbox[2] - bbox[0]
        draw.text(((A4_W - tw) // 2, MAR_TOP + 4), title,
                  fill=(0, 0, 0), font=self._font_title)

        # meta row
        meta_y = MAR_TOP + 30
        meta_items = [
            ("Student:", "Ahmed Ali"),
            ("ID:",       "STU-20240001"),
            ("Exam:",     "EXAM-2024-MATH-01"),
            ("Class:",    "Grade 10-A"),
        ]
        meta_x = GRID_X0
        col_w_meta = (GRID_X1 - GRID_X0) // len(meta_items)
        for lbl, val in meta_items:
            draw.text((meta_x + 2, meta_y), lbl,
                      fill=(120, 120, 120), font=self._font)
            draw.text((meta_x + 2, meta_y + 13), val,
                      fill=(0, 0, 0), font=self._font_bold)
            draw.line([(meta_x, meta_y + 26), (meta_x + col_w_meta - 4, meta_y + 26)],
                      fill=(180, 180, 180), width=1)
            meta_x += col_w_meta

        # legend
        leg_y = GRID_Y0 - round(8 * PX_MM)
        draw.rectangle([GRID_X0, leg_y, GRID_X1, leg_y + round(6 * PX_MM)],
                       fill=(245, 245, 245), outline=(180, 180, 180), width=1)
        legend_text = (
            f"n_questions={sp.n_q}   "
            f"n_choices={sp.n_ch}   "
            f"Fill the circle completely"
        )
        draw.text((GRID_X0 + 6, leg_y + 4), legend_text,
                  fill=(80, 80, 80), font=self._font)

        # separator
        draw.line([(GRID_X0, GRID_Y0 - 2), (GRID_X1, GRID_Y0 - 2)],
                  fill=(150, 150, 150), width=2)

    # ── grid ──────────────────────────────────────────────────────────────────
    def _draw_grid(self, draw: ImageDraw.Draw) -> List[Bubble]:
        sp      = self.sp
        bubbles = []
        ch_lbls = CHOICES_EN[:sp.n_ch]

        for col in range(sp.n_cols):
            q_start = col * sp.rpc + 1
            q_end   = min((col + 1) * sp.rpc, sp.n_q)
            if q_start > sp.n_q:
                break

            cx0    = GRID_X0 + col * sp.col_w
            row_vi = 0   # visual row counter (includes header rows)

            for i, q in enumerate(range(q_start, q_end + 1)):
                # header row every HEADER_EVERY questions
                if i % self.HEADER_EVERY == 0:
                    hy = GRID_Y0 + row_vi * sp.row_h + sp.row_h // 2
                    self._draw_col_header(draw, cx0, hy, ch_lbls)
                    row_vi += 1

                y_cen  = GRID_Y0 + row_vi * sp.row_h + sp.row_h // 2
                row_vi += 1

                # question number (right-aligned in column)
                qn_x = cx0 + sp.col_w - 2
                draw.text((qn_x, y_cen), str(q),
                          fill=(0, 0, 0), font=self._font_bold, anchor="rm")

                # bubbles in remaining zone
                zone_w  = sp.col_w - sp.qn_w - round(2 * PX_MM)
                spacing = zone_w // sp.n_ch
                bub_off = spacing // 2

                for ch in range(sp.n_ch):
                    bx = cx0 + bub_off + ch * spacing
                    r  = sp.bub_r
                    draw.ellipse([bx - r, y_cen - r, bx + r, y_cen + r],
                                 outline=(0, 0, 0), width=2)
                    bubbles.append(Bubble(cx=bx, cy=y_cen, r=r,
                                          question=q, choice=ch))

            # phantom rows: pad short columns so flex heights match
            n_actual = q_end - q_start + 1
            for i in range(n_actual, sp.rpc):
                if i % self.HEADER_EVERY == 0:
                    row_vi += 1   # skip phantom header slot
                row_vi += 1

            # column separator
            if col < sp.n_cols - 1:
                lx = cx0 + sp.col_w - 1
                draw.line([(lx, GRID_Y0), (lx, GRID_Y1)],
                          fill=(200, 200, 200), width=1)

        return bubbles

    def _draw_col_header(self, draw: ImageDraw.Draw,
                         cx0: int, y_cen: int, labels: List[str]):
        sp = self.sp
        draw.rectangle([cx0, y_cen - sp.row_h // 2 + 1,
                        cx0 + sp.col_w - 1, y_cen + sp.row_h // 2 - 1],
                       fill=(238, 238, 248))
        zone_w  = sp.col_w - sp.qn_w - round(2 * PX_MM)
        spacing = zone_w // sp.n_ch
        bub_off = spacing // 2
        for ch, lbl in enumerate(labels):
            lx = cx0 + bub_off + ch * spacing
            draw.text((lx, y_cen), lbl,
                      fill=(60, 60, 180), font=self._font_hdr, anchor="mm")


# ─────────────────────────────────────────────────────────────────────────────
# Answer simulator
# ─────────────────────────────────────────────────────────────────────────────
class AnswerSimulator:
    """Marks bubbles with realistic student handwriting styles."""

    def simulate(self,
                 img: np.ndarray,
                 bubbles: List[Bubble],
                 answers: Dict[int, List[int]],
                 style: str) -> np.ndarray:
        out = img.copy()
        for b in bubbles:
            if b.question in answers and b.choice in answers[b.question]:
                b.filled = True
                self._mark(out, b, style)
        return out

    # ── marking primitives ────────────────────────────────────────────────────
    def _mark(self, img: np.ndarray, b: Bubble, style: str):
        cx, cy, r = b.cx, b.cy, b.r

        if style == "pen_solid":
            cv2.circle(img, (cx, cy), r - 2, (15, 15, 15), -1)

        elif style == "pen_partial":
            ox, oy = random.randint(-2, 2), random.randint(-2, 2)
            cv2.circle(img, (cx + ox, cy + oy), r - 3, (25, 25, 25), -1)

        elif style == "pencil_light":
            gray = random.randint(140, 185)
            cv2.circle(img, (cx, cy), r - 2, (gray, gray, gray), -1)
            self._pencil_texture(img, cx, cy, r, lo=155, hi=220)

        elif style == "pencil_dark":
            gray = random.randint(55, 105)
            cv2.circle(img, (cx, cy), r - 2, (gray, gray, gray), -1)
            self._pencil_texture(img, cx, cy, r, lo=90, hi=150)

        elif style == "marker":
            cv2.circle(img, (cx, cy), r - 1, (8, 8, 25), -1)
            for _ in range(6):
                bx = cx + random.randint(-2, 2)
                by = cy + random.randint(-2, 2)
                cv2.circle(img, (bx, by), r, (30, 30, 55), 1)

        elif style == "x_mark":
            d = r - 2
            thick = max(1, r // 5)
            cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), (10, 10, 10), thick)
            cv2.line(img, (cx + d, cy - d), (cx - d, cy + d), (10, 10, 10), thick)

        elif style == "check":
            d = r - 2
            pts = np.array([[cx - d, cy],
                            [cx - d // 3, cy + d - 1],
                            [cx + d,      cy - d // 2]], np.int32)
            thick = max(1, r // 5)
            cv2.polylines(img, [pts], False, (10, 10, 10), thick)

        elif style == "sloppy_fill":
            mask = np.zeros(img.shape[:2], np.uint8)
            for _ in range(6):
                ox = random.randint(-r // 3, r // 3)
                oy = random.randint(-r // 3, r // 3)
                ax = r - 1 + random.randint(-2, 2)
                ay = r - 1 + random.randint(-2, 2)
                cv2.ellipse(mask, (cx + ox, cy + oy), (max(ax, 3), max(ay, 3)),
                            random.randint(0, 360), 0, 360, 255, -1)
            img[mask > 0] = [20, 20, 20]

    @staticmethod
    def _pencil_texture(img, cx, cy, r, lo=150, hi=220):
        """Add random light-scatter dots to simulate pencil grain."""
        for _ in range(18):
            px = cx + random.randint(-(r - 2), r - 2)
            py = cy + random.randint(-(r - 2), r - 2)
            if (px - cx) ** 2 + (py - cy) ** 2 < (r - 2) ** 2:
                v = random.randint(lo, hi)
                cv2.circle(img, (px, py), 1, (v, v, v), -1)

    # ── answer pattern generator ──────────────────────────────────────────────
    @staticmethod
    def random_answers(n_q: int, n_ch: int,
                       blank_prob: float  = 0.05,
                       double_prob: float = 0.08,
                       triple_prob: float = 0.03) -> Dict[int, List[int]]:
        """
        Returns {question: [list of chosen choice indices]}.
        blank   → []
        single  → [c]
        double  → [c1, c2]
        triple  → [c1, c2, c3]
        """
        answers = {}
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


# ─────────────────────────────────────────────────────────────────────────────
# Distortion engine  (also transforms bubble coordinates)
# ─────────────────────────────────────────────────────────────────────────────
class DistortionEngine:
    """
    apply(img, bubbles, level) -> (distorted_img, transformed_bubbles)
    Bubble cx/cy are updated to match the distorted image.
    """

    @staticmethod
    def apply(img: np.ndarray,
              bubbles: List[Bubble],
              level: str) -> Tuple[np.ndarray, List[Bubble]]:

        pts = np.float32([[b.cx, b.cy] for b in bubbles])  # (N,2)
        out = img.copy()

        if level == "clean":
            return out, bubbles

        # ── rotation ──────────────────────────────────────────────────────────
        max_angle = {"light": 1.5, "medium": 3.5, "heavy": 7.0}[level]
        angle = random.uniform(-max_angle, max_angle)
        out, pts = DistortionEngine._rotate(out, pts, angle)

        # ── perspective ───────────────────────────────────────────────────────
        if level in ("medium", "heavy"):
            strength = {"medium": 0.008, "heavy": 0.020}[level]
            out, pts = DistortionEngine._perspective(out, pts, strength)

        # ── brightness / contrast ─────────────────────────────────────────────
        b_range = {"light": (0.82, 1.18),
                   "medium": (0.60, 1.40),
                   "heavy":  (0.38, 1.65)}[level]
        out = DistortionEngine._brightness(out, b_range)

        # ── Gaussian noise ────────────────────────────────────────────────────
        noise_std = {"light": 6, "medium": 16, "heavy": 32}[level]
        out = DistortionEngine._noise(out, noise_std)

        # ── blur / JPEG ───────────────────────────────────────────────────────
        if level == "heavy" and random.random() < 0.55:
            out = DistortionEngine._motion_blur(out)
        if level == "heavy":
            out = DistortionEngine._jpeg(out, random.randint(38, 68))
        elif level == "medium" and random.random() < 0.35:
            out = DistortionEngine._jpeg(out, random.randint(58, 82))

        # rebuild Bubble list with updated coordinates
        new_bubbles = []
        for b, (nx, ny) in zip(bubbles, pts.tolist()):
            new_bubbles.append(Bubble(cx=int(round(nx)), cy=int(round(ny)),
                                      r=b.r, question=b.question,
                                      choice=b.choice, filled=b.filled))
        return out, new_bubbles

    # ── transforms ────────────────────────────────────────────────────────────
    @staticmethod
    def _rotate(img, pts, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        out = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(240, 240, 240))
        if len(pts):
            ones = np.ones((len(pts), 1), dtype=np.float32)
            pts_h = np.hstack([pts, ones])          # (N,3)
            pts   = (M @ pts_h.T).T                 # (N,2)
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
            pts = cv2.perspectiveTransform(
                pts.reshape(-1, 1, 2), M).reshape(-1, 2)
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
# YOLO label helper
# ─────────────────────────────────────────────────────────────────────────────
def bubbles_to_yolo(bubbles: List[Bubble], iw: int, ih: int) -> str:
    lines = []
    for b in bubbles:
        cls = 1 if b.filled else 0
        bw  = (b.r * 2 + 4) / iw
        bh  = (b.r * 2 + 4) / ih
        cx  = b.cx / iw
        cy  = b.cy / ih
        # clamp to [0,1]
        cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
        bw, bh = max(0.001, min(1.0, bw)), max(0.001, min(1.0, bh))
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generator
# ─────────────────────────────────────────────────────────────────────────────
class DatasetGenerator:

    def __init__(self, out_dir: str, n_total: int = 600, seed: int = 42):
        self.out     = Path(out_dir)
        self.n_total = n_total
        random.seed(seed)
        np.random.seed(seed)

        for split in ("train", "val"):
            (self.out / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (self.out / "samples").mkdir(exist_ok=True)

    # ── main loop ─────────────────────────────────────────────────────────────
    def generate(self):
        configs = list(itertools.product(
            [20, 50, 100],   # question counts
            [2, 3, 4, 5],    # choices per row
        ))
        n_per = max(1, self.n_total // len(configs))
        print(f"{len(configs)} sheet configs × {n_per} samples = "
              f"~{len(configs) * n_per} images\n")

        total       = 0
        sample_count = 0

        # pre-render one blank sheet per config (expensive step done once)
        for n_q, n_ch in configs:
            tag = f"{n_q}q_{n_ch}ch"
            print(f"  [{tag}] rendering base sheet …", end="", flush=True)
            sp            = spec_for(n_q, n_ch)
            base_img, base_bubbles = SheetRenderer(sp).render()
            print(f" ({len(base_bubbles)} bubbles)")

            sim = AnswerSimulator()

            for i in range(n_per):
                style   = random.choice(MARK_STYLES)
                level   = random.choice(DISTORTION_LEVELS)
                answers = AnswerSimulator.random_answers(n_q, n_ch)

                # reset fill flags
                for b in base_bubbles:
                    b.filled = False

                marked = sim.simulate(base_img, base_bubbles, answers, style)

                distorted, moved_bubbles = DistortionEngine.apply(
                    marked, list(base_bubbles), level)

                split = "val" if total % 5 == 0 else "train"
                stem  = f"omr_{tag}_{total:06d}"

                cv2.imwrite(
                    str(self.out / "images" / split / f"{stem}.jpg"),
                    distorted, [cv2.IMWRITE_JPEG_QUALITY, 95])

                (self.out / "labels" / split / f"{stem}.txt").write_text(
                    bubbles_to_yolo(moved_bubbles, A4_W, A4_H))

                # save annotated sample preview (2 per config)
                if i < 2:
                    self._save_sample(distorted, moved_bubbles,
                                      f"sample_{tag}_{style}_{level}.jpg",
                                      n_q, n_ch, style, level)
                    sample_count += 1

                total += 1

        self._write_yaml()
        print(f"\nDone — {total} images saved to {self.out.resolve()}")

    # ── sample visualisation ──────────────────────────────────────────────────
    def _save_sample(self, img, bubbles, name,
                     n_q, n_ch, style, level):
        vis = img.copy()
        filled_c  = (0,  210, 60)
        empty_c   = (180, 180, 180)
        for b in bubbles:
            color = filled_c if b.filled else empty_c
            cv2.circle(vis, (b.cx, b.cy), b.r + 2, color, 2)

        # info bar at top
        bar_h = 30
        overlay = vis[:bar_h].copy()
        cv2.rectangle(vis, (0, 0), (A4_W, bar_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.3, vis[:bar_h], 0.7, 0, vis[:bar_h])
        info = (f"Q={n_q}  CH={n_ch}  style={style}  dist={level}  "
                f"filled={sum(b.filled for b in bubbles)}/{len(bubbles)}")
        cv2.putText(vis, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)

        cv2.imwrite(str(self.out / "samples" / name), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ── data.yaml ─────────────────────────────────────────────────────────────
    def _write_yaml(self):
        yaml = textwrap.dedent(f"""\
            path:  {self.out.resolve().as_posix()}
            train: images/train
            val:   images/val

            nc: 2
            names:
              0: bubble_empty
              1: bubble_filled

            # Recommended training: imgsz=640, batch=8, epochs=100
            # Generated at DPI={DPI}, image size={A4_W}x{A4_H}
        """)
        (self.out / "data.yaml").write_text(yaml)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="OMR Dataset Generator v2")
    ap.add_argument("--out",     default="dataset_v2",
                    help="output directory (default: dataset_v2)")
    ap.add_argument("--samples", type=int, default=2400,
                    help="total images to generate (default: 2400)")
    ap.add_argument("--dpi",     type=int, default=100,
                    help="render DPI — 100 is fast (827x1169), 150 is sharper (default: 100)")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    init_geometry(args.dpi)
    print(f"DPI={args.dpi}  image size={A4_W}x{A4_H}  samples={args.samples}")
    DatasetGenerator(args.out, args.samples, args.seed).generate()


if __name__ == "__main__":
    main()

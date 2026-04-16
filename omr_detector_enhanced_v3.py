#!/usr/bin/env python3
"""
omr_detector_enhanced_v3.py
────────────────────────────
OMR pipeline v3 — template-driven, global-lattice row fitting.

Key improvements over v2:
  1. Template-driven geometry  — TemplateSpec replaces hard-coded CFG sheet
                                 geometry; no per-run global mutations.
  2. Global row lattice fitting — after per-column row detection, a global
                                  row pitch is fitted and all columns' rows
                                  are snapped to the same lattice.
  3. Illumination normalisation — each bubble crop is divided by a large
                                  Gaussian blur before classification to
                                  remove shadows and gradients.
  4. Subpixel crops             — cv2.getRectSubPix for more precise sampling.
  5. Per-bubble confidence      — CNN softmax max-prob; fill-ratio scalar for
                                  fallback.  Propagated to per-question and
                                  sheet-level confidence.
  6. Choice labels from template — no fixed ABCDE; works for 2/3/4/5 choices
                                   and both RTL and LTR sheets.

Usage:
  python omr_detector_enhanced_v3.py ans.jpg --questions 100
  python omr_detector_enhanced_v3.py ans.jpg --questions 50 --choices 4
  python omr_detector_enhanced_v3.py ans.jpg --template Q100_5ch
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from omr_templates import (
    WARP_W, WARP_H,
    WARP_GX0, WARP_GX1, WARP_GY0, WARP_GY1,
    WARP_ACX_L, WARP_ACX_R, WARP_ACY_T, WARP_ACY_B,
    TemplateSpec,
    get_template,
    infer_template,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm configuration (tuneable; NOT sheet geometry)
# ─────────────────────────────────────────────────────────────────────────────

ALGO = {
    # Preprocessing
    "bilateral_d":         9,
    "bilateral_sigma_col": 75,
    "bilateral_sigma_sp":  75,
    # Bar detection
    "bar_dark_thr":        140,
    "bar_min_width_px":    30,
    # Row detection
    "row_dark_thr":        160,
    # Bubble classification
    "fill_norm_thr":       185,   # normalized pixel < this = dark (after bg correction)
    "fill_ratio_thr":      0.35,
    "fill_ratio_ambi_lo":  0.18,
    "fill_ratio_ambi_hi":  0.55,
    "bubble_r":            9,
    "cls_r_shrink":        1,     # classification samples bubble_r minus this (avoids ring ink)
    # Lattice fitting
    "lattice_max_snap_px": 12,   # snap only if detected centre within N px
    # CNN model paths
    "onnx_path": "bubble_classifier_v3.onnx",
    "pt_path":   "bubble_classifier_v3.pt",
    # Cross-row consistency
    "max_consec_empty_rows": 10,
    # Circle-centre snapping (step 5E)
    "circle_snap_r":          6,   # ±px search radius around template position
    "circle_snap_min_gain":   3.0, # min edge-score improvement to accept a snap
}

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: uniform row grid (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_rows(y0: int, y1: int, n: int) -> List[int]:
    if n == 0:
        return []
    h = (y1 - y0) / n
    return [round(y0 + (i + 0.5) * h) for i in range(n)]


def _dbg(img: np.ndarray, name: str, d: Optional[pathlib.Path]) -> None:
    if d:
        cv2.imwrite(str(d / name), img)


# ─────────────────────────────────────────────────────────────────────────────
#  Data contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InputContract:
    image_path: str
    image:      np.ndarray
    metadata:   Dict[str, Any]


@dataclass
class PreprocessContract:
    original_bgr:  np.ndarray
    enhanced_gray: np.ndarray
    quality_score: float


@dataclass
class AlignmentContract:
    aligned_bgr:         np.ndarray
    align_method:        str
    alignment_confidence: float
    warp_matrix:         Optional[np.ndarray] = None


@dataclass
class ColumnLayout:
    col_idx: int
    x0: int; x1: int
    y0: int; y1: int
    from_bar: bool = False   # True when x0/x1 are bar-detected bubble zone bounds


@dataclass
class LayoutContract:
    aligned_bgr:        np.ndarray
    grid_x0: int; grid_y0: int; grid_x1: int; grid_y1: int
    columns:            List[ColumnLayout]
    layout_confidence:  float
    student_code:       str = ""
    exam_code:          str = ""
    qr_raw:             List[str] = field(default_factory=list)


@dataclass
class GeneratedBubble:
    cx: int; cy: int; r: int
    col_idx:     int
    row_idx:     int
    choice_rank: int
    question_id: str
    option:      str


@dataclass
class BubbleGridContract:
    aligned_bgr: np.ndarray
    bubbles:     List[GeneratedBubble]
    grid_x0: int; grid_y0: int; grid_x1: int; grid_y1: int
    lattice_quality: float   # 0-1, 1 = all rows snapped perfectly
    bubble_r:        int


@dataclass
class BubblePrediction:
    question_id: str
    option:      str
    status:      str    # "empty" | "filled" | "ambiguous"
    fill_ratio:  float
    confidence:  float  # CNN softmax max or fill-ratio derived confidence


@dataclass
class ClassificationContract:
    predictions:             List[BubblePrediction]
    bubbles:                 List[GeneratedBubble]
    aligned_bgr:             np.ndarray
    classifier_type:         str
    classification_confidence: float


@dataclass
class QuestionResult:
    choice:      Optional[str]
    note:        str          # "ok" | "blank" | "double_mark" | "ambiguous"
    all_filled:  List[str]
    fill:        float
    classifier_type: str
    confidence:  float


@dataclass
class ValidationContract:
    answers:      Dict[str, Optional[str]]
    details:      Dict[str, QuestionResult]
    unanswered:   List[str]
    double_marked: List[str]
    ambiguous:    List[str]
    low_conf_rows: List[str]   # cross-row consistency flags


@dataclass
class FinalResultContract:
    image:            str
    student_code:     str
    exam_code:        str
    qr_codes_raw:     List[str]
    align_method:     str
    answers:          Dict[str, Optional[str]]
    answer_details:   Dict[str, Any]
    total_questions:  int
    answered:         int
    unanswered:       List[str]
    double_marked:    List[str]
    ambiguous:        List[str]
    low_conf_rows:    List[str]
    valid:            bool
    debug_dir:        str
    confidence_metrics:  Dict[str, Any]
    processing_metrics:  Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — INPUT
# ─────────────────────────────────────────────────────────────────────────────

class InputLayer:
    def process(self, image_path: str,
                debug_dir: pathlib.Path) -> InputContract:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path!r}")
        _dbg(img, "00_original.jpg", debug_dir)
        h, w = img.shape[:2]
        print(f"  [1-input] {w}×{h}  {image_path}")
        return InputContract(image_path=image_path, image=img,
                             metadata={"resolution": (w, h),
                                       "timestamp":  time.time()})


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

class PreprocessingEngine:
    """CLAHE → optional grey-world white-balance → bilateral denoise."""

    def process(self, data: InputContract,
                white_balance: bool = True,
                denoise: bool = True) -> PreprocessContract:
        img = data.image

        if white_balance:
            img = self._grey_world(img)

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        if denoise:
            enhanced = cv2.bilateralFilter(
                enhanced,
                ALGO["bilateral_d"],
                ALGO["bilateral_sigma_col"],
                ALGO["bilateral_sigma_sp"],
            )

        lap     = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality = float(np.clip(lap / 500.0, 0.0, 1.0))
        print(f"  [2-preprocess] quality={quality:.3f}  "
              f"wb={white_balance}  denoise={denoise}")
        return PreprocessContract(original_bgr=data.image,
                                  enhanced_gray=enhanced,
                                  quality_score=quality)

    @staticmethod
    def _grey_world(img: np.ndarray) -> np.ndarray:
        r = img.astype(np.float32)
        means = [r[:, :, c].mean() for c in range(3)]
        gm    = float(np.mean(means))
        for c in range(3):
            if means[c] > 0:
                r[:, :, c] = np.clip(r[:, :, c] * (gm / means[c]), 0, 255)
        return r.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 — ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

class AlignmentEngine:
    """Perspective alignment using anchor detection (no YOLO)."""

    @staticmethod
    def _order_pts(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    def _find_anchors(self, img: np.ndarray
                      ) -> Optional[Dict[str, Tuple[int, int]]]:
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw   = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 51, 15)
        cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        min_s = min(W, H) * 0.012
        max_s = min(W, H) * 0.15
        cands = []
        for c in cnts:
            area = cv2.contourArea(c)
            if not (min_s ** 2 < area < max_s ** 2):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if min(w, h) / (max(w, h) + 1e-6) < 0.65:
                continue
            hull = cv2.convexHull(c)
            if area / (cv2.contourArea(hull) + 1e-6) < 0.70:
                continue
            Mv = cv2.moments(c)
            if Mv["m00"] == 0:
                continue
            cands.append((int(Mv["m10"] / Mv["m00"]),
                          int(Mv["m01"] / Mv["m00"]), area))

        if len(cands) < 4:
            return None
        cands.sort(key=lambda x: x[2], reverse=True)
        best_quad, max_area = None, 0
        for quad in itertools.combinations(cands[:12], 4):
            pts     = np.float32([[p[0], p[1]] for p in quad])
            ordered = self._order_pts(pts)
            qa      = cv2.contourArea(ordered)
            if qa <= max_area:
                continue
            tl, tr, br, bl = ordered
            wt = np.linalg.norm(tr - tl); wb = np.linalg.norm(br - bl)
            hl = np.linalg.norm(bl - tl); hr = np.linalg.norm(br - tr)
            if (min(wt, wb) / (max(wt, wb) + 1e-6) > 0.70 and
                    min(hl, hr) / (max(hl, hr) + 1e-6) > 0.70):
                max_area  = qa
                best_quad = ordered
        if best_quad is not None and max_area > W * H * 0.05:
            return {"TL": tuple(best_quad[0].astype(int)),
                    "TR": tuple(best_quad[1].astype(int)),
                    "BR": tuple(best_quad[2].astype(int)),
                    "BL": tuple(best_quad[3].astype(int))}
        return None

    def _find_document_corners(self, gray: np.ndarray
                                ) -> Optional[np.ndarray]:
        H, W = gray.shape
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        for lo, hi in [(50, 150), (30, 100), (10, 60)]:
            edges = cv2.Canny(blur, lo, hi)
            k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, k, iterations=1)
            cnts, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
                if cv2.contourArea(c) < 0.35 * H * W:
                    continue
                peri   = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    return approx.reshape(4, 2).astype(np.float32)
        return None

    def process(self, prep: PreprocessContract,
                debug_dir: pathlib.Path) -> AlignmentContract:
        img  = prep.original_bgr
        gray = prep.enhanced_gray
        W, H = WARP_W, WARP_H
        method = "full-image"
        warped = None
        M_used = None
        conf   = 0.0

        # Anchor-based (most robust)
        anchors = self._find_anchors(img)
        if anchors:
            src = np.float32([anchors["TL"], anchors["TR"],
                              anchors["BR"], anchors["BL"]])
            dst = np.float32([
                [WARP_ACX_L, WARP_ACY_T],
                [WARP_ACX_R, WARP_ACY_T],
                [WARP_ACX_R, WARP_ACY_B],
                [WARP_ACX_L, WARP_ACY_B],
            ])
            M      = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img, M, (W, H),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            M_used = M
            method = "anchor-based"
            conf   = 0.95

        # Canny-contour fallback
        if warped is None:
            corners = self._find_document_corners(gray)
            if corners is not None:
                ordered = self._order_pts(corners)
                M       = cv2.getPerspectiveTransform(ordered,
                    np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
                warped  = cv2.warpPerspective(img, M, (W, H),
                                              flags=cv2.INTER_CUBIC,
                                              borderMode=cv2.BORDER_REPLICATE)
                M_used  = M
                method  = "canny-contour"
                conf    = 0.60

        # Full-image last resort
        if warped is None:
            iH, iW = img.shape[:2]
            ins = 10
            src = np.float32([[ins, ins], [iW - ins, ins],
                              [iW - ins, iH - ins], [ins, iH - ins]])
            dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
            M       = cv2.getPerspectiveTransform(src, dst)
            warped  = cv2.warpPerspective(img, M, (W, H))
            M_used  = M
            method  = "full-image"
            conf    = 0.25

        _dbg(warped, "03_warped.jpg", debug_dir)
        print(f"  [3-align] method={method}  confidence={conf:.2f}")
        return AlignmentContract(aligned_bgr=warped, align_method=method,
                                 alignment_confidence=conf, warp_matrix=M_used)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4 — STATIC LAYOUT (template-driven)
# ─────────────────────────────────────────────────────────────────────────────

class StaticLayoutEngine:
    """
    Derive the question-grid bounding box and n_cols column layouts from the
    TemplateSpec.  No global CFG geometry read here.
    """

    def process(self, align: AlignmentContract,
                template: TemplateSpec,
                debug_dir: pathlib.Path) -> LayoutContract:
        warped = align.aligned_bgr

        # Grid bounds directly from template constants (warped-space)
        gx0, gx1 = WARP_GX0, WARP_GX1
        gy0, gy1 = WARP_GY0, WARP_GY1

        # Static columns split evenly
        col_w = (gx1 - gx0) // template.n_cols
        columns: List[ColumnLayout] = []
        for i in range(template.n_cols):
            cx0 = gx0 + i * col_w
            cx1 = cx0 + col_w
            columns.append(ColumnLayout(col_idx=i,
                                        x0=cx0, x1=cx1,
                                        y0=gy0,  y1=gy1))

        vis = warped.copy()
        for col in columns:
            cv2.rectangle(vis, (col.x0, col.y0), (col.x1, col.y1),
                          (0, 200, 80), 1)
        _dbg(vis, "04_layout.jpg", debug_dir)

        print(f"  [4-layout] grid=[{gx0},{gy0}]->[{gx1},{gy1}]  "
              f"cols={template.n_cols}  col_w~{col_w}")
        return LayoutContract(aligned_bgr=warped,
                              grid_x0=gx0, grid_y0=gy0,
                              grid_x1=gx1, grid_y1=gy1,
                              columns=columns,
                              layout_confidence=1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 5 — BUBBLE GRID (bar refinement + global lattice fitting)
# ─────────────────────────────────────────────────────────────────────────────

class BubbleGridEngine:
    """
    5A  Bar detection  — scan just outside the grid for printed column bars
    5B  Column refinement — replace static columns with bar-detected ones
    5C  Per-column row detection + global lattice fitting
    5D  Bubble position generation from template geometry
    """

    # ── 5A: bar detection ─────────────────────────────────────────────────────

    @staticmethod
    def _detect_bars(gray: np.ndarray,
                     gx0: int, gy0: int, gx1: int, gy1: int,
                     n_cols: int) -> List[Tuple[int, int, int, int]]:
        H, W        = gray.shape
        dark_thr    = ALGO["bar_dark_thr"]
        min_bar     = ALGO["bar_min_width_px"]
        col_width   = (gx1 - gx0) / max(n_cols, 1)
        min_bar_run = max(min_bar, int(col_width * 0.20))

        scan_x0 = max(0, gx0 - 30)
        scan_x1 = min(W, gx1 + 30)

        def _scan(sy0: int, sy1: int) -> List[Tuple[int, int]]:
            sy0 = max(0, sy0); sy1 = min(H, sy1)
            if sy0 >= sy1:
                return []
            strip    = gray[sy0:sy1, scan_x0:scan_x1]
            dark_col = np.sum(strip < dark_thr, axis=0)
            is_dark  = dark_col >= max(3, (sy1 - sy0) * 0.4)
            runs: List[Tuple[int, int]] = []
            in_run = False
            rs     = 0
            for xi, d in enumerate(is_dark):
                if d and not in_run:
                    in_run, rs = True, xi
                elif not d and in_run:
                    in_run = False
                    if xi - rs >= min_bar_run:
                        runs.append((scan_x0 + rs, scan_x0 + xi - 1))
            if in_run and len(is_dark) - rs >= min_bar_run:
                runs.append((scan_x0 + rs, scan_x0 + len(is_dark) - 1))
            return runs

        top = _scan(gy0 - 70, gy0 - 4)
        bot = _scan(gy1 + 4,  gy1 + 70)
        if not top:
            top = _scan(gy0 - 4, gy0 + 30)
        if not bot:
            bot = _scan(gy1 - 30, gy1 + 4)
        if not top:
            return []

        results: List[Tuple[int, int, int, int]] = []
        used_bot: set = set()
        for tx0, tx1 in top:
            tc = (tx0 + tx1) // 2
            best_j, best_d = -1, col_width * 0.4
            for j, (bx0, bx1) in enumerate(bot):
                d = abs((bx0 + bx1) // 2 - tc)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_j not in used_bot:
                bx0, bx1 = bot[best_j]
                results.append((min(tx0, bx0), max(tx1, bx1), gy0 - 4, gy1 + 4))
                used_bot.add(best_j)
            else:
                results.append((tx0, tx1, gy0 - 4, gy1 + 4))
        return sorted(results, key=lambda r: r[0])

    # ── 5B: bars → columns ────────────────────────────────────────────────────

    @staticmethod
    def _bars_to_columns(bars: List[Tuple[int, int, int, int]],
                         static_cols: List[ColumnLayout]) -> List[ColumnLayout]:
        if len(bars) == len(static_cols):
            return [ColumnLayout(col_idx=i,
                                 x0=bx0, x1=bx1,
                                 y0=cty,  y1=cby,
                                 from_bar=True)
                    for i, (bx0, bx1, cty, cby) in enumerate(bars)]
        print(f"  [5-grid] bar count {len(bars)} != {len(static_cols)} - "
              "using static columns")
        return static_cols

    # ── 5C: row detection ─────────────────────────────────────────────────────

    @staticmethod
    def _bands_from_profile(profile: np.ndarray,
                            col_w: float, y0: int,
                            rows_per_group: int,
                            allow_extrap: bool = False,
                            merge_gap_override: int = 0,
                            ) -> Tuple[List[int], str]:
        y1      = y0 + len(profile)
        bar_thr = col_w * 0.55
        bub_thr = max(8.0, col_w * 0.08)
        pitch   = (y1 - y0) / max(rows_per_group, 1)
        mgap    = (merge_gap_override if merge_gap_override > 0
                   else max(5, int(pitch * 0.20)))

        if profile.max() == 0:
            return _uniform_rows(y0, y1, rows_per_group), "fallback"

        # Find bands above bub_thr that are not main bars
        bands: List[Tuple[int, int]] = []
        in_band = False
        bs = 0
        for y, v in enumerate(profile):
            if v >= bub_thr and not in_band:
                in_band, bs = True, y
            elif v < bub_thr and in_band:
                in_band = False
                if profile[bs:y].max() < bar_thr:
                    bands.append((bs, y - 1))
        if in_band and profile[bs:].max() < bar_thr:
            bands.append((bs, len(profile) - 1))

        # Merge close fragments
        merged: List[Tuple[int, int]] = []
        for b in bands:
            if merged and b[0] - merged[-1][1] <= mgap:
                prev = merged.pop()
                merged.append((prev[0], b[1]))
            else:
                merged.append(b)
        bands = merged

        # Proximity filter
        min_prox = max(10, int(pitch * 0.30))
        proxied: List[Tuple[int, int]] = []
        for b in bands:
            bc = (b[0] + b[1]) // 2
            if proxied:
                pc = (proxied[-1][0] + proxied[-1][1]) // 2
                if bc - pc < min_prox:
                    if (b[1] - b[0]) > (proxied[-1][1] - proxied[-1][0]):
                        proxied[-1] = b
                    continue
            proxied.append(b)
        bands = proxied

        if merge_gap_override > 0:
            bubble_r = (merge_gap_override - 1) // 2
            def _cen(b: Tuple[int, int]) -> int:
                return y0 + b[0] + bubble_r
        else:
            def _cen(b: Tuple[int, int]) -> int:
                return y0 + (b[0] + b[1]) // 2

        if len(bands) == rows_per_group:
            return [_cen(b) for b in bands], "exact"
        if len(bands) > rows_per_group:
            top_b = sorted(sorted(bands, key=lambda b: b[1] - b[0],
                                  reverse=True)[:rows_per_group],
                           key=lambda b: b[0])
            return [_cen(b) for b in top_b], "trim"
        if len(bands) >= 3 and allow_extrap:
            cs = [_cen(b) for b in bands]
            sps = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
            med_s        = int(np.median(sps))
            within_pitch = int(np.median([s for s in sps
                                          if s <= int(med_s * 1.5)]))
            while len(cs) < rows_per_group:
                cs.append(cs[-1] + within_pitch)
            return cs[:rows_per_group], "extrapolated"
        return _uniform_rows(y0, y1, rows_per_group), "fallback"

    @staticmethod
    def _detect_rows_col(gray: np.ndarray,
                         col: ColumnLayout,
                         rows_per_group: int,
                         bubble_r: int,
                         combined: Optional[np.ndarray] = None,
                         comb_col_w: float = 0.0,
                         comb_y0: int = 0,
                         ) -> Tuple[List[int], str]:
        dark_thr = ALGO["row_dark_thr"]
        crop     = gray[col.y0:col.y1, col.x0:col.x1]
        if crop.size == 0:
            return _uniform_rows(col.y0, col.y1, rows_per_group), "fallback"
        col_w   = float(crop.shape[1])
        profile = np.sum(crop < dark_thr, axis=1).astype(np.float64)
        ctrs, q = BubbleGridEngine._bands_from_profile(
            profile, col_w, col.y0, rows_per_group
        )
        if q == "fallback" and combined is not None:
            sl = col.y1 - col.y0
            cp = combined[col.y0 - comb_y0: col.y0 - comb_y0 + sl]
            ctrs, q = BubbleGridEngine._bands_from_profile(
                cp, comb_col_w, col.y0, rows_per_group,
                allow_extrap=True, merge_gap_override=2 * bubble_r + 1,
            )
            if q != "fallback":
                q = "combined"
        return ctrs, q

    @staticmethod
    def _detect_question_rows_calibrated(
        gray: np.ndarray,
        template: TemplateSpec,
        gy0: int, gy1: int, gx0: int, gx1: int,
    ) -> Tuple[List[int], int]:
        """
        Detect question-row Y centres from the full-image background-corrected
        horizontal dark-pixel profile.

        Algorithm:
          1. Compute bg-corrected profile (medianBlur(71) / normalise to 0-255).
          2. Detect all dark bands (pixel-run above threshold).
          3. Keep only "wide" bands (width ≥ 12 px): question rows.
             Narrow bands are header labels, noise, or thin separators.
          4. Match detected wide-band centres to the template's expected
             row positions (greedy nearest-neighbour, tolerance = 30 px).
          5. Unmatched rows fall back to expected_y + median_shift.

        Returns: (row_y_centres_list, median_y_shift_px)
        """
        # Background-corrected projection
        bg   = cv2.medianBlur(gray, 71)
        bg   = np.maximum(bg, 1)
        norm = np.clip(
            gray.astype(np.float32) / bg.astype(np.float32) * 255, 0, 255
        ).astype(np.uint8)
        dark = (norm < ALGO["fill_norm_thr"]).astype(np.float32)
        proj = np.sum(dark[:, gx0:gx1], axis=1)

        # Find bands
        bands: List[Tuple[int, int]] = []
        in_b = False
        lo = max(0, gy0 - 40)
        hi = min(len(proj), gy1 + 40)
        for y in range(lo, hi):
            v = proj[y]
            if v > 20 and not in_b:
                bs, in_b = y, True
            elif v <= 20 and in_b:
                bands.append((bs, y - 1))
                in_b = False
        if in_b:
            bands.append((bs, hi - 1))

        # Keep only wide bands (question rows have width ≈ bubble diameter)
        min_band_w = 10
        q_bands = [(s, e) for s, e in bands if e - s + 1 >= min_band_w]
        detected  = sorted((s + e) // 2 for s, e in q_bands)

        expected  = [template.row_y(r) for r in range(template.rows_per_col)]

        # Pitch-adaptive tolerance: use 60% of median within-group pitch,
        # capped at 45px. This safely handles large group-separator gaps
        # (e.g. Q20) while keeping Q100's tight tolerance.
        if len(expected) >= 2:
            all_diffs = sorted([abs(expected[i+1] - expected[i])
                                for i in range(len(expected)-1)])
            within_pitch = float(np.median(all_diffs[:len(all_diffs)*3//4 or 1]))
        else:
            within_pitch = 30.0
        MATCH_TOL = int(min(within_pitch * 0.60, 45))

        # Greedy matching — "first within tolerance" (monotonic order).
        # Using "nearest within tolerance" causes the algorithm to skip the
        # correct band when the template's separator-gap estimate is off:
        # e.g. for Q20, expected[5]=714 sits between actual row5=678(diff=36)
        # and row6=746(diff=32), so "nearest" picks row6 instead of row5.
        # Taking the smallest cy within the window + after last match is always
        # the correct first-unassigned question row.
        sorted_detected = sorted(detected)
        shifts: List[int] = []
        exp_to_det: dict = {}
        last_matched_cy = -1
        for exp_y in expected:
            candidates = [
                cy for cy in sorted_detected
                if cy not in exp_to_det.values()
                and cy > last_matched_cy
                and abs(cy - exp_y) <= MATCH_TOL
            ]
            if candidates:
                best_cy = candidates[0]          # first (smallest) within window
                exp_to_det[exp_y] = best_cy
                shifts.append(best_cy - exp_y)
                last_matched_cy = best_cy

        median_shift = int(np.median(shifts)) if shifts else 0

        row_centers = [
            exp_to_det.get(e, e + median_shift) for e in expected
        ]
        return row_centers, median_shift

    # ── global lattice fitting ─────────────────────────────────────────────────

    @staticmethod
    def _estimate_y_shift(combined: np.ndarray, expected_rows: List[int],
                          y0_com: int) -> int:
        """
        Estimate a global vertical registration offset (in pixels) by
        correlating the horizontal dark-band profile with the expected
        template row positions.  Typical range: 0-5 px.
        """
        if len(expected_rows) < 2 or combined.sum() == 0:
            return 0
        pitch     = float(expected_rows[1] - expected_rows[0])
        max_shift = max(8, int(pitch * 0.20))
        prof_len  = len(combined)
        best_shift, best_score = 0, -1.0
        for delta in range(-max_shift, max_shift + 1):
            score = 0.0
            for ey in expected_rows:
                idx = ey - y0_com + delta
                if 0 <= idx < prof_len:
                    score += combined[idx]
            if score > best_score:
                best_score, best_shift = score, delta
        return best_shift

    @staticmethod
    def _fit_global_lattice(all_centers: List[List[int]],
                            rows_per_group: int,
                            gy0: int, gy1: int,
                            ) -> Tuple[List[int], float]:
        """
        Fit a common row pitch from all detected row centres across columns.

        Strategy:
          1. Estimate pitch per column (median of sorted within-column diffs).
          2. Global pitch = median of per-column pitch estimates.
          3. Build lattice from global pitch + best row0 (minimises snap error).
          4. Quality = fraction of all centres within max_snap_px of lattice.
        """
        # Per-column pitch estimates
        per_col_pitches: List[float] = []
        for centers in all_centers:
            if len(centers) >= 2:
                cs   = sorted(centers)
                dfs  = np.diff(cs)
                dfs  = dfs[dfs > 5]   # filter noise
                if len(dfs):
                    per_col_pitches.append(float(np.median(dfs)))

        if not per_col_pitches:
            return _uniform_rows(gy0, gy1, rows_per_group), 0.0

        pitch = float(np.median(per_col_pitches))
        if pitch < 5:
            return _uniform_rows(gy0, gy1, rows_per_group), 0.0

        # All detected centres pooled (for row0 estimation and quality)
        flat = sorted(cy for cs in all_centers for cy in cs)
        if not flat:
            return _uniform_rows(gy0, gy1, rows_per_group), 0.0

        row0 = float(min(flat))

        # Refine row0 over ±pitch/2 to minimise total snap error
        best_row0, best_err = row0, 1e9
        for delta in np.arange(-pitch / 2, pitch / 2 + 1, 1.0):
            cand    = row0 + delta
            lattice = [cand + i * pitch for i in range(rows_per_group)]
            err     = sum(min(abs(cy - l) for l in lattice) for cy in flat)
            if err < best_err:
                best_err, best_row0 = err, cand

        lattice = [round(best_row0 + i * pitch) for i in range(rows_per_group)]

        # Quality = fraction snapped within threshold
        snap_thr  = ALGO["lattice_max_snap_px"]
        n_snapped = sum(
            1 for cy in flat
            if min(abs(cy - l) for l in lattice) <= snap_thr
        )
        quality = n_snapped / max(len(flat), 1)
        return lattice, quality

    @staticmethod
    def _snap_to_lattice(centers: List[int], lattice: List[int]) -> List[int]:
        """Snap each detected centre to the nearest lattice point (in-place)."""
        snap_thr = ALGO["lattice_max_snap_px"]
        snapped  = []
        for cy in centers:
            nearest = min(lattice, key=lambda l: abs(l - cy))
            if abs(cy - nearest) <= snap_thr:
                snapped.append(nearest)
            else:
                snapped.append(cy)   # keep original if too far from lattice
        return snapped

    # ── 5D: bubble generation ─────────────────────────────────────────────────

    @staticmethod
    def _generate_bubbles(col: ColumnLayout,
                          row_centers: List[int],
                          col_idx: int,
                          template: TemplateSpec,
                          bubble_r: int,
                          ) -> List[GeneratedBubble]:
        """
        Compute choice X positions and generate bubbles for every (row, choice).

        When the column was refined via bar detection (col.from_bar=True),
        col.x0/x1 are the actual printed bar bounds in warped space, which equal
        the bubble-zone bounds by construction (generate_sheet_v3 draws the bars
        from bzone_x0 to bzone_x1).  Using them directly eliminates the residual
        horizontal drift that appears on edge columns after perspective correction.

        When bars were not detected (col.from_bar=False), fall back to the static
        template formula which is calibrated for perfect-geometry sheets.
        """
        n = template.n_choices
        if col.from_bar and col.x1 > col.x0:
            # Bar bounds == bubble zone.  The sheet uses CSS space-around, so
            # each choice gets an equal share of the width and is centered in it:
            #   share = width / n
            #   center_i = x0 + share * (0.5 + i)
            bx0 = float(col.x0)
            bx1 = float(col.x1)
            share = (bx1 - bx0) / n
            xs = [round(bx0 + share * (0.5 + i)) for i in range(n)]
            if template.rtl:
                xs.reverse()
        else:
            xs = template.bubble_x_positions(col_idx)

        bubbles: List[GeneratedBubble] = []
        for row_idx, cy in enumerate(row_centers[:template.rows_per_col]):
            q_num = template.question_for(col_idx, row_idx)
            if q_num > template.n_questions:
                break
            for rank, cx in enumerate(xs):
                bubbles.append(GeneratedBubble(
                    cx=cx, cy=cy, r=bubble_r,
                    col_idx=col_idx,
                    row_idx=row_idx,
                    choice_rank=rank,
                    question_id=str(q_num),
                    option=template.choice_labels[rank],
                ))
        return bubbles

    # ── 5E: circle-centre snapping ────────────────────────────────────────────

    @staticmethod
    def _snap_to_circles(
        gray: np.ndarray,
        bubbles: List[GeneratedBubble],
        columns: List[ColumnLayout],
        bubble_r: int,
        snap_r: int = 8,
        min_gain: float = 3.0,
    ) -> Tuple[List[GeneratedBubble], int]:
        """Snap each bubble centre to the geometric centre of its printed ring.

        Algorithm
        ---------
        1. Canny edges on the grayscale warped image, with column border pixels
           blanked out so the strong vertical bar edges cannot steal snaps.
        2. Pre-compute the circle perimeter sampling offsets (at radius bubble_r).
        3. For every bubble, evaluate the mean edge response at each candidate
           centre in a (2*snap_r+1)² neighbourhood using vectorised numpy ops.
        4. Candidate must remain inside its own column's x/y bounds (±bubble_r
           tolerance).  Candidates outside are masked to score 0.
        5. Snap only if the best candidate beats the original by ≥ min_gain.

        Works for empty bubbles (dark ink ring → edge) and filled ones
        (outer edge where filled area meets white paper).

        Radius robustness: tests r-2 … r+2 at each candidate position and
        takes the max, so a small error in bubble_r doesn't miss the ring.
        """
        edges = cv2.Canny(gray, 40, 100)

        # Blank a narrow strip centred on each column boundary (the printed bar
        # line itself) without eating into the bubble ring of edge bubbles.
        for col in columns:
            for x in (col.x0, col.x1):
                x0b = max(0, x - 1)
                x1b = min(edges.shape[1], x + 2)
                edges[:, x0b:x1b] = 0

        # Build perimeter offsets for r-2 … r+2 and merge with max-pool so a
        # small radius error still gives a strong response at the true centre.
        def _perim(r: int) -> Tuple[np.ndarray, np.ndarray]:
            n = max(24, r * 4)
            t = np.linspace(0, 2 * np.pi, n, endpoint=False)
            return (np.round(r * np.cos(t)).astype(np.int32),
                    np.round(r * np.sin(t)).astype(np.int32))

        perims = [_perim(max(3, bubble_r + dr)) for dr in range(-2, 3)]

        H, W = edges.shape

        drange = np.arange(-snap_r, snap_r + 1, dtype=np.int32)
        DY, DX = np.meshgrid(drange, drange, indexing='ij')   # (2R+1, 2R+1)
        n_win  = len(drange)

        # Distance-from-template penalty: prefer staying close to the original
        # position.  Penalise 1.5 edge-score units per pixel of displacement.
        DIST_PENALTY = 1.5
        dist_map = np.sqrt((DX.astype(np.float32)) ** 2 +
                           (DY.astype(np.float32)) ** 2) * DIST_PENALTY

        # Build per-column allowed x/y ranges (bubble centre must stay inside)
        col_bounds: Dict[int, Tuple[int, int, int, int]] = {
            col.col_idx: (col.x0 + bubble_r, col.x1 - bubble_r,
                          col.y0,            col.y1)
            for col in columns
        }

        snapped: List[GeneratedBubble] = []
        n_snapped = 0

        for b in bubbles:
            CX = b.cx + DX   # (n_win, n_win)
            CY = b.cy + DY

            # Max-pool over all tested radii → robust to small radius error
            scores = np.zeros((n_win, n_win), dtype=np.float32)
            for px_off, py_off in perims:
                XS = np.clip(
                    CX[:, :, np.newaxis] + px_off[np.newaxis, np.newaxis, :],
                    0, W - 1,
                )
                YS = np.clip(
                    CY[:, :, np.newaxis] + py_off[np.newaxis, np.newaxis, :],
                    0, H - 1,
                )
                scores = np.maximum(scores,
                                    edges[YS, XS].mean(axis=2).astype(np.float32))

            # Mask candidates whose centre falls outside this column's bounds
            bnd = col_bounds.get(b.col_idx)
            if bnd is not None:
                cx_lo, cx_hi, cy_lo, cy_hi = bnd
                outside = (CX < cx_lo) | (CX > cx_hi) | (CY < cy_lo) | (CY > cy_hi)
                scores[outside] = 0.0

            # Apply distance penalty so nearby candidates are preferred
            penalised = scores - dist_map

            orig_score = float(penalised[snap_r, snap_r])   # dist_map centre = 0
            best_flat  = int(np.argmax(penalised))
            bi, bj     = divmod(best_flat, n_win)
            best_score = float(penalised[bi, bj])

            if best_score > orig_score + min_gain and (bi != snap_r or bj != snap_r):
                new_cx = b.cx + int(drange[bj])
                snapped.append(GeneratedBubble(
                    cx=new_cx, cy=b.cy, r=b.r,
                    col_idx=b.col_idx, row_idx=b.row_idx,
                    choice_rank=b.choice_rank,
                    question_id=b.question_id, option=b.option,
                ))
                n_snapped += 1
            else:
                snapped.append(b)

        return snapped, n_snapped

    @staticmethod
    def _regularize_grid(bubbles: List[GeneratedBubble]) -> List[GeneratedBubble]:
        """Force strict X alignment: median X per (col, choice).

        Y is left untouched — it comes directly from the band-detected
        row_centers which are already fixed relative to the bars.
        """
        from collections import defaultdict

        x_groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for b in bubbles:
            x_groups[(b.col_idx, b.choice_rank)].append(b.cx)

        med_x: Dict[Tuple[int, int], int] = {
            k: int(round(float(np.median(vs)))) for k, vs in x_groups.items()
        }

        return [
            GeneratedBubble(
                cx=med_x[(b.col_idx, b.choice_rank)],
                cy=b.cy,
                r=b.r,
                col_idx=b.col_idx, row_idx=b.row_idx,
                choice_rank=b.choice_rank,
                question_id=b.question_id, option=b.option,
            )
            for b in bubbles
        ]

    # ── public ────────────────────────────────────────────────────────────────

    def process(self, layout: LayoutContract,
                template: TemplateSpec,
                debug_dir: pathlib.Path) -> BubbleGridContract:
        warped = layout.aligned_bgr
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # 5A: bars
        bars = self._detect_bars(gray,
                                  layout.grid_x0, layout.grid_y0,
                                  layout.grid_x1, layout.grid_y1,
                                  template.n_cols)
        print(f"  [5-grid] bars detected: {len(bars)}")

        # 5B: refine columns
        columns = self._bars_to_columns(bars, layout.columns)
        avg_col_w  = float(np.mean([c.x1 - c.x0 for c in columns]))

        bubble_r  = max(ALGO["bubble_r"], template.warp_bub_r)

        # 5C: Build bubbles from Playwright-extracted coords (JSON) if available,
        # otherwise fall back to template formula geometry.
        coords_file = pathlib.Path("bubble_coords") / f"{template.template_id}_ar.json"
        if coords_file.exists():
            import json as _json
            cdata = _json.loads(coords_file.read_text(encoding="utf-8"))

            # Per-column X correction: compare JSON ideal col centre
            # with bar-detected col centre in the actual warped image.
            from collections import defaultdict as _ddict
            _col_cxs: Dict[int, List[float]] = _ddict(list)
            for b in cdata["bubbles"]:
                _col_cxs[b["col"]].append(b["cx_frac"] * WARP_W)
            col_dx: Dict[int, int] = {}
            for ci, cxs in _col_cxs.items():
                json_cx = float(np.mean(cxs))
                col = columns[ci] if ci < len(columns) else None
                if col is not None and col.from_bar:
                    det_cx = (col.x0 + col.x1) / 2
                    col_dx[ci] = round(det_cx - json_cx)
                else:
                    col_dx[ci] = 0

            all_bubbles: List[GeneratedBubble] = []
            for b in cdata["bubbles"]:
                cx = round(b["cx_frac"] * WARP_W) + col_dx.get(b["col"], 0)
                cy = round(b["cy_frac"] * WARP_H)
                r_from_json = max(ALGO["bubble_r"], round(b["r_frac"] * WARP_W))
                all_bubbles.append(GeneratedBubble(
                    cx=cx, cy=cy, r=r_from_json,
                    col_idx=b["col"], row_idx=b["row"],
                    choice_rank=b["choice"],
                    question_id=str(b["q"]),
                    option=template.choice_labels[b["choice"]],
                ))
            bubble_r = all_bubbles[0].r if all_bubbles else bubble_r
            source = "playwright-json"
        else:
            tmpl_bubbles = template.all_bubble_positions_warp()
            all_bubbles = []
            for col_idx, row_idx, rank, cx, cy, r in tmpl_bubbles:
                col = columns[col_idx] if col_idx < len(columns) else None
                if col is not None and col.from_bar:
                    tmpl_col_cx = (template.col_x0(col_idx) + template.col_x1(col_idx)) / 2
                    det_col_cx  = (col.x0 + col.x1) / 2
                    cx = cx + round(det_col_cx - tmpl_col_cx)

                q_num = template.question_for(col_idx, row_idx)
                all_bubbles.append(GeneratedBubble(
                    cx=cx, cy=cy, r=bubble_r,
                    col_idx=col_idx, row_idx=row_idx,
                    choice_rank=rank,
                    question_id=str(q_num),
                    option=template.choice_labels[rank],
                ))
            source = "template-formula"

        print(f"  [5-grid] bubbles={len(all_bubbles)} "
              f"(expected={template.n_questions * template.n_choices})  "
              f"bubble_r={bubble_r}  source={source}")

        # 5E: snap each bubble to the geometric centre of its printed ring
        all_bubbles, n_snapped = self._snap_to_circles(
            gray, all_bubbles, columns, bubble_r,
            snap_r=ALGO["circle_snap_r"],
            min_gain=ALGO["circle_snap_min_gain"],
        )
        print(f"  [5-grid] circle-snap: {n_snapped}/{len(all_bubbles)} bubbles moved")

        # 5F: regularize — force strict grid alignment via median per group
        all_bubbles = self._regularize_grid(all_bubbles)
        print(f"  [5-grid] grid regularized (median X per choice-col, median Y per row-col)")

        # Debug overlay
        vis = warped.copy()
        for b in all_bubbles:
            cls_r_vis = max(4, b.r - ALGO["cls_r_shrink"])
            cv2.circle(vis, (b.cx, b.cy), int(cls_r_vis), (0, 200, 80), 1)
        for col in columns:
            cv2.rectangle(vis, (col.x0, col.y0), (col.x1, col.y1),
                          (0, 140, 255), 1)

        # Grid lines through bubble centres for alignment inspection
        from collections import defaultdict
        rows_by_col: Dict[int, List[int]] = defaultdict(list)
        for b in all_bubbles:
            rows_by_col[b.col_idx].append(b.cy)

        overlay = vis.copy()
        grid_color = (255, 160, 0)   # bright blue-ish (BGR)
        gx0 = min(c.x0 for c in columns)
        gx1 = max(c.x1 for c in columns)
        # Horizontal: one per unique Y
        seen_y: set = set()
        for ys in rows_by_col.values():
            for y in ys:
                if y not in seen_y:
                    seen_y.add(y)
                    cv2.line(overlay, (gx0, y), (gx1, y), grid_color, 1,
                             cv2.LINE_AA)
        # Vertical: one per unique X within each column
        xs_per_col: Dict[int, set] = defaultdict(set)
        for b in all_bubbles:
            xs_per_col[b.col_idx].add(b.cx)
        for ci, xs in xs_per_col.items():
            col = columns[ci]
            for x in xs:
                cv2.line(overlay, (x, col.y0), (x, col.y1), grid_color, 1,
                         cv2.LINE_AA)
        # Blend at 30% opacity so bubbles/sheet stay clearly visible
        cv2.addWeighted(overlay, 0.30, vis, 0.70, 0, vis)

        _dbg(vis, "06_grid.jpg", debug_dir)

        return BubbleGridContract(
            aligned_bgr=warped,
            bubbles=all_bubbles,
            grid_x0=layout.grid_x0, grid_y0=layout.grid_y0,
            grid_x1=layout.grid_x1, grid_y1=layout.grid_y1,
            lattice_quality=1.0,
            bubble_r=bubble_r,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 6 — BUBBLE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class BubbleClassifier:
    """
    CNN-first, fill-ratio fallback.

    New in v3:
      • Each crop is illumination-normalised (divide by large Gaussian blur)
        before classification to suppress shadows and gradients.
      • Subpixel crop via cv2.getRectSubPix.
      • Per-bubble confidence returned from CNN softmax or fill-ratio model.
    """

    _CROP_SIZE = 32

    def __init__(self):
        self._session  = None   # ONNX
        self._torch_fn = None   # PyTorch
        self._type     = "fill-ratio"
        self._load_model()

    def _load_model(self) -> None:
        # Try ONNX Runtime first
        try:
            import onnxruntime as ort
            onnx_path = pathlib.Path(ALGO["onnx_path"])
            if onnx_path.exists():
                so = ort.SessionOptions()
                so.log_severity_level = 3
                self._session = ort.InferenceSession(str(onnx_path), so)
                self._type    = "onnx-cnn"
                print(f"  [6-classifier] ONNX model loaded: {onnx_path}")
                return
        except ImportError:
            pass

        # Try PyTorch
        try:
            import torch
            pt_path = pathlib.Path(ALGO["pt_path"])
            if pt_path.exists():
                model = torch.jit.load(str(pt_path), map_location="cpu")
                model.eval()
                self._torch_fn = model
                self._type     = "pytorch-cnn"
                print(f"  [6-classifier] PyTorch model loaded: {pt_path}")
                return
        except ImportError:
            pass

        print("  [6-classifier] No CNN model found — using fill-ratio fallback")

    # ── illumination normalisation (full-image) ──────────────────────────────

    @staticmethod
    def _illum_normalise_full(gray: np.ndarray) -> np.ndarray:
        """
        Divide the full warped image by a large Gaussian blur to remove
        illumination gradients (shadows, scanner vignetting).
        Applied once to the whole image before crop extraction.
        """
        bg   = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=80)
        bg   = np.maximum(bg, 30.0)   # prevent near-zero division
        norm = np.clip(
            gray.astype(np.float32) / bg * 180.0, 0, 255
        ).astype(np.uint8)
        return norm

    # ── crop extraction ───────────────────────────────────────────────────────

    def _get_crop(self, gray: np.ndarray, b: GeneratedBubble) -> np.ndarray:
        """Subpixel-accurate crop centred at (cx, cy), resized to CROP_SIZE."""
        half = b.r + 4
        sz   = half * 2
        crop = cv2.getRectSubPix(gray.astype(np.float32),
                                  (sz, sz), (float(b.cx), float(b.cy)))
        crop = np.clip(crop, 0, 255).astype(np.uint8)
        crop = cv2.resize(crop, (self._CROP_SIZE, self._CROP_SIZE),
                          interpolation=cv2.INTER_AREA)
        return crop

    # ── fill-ratio fallback ───────────────────────────────────────────────────

    @staticmethod
    def _fill_ratio(crop: np.ndarray, r: int) -> float:
        H, W    = crop.shape
        mask    = np.zeros((H, W), np.uint8)
        cr      = min(r - 1, H // 2 - 1, W // 2 - 1, r)
        cv2.circle(mask, (W // 2, H // 2), max(cr, 4), 255, -1)
        pixels  = crop[mask > 0]
        if len(pixels) == 0:
            return 0.0
        dark    = np.sum(pixels < 140)
        return float(dark) / len(pixels)

    @staticmethod
    def _fill_ratio_direct(gray: np.ndarray, cx: int, cy: int, r: int,
                            snap_r: int = 5) -> float:
        """
        V2-compatible fill ratio using a direct integer-indexed slice.
        Expects a background-corrected normalized image (empty→255, filled→~0-100).
        Threshold: 185 (normalized scale), matching V2's fill_norm_thr.
        snap_r: scan ±snap_r in X around cx to find actual bubble peak center.
        """
        def _at(cx_: int) -> float:
            r0 = max(0, cy - r)
            r1 = min(gray.shape[0], cy + r)
            c0 = max(0, cx_ - r)
            c1 = min(gray.shape[1], cx_ + r)
            crop = gray[r0:r1, c0:c1]
            if crop.size == 0:
                return 0.0
            H, W = crop.shape
            mask = np.zeros((H, W), np.uint8)
            cr = min(r - 1, H // 2 - 1, W // 2 - 1, r)
            cv2.circle(mask, (W // 2, H // 2), max(cr, 4), 255, -1)
            pixels = crop[mask > 0]
            if len(pixels) == 0:
                return 0.0
            return float(np.sum(pixels < ALGO["fill_norm_thr"])) / len(pixels)

        best = 0.0
        for dx in range(-snap_r, snap_r + 1):
            best = max(best, _at(cx + dx))
        return best

    def _classify_fill_ratio(self, crop: np.ndarray, r: int,
                              ) -> Tuple[str, float]:
        fr     = self._fill_ratio(crop, r)
        thr    = ALGO["fill_ratio_thr"]
        lo     = ALGO["fill_ratio_ambi_lo"]
        hi     = ALGO["fill_ratio_ambi_hi"]
        if fr > thr:
            conf = min(1.0, (fr - thr) / (1.0 - thr + 1e-6) + 0.5)
            return "filled", float(np.clip(conf, 0.5, 1.0))
        if lo <= fr <= hi:
            conf = 1.0 - abs(fr - (lo + hi) / 2) / ((hi - lo) / 2)
            return "ambiguous", float(np.clip(conf, 0.4, 0.8))
        conf = min(1.0, (thr - fr) / (thr - lo + 1e-6) + 0.5)
        return "empty", float(np.clip(conf, 0.5, 1.0))

    # ── CNN classify ──────────────────────────────────────────────────────────

    def _classify_cnn(self, crop: np.ndarray) -> Tuple[str, float]:
        LABELS = ["empty", "filled", "ambiguous"]
        inp    = crop.astype(np.float32) / 255.0
        inp    = inp[np.newaxis, np.newaxis, :, :]   # (1,1,H,W)

        if self._session is not None:
            name   = self._session.get_inputs()[0].name
            logits = self._session.run(None, {name: inp})[0][0]
        else:
            import torch
            with torch.no_grad():
                out    = self._torch_fn(torch.tensor(inp))
                logits = out.numpy()[0]

        probs = np.exp(logits) / np.sum(np.exp(logits))
        idx   = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])

    # ── public ────────────────────────────────────────────────────────────────

    def process(self, grid: BubbleGridContract,
                debug_dir: pathlib.Path) -> ClassificationContract:
        warped = grid.aligned_bgr
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # V2-compatible shading-corrected normalization:
        # bg = medianBlur(71), then norm = gray / bg * 255.
        # This makes empty paper → ~255 and filled ink → ~0-100,
        # giving reliable fill_ratio with threshold 185.
        bg_blur   = cv2.medianBlur(gray, 71)
        bg_blur   = np.maximum(bg_blur, 1)
        gray_norm = np.clip(
            gray.astype(np.float32) / bg_blur.astype(np.float32) * 255,
            0, 255,
        ).astype(np.uint8)
        preds: List[BubblePrediction] = []
        use_cnn = self._session is not None or self._torch_fn is not None

        for b in grid.bubbles:
            cls_r = max(4, b.r - ALGO["cls_r_shrink"])

            fr = self._fill_ratio_direct(gray_norm, b.cx, b.cy, cls_r)

            if use_cnn:
                cls_b = GeneratedBubble(
                    cx=b.cx, cy=b.cy, r=cls_r,
                    col_idx=b.col_idx, row_idx=b.row_idx,
                    choice_rank=b.choice_rank,
                    question_id=b.question_id, option=b.option,
                )
                crop = self._get_crop(gray_norm, cls_b)
                status, conf = self._classify_cnn(crop)
                # When CNN is uncertain (ambiguous), use fill-ratio to decide.
                # This bridges the synthetic→real domain gap for marginal marks.
                if status == "ambiguous":
                    thr = ALGO["fill_ratio_thr"]
                    lo  = ALGO["fill_ratio_ambi_lo"]
                    if fr > thr:
                        conf   = float(np.clip(
                            (fr - thr) / (1.0 - thr + 1e-6) + 0.5, 0.5, 1.0))
                        status = "filled"
                    elif fr < lo:
                        conf   = float(np.clip(
                            1.0 - fr / (lo + 1e-6), 0.5, 1.0))
                        status = "empty"
                    # else keep "ambiguous" — genuinely uncertain in both
            else:
                # Threshold-based on the direct fill ratio
                thr = ALGO["fill_ratio_thr"]
                lo  = ALGO["fill_ratio_ambi_lo"]
                hi  = ALGO["fill_ratio_ambi_hi"]
                if fr > thr:
                    conf   = float(np.clip((fr - thr) / (1.0 - thr + 1e-6) + 0.5, 0.5, 1.0))
                    status = "filled"
                elif lo <= fr <= hi:
                    conf   = float(np.clip(1.0 - abs(fr - (lo + hi) / 2) / ((hi - lo) / 2 + 1e-6), 0.0, 1.0))
                    status = "ambiguous"
                else:
                    conf   = float(np.clip(1.0 - fr / (lo + 1e-6), 0.0, 1.0))
                    status = "empty"

            preds.append(BubblePrediction(
                question_id=b.question_id,
                option=b.option,
                status=status,
                fill_ratio=fr,
                confidence=conf,
            ))

        # ── Per-question relative-fill normalisation (V2's REL_RATIO filter) ──
        # Within each question's row, only keep a bubble "filled" if its fill
        # ratio is ≥ REL_RATIO × the row's maximum fill.
        # This eliminates ink-bleed false positives adjacent to a heavy mark.
        REL_RATIO = 0.65
        by_q: dict = {}
        for i, p in enumerate(preds):
            by_q.setdefault(p.question_id, []).append((i, p))
        for _qpairs in by_q.values():
            max_fr = max(p.fill_ratio for _, p in _qpairs)
            for i, p in _qpairs:
                if p.status == "filled" and p.fill_ratio < REL_RATIO * max_fr:
                    preds[i] = BubblePrediction(
                        question_id=p.question_id,
                        option=p.option,
                        status="empty",
                        fill_ratio=p.fill_ratio,
                        confidence=float(np.clip(1.0 - p.fill_ratio, 0.0, 1.0)),
                    )

        confs = [p.confidence for p in preds]
        cls_conf = float(np.mean(confs)) if confs else 0.0

        # Debug overlay
        vis = warped.copy()
        for b, p in zip(grid.bubbles, preds):
            c = {"filled": (0, 200, 80),
                 "ambiguous": (0, 165, 255),
                 "empty": (180, 180, 180)}[p.status]
            cv2.circle(vis, (b.cx, b.cy), b.r, c, -1 if p.status == "filled" else 1)
            if p.status == "ambiguous":
                cv2.putText(vis, "?", (b.cx - 3, b.cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, c, 1)
        _dbg(vis, "08_classified.jpg", debug_dir)

        print(f"  [6-classifier] type={self._type}  "
              f"mean_conf={cls_conf:.3f}  "
              f"filled={sum(1 for p in preds if p.status=='filled')}  "
              f"ambig={sum(1 for p in preds if p.status=='ambiguous')}")
        return ClassificationContract(
            predictions=preds, bubbles=grid.bubbles,
            aligned_bgr=warped,
            classifier_type=self._type,
            classification_confidence=cls_conf,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 7 — ANSWER LOGIC + CROSS-ROW CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class AnswerLogicEngine:
    """
    Aggregate per-bubble predictions → per-question answers.
    Cross-row consistency: flag sheets with suspiciously long empty runs.
    """

    def process(self, cls: ClassificationContract,
                template: TemplateSpec) -> ValidationContract:
        n_q   = template.n_questions
        n_ch  = template.n_choices
        preds = cls.predictions

        # Group predictions by question
        q_map: Dict[str, List[BubblePrediction]] = {}
        for p in preds:
            q_map.setdefault(p.question_id, []).append(p)

        answers:   Dict[str, Optional[str]] = {}
        details:   Dict[str, QuestionResult] = {}

        for q in range(1, n_q + 1):
            qstr  = str(q)
            qpred = q_map.get(qstr, [])
            if not qpred:
                answers[qstr] = None
                details[qstr] = QuestionResult(
                    choice=None, note="blank", all_filled=[],
                    fill=0.0, classifier_type=cls.classifier_type,
                    confidence=0.0,
                )
                continue

            filled    = [p for p in qpred if p.status == "filled"]
            ambiguous = [p for p in qpred if p.status == "ambiguous"]
            all_f     = [p.option for p in filled]
            avg_fill  = float(np.mean([p.fill_ratio for p in qpred]))
            avg_conf  = float(np.mean([p.confidence for p in qpred]))

            if len(filled) == 1:
                choice, note = filled[0].option, "ok"
            elif len(filled) == 0 and len(ambiguous) == 1:
                choice, note = ambiguous[0].option, "ambiguous"
            elif len(filled) == 0:
                choice, note = None, "blank"
            else:
                # Double / triple mark: pick highest fill_ratio
                best   = max(filled, key=lambda p: p.fill_ratio)
                choice = best.option
                note   = "double_mark"

            answers[qstr] = choice
            details[qstr] = QuestionResult(
                choice=choice, note=note, all_filled=all_f,
                fill=avg_fill, classifier_type=cls.classifier_type,
                confidence=avg_conf,
            )

        unanswered   = [qstr for qstr, a in answers.items() if a is None]
        double_marked = [qstr for qstr, d in details.items()
                         if d.note == "double_mark"]
        ambig_list   = [qstr for qstr, d in details.items()
                        if d.note == "ambiguous"]

        # Cross-row consistency check
        low_conf_rows = self._cross_row_check(details, n_q)

        return ValidationContract(
            answers=answers, details=details,
            unanswered=unanswered, double_marked=double_marked,
            ambiguous=ambig_list, low_conf_rows=low_conf_rows,
        )

    @staticmethod
    def _cross_row_check(details: Dict[str, QuestionResult],
                         n_q: int) -> List[str]:
        """Flag sheets with suspicious patterns (>N consecutive blank rows)."""
        max_consec = ALGO["max_consec_empty_rows"]
        run = 0
        suspect: List[str] = []
        for q in range(1, n_q + 1):
            qstr = str(q)
            d    = details.get(qstr)
            if d and d.choice is None:
                run += 1
                if run >= max_consec:
                    suspect.append(f"blank_run_ends_at_Q{q}")
            else:
                run = 0
        return suspect


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 8 — STORAGE
# ─────────────────────────────────────────────────────────────────────────────

class StorageEngine:

    def process(self,
                validation: ValidationContract,
                cls: ClassificationContract,
                layout: LayoutContract,
                align: AlignmentContract,
                template: TemplateSpec,
                image_path: str,
                debug_dir: pathlib.Path,
                start_time: float,
                ) -> FinalResultContract:

        n_q     = template.n_questions
        elapsed = int((time.time() - start_time) * 1000)
        answered = n_q - len(validation.unanswered)

        # Annotated answer overlay
        vis = layout.aligned_bgr.copy()
        pred_lut: Dict[Tuple[str, str], BubblePrediction] = {
            (p.question_id, p.option): p for p in cls.predictions
        }
        for b in cls.bubbles:
            p  = pred_lut.get((b.question_id, b.option))
            d  = validation.details.get(b.question_id)
            if p is None or d is None:
                continue
            chosen = d.choice == b.option
            double = d.note == "double_mark" and b.option in d.all_filled
            ambig  = p.status == "ambiguous"
            if chosen or double:
                color = (0, 0, 255) if double else (0, 220, 0)
                cv2.circle(vis, (b.cx, b.cy), b.r + 4, color, 2)
                cv2.circle(vis, (b.cx, b.cy), b.r, (0, 0, 0), -1)
                cv2.putText(vis, b.option, (b.cx - 5, b.cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
            elif ambig:
                cv2.circle(vis, (b.cx, b.cy), b.r + 2, (0, 200, 255), 2)
                cv2.putText(vis, "?", (b.cx - 4, b.cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 255), 1)
            else:
                cv2.circle(vis, (b.cx, b.cy), b.r, (180, 180, 180), 1)
        _dbg(vis, "10_answered.jpg", debug_dir)

        # Per-question confidence for output
        q_confs = {
            qstr: round(d.confidence, 3)
            for qstr, d in validation.details.items()
        }

        confidence_metrics: Dict[str, Any] = {
            "template_id":             template.template_id,
            "alignment_score":         round(align.alignment_confidence, 3),
            "alignment_method":        align.align_method,
            "classification_confidence": round(cls.classification_confidence, 3),
            "classifier_type":         cls.classifier_type,
            "lattice_quality":         round(cls.aligned_bgr.shape[0], 3),  # placeholder
            "ambiguous_count":         len(validation.ambiguous),
            "low_conf_rows":           len(validation.low_conf_rows),
            "per_question_confidence": q_confs,
        }

        answer_details: Dict[str, Any] = {}
        for qstr, qr in validation.details.items():
            answer_details[qstr] = {
                "choice":          qr.choice,
                "note":            qr.note,
                "all_filled":      qr.all_filled,
                "fill":            round(qr.fill, 4),
                "confidence":      round(qr.confidence, 3),
                "classifier_type": qr.classifier_type,
            }

        result = FinalResultContract(
            image=image_path,
            student_code=layout.student_code,
            exam_code=layout.exam_code,
            qr_codes_raw=layout.qr_raw,
            align_method=align.align_method,
            answers={str(q): validation.answers.get(str(q))
                     for q in range(1, n_q + 1)},
            answer_details=answer_details,
            total_questions=n_q,
            answered=answered,
            unanswered=validation.unanswered,
            double_marked=validation.double_marked,
            ambiguous=validation.ambiguous,
            low_conf_rows=validation.low_conf_rows,
            valid=(len(validation.unanswered) == 0
                   and len(validation.double_marked) == 0),
            debug_dir=str(debug_dir),
            confidence_metrics=confidence_metrics,
            processing_metrics={"time_ms": elapsed},
        )

        out_dict = {k: getattr(result, k) for k in result.__dataclass_fields__}
        out_path = debug_dir / "result_v3.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=2)
        print(f"  [8-storage] JSON -> {out_path}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class OMRPipelineV3:
    """
    Stateless orchestrator — wires the 8 stages together.
    Template is resolved once per call and passed through the chain.
    No global mutable state.
    """

    def __init__(self):
        self.s1 = InputLayer()
        self.s2 = PreprocessingEngine()
        self.s3 = AlignmentEngine()
        self.s4 = StaticLayoutEngine()
        self.s5 = BubbleGridEngine()
        self.s6 = BubbleClassifier()
        self.s7 = AnswerLogicEngine()
        self.s8 = StorageEngine()

    def run(self, image_path: str,
            template: TemplateSpec,
            debug_dir: Optional[pathlib.Path] = None,
            white_balance: bool = True,
            denoise: bool = True,
            ) -> FinalResultContract:

        stem      = pathlib.Path(image_path).stem
        debug_dir = debug_dir or pathlib.Path(f"detect_v3_{stem}")
        debug_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*64}")
        print(f"  OMR Pipeline v3   {image_path}")
        print(f"  template={template.template_id}  "
              f"Q={template.n_questions}  cols={template.n_cols}  "
              f"choices={template.n_choices}  rtl={template.rtl}")
        print(f"  debug: {debug_dir}/")
        print(f"{'='*64}")
        start = time.time()

        try:
            print("[1] Input Layer ...")
            inp = self.s1.process(image_path, debug_dir)

            print("[2] Preprocessing ...")
            prep = self.s2.process(inp, white_balance=white_balance,
                                   denoise=denoise)
            if prep.quality_score < 0.05:
                print("  [WARN] Very low quality score")

            print("[3] Alignment ...")
            align = self.s3.process(prep, debug_dir)
            if align.alignment_confidence < 0.5:
                print("  [WARN] Low alignment confidence — results may be inaccurate")

            print("[4] Static Layout ...")
            layout = self.s4.process(align, template, debug_dir)

            print("[5] Bubble Grid (lattice fitting) ...")
            grid = self.s5.process(layout, template, debug_dir)

            print("[6] Bubble Classifier ...")
            cls_result = self.s6.process(grid, debug_dir)

            print("[7] Answer Logic + Cross-row check ...")
            validation = self.s7.process(cls_result, template)

            print("[8] Storage ...")
            result = self.s8.process(
                validation, cls_result, layout, align,
                template, image_path, debug_dir, start,
            )

        except Exception as exc:
            elapsed_ms = int((time.time() - start) * 1000)
            err_path   = debug_dir / "result_v3_error.json"
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump({"image": image_path, "status": "ERROR",
                           "error": str(exc),
                           "processing_metrics": {"time_ms": elapsed_ms}},
                          f, indent=2)
            print(f"\n[FAIL] {exc}")
            raise

        elapsed = result.processing_metrics["time_ms"]
        print(f"\n[OK] Done in {elapsed} ms")
        print(f"     template:       {template.template_id}")
        print(f"     answered:       {result.answered} / {result.total_questions}")
        print(f"     unanswered:     {result.unanswered[:10]}"
              f"{'...' if len(result.unanswered) > 10 else ''}")
        print(f"     double_marked:  {result.double_marked}")
        print(f"     ambiguous:      {result.ambiguous[:10]}"
              f"{'...' if len(result.ambiguous) > 10 else ''}")
        print(f"     low_conf_rows:  {result.low_conf_rows}")
        print(f"     valid:          {result.valid}")
        print(f"     align_method:   {result.align_method}")
        print(f"     cls_type:       "
              f"{result.confidence_metrics['classifier_type']}")
        print(f"     JSON:           {debug_dir}/result_v3.json")
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="OMR Detector Enhanced v3 — template-driven, global lattice"
    )
    ap.add_argument("image",
                    help="Input answer sheet (JPEG/PNG)")
    ap.add_argument("--questions", type=int, default=100,
                    help="Number of questions (default: 100)")
    ap.add_argument("--choices",   type=int, default=5,
                    help="Choices per question (default: 5)")
    ap.add_argument("--template",  type=str, default=None,
                    help="Template ID, e.g. Q100_5ch (overrides --questions/--choices)")
    ap.add_argument("--debug-dir", type=str, default=None,
                    help="Override debug output directory")
    ap.add_argument("--no-wb",     action="store_true",
                    help="Disable grey-world white balance")
    ap.add_argument("--no-denoise", action="store_true",
                    help="Disable bilateral denoising")
    args = ap.parse_args()

    if args.template:
        template = get_template(args.template)
    else:
        template = infer_template(args.questions, args.choices)

    pipeline   = OMRPipelineV3()
    debug_path = pathlib.Path(args.debug_dir) if args.debug_dir else None

    result = pipeline.run(
        args.image,
        template=template,
        debug_dir=debug_path,
        white_balance=not args.no_wb,
        denoise=not args.no_denoise,
    )

    print("\n--- Answer Table ---")
    for row in range(0, template.n_questions, 10):
        qs = range(row + 1, min(row + 11, template.n_questions + 1))
        line = "  " + "  ".join(
            f"Q{q:3d}:{result.answers.get(str(q)) or '-'}" for q in qs
        )
        print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()

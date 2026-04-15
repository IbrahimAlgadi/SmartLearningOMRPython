#!/usr/bin/env python3
"""
omr_detector_enhanced_v2.py
────────────────────────────
OMR Detector Enhanced v2 — Geometry-First Architecture

Core philosophy: KNOW where everything is (geometry-driven), not DETECT everything.
ML is used only where deterministic methods cannot work: bubble fill classification.

Architecture (8 stages):
  1. InputLayer              — load & normalise raw image
  2. PreprocessingEngine     — CLAHE + optional white-balance + bilateral denoise
  3. AlignmentEngine         — anchor-first → Canny-contour → full-image (NO YOLO)
  4. StaticLayoutEngine      — geometry-only layout from warped anchor coords
  5. BubbleGridEngine        — bar detection → column segmentation → row projection
                               → deterministic bubble coordinate generation (NO detection)
  6. BubbleClassifier        — CNN 32×32 crop → [empty|filled|ambiguous]
                               (graceful fill-ratio fallback when no model is present)
  7. AnswerLogicEngine       — simplified 3-rule decision (no multi-tier heuristics)
  8. StorageEngine           — JSON + debug images + confidence metrics

Performance gains over v1/v1.5:
  • 2–4× faster   — no HoughCircles, no K-means, no YOLO inference
  • Zero over-detection — coordinates are generated, not detected
  • Deterministic — same image always produces same result
  • Easy to debug — every stage has a clear, single responsibility

Usage:
  python omr_detector_enhanced_v2.py image.jpg
  python omr_detector_enhanced_v2.py image.jpg --questions 100
  python omr_detector_enhanced_v2.py image.jpg --debug-dir my_debug/
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

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CFG: Dict[str, Any] = {
    # Perspective-warp output (750 × 1060 → 1 px ≈ 0.28 mm on A4)
    "warp_w": 750,
    "warp_h": 1060,

    # Corner anchors: 12 mm side, CSS inset 4 mm → centre at 10 mm from edge
    "anchor_cx_left":   int(10 / 210 * 750),           # ≈ 36 px
    "anchor_cx_right":  int((210 - 10) / 210 * 750),   # ≈ 714 px
    "anchor_cy_top":    int(10 / 297 * 1060),           # ≈ 36 px
    "anchor_cy_bottom": int((297 - 10) / 297 * 1060),  # ≈ 1024 px
    "anchor_search_r":  55,
    "anchor_min_area":  400,
    "anchor_max_area":  6000,

    # Static layout contract — answer grid bounds in the 750×1060 warped image
    # These match the generate_sheet_v3 geometry and are refined by detected anchors.
    "grid_x0_frac": 0.085,  # left edge relative to warp_w
    "grid_x1_frac": 0.915,  # right edge relative to warp_w
    "grid_y0_frac": 0.205,  # top edge relative to (anchor_span) from top anchor
    "grid_y1_frac": 0.975,  # bottom edge relative to (anchor_span) from top anchor

    # QR region — top-left area (RTL sheet, QR codes are on the left)
    "qr_x0": 0.08, "qr_x1": 0.42,
    "qr_y0": 0.08, "qr_y1": 0.25,

    # Sheet layout
    "n_col_groups":   4,
    "rows_per_group": 25,
    "n_questions":    100,
    "n_choices":      5,   # bubbles per row (A B C D E / أ ب ج د هـ)

    # Bubble radius in warped image (px) — used for crop and draw
    "bubble_r":   9,

    # Row projection — valley detection sensitivity
    "row_min_gap_px":    8,    # minimum y-gap between two consecutive row valleys
    "row_smooth_sigma":  3,    # Gaussian sigma for smoothing projection profile
    "row_dark_thr":      80,   # pixel < this counts as dark for projection

    # Bar detection via projection profile
    "bar_dark_thr":     80,    # pixel < this counts as "dark" for bar ink
    "bar_min_width_px": 60,    # minimum horizontal run to be considered a bar

    # Fill ratio fallback (used when CNN model is absent)
    "fill_norm_thr":  185,   # normalised pixel < this = dark
    "fill_ratio_thr": 0.35,

    # CNN classifier thresholds
    "cnn_filled_conf":    0.70,   # filled_prob > this → filled
    "cnn_ambiguous_conf": 0.50,   # ambiguous_prob > this → ambiguous (else empty)

    # Preprocessing
    "bilateral_d":         9,
    "bilateral_sigma_col": 75,
    "bilateral_sigma_sp":  75,
}

RANK_TO_CHOICE: Dict[int, str] = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
CHOICE_COLORS: Dict[Optional[str], Tuple[int, int, int]] = {
    "A": (0, 220, 80), "B": (255, 165, 0), "C": (0, 200, 255),
    "D": (200, 0, 255), "E": (0, 80, 255), None: (120, 120, 120),
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class InputContract:
    image_path: str
    image: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class PreprocessContract:
    original_bgr: np.ndarray
    enhanced_gray: np.ndarray
    quality_score: float


@dataclass
class AlignmentContract:
    aligned_bgr: np.ndarray
    align_method: str
    alignment_confidence: float


@dataclass
class ColumnLayout:
    """Geometry of a single answer column (bubble zone only)."""
    col_idx: int          # 0 = leftmost
    x0: int               # left edge of bubble zone (px, warped coords)
    x1: int               # right edge of bubble zone (bar right edge or col edge)
    y0: int               # top y of bubble zone
    y1: int               # bottom y of bubble zone


@dataclass
class LayoutContract:
    aligned_bgr: np.ndarray
    columns: List[ColumnLayout]
    anchors: Dict[str, Tuple[int, int]]
    student_code: str
    exam_code: str
    qr_raw: List[str]
    qr_method: str
    layout_confidence: float
    grid_x0: int
    grid_y0: int
    grid_x1: int
    grid_y1: int


@dataclass
class GeneratedBubble:
    """A bubble whose position is computed from geometry, NOT detected."""
    cx: int
    cy: int
    r: int
    col_idx: int
    row_idx: int
    choice_rank: int   # 0=rightmost=A, 1=B, …
    question_id: str
    option: str


@dataclass
class BubbleGridContract:
    aligned_bgr: np.ndarray
    bubbles: List[GeneratedBubble]
    columns: List[ColumnLayout]
    grid_confidence: float


@dataclass
class BubblePrediction:
    question_id: str
    option: str
    status: str          # "filled" | "empty" | "ambiguous"
    filled_conf: float
    ambiguous_conf: float
    empty_conf: float
    fill_ratio: float    # raw shading-corrected ratio (always computed for debug)


@dataclass
class ClassificationContract:
    aligned_bgr: np.ndarray
    predictions: List[BubblePrediction]
    bubbles: List[GeneratedBubble]
    classification_confidence: float
    classifier_type: str   # "cnn" | "fill_ratio"


@dataclass
class QuestionResult:
    question_id: str
    choice: Optional[str]
    note: str              # "" | "unanswered" | "double_mark" | "ambiguous_review"
    all_filled: List[str]  # all options flagged as filled (double-mark context)
    fill: float
    classifier_type: str


@dataclass
class ValidationContract:
    answers: Dict[str, Optional[str]]
    details: Dict[str, QuestionResult]
    unanswered: List[int]
    double_marked: List[int]
    ambiguous: List[int]


@dataclass
class FinalResultContract:
    image: str
    student_code: str
    exam_code: str
    qr_codes_raw: List[str]
    align_method: str
    answers: Dict[str, Optional[str]]
    answer_details: Dict[str, Any]
    total_questions: int
    answered: int
    unanswered: List[int]
    double_marked: List[int]
    ambiguous: List[int]
    valid: bool
    debug_dir: str
    confidence_metrics: Dict[str, Any]
    processing_metrics: Dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED DEBUG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _dbg(img: np.ndarray, name: str, debug_dir: pathlib.Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / name), img)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — INPUT LAYER
# ══════════════════════════════════════════════════════════════════════════════

class InputLayer:
    """Accept any image source (file path) and normalise to InputContract."""

    def process(self, image_path: str, debug_dir: pathlib.Path) -> InputContract:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path!r}")

        _dbg(img, "00_original.jpg", debug_dir)
        h, w = img.shape[:2]
        print(f"  [1-input] size={w}x{h}  path={image_path}")

        return InputContract(
            image_path=image_path,
            image=img,
            metadata={"source": "file", "resolution": (w, h), "timestamp": time.time()},
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — PREPROCESSING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class PreprocessingEngine:
    """
    CLAHE → optional white-balance → bilateral denoise.
    White-balance normalisation lifts shadows caused by ambient light variation.
    Bilateral filter preserves bubble edges while suppressing JPEG noise.
    """

    def process(self, data: InputContract,
                white_balance: bool = True,
                denoise: bool = True) -> PreprocessContract:
        img = data.image

        # Optional white-balance (Grey World assumption)
        if white_balance:
            img = self._grey_world(img)

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral denoise on the enhanced grey — preserves circle edges
        if denoise:
            enhanced = cv2.bilateralFilter(
                enhanced,
                CFG["bilateral_d"],
                CFG["bilateral_sigma_col"],
                CFG["bilateral_sigma_sp"],
            )

        lap     = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality = float(np.clip(lap / 500.0, 0.0, 1.0))

        print(f"  [2-preprocess] quality_score={quality:.3f}  "
              f"wb={white_balance}  denoise={denoise}")
        return PreprocessContract(
            original_bgr=data.image,
            enhanced_gray=enhanced,
            quality_score=quality,
        )

    @staticmethod
    def _grey_world(img: np.ndarray) -> np.ndarray:
        """Grey-world white balance: rescale each channel to a common mean."""
        result = img.astype(np.float32)
        means  = [result[:, :, c].mean() for c in range(3)]
        global_mean = float(np.mean(means))
        for c in range(3):
            if means[c] > 0:
                result[:, :, c] = np.clip(result[:, :, c] * (global_mean / means[c]), 0, 255)
        return result.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — ALIGNMENT ENGINE  (NO YOLO)
# ══════════════════════════════════════════════════════════════════════════════

class AlignmentEngine:
    """
    Geometry-only perspective alignment.  Priority:
      1. Anchor-based  — find the 4 OMR black anchor squares globally (most robust)
      2. Canny-contour — largest 4-point hull in edge image (works with dark borders)
      3. Full-image    — 10 px inset last resort

    YOLO is intentionally absent.  If your environment has a doc_model.pt and you
    want to try it, uncomment the _detect_paper_yolo stub and call it before step 1.
    """

    # ── geometry helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _order_pts(pts: np.ndarray) -> np.ndarray:
        """Return [TL, TR, BR, BL] order."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # TL (smallest x+y)
        rect[2] = pts[np.argmax(s)]   # BR (largest  x+y)
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]   # TR (smallest x-y)
        rect[3] = pts[np.argmax(d)]   # BL (largest  x-y)
        return rect

    @staticmethod
    def _warp(img: np.ndarray, rect: np.ndarray,
              out_w: int, out_h: int) -> np.ndarray:
        dst = np.float32([[0, 0], [out_w - 1, 0],
                          [out_w - 1, out_h - 1], [0, out_h - 1]])
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (out_w, out_h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

    # ── method 1: anchor-based ───────────────────────────────────────────────

    def _find_anchors_global(
        self, img: np.ndarray
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Detect the 4 black anchor squares anywhere in the image.
        Uses adaptive thresholding so it handles varying brightness.
        """
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw   = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 51, 15)
        cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        min_side = min(W, H) * 0.012
        max_side = min(W, H) * 0.15
        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if not (min_side ** 2 < area < max_side ** 2):
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
            cx = int(Mv["m10"] / Mv["m00"])
            cy = int(Mv["m01"] / Mv["m00"])
            candidates.append((cx, cy, area))

        if len(candidates) < 4:
            return None

        candidates.sort(key=lambda x: x[2], reverse=True)
        best_quad, max_area = None, 0

        for quad in itertools.combinations(candidates[:12], 4):
            pts     = np.float32([[p[0], p[1]] for p in quad])
            ordered = self._order_pts(pts)
            qa      = cv2.contourArea(ordered)
            if qa <= max_area:
                continue
            tl, tr, br, bl = ordered
            wt = np.linalg.norm(tr - tl);  wb = np.linalg.norm(br - bl)
            hl = np.linalg.norm(bl - tl);  hr = np.linalg.norm(br - tr)
            if (min(wt, wb) / (max(wt, wb) + 1e-6) > 0.70 and
                    min(hl, hr) / (max(hl, hr) + 1e-6) > 0.70):
                max_area  = qa
                best_quad = ordered

        if best_quad is not None and max_area > W * H * 0.05:
            return {
                "TL": tuple(best_quad[0].astype(int)),
                "TR": tuple(best_quad[1].astype(int)),
                "BR": tuple(best_quad[2].astype(int)),
                "BL": tuple(best_quad[3].astype(int)),
            }
        return None

    # ── method 2: Canny contour ───────────────────────────────────────────────

    def _find_document_corners(
        self, gray: np.ndarray
    ) -> Optional[np.ndarray]:
        H, W = gray.shape
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        for lo, hi in [(50, 150), (30, 100), (10, 60)]:
            edges = cv2.Canny(blur, lo, hi)
            k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, k, iterations=1)
            cnts, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts[:10]:
                if cv2.contourArea(c) < 0.35 * H * W:
                    continue
                peri  = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    return approx.reshape(4, 2).astype(np.float32)
        return None

    # ── public entry point ────────────────────────────────────────────────────

    def process(self, prep: PreprocessContract,
                debug_dir: pathlib.Path) -> AlignmentContract:
        img  = prep.original_bgr
        gray = prep.enhanced_gray
        W, H = CFG["warp_w"], CFG["warp_h"]
        Ih, Iw = gray.shape

        method = None
        rect   = None
        warped = None

        # ── Priority 1: anchor-based ──────────────────────────────────────────
        orig_anchors = self._find_anchors_global(img)
        if orig_anchors:
            method  = "anchor-based"
            src_pts = np.float32([orig_anchors["TL"], orig_anchors["TR"],
                                   orig_anchors["BR"], orig_anchors["BL"]])
            dst_pts = np.float32([
                [CFG["anchor_cx_left"],  CFG["anchor_cy_top"]],
                [CFG["anchor_cx_right"], CFG["anchor_cy_top"]],
                [CFG["anchor_cx_right"], CFG["anchor_cy_bottom"]],
                [CFG["anchor_cx_left"],  CFG["anchor_cy_bottom"]],
            ])
            M      = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (W, H),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            rect   = src_pts

        # ── Priority 2: Canny contour ─────────────────────────────────────────
        if warped is None:
            corners = self._find_document_corners(gray)
            if corners is not None:
                method = "canny-contour"
                rect   = self._order_pts(corners)
                warped = self._warp(img, rect, W, H)

        # ── Priority 3: full-image fallback ───────────────────────────────────
        if warped is None:
            print("  [3-align] WARN: anchor + canny both failed — full-image fallback")
            method = "full-image"
            rect   = self._order_pts(np.float32([
                [10, 10], [Iw - 10, 10], [Iw - 10, Ih - 10], [10, Ih - 10]
            ]))
            warped = self._warp(img, rect, W, H)

        conf_map = {"anchor-based": 0.97, "canny-contour": 0.80, "full-image": 0.50}
        conf     = conf_map[method]

        # Debug: annotate original with detected corners
        vis  = img.copy()
        tl, tr, br, bl = [tuple(p.astype(int)) for p in rect]
        cv2.polylines(vis, [rect.astype(np.int32)], True, (0, 255, 80), 3)
        for pt, col, lbl in zip([tl, tr, br, bl],
                                 [(0, 0, 255), (0, 200, 0), (0, 165, 255), (255, 100, 0)],
                                 ["TL", "TR", "BR", "BL"]):
            cv2.circle(vis, pt, 12, col, -1)
            cv2.putText(vis, lbl, (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(vis, method, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        _dbg(vis,    "01_doc_corners.jpg", debug_dir)
        _dbg(warped, "02_warped.jpg",      debug_dir)

        print(f"  [3-align] method={method}  out={W}x{H}  conf={conf:.2f}")
        return AlignmentContract(
            aligned_bgr=warped,
            align_method=method,
            alignment_confidence=conf,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — STATIC LAYOUT ENGINE  (NO YOLO)
# ══════════════════════════════════════════════════════════════════════════════

class StaticLayoutEngine:
    """
    Converts an aligned image to a LayoutContract purely by geometry.

    Because the sheet is:
      • normalised to 750×1060 px (Stage 3)
      • printed to a fixed design spec

    We already KNOW where the answer grid is.  We refine those positions using
    the detected anchor centres from within the warped image.  No ML needed.

    Also decodes QR codes (pyzbar → OpenCV fallback).
    """

    # ── anchor verification in warped image ──────────────────────────────────

    def _find_anchor_blob(self, patch: np.ndarray,
                          min_area: float, max_area: float
                          ) -> Optional[Tuple[int, int]]:
        _, bw = cv2.threshold(patch, 60, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            area = cv2.contourArea(c)
            if not (min_area < area < max_area):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if min(w, h) / (max(w, h) + 1e-6) < 0.50:
                continue
            if best is None or area > cv2.contourArea(best):
                best = c
        if best is None:
            return None
        Mv = cv2.moments(best)
        if Mv["m00"] == 0:
            return None
        return int(Mv["m10"] / Mv["m00"]), int(Mv["m01"] / Mv["m00"])

    def _verify_anchors(self, warped: np.ndarray,
                        debug_dir: pathlib.Path
                        ) -> Dict[str, Tuple[int, int]]:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        H, W = warped.shape[:2]
        r    = CFG["anchor_search_r"]
        expected = {
            "TL": (CFG["anchor_cx_left"],   CFG["anchor_cy_top"]),
            "TR": (CFG["anchor_cx_right"],  CFG["anchor_cy_top"]),
            "BR": (CFG["anchor_cx_right"],  CFG["anchor_cy_bottom"]),
            "BL": (CFG["anchor_cx_left"],   CFG["anchor_cy_bottom"]),
        }
        vis     = _to_bgr(gray)
        anchors: Dict[str, Tuple[int, int]] = {}

        for name, (ex, ey) in expected.items():
            x0 = max(0, ex - r); x1 = min(W, ex + r)
            y0 = max(0, ey - r); y1 = min(H, ey + r)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 200, 0), 2)
            local = self._find_anchor_blob(
                gray[y0:y1, x0:x1],
                CFG["anchor_min_area"], CFG["anchor_max_area"],
            )
            if local:
                cx, cy = x0 + local[0], y0 + local[1]
                anchors[name] = (cx, cy)
                cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(vis, name, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                anchors[name] = expected[name]
                cv2.circle(vis, expected[name], 8, (0, 0, 255), -1)
                cv2.putText(vis, f"{name}?", (expected[name][0] + 5,
                                               expected[name][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        _dbg(vis, "03_anchor_search.jpg", debug_dir)
        vis_anch = warped.copy()
        for name, (cx, cy) in anchors.items():
            cv2.circle(vis_anch, (cx, cy), 20, (0, 255, 255), 3)
            cv2.putText(vis_anch, name, (cx - 30, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        _dbg(vis_anch, "04_anchors.jpg", debug_dir)

        print(f"  [4-layout] anchors: " +
              " ".join(f"{k}={v}" for k, v in anchors.items()))
        return anchors

    # ── QR decoding ──────────────────────────────────────────────────────────

    def _decode_qr(self, warped: np.ndarray,
                   debug_dir: pathlib.Path) -> Dict[str, Any]:
        H, W = warped.shape[:2]
        x0 = int(W * CFG["qr_x0"]); x1 = int(W * CFG["qr_x1"])
        y0 = int(H * CFG["qr_y0"]); y1 = int(H * CFG["qr_y1"])
        region = warped[y0:y1, x0:x1]
        _dbg(region, "05_qr_regions.jpg", debug_dir)

        all_codes: List[str] = []
        pyzbar_ok = False

        try:
            from pyzbar import pyzbar as pzb  # type: ignore

            def _try(img_bgr: np.ndarray) -> List[str]:
                g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                found: List[str] = []
                for scale in [2, 3, 1]:
                    gs = (cv2.resize(g, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
                          if scale > 1 else g)
                    _, bw1 = cv2.threshold(gs, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    for bw in [bw1]:
                        for obj in pzb.decode(bw):
                            val = obj.data.decode("utf-8", errors="ignore").strip()
                            if val and val not in found:
                                found.append(val)
                    if found:
                        return found
                return found

            rw = region.shape[1]
            for sub in [region, region[:, :rw // 2], region[:, rw // 2:]]:
                for code in _try(sub):
                    if code not in all_codes:
                        all_codes.append(code)
                if len(all_codes) >= 2:
                    break
            pyzbar_ok = bool(all_codes)
        except Exception as e:
            print(f"  [4-layout] pyzbar unavailable: {e}")

        if not all_codes:
            det = cv2.QRCodeDetector()
            gray_r = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            for scale in [3, 2, 1]:
                g = (cv2.resize(gray_r, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
                     if scale > 1 else gray_r)
                _, bw = cv2.threshold(g, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                val, *_ = det.detectAndDecode(bw)
                if val and val not in all_codes:
                    all_codes.append(val)

        student_code = "UNKNOWN"
        exam_code    = "UNKNOWN"
        for code in all_codes:
            up = code.upper()
            if up.startswith("STU") and student_code == "UNKNOWN":
                student_code = code
            elif exam_code == "UNKNOWN":
                exam_code = code

        qr_method = "pyzbar" if pyzbar_ok else ("opencv" if all_codes else "none")
        print(f"  [4-layout] qr method={qr_method}  student={student_code}  "
              f"exam={exam_code}")
        return {
            "student_code": student_code,
            "exam_code":    exam_code,
            "qr_raw":       all_codes,
            "qr_method":    qr_method,
        }

    # ── derive grid bounds from anchors ──────────────────────────────────────

    @staticmethod
    def _derive_grid(anchors: Dict[str, Tuple[int, int]],
                     W: int, H: int) -> Tuple[int, int, int, int]:
        """
        Compute the answer grid bbox using anchor positions if available,
        or fall back to fixed fractions of the warped image.
        """
        corner_keys = {"TL", "TR", "BR", "BL"}
        if corner_keys.issubset(anchors):
            ay_top   = min(anchors["TL"][1], anchors["TR"][1])
            ay_bot   = max(anchors["BL"][1], anchors["BR"][1])
            ax_left  = min(anchors["TL"][0], anchors["BL"][0])
            ax_right = max(anchors["TR"][0], anchors["BR"][0])
            span_h   = ay_bot - ay_top
            span_w   = ax_right - ax_left
            gx0 = int(ax_left  + span_w * 0.06)
            gx1 = int(ax_right - span_w * 0.06)
            gy0 = int(ay_top   + span_h * CFG["grid_y0_frac"])
            gy1 = int(ay_top   + span_h * CFG["grid_y1_frac"])
        else:
            gx0 = int(W * CFG["grid_x0_frac"])
            gx1 = int(W * CFG["grid_x1_frac"])
            gy0 = int(H * 0.247)
            gy1 = int(H * 0.94)
        return gx0, gy0, gx1, gy1

    # ── public entry point ────────────────────────────────────────────────────

    def process(self, align: AlignmentContract,
                debug_dir: pathlib.Path,
                n_col_groups: int = 4,
                ) -> LayoutContract:
        warped  = align.aligned_bgr
        H, W    = warped.shape[:2]
        anchors = self._verify_anchors(warped, debug_dir)
        qr      = self._decode_qr(warped, debug_dir)

        gx0, gy0, gx1, gy1 = self._derive_grid(anchors, W, H)

        # Equal-width column layout as the starting point; Stage 5 will refine
        # using the actual bar positions detected in the image.
        col_w = (gx1 - gx0) // n_col_groups
        columns = [
            ColumnLayout(
                col_idx=i,
                x0=gx0 + i * col_w,
                x1=gx0 + (i + 1) * col_w,
                y0=gy0,
                y1=gy1,
            )
            for i in range(n_col_groups)
        ]

        conf = 0.98 if {"TL", "TR", "BL", "BR"}.issubset(anchors) else 0.80

        # Debug: show grid and columns
        vis = warped.copy()
        cv2.rectangle(vis, (gx0, gy0), (gx1, gy1), (0, 100, 255), 2)
        for col in columns:
            cv2.rectangle(vis, (col.x0, col.y0), (col.x1, col.y1),
                          (200, 200, 0), 1)
        cv2.putText(vis, f"static grid ({align.align_method})",
                    (gx0 + 4, gy0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        _dbg(vis, "06_layout.jpg", debug_dir)

        print(f"  [4-layout] grid x0={gx0} y0={gy0} x1={gx1} y1={gy1}  "
              f"cols={n_col_groups}  conf={conf:.2f}")

        return LayoutContract(
            aligned_bgr=warped,
            columns=columns,
            anchors=anchors,
            student_code=qr["student_code"],
            exam_code=qr["exam_code"],
            qr_raw=qr["qr_raw"],
            qr_method=qr["qr_method"],
            layout_confidence=conf,
            grid_x0=gx0, grid_y0=gy0,
            grid_x1=gx1, grid_y1=gy1,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — BUBBLE GRID ENGINE  (deterministic, NO detection model)
# ══════════════════════════════════════════════════════════════════════════════

class BubbleGridEngine:
    """
    Converts layout geometry into an exact set of bubble coordinates.

    Sub-steps:
      5A  Bar detection  — horizontal projection profile → locate printed bars
      5B  Column segmentation — bar edges define precise column x-bounds
      5C  Row detection  — vertical projection inside each column → row centres
      5D  Bubble generation — divide column width into n_choices equal parts
                              bubble_cx[j] = linspace(col_x0, col_x1, n_choices)[j]
                              bubble_cy[i] = row_centers[i]
                              radius       = CFG["bubble_r"]

    No detection model.  No K-means.  No HoughCircles.
    Exactly n_col_groups × rows_per_group × n_choices bubbles are generated.
    """

    # ── 5A: bar detection via horizontal projection ───────────────────────────

    @staticmethod
    def _detect_bars(gray_warped: np.ndarray,
                     gx0: int, gy0: int, gx1: int, gy1: int,
                     n_cols: int,
                     ) -> List[Tuple[int, int, int, int]]:
        """
        Detect printed bubble-zone divider bars using a horizontal projection
        profile of DARK pixels inside a scan strip above/below the grid.

        For each bar, returns (bar_x0, bar_x1, col_top_y, col_bot_y).
        If bar detection fails, returns an empty list (caller falls back to
        equal-width column split from Stage 4).
        """
        H, W     = gray_warped.shape
        dark_thr = CFG["bar_dark_thr"]
        min_bar  = CFG["bar_min_width_px"]

        col_width   = (gx1 - gx0) / max(n_cols, 1)
        min_bar_run = max(min_bar, int(col_width * 0.20))

        # Extend the scan 30 px beyond the grid edges so that the outermost
        # column bars (which may overlap or start just outside the grid
        # boundary) are captured in full.
        scan_x0 = max(0, gx0 - 30)
        scan_x1 = min(W, gx1 + 30)

        def _scan_strip(sy0: int, sy1: int) -> List[Tuple[int, int]]:
            """Find horizontal runs of dark columns in the strip."""
            sy0 = max(0, sy0); sy1 = min(H, sy1)
            if sy0 >= sy1:
                return []
            strip    = gray_warped[sy0:sy1, scan_x0:scan_x1]
            dark_col = np.sum(strip < dark_thr, axis=0)
            is_dark  = dark_col >= max(3, (sy1 - sy0) * 0.4)
            runs: List[Tuple[int, int]] = []
            in_run, rs = False, 0
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

        # Bars printed just outside the YOLO/anchor grid boundary
        # Scan 5..70 px above gy0 and 5..70 px below gy1
        top_runs = _scan_strip(gy0 - 70, gy0 - 4)
        bot_runs = _scan_strip(gy1 + 4,  gy1 + 70)

        # Fallback: scan inside the grid boundary if nothing found outside
        if not top_runs:
            top_runs = _scan_strip(gy0 - 4, gy0 + 30)
        if not bot_runs:
            bot_runs = _scan_strip(gy1 - 30, gy1 + 4)

        if not top_runs:
            return []

        # Match top and bottom runs by x-center proximity
        results: List[Tuple[int, int, int, int]] = []
        used_bot: set = set()
        for tx0, tx1 in top_runs:
            tc = (tx0 + tx1) // 2
            best_j, best_d = -1, col_width * 0.4
            for j, (bx0, bx1) in enumerate(bot_runs):
                d = abs((bx0 + bx1) // 2 - tc)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_j not in used_bot:
                bx0, bx1 = bot_runs[best_j]
                # Use the WIDEST coverage of top + bottom bars so that a
                # partially-detected bottom bar cannot narrow the result.
                x_l = min(tx0, bx0)
                x_r = max(tx1, bx1)
                results.append((x_l, x_r, gy0 - 4, gy1 + 4))
                used_bot.add(best_j)
            else:
                results.append((tx0, tx1, gy0 - 4, gy1 + 4))

        return sorted(results, key=lambda r: r[0])

    # ── 5B: column segmentation from bars ─────────────────────────────────────

    @staticmethod
    def _bars_to_columns(bars: List[Tuple[int, int, int, int]],
                         layout_cols: List[ColumnLayout],
                         n_choices: int,
                         ) -> List[ColumnLayout]:
        """
        Convert detected bars into refined ColumnLayout objects.

        bar[i] = (bar_x0, bar_x1, col_top_y, col_bot_y)
        bubble zone: x0=bar_x0, x1=bar_x0 + (bar_x1-bar_x0) (= bar right edge)
        The region RIGHT of bar_x1 to the next bar_x0 is the question-number zone.
        """
        if len(bars) == len(layout_cols):
            cols = []
            for i, (bx0, bx1, cty, cby) in enumerate(bars):
                # The BUBBLE ZONE is from bx0 to bx1; right of bx1 = question numbers
                cols.append(ColumnLayout(
                    col_idx=i,
                    x0=bx0, x1=bx1,
                    y0=cty, y1=cby,
                ))
            return cols
        # Bar count mismatch — fall back to layout_cols as-is
        print(f"  [5-grid] bar count {len(bars)} ≠ col count {len(layout_cols)} "
              "— using static layout columns")
        return layout_cols

    # ── 5C: row detection via dark-band finding ───────────────────────────────

    @staticmethod
    @staticmethod
    def _bands_from_profile(
        profile: np.ndarray,
        col_w: float,
        y0: int,
        rows_per_group: int,
        allow_extrapolation: bool = False,
        merge_gap_override: int = 0,
    ) -> Tuple[List[int], str]:
        """
        Core band detector: accepts a pre-computed dark-pixel profile and
        returns (row_centers, quality) where quality is "exact"|"trim"|"fallback".

        merge_gap is derived from the expected row pitch so it correctly bridges
        intra-row dips on both dense (100Q, pitch≈25px) and sparse (50Q,
        pitch≈51px) sheets without accidentally merging adjacent rows.
        """
        y1 = y0 + len(profile)

        if profile.max() == 0:
            return _uniform_rows(y0, y1, rows_per_group), "fallback"

        bar_thr = col_w * 0.55          # solid printed bar (>55 % of col width)
        bub_thr = max(8.0, col_w * 0.08)   # bubble row presence

        # Pitch-based merge gap: bridges intra-row profile dips without
        # accidentally merging adjacent rows on any sheet density.
        pitch     = (y1 - y0) / rows_per_group
        merge_gap = merge_gap_override if merge_gap_override > 0 else max(5, int(pitch * 0.20))

        # ── find contiguous bands above bub_thr that are NOT main bars ────────
        bands: List[Tuple[int, int]] = []
        in_band, bs = False, 0
        for y, v in enumerate(profile):
            if v >= bub_thr and not in_band:
                in_band, bs = True, y
            elif v < bub_thr and in_band:
                in_band = False
                if profile[bs:y].max() < bar_thr:
                    bands.append((bs, y - 1))
        if in_band and profile[bs:].max() < bar_thr:
            bands.append((bs, len(profile) - 1))

        # ── merge fragments whose gap ≤ merge_gap ────────────────────────────
        merged: List[Tuple[int, int]] = []
        for b in bands:
            if merged and b[0] - merged[-1][1] <= merge_gap:
                prev = merged.pop()
                merged.append((prev[0], b[1]))
            else:
                merged.append(b)
        bands = merged

        # ── proximity filter ─────────────────────────────────────────────────
        # Remove a band whose centre is too close to the previous band's centre
        # (< 30 % of the expected row pitch).  Such near-duplicates arise when
        # a circle's bottom edge creates a secondary peak immediately below the
        # real row centre — keeping both causes a spurious "exact" count while
        # the true last row is silently missed.
        #
        # When two bands are equally tall (e.g. both 0 px), keep the first
        # (lower abs-y = closer to the real position of that row).
        min_prox = max(10, int(pitch * 0.30))
        proxied: List[Tuple[int, int]] = []
        for b in bands:
            bc = (b[0] + b[1]) // 2
            if proxied:
                pc = (proxied[-1][0] + proxied[-1][1]) // 2
                if bc - pc < min_prox:
                    # keep the taller; on tie keep the first (lower y)
                    if (b[1] - b[0]) > (proxied[-1][1] - proxied[-1][0]):
                        proxied[-1] = b
                    continue
            proxied.append(b)
        bands = proxied

        # ── compute band centres ──────────────────────────────────────────────
        # For the combined profile (merge_gap_override > 0) the averaged signal
        # of mostly-empty circles only crosses bub_thr at the top outline peak
        # (y ≈ cy−r).  Using the midpoint of that narrow band under-estimates cy
        # by ~r pixels.  Instead use band_start + bubble_r, which equals cy for
        # BOTH top-edge-only bands AND full-circle bands (since band_start ≈ cy−r
        # in all cases).
        if merge_gap_override > 0:
            bubble_r = (merge_gap_override - 1) // 2   # recover r from 2r+1
            def _center(b: Tuple[int, int]) -> int:
                return y0 + b[0] + bubble_r
        else:
            def _center(b: Tuple[int, int]) -> int:
                return y0 + (b[0] + b[1]) // 2

        if len(bands) == rows_per_group:
            centers = [_center(b) for b in bands]
            return centers, "exact"
        elif len(bands) > rows_per_group:
            heights   = [(b[1] - b[0], b) for b in bands]
            heights.sort(reverse=True)
            top_bands = sorted([b for _, b in heights[:rows_per_group]],
                               key=lambda b: b[0])
            centers = [_center(b) for b in top_bands]
            return centers, "trim"
        elif len(bands) >= 3 and allow_extrapolation:
            # ── extrapolation: too few bands but enough to infer spacing ─────
            # Compute the characteristic within-group spacing (the median of
            # "small" gaps, excluding the larger between-group sub-header gaps).
            # Use it to append the missing trailing rows instead of falling
            # back to a uniform grid that ignores sub-group structure.
            centers = [_center(b) for b in bands]
            spacings = [centers[i + 1] - centers[i]
                        for i in range(len(centers) - 1)]
            med_s        = int(np.median(spacings))
            within_pitch = int(np.median(
                [s for s in spacings if s <= int(med_s * 1.5)]
            ))
            while len(centers) < rows_per_group:
                centers.append(centers[-1] + within_pitch)
            return centers[:rows_per_group], "extrapolated"
        else:
            return _uniform_rows(y0, y1, rows_per_group), "fallback"

    @staticmethod
    def _detect_rows(gray_warped: np.ndarray,
                     col: ColumnLayout,
                     rows_per_group: int,
                     bubble_r: int,
                     combined_profile: Optional[np.ndarray] = None,
                     combined_col_w: float = 0.0,
                     combined_y0: int = 0,
                     ) -> Tuple[List[int], str]:
        """
        Detect bubble row centers.

        Uses the per-column profile by default.  If the per-column detection
        falls back to uniform spacing (too-faint signal, e.g. mostly-empty
        columns on a sparse sheet), the caller-supplied *combined_profile*
        (averaged across all columns) is tried instead — combining the signal
        from well-filled columns rescues detection for faint columns.
        """
        dark_thr = CFG["row_dark_thr"]
        crop     = gray_warped[col.y0:col.y1, col.x0:col.x1]
        if crop.size == 0:
            return _uniform_rows(col.y0, col.y1, rows_per_group), "fallback"

        col_w   = float(crop.shape[1])
        profile = np.sum(crop < dark_thr, axis=1).astype(np.float64)

        centers, quality = BubbleGridEngine._bands_from_profile(
            profile, col_w, col.y0, rows_per_group
        )

        # If per-column failed and a combined profile is available, retry
        if quality == "fallback" and combined_profile is not None:
            slice_len = col.y1 - col.y0
            comb_slice = combined_profile[col.y0 - combined_y0:
                                          col.y0 - combined_y0 + slice_len]
            centers, quality = BubbleGridEngine._bands_from_profile(
                comb_slice, combined_col_w, col.y0, rows_per_group,
                allow_extrapolation=True,
                # Larger merge gap for combined: the averaged profile of mostly-empty
                # circles is double-peaked (top + bottom outline), with a dip spanning
                # up to ~2r-2 pixels.  Bridging it requires merge_gap ≥ 2*r-2 ≈ 14.
                # We use 2*r+1 to give a safe margin for in-memory vs JPEG pixel
                # differences.  Per-column profiles use the default pitch*0.20 gap.
                merge_gap_override=2 * bubble_r + 1,
            )
            if quality != "fallback":
                quality = "combined"  # annotate source

        return centers, quality

    # ── 5D: bubble position generation ───────────────────────────────────────

    @staticmethod
    def _generate_bubbles(col: ColumnLayout,
                          row_centers: List[int],
                          col_idx: int,
                          n_col_groups: int,
                          rows_per_group: int,
                          n_choices: int,
                          n_questions: int,
                          bubble_r: int,
                          ) -> List[GeneratedBubble]:
        """
        For each (row, choice) pair, generate an exact bubble coordinate.

        RTL layout:
          col_group 0 (leftmost)  → Q 76–100
          col_group 1             → Q 51–75
          col_group 2             → Q 26–50
          col_group 3 (rightmost) → Q  1–25

        Within each row, choice rank 0 = rightmost = A (أ).
        """
        base_q  = (n_col_groups - 1 - col_idx) * rows_per_group + 1
        r       = bubble_r
        bubbles: List[GeneratedBubble] = []

        # Evenly-spaced choice X positions within the bubble zone.
        # The question-number label occupies the RIGHT portion of each bar.
        # The left inset is purely proportional (8.7% of col_span).
        # The right inset has a fixed component (~10 px) plus a proportional
        # one (8.9%), because the question-number text is partially a fixed
        # absolute width regardless of column size:
        #   right_inset_px = 10 + col_span × 0.089
        # Calibrated: 50Q col_w≈114 → 20 px (matches old 0.174×114=20),
        #             20Q col_w≈254 → 33 px (old 0.174×254=44 was 11 px too far).
        col_span  = col.x1 - col.x0
        right_inset = 10.0 + col_span * 0.089
        choice_xs = np.linspace(
            col.x0 + col_span * 0.087,
            col.x1 - right_inset,
            n_choices,
        )
        # RTL: rank 0 = rightmost → reverse so rank 0 = last element
        choice_xs_rtl = choice_xs[::-1]

        for row_idx, cy in enumerate(row_centers[:rows_per_group]):
            q_num = base_q + row_idx
            if q_num > n_questions:
                break
            for rank in range(n_choices):
                cx     = int(choice_xs_rtl[rank])
                option = RANK_TO_CHOICE.get(rank, "?")
                bubbles.append(GeneratedBubble(
                    cx=cx, cy=cy, r=r,
                    col_idx=col_idx,
                    row_idx=row_idx,
                    choice_rank=rank,
                    question_id=str(q_num),
                    option=option,
                ))

        return bubbles

    # ── public entry point ────────────────────────────────────────────────────

    def process(self, layout: LayoutContract,
                debug_dir: pathlib.Path,
                n_questions: int = 100,
                rows_per_group: int = 25,
                n_choices: int = 5,
                ) -> BubbleGridContract:
        warped = layout.aligned_bgr
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        H, W   = warped.shape[:2]
        n_cg   = len(layout.columns)

        # ── 5A: detect bars ───────────────────────────────────────────────────
        bars = self._detect_bars(
            gray,
            layout.grid_x0, layout.grid_y0,
            layout.grid_x1, layout.grid_y1,
            n_cg,
        )
        print(f"  [5-grid] bars detected: {len(bars)}")
        for i, b in enumerate(bars):
            print(f"           col_{i}  bar=[{b[0]},{b[1]}]  y=[{b[2]},{b[3]}]")

        # ── 5B: refine columns ────────────────────────────────────────────────
        columns = self._bars_to_columns(bars, layout.columns, n_choices)

        # Debug 06b: bar detection overlay
        vis_bar = warped.copy()
        for i, col in enumerate(columns):
            color = [(0, 200, 80), (0, 140, 255), (200, 80, 0), (80, 0, 200)][i % 4]
            cv2.rectangle(vis_bar, (col.x0, col.y0), (col.x1, col.y1), color, 2)
            cv2.putText(vis_bar, f"col_{i}", (col.x0 + 4, col.y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        _dbg(vis_bar, "07_columns.jpg", debug_dir)

        # ── 5C: pre-compute combined profile (max-normalised across cols) ───────
        # Using the per-row MAX (not mean) of normalised dark-pixel fractions
        # preserves weak signals from partially-filled columns.  Mean dilutes
        # a sparse column's signal to zero when other columns are empty at the
        # same Y, causing leading rows (Q1, Q2 …) to be missed entirely.
        dark_thr  = CFG["row_dark_thr"]
        y0_common = min(c.y0 for c in columns)
        y1_common = max(c.y1 for c in columns)
        combined  = np.zeros(y1_common - y0_common, dtype=np.float64)
        for col in columns:
            crop    = gray[col.y0:col.y1, col.x0:col.x1]
            col_w_  = float(col.x1 - col.x0)
            p       = np.sum(crop < dark_thr, axis=1).astype(np.float64) / col_w_
            start   = col.y0 - y0_common
            end     = col.y1 - y0_common
            combined[start:end] = np.maximum(combined[start:end], p)  # max, not sum
        avg_col_w  = float(np.mean([c.x1 - c.x0 for c in columns]))
        combined  *= avg_col_w                                 # → pixel counts

        # Dynamic bubble radius: scales with column width to match the actual
        # printed circle size.  Calibrated: 50Q/100Q col_w≈114 → r=9,
        # 20Q col_w≈254 → r=12.  Formula: max(cfg_r, round(col_w × 0.047)).
        bubble_r   = max(CFG["bubble_r"], round(avg_col_w * 0.047))

        # ── 5C: detect rows per column ────────────────────────────────────────
        all_row_centers: List[List[int]] = []
        for col in columns:
            row_centers, quality = self._detect_rows(
                gray, col, rows_per_group,
                bubble_r,
                combined_profile=combined,
                combined_col_w=avg_col_w,
                combined_y0=y0_common,
            )
            all_row_centers.append(row_centers)
            print(f"  [5-grid] col_{col.col_idx}  rows detected: "
                  f"{len(row_centers)}  ({quality}, expected {rows_per_group})")

        # ── 5D: generate bubble positions ─────────────────────────────────────
        all_bubbles: List[GeneratedBubble] = []
        for i, (col, row_centers) in enumerate(zip(columns, all_row_centers)):
            bs = self._generate_bubbles(
                col, row_centers, i, n_cg, rows_per_group,
                n_choices, n_questions, bubble_r,
            )
            all_bubbles.extend(bs)

        print(f"  [5-grid] total bubbles generated: {len(all_bubbles)}  "
              f"(expected {n_cg * rows_per_group * n_choices})")

        # Grid confidence: 1.0 if bars detected, 0.85 if using layout fallback
        grid_conf = 1.0 if len(bars) == n_cg else 0.85

        # Debug 08: generated bubble grid
        vis_grid = warped.copy()
        for b in all_bubbles:
            col_color = [(0, 200, 80), (0, 140, 255),
                         (200, 80, 0), (80, 0, 200)][b.col_idx % 4]
            cv2.circle(vis_grid, (b.cx, b.cy), b.r, col_color, 1)
            if b.choice_rank == 0:
                cv2.putText(vis_grid, b.option,
                            (b.cx - 5, b.cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, col_color, 1)
        cv2.putText(vis_grid, f"generated={len(all_bubbles)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
        _dbg(vis_grid, "08_bubble_grid.jpg", debug_dir)

        return BubbleGridContract(
            aligned_bgr=warped,
            bubbles=all_bubbles,
            columns=columns,
            grid_confidence=grid_conf,
        )


def _uniform_rows(y0: int, y1: int, n: int) -> List[int]:
    """Return n evenly-spaced row centres between y0 and y1."""
    if n <= 0:
        return []
    step = (y1 - y0) / n
    return [int(y0 + step * (i + 0.5)) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 6 — BUBBLE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class BubbleClassifier:
    """
    Classify each bubble crop as empty | filled | ambiguous.

    Primary:  CNN model (ONNX or PyTorch) loaded from models/bubble_classifier.onnx
              (or models/bubble_classifier.pt for PyTorch).
              Input: 32×32 greyscale, normalised 0–1.
              Output: 3-class softmax [empty, filled, ambiguous].

    Fallback: Shading-corrected fill ratio when no model file is present.
              filled    → fill_ratio > fill_ratio_thr
              ambiguous → fill_ratio in [thr*0.5, thr*1.5]
              empty     → fill_ratio < thr*0.5

    Train the CNN using the bubble crops saved by this script.
    See train_bubble_classifier.py for the training recipe.
    """

    CROP_SIZE = 32
    _ONNX_PATH   = pathlib.Path(__file__).parent / "models" / "bubble_classifier.onnx"
    _PT_PATH     = pathlib.Path(__file__).parent / "models" / "bubble_classifier.pt"

    def __init__(self):
        self._session = None    # ONNX runtime session
        self._pt_model = None   # PyTorch model
        self._type     = "fill_ratio"
        self._load_model()

    def _load_model(self) -> None:
        # Try ONNX first (fastest, no framework dependency)
        if self._ONNX_PATH.exists():
            try:
                import onnxruntime as ort  # type: ignore
                self._session = ort.InferenceSession(
                    str(self._ONNX_PATH),
                    providers=["CPUExecutionProvider"],
                )
                self._type = "cnn_onnx"
                print(f"  [6-classify] CNN (ONNX) loaded: {self._ONNX_PATH}")
                return
            except Exception as e:
                print(f"  [6-classify] ONNX load failed ({e}) — trying PyTorch")

        # Try PyTorch
        if self._PT_PATH.exists():
            try:
                import torch  # type: ignore
                self._pt_model = torch.load(
                    str(self._PT_PATH), map_location="cpu", weights_only=False
                )
                self._pt_model.eval()
                self._type = "cnn_pytorch"
                print(f"  [6-classify] CNN (PyTorch) loaded: {self._PT_PATH}")
                return
            except Exception as e:
                print(f"  [6-classify] PyTorch load failed ({e}) — using fill_ratio")

        print("  [6-classify] No CNN model found — using fill_ratio fallback")
        print(f"               (place bubble_classifier.onnx in {self._ONNX_PATH.parent}/)")

    # ── crop helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _crop(gray: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
        """Extract a square crop around (cx, cy) and resize to CROP_SIZE×CROP_SIZE."""
        H, W    = gray.shape
        pad     = max(r + 2, BubbleClassifier.CROP_SIZE // 2)
        x0      = max(0, cx - pad);  x1 = min(W, cx + pad)
        y0      = max(0, cy - pad);  y1 = min(H, cy + pad)
        crop    = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return np.full((BubbleClassifier.CROP_SIZE,
                            BubbleClassifier.CROP_SIZE), 128, dtype=np.uint8)
        return cv2.resize(crop,
                          (BubbleClassifier.CROP_SIZE, BubbleClassifier.CROP_SIZE),
                          interpolation=cv2.INTER_AREA)

    # ── fill-ratio fallback ───────────────────────────────────────────────────

    @staticmethod
    def _fill_ratio_classify(gray_warped: np.ndarray,
                             bubbles: List[GeneratedBubble],
                             ) -> List[BubblePrediction]:
        """
        Shading-corrected fill ratio → 3-class softmax approximation.

        Two-pass algorithm:
          Pass 1: compute raw shading-corrected fill for every bubble.
          Pass 2: per question, normalise by the row maximum so that ink-bleed
                  from a heavily-filled bubble onto its neighbour is filtered out.
                  A secondary bubble only counts as "filled" if its fill is ≥
                  REL_RATIO (0.65) × the row's maximum fill AND exceeds the
                  absolute fill_ratio_thr.  This eliminates the most common
                  source of false double-marks in the fill_ratio fallback.
        """
        from collections import defaultdict

        REL_RATIO = 0.65   # secondary fill must be ≥ this fraction of the row max

        bg   = cv2.medianBlur(gray_warped, 71)
        bg   = np.maximum(bg, 1)
        norm = np.clip(
            (gray_warped.astype(np.float32) / bg.astype(np.float32)) * 255,
            0, 255,
        ).astype(np.uint8)
        dark_binary = (norm < CFG["fill_norm_thr"]).astype(np.uint8)

        H, W = gray_warped.shape
        thr  = CFG["fill_ratio_thr"]

        # ── Pass 1: raw fill for every bubble ────────────────────────────────
        raw_fills: List[float] = []
        for b in bubbles:
            cx, cy, r = b.cx, b.cy, b.r
            if cx - r < 1 or cy - r < 1 or cx + r >= W - 1 or cy + r >= H - 1:
                raw_fills.append(0.0)
                continue
            mask = np.zeros((H, W), np.uint8)
            cv2.circle(mask, (cx, cy), max(r - 2, 2), 255, -1)
            nz   = np.count_nonzero(mask)
            dark = int(np.sum(dark_binary[mask > 0]))
            raw_fills.append(dark / nz if nz > 0 else 0.0)

        # ── Pass 2: per-question row max → relative threshold ─────────────────
        # Group bubble indices by question_id
        q_indices: dict = defaultdict(list)
        for i, b in enumerate(bubbles):
            q_indices[b.question_id].append(i)

        # For each question compute the row max
        row_max: List[float] = [0.0] * len(bubbles)
        for q_str, indices in q_indices.items():
            mx = max(raw_fills[i] for i in indices)
            for i in indices:
                row_max[i] = mx

        # ── Build predictions ─────────────────────────────────────────────────
        preds: List[BubblePrediction] = []
        for i, b in enumerate(bubbles):
            fill = raw_fills[i]
            mx   = row_max[i]

            # Absolute threshold AND relative dominance
            abs_ok = fill >= thr
            rel_ok = mx < thr * 0.5 or fill >= mx * REL_RATIO

            # In fill_ratio fallback mode only "filled" or "empty" are returned.
            # "ambiguous" is reserved for the CNN classifier which has richer
            # per-crop information.  Using ambiguous here inflates the
            # ambiguous count and confuses the answered/unanswered tally.
            if abs_ok and rel_ok:
                filled_conf    = min(1.0, fill / (thr + 1e-6))
                ambiguous_conf = 0.0
                empty_conf     = max(0.0, 1.0 - filled_conf)
                status         = "filled"
            else:
                empty_conf     = 1.0
                ambiguous_conf = 0.0
                filled_conf    = 0.0
                status         = "empty"

            preds.append(BubblePrediction(
                question_id=b.question_id, option=b.option,
                status=status,
                filled_conf=round(filled_conf, 3),
                ambiguous_conf=round(ambiguous_conf, 3),
                empty_conf=round(empty_conf, 3),
                fill_ratio=round(fill, 3),
            ))

        return preds

    # ── CNN inference ─────────────────────────────────────────────────────────

    def _cnn_classify(self, gray_warped: np.ndarray,
                      bubbles: List[GeneratedBubble],
                      ) -> List[BubblePrediction]:
        """Run CNN model on every bubble crop."""
        crops = np.array([
            self._crop(gray_warped, b.cx, b.cy, b.r)
            for b in bubbles
        ], dtype=np.float32) / 255.0
        # Shape: (N, CROP_SIZE, CROP_SIZE) → (N, 1, CROP_SIZE, CROP_SIZE)
        crops = crops[:, np.newaxis, :, :]

        if self._type == "cnn_onnx":
            inp_name = self._session.get_inputs()[0].name
            logits   = self._session.run(None, {inp_name: crops})[0]  # (N, 3)
        else:
            import torch  # type: ignore
            with torch.no_grad():
                t      = torch.tensor(crops)
                logits = self._pt_model(t).numpy()

        # Softmax
        e    = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)  # (N, 3): [empty, filled, ambiguous]

        # Back-compute approximate fill ratio from filled confidence for debug
        bg   = cv2.medianBlur(gray_warped, 71)
        bg   = np.maximum(bg, 1)
        norm = np.clip(
            (gray_warped.astype(np.float32) / bg.astype(np.float32)) * 255, 0, 255
        ).astype(np.uint8)
        dark_binary = (norm < CFG["fill_norm_thr"]).astype(np.uint8)
        H, W = gray_warped.shape

        preds: List[BubblePrediction] = []
        for i, b in enumerate(bubbles):
            empty_c    = float(probs[i, 0])
            filled_c   = float(probs[i, 1])
            ambig_c    = float(probs[i, 2])

            # Decision
            if filled_c >= CFG["cnn_filled_conf"]:
                status = "filled"
            elif ambig_c >= CFG["cnn_ambiguous_conf"]:
                status = "ambiguous"
            else:
                status = "empty"

            # Fill ratio for debug/logging
            cx, cy, r = b.cx, b.cy, b.r
            fill = 0.0
            if 0 < cx - r and 0 < cy - r and cx + r < W - 1 and cy + r < H - 1:
                mask = np.zeros((H, W), np.uint8)
                cv2.circle(mask, (cx, cy), max(r - 2, 2), 255, -1)
                nz   = np.count_nonzero(mask)
                dark = int(np.sum(dark_binary[mask > 0]))
                fill = dark / nz if nz > 0 else 0.0

            preds.append(BubblePrediction(
                question_id=b.question_id, option=b.option,
                status=status,
                filled_conf=round(filled_c, 3),
                ambiguous_conf=round(ambig_c, 3),
                empty_conf=round(empty_c, 3),
                fill_ratio=round(fill, 3),
            ))

        return preds

    # ── public entry point ────────────────────────────────────────────────────

    def process(self, grid: BubbleGridContract,
                debug_dir: pathlib.Path) -> ClassificationContract:
        warped = grid.aligned_bgr
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        if self._type.startswith("cnn"):
            preds = self._cnn_classify(gray, grid.bubbles)
        else:
            preds = self._fill_ratio_classify(gray, grid.bubbles)

        n_filled = sum(1 for p in preds if p.status == "filled")
        n_amb    = sum(1 for p in preds if p.status == "ambiguous")
        n_total  = len(preds)
        conf     = round(n_filled / max(n_total, 1), 3)

        # Per-question summary
        from collections import defaultdict
        q_preds: Dict[str, List[BubblePrediction]] = defaultdict(list)
        for p in preds:
            q_preds[p.question_id].append(p)
        rows_info = []
        for qid in sorted(q_preds, key=lambda x: int(x)):
            ps   = sorted(q_preds[qid], key=lambda p: p.filled_conf, reverse=True)
            top2 = "  ".join(f"{p.option}={p.fill_ratio:.2f}" for p in ps[:2])
            rows_info.append(f"Q{qid}:{top2}")
        print(f"  [6-classify] type={self._type}  total={n_total}  "
              f"filled={n_filled}  ambiguous={n_amb}")
        print("  [6-classify] fills: " + "  ".join(rows_info))

        # Debug: draw classified bubbles
        vis = warped.copy()
        for b, p in zip(grid.bubbles, preds):
            if p.status == "filled":
                color, thick = (0, 220, 0), 3
            elif p.status == "ambiguous":
                color, thick = (0, 200, 255), 2
            else:
                color, thick = (0, 100, 200), 1
            cv2.circle(vis, (b.cx, b.cy), b.r, color, thick)
            cv2.putText(vis, f"{p.fill_ratio:.2f}",
                        (b.cx - 12, b.cy - b.r - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.20, color, 1)
        _dbg(vis, "09_classified.jpg", debug_dir)

        return ClassificationContract(
            aligned_bgr=warped,
            predictions=preds,
            bubbles=grid.bubbles,
            classification_confidence=conf,
            classifier_type=self._type,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 7 — ANSWER LOGIC ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AnswerLogicEngine:
    """
    Convert per-bubble predictions into per-question answers using simple rules:

      Rule 1:  exactly 1 filled  →  answer = that option
      Rule 2:  0 filled          →  unanswered (check for ambiguous)
      Rule 3:  >1 filled         →  double_mark (report all, pick highest confidence)
      Rule 4:  any ambiguous     →  flag for review (non-blocking)

    No multi-tier heuristics.  No dominance ratios.  No fill-mean adjustments.
    The classifier has already made the hard decisions — we just aggregate.
    """

    def process(self, cls: ClassificationContract,
                n_questions: int = 100,
                rows_per_group: int = 25,
                n_col_groups: int = 4,
                ) -> ValidationContract:
        from collections import defaultdict

        q_preds: Dict[str, List[BubblePrediction]] = defaultdict(list)
        for p in cls.predictions:
            q_preds[p.question_id].append(p)

        answers: Dict[str, Optional[str]] = {}
        details: Dict[str, QuestionResult] = {}

        for q_num in range(1, n_questions + 1):
            q_str  = str(q_num)
            row_ps = q_preds.get(q_str, [])

            if not row_ps:
                answers[q_str] = None
                details[q_str] = QuestionResult(
                    question_id=q_str, choice=None,
                    note="no_circles", all_filled=[], fill=0.0,
                    classifier_type=cls.classifier_type,
                )
                continue

            filled   = [p for p in row_ps if p.status == "filled"]
            ambig    = [p for p in row_ps if p.status == "ambiguous"]
            best_fill = max(p.fill_ratio for p in row_ps)

            if len(filled) == 1:
                # Clean single answer
                p          = filled[0]
                answers[q_str] = p.option
                note       = "ambiguous_review" if ambig else ""
                details[q_str] = QuestionResult(
                    question_id=q_str, choice=p.option,
                    note=note, all_filled=[p.option],
                    fill=p.fill_ratio,
                    classifier_type=cls.classifier_type,
                )

            elif len(filled) == 0:
                # Unanswered — check if anything is ambiguous
                answers[q_str] = None
                note = "ambiguous_review" if ambig else "unanswered"
                details[q_str] = QuestionResult(
                    question_id=q_str, choice=None,
                    note=note, all_filled=[],
                    fill=best_fill,
                    classifier_type=cls.classifier_type,
                )

            else:
                # Multiple filled → double mark; pick highest confidence
                best = max(filled, key=lambda p: p.filled_conf)
                answers[q_str] = best.option
                details[q_str] = QuestionResult(
                    question_id=q_str, choice=best.option,
                    note="double_mark",
                    all_filled=[p.option for p in filled],
                    fill=best.fill_ratio,
                    classifier_type=cls.classifier_type,
                )

        unanswered    = [q for q in range(1, n_questions + 1)
                         if answers.get(str(q)) is None]
        double_marked = [q for q in range(1, n_questions + 1)
                         if details.get(str(q),
                            QuestionResult("", None, "", [], 0.0, "")).note == "double_mark"]
        ambiguous     = [q for q in range(1, n_questions + 1)
                         if "ambiguous" in details.get(
                            str(q), QuestionResult("", None, "", [], 0.0, "")).note]

        print(f"  [7-logic] answered={n_questions - len(unanswered)}  "
              f"unanswered={len(unanswered)}  "
              f"double_mark={len(double_marked)}  "
              f"ambiguous_review={len(ambiguous)}")

        return ValidationContract(
            answers=answers,
            details=details,
            unanswered=unanswered,
            double_marked=double_marked,
            ambiguous=ambiguous,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 8 — STORAGE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class StorageEngine:
    """
    Build FinalResultContract, save JSON, and write the annotated answer sheet.
    """

    def process(self,
                validation: ValidationContract,
                cls: ClassificationContract,
                layout: LayoutContract,
                align: AlignmentContract,
                image_path: str,
                debug_dir: pathlib.Path,
                start_time: float,
                n_questions: int,
                ) -> FinalResultContract:

        elapsed_ms = int((time.time() - start_time) * 1000)
        answered   = n_questions - len(validation.unanswered)

        # ── debug: annotated answer sheet ─────────────────────────────────────
        vis = layout.aligned_bgr.copy()
        # Build lookup: (question_id, option) → prediction
        pred_lookup: Dict[Tuple[str, str], BubblePrediction] = {
            (p.question_id, p.option): p for p in cls.predictions
        }
        for b in cls.bubbles:
            key    = (b.question_id, b.option)
            pred   = pred_lookup.get(key)
            detail = validation.details.get(b.question_id)
            if pred is None or detail is None:
                continue

            chosen  = detail.choice == b.option
            double  = detail.note == "double_mark" and b.option in detail.all_filled
            ambig   = pred.status == "ambiguous"

            if chosen or double:
                ring_col = (0, 0, 255) if double else (0, 220, 0)
                cv2.circle(vis, (b.cx, b.cy), b.r + 4, ring_col, 2)
                cv2.circle(vis, (b.cx, b.cy), b.r, (0, 0, 0), -1)
                cv2.putText(vis, b.option,
                            (b.cx - 5, b.cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
            elif ambig:
                cv2.circle(vis, (b.cx, b.cy), b.r + 2, (0, 200, 255), 2)
                cv2.putText(vis, "?",
                            (b.cx - 4, b.cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 255), 1)
            else:
                cv2.circle(vis, (b.cx, b.cy), b.r, (180, 180, 180), 1)

        _dbg(vis, "10_answered.jpg", debug_dir)

        # ── confidence metrics ─────────────────────────────────────────────────
        confidence_metrics: Dict[str, Any] = {
            "alignment_score":           round(align.alignment_confidence, 3),
            "alignment_method":          align.align_method,
            "layout_confidence":         round(layout.layout_confidence, 3),
            "classification_confidence": round(cls.classification_confidence, 3),
            "classifier_type":           cls.classifier_type,
            "ambiguous_count":           len(validation.ambiguous),
        }

        # ── answer details dict (JSON-serialisable) ────────────────────────────
        answer_details: Dict[str, Any] = {}
        for q_str, qr in validation.details.items():
            answer_details[q_str] = {
                "choice":          qr.choice,
                "note":            qr.note,
                "all_filled":      qr.all_filled,
                "fill":            qr.fill,
                "classifier_type": qr.classifier_type,
            }

        result = FinalResultContract(
            image=image_path,
            student_code=layout.student_code,
            exam_code=layout.exam_code,
            qr_codes_raw=layout.qr_raw,
            align_method=align.align_method,
            answers={str(q): validation.answers.get(str(q))
                     for q in range(1, n_questions + 1)},
            answer_details=answer_details,
            total_questions=n_questions,
            answered=answered,
            unanswered=validation.unanswered,
            double_marked=validation.double_marked,
            ambiguous=validation.ambiguous,
            valid=(len(validation.unanswered) == 0
                   and len(validation.double_marked) == 0),
            debug_dir=str(debug_dir),
            confidence_metrics=confidence_metrics,
            processing_metrics={"time_ms": elapsed_ms},
        )

        out_dict = {
            "image":             result.image,
            "student_code":      result.student_code,
            "exam_code":         result.exam_code,
            "qr_codes_raw":      result.qr_codes_raw,
            "align_method":      result.align_method,
            "answers":           result.answers,
            "answer_details":    result.answer_details,
            "total_questions":   result.total_questions,
            "answered":          result.answered,
            "unanswered":        result.unanswered,
            "double_marked":     result.double_marked,
            "ambiguous":         result.ambiguous,
            "valid":             result.valid,
            "debug_dir":         result.debug_dir,
            "confidence_metrics": result.confidence_metrics,
            "processing_metrics": result.processing_metrics,
        }
        out_path = debug_dir / "result_v2.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=2)

        print(f"  [8-storage] JSON saved -> {out_path}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR — OMRPipelineV2
# ══════════════════════════════════════════════════════════════════════════════

class OMRPipelineV2:
    """
    Stateless orchestrator — wires the 8 geometry-first stages together.
    No global mutable state; every call is fully independent.
    """

    @staticmethod
    def _apply_layout_cfg(n_questions: int) -> Tuple[int, int]:
        """Return (n_col_groups, rows_per_group) for a given question count."""
        if n_questions <= 20:
            n_cg = 2
        else:
            n_cg = 4
        n_rpg = (n_questions + n_cg - 1) // n_cg
        CFG["n_col_groups"]   = n_cg
        CFG["rows_per_group"] = n_rpg
        CFG["n_questions"]    = n_questions
        return n_cg, n_rpg

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
            n_questions: int = 100,
            debug_dir: Optional[pathlib.Path] = None,
            ) -> FinalResultContract:

        stem      = pathlib.Path(image_path).stem
        debug_dir = debug_dir or pathlib.Path(f"detect_v2_{stem}")
        debug_dir.mkdir(parents=True, exist_ok=True)

        n_cg, n_rpg = self._apply_layout_cfg(n_questions)

        print(f"\n{'='*64}")
        print(f"  OMR Pipeline v2 (geometry-first)   {image_path}")
        print(f"  questions={n_questions}  cols={n_cg}  rows_per_col={n_rpg}")
        print(f"  debug dir: {debug_dir}/")
        print(f"{'='*64}")
        start = time.time()

        try:
            print("[1] Input Layer ...")
            in_data = self.s1.process(image_path, debug_dir)

            print("[2] Preprocessing Engine ...")
            prep = self.s2.process(in_data)
            if prep.quality_score < 0.05:
                print("  [WARN] Very low quality — image may be unreadable")

            print("[3] Alignment Engine ...")
            align = self.s3.process(prep, debug_dir)
            if align.alignment_confidence < 0.5:
                print("  [WARN] Low alignment confidence")

            print("[4] Static Layout Engine ...")
            layout = self.s4.process(align, debug_dir, n_col_groups=n_cg)

            print("[5] Bubble Grid Engine ...")
            grid = self.s5.process(layout, debug_dir,
                                   n_questions=n_questions,
                                   rows_per_group=n_rpg,
                                   n_choices=CFG["n_choices"])

            print("[6] Bubble Classifier ...")
            cls_result = self.s6.process(grid, debug_dir)

            print("[7] Answer Logic Engine ...")
            validation = self.s7.process(cls_result,
                                          n_questions=n_questions,
                                          rows_per_group=n_rpg,
                                          n_col_groups=n_cg)

            print("[8] Storage Engine ...")
            result = self.s8.process(
                validation, cls_result, layout, align,
                image_path, debug_dir, start, n_questions,
            )

        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            err_payload = {
                "image": image_path,
                "status": "ERROR",
                "error": str(exc),
                "processing_metrics": {"time_ms": elapsed},
            }
            err_path = debug_dir / "result_v2_error.json"
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(err_payload, f, indent=2)
            print(f"\n[FAIL] Pipeline error: {exc}")
            raise

        elapsed = result.processing_metrics["time_ms"]
        print(f"\n[OK] Done in {elapsed} ms")
        print(f"     Answered:      {result.answered} / {n_questions}")
        print(f"     Unanswered:    {result.unanswered[:10]}"
              f"{'...' if len(result.unanswered) > 10 else ''}")
        print(f"     Double-marked: {result.double_marked}")
        print(f"     Ambiguous:     {result.ambiguous[:10]}"
              f"{'...' if len(result.ambiguous) > 10 else ''}")
        print(f"     Valid:         {result.valid}")
        print(f"     Confidence:    {result.confidence_metrics}")
        print(f"     JSON:          {debug_dir}/result_v2.json")
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="OMR Detector Enhanced v2 — geometry-first Arabic answer sheet"
    )
    ap.add_argument("image",     help="Input answer sheet image (JPEG/PNG/…)")
    ap.add_argument("--questions", type=int, default=100,
                    help="Number of questions (default: 100)")
    ap.add_argument("--debug-dir", type=str, default=None,
                    help="Override debug output directory")
    ap.add_argument("--choices",   type=int, default=5,
                    help="Number of choices per question (default: 5)")
    ap.add_argument("--no-denoise", action="store_true",
                    help="Disable bilateral denoising in Stage 2")
    ap.add_argument("--no-wb", action="store_true",
                    help="Disable grey-world white balance in Stage 2")
    args = ap.parse_args()

    CFG["n_choices"] = args.choices

    pipeline   = OMRPipelineV2()
    debug_path = pathlib.Path(args.debug_dir) if args.debug_dir else None

    # Override preprocessing flags
    _orig_process = pipeline.s2.process
    def _patched_preprocess(data):
        return _orig_process(data,
                             white_balance=not args.no_wb,
                             denoise=not args.no_denoise)
    pipeline.s2.process = _patched_preprocess

    result = pipeline.run(args.image, n_questions=args.questions,
                          debug_dir=debug_path)

    print("\n--- Answer Table ---")
    for row in range(0, args.questions, 10):
        qs = range(row + 1, min(row + 11, args.questions + 1))
        print("  " + "  ".join(
            f"Q{q:3d}:{result.answers.get(str(q)) or '-'}" for q in qs
        ))


if __name__ == "__main__":
    main()

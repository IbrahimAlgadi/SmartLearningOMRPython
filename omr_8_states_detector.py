#!/usr/bin/env python3
"""
OMR 8-Stage Modular Detector — Arabic Answer Sheet (100 questions)

Architecture: 8 stateless stages, each communicating via a strict @dataclass contract.
Every stage takes a contract, produces a contract — no shared mutable state.

Stages:
  1. InputLayer              — load & normalise raw image
  2. PreprocessingEngine     — CLAHE illumination normalisation
  3. AlignmentEngine         — document detection + perspective warp
  4. LayoutUnderstandingModel— anchor verification + QR decode → logical regions
  5. BubbleDetectionModel    — HoughCircles inside answer grid → located bubbles
  6. AnswerClassificationModel — shading-corrected fill ratio → fill probabilities
  7. PostProcessingEngine    — 3-tier decision rules → final answers + flags
  8. StorageEngine           — build result JSON + debug images

Usage:
  python omr_8_states_detector.py ar-student-answer.jpeg
  python omr_8_states_detector.py ar-student-answer.jpeg --questions 100
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CFG: Dict[str, Any] = {
    # Perspective-warp output size (750 × 1060 → 1 px ≈ 0.28 mm)
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

    # Mid-edge anchors: 7 mm side, flush to edge → centre at 3.5 mm
    "anchor_ml_x":      int(3.5  / 210 * 750),         # ≈ 12 px
    "anchor_mr_x":      int((210 - 3.5) / 210 * 750),  # ≈ 738 px
    "anchor_mid_y":     int(148.5 / 297 * 1060),        # ≈ 530 px
    "anchor_search_r_mid": 40,
    "anchor_min_area_mid": 150,
    "anchor_max_area_mid": 2500,

    # Column-top dots: 6 mm, centred in each of 4 grid columns
    # Content padding 18 mm each side → grid x: 64–686 px, 4 cols × 155.5 px
    "col_dot_ys":  [int(244 / 1060 * 1060)],           # single expected y ≈ 244 px
    "col_dot_xs":  [int(x / 750 * 750) for x in [142, 297, 453, 608]],
    "col_dot_search_r": 35,
    "col_dot_min_area": 100,
    "col_dot_max_area": 2000,

    # Answer grid bounds relative to anchor centres
    "grid_anchor_top":   0.205,
    "grid_anchor_bot":   0.975,
    "grid_anchor_left":  0.06,
    "grid_anchor_right": 0.06,

    # Fallback fractions when anchors are unavailable
    "grid_x0": 0.10,
    "grid_x1": 0.90,
    "grid_y0": 0.247,
    "grid_y1": 0.94,

    # QR code region (top-left area, RTL sheet)
    "qr_x0": 0.08,
    "qr_x1": 0.42,
    "qr_y0": 0.08,
    "qr_y1": 0.25,

    # HoughCircles — bubble radius ≈ 8–9 px in warped image
    "hough_dp":       1.2,
    "hough_min_dist": 14,
    "hough_param1":   40,
    "hough_param2":   18,
    "hough_min_r":    6,
    "hough_max_r":    16,

    # NMS
    "nms_dist": 14,

    # Fill classification (shading corrected)
    "fill_norm_thr":  185,
    "fill_ratio_thr": 0.35,

    # Layout
    "n_col_groups":   4,
    "rows_per_group": 25,
    "n_questions":    100,
    "n_choices":      5,   # bubbles per row (A B C D E / أ ب ج د هـ)
}

# RTL choice mapping — rank from rightmost (0) → answer letter
RANK_TO_CHOICE: Dict[int, str] = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

CHOICE_COLORS: Dict[Optional[str], Tuple[int, int, int]] = {
    "A": (0, 220, 80),
    "B": (255, 165, 0),
    "C": (0, 200, 255),
    "D": (200, 0, 255),
    "E": (0, 80, 255),
    None: (120, 120, 120),
}

# ══════════════════════════════════════════════════════════════════════════════
#  ① DATA CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class InputContract:
    """Raw image + source metadata."""
    image_path: str
    image: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class PreprocessContract:
    """Illumination-normalised grayscale image."""
    original_bgr: np.ndarray        # keep original for downstream colour ops
    enhanced_gray: np.ndarray       # CLAHE-processed grayscale
    quality_score: float            # 0–1; placeholder for future DL quality model


@dataclass
class AlignedRegion:
    """A named logical region with pixel bbox in warped-image coordinates."""
    name: str                       # e.g. "answer_grid", "qr_area"
    region_type: str                # "answer_grid" | "qr" | "student_id"
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class AlignmentContract:
    """Perspective-corrected BGR image + how the warp was achieved."""
    aligned_bgr: np.ndarray
    transform_matrix: Optional[np.ndarray]
    align_method: str               # "anchor-based" | "canny-contour" | "full-image"
    alignment_confidence: float


@dataclass
class LayoutContract:
    """Logical regions derived from anchor positions + QR decoded data."""
    aligned_bgr: np.ndarray         # pass-through for downstream stages
    regions: List[AlignedRegion]
    anchors: Dict[str, Tuple[int, int]]
    student_code: str
    exam_code: str
    qr_raw: List[str]
    qr_method: str
    layout_confidence: float
    # Precise bubble-zone right edges per column (col_idx → warped-x).
    # Populated when the sheet has printed bubble-zone bars (new v3 sheets).
    # Empty dict for old sheets — detector falls back to CT-anchor estimation.
    bubble_zone_xs: Dict[int, int] = field(default_factory=dict)


@dataclass
class DetectedBubble:
    """A single detected circle — location only, no fill decision yet."""
    cx: int
    cy: int
    r: int
    col_group: int                  # 0 = leftmost group (Q76-100) … 3 = rightmost (Q1-25)
    row_idx: int                    # 0-based row within the column group
    question_id: str                # derived: e.g. "1", "25", "76"


@dataclass
class BubblesContract:
    """All located bubbles, sorted into the col_group × row grid."""
    aligned_bgr: np.ndarray
    bubbles: List[DetectedBubble]
    raw_circles: List[Dict[str, Any]]   # raw Hough output for classification stage
    detection_confidence: float
    col_regions: List["AlignedRegion"] = field(default_factory=list)
    ct_xs: List[int] = field(default_factory=list)   # CT anchor X positions (sorted)


@dataclass
class BubblePrediction:
    """Fill state prediction for one bubble."""
    question_id: str
    option: str                     # "A" | "B" | "C" | "D" | "E"
    fill_ratio: float               # raw shading-corrected fill value
    probabilities: Dict[str, float] # {"filled": 0.92, "empty": 0.08}
    status: str                     # "filled" | "empty"
    yolo_conf: float = 0.0          # YOLO detection confidence (0 = no YOLO)


@dataclass
class ClassificationContract:
    """Per-bubble fill predictions for every detected bubble."""
    aligned_bgr: np.ndarray
    predictions: List[BubblePrediction]
    bubbles: List[DetectedBubble]   # pass-through for visualisation
    raw_circles: List[Dict[str, Any]]
    classification_confidence: float


@dataclass
class QuestionResult:
    """Post-processed answer for a single question."""
    question_id: str
    choice: Optional[str]           # "A"–"D", None = unanswered
    fill: float
    tier: str                       # "absolute" | "relative" | "dominant" | "none"
    note: str                       # "" | "double_mark" | "unanswered" | "low_confidence"
    all_filled: List[str]           # all choices flagged as filled (for double-mark)
    row_mean_fill: float
    n_choices: int


@dataclass
class ValidationContract:
    """Final answers, flags, and optional score."""
    answers: Dict[str, Optional[str]]       # q_str → choice or None
    details: Dict[str, QuestionResult]
    flags: List[str]                        # e.g. ["MULTIPLE_Q3", "EMPTY_Q7"]
    unanswered: List[int]
    double_marked: List[int]
    score: Optional[int]


@dataclass
class FinalResultContract:
    """Persisted JSON payload."""
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
    low_confidence: List[int]
    valid: bool
    debug_dir: str
    processing_metrics: Dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED DEBUG HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _dbg(img: np.ndarray, name: str, debug_dir: pathlib.Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / name), img)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — INPUT LAYER
# ══════════════════════════════════════════════════════════════════════════════

class InputLayer:
    """
    Role: Accept any image source (file path) and produce a normalised
    InputContract. Everything downstream is source-agnostic.
    """

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
            metadata={
                "source": "file",
                "resolution": (w, h),
                "timestamp": time.time(),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — PREPROCESSING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class PreprocessingEngine:
    """
    Role: Improve image quality without destroying signal.
    CLAHE handles uneven phone-camera lighting.
    Keeps the original BGR for stages that need colour.
    """

    def process(self, data: InputContract) -> PreprocessContract:
        gray  = cv2.cvtColor(data.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Simple sharpness proxy — placeholder for a DL quality classifier
        lap     = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality = float(np.clip(lap / 500.0, 0.0, 1.0))

        print(f"  [2-preprocess] quality_score={quality:.3f}")
        return PreprocessContract(
            original_bgr=data.image,
            enhanced_gray=enhanced,
            quality_score=quality,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — ALIGNMENT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AlignmentEngine:
    """
    Role: Normalise geometry — detect paper boundary, correct perspective.

    Priority:
      0. YOLO doc_model   — if models/doc_model.pt exists, use it (most robust)
      1. anchor-based     — find the 4 black OMR anchor squares globally
      2. canny-contour    — Canny edge + largest quadrilateral
      3. full-image       — 10 px inset fallback

    Drop a trained doc_model.pt into the models/ folder (same dir as this script)
    and YOLO will be used automatically, with the OpenCV methods as fallback.
    """

    # Path relative to this script's directory
    _YOLO_MODEL_PATH = pathlib.Path(__file__).parent / "models" / "doc_model.pt"
    _yolo_model = None   # lazy-loaded

    @classmethod
    def _load_yolo(cls):
        if cls._yolo_model is not None:
            return cls._yolo_model
        if not cls._YOLO_MODEL_PATH.exists():
            return None
        try:
            from ultralytics import YOLO  # type: ignore
            cls._yolo_model = YOLO(str(cls._YOLO_MODEL_PATH))
            print(f"  [3-align] YOLO doc_model loaded: {cls._YOLO_MODEL_PATH}")
            return cls._yolo_model
        except Exception as e:
            print(f"  [3-align] YOLO load failed ({e}) — falling back to OpenCV")
            return None

    def _detect_paper_yolo(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Use YOLO to detect the largest 'paper' bounding box.
        Returns 4 corner points as np.float32 shape (4,2) or None.
        """
        model = self._load_yolo()
        if model is None:
            return None
        try:
            results = model(img, verbose=False)[0]
            if results.boxes is None or len(results.boxes) == 0:
                return None
            # Pick highest-confidence paper box
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            best  = boxes[confs.argmax()]
            x0, y0, x1, y1 = best[:4]
            # Return as 4 corners (TL, TR, BR, BL)
            return np.float32([
                [x0, y0], [x1, y0],
                [x1, y1], [x0, y1],
            ])
        except Exception as e:
            print(f"  [3-align] YOLO inference error: {e}")
            return None

    # ── internal geometry helpers ────────────────────────────────────────────

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # TL
        rect[2] = pts[np.argmax(s)]   # BR
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]   # TR
        rect[3] = pts[np.argmax(d)]   # BL
        return rect

    @staticmethod
    def _warp(img: np.ndarray, rect: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        dst = np.float32([[0, 0], [out_w - 1, 0],
                          [out_w - 1, out_h - 1], [0, out_h - 1]])
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (out_w, out_h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

    def _find_anchors_global(self, img: np.ndarray) -> Optional[Dict[str, Tuple[int, int]]]:
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255,
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
            ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            if ratio < 0.65:
                continue
            hull = cv2.convexHull(c)
            if area / (cv2.contourArea(hull) + 1e-6) < 0.70:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            candidates.append((cx, cy, area))

        if len(candidates) < 4:
            return None

        candidates.sort(key=lambda x: x[2], reverse=True)
        best_quad, max_area = None, 0

        for quad in itertools.combinations(candidates[:12], 4):
            pts = np.float32([[p[0], p[1]] for p in quad])
            ordered = self._order_points(pts)
            qa = cv2.contourArea(ordered)
            if qa <= max_area:
                continue
            tl, tr, br, bl = ordered
            w_top = np.linalg.norm(tr - tl)
            w_bot = np.linalg.norm(br - bl)
            h_l   = np.linalg.norm(bl - tl)
            h_r   = np.linalg.norm(br - tr)
            if (min(w_top, w_bot) / (max(w_top, w_bot) + 1e-6) > 0.7 and
                    min(h_l, h_r) / (max(h_l, h_r) + 1e-6) > 0.7):
                max_area = qa
                best_quad = ordered

        if best_quad is not None and max_area > W * H * 0.05:
            return {
                "TL": tuple(best_quad[0].astype(int)),
                "TR": tuple(best_quad[1].astype(int)),
                "BR": tuple(best_quad[2].astype(int)),
                "BL": tuple(best_quad[3].astype(int)),
            }
        return None

    def _find_document_corners(self, gray: np.ndarray) -> Optional[np.ndarray]:
        H, W = gray.shape
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        for lo, hi in [(50, 150), (30, 100), (10, 60)]:
            edges = cv2.Canny(blur, lo, hi)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, k, iterations=1)
            cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts[:10]:
                if cv2.contourArea(c) < 0.35 * H * W:
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    return approx.reshape(4, 2).astype(np.float32)
        return None

    # ── public entry point ───────────────────────────────────────────────────

    def process(self, prep: PreprocessContract,
                debug_dir: pathlib.Path) -> AlignmentContract:
        img  = prep.original_bgr
        gray = prep.enhanced_gray
        W, H = CFG["warp_w"], CFG["warp_h"]
        Ih, Iw = gray.shape

        method = "full-image"
        rect   = None
        warped = None
        M_out  = None

        # Priority 0: YOLO doc_model (if available)
        yolo_corners = self._detect_paper_yolo(img)
        if yolo_corners is not None:
            method = "yolo-doc"
            rect   = self._order_points(yolo_corners)
            warped = self._warp(img, rect, W, H)
            M_out  = cv2.getPerspectiveTransform(
                rect,
                np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
            )

        # Priority 1: anchor-based
        if warped is None:
            orig_anchors = self._find_anchors_global(img)
        else:
            orig_anchors = None

        if orig_anchors:
            method = "anchor-based"
        if orig_anchors:
            method = "anchor-based"
            dst_pts = np.float32([
                [CFG["anchor_cx_left"],  CFG["anchor_cy_top"]],
                [CFG["anchor_cx_right"], CFG["anchor_cy_top"]],
                [CFG["anchor_cx_right"], CFG["anchor_cy_bottom"]],
                [CFG["anchor_cx_left"],  CFG["anchor_cy_bottom"]],
            ])
            src_pts = np.float32([orig_anchors["TL"], orig_anchors["TR"],
                                   orig_anchors["BR"], orig_anchors["BL"]])
            M_out  = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M_out, (W, H),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            rect = src_pts

        # Priority 2: Canny contour
        if warped is None:
            corners = self._find_document_corners(gray)
            if corners is not None:
                method = "canny-contour"
                rect   = self._order_points(corners)
                warped = self._warp(img, rect, W, H)

        # Priority 3: full-image fallback
        if warped is None:
            print("  [3-align] WARN: both methods failed — full-image fallback")
            rect = self._order_points(np.float32([
                [10, 10], [Iw - 10, 10],
                [Iw - 10, Ih - 10], [10, Ih - 10],
            ]))
            warped = self._warp(img, rect, W, H)

        # Confidence proxy: anchor-based is most reliable
        conf = {"yolo-doc": 0.98, "anchor-based": 0.95, "canny-contour": 0.80, "full-image": 0.50}[method]

        # Debug: annotate original image with detected corners
        vis = img.copy()
        tl, tr, br, bl = [tuple(p.astype(int)) for p in rect]
        cv2.polylines(vis, [rect.astype(np.int32)], True, (0, 255, 80), 3)
        for pt, col, lbl in zip([tl, tr, br, bl],
                                 [(0, 0, 255), (0, 200, 0), (0, 165, 255), (255, 100, 0)],
                                 ["TL", "TR", "BR", "BL"]):
            cv2.circle(vis, pt, 12, col, -1)
            cv2.putText(vis, lbl, (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(vis, method, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        _dbg(vis,    "01_doc_corners.jpg", debug_dir)
        _dbg(warped, "02_warped.jpg",      debug_dir)

        print(f"  [3-align] method={method}  out={W}x{H}  conf={conf:.2f}")
        return AlignmentContract(
            aligned_bgr=warped,
            transform_matrix=M_out,
            align_method=method,
            alignment_confidence=conf,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — LAYOUT UNDERSTANDING MODEL
# ══════════════════════════════════════════════════════════════════════════════

class LayoutUnderstandingModel:
    """
    Role: Convert aligned image → structured exam layout.

    Responsibilities:
      • Detect logical regions (answer_grid, qr_area, student_id_block)
      • Verify the 4 corner anchor squares in the warped image
      • Decode QR codes (student + exam)

    Detection priority for region bboxes:
      0. YOLO layout_model  — if models/layout_model.pt exists
      1. Anchor-based fractions — current OpenCV baseline

    Everything downstream uses AlignedRegion bboxes — no hardcoded fractions.
    """

    _YOLO_MODEL_PATH = pathlib.Path(__file__).parent / "models" / "layout_model.pt"
    _yolo_model = None

    # YOLO class names → AlignedRegion.region_type mapping
    _YOLO_CLASS_MAP = {
        "answer_grid":      "answer_grid",
        "qr_area":          "qr",
        "student_id_block": "student_id",
    }

    @classmethod
    def _load_yolo(cls):
        if cls._yolo_model is not None:
            return cls._yolo_model
        if not cls._YOLO_MODEL_PATH.exists():
            return None
        try:
            from ultralytics import YOLO  # type: ignore
            cls._yolo_model = YOLO(str(cls._YOLO_MODEL_PATH))
            print(f"  [4-layout] YOLO layout_model loaded: {cls._YOLO_MODEL_PATH}")
            return cls._yolo_model
        except Exception as e:
            print(f"  [4-layout] YOLO load failed ({e}) — falling back to anchor fractions")
            return None

    def _detect_regions_yolo(self, warped: np.ndarray) -> Optional[List[AlignedRegion]]:
        """
        Run YOLO layout model; return a list of AlignedRegion or None if
        the model is unavailable or finds nothing useful.
        """
        model = self._load_yolo()
        if model is None:
            return None
        try:
            results = model(warped, verbose=False)[0]
            if results.boxes is None or len(results.boxes) == 0:
                return None

            regions: List[AlignedRegion] = []
            for box, cls_id, conf in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.cls.cpu().numpy().astype(int),
                results.boxes.conf.cpu().numpy(),
            ):
                if conf < 0.30:
                    continue
                cls_name = results.names[cls_id]
                rtype    = self._YOLO_CLASS_MAP.get(cls_name, cls_name)
                x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                regions.append(AlignedRegion(cls_name, rtype, x0, y0, x1, y1))

            if not regions:
                return None
            print(f"  [4-layout] YOLO regions: "
                  + " ".join(r.name for r in regions))
            return regions
        except Exception as e:
            print(f"  [4-layout] YOLO inference error: {e}")
            return None

    # ── anchor verification ──────────────────────────────────────────────────

    def _find_anchor_blob(self, patch: np.ndarray) -> Optional[Tuple[int, int]]:
        return self._find_anchor_blob_ex(patch,
                                        CFG["anchor_min_area"],
                                        CFG["anchor_max_area"])

    def _find_anchor_blob_ex(
        self, patch: np.ndarray, min_area: float, max_area: float
    ) -> Optional[Tuple[int, int]]:
        _, bw = cv2.threshold(patch, 60, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            area = cv2.contourArea(c)
            if not (min_area < area < max_area):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if min(w, h) / (max(w, h) + 1e-6) < 0.50:   # slightly relaxed for col-dots
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
                        debug_dir: pathlib.Path) -> Dict[str, Tuple[int, int]]:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        H, W = warped.shape[:2]
        r    = CFG["anchor_search_r"]
        expected = {
            "TL": (CFG["anchor_cx_left"],   CFG["anchor_cy_top"]),
            "TR": (CFG["anchor_cx_right"],  CFG["anchor_cy_top"]),
            "BR": (CFG["anchor_cx_right"],  CFG["anchor_cy_bottom"]),
            "BL": (CFG["anchor_cx_left"],   CFG["anchor_cy_bottom"]),
            # mid-edge anchors (7 mm flush to page edge)
            "ML": (CFG["anchor_ml_x"],      CFG["anchor_mid_y"]),
            "MR": (CFG["anchor_mr_x"],      CFG["anchor_mid_y"]),
        }
        vis = _to_bgr(gray)
        anchors: Dict[str, Tuple[int, int]] = {}

        for name, (ex, ey) in expected.items():
            sr  = CFG["anchor_search_r_mid"] if name in ("ML", "MR") else r
            min_a = (CFG["anchor_min_area_mid"] if name in ("ML", "MR")
                     else CFG["anchor_min_area"])
            max_a = (CFG["anchor_max_area_mid"] if name in ("ML", "MR")
                     else CFG["anchor_max_area"])
            x0 = max(0, ex - sr); x1 = min(W, ex + sr)
            y0 = max(0, ey - sr); y1 = min(H, ey + sr)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 200, 0), 2)
            local = self._find_anchor_blob_ex(gray[y0:y1, x0:x1], min_a, max_a)
            if local:
                cx, cy = x0 + local[0], y0 + local[1]
                anchors[name] = (cx, cy)
                cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(vis, name, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            else:
                anchors[name] = (ex, ey)
                cv2.circle(vis, (ex, ey), 8, (0, 0, 255), -1)
                cv2.putText(vis, f"{name}?", (ex + 5, ey - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # ── Column-top dot detection ──────────────────────────────────────────
        # Look for 4 small square blobs near the expected col-top positions.
        # Their Y gives the precise grid top; their X gives column boundaries.
        sr_c   = CFG["col_dot_search_r"]
        min_ac = CFG["col_dot_min_area"]
        max_ac = CFG["col_dot_max_area"]
        col_dot_ys = CFG["col_dot_ys"]
        ey_col = col_dot_ys[0] if col_dot_ys else int(H * 0.23)
        col_dots_found: List[Tuple[str, Tuple[int, int]]] = []
        for i, ex_c in enumerate(CFG["col_dot_xs"]):
            name_c = f"CT{i}"
            x0 = max(0, ex_c - sr_c); x1 = min(W, ex_c + sr_c)
            y0 = max(0, ey_col - sr_c); y1 = min(H, ey_col + sr_c)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (180, 100, 0), 1)
            local = self._find_anchor_blob_ex(gray[y0:y1, x0:x1], min_ac, max_ac)
            if local:
                cx, cy = x0 + local[0], y0 + local[1]
                anchors[name_c] = (cx, cy)
                col_dots_found.append((name_c, (cx, cy)))
                cv2.circle(vis, (cx, cy), 6, (0, 255, 180), -1)
                cv2.putText(vis, name_c, (cx + 4, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1)

        _dbg(vis, "03_anchor_search.jpg", debug_dir)

        vis_anch = warped.copy()
        for name, (cx, cy) in anchors.items():
            color = (0, 255, 255) if name in ("TL","TR","BL","BR") else (
                    (255, 200, 0) if name in ("ML","MR") else (0, 255, 180))
            r_vis = 20 if name in ("TL","TR","BL","BR") else 10
            cv2.circle(vis_anch, (cx, cy), r_vis, color, 3)
            cv2.putText(vis_anch, name, (cx - 30, cy - r_vis - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        _dbg(vis_anch, "04_anchors.jpg", debug_dir)

        detected = sorted(anchors.keys())
        print(f"  [4-layout] anchors detected: {detected}")
        return anchors

    # ── QR decoding ──────────────────────────────────────────────────────────

    def _decode_qr(self, warped: np.ndarray,
                   debug_dir: pathlib.Path) -> Dict[str, Any]:
        H, W = warped.shape[:2]
        x0 = int(W * CFG["qr_x0"]); x1 = int(W * CFG["qr_x1"])
        y0 = int(H * CFG["qr_y0"]); y1 = int(H * CFG["qr_y1"])
        region = warped[y0:y1, x0:x1]
        _dbg(region, "05_qr_regions.jpg", debug_dir)

        rw = region.shape[1]
        all_codes: List[str] = []
        pyzbar_ok = False

        # ── pyzbar (best) ────────────────────────────────────────────────────
        try:
            from pyzbar import pyzbar as pzb  # type: ignore

            def _try_pyzbar(img_bgr: np.ndarray) -> List[str]:
                g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                found: List[str] = []
                for scale in [2, 3, 1]:
                    gs = cv2.resize(g, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_CUBIC) if scale > 1 else g
                    _, bw1 = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                    _, bw2 = cv2.threshold(clahe.apply(gs), 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    bw3 = cv2.adaptiveThreshold(gs, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 15, 4)
                    for bw in [bw1, bw2, bw3]:
                        for obj in pzb.decode(bw):
                            v = obj.data.decode("utf-8", errors="ignore").strip()
                            if v and v not in found:
                                found.append(v)
                    if found:
                        return found
                return found

            for sub in [region, region[:, :rw // 2], region[:, rw // 2:]]:
                for code in _try_pyzbar(sub):
                    if code not in all_codes:
                        all_codes.append(code)
                if len(all_codes) >= 2:
                    break
            pyzbar_ok = bool(all_codes)

        except Exception as e:
            print(f"  [4-layout/qr] pyzbar unavailable: {e}")

        # ── OpenCV fallback ──────────────────────────────────────────────────
        if not all_codes:
            detector = cv2.QRCodeDetector()
            gray_reg = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            for scale in [3, 2, 1]:
                g = (cv2.resize(gray_reg, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
                     if scale > 1 else gray_reg)
                _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                val, *_ = detector.detectAndDecode(bw)
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

        method = "pyzbar" if pyzbar_ok else ("opencv" if all_codes else "none")
        print(f"  [4-layout/qr] method={method}  student={student_code}  exam={exam_code}")
        return {
            "student_code": student_code,
            "exam_code": exam_code,
            "qr_raw": all_codes,
            "qr_method": method,
        }

    # ── bubble-zone bar detection ─────────────────────────────────────────────

    def _detect_bubble_zone_bar(
        self,
        warped: np.ndarray,
        col_regions: List[AlignedRegion],
        gy0: int,
        gy1: int = -1,
    ) -> Dict[int, int]:
        """
        Detect the horizontal bubble-zone bars printed at the top and bottom
        of each column group on new-style sheets.

        The bar spans exactly the bubble zone (.bs flex-1) while a transparent
        spacer sits beside the number zone (.qn).  In RTL layout the bar is on
        the LEFT, so its RIGHT edge = the precise bubble/number boundary.

        Scans both the TOP bar (near gy0) and BOTTOM bar (near gy1), and
        uses an absolute dark-pixel count per x-column (not a ratio) so the
        result is robust to scan-strip height variations and sheet tilt.

        Returns {col_idx: bar_right_x_in_warped} for all detected columns.
        """
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        H    = warped.shape[0]
        if gy1 < 0:
            gy1 = H

        def _scan_strip(anchor_y: int) -> Dict[int, int]:
            """Scan ±25 px around anchor_y; return {col_idx: right_x}."""
            sy0 = max(0, anchor_y - 25)
            sy1 = min(H, anchor_y + 30)
            if sy0 >= sy1:
                return {}
            strip = gray[sy0:sy1, :]
            found: Dict[int, int] = {}
            for col_idx, cr in enumerate(col_regions):
                col_strip  = strip[:, cr.x0:cr.x1]
                if col_strip.shape[1] < 10:
                    continue
                # Absolute count: a column belongs to the bar if ≥ 8 of its
                # pixels across the scan strip are dark (< 80).
                dark_count = np.sum(col_strip < 80, axis=0)
                xs = np.where(dark_count >= 8)[0]
                if len(xs) < 8:
                    continue
                bar_x1_abs = cr.x0 + int(xs[-1])
                bar_width  = int(xs[-1] - xs[0])
                if bar_width < (cr.x1 - cr.x0) * 0.15:
                    continue
                found[col_idx] = bar_x1_abs
                print(f"  [4-layout] BZ bar col_{col_idx} "
                      f"(y≈{anchor_y}): x=[{cr.x0+int(xs[0])}, {bar_x1_abs}]  "
                      f"w={bar_width}px")
            return found

        top_hits = _scan_strip(gy0)
        bot_hits = _scan_strip(gy1)

        # Merge: prefer top; use bottom as fallback; average when both found
        result: Dict[int, int] = {}
        all_cols = set(top_hits) | set(bot_hits)
        for i in all_cols:
            if i in top_hits and i in bot_hits:
                result[i] = (top_hits[i] + bot_hits[i]) // 2   # average for robustness
            else:
                result[i] = top_hits.get(i, bot_hits.get(i, 0))

        return result

    # ── region derivation ─────────────────────────────────────────────────────

    def _derive_regions(self, warped: np.ndarray,
                        anchors: Dict[str, Tuple[int, int]]
                        ) -> Tuple[List[AlignedRegion], Dict[int, int]]:
        H, W = warped.shape[:2]
        regions: List[AlignedRegion] = []

        corner_keys = {"TL", "TR", "BL", "BR"}
        have_corners = corner_keys.issubset(anchors)

        if have_corners:
            ay_top   = min(anchors["TL"][1], anchors["TR"][1])
            ay_bot   = max(anchors["BL"][1], anchors["BR"][1])
            ax_left  = min(anchors["TL"][0], anchors["BL"][0])
            ax_right = max(anchors["TR"][0], anchors["BR"][0])
            ah = ay_bot - ay_top
            aw = ax_right - ax_left

            # --- X boundaries: use ML/MR if detected, else fraction --------
            if "ML" in anchors and "MR" in anchors:
                gx0 = anchors["ML"][0] + 20   # step 20 px inward from ML
                gx1 = anchors["MR"][0] - 20
            else:
                gx0 = int(ax_left  + aw * CFG["grid_anchor_left"])
                gx1 = int(ax_right - aw * CFG["grid_anchor_right"])

            # --- Y top: use col-top dots if ≥ 2 detected, else fraction ----
            ct_ys = [anchors[k][1] for k in anchors if k.startswith("CT")]
            if len(ct_ys) >= 2:
                # Grid starts just above the col-top dots (minus 1 dot radius)
                gy0 = int(min(ct_ys)) - 12
            else:
                gy0 = int(ay_top + ah * CFG["grid_anchor_top"])

            # --- Y bottom: fraction from corners ----------------------------
            gy1 = int(ay_top + ah * CFG["grid_anchor_bot"])

        else:
            gx0 = int(W * CFG["grid_x0"]); gx1 = int(W * CFG["grid_x1"])
            gy0 = int(H * CFG["grid_y0"]); gy1 = int(H * CFG["grid_y1"])

        # Clamp to image
        gx0 = max(0, gx0); gy0 = max(0, gy0)
        gx1 = min(W, gx1); gy1 = min(H, gy1)

        regions.append(AlignedRegion("answer_grid", "answer_grid", gx0, gy0, gx1, gy1))

        # Store column X boundaries in regions if col-top dots were detected
        ct_xs = sorted([anchors[k][0] for k in anchors if k.startswith("CT")])
        if len(ct_xs) >= 2:
            # CT anchors sit above/near the right side (question-number column) of
            # each RTL column group.  Using them DIRECTLY as boundaries would split
            # every column in half and let question-number digits bleed into the
            # adjacent group's filter zone.
            #
            # Correct approach: use MIDPOINTS between adjacent CT anchors as the
            # inter-column boundaries so each col_group gets its full width.
            mids       = [(ct_xs[i] + ct_xs[i + 1]) // 2 for i in range(len(ct_xs) - 1)]
            boundaries = [gx0] + mids + [gx1]          # n_ct - 1 + 2 = n_ct + 1 → n_ct regions
            for i in range(len(boundaries) - 1):
                regions.append(AlignedRegion(
                    f"col_group_{i}", "col_group",
                    boundaries[i], gy0, boundaries[i + 1], gy1,
                ))
            print(f"  [4-layout] col-group X boundaries (CT): {ct_xs}")
            print(f"  [4-layout] col-group X boundaries (mid): {mids}")

        qr_x0 = int(W * CFG["qr_x0"]); qr_x1 = int(W * CFG["qr_x1"])
        qr_y0 = int(H * CFG["qr_y0"]); qr_y1 = int(H * CFG["qr_y1"])
        regions.append(AlignedRegion("qr_area", "qr", qr_x0, qr_y0, qr_x1, qr_y1))

        print(f"  [4-layout] answer_grid  x0={gx0} y0={gy0} x1={gx1} y1={gy1}")

        # ── Detect printed bubble-zone bars (new-style sheets only) ──────────
        col_group_regions = [r for r in regions if r.region_type == "col_group"]
        bubble_zone_xs: Dict[int, int] = {}
        if col_group_regions:
            bubble_zone_xs = self._detect_bubble_zone_bar(
                warped, col_group_regions, gy0, gy1)
            if bubble_zone_xs:
                print(f"  [4-layout] bubble-zone bars detected: "
                      + ", ".join(f"col_{i}→x={x}" for i, x in sorted(bubble_zone_xs.items())))
            else:
                print("  [4-layout] no bubble-zone bars found — using CT-anchor fallback")

        return regions, bubble_zone_xs

    # ── public entry point ───────────────────────────────────────────────────

    def process(self, align: AlignmentContract,
                debug_dir: pathlib.Path) -> LayoutContract:
        warped  = align.aligned_bgr
        anchors = self._verify_anchors(warped, debug_dir)
        qr      = self._decode_qr(warped, debug_dir)

        # Anchor quality: "strong" = 4 corners + at least 2 col-top dots
        corner_keys = {"TL", "TR", "BL", "BR"}
        ct_found    = [k for k in anchors if k.startswith("CT")]
        strong_anchors = corner_keys.issubset(anchors) and len(ct_found) >= 2

        if strong_anchors:
            # Anchor-based derivation is more precise for v3 sheets
            regions, bubble_zone_xs = self._derive_regions(warped, anchors)
            conf    = 0.98
            print(f"  [4-layout] using anchor-derived regions "
                  f"(corners + {len(ct_found)} col-top dots)")
            # Still run YOLO to pick up qr/student-id regions it detects
            yolo_regions = self._detect_regions_yolo(warped)
            if yolo_regions:
                non_grid = [r for r in yolo_regions
                            if r.region_type not in ("answer_grid",)]
                regions = [r for r in regions
                           if r.region_type == "answer_grid"] + non_grid
        else:
            # Fallback: YOLO layout model → anchor fractions
            bubble_zone_xs: Dict[int, int] = {}
            yolo_regions = self._detect_regions_yolo(warped)
            if yolo_regions is not None:
                regions = yolo_regions
                conf    = 0.97
            else:
                regions, bubble_zone_xs = self._derive_regions(warped, anchors)
                conf    = 0.90 if corner_keys.issubset(anchors) else 0.70

        return LayoutContract(
            aligned_bgr=warped,
            regions=regions,
            anchors=anchors,
            student_code=qr["student_code"],
            exam_code=qr["exam_code"],
            qr_raw=qr["qr_raw"],
            qr_method=qr["qr_method"],
            layout_confidence=conf,
            bubble_zone_xs=bubble_zone_xs,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — BUBBLE DETECTION MODEL
# ══════════════════════════════════════════════════════════════════════════════

class BubbleDetectionModel:
    """
    Role: Find WHERE all candidate answer bubbles are.
    Does NOT decide whether they are filled — that is Stage 6's job.

    Detection priority:
      0. YOLO bubble_model — if models/bubble_model.pt exists
         Outputs bboxes for bubble_filled AND bubble_empty classes.
         The fill status from YOLO feeds directly into Stage 6 probs.
      1. HoughCircles + NMS + K-means (current OpenCV baseline)

    Responsibilities:
      • Locate every bubble bbox within the answer_grid region
      • K-means cluster into (col_group × row) cells
      • Assign each bubble a question_id
    """

    _YOLO_MODEL_PATH = pathlib.Path(__file__).parent / "models" / "bubble_model_v2.pt"
    _yolo_model = None

    @classmethod
    def _load_yolo(cls):
        if cls._yolo_model is not None:
            return cls._yolo_model
        if not cls._YOLO_MODEL_PATH.exists():
            return None
        try:
            from ultralytics import YOLO  # type: ignore
            cls._yolo_model = YOLO(str(cls._YOLO_MODEL_PATH))
            print(f"  [5-bubbles] YOLO bubble_model loaded: {cls._YOLO_MODEL_PATH}")
            return cls._yolo_model
        except Exception as e:
            print(f"  [5-bubbles] YOLO load failed ({e}) — falling back to HoughCircles")
            return None

    def _detect_bubbles_yolo(
        self, region: np.ndarray, offset_x: int, offset_y: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Run YOLO on the answer-grid crop.
        Returns a list of raw circle dicts {cx, cy, r, yolo_filled} or None.
        """
        model = self._load_yolo()
        if model is None:
            return None
        try:
            results = model(region, verbose=False, max_det=600, conf=0.28)[0]
            if results.boxes is None or len(results.boxes) == 0:
                return None

            circles: List[Dict[str, Any]] = []
            for box, cls_id, conf in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.cls.cpu().numpy().astype(int),
                results.boxes.conf.cpu().numpy(),
            ):
                if conf < 0.25:
                    continue
                x0, y0, x1, y1 = box[:4]
                cx    = int((x0 + x1) / 2) + offset_x
                cy    = int((y0 + y1) / 2) + offset_y
                r     = int(max(x1 - x0, y1 - y0) / 2)
                label = results.names[cls_id]
                circles.append({
                    "cx": cx, "cy": cy, "r": r,
                    "yolo_filled": label == "bubble_filled",
                    "yolo_conf":   float(conf),
                })
            return circles if circles else None
        except Exception as e:
            print(f"  [5-bubbles] YOLO inference error: {e}")
            return None

    # ── NMS ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _nms(circles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not circles:
            return circles
        nd = CFG["nms_dist"]
        sorted_c = sorted(circles, key=lambda c: c["r"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for cand in sorted_c:
            if any(np.hypot(cand["cx"] - k["cx"], cand["cy"] - k["cy"]) < nd
                   for k in kept):
                continue
            kept.append(cand)
        return kept

    # ── YOLO fill-status overlay ─────────────────────────────────────────────

    @staticmethod
    def _overlay_yolo_fill(
        hough_circles: List[Dict[str, Any]],
        yolo_circles:  List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For every HoughCircle, find the nearest YOLO detection within 1.5r.
        If found, tag the circle with yolo_filled / yolo_conf so the
        classifier can use it.  Positions always come from HoughCircles.
        """
        for hc in hough_circles:
            best_conf   = 0.0
            best_filled = None
            for yc in yolo_circles:
                dist = np.hypot(hc["cx"] - yc["cx"], hc["cy"] - yc["cy"])
                if dist < hc["r"] * 1.5 and yc["yolo_conf"] > best_conf:
                    best_conf   = yc["yolo_conf"]
                    best_filled = yc["yolo_filled"]
            if best_filled is not None:
                hc["yolo_filled"] = best_filled
                hc["yolo_conf"]   = best_conf
        return hough_circles

    # ── HoughCircles ─────────────────────────────────────────────────────────

    def _hough(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        # CLAHE normalises local contrast — recovers circles in watermark /
        # low-contrast regions that plain GaussianBlur cannot handle.
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
        raw = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=CFG["hough_dp"],
            minDist=CFG["hough_min_dist"],
            param1=CFG["hough_param1"],
            param2=CFG["hough_param2"],
            minRadius=CFG["hough_min_r"],
            maxRadius=CFG["hough_max_r"],
        )
        if raw is None:
            return []
        H, W = gray.shape
        results = []
        for (cx, cy, r) in np.uint16(np.around(raw[0])):
            cx, cy, r = int(cx), int(cy), int(r)
            if cx - r < 0 or cy - r < 0 or cx + r >= W or cy + r >= H:
                continue
            results.append({"cx": cx, "cy": cy, "r": r})
        return self._nms(results)

    # ── 1D K-means clustering ─────────────────────────────────────────────────

    @staticmethod
    def _kmeans1d(values: List[float], k: int) -> Tuple[List[int], List[float]]:
        if len(values) < k:
            k = max(1, len(values))
        arr = np.array(values, dtype=np.float32).reshape(-1, 1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.5)
        best_labels, best_centers, best_compact = None, None, float("inf")
        for _ in range(6):
            compact, labels, centers = cv2.kmeans(
                arr, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
            if compact < best_compact:
                best_compact = compact
                best_labels  = labels.flatten().tolist()
                best_centers = centers.flatten().tolist()
        sorted_pairs = sorted(enumerate(best_centers), key=lambda x: x[1])
        old_to_new   = {old: new for new, (old, _) in enumerate(sorted_pairs)}
        new_labels   = [old_to_new[l] for l in best_labels]
        new_centers  = [c for _, c in sorted_pairs]
        return new_labels, new_centers

    # ── cluster into grid ─────────────────────────────────────────────────────

    def _cluster(self, circles: List[Dict[str, Any]],
                 n_col_groups: int,
                 rows_per_group: int,
                 rows_per_col: Optional[List[int]] = None,
                 ) -> Dict[int, Dict[int, List[Dict[str, Any]]]]:
        if not circles:
            return {}
        xs = [c["cx"] for c in circles]
        col_labels, _ = self._kmeans1d(xs, n_col_groups)

        col_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for circle, col_label in zip(circles, col_labels):
            col_groups[col_label].append(circle)

        result: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
        for col_idx in range(n_col_groups):
            group = col_groups.get(col_idx, [])
            if not group:
                result[col_idx] = {}
                continue
            ys = [c["cy"] for c in group]
            # Use per-column actual row count to avoid merging rows in short columns
            actual_rows = (rows_per_col[col_idx]
                           if rows_per_col and col_idx < len(rows_per_col)
                           else rows_per_group)
            k_rows = max(1, min(actual_rows, len(group)))
            row_labels, _ = self._kmeans1d(ys, k_rows)
            rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for circle, row_label in zip(group, row_labels):
                rows[row_label].append(circle)
            result[col_idx] = dict(rows)

        return result

    # ── per-row pruning to max n_choices bubbles ──────────────────────────────

    def _trim_rows(
        self,
        result: Dict[int, Dict[int, List[Dict[str, Any]]]],
        n_choices: int = 4,
    ) -> Dict[int, Dict[int, List[Dict[str, Any]]]]:
        """
        Cap each row at n_choices circles by keeping the one closest to each
        expected X centroid.  Handles cases where YOLO double-detects a bubble
        (e.g. two overlapping detections for the same filled circle).
        """
        trimmed: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
        for col_idx, rows in result.items():
            all_circles = [c for row_cs in rows.values() for c in row_cs]
            if len(all_circles) < n_choices:
                trimmed[col_idx] = rows
                continue

            xs = [c["cx"] for c in all_circles]
            _, x_centers = self._kmeans1d(xs, min(n_choices, len(set(xs))))
            x_centers = sorted(x_centers)

            new_rows: Dict[int, List[Dict[str, Any]]] = {}
            for row_idx, row_cs in rows.items():
                if len(row_cs) <= n_choices:
                    new_rows[row_idx] = row_cs
                    continue
                # Assign one circle per expected X centroid (greedy nearest)
                selected: List[Dict[str, Any]] = []
                used: set = set()
                for xc in x_centers:
                    best_j, best_d = -1, float("inf")
                    for j, c in enumerate(row_cs):
                        if j in used:
                            continue
                        d = abs(c["cx"] - xc)
                        if d < best_d:
                            best_d, best_j = d, j
                    if best_j >= 0:
                        selected.append(row_cs[best_j])
                        used.add(best_j)
                new_rows[row_idx] = selected
            trimmed[col_idx] = new_rows

        return trimmed

    @staticmethod
    def _preprocess_col_crop(crop_bgr: np.ndarray) -> np.ndarray:
        """
        White out horizontal dark strips (bar, mid-column separator) and the
        ~30 px immediately following each strip (column choice-letter headers).
        This prevents YOLO from detecting the bar ink and Arabic choice letters
        (أ ب ج د ه) as bubbles, which would shift the row clustering.
        """
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        dark_per_row = np.sum(gray < 80, axis=1)
        # Solid bars (5 mm black) fill nearly the entire row width.
        # Empty bubble rings reach at most ~28% of row width (5 circles × 2×3px outline
        # in a ~110px crop).  Filled bubbles at most ~42%.  Use 0.55 so only bars and
        # solid separator lines are masked, never actual bubble rows.
        is_strip = dark_per_row > W * 0.55

        result    = crop_bgr.copy()
        last_dark = -99

        for y in range(H):
            if is_strip[y]:
                result[y, :] = 255
                last_dark = y
            elif y - last_dark <= 30:
                # header letters follow immediately after a strip
                result[y, :] = 255

        return result

    # ── bar detection helper ────────────────────────────────────────────────────

    @staticmethod
    def _detect_bars(
        gray_warped: np.ndarray,
        x0: int, y0: int, x1: int, y1: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect the printed bubble-zone bars.

        Returns a list of (bar_x0, bar_x1, col_top_y, col_bot_y) tuples
        sorted by bar_x0, one per column group.

        bar_x0 / bar_x1   : left / right edge of the bubble zone (right = mask boundary)
        col_top_y          : top y of the top bar  → answer area y0 for this column
        col_bot_y          : bottom y of the bottom bar → answer area y1 for this column

        The bars sit ABOVE/BELOW the YOLO answer_grid bounding box (YOLO was trained
        on sheets without bars, so it doesn't include them in its grid bounding box).
        We scan above y0 and below y1.  Minimum run width = max(80px, 25% col_width)
        to reject single-bubble arcs (~26 px) from being misdetected as bars.
        """
        H, W = gray_warped.shape[:2]
        col_width  = (x1 - x0) / max(1, 4)
        min_bar_w  = max(80, int(col_width * 0.25))

        BarRun = Tuple[int, int, int, int]  # (abs_x0, abs_x1, abs_y0, abs_y1)

        def _runs_at(sy0: int, sy1: int) -> List[BarRun]:
            if sy0 < 0 or sy1 > H or sy0 >= sy1:
                return []
            strip    = gray_warped[sy0:sy1, x0:x1]
            dark_col = np.sum(strip < 80, axis=0)
            is_bar   = dark_col >= 8

            x_runs: List[Tuple[int, int]] = []
            in_run, rstart = False, 0
            for xi, b in enumerate(is_bar):
                if b and not in_run:
                    in_run, rstart = True, xi
                elif not b and in_run:
                    in_run = False
                    if xi - rstart >= min_bar_w:
                        x_runs.append((rstart, xi - 1))
            if in_run and len(is_bar) - rstart >= min_bar_w:
                x_runs.append((rstart, len(is_bar) - 1))

            result: List[BarRun] = []
            for rx0_rel, rx1_rel in x_runs:
                # find y extent of bar within this run
                run_strip  = strip[:, rx0_rel:rx1_rel + 1]
                dark_row   = np.sum(run_strip < 80, axis=1)
                bar_rows   = np.where(dark_row >= max(1, (rx1_rel - rx0_rel) * 0.25))[0]
                if len(bar_rows):
                    by0 = sy0 + int(bar_rows[0])
                    by1 = sy0 + int(bar_rows[-1])
                else:
                    by0, by1 = sy0, sy1
                result.append((x0 + rx0_rel, x0 + rx1_rel, by0, by1))
            return result

        # Bars sit just outside the YOLO answer_grid boundary
        top_runs = _runs_at(max(0, y0 - 80), max(0, y0 - 2))
        bot_runs = _runs_at(min(H, y1 + 2),  min(H, y1 + 80))
        # Fallback: try inside grid boundary if nothing found outside
        if not top_runs:
            top_runs = _runs_at(max(0, y0 - 5), min(H, y0 + 35))
        if not bot_runs:
            bot_runs = _runs_at(max(0, y1 - 35), min(H, y1 + 5))

        if not top_runs and not bot_runs:
            return []

        # Merge top + bottom runs by x-centre proximity, average x extents
        if not bot_runs:
            # Use top bar y for both top and bottom (best effort)
            return sorted([(tx0, tx1, ty0, ty1) for tx0, tx1, ty0, ty1 in top_runs],
                          key=lambda r: r[0])
        if not top_runs:
            return sorted([(bx0, bx1, by0, by1) for bx0, bx1, by0, by1 in bot_runs],
                          key=lambda r: r[0])

        merged: List[BarRun] = []
        used_bot: set = set()
        for tx0, tx1, ty0, ty1 in top_runs:
            tc = (tx0 + tx1) // 2
            best, best_d = -1, 80
            for j, (bx0, bx1, by0, by1) in enumerate(bot_runs):
                d = abs((bx0 + bx1) // 2 - tc)
                if d < best_d:
                    best_d, best = d, j
            if best >= 0 and best not in used_bot:
                bx0, bx1, by0, by1 = bot_runs[best]
                merged.append(((tx0 + bx0) // 2, (tx1 + bx1) // 2, ty0, by1))
                used_bot.add(best)
            else:
                merged.append((tx0, tx1, ty0, ty1))
        for j, run in enumerate(bot_runs):
            if j not in used_bot:
                merged.append(run)

        return sorted(merged, key=lambda r: r[0])

    # ── public entry point ────────────────────────────────────────────────────

    def process(self, layout: LayoutContract,
                debug_dir: pathlib.Path) -> BubblesContract:
        warped = layout.aligned_bgr
        H, W   = warped.shape[:2]

        # Extract answer_grid region
        grid_region = next((r for r in layout.regions
                            if r.region_type == "answer_grid"), None)
        if grid_region:
            x0, y0, x1, y1 = grid_region.x0, grid_region.y0, grid_region.x1, grid_region.y1
        else:
            x0 = int(W * CFG["grid_x0"]); x1 = int(W * CFG["grid_x1"])
            y0 = int(H * CFG["grid_y0"]); y1 = int(H * CFG["grid_y1"])

        # Debug: YOLO grid boundary in grey (bars will be drawn later in orange)
        vis_area = warped.copy()
        cv2.rectangle(vis_area, (x0, y0), (x1, y1), (128, 128, 128), 1)
        cv2.putText(vis_area, "YOLO grid", (x0 + 4, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (128, 128, 128), 1)
        # Shade the bar scan zones (above + below grid)
        for sy0_d, sy1_d in [(max(0, y0-80), max(0, y0-2)),
                              (min(H, y1+2),  min(H, y1+80))]:
            ov = vis_area.copy()
            cv2.rectangle(ov, (x0, sy0_d), (x1, sy1_d), (255, 200, 0), -1)
            cv2.addWeighted(ov, 0.28, vis_area, 0.72, 0, vis_area)
        cv2.putText(vis_area, "bar scan zone", (x0 + 4, max(4, y0 - 84)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 140, 0), 1)
        # (vis_area will be saved as 07_answer_area.jpg after bar detection below)

        # Initial region slice (may be expanded after bar detection)
        region   = warped[y0:y1, x0:x1]
        gray_reg = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        n_cg  = CFG["n_col_groups"]
        n_rpg = CFG["rows_per_group"]
        n_ch  = CFG["n_choices"]

        # ── Step 1: Detect bubble-zone bars → column count + boundaries ────────
        gray_full = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        bars = self._detect_bars(gray_full, x0, y0, x1, y1)
        # bars = [(bar_x0, bar_x1), ...] sorted left→right, one per column.
        # bar_x0 = leftmost dark pixel of bar  (≈ bubble zone left edge)
        # bar_x1 = rightmost dark pixel of bar (≈ bubble zone RIGHT edge = mask boundary)

        if bars and len(bars) in (2, 4):
            n_detected = len(bars)
            if n_detected != n_cg:
                print(f"  [5-bubbles] bar count ({n_detected}) overrides "
                      f"n_col_groups ({n_cg})")
                n_cg  = n_detected
                n_rpg = (CFG["n_questions"] + n_cg - 1) // n_cg
                CFG["n_col_groups"]   = n_cg
                CFG["rows_per_group"] = n_rpg
            print(f"  [5-bubbles] bars detected: {len(bars)}  "
                  + ", ".join(f"col_{i}=[{b[0]},{b[1]}] y=[{b[2]},{b[3]}]"
                               for i, b in enumerate(bars)))

        expected      = n_cg * n_rpg * n_ch
        min_yolo_prim = max(20, int(expected * 0.50))

        # ── Step 2: Build col_group regions from bars (or LAYOUT anchors) ──────
        # bar = (bar_x0, bar_x1, col_top_y, col_bot_y)
        # col answer area: x = [bar_x0 .. next bar_x0]  y = [col_top_y .. col_bot_y]
        # bubble_right_xs[i] = bar_x1  (right edge of bar = masking boundary)
        if bars and len(bars) == n_cg:
            col_left_edges  = [b[0] for b in bars]
            col_right_edges = col_left_edges[1:] + [x1]
            col_top_ys      = [b[2] for b in bars]
            col_bot_ys      = [b[3] for b in bars]
            col_regions = [
                AlignedRegion(f"col_group_{i}", "col_group",
                              col_left_edges[i], col_top_ys[i],
                              col_right_edges[i], col_bot_ys[i])
                for i in range(n_cg)
            ]
            bubble_right_xs = [b[1] for b in bars]
            print(f"  [5-bubbles] bubble zone: bar-detected  "
                  + ", ".join(f"col_{i}←{bx}" for i, bx in enumerate(bubble_right_xs)))
            # Expand region to cover bars (they're outside YOLO's y0/y1)
            ey0 = min(b[2] for b in bars)
            ey1 = max(b[3] for b in bars)
            if ey0 < y0 or ey1 > y1:
                y0, y1   = min(y0, ey0), max(y1, ey1)
                region   = warped[y0:y1, x0:x1]
                gray_reg = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                # Update col_regions y with expanded bounds
                col_regions = [
                    AlignedRegion(cr.name, cr.region_type,
                                  cr.x0, bars[i][2], cr.x1, bars[i][3])
                    for i, cr in enumerate(col_regions)
                ]
        else:
            # Fallback: use layout anchors or equal-width split
            col_regions = sorted(
                [r for r in layout.regions if r.region_type == "col_group"],
                key=lambda r: r.x0,
            )
            ct_xs_sorted = sorted(
                [v[0] for k, v in layout.anchors.items() if k.startswith("CT")]
            )
            if not col_regions:
                col_w = (x1 - x0) // n_cg
                col_regions = [
                    AlignedRegion(f"col_group_{i}", "col_group",
                                  x0 + i * col_w, y0,
                                  x0 + (i + 1) * col_w, y1)
                    for i in range(n_cg)
                ]
                print(f"  [5-bubbles] col_group regions: equal-width fallback")

            # Bubble zone right edges: prefer layout contract bars, then CT anchors
            if layout.bubble_zone_xs and len(layout.bubble_zone_xs) == n_cg:
                bubble_right_xs = [layout.bubble_zone_xs[i] for i in range(n_cg)]
                print(f"  [5-bubbles] bubble zone: layout-contract bars  "
                      + ", ".join(f"col_{i}←{x}" for i, x in enumerate(bubble_right_xs)))
            elif ct_xs_sorted and len(ct_xs_sorted) == n_cg:
                bubble_right_xs = ct_xs_sorted
                print(f"  [5-bubbles] bubble zone: CT-anchor fallback  {ct_xs_sorted}")
            else:
                bubble_right_xs = []
                print("  [5-bubbles] bubble zone: no boundary info — no masking")

        # ── Step 3: Debug images — 06 bar detection, 07 answer area ─────────────
        # Both show the same visual: orange = bubble zone, grey = full col, nums label

        def _draw_col_zones(canvas: np.ndarray) -> np.ndarray:
            for i, (bx0, bx1, cty, cby) in enumerate(bars):
                col_right = bars[i + 1][0] if i + 1 < len(bars) else x1
                # Grey: full column span
                cv2.rectangle(canvas, (bx0, cty), (col_right, cby), (160, 160, 160), 1)
                # Green tint: bubble zone
                ov = canvas.copy()
                cv2.rectangle(ov, (bx0, cty), (bx1, cby), (0, 200, 80), -1)
                cv2.addWeighted(ov, 0.12, canvas, 0.88, 0, canvas)
                # Orange border: bubble zone answer area
                cv2.rectangle(canvas, (bx0, cty), (bx1, cby), (0, 100, 255), 2)
                # Green bar highlight at top/bottom
                cv2.rectangle(canvas, (bx0, cty), (bx1, cty + 4), (0, 200, 80), -1)
                cv2.rectangle(canvas, (bx0, cby - 4), (bx1, cby), (0, 200, 80), -1)
                # Red line at masking boundary
                cv2.line(canvas, (bx1, cty), (bx1, cby), (0, 60, 255), 2)
                cv2.putText(canvas, f"col_{i} bub=[{bx0},{bx1}]",
                            (bx0 + 3, cty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 80), 1)
                cv2.putText(canvas, "nums", (bx1 + 3, cty + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
            if not bars:
                cv2.putText(canvas, "NO BARS DETECTED", (x0 + 10, y0 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(canvas, f"bars={len(bars)}  cols={n_cg}", (x0, y0 - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
            return canvas

        # 06_bar_detection.jpg — scan strips + detected bars
        vis_bars = warped.copy()
        cv2.rectangle(vis_bars, (x0, y0), (x1, y1), (128, 128, 128), 1)  # YOLO grid ref
        for sy0_d, sy1_d in [(max(0, y0 - 80), max(0, y0 - 2)),
                              (min(H, y1 + 2),  min(H, y1 + 80))]:
            ov = vis_bars.copy()
            cv2.rectangle(ov, (x0, sy0_d), (x1, sy1_d), (255, 200, 0), -1)
            cv2.addWeighted(ov, 0.30, vis_bars, 0.70, 0, vis_bars)
        _draw_col_zones(vis_bars)
        _dbg(vis_bars, "06_bar_detection.jpg", debug_dir)

        # 07_answer_area.jpg — col zones + each crop highlighted
        _draw_col_zones(vis_area)
        for i, (bx0, bx1, cty, cby) in enumerate(bars):
            ov = vis_area.copy()
            cv2.rectangle(ov, (bx0, cty), (bx1, cby), (0, 200, 80), -1)
            cv2.addWeighted(ov, 0.08, vis_area, 0.92, 0, vis_area)
            cv2.putText(vis_area, f"YOLO crop {i}", (bx0 + 4, cty + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 60, 200), 1)
        _dbg(vis_area, "07_answer_area.jpg", debug_dir)

        # ── Step 4: YOLO per-column crops (no masking needed — crop IS the bubble zone) ──
        yolo_circles: List[Dict[str, Any]] = []
        if bars and len(bars) == n_cg:
            for i, (bx0, bx1, cty, cby) in enumerate(bars):
                col_crop   = warped[cty:cby, bx0:bx1]
                col_clean  = self._preprocess_col_crop(col_crop)
                col_yolo   = self._detect_bubbles_yolo(col_clean, bx0, cty)

                # HoughCircles gap-fill: add any circle YOLO missed
                gray_col   = cv2.cvtColor(col_clean, cv2.COLOR_BGR2GRAY)
                col_hough  = [dict(c, cx=c["cx"] + bx0, cy=c["cy"] + cty)
                              for c in self._hough(gray_col)]
                added = 0
                for hc in col_hough:
                    if all(abs(hc["cx"] - yc["cx"]) > 14 or
                           abs(hc["cy"] - yc["cy"]) > 14
                           for yc in col_yolo):
                        col_yolo.append(hc)
                        added += 1
                print(f"  [5-bubbles] YOLO col_{i}: {len(col_yolo)-added} det "
                      f"+ {added} Hough gap-fill  (crop {bx1-bx0}×{cby-cty}px)")
                yolo_circles.extend(col_yolo)
        else:
            # Fallback: single region with number-zone masking (old sheets / no bars)
            yolo_region = region.copy()
            if bubble_right_xs and col_regions:
                for cr, bx in zip(col_regions, bubble_right_xs):
                    dz_x0 = max(0,               bx   - x0)
                    dz_x1 = min(region.shape[1], cr.x1 - x0)
                    if dz_x0 < dz_x1:
                        yolo_region[:, dz_x0:dz_x1] = 255
                print(f"  [5-bubbles] number-zone mask: "
                      + ", ".join(f"[{bx}–{cr.x1}]"
                                   for cr, bx in zip(col_regions, bubble_right_xs)))
            yolo_circles = self._detect_bubbles_yolo(yolo_region, x0, y0)
            print(f"  [5-bubbles] YOLO primary: {len(yolo_circles)} detections")
        if yolo_circles and len(yolo_circles) >= min_yolo_prim:
            raw_global = yolo_circles
            print(f"  [5-bubbles] YOLO primary: {len(yolo_circles)} detections")
        else:
            # Fallback: HoughCircles for location, YOLO overlay for fill status
            raw_local  = self._hough(gray_reg)
            raw_global = [dict(c, cx=c["cx"] + x0, cy=c["cy"] + y0)
                          for c in raw_local]
            print(f"  [5-bubbles] HoughCircles fallback: {len(raw_global)} bubbles"
                  f"  (YOLO had {len(yolo_circles) if yolo_circles else 0})")
            if yolo_circles:
                raw_global = self._overlay_yolo_fill(raw_global, yolo_circles)
                print("  [5-bubbles] YOLO fill overlay applied")
            else:
                print("  [5-bubbles] fill-ratio classification only")

        # Debug: raw detections (may include false-positives on printed numbers)
        vis_raw = warped.copy()
        for c in raw_global:
            if "yolo_filled" in c:
                color = (0, 200, 80) if c["yolo_filled"] else (0, 140, 255)
                tag   = f"{c['yolo_conf']:.2f}"
            else:
                color = (160, 160, 160)
                tag   = "?"
            cv2.circle(vis_raw, (c["cx"], c["cy"]), c["r"], color, 2)
            cv2.putText(vis_raw, tag,
                        (c["cx"] - 8, c["cy"] - c["r"] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, color, 1)
        cv2.putText(vis_raw, f"raw={len(raw_global)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
        _dbg(vis_raw, "08a_circles_raw.jpg", debug_dir)

        # Cluster into grid — per-column actual row count handles short last columns
        n_questions = CFG["n_questions"]
        rows_per_col = [
            min(n_rpg, max(0, n_questions - (n_cg - 1 - col_idx) * n_rpg))
            for col_idx in range(n_cg)
        ]
        grid = self._cluster(raw_global, n_cg, n_rpg, rows_per_col)
        # _trim_rows caps rows at n_choices if YOLO double-detects a bubble
        grid = self._trim_rows(grid, n_ch)

        # Build DetectedBubble list — skip any row that falls beyond n_questions
        bubbles: List[DetectedBubble] = []
        for col_idx in range(n_cg):
            base_q = (n_cg - 1 - col_idx) * n_rpg + 1
            for row_idx, row_circles in grid.get(col_idx, {}).items():
                q_num = base_q + row_idx
                if q_num > n_questions:
                    continue
                for c in row_circles:
                    bubbles.append(DetectedBubble(
                        cx=c["cx"], cy=c["cy"], r=c["r"],
                        col_group=col_idx, row_idx=row_idx,
                        question_id=str(q_num),
                    ))

        for cg, rows in grid.items():
            print(f"  [5-bubbles] col_group={cg}  rows={len(rows)}"
                  f"  circles={sum(len(v) for v in rows.values())}")
        print(f"  [5-bubbles] total={len(raw_global)}  "
              f"after cluster={len(bubbles)}")

        return BubblesContract(
            aligned_bgr=warped,
            bubbles=bubbles,
            raw_circles=raw_global,
            detection_confidence=min(1.0, len(raw_global) / (n_cg * n_rpg * 4) * 1.1),
            col_regions=col_regions,
            ct_xs=bubble_right_xs,   # precise right edge per col (bar or CT fallback)
        )

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 6 — ANSWER CLASSIFICATION MODEL
# ══════════════════════════════════════════════════════════════════════════════

class AnswerClassificationModel:
    """
    Role: Determine the fill state of each bubble using a shading-corrected
    fill ratio.  Outputs probabilities, not binary decisions.

    Design: Separate WHERE from WHETHER — Stage 5 found the circles,
    Stage 6 decides how dark they are.
    """

    def _classify_circles(self, gray: np.ndarray,
                          circles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add fill ratio and filled flag to each raw circle dict.

        If the circle already has a 'yolo_filled' key (set by BubbleDetectionModel
        when running the YOLO bubble_model), that value takes precedence over the
        fill-ratio threshold — the OpenCV fill ratio is still computed and stored
        for analytics / confidence scoring but the binary filled decision uses YOLO.
        """
        H, W = gray.shape

        # Shading correction — local background map via large median blur
        bg   = cv2.medianBlur(gray, 71)
        bg   = np.maximum(bg, 1)
        norm = np.clip((gray.astype(np.float32) / bg.astype(np.float32)) * 255,
                       0, 255).astype(np.uint8)
        dark_binary = (norm < CFG["fill_norm_thr"]).astype(np.uint8)

        enriched = []
        for c in circles:
            cx, cy, r = c["cx"], c["cy"], c["r"]
            if cx - r < 0 or cy - r < 0 or cx + r >= W or cy + r >= H:
                enriched.append(dict(c, fill=0.0, filled=False))
                continue
            mask = np.zeros((H, W), np.uint8)
            cv2.circle(mask, (cx, cy), max(r - 2, 2), 255, -1)   # exclude border ring
            nz = np.count_nonzero(mask)
            if nz == 0:
                enriched.append(dict(c, fill=0.0, filled=False))
                continue
            dark = int(np.sum(dark_binary[mask > 0]))
            fill = dark / nz

            # YOLO label overrides threshold-based decision only when confident
            if "yolo_filled" in c and c.get("yolo_conf", 0) >= 0.25:
                filled = c["yolo_filled"]
            else:
                filled = fill > CFG["fill_ratio_thr"]

            enriched.append(dict(c, fill=round(fill, 3), filled=filled))
        return enriched

    def process(self, detection: BubblesContract,
                debug_dir: pathlib.Path) -> ClassificationContract:
        warped = detection.aligned_bgr
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Mask the question-number zone in the gray image so digit ink doesn't
        # bleed into adjacent bubble fill measurements.  detection.ct_xs now
        # carries the precise bubble-zone right edges (bar-detected on new sheets,
        # CT-anchor fallback on old sheets).
        if detection.col_regions and detection.ct_xs and \
                len(detection.col_regions) == len(detection.ct_xs):
            gray = gray.copy()
            for cr, bx in zip(detection.col_regions, detection.ct_xs):
                if bx < cr.x1:
                    gray[:, bx:cr.x1] = 255   # treat as white (unfilled)

        # Enrich raw circles with fill data
        enriched = self._classify_circles(gray, detection.raw_circles)

        # Build lookup: (cx, cy) → enriched circle
        lookup = {(c["cx"], c["cy"]): c for c in enriched}

        # Debug: draw filled (green) vs empty (orange) on warped
        vis = warped.copy()
        for c in enriched:
            color = (0, 200, 0) if c["filled"] else (0, 140, 255)
            cv2.circle(vis, (c["cx"], c["cy"]), c["r"], color, 2)
            cv2.putText(vis, f"{c['fill']:.2f}",
                        (c["cx"] - 14, c["cy"] - c["r"] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, color, 1)
        _dbg(vis, "08_circles_raw.jpg", debug_dir)   # overwrites Stage 5's placeholder

        # Build BubblePrediction list
        predictions: List[BubblePrediction] = []
        n_cg  = CFG["n_col_groups"]
        n_rpg = CFG["rows_per_group"]

        # Re-cluster enriched circles into grid to assign rank (choice letter)
        # Group by col_group × row_idx from the DetectedBubble list
        grid_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for b in detection.bubbles:
            ec = lookup.get((b.cx, b.cy))
            if ec:
                grid_map[(b.col_group, b.row_idx)].append(ec)

        for b in detection.bubbles:
            ec = lookup.get((b.cx, b.cy))
            if ec is None:
                ec = {"cx": b.cx, "cy": b.cy, "r": b.r, "fill": 0.0, "filled": False}

            # Determine rank within this row by X descending (RTL: rightmost = A)
            row_key = (b.col_group, b.row_idx)
            row_cs  = sorted(grid_map[row_key], key=lambda c: c["cx"], reverse=True)
            rank    = next((i for i, c in enumerate(row_cs)
                            if c["cx"] == ec["cx"] and c["cy"] == ec["cy"]), 0)
            option  = RANK_TO_CHOICE.get(rank, "?")

            fill   = ec["fill"]
            status = "filled" if ec["filled"] else "empty"
            probs  = {
                "filled": round(min(fill / CFG["fill_ratio_thr"], 1.0), 3),
                "empty":  round(max(1.0 - fill / CFG["fill_ratio_thr"], 0.0), 3),
            }
            predictions.append(BubblePrediction(
                question_id=b.question_id,
                option=option,
                fill_ratio=fill,
                probabilities=probs,
                status=status,
                yolo_conf=float(ec.get("yolo_conf", 0.0)),
            ))

        n_filled = sum(1 for p in predictions if p.status == "filled")
        n_total  = len(predictions)
        conf     = float(n_filled) / max(n_total, 1)
        print(f"  [6-classify] total={n_total}  filled={n_filled}  "
              f"fill_thr={CFG['fill_ratio_thr']}")

        # Per-question fill-ratio dump: shows top-2 fills with option labels
        q_preds: Dict[str, List] = defaultdict(list)
        for p in predictions:
            q_preds[p.question_id].append(p)
        rows_info = []
        for qid in sorted(q_preds, key=lambda x: int(x)):
            ps = sorted(q_preds[qid], key=lambda p: p.fill_ratio, reverse=True)
            if len(ps) >= 2:
                dom = f"{ps[0].option}={ps[0].fill_ratio:.2f}/{ps[1].option}={ps[1].fill_ratio:.2f}"
            elif ps:
                dom = f"{ps[0].option}={ps[0].fill_ratio:.2f}"
            else:
                dom = "?"
            rows_info.append(f"Q{qid}:{dom}")
        print("  [6-classify] fills: " + "  ".join(rows_info))

        return ClassificationContract(
            aligned_bgr=warped,
            predictions=predictions,
            bubbles=detection.bubbles,
            raw_circles=enriched,
            classification_confidence=conf,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 7 — POST-PROCESSING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class PostProcessingEngine:
    """
    Role: Convert raw probabilities → final answers.
    This layer is business logic + rules, NOT ML.

    Three-tier decision:
      Tier 1 (Absolute)  fill ≥ fill_ratio_thr
      Tier 2 (Relative)  fill ≥ row_mean + REL_MARGIN  AND  fill ≥ MIN_SIGNAL
      Tier 3 (Dominant)  one circle clearly dominates (≥ DOM_R × second-highest)
    """

    FILL_ABS_THR      = CFG["fill_ratio_thr"]  # 0.35 — adj fill must meet this (Tier 1)
    FILL_REL_MARGIN   = 0.12   # adj fill must exceed adj_mean by this (Tier 2)
    FILL_MIN_SIGNAL   = 0.20   # minimum adj fill to consider at all (Tiers 3 & 0)
    FILL_DOM_R        = 2.00   # adj max / adj 2nd must be ≥ this ratio (Tier 3)
    DOUBLE_MARK_RATIO = 0.70   # Tier-1 secondary must be ≥ this fraction of primary
                                # (prevents ink-bleed from a heavy mark being flagged)

    def process(self, cls: ClassificationContract,
                debug_dir: pathlib.Path,
                n_questions: int = 100) -> ValidationContract:
        n_cg  = CFG["n_col_groups"]
        n_rpg = CFG["rows_per_group"]

        # Group predictions by (question_id, option)
        # We need per-row (question) collections of all bubble predictions
        q_rows: Dict[str, List[BubblePrediction]] = defaultdict(list)
        for p in cls.predictions:
            q_rows[p.question_id].append(p)

        answers: Dict[str, Optional[str]] = {}
        details: Dict[str, QuestionResult] = {}

        for col_idx in range(n_cg):
            base_q      = (n_cg - 1 - col_idx) * n_rpg + 1
            actual_rows = min(n_rpg, max(0, n_questions - base_q + 1))
            for row_idx in range(actual_rows):
                q_num = base_q + row_idx
                q_str = str(q_num)

                row_preds = sorted(
                    q_rows.get(q_str, []),
                    key=lambda p: p.fill_ratio, reverse=False,
                )
                # Sort RTL: rightmost bubble first = rank 0 = choice A
                # Re-sort by option to match RANK_TO_CHOICE
                row_preds_by_option = sorted(
                    q_rows.get(q_str, []),
                    key=lambda p: list(RANK_TO_CHOICE.values()).index(p.option)
                    if p.option in RANK_TO_CHOICE.values() else 99,
                )

                if not row_preds_by_option:
                    answers[q_str] = None
                    details[q_str] = QuestionResult(
                        question_id=q_str, choice=None,
                        fill=0.0, tier="none", note="no_circles",
                        all_filled=[], row_mean_fill=0.0, n_choices=0,
                    )
                    continue

                fills     = [p.fill_ratio for p in row_preds_by_option]
                row_min   = min(fills)
                # Subtract per-row background (row minimum) so lighting
                # variation doesn't inflate the effective fill signal.
                adj_fills = [f - row_min for f in fills]
                row_mean  = float(np.mean(adj_fills))
                row_max   = max(adj_fills)
                n_choices = len(fills)

                # ── Tier 0: YOLO-confirmed ───────────────────────────────────
                yolo_hits = [
                    (i, p) for i, p in enumerate(row_preds_by_option)
                    if p.status == "filled"
                    and p.yolo_conf >= 0.30
                    and adj_fills[i] == row_max
                    and adj_fills[i] >= 0.20   # require meaningful adj-fill
                ]

                # ── Tier 1: absolute ────────────────────────────────────────
                abs_hits = [(i, p) for i, p in enumerate(row_preds_by_option)
                            if adj_fills[i] >= self.FILL_ABS_THR]

                # Within Tier-1 hits, apply a dominance ratio: a secondary hit
                # must be ≥ DOUBLE_MARK_RATIO × the primary adj_fill to count
                # as a deliberate double-mark.  Lower ratios are typically
                # ink-bleed / smear from a heavy neighbouring mark.
                if len(abs_hits) > 1:
                    sorted_abs   = sorted(abs_hits, key=lambda x: adj_fills[x[0]], reverse=True)
                    primary_adj  = adj_fills[sorted_abs[0][0]]
                    abs_hits     = [h for h in abs_hits
                                    if adj_fills[h[0]] >= primary_adj * self.DOUBLE_MARK_RATIO]

                # ── Tier 2: relative ────────────────────────────────────────
                rel_hits = [(i, p) for i, p in enumerate(row_preds_by_option)
                            if adj_fills[i] >= row_mean + self.FILL_REL_MARGIN
                            and adj_fills[i] >= self.FILL_MIN_SIGNAL]

                # Combine — only add relative hits close to the absolute winner
                abs_idxs     = {i for i, _ in abs_hits}
                primary_fill = max((adj_fills[i] for i, _ in abs_hits), default=0.0)
                extra_rel    = [
                    (i, p) for i, p in rel_hits
                    if i not in abs_idxs
                    and (primary_fill == 0.0 or adj_fills[i] >= primary_fill * 0.70)
                ]
                if abs_hits:
                    candidates = abs_hits + extra_rel
                elif len(rel_hits) > 1:
                    rel_sorted = sorted(rel_hits, key=lambda x: adj_fills[x[0]], reverse=True)
                    top_fill   = adj_fills[rel_sorted[0][0]]
                    candidates = (rel_sorted
                                  if adj_fills[rel_sorted[1][0]] >= top_fill * 0.85
                                  else [rel_sorted[0]])
                else:
                    candidates = rel_hits

                # ── Tier 3: dominant-max ────────────────────────────────────
                if not candidates and row_max >= self.FILL_MIN_SIGNAL:
                    adj_desc = sorted(adj_fills, reverse=True)
                    ratio_ok = (len(adj_desc) == 1
                                or adj_desc[1] < 1e-6          # only one non-zero
                                or adj_desc[0] >= adj_desc[1] * self.FILL_DOM_R)
                    if ratio_ok:
                        max_i = adj_fills.index(row_max)
                        candidates = [(max_i, row_preds_by_option[max_i])]

                if not candidates:
                    tier = "none"
                    # Last resort: YOLO-confirmed highest fill (light pencil marks)
                    if yolo_hits and len(yolo_hits) == 1:
                        idx, p = yolo_hits[0]
                        answers[q_str] = p.option
                        details[q_str] = QuestionResult(
                            question_id=q_str, choice=p.option,
                            fill=p.fill_ratio, tier="yolo",
                            note=f"yolo_conf={p.yolo_conf:.2f}",
                            all_filled=[p.option],
                            row_mean_fill=round(row_mean, 3),
                            n_choices=n_choices,
                        )
                        continue
                    answers[q_str] = None
                    details[q_str] = QuestionResult(
                        question_id=q_str, choice=None,
                        fill=row_max, tier=tier, note="unanswered",
                        all_filled=[], row_mean_fill=round(row_mean, 3),
                        n_choices=n_choices,
                    )
                elif len(candidates) == 1:
                    idx, p = candidates[0]
                    tier = "absolute" if abs_hits else ("relative" if rel_hits else "dominant")
                    answers[q_str] = p.option
                    details[q_str] = QuestionResult(
                        question_id=q_str, choice=p.option,
                        fill=p.fill_ratio, tier=tier, note="",
                        all_filled=[p.option],
                        row_mean_fill=round(row_mean, 3),
                        n_choices=n_choices,
                    )
                else:
                    # Multiple candidates — try to resolve via YOLO binary status.
                    # If exactly one candidate has YOLO-confirmed "filled" status,
                    # trust it (the others are high-fill-ratio noise / smear /
                    # adjacent digit ink, but YOLO correctly recognises they don't
                    # look like filled bubbles).
                    yolo_ok = [(i, p) for i, p in candidates if p.status == "filled"]
                    if len(yolo_ok) == 1:
                        idx, p = yolo_ok[0]
                        tier = "absolute" if abs_hits else "relative"
                        answers[q_str] = p.option
                        details[q_str] = QuestionResult(
                            question_id=q_str, choice=p.option,
                            fill=p.fill_ratio, tier=tier, note="yolo_resolved",
                            all_filled=[p.option],
                            row_mean_fill=round(row_mean, 3),
                            n_choices=n_choices,
                        )
                    else:
                        # True double-mark (YOLO confirms 2+ filled, or YOLO
                        # is unavailable for both) — pick highest adj_fill, flag it
                        best_i, best_p = max(candidates, key=lambda x: adj_fills[x[0]])
                        tier = "absolute" if abs_hits else "relative"
                        all_filled_opts = [p.option for _, p in candidates]
                        answers[q_str] = best_p.option
                        details[q_str] = QuestionResult(
                            question_id=q_str, choice=best_p.option,
                            fill=best_p.fill_ratio, tier=tier, note="double_mark",
                            all_filled=all_filled_opts,
                            row_mean_fill=round(row_mean, 3),
                            n_choices=n_choices,
                        )

        # Flags & summary lists
        flags:         List[str] = []
        unanswered:    List[int] = []
        double_marked: List[int] = []
        for q in range(1, n_questions + 1):
            q_str = str(q)
            d = details.get(q_str)
            if d is None or d.choice is None:
                unanswered.append(q)
                flags.append(f"EMPTY_Q{q}")
            if d and d.note == "double_mark":
                double_marked.append(q)
                flags.append(f"MULTIPLE_Q{q}")

        # Debug: classified + answered debug images
        self._draw_debug(cls.aligned_bgr, cls.bubbles, cls.raw_circles,
                         answers, details, debug_dir)

        print(f"  [7-postprocess] answered={n_questions - len(unanswered)}/{n_questions}"
              f"  double={len(double_marked)}")
        return ValidationContract(
            answers=answers,
            details=details,
            flags=flags,
            unanswered=unanswered,
            double_marked=double_marked,
            score=None,
        )

    def _draw_debug(self, warped: np.ndarray,
                    bubbles: List[DetectedBubble],
                    enriched: List[Dict[str, Any]],
                    answers: Dict[str, Optional[str]],
                    details: Dict[str, QuestionResult],
                    debug_dir: pathlib.Path) -> None:
        lookup = {(c["cx"], c["cy"]): c for c in enriched}

        # Build col_group × row_idx → sorted-by-cx-desc list (for rank)
        from collections import defaultdict as _dd
        grid_map: Dict[Tuple[int, int], List] = _dd(list)
        for b in bubbles:
            grid_map[(b.col_group, b.row_idx)].append(b)

        vis_cls = warped.copy()
        vis_ans = warped.copy()

        for b in bubbles:
            ec = lookup.get((b.cx, b.cy), {"cx": b.cx, "cy": b.cy, "r": b.r,
                                           "fill": 0.0, "filled": False})
            row_key   = (b.col_group, b.row_idx)
            row_sorted = sorted(grid_map[row_key], key=lambda bb: bb.cx, reverse=True)
            rank       = next((i for i, bb in enumerate(row_sorted)
                               if bb.cx == b.cx and bb.cy == b.cy), 0)
            choice     = RANK_TO_CHOICE.get(rank, "?")
            col        = CHOICE_COLORS.get(choice, (200, 200, 200))
            thick      = 3 if ec["filled"] else 1
            cv2.circle(vis_cls, (b.cx, b.cy), b.r, col, thick)
            cv2.putText(vis_cls, choice, (b.cx - 5, b.cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, col, 1)

            q_str    = b.question_id
            chosen   = answers.get(q_str)
            d        = details.get(q_str)
            is_dbl   = d is not None and d.note == "double_mark"
            all_fill = set(d.all_filled) if d else set()
            is_chosen       = (choice == chosen and chosen is not None)
            is_filled_other = is_dbl and choice in all_fill

            if is_chosen or is_filled_other:
                ring = (0, 0, 255) if is_dbl else (0, 220, 0)
                cv2.circle(vis_ans, (b.cx, b.cy), b.r + 4, ring, 2)
                cv2.circle(vis_ans, (b.cx, b.cy), b.r, (0, 0, 0), -1)
                cv2.putText(vis_ans, choice, (b.cx - 5, b.cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
            else:
                cv2.circle(vis_ans, (b.cx, b.cy), b.r, (200, 200, 200), 1)

        _dbg(vis_cls, "09_circles_classified.jpg", debug_dir)
        _dbg(vis_ans, "10_answered.jpg",            debug_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 8 — STORAGE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class StorageEngine:
    """
    Role: Persist results, attach analytics metadata.
    Output is 100% JSON-serialisable and schema-compatible with omr_detector.py.
    """

    def process(self, validation: ValidationContract,
                layout: LayoutContract,
                align: AlignmentContract,
                image_path: str,
                debug_dir: pathlib.Path,
                start_time: float,
                n_questions: int) -> FinalResultContract:

        elapsed_ms = int((time.time() - start_time) * 1000)
        conf_vals  = [d.fill for d in validation.details.values() if d.choice]
        conf_avg   = float(np.mean(conf_vals)) if conf_vals else 0.0
        low_conf   = [q for q in range(1, n_questions + 1)
                      if validation.details.get(str(q), QuestionResult(
                          str(q), None, 0, "none", "", [], 0, 0)).note == "low_confidence"]

        # Convert details to plain dicts for JSON
        details_json: Dict[str, Any] = {}
        for q_str, d in validation.details.items():
            details_json[q_str] = {
                "choice":        d.choice,
                "fill":          round(d.fill, 3),
                "tier":          d.tier,
                "note":          d.note,
                "all_filled":    d.all_filled,
                "row_mean_fill": d.row_mean_fill,
                "n_choices":     d.n_choices,
            }

        result = FinalResultContract(
            image=image_path,
            student_code=layout.student_code,
            exam_code=layout.exam_code,
            qr_codes_raw=layout.qr_raw,
            align_method=align.align_method,
            answers={str(q): validation.answers.get(str(q))
                     for q in range(1, n_questions + 1)},
            answer_details=details_json,
            total_questions=n_questions,
            answered=n_questions - len(validation.unanswered),
            unanswered=validation.unanswered,
            double_marked=validation.double_marked,
            low_confidence=low_conf,
            valid=(len(validation.unanswered) == 0
                   and len(validation.double_marked) == 0),
            debug_dir=str(debug_dir),
            processing_metrics={
                "time_ms":        elapsed_ms,
                "confidence_avg": round(conf_avg, 3),
                "align_method":   align.align_method,
                "qr_method":      layout.qr_method,
                "pipeline":       "omr_8_states_detector_v1",
            },
        )

        out_path = debug_dir / "result.json"
        out_dict = {
            "image":            result.image,
            "student_code":     result.student_code,
            "exam_code":        result.exam_code,
            "qr_codes_raw":     result.qr_codes_raw,
            "align_method":     result.align_method,
            "answers":          result.answers,
            "answer_details":   result.answer_details,
            "total_questions":  result.total_questions,
            "answered":         result.answered,
            "unanswered":       result.unanswered,
            "double_marked":    result.double_marked,
            "low_confidence":   result.low_confidence,
            "valid":            result.valid,
            "debug_dir":        result.debug_dir,
            "processing_metrics": result.processing_metrics,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=2)

        print(f"  [8-storage] JSON saved → {out_path}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR — OMRPipeline
# ══════════════════════════════════════════════════════════════════════════════

class OMRPipeline:
    """
    Stateless orchestrator.  Wires the 8 stages together.
    Each stage receives its input contract and the shared debug_dir.
    No global mutable state; each call is fully independent.
    """

    # ── Layout mapping: n_questions → (n_col_groups, rows_per_group) ───────────
    @staticmethod
    def _apply_layout_cfg(n_questions: int) -> None:
        """
        Override global CFG with sheet-format-specific layout parameters.
        Called at the start of every run() so all stages see correct values.
        """
        if n_questions <= 20:
            n_cg  = 2
        elif n_questions <= 50:
            n_cg  = 4
        else:
            n_cg  = 4
        n_rpg = (n_questions + n_cg - 1) // n_cg   # ceiling division

        # Column-top dot X positions in warped image (750 px wide).
        # Content spans x ≈ 64 .. 686 (622 px), equally split into n_cg columns.
        content_l, content_w = 64, 622
        col_dot_xs = [
            int(content_l + (2*i + 1) * content_w / (2 * n_cg))
            for i in range(n_cg)
        ]

        CFG["n_col_groups"]   = n_cg
        CFG["rows_per_group"] = n_rpg
        CFG["n_questions"]    = n_questions
        CFG["col_dot_xs"]     = col_dot_xs

    def __init__(self):
        self.s1_input      = InputLayer()
        self.s2_preprocess = PreprocessingEngine()
        self.s3_align      = AlignmentEngine()
        self.s4_layout     = LayoutUnderstandingModel()
        self.s5_bubbles    = BubbleDetectionModel()
        self.s6_classify   = AnswerClassificationModel()
        self.s7_postproc   = PostProcessingEngine()
        self.s8_storage    = StorageEngine()

    def run(self, image_path: str,
            n_questions: int = 100) -> FinalResultContract:
        stem      = pathlib.Path(image_path).stem
        debug_dir = pathlib.Path(f"detect_8_stage_{stem}")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Apply layout overrides for this sheet format before any stage runs
        self._apply_layout_cfg(n_questions)

        print(f"\n{'='*60}")
        print(f"  OMR 8-Stage Pipeline: {image_path}")
        print(f"  debug dir: {debug_dir}/")
        print(f"{'='*60}")
        start = time.time()

        try:
            print("[1] Input Layer ...")
            in_data = self.s1_input.process(image_path, debug_dir)

            print("[2] Preprocessing Engine ...")
            prep = self.s2_preprocess.process(in_data)
            if prep.quality_score < 0.05:
                print("  [WARN] Very low quality score — image may be unreadable")

            print("[3] Alignment Engine ...")
            align = self.s3_align.process(prep, debug_dir)
            if align.alignment_confidence < 0.5:
                print("  [WARN] Low alignment confidence — sheet may be misaligned")

            print("[4] Layout Understanding Model ...")
            layout = self.s4_layout.process(align, debug_dir)

            print("[5] Bubble Detection Model ...")
            bubbles = self.s5_bubbles.process(layout, debug_dir)
            if len(bubbles.raw_circles) < 50:
                print(f"  [WARN] Only {len(bubbles.raw_circles)} circles found. "
                      "Try adjusting hough_param2 or fill_ratio_thr in CFG.")

            print("[6] Answer Classification Model ...")
            classification = self.s6_classify.process(bubbles, debug_dir)

            print("[7] Post-processing Engine ...")
            validation = self.s7_postproc.process(classification, debug_dir,
                                                   n_questions=n_questions)

            print("[8] Storage Engine ...")
            result = self.s8_storage.process(
                validation, layout, align,
                image_path, debug_dir, start, n_questions,
            )

        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            error_payload = {
                "image": image_path,
                "status": "REVIEW_REQUIRED",
                "error_stage": type(exc).__name__,
                "error_msg": str(exc),
                "processing_metrics": {"time_ms": elapsed},
            }
            err_path = debug_dir / "result_error.json"
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(error_payload, f, indent=2)
            print(f"\n[FAIL] Pipeline error: {exc}")
            print(f"       Error payload written to {err_path}")
            raise

        print(f"\n[OK] Done in {result.processing_metrics['time_ms']} ms")
        print(f"     Answered:      {result.answered} / {n_questions}")
        print(f"     Unanswered:    {result.unanswered[:10]}"
              f"{'...' if len(result.unanswered) > 10 else ''}")
        print(f"     Double-marked: {result.double_marked}")
        print(f"     Valid:         {result.valid}")
        print(f"     JSON:          {debug_dir}/result.json")
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="OMR 8-Stage Detector — Arabic Answer Sheet"
    )
    ap.add_argument("image", help="Input answer sheet image (JPEG/PNG/etc.)")
    ap.add_argument("--questions", type=int, default=100,
                    help="Number of questions (default: 100)")
    ap.add_argument("--param2", type=int, default=None,
                    help="Override hough_param2 for circle sensitivity")
    ap.add_argument("--min-r",  type=int,   default=None,
                    help="Override hough_min_r (bubble radius lower bound)")
    ap.add_argument("--max-r",  type=int,   default=None,
                    help="Override hough_max_r (bubble radius upper bound)")
    ap.add_argument("--fill-thr", type=float, default=None,
                    help="Override fill_ratio_thr (0–1)")
    args = ap.parse_args()

    if args.param2:
        CFG["hough_param2"] = args.param2
    if args.min_r:
        CFG["hough_min_r"] = args.min_r
    if args.max_r:
        CFG["hough_max_r"] = args.max_r
    if args.fill_thr:
        CFG["fill_ratio_thr"] = args.fill_thr

    pipeline = OMRPipeline()
    result   = pipeline.run(args.image, args.questions)

    print("\n--- Answers ---")
    for row in range(0, args.questions, 10):
        qs = range(row + 1, min(row + 11, args.questions + 1))
        print("  " + "  ".join(
            f"Q{q:3d}:{result.answers.get(str(q)) or '-'}" for q in qs
        ))


if __name__ == "__main__":
    main()

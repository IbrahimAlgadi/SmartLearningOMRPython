#!/usr/bin/env python3
"""
omr_templates.py
─────────────────
Canonical sheet-template definitions for OMR v3.

Used by BOTH:
  generate_omr_dataset_v3.py  — synthetic dataset generation
  omr_detector_enhanced_v3.py — runtime detection

Three production families (matching generate_sheet_v3.py / reference scans):
  Q20_5ch   : 20 questions, 2 columns, 5 choices
  Q50_5ch   : 50 questions, 4 columns, 5 choices
  Q100_5ch  : 100 questions, 4 columns, 5 choices

Extended templates for CNN training: all (n_q, n_ch) in {20,50,100} × {2,3,4,5}.

All geometry is anchored to the physical A4 sheet in mm, then derived
in the warped 750×1060 pixel space that the v3 detector produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Physical A4 constants (mm) — matches generate_sheet_v3.py HTML/CSS
# ─────────────────────────────────────────────────────────────────────────────

A4_W_MM = 210.0
A4_H_MM = 297.0

# Corner anchors — .anchor.corner { width:12mm; height:12mm } at 4mm inset
ANCHOR_CORNER_MM       = 12.0
ANCHOR_CORNER_INSET_MM = 4.0

# Mid anchors — .anchor.mid { width:7mm; height:7mm } flush to left/right edges
ANCHOR_MID_MM = 7.0

# Content padding — .content { padding: 16mm 18mm 8mm }
CONTENT_TOP_MM  = 16.0
CONTENT_SIDE_MM = 18.0
CONTENT_BOT_MM  = 8.0

# Column-anchor bar height — .col-bubble-bar { height: 5mm }
COL_BAR_H_MM = 5.0

# Grid vertical extent.
# Calibrated so that a perspective-warped 750×1060 image from these sheets
# places the question-row grid at y=[239, 999] (matching v2 detector calibration).
#   grid_y0_mm = 239 / (1060/297) ≈ 67.0 mm from top of sheet
#   grid_y1_mm = 999 / (1060/297) ≈ 280.0 mm from top of sheet
GRID_Y0_MM = 67.0    # top of question-row area (below bars)
GRID_Y1_MM = 280.0   # bottom of question-row area (above bottom bars)

# Horizontal grid == content width (18 mm side padding each side)
GRID_X0_MM = CONTENT_SIDE_MM           # 18 mm
GRID_X1_MM = A4_W_MM - CONTENT_SIDE_MM  # 192 mm
GRID_W_MM  = GRID_X1_MM - GRID_X0_MM   # 174 mm
GRID_H_MM  = GRID_Y1_MM - GRID_Y0_MM   # 213 mm

# Bubble placement inset fractions — calibrated to match v2 detector heuristics.
# Bubble zone: [ col_x + col_w * LEFT_FRAC, col_x1 - (BASE_PX_WARP + col_w * RIGHT_FRAC) ]
BZONE_LEFT_FRAC    = 0.100   # calibrated: actual left inset ≈ span * 0.100 (~16px for span=155)
BZONE_RIGHT_FRAC   = 0.089   # unchanged from v2 calibration
BZONE_RIGHT_BASE_W = 37.0    # calibrated: actual right inset ≈ 37 + span*0.089 (~51px for span=155)

# ─────────────────────────────────────────────────────────────────────────────
#  Warped-space constants (750 × 1060 px) — detector canonical space
# ─────────────────────────────────────────────────────────────────────────────

WARP_W = 750
WARP_H = 1060

# Scale factors: physical mm → warped pixels
_SX = WARP_W / A4_W_MM   # ≈ 3.571 px/mm
_SY = WARP_H / A4_H_MM   # ≈ 3.569 px/mm


def mm_to_warp_x(mm: float) -> int:
    return round(mm * _SX)


def mm_to_warp_y(mm: float) -> int:
    return round(mm * _SY)


# Grid in warped pixels
WARP_GX0 = mm_to_warp_x(GRID_X0_MM)   # 64
WARP_GX1 = mm_to_warp_x(GRID_X1_MM)   # 686
WARP_GY0 = mm_to_warp_y(GRID_Y0_MM)   # 239
WARP_GY1 = mm_to_warp_y(GRID_Y1_MM)   # 999
WARP_GW  = WARP_GX1 - WARP_GX0        # 622
WARP_GH  = WARP_GY1 - WARP_GY0        # 760

# Corner anchor centres in warped space
WARP_ACX_L = mm_to_warp_x(ANCHOR_CORNER_INSET_MM + ANCHOR_CORNER_MM / 2)   # ≈ 36
WARP_ACX_R = mm_to_warp_x(A4_W_MM - ANCHOR_CORNER_INSET_MM - ANCHOR_CORNER_MM / 2)  # ≈ 714
WARP_ACY_T = mm_to_warp_y(ANCHOR_CORNER_INSET_MM + ANCHOR_CORNER_MM / 2)   # ≈ 36
WARP_ACY_B = mm_to_warp_y(A4_H_MM  - ANCHOR_CORNER_INSET_MM - ANCHOR_CORNER_MM / 2) # ≈ 1024

# ─────────────────────────────────────────────────────────────────────────────
#  Choice label sets
# ─────────────────────────────────────────────────────────────────────────────

CHOICES_AR = ["أ", "ب", "ج", "د", "هـ"]
CHOICES_EN = list("ABCDE")


def choice_labels(n: int, lang: str = "ar") -> List[str]:
    src = CHOICES_AR if lang == "ar" else CHOICES_EN
    return src[:n]


# ─────────────────────────────────────────────────────────────────────────────
#  TemplateSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemplateSpec:
    """
    Complete geometry contract for one OMR sheet variant.

    All warped-space coordinates assume a 750×1060 image produced by
    perspective-warping the physical A4 sheet using its four corner anchors.

    Both the dataset generator and the v3 detector import this class —
    that shared dependency is what eliminates the geometry drift between
    what was rendered and what the detector expects to find.
    """

    # ── identity ──────────────────────────────────────────────────────────────
    template_id:   str
    n_questions:   int
    n_choices:     int
    n_cols:        int           # 2 for Q20, 4 for Q50/Q100
    header_every:  int           # choice-label header row every N question rows
    rtl:           bool          # True = Arabic/RTL bubble order
    choice_labels: List[str]     # length == n_choices

    # ── visual sizing (from generate_sheet_v3.py layout_for()) ────────────────
    bubble_css_px: int           # CSS bubble diameter (px @96dpi)
    qn_css_px:     int           # question-number font size
    ch_css_px:     int           # choice-label font size
    row_py_css:    int           # row top+bottom padding
    col_gap_css:   int           # gap between columns

    # ── per-template BZONE overrides (None → use global BZONE_* constants) ──────
    # Calibrated from reference scans; wider columns (Q20) need different values.
    bzone_left_frac:    Optional[float] = None  # fraction of col_span from left edge to leftmost bubble
    bzone_right_base_w: Optional[float] = None  # fixed right-side base inset (warped px)

    # ── derived (computed once in __post_init__) ───────────────────────────────
    rows_per_col:  int   = field(init=False)
    n_hdr_rows:    int   = field(init=False)   # header rows per column
    visual_rows:   int   = field(init=False)   # question rows + header rows
    warp_col_w:    int   = field(init=False)   # column width in warped px
    warp_row_h:    float = field(init=False)   # row height in warped px
    warp_bub_r:    int   = field(init=False)   # bubble radius in warped px

    def __post_init__(self) -> None:
        self.rows_per_col = math.ceil(self.n_questions / self.n_cols)
        self.n_hdr_rows   = math.ceil(self.rows_per_col / self.header_every)
        self.visual_rows  = self.rows_per_col + self.n_hdr_rows
        self.warp_col_w   = WARP_GW // self.n_cols
        self.warp_row_h   = WARP_GH / self.visual_rows
        # Bubble radius in warped pixels: CSS px → mm → warped px
        bub_mm = self.bubble_css_px * (25.4 / 96.0)
        self.warp_bub_r = max(6, round(bub_mm * _SX / 2))

    # ── warped-space geometry helpers ─────────────────────────────────────────

    def col_x0(self, col_idx: int) -> int:
        """Left edge of column col_idx in warped pixels."""
        return WARP_GX0 + col_idx * self.warp_col_w

    def col_x1(self, col_idx: int) -> int:
        """Right edge of column col_idx in warped pixels."""
        return self.col_x0(col_idx) + self.warp_col_w

    def row_y(self, row_idx: int) -> int:
        """
        Y centre of question row row_idx (0-based within a column) in warped px.

        Visual layout within a column (header_every=5 case):
          vis_row 0: group-0 choice-label header
          vis_row 1: Q1      vis_row 2: Q2  …  vis_row 5: Q5
          vis_row 6: group-1 header
          vis_row 7: Q6  …

        Formula:  vis_row = row_idx + (row_idx // header_every) + 1
        """
        group   = row_idx // self.header_every
        vis_row = row_idx + group + 1
        return round(WARP_GY0 + (vis_row + 0.5) * self.warp_row_h)

    def header_y(self, group_idx: int) -> int:
        """Y centre of the choice-label header row for group group_idx (0-based)."""
        vis_row = group_idx * (self.header_every + 1)
        return round(WARP_GY0 + (vis_row + 0.5) * self.warp_row_h)

    def bubble_x_positions(self, col_idx: int) -> List[int]:
        """
        X centres for n_choices bubbles in column col_idx (warped px).

        Returned in choice-rank order:
          RTL (Arabic): rank 0 = rightmost bubble (choice أ)
          LTR (English): rank 0 = leftmost bubble (choice A)

        Uses the same linspace/inset formula as the v2 detector so that
        synthetic crops and real-scan crops are geometrically consistent.
        """
        x0   = self.col_x0(col_idx)
        x1   = self.col_x1(col_idx)
        span = float(x1 - x0)

        bzone_lf  = self.bzone_left_frac    if self.bzone_left_frac    is not None else BZONE_LEFT_FRAC
        bzone_rbw = self.bzone_right_base_w if self.bzone_right_base_w is not None else BZONE_RIGHT_BASE_W

        bzone_x0 = x0 + span * bzone_lf
        bzone_x1 = x1 - (bzone_rbw + span * BZONE_RIGHT_FRAC)

        if self.n_choices == 1:
            xs = [round((bzone_x0 + bzone_x1) / 2)]
        else:
            step = (bzone_x1 - bzone_x0) / (self.n_choices - 1)
            xs   = [round(bzone_x0 + i * step) for i in range(self.n_choices)]

        # RTL: rank 0 is rightmost
        if self.rtl:
            xs.reverse()
        return xs

    def question_for(self, col_idx: int, row_idx: int) -> int:
        """
        Question number (1-based) at (col_idx, row_idx).

        RTL convention: col_idx=0 is the LEFTMOST column in warped space, which
        corresponds to the HIGHEST question numbers (matching generate_sheet_v3.py
        and the v2 detector formula).
        """
        base_q = (self.n_cols - 1 - col_idx) * self.rows_per_col + 1
        return base_q + row_idx

    def all_bubble_positions_warp(
        self,
    ) -> List[Tuple[int, int, int, int, int, int]]:
        """
        Return (col_idx, row_idx, rank, cx_warp, cy_warp, r_warp) for every
        bubble, sorted ascending by question number then choice rank.
        """
        out: List[Tuple[int, int, int, int, int, int]] = []
        for col_idx in range(self.n_cols):
            xs = self.bubble_x_positions(col_idx)
            for row_idx in range(self.rows_per_col):
                q = self.question_for(col_idx, row_idx)
                if q > self.n_questions:
                    continue
                cy = self.row_y(row_idx)
                for rank, cx in enumerate(xs):
                    out.append((col_idx, row_idx, rank, cx, cy, self.warp_bub_r))
        out.sort(key=lambda t: (self.question_for(t[0], t[1]), t[2]))
        return out

    def bzone_x_bounds(self, col_idx: int) -> Tuple[float, float]:
        """Return (bzone_x0, bzone_x1) for column col_idx in warped px (float)."""
        x0   = self.col_x0(col_idx)
        x1   = self.col_x1(col_idx)
        span = float(x1 - x0)
        bzone_lf  = self.bzone_left_frac    if self.bzone_left_frac    is not None else BZONE_LEFT_FRAC
        bzone_rbw = self.bzone_right_base_w if self.bzone_right_base_w is not None else BZONE_RIGHT_BASE_W
        return (x0 + span * bzone_lf,
                x1 - (bzone_rbw + span * BZONE_RIGHT_FRAC))


# ─────────────────────────────────────────────────────────────────────────────
#  Template factory
# ─────────────────────────────────────────────────────────────────────────────

def _lp(n_q: int) -> dict:
    """CSS layout params from generate_sheet_v3.py layout_for()."""
    if n_q <= 20:
        return dict(bubble=26, qn=16, ch=14, row_py=8,  col_gap=28)
    if n_q <= 50:
        return dict(bubble=20, qn=14, ch=12, row_py=5,  col_gap=12)
    return     dict(bubble=18, qn=13, ch=11, row_py=3,  col_gap=12)


_VALID_Q = (10, 15, 18, 20, 30, 40, 50, 60, 80, 100)


def make_template(n_q: int, n_ch: int, lang: str = "ar") -> TemplateSpec:
    """
    Build a TemplateSpec for any valid (n_questions, n_choices) combination.

    Supported question counts: 10, 15, 18, 20, 30, 40, 50, 60, 80, 100
    Supported choice counts:   2, 3, 4, 5
    """
    if n_q not in _VALID_Q:
        raise ValueError(f"n_q must be one of {_VALID_Q}; got {n_q}")
    if n_ch not in (2, 3, 4, 5):
        raise ValueError(f"n_ch must be 2-5; got {n_ch}")

    n_cols = 2 if n_q <= 20 else 4
    lp     = _lp(n_q)
    rtl    = (lang == "ar")
    tid    = f"Q{n_q}_{n_ch}ch"

    return TemplateSpec(
        template_id   = tid,
        n_questions   = n_q,
        n_choices     = n_ch,
        n_cols        = n_cols,
        header_every  = 5,
        rtl           = rtl,
        choice_labels = choice_labels(n_ch, lang),
        bubble_css_px = lp["bubble"],
        qn_css_px     = lp["qn"],
        ch_css_px     = lp["ch"],
        row_py_css    = lp["row_py"],
        col_gap_css   = lp["col_gap"],
        # Q20 has 2 wide columns (311px each) — calibrated from reference scan.
        # The narrow-column constants (Q100: LEFT_FRAC=0.10, RIGHT_BASE=37) don't
        # scale to 311px: actual right inset is 76px vs formula's 64.7px,
        # and actual left inset is ~43px vs formula's 31px.
        bzone_left_frac    = 0.138 if n_q == 20 else None,
        bzone_right_base_w = 48.0  if n_q == 20 else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Template registry — all 12 combinations
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY: Dict[str, TemplateSpec] = {}

for _nq in _VALID_Q:
    for _nch in (2, 3, 4, 5):
        _t = make_template(_nq, _nch)
        REGISTRY[_t.template_id] = _t


def get_template(template_id: str) -> TemplateSpec:
    if template_id not in REGISTRY:
        raise KeyError(
            f"Unknown template '{template_id}'. "
            f"Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[template_id]


def infer_template(n_questions: int, n_choices: int = 5,
                   lang: str = "ar") -> TemplateSpec:
    """Convenience wrapper — infer template from CLI args."""
    return get_template(f"Q{n_questions}_{n_choices}ch")

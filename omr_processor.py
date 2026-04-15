"""
OMR Processor for Arabic Answer Sheets
Pipeline: Preprocessing → Alignment → Extraction → Validation → JSON Output
"""

import cv2
import numpy as np
import json
import sys
import os
import re
from pathlib import Path

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CFG = {
    "warp_w": 750,
    "warp_h": 1050,

    # Corner marker search (legacy, kept for fallback reference)
    "marker_search_margin": 110,
    "marker_patch_size": 120,
    "marker_thresh": 80,

    # HoughCircles params
    "hough_dp": 1.2,
    "hough_min_dist": 22,
    "hough_param1": 50,
    "hough_param2": 18,
    "hough_min_r": 8,
    "hough_max_r": 22,

    # Fill threshold: fraction of non-white pixels inside a bubble
    "fill_threshold": 0.42,

    # Region fractions (relative to warped image)
    # Answer box: right portion
    "ans_x_frac": 0.50,   # left edge of answer column area
    "ans_y0_frac": 0.09,
    "ans_y1_frac": 0.56,

    # Student code box: top-left  (رمز الطالب)
    "stud_x0_frac": 0.02,
    "stud_x1_frac": 0.36,
    "stud_y0_frac": 0.13,
    "stud_y1_frac": 0.44,

    # Test/exam code box: lower-left  (رمز الاختيار)
    "test_x0_frac": 0.02,
    "test_x1_frac": 0.36,
    "test_y0_frac": 0.48,
    "test_y1_frac": 0.86,

    "debug": True,
    "debug_dir": "debug_output",
}

# Answer choices: the answer grid is right-to-left (Arabic reading order)
# col 0 (leftmost in the extracted box) = choice A (أ)
# col 1 = B (ب), col 2 = C (ج), col 3 = D (د)
# BUT visually from the image: rightmost = A (أ), so in image coords col0=D, col3=A
# After extracting the right portion, left→right = D C B A (RTL)
# So: detected_col 0→D, 1→C, 2→B, 3→A  ... let's map after detection
CHOICE_MAP = {0: "D", 1: "C", 2: "B", 3: "A"}  # col index → choice label


# ─────────────────────────────────────────────
#  DEBUG HELPERS
# ─────────────────────────────────────────────
def _dbg(name: str, img: np.ndarray):
    if not CFG["debug"]:
        return
    os.makedirs(CFG["debug_dir"], exist_ok=True)
    cv2.imwrite(os.path.join(CFG["debug_dir"], name), img)


# ─────────────────────────────────────────────
#  1. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(img: np.ndarray):
    """Returns (gray_clahe). CLAHE handles uneven lighting from phone photos."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ─────────────────────────────────────────────
#  2. ALIGNMENT  (CamScanner-style)
# ─────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four (x,y) points as [TL, TR, BR, BL].
    TL = smallest sum, BR = largest sum,
    TR = smallest diff (x-y), BL = largest diff.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # TR
    rect[3] = pts[np.argmax(d)]   # BL
    return rect


def _four_point_warp(img: np.ndarray, pts: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Perspective-warp img so that pts maps to a perfect out_w × out_h rectangle."""
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def _find_document_corners(gray: np.ndarray):
    """
    CamScanner-style document detection:
      1. Gaussian blur → Canny edge detection
      2. Find contours, pick the largest roughly-quadrilateral one
      3. Approximate to 4 corners
    Returns np.ndarray of shape (4,2) or None.
    """
    H, W = gray.shape[:2]

    # Step 1: blur + Canny (two-pass: tight then relaxed)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    for lo, hi in [(50, 150), (30, 100), (10, 60)]:
        edges = cv2.Canny(blur, lo, hi)
        # Dilate edges to close small gaps
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, k, iterations=1)

        # Step 2: contours sorted by area descending
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Step 3: find first contour that approximates to a quadrilateral
        for c in cnts[:10]:
            area = cv2.contourArea(c)
            if area < 0.10 * H * W:   # ignore tiny contours
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

    return None


def _warp_dimensions(rect: np.ndarray):
    """Compute natural width and height of the warped document from the 4 corners."""
    tl, tr, br, bl = rect
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    return int(max(w1, w2)), int(max(h1, h2))


def align(img: np.ndarray) -> np.ndarray:
    """
    CamScanner-style alignment:
      1. Grayscale + Gaussian blur
      2. Canny edge detection
      3. Largest 4-corner contour = document boundary
      4. Perspective warp to a flat rectangle
      5. Fallback: Otsu paper-boundary detection if Canny finds nothing
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    H, W = gray.shape[:2]

    corners = _find_document_corners(gray)

    if corners is not None:
        rect = _order_points(corners)
        tl, tr, br, bl = rect
        out_w, out_h = _warp_dimensions(rect)
        # Clamp to configured max to avoid absurd dimensions
        out_w = min(out_w, CFG["warp_w"])
        out_h = min(out_h, CFG["warp_h"])
        method = "canny-contour"
    else:
        # Fallback: Otsu threshold → largest blob bounding box
        print("[WARN] Canny contour failed — falling back to Otsu paper boundary")
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            paper = max(cnts, key=cv2.contourArea)
            px, py, pw, ph = cv2.boundingRect(paper)
        else:
            px, py, pw, ph = 0, 0, W, H
        pad = 4
        rect = _order_points(np.float32([
            [px + pad,       py + pad],
            [px + pw - pad,  py + pad],
            [px + pw - pad,  py + ph - pad],
            [px + pad,       py + ph - pad],
        ]))
        tl, tr, br, bl = rect
        out_w, out_h = CFG["warp_w"], CFG["warp_h"]
        method = "otsu-fallback"

    print(f"[Align] Method={method}  TL={tuple(tl.astype(int))} TR={tuple(tr.astype(int))} "
          f"BR={tuple(br.astype(int))} BL={tuple(bl.astype(int))}  "
          f"-> {out_w}x{out_h}")

    # Ensure a minimum output size so ROI fractions stay sane
    out_w = max(out_w, 600)
    out_h = max(out_h, 840)

    # Debug: draw detected document outline on original
    vis = img.copy()
    poly = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.polylines(vis, [poly], True, (0, 255, 80), 3)
    for pt, color, lbl in zip([tl, tr, br, bl],
                               [(0, 0, 255), (0, 255, 0), (0, 165, 255), (255, 128, 0)],
                               ["TL", "TR", "BR", "BL"]):
        p = tuple(pt.astype(int))
        cv2.circle(vis, p, 10, color, -1)
        cv2.putText(vis, lbl, (p[0] + 7, p[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    _dbg("01_corners.jpg", vis)

    warped = _four_point_warp(img, np.array([tl, tr, br, bl]), out_w, out_h)
    _dbg("02_warped.jpg", warped)
    return warped


# ─────────────────────────────────────────────
#  BUBBLE DETECTION (HoughCircles-based)
# ─────────────────────────────────────────────
def _detect_circles(gray_roi: np.ndarray, min_r=None, max_r=None,
                     min_dist=None) -> list:
    """
    Returns list of dicts: {cx, cy, r, fill, filled}
    fill = fraction of dark pixels inside circle.
    """
    if min_r is None:
        min_r = CFG["hough_min_r"]
    if max_r is None:
        max_r = CFG["hough_max_r"]
    if min_dist is None:
        min_dist = CFG["hough_min_dist"]

    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=CFG["hough_dp"],
        minDist=min_dist,
        param1=CFG["hough_param1"],
        param2=CFG["hough_param2"],
        minRadius=min_r,
        maxRadius=max_r,
    )

    results = []
    if circles is None:
        return results

    for (cx, cy, r) in np.uint16(np.around(circles[0])):
        cx, cy, r = int(cx), int(cy), int(r)
        # Clamp to image bounds
        if cx - r < 0 or cy - r < 0 or cx + r >= gray_roi.shape[1] or cy + r >= gray_roi.shape[0]:
            continue
        # Build mask
        mask = np.zeros(gray_roi.shape, np.uint8)
        cv2.circle(mask, (cx, cy), max(r - 2, 2), 255, -1)
        nz = np.count_nonzero(mask)
        if nz == 0:
            continue
        dark_pixels = np.sum(gray_roi[mask > 0] < 110)
        fill = dark_pixels / nz
        filled = fill > CFG["fill_threshold"]
        results.append({"cx": cx, "cy": cy, "r": r, "fill": round(float(fill), 3), "filled": filled})

    return results


def _vis_circles(base: np.ndarray, circles: list, name: str):
    if not CFG["debug"]:
        return
    vis = base.copy() if len(base.shape) == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for b in circles:
        color = (0, 220, 0) if b["filled"] else (0, 140, 255)
        cv2.circle(vis, (b["cx"], b["cy"]), b["r"], color, 2)
        cv2.putText(vis, f"{b['fill']:.2f}",
                    (b["cx"] - 14, b["cy"] - b["r"] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)
    _dbg(name, vis)


# ─────────────────────────────────────────────
#  3a. SHEET CODE
# ─────────────────────────────────────────────
def extract_sheet_code(warped: np.ndarray) -> str:
    """
    Sheet code printed at bottom: e.g. '0004-720004' or 'T20004'.
    Strategy: OCR bottom strip → extract alphanumeric token.
    """
    H, W = warped.shape[:2]

    # Try bottom 7%
    strip = warped[int(H * 0.93):, :]
    _dbg("06_sheet_code_strip.jpg", strip)
    code = _ocr_strip(strip, pattern=r"[A-Z]?\d{4,6}(?:[-]\d+)?")
    if code:
        return code

    # Widen to bottom 15%
    strip2 = warped[int(H * 0.85):, :]
    code = _ocr_strip(strip2, pattern=r"[A-Z]?\d{4,6}(?:[-]\d+)?")
    return code or "UNKNOWN"


def _ocr_strip(region: np.ndarray, pattern: str = None) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        # Upscale small strips
        h = gray.shape[0]
        if h < 60:
            scale = max(2, 80 // h)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(
            bw,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
        ).strip()
        if not text:
            return ""
        if pattern:
            m = re.search(pattern, text, re.IGNORECASE)
            return m.group(0) if m else ""
        return text
    except Exception as e:
        print(f"[OCR] {e}")
        return ""


# ─────────────────────────────────────────────
#  CLUSTER HELPER
# ─────────────────────────────────────────────
def _cluster_1d(values: list, gap: float = 25.0) -> list:
    """Group sorted values into clusters, return centroids."""
    if not values:
        return []
    values = sorted(float(v) for v in values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] < gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def _cluster_1d_auto(values: list, n_expected: int) -> list:
    """
    Cluster 1D values into n_expected groups using k-means.
    Returns sorted list of cluster centroids.
    """
    if not values:
        return []
    arr = np.array(sorted(float(v) for v in values), dtype=np.float32).reshape(-1, 1)
    if len(arr) < n_expected:
        # Too few points; gap-based fallback
        return _cluster_1d(values, gap=15)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.5)
    best_centers = None
    best_compactness = float("inf")
    for _ in range(5):   # multiple restarts for stability
        compact, _, centers = cv2.kmeans(
            arr, n_expected, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        if compact < best_compactness:
            best_compactness = compact
            best_centers = centers

    return sorted(float(c[0]) for c in best_centers)


def _nearest_idx(val: float, centers: list) -> int:
    return int(np.argmin([abs(val - c) for c in centers]))


# ─────────────────────────────────────────────
#  3b. STUDENT CODE
# ─────────────────────────────────────────────
def extract_student_code(warped: np.ndarray) -> dict:
    """
    رمز الطالب — top-left bubble grid, 4 cols x 10 rows (digit 0-9).
    Returns {"code": "0004", "method": "ocr|bubbles"}.
    """
    H, W = warped.shape[:2]
    box = warped[int(H * CFG["stud_y0_frac"]): int(H * CFG["stud_y1_frac"]),
                 int(W * CFG["stud_x0_frac"]): int(W * CFG["stud_x1_frac"])]
    _dbg("03_student_box.jpg", box)

    # Try OCR on printed barcode below the grid (bottom 12% of box)
    barcode_strip = box[int(box.shape[0] * 0.88):, :]
    _dbg("03b_student_barcode.jpg", barcode_strip)
    ocr_code = _ocr_strip(barcode_strip, pattern=r"\d{4,6}")
    if ocr_code and re.fullmatch(r"\d{4,6}", ocr_code):
        return {"code": ocr_code, "method": "ocr"}

    # Fallback: cluster-based bubble grid (4 cols x 10 rows)
    code = _decode_bubble_code(box, n_cols=4, debug_name="03c_student_circles.jpg")
    return {"code": code, "method": "bubbles"}


def _decode_bubble_code(box: np.ndarray, n_cols: int, debug_name: str,
                         header_frac: float = 0.13, footer_frac: float = 0.07) -> str:
    """
    Decode a numeric bubble grid: n_cols columns × 10 rows (row 0 = digit 0, row 9 = digit 9).

    Row calibration: detect all circles, then fit the row grid (row-0 Y, row spacing) to
    the detected Y values using a periodicity search. This is robust to header/footer crop
    errors since it uses the actual circle positions.
    """
    gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY) if len(box.shape) == 3 else box
    bh, bw = gray.shape[:2]

    # Broad crop — just exclude the very top and bottom borders
    grid_y0 = int(bh * header_frac)
    grid_y1 = int(bh * (1.0 - footer_frac))
    grid_gray = gray[grid_y0:grid_y1, :]

    # Pad grid_gray so circles near the border are detected reliably
    PAD = 15
    padded = cv2.copyMakeBorder(grid_gray, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    circles_raw = _detect_circles(padded, min_r=6, max_r=18, min_dist=14)
    # Subtract padding offset; keep only circles whose center is inside the original grid area
    circles = [dict(b, cx=b["cx"] - PAD, cy=b["cy"] - PAD)
               for b in circles_raw
               if 0 <= b["cy"] - PAD < grid_y1 - grid_y0
               and 0 <= b["cx"] - PAD < grid_gray.shape[1]]
    circles_vis = [dict(b, cy=b["cy"] + grid_y0) for b in circles]
    _vis_circles(box, circles_vis, debug_name)

    if not circles:
        return "?" * n_cols

    # Column centers
    xs = [b["cx"] for b in circles]
    col_centers = _cluster_1d_auto(xs, n_cols)

    # ── Fixed-grid row model ───────────────────────────────────────────────────
    # The grid spans the full height of the cropped region (after header/footer removal).
    # 10 rows are evenly spaced; row i center = (i + 0.5) * grid_h / 10.
    grid_h = grid_y1 - grid_y0
    row_ys = [(i + 0.5) * grid_h / 10.0 for i in range(10)]

    def cy_to_digit(cy: float) -> int:
        return int(_nearest_idx(cy, row_ys))

    # Build grid: (digit, col_index) → highest-fill bubble
    grid: dict = {}
    for b in circles:
        digit = cy_to_digit(b["cy"])
        ci    = _nearest_idx(b["cx"], col_centers)
        key   = (digit, ci)
        if key not in grid or b["fill"] > grid[key]["fill"]:
            grid[key] = b

    import sys
    print(f"  [DBG {debug_name}] grid_h={grid_h} row_ys={[round(r) for r in row_ys]} "
          f"cols={[round(c) for c in col_centers]}", file=sys.stderr)
    for ci in range(min(n_cols, len(col_centers))):
        col_cells = [(d, b["cy"], b["fill"]) for (d, c), b in grid.items() if c == ci]
        col_cells.sort(key=lambda x: x[0])
        print(f"  col{ci}: {[(d, round(cy), round(f, 3)) for d, cy, f in col_cells]}",
              file=sys.stderr)

    # Pick best digit per column — always take max-fill if above noise floor
    digits = []
    for ci in range(min(n_cols, len(col_centers))):
        col_cells = [(digit, data) for (digit, c), data in grid.items() if c == ci]
        if not col_cells:
            digits.append("?")
            continue
        best_digit, best_data = max(col_cells, key=lambda x: x[1]["fill"])
        if best_data["fill"] > 0.20:
            digits.append(str(best_digit))
        else:
            digits.append("?")

    while len(digits) < n_cols:
        digits.append("?")

    # Arabic RTL: leftmost column = last digit of the code
    return "".join(reversed(digits))


# ─────────────────────────────────────────────
#  3c. TEST CODE
# ─────────────────────────────────────────────
def extract_test_code(warped: np.ndarray) -> dict:
    """
    رمز الاختبار — lower-left bubble grid, 6 cols x 10 rows.
    Returns {"code": "720004", "method": "ocr|bubbles"}.
    """
    H, W = warped.shape[:2]
    box = warped[int(H * CFG["test_y0_frac"]): int(H * CFG["test_y1_frac"]),
                 int(W * CFG["test_x0_frac"]): int(W * CFG["test_x1_frac"])]
    _dbg("07_test_code_box.jpg", box)

    # Try OCR barcode (just below bubble grid, ~70-85% of box height)
    barcode_strip = box[int(box.shape[0] * 0.70): int(box.shape[0] * 0.86), :]
    _dbg("07b_test_barcode.jpg", barcode_strip)
    ocr_code = _ocr_strip(barcode_strip, pattern=r"[A-Z]?\d{5,6}")
    if ocr_code:
        return {"code": ocr_code, "method": "ocr"}

    # Fallback: 6-col bubble grid (test box crop already excludes header)
    code = _decode_bubble_code(box, n_cols=6, debug_name="07c_test_circles.jpg",
                               header_frac=0.02, footer_frac=0.37)
    return {"code": code, "method": "bubbles"}


def extract_answers(warped: np.ndarray, n_questions: int = 12) -> dict:
    """
    Answer grid: right portion. n_questions × 4 choices.
    Sheet is Arabic (RTL): rightmost circle column = A (أ), leftmost = D (د).
    Uses cluster-based row/col assignment rather than fixed cell size.
    Returns {question_number_str: choice_letter or None}.
    """
    H, W = warped.shape[:2]
    box = warped[int(H * CFG["ans_y0_frac"]): int(H * CFG["ans_y1_frac"]),
                 int(W * CFG["ans_x_frac"]):]
    _dbg("04_answer_box.jpg", box)

    gray_box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    circles = _detect_circles(gray_box)
    _vis_circles(box, circles, "05_answer_circles.jpg")

    if not circles:
        return {str(q + 1): None for q in range(n_questions)}

    # ── Cluster circles into rows (by Y) and columns (by X) ──────────────
    ys = [b["cy"] for b in circles]
    xs = [b["cx"] for b in circles]

    row_centers = _cluster_1d_auto(ys, n_questions)
    col_centers = _cluster_1d_auto(xs, 4)

    # Assign each circle to nearest row/col cluster
    grid = {}  # (row_idx, col_idx) -> best circle dict
    for b in circles:
        ri = _nearest_idx(b["cy"], row_centers)
        ci = _nearest_idx(b["cx"], col_centers)
        key = (ri, ci)
        if key not in grid or b["fill"] > grid[key]["fill"]:
            grid[key] = b

    n_rows = len(row_centers)
    n_cols = len(col_centers)

    # col_centers sorted ascending = leftmost first.
    # In the extracted box, right side (higher x) = Arabic A (أ).
    # So: col_idx (n_cols-1) = rightmost = A, col_idx 0 = leftmost = D (if 4 cols)
    # Map col_idx to choice: rightmost→A, then B, C, D left
    def col_to_choice(ci, total_cols):
        # rightmost = A
        rank = (total_cols - 1) - ci   # 0 for rightmost
        return ["A", "B", "C", "D"][rank] if rank < 4 else "?"

    # Map row clusters to question numbers
    # We expect exactly n_questions row clusters; if more, ignore outliers
    # row_centers sorted ascending = Q1 at top (lower y), Q12 at bottom
    if n_rows < n_questions:
        print(f"[WARN] Only {n_rows} row clusters found (expected {n_questions})")
    if n_cols < 4:
        print(f"[WARN] Only {n_cols} column clusters found (expected 4)")

    answers = {}
    for q in range(n_questions):
        if q >= n_rows:
            answers[str(q + 1)] = None
            continue

        row_cells = {ci: data for (ri, ci), data in grid.items() if ri == q}
        filled_cols = {ci: data["fill"] for ci, data in row_cells.items()
                       if data["filled"]}

        if len(filled_cols) > 1:
            # Multiple fills → double mark; pick highest, flag it
            best_ci = max(filled_cols, key=filled_cols.get)
            answers[str(q + 1)] = col_to_choice(best_ci, n_cols)
            answers[str(q + 1) + "_double_mark"] = True
        elif filled_cols:
            best_ci = max(filled_cols, key=filled_cols.get)
            answers[str(q + 1)] = col_to_choice(best_ci, n_cols)
        else:
            # Soft fallback: pick darkest if slightly above 70% of threshold
            if row_cells:
                darkest_ci = max(row_cells, key=lambda c: row_cells[c]["fill"])
                if row_cells[darkest_ci]["fill"] > CFG["fill_threshold"] * 0.72:
                    answers[str(q + 1)] = col_to_choice(darkest_ci, n_cols)
                    answers[str(q + 1) + "_confidence"] = "low"
                else:
                    answers[str(q + 1)] = None
            else:
                answers[str(q + 1)] = None

    return answers


# ─────────────────────────────────────────────
#  4. VALIDATION + JSON OUTPUT
# ─────────────────────────────────────────────
def validate(student_code: str, test_code: dict, answers: dict, n_questions: int) -> dict:
    issues = []

    if "?" in student_code:
        issues.append(f"student_code uncertain: {student_code!r}")

    unanswered = [q for q in range(1, n_questions + 1) if answers.get(str(q)) is None]
    if unanswered:
        issues.append(f"unanswered questions: {unanswered}")

    low_conf = [q for q in range(1, n_questions + 1) if str(q) + "_confidence" in answers]
    if low_conf:
        issues.append(f"low-confidence answers: {low_conf}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


def build_output(image_path: str, sheet_code: str, student: dict, test: dict,
                 answers: dict, n_questions: int) -> dict:
    # Clean answers dict (remove confidence markers for count)
    clean_answers = {k: v for k, v in answers.items() if not k.endswith("_confidence")}
    answered = sum(1 for v in clean_answers.values() if v is not None)

    validation = validate(student["code"], test, clean_answers, n_questions)

    return {
        "image": str(image_path),
        "sheet_code": sheet_code,
        "student_code": student["code"],
        "student_code_method": student["method"],
        "test_code": test["code"],
        "test_code_method": test["method"],
        "answers": clean_answers,
        "answer_details": answers,
        "total_questions": n_questions,
        "answered": answered,
        "validation": validation,
    }


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def process(image_path: str, n_questions: int = 12, output_json: str = None) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    _dbg("00_original.jpg", img)

    print("[1] Aligning sheet ...")
    warped = align(img)

    print("[2] Extracting sheet barcode ...")
    sheet_code = extract_sheet_code(warped)
    print(f"    sheet_code: {sheet_code}")

    print("[3] Extracting student code ...")
    student = extract_student_code(warped)
    print(f"    student_code: {student}")

    print("[4] Extracting test/exam code ...")
    test = extract_test_code(warped)
    print(f"    test_code: {test}")

    print("[5] Extracting answers ...")
    answers = extract_answers(warped, n_questions)
    clean = {k: v for k, v in answers.items() if not k.endswith("_confidence")}
    print(f"    answers: {clean}")

    print("[6] Validating ...")
    result = build_output(image_path, sheet_code, student, test, answers, n_questions)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON saved to {output_json}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "student-answer.png"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "result.json"
    process(img_path, n_questions=12, output_json=out_path)

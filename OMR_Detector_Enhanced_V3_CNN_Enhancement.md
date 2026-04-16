# OMR Detector Enhanced V3 — Alignment & Geometry Enhancement

**Project:** e-Learning Grade OMR  
**Engine version:** v3.1 (Playwright-guided geometry + perimeter snapping)  
**Date:** April 2026  
**Builds on:** [OMR_Detector_Enhanced_V3_CNN.md](OMR_Detector_Enhanced_V3_CNN.md)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [What Was Achieved](#what-was-achieved)
3. [Expanded Template Coverage](#expanded-template-coverage)
4. [Playwright-Extracted Bubble Coordinates](#playwright-extracted-bubble-coordinates)
5. [Per-Column X Correction from Bar Detection](#per-column-x-correction-from-bar-detection)
6. [Circle-Centre Perimeter Snapping (Stage 5E)](#circle-centre-perimeter-snapping)
7. [Grid Regularisation (Stage 5F)](#grid-regularisation)
8. [Classification Radius Shrinkage](#classification-radius-shrinkage)
9. [Debug Overlay — Alignment Grid Lines](#debug-overlay)
10. [Revised Pipeline Flow](#revised-pipeline-flow)
11. [Files Reference](#files-reference)
12. [Key Engineering Lessons](#key-engineering-lessons)

---

## Problem Statement

The v3 CNN pipeline (see [base document](OMR_Detector_Enhanced_V3_CNN.md)) achieved excellent classification accuracy but assumed the **template formula geometry** would place detection circles accurately on every scan. In practice, several alignment problems emerged:

| Symptom | Root Cause |
|---------|-----------|
| Edge bubbles (choices A and E) drawn outside column bounds | `linspace` used `n-1` divisor — no margin; bubbles sat on the column boundary |
| 4th column bubbles shifted left, 1st column shifted right | No per-column correction against actual bar positions in the warped image |
| Detection circles centred on ink centroid, not printed ring | Classifier sampled at formula-derived position, which drifts with handwriting |
| Grid lines through bubble centres were visibly wobbly | Each bubble snapped independently — no collective regularisation |
| Q10/Q15/Q18 layouts completely misaligned | Template registry only contained Q20/Q50/Q100; CSS layout differences (larger bubbles, variable row heights) made formula-based row/column geometry inaccurate for other sizes |

These are **geometry** problems, not classification problems — the CNN was correctly classifying whatever it was pointed at, but it was being pointed at slightly wrong locations.

---

## What Was Achieved

| Enhancement | Status | Detail |
|-------------|--------|--------|
| Playwright coordinate extraction | ✅ Complete | Ground-truth `(cx, cy, r)` measured from the HTML renderer |
| All 10 sheet sizes supported | ✅ Complete | Q10, Q15, Q18, Q20, Q30, Q40, Q50, Q60, Q80, Q100 |
| JSON coords for all templates × 2 languages | ✅ Complete | 20 files in `bubble_coords/` |
| Per-column X correction (bar-based) | ✅ Complete | Shifts JSON coords to match actual warped-image column positions |
| Circle-centre perimeter snapping | ✅ Complete | Canny-based ring detection with distance penalty |
| Median grid regularisation | ✅ Complete | Forces strict vertical alignment per choice column |
| Classification radius shrinkage | ✅ Complete | Avoids sampling the printed ring ink |
| Debug grid overlay | ✅ Complete | Semi-transparent alignment lines through bubble centres |
| `space-around` bubble X formula | ✅ Complete | Correct CSS-matched spacing with proper margins |

---

## Expanded Template Coverage

**File:** `omr_templates.py`

The template registry was expanded from 3 to 10 question counts:

```python
_VALID_Q = (10, 15, 18, 20, 30, 40, 50, 60, 80, 100)
```

Each is registered for all choice counts (2–5) and both languages (AR/EN), yielding **80 template variants**. The `make_template()` factory handles column layout selection automatically:

| Questions | Columns | Rows/Col |
|-----------|---------|----------|
| 10        | 1       | 10       |
| 15        | 1       | 15       |
| 18        | 2       | 9        |
| 20        | 2       | 10       |
| 30        | 3       | 10       |
| 40        | 4       | 10       |
| 50        | 4       | 13       |
| 60        | 4       | 15       |
| 80        | 4       | 20       |
| 100       | 4       | 25       |

---

## Playwright-Extracted Bubble Coordinates

**File:** `generate_sheet_v3.py` — `extract_bubble_coords()`

### The Problem with Formula-Based Geometry

The original pipeline computed bubble positions from `TemplateSpec` fields using analytical formulas (`row_y()`, `bubble_x_positions()`, etc.). These approximate the CSS layout but accumulate errors because:

- CSS `flex` layout distributes space differently from linear interpolation
- Headers, separators, and padding vary per sheet size
- Bubble diameter scales non-linearly between Q10 (large) and Q100 (small)

For Q50/Q100 the errors were tolerable (~2–5 px). For Q10, they were catastrophic (~15–25 px) because the CSS layout has much larger bubbles and proportionally different spacing.

### The Solution: Measure, Don't Compute

`generate_sheet_v3.py` now renders each HTML sheet in a headless Chromium browser (via Playwright), queries every `.bubble` element's bounding box, and records the **exact pixel-centre coordinates** as resolution-independent fractions:

```python
def extract_bubble_coords(html_path, n_questions, n_choices, n_cols, rtl):
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page    = browser.new_page()
        page.goto(f"file:///{Path(html_path).resolve()}")

        sheet_box = page.query_selector(".sheet").bounding_box()
        sw, sh = sheet_box["width"], sheet_box["height"]

        for col_el in page.query_selector_all(".qc"):
            for qr_el in col_el.query_selector_all(".qr:not(.qr-phantom)"):
                for bub_el in qr_el.query_selector_all(".bubble"):
                    box = bub_el.bounding_box()
                    cx = (box["x"] + box["width"]/2 - sheet_box["x"]) / sw
                    cy = (box["y"] + box["height"]/2 - sheet_box["y"]) / sh
                    r  = (box["width"] / 2) / sw
                    # → store as cx_frac, cy_frac, r_frac
```

### Output Format

Each JSON file (`bubble_coords/Q{n}_5ch_{lang}.json`) contains:

```json
{
  "template": "Q50_5ch",
  "n_questions": 50,
  "n_choices": 5,
  "n_cols": 4,
  "rtl": true,
  "bubbles": [
    {
      "q": 40,
      "choice": 0,
      "col": 0,
      "row": 0,
      "cx_frac": 0.844633,
      "cy_frac": 0.286327,
      "r_frac": 0.012599
    }
  ]
}
```

Coordinates are fractions of sheet size (0–1), mapped to warped pixels at detection time:

```python
cx = round(b["cx_frac"] * WARP_W)   # WARP_W = 750
cy = round(b["cy_frac"] * WARP_H)   # WARP_H = 1060
```

### Generation Pipeline

Running `python generate_sheet_v3.py --pdf` generates all sheets and automatically extracts coordinates:

```
[AR  50Q] -> answer_sheet_v3_50q_ar.html
[PDF] saved -> answer_sheet_v3_50q_ar.pdf
[COORDS] saved -> bubble_coords/Q50_5ch_ar.json
```

This produces **20 JSON files** — one per (Q-count × language) combination.

---

## Per-Column X Correction from Bar Detection

**File:** `omr_detector_enhanced_v3.py` — Stage 5C

### Why It's Needed

Playwright JSON gives **ideal** positions relative to the A4 page. The detector warps the scanned image using 4 corner anchors to the canonical 750×1060 space. But:

- Corner anchor detection has sub-pixel uncertainty
- Paper can warp slightly during printing/scanning
- Small homography errors accumulate across the page width

This creates a systematic per-column horizontal drift of 2–8 px — enough to visibly offset detection circles from their printed bubbles.

### How It Works

Bar detection finds the actual column boundaries in the warped image. The correction compares the JSON-derived column centre against the bar-detected column centre and shifts all bubbles in that column:

```python
# Compute expected column centre from JSON bubble coordinates
for b in cdata["bubbles"]:
    _col_cxs[b["col"]].append(b["cx_frac"] * WARP_W)

# Per-column shift: detected bar centre vs JSON-predicted centre
for col_idx, cxs in _col_cxs.items():
    json_cx = mean(cxs)
    det_cx  = (col.x0 + col.x1) / 2   # from bar detection
    col_dx[col_idx] = round(det_cx - json_cx)

# Applied to each bubble
cx = round(b["cx_frac"] * WARP_W) + col_dx[b["col"]]
```

The same correction already existed for the formula-based fallback path — this enhancement brings parity to the Playwright-JSON path.

---

## Circle-Centre Perimeter Snapping

**File:** `omr_detector_enhanced_v3.py` — `_snap_to_circles()` (Stage 5E)

After initial placement (from JSON or formula), each bubble is fine-tuned to lock onto the **geometric centre of its printed ring** — not the centroid of any handwritten ink.

### Algorithm

```
1. Canny(gray, 40, 100) → edge map
2. Blank ±1 px strips at column borders (prevents bar edges from stealing snaps)
3. Pre-compute circle perimeter offsets for radii r-2 … r+2
4. For each bubble, evaluate all candidates in a ±6 px window:
   a. Sum edge response along the circle perimeter (max-pooled across radii)
   b. Subtract distance penalty: 1.5 × displacement_px
   c. Mask candidates that fall outside column bounds
5. Accept snap only if best_score > original_score + min_gain (3.0)
6. Only adjust X — Y stays fixed from row detection
```

### Key Design Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `circle_snap_r` | 6 px | Small enough to avoid cross-talk between adjacent bubbles (~18 px apart) |
| `circle_snap_min_gain` | 3.0 | Prevents noise-driven jitter; only snap when there's a clear ring signal |
| `DIST_PENALTY` | 1.5/px | Biases towards the template position; prevents long-range pulls |
| Border blank width | ±1 px | Narrowly removes column bar edges without erasing nearby bubble edges |
| Radius range | r±2 | Robust to small radius estimation errors; max-pool across all tested radii |
| X-only snap | — | Y is reliably fixed by horizontal band detection; moving Y would only add noise |

### Why Perimeter Voting, Not Centroid

Standard approaches find the centroid (centre of mass) of dark pixels. For OMR, this fails because:

- Filled bubbles have asymmetric ink distribution (scribbles, partial fills)
- The centroid of the **ink** is not the centre of the **printed circle**
- The printed ring has a consistent edge signature even when filled

Perimeter voting finds the point where a circle of the expected radius produces the strongest Canny edge response — this is the geometric centre of the printed ring regardless of how the student marked it.

---

## Grid Regularisation

**File:** `omr_detector_enhanced_v3.py` — `_regularize_grid()` (Stage 5F)

After snapping, each bubble has been adjusted independently. Due to noise in the Canny edge map, snapped X positions can wobble ±1–2 px within a choice column.

Regularisation forces **strict vertical alignment** by replacing each bubble's X with the **median X of all bubbles in the same (column, choice) group**:

```python
# Group by (column_index, choice_rank) → list of cx values
x_groups[(b.col_idx, b.choice_rank)].append(b.cx)

# Replace each bubble's cx with the group median
med_x = {k: median(vs) for k, vs in x_groups.items()}
b.cx = med_x[(b.col_idx, b.choice_rank)]
```

Y coordinates are **not regularised** — they are derived from horizontal band detection which is already deterministic and reliable.

This produces perfectly straight vertical grid lines in the debug overlay.

---

## Classification Radius Shrinkage

**File:** `omr_detector_enhanced_v3.py` — `ALGO["cls_r_shrink"]`

The printed bubble ring has a stroke width of ~1–2 px. If the classifier samples at the full template radius, it includes ring ink pixels, which inflate the fill ratio of empty bubbles.

```python
cls_r = max(4, b.r - ALGO["cls_r_shrink"])   # cls_r_shrink = 1
```

This shrunken radius is used for:
- `_fill_ratio_direct()` dark-pixel counting
- CNN crop extraction (`cv2.getRectSubPix` at `2 × cls_r` window)
- Debug overlay circle rendering (green circles in `06_grid.jpg`)

---

## Debug Overlay

**File:** `omr_detector_enhanced_v3.py` — Stage 5 debug output

The `06_grid.jpg` debug image now includes:

1. **Green circles** at each bubble's final (post-snap, post-regularise) position, drawn at the classification radius
2. **Orange rectangles** around detected column bounds
3. **Semi-transparent blue grid lines** through bubble centres:
   - **Horizontal lines** — one per unique Y (from row detection), spanning the full grid width
   - **Vertical lines** — one per unique X within each column, spanning the column's height

Grid lines are 1 px thick at 30% opacity (alpha-blended) to avoid obscuring the underlying image. Straight, evenly-spaced lines confirm correct alignment; wobbly lines indicate snapping or regularisation issues.

---

## Revised Pipeline Flow

The enhancements insert new sub-stages into Stage 5 and modify Stage 6:

```
[5] Bubble Grid (Lattice Fitting)
    │
    ├─ 5A: Detect bars (vertical projection)
    ├─ 5B: Refine columns from bars
    ├─ 5C: Load bubble positions
    │       ├─ Primary:  Playwright JSON (bubble_coords/{id}_{lang}.json)
    │       │            + per-column X correction from bar detection
    │       └─ Fallback: Template formula geometry
    │                    + per-column X correction from bar detection
    ├─ 5E: Circle-centre perimeter snap (X-only, ±6 px)
    └─ 5F: Grid regularisation (median X per choice-column)
        │
        ▼
[6] Bubble Classifier
    Uses cls_r = bubble_r − cls_r_shrink for sampling
    (avoids printed ring ink → cleaner fill-ratio and CNN crops)
```

### Source Selection Logic

```
if bubble_coords/{template_id}_ar.json exists:
    source = "playwright-json"
    Load fractional coords → scale to warped pixels
    Apply per-column X shift (bar-detected centre − JSON centre)
else:
    source = "template-formula"
    Compute from TemplateSpec.all_bubble_positions_warp()
    Apply per-column X shift (bar-detected centre − template centre)
```

---

## Files Reference

| File | Role |
|------|------|
| `omr_templates.py` | Template registry — expanded to 10 Q-counts × 4 choice-counts × 2 languages |
| `omr_detector_enhanced_v3.py` | 8-stage detector with snap/regularise/shrink enhancements |
| `generate_sheet_v3.py` | HTML/CSS + PDF generator + Playwright coordinate extraction |
| `bubble_coords/*.json` | 20 ground-truth coordinate files (fractional cx, cy, r per bubble) |
| `bubble_classifier_v3.onnx` | CNN model (unchanged from base v3) |

### Detector CLI

```bash
# With explicit question count (selects matching template + JSON coords)
python omr_detector_enhanced_v3.py image.jpg --questions 50
python omr_detector_enhanced_v3.py image.jpg --questions 10

# Regenerate all coordinate JSONs (run whenever sheet HTML changes)
python generate_sheet_v3.py --pdf
```

---

## Key Engineering Lessons

1. **Measure, don't compute.** Analytical formulas for CSS layout geometry accumulate errors because they approximate flex/grid behaviour. Directly measuring element positions from the rendering engine eliminates this entire class of bugs. The Playwright extraction adds ~7 seconds per sheet at generation time but produces sub-pixel-accurate coordinates that never need manual calibration.

2. **Per-column correction bridges the scan-to-template gap.** Even with perfect ground-truth coordinates, a scanned/warped image will have small systematic offsets due to homography imprecision. Bar detection provides a per-column anchor to correct this — a single shift per column that costs nothing computationally but recovers 2–8 px of alignment.

3. **Perimeter voting beats centroid detection for pre-printed circles.** The printed ring maintains its edge signature regardless of how the bubble is filled. Sampling the Canny response along the expected circle perimeter finds the ring centre, not the ink centre — critical when students mark off-centre, partially, or with variable pressure.

4. **Snap then regularise — two-pass refinement.** Individual snapping captures per-bubble local edge evidence, but introduces noise. Regularisation (median per column-choice group) removes the noise by enforcing what we know must be true: all choice-A bubbles in column 0 must share the same X coordinate. Neither step alone is sufficient.

5. **Shrink the classification radius.** The printed ring is part of the "bubble" but not part of the "answer." Sampling at `r − 1` avoids the ring ink, giving the fill-ratio classifier a cleaner signal and the CNN a crop that contains only paper-vs-ink, not paper-vs-ring-vs-ink.

6. **Y from rows, X from everything else.** Row detection via horizontal projection is highly reliable — the dark bands are unambiguous. X positioning is harder because bubbles are closely spaced and printing/scanning introduces lateral drift. Fixing Y to the row-detected position and only adjusting X with snap/regularise simplifies the problem and prevents Y noise from compounding.

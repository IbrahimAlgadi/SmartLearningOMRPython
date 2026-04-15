import cv2
import numpy as np
import sys
import json
import pathlib
import omr_detector as omr

img = cv2.imread("detected-ans7/02_warped.jpg")
debug_dir = pathlib.Path("detected-ans7")

# We'll monkey-patch omr_detector to use adaptive thresholding for the dark_binary
def my_detect_circles(gray):
    # This is a copy of omr_detector.detect_circles, with adaptiveThreshold
    raw = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=omr.CFG["hough_dp"],
        minDist=omr.CFG["hough_min_dist"],
        param1=omr.CFG["hough_param1"],
        param2=omr.CFG["hough_param2"],
        minRadius=omr.CFG["hough_min_r"],
        maxRadius=omr.CFG["hough_max_r"],
    )
    results = []
    if raw is None:
        return results
    H, W = gray.shape

    # ADAPTIVE THRESHOLD
    # block size 51, C=15.  Using ADAPTIVE_THRESH_GAUSSIAN_C
    dark_binary = cv2.adaptiveThreshold(
        gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15
    )

    for (cx, cy, r) in np.uint16(np.around(raw[0])):
        cx, cy, r = int(cx), int(cy), int(r)
        if cx - r < 0 or cy - r < 0 or cx + r >= W or cy + r >= H:
            continue
        mask = np.zeros((H, W), np.uint8)
        cv2.circle(mask, (cx, cy), max(r - 1, 2), 1, -1)
        nz = np.count_nonzero(mask)
        if nz == 0:
            continue
        dark = int(np.sum(dark_binary[mask > 0]))
        fill = dark / nz
        results.append({
            "cx": cx, "cy": cy, "r": r,
            "fill": round(fill, 3),
            "filled": fill > omr.CFG["fill_ratio_thr"],
        })
    return omr._nms_circles(results)

# Monkey-patch it
omr.detect_circles = my_detect_circles

anchors = omr.verify_anchors(img, debug_dir)
circles, _ = omr.detect_answer_circles(img, debug_dir, anchors=anchors)
cluster_grid = omr.cluster_circles(circles, 4, 25)
answers, details = omr.extract_answers(cluster_grid, 4, 25)

# print the answers 
# We'll just look at Q76..80 which might be shadowed
print("Results with Adaptive Threshold:")
for q in range(76, 81):
    q_str = str(q)
    if q_str in details:
        d = details[q_str]
        print(f"Q{q}: choice={d.get('choice', '-')}  fill={d.get('fill', '-')}  mean={d.get('row_mean_fill', '-')}  candidates={len(d.get('candidates_debug', []))}")

# Check Q51..55
for q in range(51, 56):
    q_str = str(q)
    if q_str in details:
        d = details[q_str]
        print(f"Q{q}: choice={d.get('choice', '-')}  fill={d.get('fill', '-')}  mean={d.get('row_mean_fill', '-')}  candidates={len(d.get('candidates_debug', []))}")

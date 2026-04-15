import cv2
import numpy as np
import sys
import json
import pathlib
import omr_detector as omr

img_path = "ans7.png"
debug_dir = pathlib.Path("detected-ans7-adaptive")
debug_dir.mkdir(exist_ok=True)

# Re-monkey-patch
def my_detect_circles(gray):
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

    # Adaptive thresholding
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

omr.detect_circles = my_detect_circles

# We will just run omr.process
res = omr.process("detected-ans7/00_original.jpg")
print("Done processing 00_original.jpg with adaptive threshold.")

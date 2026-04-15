import cv2
import numpy as np
import omr_detector as omr
import pathlib

# Monkey-patch to use background division
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

    # ── SHADING CORRECTION ──
    # Estimate background illumination using a large median or gaussian blur
    # The bubble diameter is ~20px. So 51 or 71 should work.
    bg = cv2.medianBlur(gray, 71)
    
    # Avoid division by zero
    bg = np.maximum(bg, 1)
    
    # Normalize: if gray == bg, norm == 255. If gray < bg, norm < 255.
    norm = np.clip((gray.astype(np.float32) / bg.astype(np.float32)) * 255, 0, 255).astype(np.uint8)

    # Now use a relative threshold. Let's say anything 70% of the background is "dark"
    # i.e., norm < 255 * 0.70 = 178
    # We can tune this `norm_thr`. Let's save `norm` to debug it.
    cv2.imwrite("detected-ans7/debug_norm.jpg", norm)

    # Use the threshold on the normalized image
    norm_thr = 180
    dark_binary = (norm < norm_thr).astype(np.uint8)

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

res = omr.process("detected-ans7/00_original.jpg")
print("Done processing 00_original.jpg with background division.")

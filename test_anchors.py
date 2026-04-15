import cv2
import numpy as np

img = cv2.imread("ans8.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

margin = int(min(H, W) * 0.25)
corner_boxes = {
    "TL": (0,          0,          margin, margin),
    "TR": (W - margin, 0,          W,      margin),
    "BR": (W - margin, H - margin, W,      H),
    "BL": (0,          H - margin, margin, H),
}

min_side = W * 0.015  # lowered from 0.020
max_side = W * 0.10
min_area = min_side ** 2
max_area = max_side ** 2

for name, (x1, y1, x2, y2) in corner_boxes.items():
    patch = gray[y1:y2, x1:x2]
    print(f"\n--- {name} ---")
    best = None
    for thr in [60, 80, 100, 120, 140, 160, 180]:
        _, bw = cv2.threshold(patch, thr, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_any = False
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            if area > 100:
                print(f"  Thr {thr}: area={area:.1f}, w={w}, h={h}, ratio={ratio:.2f}")
            if area < min_area or area > max_area:
                continue
            if ratio < 0.65:
                continue
            hull = cv2.convexHull(c)
            solidity = area / (cv2.contourArea(hull) + 1e-6)
            if solidity < 0.70:
                continue
            score = area * ratio * solidity
            print(f"  => VALID at thr {thr}: score={score:.1f}")
            found_any = True
        if found_any: break

import cv2
import numpy as np

img = cv2.imread("ans8.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

# Adaptive threshold to handle varied lighting
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)

cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

candidates = []
for c in cnts:
    area = cv2.contourArea(c)
    if area < 100 or area > (W * H * 0.1):
        continue
    x, y, w, h = cv2.boundingRect(c)
    ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
    if ratio < 0.65:
        continue
    hull = cv2.convexHull(c)
    solidity = area / (cv2.contourArea(hull) + 1e-6)
    if solidity < 0.70:
        continue
    
    # We found a dark square
    M = cv2.moments(c)
    if M["m00"] == 0: continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    candidates.append({"cx": cx, "cy": cy, "area": area, "rect": (x, y, w, h)})

print(f"Found {len(candidates)} square candidates.")
# Sort candidates by area descending
candidates.sort(key=lambda x: x["area"], reverse=True)

for i, cand in enumerate(candidates[:20]):
    print(f"  Cand {i}: cx={cand['cx']}, cy={cand['cy']}, area={cand['area']}")

# Let's draw them to visualize
vis = img.copy()
for cand in candidates[:20]:
    x, y, w, h = cand["rect"]
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite("detected-ans8/debug_robust_anchors.jpg", vis)
print("Saved debug_robust_anchors.jpg")

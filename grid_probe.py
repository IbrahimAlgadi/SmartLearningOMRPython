import cv2, numpy as np, sys
sys.path.insert(0, '.')
from omr_templates import infer_template, WARP_GY0, WARP_GY1, WARP_GX0, WARP_GX1

img = cv2.imread('detect_v2_ans19/02_warped.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bg = cv2.medianBlur(gray, 71)
bg = np.maximum(bg, 1)
norm = np.clip(gray.astype(np.float32) / bg.astype(np.float32) * 255, 0, 255).astype(np.uint8)
dark = (norm < 185).astype(np.float32)
proj = np.sum(dark[:, WARP_GX0:WARP_GX1], axis=1)

tmpl = infer_template(100, 5)

bands = []
in_b = False
for y in range(WARP_GY0 - 20, WARP_GY1 + 20):
    v = proj[y] if 0 <= y < len(proj) else 0
    if v > 30 and not in_b:
        bs, in_b = y, True
    elif v <= 30 and in_b:
        peak = int(np.argmax(proj[bs:y]) + bs)
        bands.append((bs, y-1, peak))
        in_b = False

print('Detected bands (start, end, peak_y):')
for i, (s, e, p) in enumerate(bands[:40]):
    print(f'  band {i:2d}: y={s}-{e}  peak={p}  width={e-s+1}')

print()
print('Template question row_y (all 25):')
for ri in range(tmpl.rows_per_col):
    print(f'  row_idx={ri:2d}: y={tmpl.row_y(ri)}')

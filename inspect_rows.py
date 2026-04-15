import cv2, numpy as np

img = cv2.imread("detected-ans6/02_warped.jpg", cv2.IMREAD_GRAYSCALE)
H, W = img.shape
print(f"Warped image size: {W}x{H}")

# Q1-Q25 column is the rightmost col_group.
# Grid: y0=240, y1=996, x0=79, x1=669
# Col groups span: (669-79)/4 = 147.5px each
# Col 3 (rightmost) x: 79 + 3*147.5 = 521.5 to 669
# Rows: (996-240)/25 = 30.24px per row
# Q23 = row 22, y = 240 + 22*30.24 = 905
# Q24 = row 23, y = 240 + 23*30.24 = 935
# Q25 = row 24, y = 240 + 24*30.24 = 966

row_h = (996 - 240) / 25
for q, row_idx in [(21,20), (22,21), (23,22), (24,23), (25,24)]:
    cy = int(240 + row_idx * row_h + row_h/2)
    # sample the full row across Q1-25 column (x=522 to 669)
    strip = img[cy-12:cy+12, 522:669]
    mean_px = strip.mean()
    dark_frac = (strip < 160).sum() / strip.size
    print(f"Q{q:>3} (cy={cy}): mean_pixel={mean_px:.1f}  dark_frac(thr<160)={dark_frac:.3f}")

# Also show a strip below Q25 (possible footer/border artifact)
print("\nArea below Q25 (y=985-1005, x=522-669):")
strip_bot = img[985:1005, 522:669]
print(f"  mean_pixel={strip_bot.mean():.1f}  dark_frac={((strip_bot<160).sum()/strip_bot.size):.3f}")

"""Print actual circle positions and fills for col_group 3 (Q1-Q25 column)."""
import sys, pathlib
sys.path.insert(0, '.')

# Monkey-patch to capture cluster data
import omr_detector as _omr
import cv2, numpy as np, json

img_path = "ans6.png"
debug_dir = pathlib.Path("detected-ans6-debug2")
debug_dir.mkdir(exist_ok=True)

# Load + warp
img = cv2.imread(img_path)
warped, method = _omr.align(img, debug_dir)
anchors = _omr.verify_anchors(warped, debug_dir)
circles, _ = _omr.detect_answer_circles(warped, debug_dir, anchors=anchors)

# Cluster
groups = _omr.cluster_circles(circles, 4, 25)

# Print col_group 3 (rightmost = Q1-Q25 in RTL)
# groups is {col_group: {row: [circle_dicts]}}
col3 = groups.get(3, {})
print(f"Col_group 3 has {len(col3)} rows")

for row_id in sorted(col3.keys()):
    row_circles = col3[row_id]
    ys = [c['cy'] for c in row_circles]
    fills = [c['fill'] for c in row_circles]
    print(f"  row {row_id:>2}  cy={int(np.mean(ys)):>4}  fills={[f'{f:.2f}' for f in fills]}  mean={np.mean(fills):.3f}")

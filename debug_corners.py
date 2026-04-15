import cv2, numpy as np

img = cv2.imread('student-answer.png')
H, W = img.shape[:2]
tl, tr, bl, br = (55,55), (613,55), (55,868), (613,868)
src = np.float32([tl, tr, bl, br])
wW, wH = 750, 1050
dst = np.float32([[0,0],[wW,0],[0,wH],[wW,wH]])
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, (wW, wH))
wH2, wW2 = warped.shape[:2]

stud_box = warped[int(wH2*0.09):int(wH2*0.50), int(wW2*0.02):int(wW2*0.38)]
bh, bw = stud_box.shape[:2]
print(f'Student box: {bw}x{bh}')

grid_y0 = int(bh * 0.10)
grid_y1 = int(bh * 0.90)
gray = cv2.cvtColor(stud_box, cv2.COLOR_BGR2GRAY)
grid_gray = gray[grid_y0:grid_y1, :]

blurred = cv2.GaussianBlur(grid_gray, (5,5), 0)
circles_raw = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12,
                            param1=50, param2=18, minRadius=6, maxRadius=18)

def get_fill(g, cx, cy, r):
    mask = np.zeros(g.shape, np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), max(int(r)-2, 2), 255, -1)
    nz = np.count_nonzero(mask)
    if nz == 0: return 0
    dark = np.sum(g[mask > 0] < 110)
    return dark / nz

circles_data = []
if circles_raw is not None:
    for c in np.uint16(np.around(circles_raw[0])):
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if cx-r >= 0 and cy-r >= 0 and cx+r < grid_gray.shape[1] and cy+r < grid_gray.shape[0]:
            f = get_fill(grid_gray, cx, cy, r)
            circles_data.append({'cx': cx, 'cy': cy, 'r': r, 'fill': f, 'filled': f > 0.42})

print(f'Circles: {len(circles_data)}')

# Gap-based clustering
def cluster_1d(values, gap=15.0):
    if not values: return []
    values = sorted(float(v) for v in values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] < gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]

xs = [b['cx'] for b in circles_data]
ys = sorted(b['cy'] for b in circles_data)

# k-means for cols
arr_x = np.array(xs, dtype=np.float32).reshape(-1,1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.5)
_, _, cx_centers = cv2.kmeans(arr_x, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
col_centers = sorted([float(c[0]) for c in cx_centers])

row_centers = cluster_1d(ys, gap=15.0)
print(f'Col centers: {[round(c) for c in col_centers]}')
print(f'Row centers: {[round(c) for c in row_centers]}')

# Build grid
grid = {}
for b in circles_data:
    ri = int(np.argmin([abs(b['cy']-rc) for rc in row_centers]))
    ci = int(np.argmin([abs(b['cx']-cc) for cc in col_centers]))
    key = (ri, ci)
    if key not in grid or b['fill'] > grid[key]['fill']:
        grid[key] = b

print('\nPer-column best fill:')
for ci in range(4):
    col_cells = [(ri, data) for (r, c), data in grid.items() if c == ci]
    if col_cells:
        best = max(col_cells, key=lambda x: x[1]['fill'])
        fills = sorted([d['fill'] for _, d in col_cells], reverse=True)
        second = fills[1] if len(fills) > 1 else 0
        print(f'  col{ci}: best row_idx={best[0]} fill={best[1]["fill"]:.3f} second={second:.3f} margin={best[1]["fill"]-second:.3f}  {"ACCEPT" if best[1]["fill"] > 0.35 and best[1]["fill"]-second > 0.06 else "REJECT"}')
        # Print all filled in this col
        filled = [(ri, d["fill"]) for ri, d in col_cells if d["filled"]]
        print(f'    filled: {filled}')

print('\nExpected: col3=row0=digit0, col2=row0=digit0, col1=row0=digit0, col0=row4=digit4')
print('Reversed output would be: digits[col3,col2,col1,col0] reversed = [0,0,0,4] -> "0004"')

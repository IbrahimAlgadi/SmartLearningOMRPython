"""Measure test code box proportions by finding the grid area."""
import cv2, numpy as np, sys
sys.path.insert(0, ".")

# Run the align step inline
img = cv2.imread("student-answer.png")
gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_orig, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)
k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
edges = cv2.dilate(edges, k, iterations=1)
cnts,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
corners4 = None
H,W = img.shape[:2]
for c in cnts[:10]:
    if cv2.contourArea(c) < 0.10*H*W: continue
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx)==4:
        corners4 = approx.reshape(4,2).astype(np.float32)
        break

def order(pts):
    rect = np.zeros((4,2),np.float32)
    s = pts.sum(axis=1); d = np.diff(pts,axis=1)
    rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    rect[1]=pts[np.argmin(d)]; rect[3]=pts[np.argmax(d)]
    return rect

rect = order(corners4)
tl,tr,br,bl = rect
w1=np.linalg.norm(br-bl); w2=np.linalg.norm(tr-tl)
h1=np.linalg.norm(tr-br); h2=np.linalg.norm(tl-bl)
out_w=max(int(max(w1,w2)),600); out_h=max(int(max(h1,h2)),840)
dst=np.float32([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]])
M=cv2.getPerspectiveTransform(rect,dst)
warped=cv2.warpPerspective(img,M,(out_w,out_h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
wH,wW = warped.shape[:2]
print(f"Warped: {wW}x{wH}")

# Extract test box
x0 = int(wW*0.02); x1 = int(wW*0.36)
y0 = int(wH*0.48); y1 = int(wH*0.85)
box = warped[y0:y1, x0:x1]
bh,bw = box.shape[:2]
print(f"Test box: {bw}x{bh}")
cv2.imwrite("debug_output/test_box_measure.jpg", box)

# Draw horizontal lines at various % to find grid start/end
vis = box.copy()
for pct in range(0,101,5):
    y = int(bh*pct/100)
    cv2.line(vis,(0,y),(bw,y),(0,165,255),1)
    cv2.putText(vis,f"{pct}%",(2,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,165,255),1)
cv2.imwrite("debug_output/test_box_measure_lines.jpg", vis)
print("Saved test_box_measure_lines.jpg")

# Try OCR on bottom portion
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    for pct in [85, 88, 90, 92, 95]:
        strip = box[int(bh*pct/100):, :]
        # scale up for OCR
        strip_big = cv2.resize(strip, (strip.shape[1]*3, strip.shape[0]*3), interpolation=cv2.INTER_CUBIC)
        gray_strip = cv2.cvtColor(strip_big, cv2.COLOR_BGR2GRAY)
        _, bw_strip = cv2.threshold(gray_strip, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        txt = pytesseract.image_to_string(bw_strip, config="--psm 7 digits").strip()
        print(f"  OCR strip from {pct}%: '{txt}'")
except Exception as e:
    print(f"OCR error: {e}")

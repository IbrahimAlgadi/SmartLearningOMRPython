import cv2, pytesseract, numpy as np
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Read the full test code box
img = cv2.imread("debug_output/07_test_code_box.jpg")
bh, bw = img.shape[:2]
print(f"test box: {bw}x{bh}")

# The "T20004" text appears to be between ~77% and ~87% of the box
for y0p, y1p in [(0.75, 0.90), (0.77, 0.87), (0.78, 0.86), (0.80, 0.88)]:
    strip = img[int(bh * y0p): int(bh * y1p), :]
    sh, sw = strip.shape[:2]
    cv2.imwrite(f"debug_output/strip_{int(y0p*100)}_{int(y1p*100)}.jpg", strip)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    # Try different preprocessing
    for method in ["otsu", "clahe"]:
        if method == "otsu":
            _, bwi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            enhanced = clahe.apply(gray)
            _, bwi = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        big = cv2.resize(bwi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        txt = pytesseract.image_to_string(
            big,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).strip()
        if txt:
            print(f"  strip {y0p:.2f}-{y1p:.2f} {method}: {repr(txt)}")

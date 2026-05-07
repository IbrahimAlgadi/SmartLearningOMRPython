import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
from pdf2image import convert_from_path

try:
    # Convert first page of PDF to image
    images = convert_from_path(str(_ROOT / "sheets" / "answer_page_ar.pdf"), dpi=200)
    if images:
        img_np = np.array(images[0])
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test_blank_ar.png', img_bgr)
        print("Created test_blank_ar.png")
except Exception as e:
    print(f"Error converting PDF: {e}")

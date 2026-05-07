# ── repo root on sys.path (script lives in scripts/) ──────────────────────────
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import subprocess

_SCRIPTS = Path(__file__).resolve().parent
base = _SCRIPTS.parent
det = str(_SCRIPTS / "omr_detector_enhanced_v3.py")

tests = [
    ("detect_v2_ans19/02_warped.jpg", "detect_v3cnn_ans19", "Q100", "Q100_5ch"),
    ("detect_v2_ans18/02_warped.jpg", "detect_v3cnn_ans18", "Q50",  "Q50_5ch"),
    ("detect_v2_ans17/02_warped.jpg", "detect_v3cnn_ans17", "Q20",  "Q20_5ch"),
]

for img, out, label, tpl in tests:
    r = subprocess.run(
        [sys.executable, "-u", det, str(base / img),
         "--template", tpl,
         "--debug-dir", str(base / out)],
        capture_output=True, text=True
    )
    print(f"\n{'='*50}")
    print(f"  {label}: {img}")
    print('='*50)
    for line in r.stdout.splitlines():
        if any(k in line for k in ["answered:", "unanswered:", "double_mark",
                                    "ambiguous:", "cls_type", "classifier"]):
            print(line)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-500:])
